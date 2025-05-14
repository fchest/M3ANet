import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import random

random.seed(100)
torch.manual_seed(100)
np.random.seed(100)

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor

from dataset import load_CleanNoisyPairDataset
from sisdr_loss import si_sidrloss
from util import rescale, find_max_epoch, print_size
from util import LinearWarmupCosineDecay, loss_fn, loss_contrast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from M3ANET import M3ANET
from torchinfo import summary


def val(dataloader, model, loss_fn):

    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for noisy, eeg, clean in dataloader:
            noisy, eeg, clean = noisy.cuda(), eeg.cuda(), clean.cuda()
            pred, _, _ = model(noisy, eeg)

            test_loss += loss_fn(pred, clean).item()
    test_loss /= num_batches
    print(f"Val Avg loss: {test_loss:>8f} \n")
    return test_loss


def train(num_gpus, rank, group_name,
          exp_path, log, optimization):
    
    # setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path)

    # Create tensorboard logger.
    log_directory = os.path.join(log["directory"], exp_path)
    if rank == 0:
        tb = SummaryWriter(os.path.join(log_directory, 'tensorboard'))

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    # Get shared ckpt_directory ready
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if rank == 0:
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
            os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True)

    # load training data
    trainloader = load_CleanNoisyPairDataset(**trainset_config,
                                             subset='train',
                                             batch_size=optimization["batch_size_per_gpu"],
                                             num_gpus=num_gpus)
    valloader = load_CleanNoisyPairDataset(**trainset_config,
                                             subset='val',
                                             batch_size=optimization["batch_size_per_gpu"],
                                             num_gpus=num_gpus)
    print('Data loaded')
    
    # predefine model
    net = M3ANET(L1=network_config["L1"], L2=network_config["L2"], L3=network_config["L3"], L4=network_config["L4"], enc_channel=network_config["enc_channel"], 
                feature_channel=network_config["feature_channel"],encoder_kernel_size=network_config["encoder_kernel_size"],
                layers=network_config["layers"], rnn_type=network_config["rnn_type"], norm=network_config["norm"],
                K=network_config["K"], dropout=network_config["dropout"], bidirectional=network_config["bidirectional"],
                CMCA_kernel=network_config["kernel"],CMCA_layer_num=network_config["CMCA_layer_num"]).cuda()
    
    # summary(net, input_size=[(8, 1, 29184), (8, 128, 29184)])
    print_size(net) 

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=optimization["learning_rate"], weight_decay=0.001)#1e-3
    # optimizer = torch.optim.Adam(net.parameters(), lr=optimization["learning_rate"])

    # load checkpoint
    time0 = time.time()
    if log["ckpt_iter"] == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
    else:
        ckpt_iter = log["ckpt_iter"]
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (
            ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully')
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # training
    n_iter = ckpt_iter + 1

    # define learning rate scheduler
    scheduler = LinearWarmupCosineDecay(
        optimizer,
        lr_max=optimization["learning_rate"],
        n_iter=optimization["epochs"],
        iteration=n_iter,
        divider=25,
        warmup_proportion=0.04,
        phase=('linear','cosine'),
    )


    sisdr = si_sidrloss().cuda()

    last_val_loss = 100.0
    epoch = 0
    
    
    # 2684 iterations for 1 epoch
    while epoch < optimization["epochs"]:
        # for each iteration
        for noisy_audio, eeg, clean_audio in trainloader:

            clean_audio = clean_audio.cuda()
            noisy_audio = noisy_audio.cuda()
            eeg = eeg.cuda()


            optimizer.zero_grad()
            X = (noisy_audio, eeg, clean_audio)

            loss, info_nce = loss_contrast(net, X, mrstftloss=sisdr)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            scheduler.step()
            optimizer.step()
           
           
            # save checkpoint
            if n_iter > 0 and n_iter % log["iters_per_ckpt"] == 0 and rank == 0:
                print("iteration: {} \treduced loss: {:.7f} \tloss: {:.7f}".format(
                    n_iter, reduced_loss, loss.item()), flush=True)

                val_loss = val(valloader, net, sisdr)
                net.train()
                
                if rank == 0:
                    # save to tensorboard
                    tb.add_scalar("Train/Train-Loss", loss.item(), n_iter)
                    tb.add_scalar("Train/Train-Reduced-Loss", reduced_loss, n_iter)
                    tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                    tb.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter)
                    tb.add_scalar("Train/InfoNCE-Loss", info_nce, n_iter)
                    tb.add_scalar("Val/Val-Loss", val_loss, n_iter)
                if val_loss < last_val_loss:
                    print('validation loss decreases from {} to {}, save checkpoint'.format(last_val_loss, val_loss))
                    checkpoint_name = '{}.pkl'.format(n_iter)
                    torch.save({'iter': n_iter,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'training_time_seconds': int(time.time() - time0)},
                               os.path.join(ckpt_directory, checkpoint_name))
                    print('model at iteration %s is saved' % n_iter)
                    last_val_loss = val_loss 
                    
                else:                    
                    print('validation loss did not decrease')  
                torch.save({'iter': n_iter,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_time_seconds': int(time.time() - time0)},
                        os.path.join(ckpt_directory, 'last.pkl'))          
            n_iter += 1
                 
        epoch += 1
        print('epoch {} done'.format(epoch))
    # After training, close TensorBoard.
    if rank == 0:
        tb.close()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/M3ANET.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='xy',
                        help='name of group for distributed')
    args = parser.parse_args()
    args.config= 'configs/M3ANET.json'
    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]  # training parameters
    global dist_config
    dist_config = config["dist_config"]  # to initialize distributed training
    global network_config
    network_config = config["network_config"]  # to define network
    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, **train_config)
