## code to calculate speech quality and intelligibility  using different metrics


import numpy as np
import os
import sys

sys.path.append('../')
sys.path.append('../..')
from tools.utils.losses import *
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
#from tools.nice_tools import *
from pesq import pesq
import pystoi
from pystoi.stoi import stoi
from pystoi import utils
from mir_eval.separation import bss_eval_sources

Fs_stoi = 10000
Fs_pesq = 8000  # used to be 16000
DYN_RANGE = 40  # Speech dynamic range
N_FRAME = 256  # Window support


def find_intel(true, prediction, metric='pesq', fs=14700):
    # to do: add an statement that checks the sizes of true and pred-done
    # to do: make a repository for estoi. For now the tf version is fine

    if true.shape != prediction.shape:
        raise Exception('True and prediction must have the same shape' + 'found shapes {} and {}'.format(true.shape,
                                                                                                         prediction.shape))
    if true.shape[0] != 1 or prediction.shape[0] != 1:
        raise Exception('Inputs must have the first dimension equal to 1')

    if metric == 'pesq':
        true_batch = np.squeeze(true)
        prediction_batch = np.squeeze(prediction)
        if fs != Fs_pesq:
            true = utils.resample_oct(np.squeeze(true), Fs_pesq, fs)
            prediction = utils.resample_oct(np.squeeze(prediction), Fs_pesq, fs)


        out_metric = pesq(Fs_pesq, true, prediction, 'nb')

    elif metric == 'stoi':
        true_batch = np.squeeze(true)
        prediction_batch = np.squeeze(prediction)
        out_metric = stoi(true_batch, prediction_batch, fs, extended=False)
        # print(out_metric)


    elif metric == 'estoi':
        true_batch = np.squeeze(true)
        prediction_batch = np.squeeze(prediction)
        if fs != Fs_stoi:
            true = utils.resample_oct(np.squeeze(true), Fs_stoi, fs)
            prediction = utils.resample_oct(np.squeeze(prediction), Fs_stoi, fs)

        true_batch, prediction_batch = utils.remove_silent_frames(true, prediction, DYN_RANGE, N_FRAME,
                                                                  int(N_FRAME / 2))

        true_batch = true_batch[np.newaxis]
        prediction_batch = prediction_batch[np.newaxis]
        true_batch = true_batch[..., np.newaxis]
        prediction_batch = prediction_batch[..., np.newaxis]
        nbf = (true_batch.shape[1] / 128) - 1
        batch_size = true_batch.shape[0]
        loss = estoi_loss(batch_size=batch_size, nbf=nbf)

        l_ = loss(true_batch.astype(np.float32), prediction_batch.astype(np.float32))[0]

        out_metric = 1 - l_.numpy()

    elif metric == 'si-sdr':

        true_batch, prediction_batch = utils.remove_silent_frames(np.squeeze(true), np.squeeze(prediction), DYN_RANGE,
                                                                  N_FRAME, int(N_FRAME / 2))

        loss = si_sdr_loss
        true_batch = true_batch[np.newaxis]
        prediction_batch = prediction_batch[np.newaxis]
        true_batch = true_batch[..., np.newaxis]
        prediction_batch = prediction_batch[..., np.newaxis]
        l_ = loss(true_batch.astype(np.float32), prediction_batch.astype(np.float32))

        # print(l_)
        l = l_.numpy()
        out_metric = l

        out_metric = -out_metric
        
    elif metric == 'sdr':
        #out_metric = calc_sdr(prediction, true)
        true_batch, prediction_batch = utils.remove_silent_frames(np.squeeze(true), np.squeeze(prediction), DYN_RANGE,
                                                                  N_FRAME, int(N_FRAME / 2))
      
        
        true_batch = true_batch[np.newaxis]
        prediction_batch = prediction_batch[np.newaxis]
        true_batch = true_batch[..., np.newaxis]
        prediction_batch = prediction_batch[..., np.newaxis]
        sdr, _, _, _ = bss_eval_sources(np.squeeze(true_batch.astype(np.float32)), np.squeeze(prediction_batch.astype(np.float32)))
        l = float(sdr)

        out_metric = l

    elif metric == 'mfcc':
        loss = mfcc_loss(fs)
        l_ = loss(true.astype(np.float32), prediction.astype(np.float32))

        l = l_.numpy()
        out_metric = l

        out_metric = out_metric


    elif metric == 'seg_SNR':

        loss = segSNR_loss(fs)
        l_ = loss(true.astype(np.float32), prediction.astype(np.float32))

        l = l_.numpy()
        out_metric = l

        out_metric = -out_metric


    elif metric == 'stsa-mse':

        true_batch, prediction_batch = removeSilentFrames(true, prediction, 40, 256, 128)

        loss = stsa_mse
        l_ = loss(true_batch.astype(np.float32), prediction_batch.astype(np.float32))

        l = l_.numpy()
        out_metric = l

        out_metric = out_metric


    else:
        raise NotImplementedError

    return out_metric
