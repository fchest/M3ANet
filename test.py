import os, sys, shutil, json, time
import argparse
from tqdm import tqdm
import torch
from dataset import cock_tail
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
sys.path.append('../')
import matplotlib.font_manager
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from M3ANET import M3ANET
import pandas as pd
from tools.VeryCustomSacred import CustomExperiment, ChooseGPU
from tools.utilities import timeStructured

from tools.plotting import save_wav, evaluations_to_violins, one_plot_test

from tools.calculate_intelligibility import find_intel


import pickle
import numpy as np


from tools.utilities import setReproducible

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
ex = CustomExperiment(random_string + '-mc-test', base_dir=CDIR, seed=100) 

@ex.config
def test_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/experiments.json',
                        help='JSON file for configuration')
    args = parser.parse_args()
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    model_path = config["test"]["model_path"]
    dataset_root = config["test"]["dataset_root"]
# the test code adapted from the UBESD's test code
@ex.automain
def test(model_path, dataset_root):
    testing = True
    exp_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[exp_dir, 'images'])
    other_dir = os.path.join(*[exp_dir, 'other_outputs'])
    device = "cuda"
    model = M3ANET().to('cuda')

    print('loading model')
    check_point = torch.load(model_path)
    print('loading {}'.format(model_path))
    #print(model.state_dict().keys())
    model.load_state_dict(check_point['model_state_dict'], strict=False)
    model.eval()

    ##############################################################
    #                    tests training
    ##############################################################
    with torch.no_grad():
        if testing:
            root_dir = dataset_root
            batch_size = 1
            print('testing the model')
            print('\n loading the test data')

            prediction_metrics = ['sdr','si-sdr', 'stoi', 'estoi', 'pesq']
            noisy_metrics = [m + '_noisy' for m in prediction_metrics]

            inference_time = []

            subjects = [None] + list(range(51))  # subject None means the whole testset

            for generator_type in ['test']:  # , 'test_unattended']:# ['test']:

                print(generator_type)
                all_subjects_evaluations = {}
                print('going through subjects')
                for subject in subjects:

                    prediction = []
                    df = pd.DataFrame(columns=prediction_metrics + noisy_metrics)
                    test_data = cock_tail(root_dir, 'test', subject=subject)

                    test_loader = DataLoader(test_data, batch_size=batch_size)
                    print('Subject {}'.format(subject))
                    try:
                        for batch_sample, (noisy, eeg, clean) in tqdm(enumerate(test_loader)):
                            print('batch_sample {} for subject {}'.format(batch_sample, subject))
                            noisy, eeg, clean = noisy.to(device), eeg.to(device), clean.to(device)

                            noisy_snd, clean = noisy, clean
                            noisy_snd = noisy_snd.squeeze().unsqueeze(0).cpu().detach().numpy()

                            clean = clean.squeeze().unsqueeze(0).cpu().detach().numpy()
                            intel_list, intel_list_noisy = [], []
                            inf_start_s = time.time()
                            print('predicting')
                            noisy = torch.cat(torch.split(noisy, 29184, dim=2), dim=0)
                            eeg = torch.cat(torch.split(eeg, 29184, dim=2), dim=0)

                            pred,_ ,_ = model(noisy, eeg)
                            print(pred.shape)
                            inf_t = time.time() - inf_start_s
                            if subject is None:
                                inference_time.append(inf_t)
                            pred = torch.cat(torch.split(pred, 1, dim=0), dim=2)
                            pred = pred.squeeze().unsqueeze(0).cpu().detach().numpy()
                            prediction.append(pred)
                            prediction_concat = np.concatenate(prediction, axis=0)
                            fig_path = os.path.join(
                                images_dir,
                                'prediction_b{}_s{}_g{}.png'.format(batch_sample, subject, generator_type))
                            print('saving plot')
                            one_plot_test(pred, clean, noisy_snd, '', fig_path)

                            print('finding metrics')
                            for m in prediction_metrics:
                                print('     ', m)
                                pred_m = find_intel(clean, pred, metric=m)
                                intel_list.append(pred_m)
                                

                                noisy_m = find_intel(clean, noisy_snd, metric=m)
                                intel_list_noisy.append(noisy_m)
                                

                            e_series = pd.Series(intel_list + intel_list_noisy, index=df.columns)
                            df = df.append(e_series, ignore_index=True)

                        if subject is None:
                            prediction_filename = os.path.join(
                                *[images_dir, 'prediction_{}_s{}_g{}.npy'.format('test', subject, generator_type)])
                            print('saving predictions')
                            np.save(prediction_filename, prediction_concat)

                        del prediction, intel_list, intel_list_noisy, pred, prediction_concat, e_series
                        df.to_csv(os.path.join(*[other_dir, 'evaluation_s{}_g{}.csv'.format(subject, generator_type)]),
                                  index=False)
                        if not subject is None:
                            print(df)
                            all_subjects_evaluations['Subject {}'.format(subject)] = df

                        fig, axs = plt.subplots(1, len(df.columns), figsize=(9, 4))
    
                        for ax, column in zip(axs, df.columns):
                            ax.set_title(column)
                            violin_handle = ax.violinplot(df[column])
                            violin_handle['bodies'][0].set_edgecolor('black')
                        fig.savefig(os.path.join(*[images_dir, 'metrics_s{}_g{}.png'.format(subject, generator_type)]),
                                    bbox_inches='tight')
                        plt.close('all')

                    except Exception as e:
                        print(e)

                print('end of code, plotting violins')

                # this part is added to find out why "evaluations_to_violins"  doesnt work
                path_to_test = os.path.join(*[other_dir, 'all_subjects_evaluations_{}.pkl'.format(generator_type)])
                a_file = open(path_to_test, "wb")
                pickle.dump(all_subjects_evaluations, a_file)
                a_file.close()

                #for loading the pickles


                path = path_to_test
                a_file = open(path, "rb")
                all_subjects_evaluations = pickle.load(a_file)
                a_file.close()
    
                prediction_metrics = ['sdr','si-sdr', 'stoi','estoi','pesq']
                noisy_metrics = [m + '_noisy' for m in prediction_metrics]
                generator_type = 'test'

                evaluations_to_violins({k: v[noisy_metrics] for k, v in all_subjects_evaluations.items()}, images_dir,
                                     generator_type + 'noisy')
                evaluations_to_violins({k: v[prediction_metrics] for k, v in all_subjects_evaluations.items()}, images_dir,
                                      generator_type + '')

    shutil.make_archive(ex.observers[0].basedir, 'zip', exp_dir)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config', type=str, default='configs/experiments.json',
#                         help='JSON file for configuration')
#     args = parser.parse_args()
#     with open(args.config) as f:
#         data = f.read()
#     config = json.loads(data)
#     test(model_path=config["test"]["model_path"],
#                    dataset_root=config["test"]["dataset_root"])
