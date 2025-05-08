import logging, os, pathlib, shutil, time
from time import strftime, localtime
import tensorflow as tf

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from tools.utilities import setReproducible, timeStructured


class CustomFileStorageObserver(FileStorageObserver):

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        if _id is None:
            # create your wanted log dir
            time_string = timeStructured(False)
            timestamp = "{}-{}_".format(time_string, ex_info['name'])
            options = '_'.join(meta_info['options']['UPDATE'])
            self.updated_config = meta_info['options']['UPDATE']
            #run_id = timestamp + options
            run_id = timestamp

            # update the basedir of the observer
            self.basedir = os.path.join(self.basedir, run_id)
            self.basedir = os.path.join(ex_info['base_dir'], self.basedir)

            # and again create the basedir
            pathlib.Path(self.basedir).mkdir(exist_ok=True, parents=True)

        # create convenient folders for current experiment
        for relative_path in ['images', 'text', 'other_outputs', 'trained_models']:
            absolute_path = os.path.join(*[self.basedir, relative_path])
            self.__dict__.update(relative_path=absolute_path)
            os.mkdir(absolute_path)

        return super().started_event(ex_info, command, host_info, start_time, config, meta_info, _id)

    def save_comments(self, comments=''):
        if not comments is '':
            comments_txt = os.path.join(*[self.basedir, '1', 'comments.txt'])
            with open(comments_txt, "w") as text_file:
                text_file.write(comments)


def CustomExperiment(experiment_name, base_dir=None, GPU=None, seed=10, ingredients=[]):
    ex = Experiment(name=experiment_name, base_dir=base_dir, ingredients=ingredients, save_git_info=False)
    ex.observers.append(CustomFileStorageObserver("experiments"))

    ex.captured_out_filter = apply_backspaces_and_linefeeds

    # create convenient folders for all experiments
    for path in ['data', 'experiments']:
        complete_path = os.path.join(base_dir, path)
        if not os.path.isdir(complete_path):
            os.mkdir(complete_path)

    # choose GPU
    if not GPU == None:
        ChooseGPU(GPU)

    # set reproducible
    if not seed is None:
        setReproducible(seed)
    else:
        import numpy as np
        print(np.random.rand())
    return ex


def ChooseGPU(GPU=None, memory_growth=True):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, memory_growth)
        except RuntimeError as e:
            print(e)

    if not GPU is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
        config = tf.compat.v1.ConfigProto()  # tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)  # tf.Session(config=config)


def remove_folder(folder_path):
    time.sleep(2)
    shutil.rmtree(folder_path, ignore_errors=True)


if __name__ == '__main__':
    a = [26.2, 37.0, 42.9, 47.5,
         55.0, 57.9, 58.4, 66.7]

    for f in a:
        new_f = f / 120 / 1e-9 * 1e6 * 5
        c = 3e8
        new_lambd = c / new_f
        new_f = int(10 * new_f * 1e-15) / 10
        new_lambd = int(10 * new_lambd * 1e9) / 10
        # print('{} MHz -> {} Hz'.format(f, new_f))
        print('{} MHz -> {} pHz = {} nm'.format(f, new_f, new_lambd))
