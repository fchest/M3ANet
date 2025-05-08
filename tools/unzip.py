import os, shutil
from zipfile import ZipFile
import numpy as np
from tqdm import tqdm

CDIR = os.path.dirname(os.path.realpath(__file__))


def unzip_good_exps(exp_origin, exp_destination, exp_identifiers=[''], except_folders=[], except_files=[],
                    unzip_what=None):
    tmp_ds = [os.path.join(*[exp_origin, e]) for e in os.listdir(exp_origin) if 'zip' in e]
    if not os.path.isdir(exp_destination): os.mkdir(exp_destination)

    ds = []
    for d in tmp_ds:
        for t_in in exp_identifiers:
            if t_in in d:
                any_exception = False
                for t_out in except_folders:
                    if t_out in d:
                        any_exception = True

                if not any_exception:
                    ds += [d]

    ds = sorted(ds)
    destinations = []
    for d in tqdm(ds):

        # Create a ZipFile Object and load sample.zip in it
        with ZipFile(d, 'r') as zipObj:
            # tail = 'good_' + ''.join([str(i) for i in np.random.choice(9, 5).tolist()])
            tail = d.split('\\')[-1].replace('.zip', '')
            destination = os.path.join(*[exp_destination, tail])
            destinations.append(destination)
            if not os.path.isdir(destination):
                os.mkdir(destination)

                # Extract all the contents of zip file in different directory
                # zipObj.extractall(destination)
                for z in zipObj.infolist():
                    do_unzip = not np.any([e in z.filename for e in except_files])
                    if do_unzip:
                        try:
                            if 'other_outputs' in z.filename:
                                zipObj.extract(z, destination)
                            elif 'config' in z.filename:
                                zipObj.extract(z, destination)
                            if not unzip_what is None:
                                if isinstance(unzip_what, list):
                                    for string in unzip_what:
                                        if string in z.filename:
                                            zipObj.extract(z, destination)
                                elif unzip_what == 'all':
                                    zipObj.extract(z, destination)
                                else:
                                    raise NotImplementedError
                        except Exception as e:
                            print(e)
    return destinations


def put_together_tensorboards(EXPERIMENTS):
    exps = [d for d in os.listdir(EXPERIMENTS) if not 'other_outputs' in d]
    for d in exps:
        path = os.path.join(*[EXPERIMENTS, d, 'other_outputs'])
        print(d)
        print(os.listdir(path))
        tb = os.listdir(path)[0]
        path_tb = os.path.join(*[path, tb])

        net_name = d.replace('.zip', '').split('=')[-1]
        random_string = 'logs_' + net_name + '_' + ''.join([str(i) for i in np.random.choice(4, 5).tolist()])
        dst = os.path.join(TENSORBOARDS, random_string)
        os.mkdir(dst)
        shutil.copy(path_tb, dst)


if __name__ == '__main__':
    GEXPERIMENTS = os.path.join(CDIR, 'good_experiments')
    EXPERIMENTS = os.path.join(CDIR, 'experiments')
    TENSORBOARDS = os.path.join(EXPERIMENTS, 'other_outputs')

    for d in [EXPERIMENTS, TENSORBOARDS]:
        if not os.path.isdir(d):
            os.mkdir(d)

    exp_identifiers = [
        '',
        # '2020-09-09--11',
        # '2020-10-09--',
        # '2020-09-23--',
        # 'freeze',
    ]

    except_identifiers = [
        # '2020-09-09--11',
        # 'freeze',
        # 'time_ae'
        # 'TwoWays'
        # 'experiment-',
        # 'time_ae_merge'
    ]
    unzip_good_exps(GEXPERIMENTS, exp_identifiers=exp_identifiers, except_identifiers=except_identifiers)
