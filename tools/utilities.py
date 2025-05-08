
import argparse, logging, os, random, time
from time import strftime, localtime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger('mylogger')


def make_directories(time_string=None):
    experiments_folder = "experiments"
    if not os.path.isdir(experiments_folder):
        os.mkdir(experiments_folder)

    if time_string == None:
        time_string = strftime("%Y-%m-%d-at-%H:%M:%S", localtime())

    experiment_folder = experiments_folder + '/experiment-' + time_string + '/'

    if not os.path.isdir(experiment_folder):
        os.mkdir(experiment_folder)

        # create folder to save new models trained
    model_folder = experiment_folder + '/model/'
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

        # create folder to save TensorBoard
    log_folder = experiment_folder + '/log/'
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)

    return experiment_folder


def plot_softmax_evolution(softmaxes_list, name='softmaxes'):
    import matplotlib.pylab as plt

    f = plt.figure()
    index = range(len(softmaxes_list[0]))
    for softmax in softmaxes_list:
        plt.bar(index, softmax)

    plt.xlabel('Token')
    plt.ylabel('Probability')
    plt.title('softmax evoluti\on during training')
    plt.show()
    f.savefig(name + ".pdf", bbox_inches='tight')


def checkDuringTraining(generator_class, indices_sentences, encoder_model, decoder_model, batch_size, lat_dim):
    # original sentences
    sentences = generator_class.indicesToSentences(indices_sentences)

    # reconstructed sentences
    point = encoder_model.predict(indices_sentences)
    indices_reconstructed, _ = decoder_model.predict(point)

    sentences_reconstructed = generator_class.indicesToSentences(indices_reconstructed)

    # generated sentences
    noise = np.random.rand(batch_size, lat_dim)
    indicess, softmaxes = decoder_model.predict(noise)
    sentences_generated = generator_class.indicesToSentences(indicess)

    from prettytable import PrettyTable

    table = PrettyTable(['original', 'reconstructed', 'generated'])
    for b, a, g in zip(sentences, sentences_reconstructed, sentences_generated):
        table.add_row([b, a, g])
    for column in table.field_names:
        table.align[column] = "l"
    print(table)

    print('')
    print('number unique generated sentences:   ', len(set(sentences_generated)))
    print('')
    print(softmaxes[0][0])
    print('')

    return softmaxes


def get_random_string():
    return ''.join([str(r) for r in np.random.choice(10, 4)])


def timeStructured(random_string=True, seconds=False):
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S-", named_tuple)
    if random_string:
        # random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
        # random_string = str(abs(hash(named_tuple)))[:4]
        time_string += '-' + get_random_string()

    if seconds:
        return time_string, time.time()
    return time_string


def collect_information():
    f = open('./cout.txt', 'r')
    i = 0

    for line in f:
        if '>> "           AriEL' in line:
            print(i)
            i = 0
            print(line)
            g = open(line[-11:-2] + '.txt', 'w')
            g.write('\n\n')
            g.write(line)
            g.write('\n\n')

        if 'cpu' in line or 'CPU' in line:
            if not 'I tensorflow' in line:
                i += 1
                g.write(line)
                g.write('\n')
    print(i)


def setReproducible(seed=0, disableGpuMemPrealloc=True, prove_seed=True):
    # Fix the seed of all random number generator
    random.seed(seed)
    np.random.seed(seed)

    if prove_seed:
        print(np.random.rand())
    if tf.__version__[0] == '2':
        tf.random.set_seed(seed)
    else:
        tf.random.set_random_seed(seed)

    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        device_count={'CPU': 1})
    if disableGpuMemPrealloc:
        config.gpu_options.allow_growth = True
    # K.clear_session()
    # K.set_session(tf.Session(config=config))


def Dict2ArgsParser(args_dict):
    parser = argparse.ArgumentParser()
    for k, v in args_dict.items():
        if type(v) == bool and v == True:
            parser.add_argument("--" + k, action="store_true")
        elif type(v) == bool and v == False:
            parser.add_argument("--" + k, action="store_false")
        else:
            parser.add_argument("--" + k, type=type(v), default=v)
    args = parser.parse_args()
    return args


def move_mouse():
    import pyautogui, keyboard

    i = 0
    while True:
        i += 1
        print((-1) ** i * 1, (-1) ** (i + 1) * 1)
        # pyautogui.moveTo((-1)**i*.01, (-1)**(i+1)*.01, duration=1)
        # pyautogui.moveTo(1, 1, duration=1)
        pyautogui.moveRel((-1) ** i * 100, (-1) ** (i + 1) * 100, duration=1)
        if keyboard.is_pressed('q'):  # if key 'q' is pressed
            print('You Pressed A Key!')
            break  # finishing the loop

str2val = lambda comments, x, f: f(
    [s for s in comments.split('_') if '{}:'.format(x) in s][0].replace('{}:'.format(x), ''))


if __name__ == '__main__':
    move_mouse()
