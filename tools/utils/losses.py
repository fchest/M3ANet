"""
source code:
    x
    https://github.com/chtaal/pystoi
"""

import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import functools
from tools.utils.OBM import OBM
import tools.utils.pmsqe as pmsqe
import tools.utils.perceptual_constants as perceptual_constants

tf.keras.backend.set_floatx('float32')

if tf.__version__[:2] == '1.':
    window_fn = functools.partial(tf.signal.hann_window, periodic=True)
    tf_signal = tf.signal
    tf_log = tf.log
elif tf.__version__[:2] == '2.':
    window_fn = functools.partial(tf.signal.hann_window, periodic=True)
    tf_signal = tf.signal
    tf_log = tf.math.log
else:
    raise NotImplementedError


def thirdoct(fs, nfft, num_bands, min_freq):
    """ Returns the 1/3 octave band matrix and its center frequencies
    # Arguments :
        fs : sampling rate
        nfft : FFT size
        num_bands : number of 1/3 octave bands
        min_freq : center frequency of the lowest 1/3 octave band
    # Returns :
        obm : Octave Band Matrix
        cf : center frequencies
    """
    f = np.linspace(0, fs, nfft + 1)
    f = f[:int(nfft / 2 + 1)]
    k = np.array(range(num_bands)).astype(float)
    cf = np.power(2. ** (1. / 3), k) * min_freq
    freq_low = min_freq * np.power(2., (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2., (2 * k + 1) / 6)
    obm = np.zeros((num_bands, len(f)))  # a verifier

    for i in range(len(cf)):
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        obm[i, fl_ii:fh_ii] = 1
    return obm.astype(np.float32), cf


def log10(x):
    numerator = tf_log(x)
    denominator = tf_log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def threshold(x):
    x_max=tf.math.maximum(x,-20)
    
    return tf.math.minimum(x_max,35)
    
def segSNR_loss(fs=15000):
    # proper definition: Speech enhancement using super-Gaussian speech models and noncausal a priori SNR estimation
    # half window of 32ms, N in the paper, eq (27)
    w = tf.cast(32/1000*fs, tf.int32)  #windows of 32ms
    #shift=tf.cast(16/1000*fs, tf.int32) #shift half a window

    def rsSNR(y_true, y_pred):  ##both have to be the same type of float
        sound_len = tf.shape(y_true)[1]
        nw = tf.cast((sound_len/w), tf.int32)#/shift, tf.int32)
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.squeeze(y_pred, axis=-1)
        loss=0.0    

        for l in range(0,nw):
            num = tf.reduce_sum(tf.square(tf.slice(y_true,[0,int(l*w)],[-1,w-1])), axis=1)#int(l*w/2)],[-1,w-1])), axis=1)

            den = tf.reduce_sum(tf.square(tf.slice(y_true,[0,int(l*w)],[-1,w-1]) - tf.slice(y_pred,[0,int(l*w)],[-1,w-1])), axis=1)
                                          
            loss_i= 10 * log10(num) - 10 * log10(den)            
            loss_i=threshold(loss_i)
            loss=loss+loss_i
            
        return -loss/tf.cast(nw,tf.float32)

    return rsSNR



def si_sdr_loss(y_true, y_pred):
    # print("######## SI-SDR LOSS ########")
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    x = tf.squeeze(y_true, axis=-1)
    y = tf.squeeze(y_pred, axis=-1)
    smallVal = 1e-9  # To avoid divide by zero
    a = K.sum(y * x, axis=-1, keepdims=True) / (K.sum(x * x, axis=-1, keepdims=True) + smallVal)

    xa = a * x
    xay = xa - y
    d = K.sum(xa * xa, axis=-1, keepdims=True) / (K.sum(xay * xay, axis=-1, keepdims=True) + smallVal)
    # d1=tf.zeros(d.shape)
    d1 = d == 0
    d1 = 1 - tf.cast(d1, tf.float32)

    d = -K.mean(10 * d1 * log10(d + smallVal))
    return d


def calc_sdr(estimation, origin):
    """
    batch-wise SDR caculation for one audio file.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    """

    origin_power = tf.reduce_sum(origin ** 2, 1, keepdims=True) + 1e-12  # (batch, 1)
    scale = tf.reduce_sum(origin * estimation, 1, keepdims=True) / origin_power  # (batch, 1)

    est_true = scale * origin  # (batch, nsample)
    est_res = estimation - est_true  # (batch, nsample)
    # est_true = est_true.T
    # est_res = est_res.T
    true_power = tf.reduce_sum(est_true ** 2, 1)
    res_power = tf.reduce_sum(est_res ** 2, 1)

    return 10 * log10(true_power) - 10 * log10(res_power)  # (batch, 1)


def estoi_sisdr_loss(batch_size=8, nbf=200, fs=10000, nfft=512, N=30, J=15, min_freq=150):
    estoi = estoi_loss(batch_size=batch_size, nbf=nbf, fs=fs, nfft=nfft, N=N, J=J, min_freq=min_freq)

    def esloss(y_true, y_pred):
        loss = si_sdr_loss(y_true, y_pred) + 50 * estoi(y_true, y_pred)
        return tf.reduce_mean(loss)

    return esloss


def audio_to_mfcc(audio, sample_rate, frame_length=1024, frame_step=256, fft_length=1024):
    # from https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms
    stfts = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]  # .value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7350.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., 1:13]
    return mfccs


def mfcc_loss(sample_rate, frame_length=1024, frame_step=256, fft_length=1024):
    def mfccloss(y_true, y_pred):
        # to make sure the signal is a tensor of [batch_size, num_samples]
        y_true = tf.reduce_mean(y_true, axis=2)
        y_pred = tf.reduce_mean(y_pred, axis=2)

        mfcc_true = audio_to_mfcc(y_true, sample_rate, frame_length=frame_length,
                                  frame_step=frame_step, fft_length=fft_length)
        mfcc_pred = audio_to_mfcc(y_pred, sample_rate, frame_length=frame_length,
                                  frame_step=frame_step, fft_length=fft_length)

        mse = tf.reduce_mean(tf.square(mfcc_true - mfcc_pred))
        return mse

    return mfccloss


def estoi_loss(batch_size=8, nbf=200, fs=10000, nfft=512,
               N=30,  # 30  # length of temporal envelope vectors
               J=15,  # Number of one-third octave bands (cannot be varied)
               min_freq=150  # 1050
               ):
    def estoi_loss_inner(y_true, y_pred):

        # print("######## ESTOI LOSS ########")
        M = int(nbf - (N - 1))  # number of temporal envelope vectors
        epsilon = 1e-9  # To avoid divide by zero

        O, _ = thirdoct(fs, nfft, J, min_freq)

        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.squeeze(y_pred, axis=-1)
        y_pred_shape = K.shape(y_pred)

        stft_true = tf_signal.stft(y_true, 256, 128, 512, window_fn, pad_end=False)
        stft_pred = tf_signal.stft(y_pred, 256, 128, 512, window_fn, pad_end=False)

        OBM1 = tf.convert_to_tensor(O)
        OBM1 = K.tile(OBM1, [y_pred_shape[0], 1, ])
        #        OBM1 = K.reshape(OBM1, [y_pred_shape[0], J, -1, ])
        OBM1 = K.reshape(OBM1, [y_pred_shape[0], J, 257, ])

        OCT_pred = K.sqrt(tf.matmul(OBM1, K.square(K.abs(tf.transpose(stft_pred, perm=[0, 2, 1])))))
        OCT_true = K.sqrt(tf.matmul(OBM1, K.square(K.abs(tf.transpose(stft_true, perm=[0, 2, 1])))))
        d = 0.0  # K.variable(0.0, 'float32')
        for i in range(0, batch_size):
            for m in range(0, M):
                x = K.squeeze(tf.slice(OCT_true, [i, 0, m], [1, J, N]), axis=0)
                y = K.squeeze(tf.slice(OCT_pred, [i, 0, m], [1, J, N]), axis=0)
                xn = x - K.mean(x, axis=-1, keepdims=True)
                yn = y - K.mean(y, axis=-1, keepdims=True)
                xn = xn / (K.sqrt(K.sum(xn * xn, axis=-1, keepdims=True)) + epsilon)
                yn = yn / (K.sqrt(K.sum(yn * yn, axis=-1, keepdims=True)) + epsilon)
                xn = xn - K.tile(K.mean(xn, axis=-2, keepdims=True), [J, 1, ])
                yn = yn - K.tile(K.mean(yn, axis=-2, keepdims=True), [J, 1, ])
                xn = xn / (K.sqrt(K.sum(xn * xn, axis=-2, keepdims=True)) + epsilon)
                yn = yn / (K.sqrt(K.sum(yn * yn, axis=-2, keepdims=True)) + epsilon)
                di = K.sum(xn * yn, axis=-1, keepdims=True)
                di = 1 / N * K.sum(di, axis=0, keepdims=False)
                d = d + di
        return 1 - (d / K.cast(batch_size * M, dtype='float'))

    return estoi_loss_inner


def stoi_loss(batch_size=8, nbf=200):
    def stoi_loss_inner(y_true, y_pred):
        # print("######## STOI LOSS ########")
        y_true = K.squeeze(y_true, axis=-1)
        y_pred = K.squeeze(y_pred, axis=-1)
        y_pred_shape = K.shape(y_pred)

        stft_true = tf_signal.stft(y_true, 256, 128, 512, window_fn, pad_end=False)
        stft_pred = tf_signal.stft(y_pred, 256, 128, 512, window_fn, pad_end=False)

        N = 44  # 230  # 30  # length of temporal envelope vectors
        J = 15  # Number of one-third octave bands (cannot be varied)
        M = int(nbf - (N - 1))  # number of temporal envelope vectors
        smallVal = 1e-9  # To avoid divide by zero

        fs = 10000  # 97656.25
        nfft = 512  # 256
        min_freq = 150  # 1150  # 1050
        O, _ = thirdoct(fs, nfft, J, min_freq)
        print(O.shape)

        OBM1 = tf.convert_to_tensor(O)
        OBM1 = K.tile(OBM1, [y_pred_shape[0], 1, ])
        OBM1 = K.reshape(OBM1, [y_pred_shape[0], 15, 257, ])

        OCT_pred = K.sqrt(tf.matmul(OBM1, K.square(K.abs(tf.transpose(stft_pred, perm=[0, 2, 1])))))
        OCT_true = K.sqrt(tf.matmul(OBM1, K.square(K.abs(tf.transpose(stft_true, perm=[0, 2, 1])))))

        doNorm = True

        c = K.constant(5.62341325, 'float')  # 10^(-Beta/20) with Beta = -15
        d = K.variable(0.0, 'float')
        for i in range(0, batch_size):  # Run over mini-batches
            for m in range(0, M):  # Run over temporal envelope vectors
                x = K.squeeze(tf.slice(OCT_true, [i, 0, m], [1, J, N]), axis=0)
                y = K.squeeze(tf.slice(OCT_pred, [i, 0, m], [1, J, N]), axis=0)
                if doNorm:
                    alpha = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True) / (
                        K.sum(K.square(y), axis=-1, keepdims=True)) + smallVal)
                    alpha = K.tile(alpha, [1, N, ])
                    ay = y * alpha
                    y = K.minimum(ay, x + x * c)
                xn = x - K.mean(x, axis=-1, keepdims=True)
                xn = xn / (K.sqrt(K.sum(xn * xn, axis=-1, keepdims=True)) + smallVal)
                yn = y - K.mean(y, axis=-1, keepdims=True)
                yn = yn / (K.sqrt(K.sum(yn * yn, axis=-1, keepdims=True)) + smallVal)
                di = K.sum(xn * yn, axis=-1, keepdims=True)
                d = d + K.sum(di, axis=0, keepdims=False)
        return 1 - (d / K.cast(batch_size * J * M, dtype='float'))

    return stoi_loss_inner


def stsa_mse(y_true, y_pred):
    # print("######## STSA-MSE LOSS ########")
    y_true = K.squeeze(y_true, axis=-1)
    y_pred = K.squeeze(y_pred, axis=-1)

    stft_true = K.abs(tf_signal.stft(y_true, 256, 128, 256, window_fn, pad_end=False))
    stft_pred = K.abs(tf_signal.stft(y_pred, 256, 128, 256, window_fn, pad_end=False))
    d = K.mean(K.square(stft_true - stft_pred))
    return d


def pmsqe_log_mse_loss(batch_size=8):
    def pmsqe_log_mse_loss_inner(y_true, y_pred):
        print("######## PMSQE Log-MSE LOSS ########")
        print("y_true shape:      ", K.int_shape(y_true))
        print("y_pred shape:      ", K.int_shape(y_pred))

        y_true = K.squeeze(y_true, axis=-1)
        y_pred = K.squeeze(y_pred, axis=-1)
        y_pred_shape = K.shape(y_pred)

        stft_true = K.square(K.abs(tf_signal.stft(y_true, 256, 128, 256, window_fn, pad_end=True)))
        stft_pred = K.square(K.abs(tf_signal.stft(y_pred, 256, 128, 256, window_fn, pad_end=True)))
        print("stft_true shape:   ", K.int_shape(stft_true))
        print("stft_pred shape:   ", K.int_shape(stft_pred))

        # Init PMSQE required constants
        pmsqe.init_constants(Fs=8000, Pow_factor=pmsqe.perceptual_constants.Pow_correc_factor_Hann,
                             apply_SLL_equalization=True,
                             apply_bark_equalization=True, apply_on_degraded=True, apply_degraded_gain_correction=True)

        d = K.variable(0.0, 'float')
        for i in range(0, batch_size):
            x = K.squeeze(tf.slice(stft_true, [i, 0, 0], [1, -1, -1]), axis=0)
            y = K.squeeze(tf.slice(stft_pred, [i, 0, 0], [1, -1, -1]), axis=0)
            # print("x shape:      ", K.int_shape(x))
            # print("y shape:      ", K.int_shape(y))

            x_log = tf_log(x + K.epsilon())
            y_log = tf_log(y + K.epsilon())
            logmse = tf.math.square(x_log - y_log)
            logmse = tf.math.multiply(tf.math.divide(1, sig_freq), logmse)
            logmse = K.mean(logmse, axis=-1, keepdims=False)

            d = d + K.mean(logmse + pmsqe.per_frame_PMSQE(x, y))
            # print("d shape:   ", K.eval(K.sum( di ,axis=0,keepdims=False)))

        print("Compiling PMSQE Log-MSE LOSS Done!")
        # print("d:                    ", K.eval(d))
        # return (d/K.cast(OCT_pred_shape[0]*OCT_pred_shape[1]*OCT_pred_shape[2],dtype='float'))
        return (d / K.cast(batch_size, dtype='float'))

    return pmsqe_log_mse_loss_inner



if __name__ == '__main__':
    batch_size = 16
    sound_len = 30000
    y_true = np.random.randn(batch_size, sound_len, 1).astype(np.float32)
    y_pred = np.random.randn(batch_size, sound_len, 1).astype(np.float32)
    losses = [mfcc_loss(80000),
              estoi_sisdr_loss(),
              si_sdr_loss,
              estoi_loss(),
              segSNR_loss()
              ]
    for loss in losses:
        loss_value = loss(y_true, y_pred)
        print(loss.__name__, '\t', loss_value)
