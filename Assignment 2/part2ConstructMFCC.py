# code referenced from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

import pickle
from scipy.fftpack import dct
import numpy
import matplotlib.pyplot as plt

def emphasizeSignal(signal, pre_emphasis):
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal

def framing(sample_rate, emphasized_signal):
    frame_size = 0.025
    frame_stride = 0.01
    frame_length = frame_size * sample_rate 
    frame_step =  frame_stride * sample_rate 
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z)

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    frames *= numpy.hamming(frame_length)
    
    return frames 

def returnFilterBanks(frames, sample_rate):
    NFFT = 512
    nfilt = 40
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])   
        f_m_plus = int(bin[m + 1])    

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)
    filter_banks = 20 * numpy.log10(filter_banks)
    return filter_banks

def calcMFCC(filter_banks):
    num_ceps = 12
    cep_lifter = 22
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift
    return mfcc 

def normalizeFilterBanks(filter_banks):
    filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    return filter_banks

def normalizeMFCC(mfcc):
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    return mfcc

def returnMFCC(d):
    sample_rate, signal = d
    duration_of_sound = len(signal)/sample_rate
    emphasized_signal = emphasizeSignal(signal, 0.97)
    frames = framing(sample_rate, emphasized_signal)
    filter_banks = returnFilterBanks(frames, sample_rate)
    mfcc =  calcMFCC(filter_banks)
    filter_banks = normalizeFilterBanks(filter_banks)
    mfcc = normalizeMFCC(mfcc)
    return mfcc

def plot_mfcc(mfcc):
    plt.imshow(mfcc)
    plt.colorbar(use_gridspec = True)
    plt.show()

names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
for i in range(len(names)):
    data = pickle.load(open(names[i] + "validationDataFile.p", "rb"))
    classPickle = []
    for j in range(len(data)):
        # print(i, j)
        mfcc = returnMFCC(data[j])
        classPickle.append(mfcc)
        plot_mfcc(mfcc)
    pickle.dump(classPickle, open(names[i] + "validationMFCC.p", "wb"))
