import glob
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt

def fft_plot(sampleRate, samples):
    n = len(samples)
    T = 1/sampleRate
    yf = scipy.fft(samples)
    xf = np.linspace(0.0, 1.0/(2.0*T), n/2)
    fig, ax = plt.subplots()
    ax.plot(xf, (2.0/n)*(np.abs(yf[:n//2])))
    plt.grid()
    plt.ylabel("Frequency")
    plt.xlabel("Magnitude")
    return plt.show()

def spectrogram(samples, sample_rate, stride_ms = 10.0, 
                          window_ms = 20.0, max_freq = 100000, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram

def plot_spectogram(spec):
    plt_spec = plt.imshow(spec,origin='lower')
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.colorbar(use_gridspec=True)
    plt.show()

names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
for name in names:
    data = pickle.load(open(name + "DataFile.p", "rb"))
    classSpectograms = []
    for d in data:
        sampleRate, samples = d
        duration_of_sound = len(samples)/sampleRate
        spec = spectrogram(samples, sampleRate)
        classSpectograms.append(spec)
        plot_spectogram(spec)
    pickle.dump(classSpectograms, open(name+"Spectrograms.p","wb" ))
    