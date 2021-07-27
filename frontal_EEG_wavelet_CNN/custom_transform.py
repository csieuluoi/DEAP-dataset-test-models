
import os
import pywt
# print("pywavelet version:", pywt.__version__)
#from wavelets.wave_python.waveletFunctions import *
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import generate_scalogram, from_fig_to_image

def GenerateNoise(signal, SNR_dB): # desiredSNR [dB]; signal is an array with complex values
    n = np.zeros((len(signal),1), dtype=complex)

    snr = 10.0**(SNR_dB/10.0) # Desired linear SNR
    var_signal = signal.var() # Measure power of signal
    var_n = var_signal / snr # Calculate required noise power for desired SNR
    if (var_n == 0): # In case of a null signal
        var_n = snr
    noise = np.random.normal(0, np.sqrt(var_n), size=(len(signal)))

#     e = np.random.normal(0, np.sqrt(var_n*2.0)/2.0, size=(len(signal)))

    return noise

class CustomTransform(object):

    """normalize the image in
    """

    def __init__(self, scale = np.arange(3, 19), wavelet_name= 'morl', frame_size = 3, overlap_size = 1, sampling_rate = 128, adding_noise = False):
        self.scale = scale
        self.wavelet_name = wavelet_name
        self.frame_size = frame_size
        self.overlap_size = overlap_size
        self.sampling_rate = sampling_rate
        self.adding_noise = adding_noise

    def StandardNormalize(self, x):
        mean = np.mean(x)
        std = np.std(x)
        x = (x - mean)/(std)
        return x

    def MinMaxNormalize(self, x):
        max = np.max(x)
        min = np.min(x)
        x = (x - min)/(max - min)

        return x


    def __call__(self, tensor):
        # (13, 768)
        tensor = np.array(tensor)
        # (13, n_scale, 768)
        scalogram = generate_scalogram(tensor, scale = self.scale, wavelet = self.wavelet_name, sampling_rate = self.sampling_rate)
        # segmenting:
        scalogram_segment = []
        for i in np.arange(0, int(scalogram.shape[-1] / self.sampling_rate) - self.frame_size, self.frame_size - self.overlap_size):
            start = int(i*self.sampling_rate)
            end = start + self.frame_size*self.sampling_rate
            segment = scalogram[:, :, start:end]
            # segment = from_fig_to_image(segment, resize = True, new_size = (64, 64), grayscale = True)
            scalogram_segment.append(segment)



        return np.array(scalogram_segment)
