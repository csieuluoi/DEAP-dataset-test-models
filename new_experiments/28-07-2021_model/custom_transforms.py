
import os
# import pywt
# print("pywavelet version:", pywt.__version__)
#from wavelets.wave_python.waveletFunctions import *
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import generate_scalogram

# def GenerateNoise(signal, SNR_dB): # desiredSNR [dB]; signal is an array with complex values
#     n = np.zeros((len(signal),1), dtype=complex)

#     snr = 10.0**(SNR_dB/10.0) # Desired linear SNR
#     var_signal = signal.var() # Measure power of signal
#     var_n = var_signal / snr # Calculate required noise power for desired SNR
#     if (var_n == 0): # In case of a null signal
#         var_n = snr
#     noise = np.random.normal(0, np.sqrt(var_n), size=(len(signal)))

# #     e = np.random.normal(0, np.sqrt(var_n*2.0)/2.0, size=(len(signal)))

#     return noise

class CustomTransform(object):

    """normalize the image in
    """

    def __init__(self, scale = None, n_scale = 32, wavelet = "morl", sampling_rate = 128, normalized=True):
        self.normalized = normalized
        self.scale = scale
        self.n_scale = n_scale
        self.wavelet = wavelet
        self.sampling_rate = sampling_rate

    # def StandardNormalize(self, x):
    #     mean = np.mean(x, axis = 1)
    #     std = np.std(x, axis = 1)
    #     x = (x - mean[:, np.newaxis]) / std[:, np.newaxis]
    #     return x

    # def MinMaxNormalize(self, x):
    #     max_ = np.max(x, axis = 1)
    #     min_ = np.min(x, axis = 1)
    #     x = (x - min_[:, np.newaxis])/(max_ - min_)[:, np.newaxis]
    #     x = x*2 - 1
    #     return x


    def __call__(self, tensor):
        x = generate_scalogram(
            tensor, 
            scale = self.scale, 
            n_scale = self.n_scale, 
            wavelet = self.wavelet, 
            sampling_rate = self.sampling_rate, 
            normalized = True
        )
        return x
