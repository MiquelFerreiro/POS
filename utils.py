import math

import cv2
import numpy as np
from scipy import io as scio
from scipy import linalg
from scipy import signal
from scipy import sparse
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error

from scipy.signal import butter, filtfilt, czt

def lowpass(signal, fs, cutoff):
        b, a = butter(
            N=4,
            Wn=cutoff / (fs / 2),
            btype='low'
        )
        return filtfilt(b, a, signal)


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def process_video(frames):
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)

def cut_bvp(bvp, t_start, t_end, fs = 60):

    n_start = int(t_start * fs)
    n_end   = int(t_end * fs) if t_end is not None else len(bvp)
    return bvp[n_start:n_end]

def CZT(bvp, fs = 60):
    
    cutoff = 4

    # 1 — Low-pass filter
    bvp_lpf = lowpass(bvp, fs, cutoff)

    # 2 — Define the CZT frequency range you want to zoom into
    f_min = 0.66   # 40 bpm
    f_max = 3.0   # 180 bpm

    # Number of CZT frequency bins (resolution)
    M = 4096  # high spectral resolution

    # 3 — Convert to angular frequencies for CZT
    w = np.exp(-1j * 2 * np.pi * (f_max - f_min) / (M * fs))
    a = np.exp(1j * 2 * np.pi * f_min / fs)

    # 4 — Apply CZT
    czt_vals = czt(bvp_lpf, m=M, w=w, a=a)

    # 5 — Create the frequency axis
    czt_freqs = np.linspace(f_min, f_max, M)

    # 6 — Power spectrum
    czt_power = np.abs(czt_vals)

    return czt_power, czt_freqs