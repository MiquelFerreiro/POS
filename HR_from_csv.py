import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, czt
import pandas as pd
from HR_from_bvp import HR_from_bvp

def lowpass(signal, fs, cutoff):
    b, a = butter(
        N=4,
        Wn=cutoff / (fs / 2),
        btype='low'
    )
    return filtfilt(b, a, signal)

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

def HR_from_csv(path, fs = 60, t_start = 0, t_end = None, plot_bvp = True, plot_czt = True):
    # Load CSV (comma-separated)
    df = pd.read_csv(path, sep=",")

    # Split the date + time
    df[['date', 'clock']] = df['time'].str.split(expand=True)

    # Create proper datetime and convert to t=0 seconds
    df['time'] = pd.to_datetime(df['date'] + " " + df['clock'])
    t0 = df['time'].iloc[0]
    df['time'] = (df['time'] - t0).dt.total_seconds()

    # Keep only ppg and time
    df = df[['time', 'ppg']].astype(float)

    # If t_end is None, use the last time in the data
    if t_end is None:
        t_end = df['time'].iloc[-1]

    # Filter the time range
    df = df[(df['time'] >= t_start) & (df['time'] <= t_end)]

    hr = HR_from_bvp(df['ppg'], fs, plot_bvp, plot_czt)

    return hr
