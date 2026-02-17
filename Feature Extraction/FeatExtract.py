import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
import sys
import os

sys.path.append(os.path.abspath(".."))

from utils import cut_bvp
from constants import Timestamps, expressive

base_path_video = "../BVPs"

failed_masks = [
    [2, "Q1_1"],
    [52, "Q7_2"],
    [53, "Q4_2"]
]

paths = [
    "Q1_1",
    "Q1_2",
    # "Q2_1",
    # "Q2_2",
    "Q3_1",
    "Q3_2",
    # "Q4_1",
    # "Q4_2",
    # "Q5_1",
    # "Q5_2",
    # "Q6_1",
    # "Q6_2",
    "Q7_1",
    "Q7_2",
    # "Q8_1",
    # "Q8_2",
    "Q9_1",
    "Q9_2"
]

patients = list(range(1, 62))
patients.remove(23)

#patients = expressive

class BVP:
    def __init__(self, patient, path, signal, features):
        self.patient = patient
        self.path = path
        self.signal = signal
        self.features = features



import numpy as np
from scipy.signal import find_peaks, czt
from scipy.stats import linregress
from itertools import permutations


def compute_bvp_arousal_features(bvp, fs,
                                 fmin=0.7,
                                 fmax=4.0,
                                 n_czt_bins=512):
    """
    Compute arousal-related features from short POS-based BVP signals
    using Chirp Z-Transform (CZT) for spectral analysis.

    Parameters
    ----------
    bvp : np.ndarray
        1D Blood Volume Pulse signal
    fs : float
        Sampling frequency (Hz)
    fmin, fmax : float
        Frequency band of interest (Hz)
    n_czt_bins : int
        Number of CZT frequency bins

    Returns
    -------
    features : dict
        Dictionary of extracted features
    """

    features = {}
    bvp = np.asarray(bvp, dtype=float)

    # if len(bvp) < fs * 5:
    #     # Too short to be meaningful
    #     for k in [
    #         "mean_hr", "hr_slope", "rmssd", "ibi_cv",
    #         "dom_freq", "peak_power_ratio", "spec_entropy",
    #         "freq_variance", "hr_snr",
    #         "amp_mean", "amp_std",
    #         "sample_entropy", "perm_entropy",
    #         "peak_success_ratio"
    #     ]:
    #         features[k] = np.nan
    #     return features

    # Remove DC
    bvp -= np.mean(bvp)

    # --------------------------------------------------
    # 1. Peak detection & IBI
    # --------------------------------------------------
    min_dist = int(0.4 * fs)  # ~150 bpm upper bound
    peaks, _ = find_peaks(bvp, distance=min_dist)

    if len(peaks) < 3:
        for k in features:
            features[k] = np.nan
            print("small")
        return features

    ibi = np.diff(peaks) / fs
    hr = 60.0 / ibi

    # --------------------------------------------------
    # 2. HR / IBI features
    # --------------------------------------------------
    features["mean_hr"] = np.mean(hr)

    t = np.arange(len(hr))
    features["hr_slope"] = linregress(t, hr).slope

    diff_ibi = np.diff(ibi)
    features["rmssd"] = (
        np.sqrt(np.mean(diff_ibi ** 2)) if len(diff_ibi) > 0 else np.nan
    )

    features["ibi_cv"] = np.std(ibi) / np.mean(ibi)

    # --------------------------------------------------
    # 3. CZT-based spectral features
    # --------------------------------------------------
    N = len(bvp)

    w = np.exp(-1j * 2 * np.pi * (fmax - fmin) / (n_czt_bins * fs))
    a = np.exp(1j * 2 * np.pi * fmin / fs)

    spectrum = czt(bvp, n_czt_bins, w, a)
    power = np.abs(spectrum) ** 2
    freqs = np.linspace(fmin, fmax, n_czt_bins)

    total_power = np.sum(power)

    if total_power > 0:
        idx_peak = np.argmax(power)
        dom_freq = freqs[idx_peak]

        features["dom_freq"] = dom_freq
        features["peak_power_ratio"] = power[idx_peak] / total_power

        p_norm = power / total_power
        features["spec_entropy"] = -np.sum(
            p_norm * np.log2(p_norm + 1e-12)
        )

        features["freq_variance"] = np.sum(
            power * (freqs - dom_freq) ** 2
        ) / total_power

        features["hr_snr"] = np.max(power) / (np.mean(power) + 1e-12)

    else:
        features["dom_freq"] = np.nan
        features["peak_power_ratio"] = np.nan
        features["spec_entropy"] = np.nan
        features["freq_variance"] = np.nan
        features["hr_snr"] = np.nan

    # --------------------------------------------------
    # 4. Pulse amplitude features
    # --------------------------------------------------
    troughs, _ = find_peaks(-bvp, distance=min_dist)
    n_beats = min(len(peaks), len(troughs))

    if n_beats > 0:
        amp = bvp[peaks[:n_beats]] - bvp[troughs[:n_beats]]
        features["amp_mean"] = np.mean(amp)
        features["amp_std"] = np.std(amp)
    else:
        features["amp_mean"] = np.nan
        features["amp_std"] = np.nan

    # --------------------------------------------------
    # 5. Sample entropy
    # --------------------------------------------------
    def sample_entropy(x, m=2, r=0.2):
        x = np.asarray(x)
        r *= np.std(x)
        N = len(x)

        def _phi(m):
            x_m = np.array([x[i:i + m] for i in range(N - m)])
            C = np.sum(
                np.max(
                    np.abs(x_m[:, None] - x_m[None, :]), axis=2
                ) <= r,
                axis=0
            ) - 1
            return np.sum(C) / ((N - m) * (N - m - 1))

        return -np.log(_phi(m + 1) / _phi(m))

    try:
        features["sample_entropy"] = sample_entropy(bvp)
    except Exception:
        features["sample_entropy"] = np.nan

    # --------------------------------------------------
    # 6. Permutation entropy
    # --------------------------------------------------
    def permutation_entropy(x, order=3, delay=1):
        x = np.asarray(x)
        perms = list(permutations(range(order)))
        counts = np.zeros(len(perms))

        for i in range(len(x) - delay * (order - 1)):
            pattern = x[i:i + delay * order:delay]
            idx = perms.index(tuple(np.argsort(pattern)))
            counts[idx] += 1

        p = counts / np.sum(counts)
        return -np.sum(p * np.log2(p + 1e-12))

    try:
        features["perm_entropy"] = permutation_entropy(bvp)
    except Exception:
        features["perm_entropy"] = np.nan

    # --------------------------------------------------
    # 7. Signal quality
    # --------------------------------------------------
    expected_beats = len(bvp) / fs * (features["mean_hr"] / 60.0)
    features["peak_success_ratio"] = (
        len(peaks) / expected_beats if expected_beats > 0 else np.nan
    )

    return features

import numpy as np


def compute_bvp_arousal_features_windowed(
    bvp,
    fs,
    window_sec=5.0,
    fmin=0.7,
    fmax=4.0,
    n_czt_bins=256
):
    """
    Compute arousal features using 5s windows and temporal aggregation.

    Parameters
    ----------
    bvp : np.ndarray
        1D BVP signal
    fs : float
        Sampling rate (Hz)
    window_sec : float
        Window length in seconds (default: 5s)
    fmin, fmax : float
        CZT frequency band
    n_czt_bins : int
        Number of CZT bins

    Returns
    -------
    features : dict
        Aggregated window-based features
    """

    bvp = np.asarray(bvp, dtype=float)
    win_len = int(window_sec * fs)

    if len(bvp) < 2 * win_len:
        # Need at least 2 windows to compute dynamics
        return None

    # --------------------------------------------------
    # 1. Window the signal
    # --------------------------------------------------
    windows = []

    # Regular windows
    for start in range(0, len(bvp) - win_len + 1, win_len):
        windows.append(bvp[start:start + win_len])

    # Force final window (end-aligned)
    end_start = len(bvp) - win_len
    if end_start > 0 and end_start not in range(0, len(bvp) - win_len + 1, win_len):
        windows.append(bvp[end_start:end_start + win_len])


    # --------------------------------------------------
    # 2. Extract features per window
    # --------------------------------------------------
    window_features = []

    for w in windows:
        feats = compute_bvp_arousal_features(
            w,
            fs,
            fmin=fmin,
            fmax=fmax,
            n_czt_bins=n_czt_bins
        )

        if any(np.isnan(v) for v in feats.values()):
            continue

        window_features.append(feats)

    if len(window_features) < 2:
        return None

    # --------------------------------------------------
    # 3. Aggregate across windows
    # --------------------------------------------------
    feature_names = list(window_features[0].keys())
    features = {}

    for name in feature_names:
        values = np.array([wf[name] for wf in window_features])

        features[f"{name}_mean"] = np.mean(values)
        features[f"{name}_std"] = np.std(values)
        features[f"{name}_range"] = np.max(values) - np.min(values)

    return features

##########################################################################################################################

def get_bvp_features(window_length = 5):
    fs = 60

    BVPs = []

    for patient in patients:

        for path in paths:

            if [patient, path] in failed_masks:
                print(f"Skipping Patient_{patient}, {path}")
                continue

            data = np.load(f"{base_path_video}/Patient_{patient}/{path}.npy")

            #t_start, t_end = getattr(Timestamps, path)

            #data_cut = cut_bvp(data, t_start, t_end, fs)

            #bvp = BVP(patient, path, data_cut, [])

            bvp = BVP(patient, path, data, [])

            BVPs.append(bvp)

            #print(f"Patient_{patient}, {path}: {data.shape}")

    print(f"Loaded {len(BVPs)} BVP signals")

    fs = 60  # sampling rate

    valid = []
    failed = []

    for bvp in BVPs:  # use a copy to safely remove items

        print(f"Computing Features for Patient_{bvp.patient}...", end="\r", flush=True)
        
        try:
            # Use the new windowed feature extraction
            feats = compute_bvp_arousal_features_windowed(
                bvp.signal,
                fs,
                window_sec=window_length,   # 5-second windows
                fmin=0.66,
                fmax=3.0,
                n_czt_bins=256
            )

            if feats is None or feats == []:
                print(f"Failed: Patient_{bvp.patient}, {bvp.path}")
                failed.append(f"Patient_{bvp.patient}, {bvp.path}")
                BVPs.remove(bvp)  # remove problematic signal
            else:
                bvp.features = feats
                valid.append(f"Patient_{bvp.patient}, {bvp.path}")

        except Exception as e:
            print(f"Error for Patient_{bvp.patient}, {bvp.path}: {e}")
            failed.append(f"Patient_{bvp.patient}, {bvp.path}")
            BVPs.remove(bvp)

    print(f"Extracted features for {len(valid)} videos")



    # Example inspection
    if valid:
        print(list(BVPs[0].features.items()))

    return BVPs





