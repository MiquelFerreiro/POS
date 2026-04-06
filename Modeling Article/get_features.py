


import numpy as np
from scipy.signal import find_peaks, czt


def compute_bvp_short_window_features(bvp_signal, fs, patient=None, path=None):
    """
    Compute PPG features (14 features from paper) using CZT.
    Includes debug prints when something fails.
    """

    bvp = np.asarray(bvp_signal)

    if len(bvp) < fs * 2:
        print(f"[SHORT SIGNAL] Patient {patient}, {path}")
        return None

    # ---- Remove DC ----
    bvp = bvp - np.mean(bvp)

    features = {}

    # --------------------------------------------------
    # 1. Peak detection → RR intervals
    # --------------------------------------------------
    min_dist = int(0.4 * fs)
    peaks, _ = find_peaks(bvp, distance=min_dist)

    if len(peaks) < 3:
        print(f"[PEAK DETECTION FAILED] Patient {patient}, {path}")
        return None

    rr = np.diff(peaks) / fs
    hr = 60.0 / rr

    # --------------------------------------------------
    # 2. HR features
    # --------------------------------------------------
    features["hr_mean"] = np.mean(hr)
    features["hr_std"] = np.std(hr)

    # --------------------------------------------------
    # 3. HRV time-domain features
    # --------------------------------------------------
    features["rr_mean"] = np.mean(rr)
    features["rr_std"] = np.std(rr)

    diff_rr = np.diff(rr)

    if len(diff_rr) > 0:
        features["rmssd"] = np.sqrt(np.mean(diff_rr ** 2))
        features["sdsd"] = np.std(diff_rr)
        features["pnn50"] = np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr)
    else:
        print(f"[RR DIFFERENCES FAILED] Patient {patient}, {path}")
        features["rmssd"] = np.nan
        features["sdsd"] = np.nan
        features["pnn50"] = np.nan

    # --------------------------------------------------
    # 4. TINN
    # --------------------------------------------------
    try:
        hist, bin_edges = np.histogram(rr, bins=20)
        features["tinn"] = bin_edges[-1] - bin_edges[0]
    except Exception:
        print(f"[TINN FAILED] Patient {patient}, {path}")
        features["tinn"] = np.nan

    # --------------------------------------------------
    # 5. Poincaré
    # --------------------------------------------------
    try:
        if len(diff_rr) > 0:
            sd1 = np.sqrt(np.var(diff_rr) / 2)
            sd2 = np.sqrt(2 * np.var(rr) - 0.5 * np.var(diff_rr))

            features["sd1"] = sd1
            features["sd2"] = sd2
            features["sd1_sd2"] = sd1 / sd2 if sd2 != 0 else np.nan
        else:
            raise ValueError
    except Exception:
        print(f"[POINCARE FAILED] Patient {patient}, {path}")
        features["sd1"] = np.nan
        features["sd2"] = np.nan
        features["sd1_sd2"] = np.nan

    # --------------------------------------------------
    # 6. LF / HF using CZT
    # --------------------------------------------------
    try:
        rr_times = np.cumsum(rr)
        rr_times = np.insert(rr_times, 0, 0)

        fs_interp = 4.0
        t_uniform = np.arange(0, rr_times[-1], 1 / fs_interp)

        rr_interp = np.interp(
            t_uniform,
            rr_times,
            np.append(rr, rr[-1])
        )

        fmin = 0.04
        fmax = 0.4
        n_bins = 256

        w = np.exp(-1j * 2 * np.pi * (fmax - fmin) / (n_bins * fs_interp))
        a = np.exp(1j * 2 * np.pi * fmin / fs_interp)

        spectrum = czt(rr_interp, n_bins, w, a)
        power = np.abs(spectrum) ** 2
        freqs = np.linspace(fmin, fmax, n_bins)

        lf_band = (freqs >= 0.04) & (freqs < 0.15)
        hf_band = (freqs >= 0.15) & (freqs < 0.4)

        lf = np.sum(power[lf_band])
        hf = np.sum(power[hf_band])

        features["lf"] = lf
        features["hf"] = hf
        features["lf_hf"] = lf / hf if hf > 0 else np.nan

    except Exception:
        print(f"[LF/HF FAILED] Patient {patient}, {path}")
        features["lf"] = np.nan
        features["hf"] = np.nan
        features["lf_hf"] = np.nan

    return features