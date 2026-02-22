import numpy as np
from scipy.signal import find_peaks, czt
from scipy.stats import linregress, skew, kurtosis
from itertools import permutations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix

base_path_video = "../BVPs"

failed_masks = [
    [2, "Q1_1"],
    [52, "Q7_2"],
    [53, "Q4_2"]
]

label_map = {
    "Q1": "Q1: ↑Arousal ↓Val",
    "Q2": "Q2: ↑Arousal -Val",
    "Q3": "Q3: ↑Arousal ↑Val",
    "Q4": "Q4: -Arousal ↓Val",
    "Q5": "Q5: -Arousal -Val",
    "Q6": "Q6: -Arousal ↑Val",
    "Q7": "Q7: ↓Arousal ↓Val",
    "Q8": "Q8: ↓Arousal -Val",
    "Q9": "Q9: ↓Arousal ↑Val",
}

class Timestamps:
    Q1_1 = [[9, 14],[14, 19]]
    Q1_2 = [[24, 29]]

    Q2_1 = [[1, 6],[6, 11]]
    Q2_2 = [[7, 12], [12, 17]]

    Q3_1 = [[14, 19], [19, 24]]
    Q3_2 = [[34, 39], [40, 44], [45, 49]]

    Q4_1 = [[9, 14], [16, 21]]
    Q4_2 = [[10, 15], [16, 21]]

    Q5_1 = [[18, 23], [10, 15]]
    Q5_2 = [[13, 18], [5, 10]]

    Q6_1 = [[80, 85], [85, 90]]
    Q6_2 = [[10, 15], [18, 23]]

    Q7_1 = [[43, 48], [30, 35]]
    Q7_2 = [[36, 41], [41, 46]]

    Q8_1 = [[12, 17], [17, 22]]
    Q8_2 = [[7, 12], [12, 17]]

    Q9_1 = [[15, 20], [25, 30]]
    Q9_2 = [[13, 18], [19, 24]]

paths = [
    "Q1_1",
    "Q1_2",
    # "Q2_1",
    # "Q2_2",
    # "Q3_1",
    # "Q3_2",
    # "Q4_1",
    # "Q4_2",
    #"Q5_1",
    #"Q5_2",
    # "Q6_1",
    # "Q6_2",
    # "Q7_1",
    # "Q7_2",
    # "Q8_1",
    # "Q8_2",
    "Q9_1",
    "Q9_2"
]

patients = list(range(1, 62))
patients.remove(23)

#patients = expressive

class BVP:
    def __init__(self, patient, path, signal, features, id):
        self.patient = patient
        self.path = path
        self.signal = signal
        self.features = features
        self.id = id


def cut_bvp(bvp, t_start, t_end, fs = 60):

    n_start = int(t_start * fs)
    n_end   = int(t_end * fs) if t_end is not None else len(bvp)
    return bvp[n_start:n_end]


def compute_bvp_short_window_features(
    bvp,
    fs,
    fmin=0.66,
    fmax=3.0,
    n_czt_bins=512
):
    """
    Robust short-window (≈3s) BVP features for arousal classification.
    Designed for small manually labeled emotional segments.

    Parameters
    ----------
    bvp : np.ndarray
        1D BVP signal
    fs : float
        Sampling frequency (Hz)
    """

    features = {}
    bvp = np.asarray(bvp, dtype=float)

    if len(bvp) < fs * 2:  # too short
        return {k: np.nan for k in [
            "mean_hr", "hr_slope",
            "dom_freq", "peak_power_ratio",
            "spec_entropy", "freq_variance", "hr_snr",
            "amp_mean", "amp_std",
            "signal_energy", "signal_std",
            "skewness", "kurtosis",
            "sample_entropy", "perm_entropy",
            "peak_success_ratio"
        ]}

    # Remove DC
    bvp = bvp - np.mean(bvp)

    # --------------------------------------------------
    # 1. Peak detection
    # --------------------------------------------------
    min_dist = int(0.4 * fs)
    peaks, _ = find_peaks(bvp, distance=min_dist)

    if len(peaks) >= 2:
        ibi = np.diff(peaks) / fs
        hr = 60.0 / ibi
        features["mean_hr"] = np.mean(hr)

        t = np.arange(len(hr))
        features["hr_slope"] = linregress(t, hr).slope
    else:
        features["mean_hr"] = np.nan
        features["hr_slope"] = np.nan

    # --------------------------------------------------
    # 2. Spectral features (CZT)
    # --------------------------------------------------
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
    # 3. Pulse amplitude features
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
    # 4. Signal statistics
    # --------------------------------------------------
    features["signal_energy"] = np.sum(bvp ** 2)
    features["signal_std"] = np.std(bvp)
    features["skewness"] = skew(bvp)
    features["kurtosis"] = kurtosis(bvp)

    # --------------------------------------------------
    # 5. Entropy features
    # --------------------------------------------------
    def sample_entropy(x, m=2, r=0.2):
        x = np.asarray(x)
        r *= np.std(x)
        N = len(x)

        if N < m + 2:
            return np.nan

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
    # 6. Signal quality
    # --------------------------------------------------
    if not np.isnan(features["mean_hr"]):
        expected_beats = len(bvp) / fs * (features["mean_hr"] / 60.0)
        features["peak_success_ratio"] = (
            len(peaks) / expected_beats
            if expected_beats > 0 else np.nan
        )
    else:
        features["peak_success_ratio"] = np.nan

    return features

# def get_label(path):
#     q = path.split("_")[0]  # "Q3_2" → "Q3"
#     return label_map[q]

def get_label(path):
    q = path.split("_")[0]

    if q in ["Q1", "Q3"]:
        return "HighArousal"
    else:
        return "LowArousal"

def getmodelresults(timestamps):

    fs = 60  # sampling rate

    BVPs = []

    for patient in patients:

        for path in paths:

            if [patient, path] in failed_masks:
                print(f"Skipping Patient_{patient}, {path}")
                continue

            data = np.load(f"{base_path_video}/Patient_{patient}/{path}.npy")

            for t_start, t_end in getattr(timestamps, path):

                data_cut = cut_bvp(data, t_start, t_end, fs)

                id = f"{patient}{path}"

                bvp = BVP(patient, path, data_cut, [], id)

                BVPs.append(bvp)

                #print(f"Patient_{patient}, {path}: {data.shape}")

    print(f"Loaded {len(BVPs)} BVP signals")

    valid = []
    failed = []

    for bvp in BVPs:  # use a copy to safely remove items

        print(f"Computing Features for Patient_{bvp.patient}...", end="\r", flush=True)
        
        try:
            # Use the new windowed feature extraction
            feats = compute_bvp_short_window_features(bvp.signal, fs)

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
    print(f"Failed: {failed}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])


    X = []
    y = []
    groups = []

    for bvp in BVPs:
        if bvp is None or bvp.features is []:
            print("Error: Patient", bvp.patient, bvp.path)
            continue

        feat_values = list(bvp.features.values())
        X.append(feat_values)
        y.append(get_label(bvp.path))
        groups.append(bvp.id)  
        

    X = np.array(X)
    y = np.array(y)

    print(X.shape, y.shape)
    print(np.unique(y, return_counts=True))
    print("Example of data: ", X[0])

    cv = StratifiedGroupKFold(n_splits=5)

    classif_reports = []
    conf_matrix = []
    f1_scores = []

    from collections import Counter
    print("Class distribution:", Counter(y))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"Fold {fold}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print("Train dist:", Counter(y_train))
        print("Test dist:", Counter(y_test))

        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)

        print("Score:", score)

        y_pred = pipe.predict(X_test)

        report_dict = classification_report(y_test, y_pred, output_dict=True)

        classif_reports.append(classification_report(y_test, y_pred))
        conf_matrix.append(confusion_matrix(y_test, y_pred))
        f1_scores.append(report_dict["macro avg"]["f1-score"])

    return classif_reports, conf_matrix, f1_scores
