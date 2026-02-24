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


# def cut_bvp(bvp, t_start, t_end, fs = 60):

#     n_start = int(t_start * fs)
#     n_end   = int(t_end * fs) if t_end is not None else len(bvp)
#     return bvp[n_start:n_end]

def cut_bvp(bvp, t_start, t_end, fs=60):
    n = len(bvp)

    # Convert time to sample indices
    n_start = int(t_start * fs)
    n_end = int(t_end * fs)

    # Clamp to valid range
    n_start = max(0, min(n_start, n))
    n_end = max(0, min(n_end, n))

    return bvp[n_start:n_end]


import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

def compute_bvp_short_window_features(bvp_signal, fs):
    """
    Compute BVP features from a single short window.
    Returns a dictionary (same format as previous function).
    """

    bvp = np.asarray(bvp_signal)

    if len(bvp) < fs:  # less than 1 second
        return None

    # ---- Remove DC component ----
    bvp = bvp - np.mean(bvp)

    # ---- Time-domain features ----
    features = {}

    features["bvp_std"] = np.std(bvp)
    features["bvp_min"] = np.min(bvp)
    features["bvp_max"] = np.max(bvp)
    features["bvp_ptp"] = np.ptp(bvp)              # peak-to-peak
    features["bvp_energy"] = np.sum(bvp ** 2)
    features["bvp_skew"] = skew(bvp)
    features["bvp_kurtosis"] = kurtosis(bvp)

    # ---- Frequency-domain features ----
    freqs, psd = welch(bvp, fs=fs)

    features["bvp_total_power"] = np.sum(psd)
    features["bvp_dominant_freq"] = freqs[np.argmax(psd)]

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

def getmodelresults(timestamps, paths):

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
        if bvp is None or bvp.features == []:
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
