import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

patients = list(range(1, 62))
patients.remove(23)

def get_label(quadrant):
    return quadrant.split("_")[0]

# def get_label(quadrant):
#     q = int(quadrant[1])
#     if q < 4:
#         return "High Arousal"
#     else:
#         return "Low Arousal"

# def get_label(quadrant):
#     q = int(quadrant[1])
#     if q in [1, 4, 7]:
#         return "Negative Valence"
#     else:
#         return "Positive Valence"


def getmodelresults(timestamps, quadrants):

    data = []
    labels = []
    groups = [] 

    for patient in patients:

        patient = f"Patient_{patient}"

        print(f"Computing Features for {patient}...", end="\r", flush=True)

        for idx, quadrant in enumerate(quadrants):

            path = rf"C:\Users\mique\OneDrive\Dokumenty\GitHub\POS\Gesture Emotion Prediction\Results\{patient}\{quadrant}\vid_crop.csv"

            df = pd.read_csv(path)

            # keep only valid frames
            if "success" in df.columns:
                df = df[df["success"] == 1]

            features = {}
                
            for t_start, t_end in getattr(timestamps, quadrant):

                df_window = df[(df[" timestamp"] >= t_start) & (df[" timestamp"] <= t_end)]

                # select AU intensity columns
                au_cols = [c for c in df_window.columns if "_r" in c]

                for col in au_cols:
                    features[f"{col}_mean"] = df_window[col].mean()
                    # features[f"{col}_std"] = df_window[col].std()
                    # features[f"{col}_max"] = df_window[col].max()

                # for col in au_cols:
                #     features[f"{col}_mean"] = df_window[col].mean()
                #     features[f"{col}_std"] = df_window[col].std()
                #     features[f"{col}_max"] = df_window[col].max()

            # example label extraction
            label = get_label(quadrant)   # Q1, Q2, Q3...

            id = f"{patient}{path}"

            data.append(features)
            labels.append(label)
            groups.append(id)

    X = pd.DataFrame(data)
    y = np.array(labels) 

    print()
    print("Dataset shape:", X.shape)

    print(np.unique(y, return_counts=True))
    print("Example of data:", X.iloc[0])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])   

    cv = StratifiedGroupKFold(n_splits=5)

    classif_reports = []
    conf_matrix = []
    f1_scores = []

    from collections import Counter
    print("Class distribution:", Counter(y))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"Fold {fold}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
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
