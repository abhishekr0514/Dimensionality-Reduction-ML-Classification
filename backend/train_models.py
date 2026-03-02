import os
import joblib
import json

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score

# Custom selectors
from src.selectors import CorrelationSelector
from src.selectors import SFSSelector, SBSSelector, SFFSSelector, SFBSSelector
from src.data_service import get_dataset


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = get_dataset()
X = df.drop("target", axis=1)
y = df["target"]

print("Dataset loaded successfully.")


# --------------------------------------------------
# COMMON MODEL
# --------------------------------------------------

svm_model = SVC(probability=True)


# --------------------------------------------------
# PIPELINES
# --------------------------------------------------

pipelines = {
    "baseline_svm": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm_model)
    ]),

    "correlation_svm": Pipeline([
        ("correlation", CorrelationSelector(threshold=0.9)),
        ("scaler", StandardScaler()),
        ("svm", svm_model)
    ]),

    "pca_svm": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10)),
        ("svm", svm_model)
    ]),

    "lda_svm": Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(n_components=1)),
        ("svm", svm_model)
    ]),

    "svd_svm": Pipeline([
        ("scaler", StandardScaler()),
        ("svd", TruncatedSVD(n_components=10)),
        ("svm", svm_model)
    ]),

    "sfs_svm": Pipeline([
        ("sfs", SFSSelector(k_features=8)),
        ("scaler", StandardScaler()),
        ("svm", svm_model)
    ]),

    "sbs_svm": Pipeline([
        ("sbs", SBSSelector(k_features=8)),
        ("scaler", StandardScaler()),
        ("svm", svm_model)
    ]),

    "sffs_svm": Pipeline([
        ("sffs", SFFSSelector(k_features=8)),
        ("scaler", StandardScaler()),
        ("svm", svm_model)
    ]),

    "sfbs_svm": Pipeline([
        ("sfbs", SFBSSelector(k_features=8)),
        ("scaler", StandardScaler()),
        ("svm", svm_model)
    ])
}


# --------------------------------------------------
# TRAIN, EVALUATE & SAVE
# --------------------------------------------------

metrics_dict = {}

for name, pipeline in pipelines.items():
    print(f"Training {name}...")
    pipeline.fit(X, y)

    # Predictions
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    # Compute metrics
    metrics_dict[name] = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob)
    }

    # Save model
    save_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(pipeline, save_path)

    print(f"{name}.pkl saved successfully!")


# --------------------------------------------------
# SAVE METRICS FILE
# --------------------------------------------------

metrics_path = os.path.join(MODEL_DIR, "metrics.json")

with open(metrics_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)

print("\nAll models trained and metrics saved successfully.")