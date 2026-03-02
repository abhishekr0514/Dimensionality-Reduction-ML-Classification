import os
import joblib

# Directory where models are stored
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Allowed model names
AVAILABLE_MODELS = {
    "baseline_svm",
    "correlation_svm",
    "pca_svm",
    "lda_svm",
    "svd_svm",
    "sfs_svm",
    "sbs_svm",
    "sffs_svm",
    "sfbs_svm"
}


def load_model(method: str):
    """
    Loads trained model pipeline based on method name.

    Parameters:
    method (str): Model identifier

    Returns:
    sklearn Pipeline object
    """

    if method not in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid method '{method}'. "
            f"Available models: {list(AVAILABLE_MODELS)}"
        )

    model_path = os.path.join(MODEL_DIR, f"{method}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{method}.pkl' not found in models directory."
        )

    return joblib.load(model_path)