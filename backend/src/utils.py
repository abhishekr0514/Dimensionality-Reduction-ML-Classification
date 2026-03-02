# Label mapping for prediction output
LABEL_MAP = {
    0: "Malignant",
    1: "Benign"
}


def get_label(prediction: int) -> str:
    return LABEL_MAP.get(prediction, "Unknown")


def get_available_methods():
    return [
        "baseline_svm",
        "correlation_svm",
        "pca_svm",
        "lda_svm",
        "svd_svm",
        "sfs_svm",
        "sbs_svm",
        "sffs_svm",
        "sfbs_svm"
    ]


def get_feature_names():
    """
    Returns ordered feature names.
    Must match training dataset order.
    """

    return [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "mean compactness",
        "mean concavity",
        "mean concave points",
        "mean symmetry",
        "mean fractal dimension",
        "radius error",
        "texture error",
        "perimeter error",
        "area error",
        "smoothness error",
        "compactness error",
        "concavity error",
        "concave points error",
        "symmetry error",
        "fractal dimension error",
        "worst radius",
        "worst texture",
        "worst perimeter",
        "worst area",
        "worst smoothness",
        "worst compactness",
        "worst concavity",
        "worst concave points",
        "worst symmetry",
        "worst fractal dimension"
    ]