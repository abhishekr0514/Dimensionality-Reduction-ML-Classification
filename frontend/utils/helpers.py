# ------------------------------------------
# Label Mapping
# ------------------------------------------

LABEL_MAP = {
    0: "Malignant",
    1: "Benign"
}


def map_label(label_value: int) -> str:
    """
    Convert numeric label to readable text.
    """
    return LABEL_MAP.get(label_value, "Unknown")


# ------------------------------------------
# Confidence Formatting
# ------------------------------------------

def format_confidence(probability: float) -> str:
    """
    Format probability into percentage string.
    """
    return f"{probability * 100:.2f}%"


# ------------------------------------------
# Get Best Model From Metrics
# ------------------------------------------

def get_best_model(metrics: dict, metric_name: str = "accuracy"):
    """
    Returns best model name based on selected metric.
    """

    if not metrics:
        return None

    best_model = max(
        metrics,
        key=lambda model: metrics[model].get(metric_name, 0)
    )

    return best_model


# ------------------------------------------
# Count Selected Features (Optional Upgrade)
# ------------------------------------------

def count_selected_features(method_name: str):
    """
    Estimate feature count after dimensionality reduction.
    Useful for displaying reduction comparison.
    """

    if method_name and isinstance(method_name, str) and "baseline" in method_name:
        return 30
    elif method_name and isinstance(method_name, str) and "pca" in method_name:
        return 10
    elif method_name and isinstance(method_name, str) and "svd" in method_name:
        return 10
    elif method_name and isinstance(method_name, str) and "lda" in method_name:
        return 1
    elif method_name and isinstance(method_name, str) and "correlation" in method_name:
        return "Reduced (correlation-based)"
    elif method_name and isinstance(method_name, str) and "sfs" in method_name:
        return 8
    elif method_name and isinstance(method_name, str) and "sbs" in method_name:
        return 8
    elif method_name and isinstance(method_name, str) and "sffs" in method_name:
        return 8
    elif method_name and isinstance(method_name, str) and "sfbs" in method_name:
        return 8
    else:
        return "Unknown"


# ------------------------------------------
# Safe Dictionary Access
# ------------------------------------------

def safe_get(dictionary: dict, key: str, default=None):
    """
    Safely get value from dictionary.
    """
    return dictionary.get(key, default)


def get_feature_reduction_stats(method_name):
    original = 30

    if method_name and isinstance(method_name, str) and "baseline" in method_name:
        reduced = 30
    elif method_name and isinstance(method_name, str) and "pca" in method_name:
        reduced = 10
    elif method_name and isinstance(method_name, str) and "svd" in method_name:
        reduced = 10
    elif method_name and isinstance(method_name, str) and "lda" in method_name:
        reduced = 1
    elif method_name and isinstance(method_name, str) and "correlation" in method_name:
        reduced = 20  # estimate or compute dynamically
    elif method_name and isinstance(method_name, str) and any(w in method_name for w in ["sfs", "sbs", "sffs", "sfbs"]):
        reduced = 8
    else:
        reduced = 30

    reduction_percent = round((1 - reduced / original) * 100, 2)

    return original, reduced, reduction_percent