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

    if "baseline" in method_name:
        return 30
    elif "pca" in method_name:
        return 10
    elif "svd" in method_name:
        return 10
    elif "lda" in method_name:
        return 1
    elif "correlation" in method_name:
        return "Reduced (correlation-based)"
    elif "sfs" in method_name:
        return 8
    elif "sbs" in method_name:
        return 8
    elif "sffs" in method_name:
        return 8
    elif "sfbs" in method_name:
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

    if "baseline" in method_name:
        reduced = 30
    elif "pca" in method_name:
        reduced = 10
    elif "svd" in method_name:
        reduced = 10
    elif "lda" in method_name:
        reduced = 1
    elif "correlation" in method_name:
        reduced = 20  # estimate or compute dynamically
    elif any(w in method_name for w in ["sfs", "sbs", "sffs", "sfbs"]):
        reduced = 8
    else:
        reduced = 30

    reduction_percent = round((1 - reduced / original) * 100, 2)

    return original, reduced, reduction_percent