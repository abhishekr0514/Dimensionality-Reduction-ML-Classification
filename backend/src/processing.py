import numpy as np


EXPECTED_FEATURE_COUNT = 30


def validate_and_format_features(features):
    """
    Validates and formats incoming feature list.

    Parameters:
    features (list): Raw feature values from frontend

    Returns:
    np.ndarray: Properly formatted input array
    """

    if not isinstance(features, list):
        raise ValueError("Features must be provided as a list.")

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Exactly {EXPECTED_FEATURE_COUNT} features are required.")

    try:
        numeric_features = [float(x) for x in features]
    except ValueError:
        raise ValueError("All feature values must be numeric.")

    return np.array(numeric_features).reshape(1, -1)