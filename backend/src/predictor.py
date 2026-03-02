from src.model_factory import load_model
from src.processing import validate_and_format_features


def predict(method: str, features: list):
    """
    Performs prediction using selected model.

    Parameters:
    method (str): Model name (e.g., 'baseline_svm')
    features (list): List of 30 numeric feature values

    Returns:
    dict: Prediction result with label and probability
    """

    # Validate & format input
    input_data = validate_and_format_features(features)

    # Load trained pipeline
    model = load_model(method)

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    label_map = {
        0: "Malignant",
        1: "Benign"
    }

    return {
        "method": method,
        "prediction": int(prediction),
        "label": label_map[int(prediction)],
        "probability_malignant": float(probability[0]),
        "probability_benign": float(probability[1])
    }