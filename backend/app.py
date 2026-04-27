from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os  
from src.data_service import BASE_DIR, get_dataset
from sklearn.inspection import permutation_importance
from src.model_factory import load_model
# Import backend logic
from src.predictor import predict
from src.utils import get_available_methods, get_feature_names
import shap
import numpy as np
# from src.model_factory import load_model
# from src.data_service import get_dataset
# --------------------------------------------------
# App Initialization
# --------------------------------------------------

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

df = None

def get_df():
    global df
    if df is None:
        df = get_dataset()
    return df

# --------------------------------------------------
# Routes
# --------------------------------------------------



@app.route("/")
def home():
    return jsonify({"message": "Breast Cancer ML API is running."})


@app.route("/methods", methods=["GET"])
def methods():
    return jsonify({
        "available_methods": get_available_methods()
    })


@app.route("/features", methods=["GET"])
def features():
    return jsonify({
        "features": get_feature_names()
    })


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON body provided."}), 400

        method = data.get("method")
        features = data.get("features")

        if not method or not features:
            return jsonify({"error": "Both 'method' and 'features' are required."}), 400

        result = predict(method, features)

        return jsonify(result)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route("/dataset/sample/<int:index>", methods=["GET"])
def get_sample(index):
    try:
        current_df = get_df()
        if index < 0 or index >= len(current_df):
            return jsonify({"error": "Index out of range"}), 400

        row = current_df.iloc[index]
        features = row.drop("target").tolist()
        actual_label = int(row["target"])

        return jsonify({
            "index": index,
            "features": features,
            "actual_label": actual_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/dataset/random", methods=["GET"])
def get_random_sample():
    try:
        current_df = get_df()
        row = current_df.sample(1).iloc[0]
        features = row.drop("target").tolist()
        actual_label = int(row["target"])

        return jsonify({
            "features": features,
            "actual_label": actual_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
import json

@app.route("/metrics", methods=["GET"])
def get_metrics():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        return jsonify(metrics)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/feature-importance/<model_name>", methods=["GET"])
def feature_importance(model_name):
    try:
        model = load_model(model_name)

        df = get_dataset()
        X = df.drop("target", axis=1)
        y = df["target"]

        result = permutation_importance(
            model, X, y,
            n_repeats=5,
            random_state=42,
            scoring="accuracy"
        )

        importances = result.importances_mean

        feature_names = X.columns.tolist()

        importance_dict = dict(
            sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
        )

        return jsonify(importance_dict)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/dataset", methods=["GET"])
def get_dataset_route():
    current_df = get_df()
    return jsonify(current_df.to_dict(orient="records"))

@app.route("/shap/<model_name>", methods=["POST"])
def shap_explanation(model_name):
    try:
        data = request.get_json()
        features = data.get("features")

        model = load_model(model_name)
        df = get_dataset()
        X = df.drop("target", axis=1)

        background = X.sample(20, random_state=42).values

        # Wrap prediction to avoid sklearn attribute issue
        def model_predict(data):
            return model.predict_proba(data)

        explainer = shap.KernelExplainer(
            model_predict,
            background
        )

        input_array = np.array(features).reshape(1, -1)

        # shap_values = explainer.shap_values(input_array)

        shap_values = explainer.shap_values(input_array)

        # Handle different SHAP output formats safely
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # class 1
        else:
            shap_vals = shap_values[0]

        # Handle base_value safely
        expected_value = explainer.expected_value

        if isinstance(expected_value, (list, np.ndarray)):
            base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            base_value = expected_value

        return jsonify({
            "base_value": float(base_value),
            "shap_values": shap_vals.tolist(),
            "feature_names": X.columns.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)