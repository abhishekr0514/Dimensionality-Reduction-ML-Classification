import requests
import os

# ------------------------------------------
# Backend Configuration
# ------------------------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")

# ------------------------------------------
# Generic Request Handler (Safe Wrapper)
# ------------------------------------------

def handle_response(response):
    try:
        if response.status_code != 200:
            return {"error": f"Backend error: {response.status_code}"}
        return response.json()
    except Exception:
        return {"error": "Invalid response from backend"}


# ------------------------------------------
# Get Available Models
# ------------------------------------------

def get_methods():
    try:
        response = requests.get(f"{BACKEND_URL}/methods", timeout=10)
        data = handle_response(response)
        return data.get("available_methods", [])
    except Exception:
        return []


# ------------------------------------------
# Get Feature Names
# ------------------------------------------

def get_features():
    try:
        response = requests.get(f"{BACKEND_URL}/features")
        return handle_response(response)["features"]
    except Exception:
        return []


# ------------------------------------------
# Predict Endpoint
# ------------------------------------------

def predict(payload: dict):
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json=payload
        )
        return handle_response(response)
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------
# Get Dataset Sample by Index
# ------------------------------------------

def get_sample(index: int):
    try:
        response = requests.get(
            f"{BACKEND_URL}/dataset/sample/{index}"
        )
        return handle_response(response)
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------
# Get Random Dataset Sample
# ------------------------------------------

def get_random_sample():
    try:
        response = requests.get(
            f"{BACKEND_URL}/dataset/random"
        )
        return handle_response(response)
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------
# Get Model Metrics
# ------------------------------------------

def get_metrics():
    try:
        response = requests.get(f"{BACKEND_URL}/metrics")
        return handle_response(response)
    except Exception as e:
        return {}
    
# ------------------------------------------
# Get Feature Importance for a Model
# ------------------------------------------

def get_feature_importance(model_name: str):
    try:
        response = requests.get(
            f"{BACKEND_URL}/feature-importance/{model_name}"
        )
        return handle_response(response)
    except Exception as e:
        return {"error": str(e)}
    
def get_dataset():
    try:
        response = requests.get(f"{BACKEND_URL}/dataset", timeout=10)
        return handle_response(response)
    except Exception as e:
        return {"error": str(e)}

def get_shap_values(model_name, features):
    response = requests.post(
        f"{BACKEND_URL}/shap/{model_name}",
        json={"features": features}
    )
    return response.json()