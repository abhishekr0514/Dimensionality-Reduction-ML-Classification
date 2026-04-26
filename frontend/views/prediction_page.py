import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np

from services.api_client import (
    get_features,
    predict,
    get_feature_importance
)

from components.sidebar import render_sidebar
from components.prediction_card import render_prediction_card
from components.charts import render_probability_chart
from utils.helpers import get_feature_reduction_stats

from services.api_client import get_shap_values
import plotly.express as px

TOP_FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "worst concave points"
]


# ============================================================
# MAIN PAGE
# ============================================================

def render_prediction_page():

    config = render_sidebar()

    selected_model = config.get("method")
    input_mode = config.get("input_mode")
    sample_data = config.get("sample_data")
    actual_label = config.get("actual_label")

    # 🔥 ADD THIS BLOCK
    if not selected_model or not isinstance(selected_model, str):
        st.warning("⚠️ Please select a model from the sidebar")
        return

    render_header(selected_model)
    render_reduction_stats(selected_model)
    render_model_justification(selected_model)
    if not selected_model:
        return
    features = get_features()
    st.write("DEBUG selected_model:", selected_model)

    slider_values = render_sensitivity_sliders()

    input_values = handle_input_mode(
        input_mode,
        features,
        sample_data
    )

    if input_values:
        final_input = build_final_input(
            features,
            input_values,
            slider_values
        )

        result = run_prediction(
            selected_model,
            final_input
        )

        if result:
            render_results(
                result,
                actual_label
            )
            st.subheader("🔍 SHAP Waterfall Explanation")

            shap_data = get_shap_values(selected_model, final_input)

            if "error" in shap_data:
                st.error(shap_data["error"])
            else:
                base_value = shap_data["base_value"]
                shap_values = np.array(shap_data["shap_values"])
                feature_names = shap_data["feature_names"]

                # 🔥 FIX: If matrix (30,2), select class 1
                if len(shap_values.shape) == 2:
                    shap_values = shap_values[:, 1]

                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=base_value,
                    data=final_input,
                    feature_names=feature_names
                )

                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(explanation, max_display=10, show=False)
                st.pyplot(fig)

    render_feature_importance(selected_model)



# ============================================================
# HEADER
# ============================================================

def render_header(selected_model):

    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🧬 Breast Cancer Classification")

    with col2:
        st.info(f"Selected Model: {selected_model}")

    st.divider()


# ============================================================
# REDUCTION STATS
# ============================================================

def render_reduction_stats(selected_model):

    orig, red, percent = get_feature_reduction_stats(selected_model)

    st.subheader("📉 Dimensionality Reduction Impact")

    col1, col2, col3 = st.columns(3)

    col1.metric("Original Features", orig)
    col2.metric("Reduced Features", red)
    col3.metric("Reduction %", f"{percent}%")

    st.divider()

def render_model_justification(selected_model):

    st.subheader("📌 Why SVM is Used as Primary Model")

    if isinstance(selected_model, str) and "svm" in selected_model.lower():

        st.markdown("""
        After extensive experimental evaluation across multiple classification models 
        (Random Forest, Decision Tree, KNN, Naive Bayes, etc.), **Support Vector Machine (SVM)** 
        consistently achieved the highest overall performance in terms of:

        - ✅ Accuracy
        - ✅ Precision
        - ✅ F1 Score
        - ✅ ROC-AUC

        ### Why SVM Performs Better in This Case

        - The Breast Cancer dataset is **moderately sized and high-dimensional**.
        - SVM is effective in handling high-dimensional feature spaces.
        - It maximizes the margin between classes, improving generalization.
        - It performs well when classes are clearly separable.

        Based on empirical comparison results in the analysis notebook, 
        SVM demonstrated the most stable and robust classification performance 
        across baseline and dimensionality-reduced datasets.

        Therefore, SVM is selected as the primary model for prediction deployment.
        """)
    else:
        st.info("SVM was identified as the best-performing model during evaluation.")


# ============================================================
# SENSITIVITY SLIDERS
# ============================================================

def render_sensitivity_sliders():

    st.subheader("🎛 Real-Time Feature Sensitivity")

    slider_values = {}

    for feature in TOP_FEATURES:
        slider_values[feature] = st.slider(
            feature,
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.1
        )

    return slider_values


# ============================================================
# INPUT MODE HANDLER
# ============================================================

def handle_input_mode(input_mode, features, sample_data):

    input_values = []

    if input_mode == "Manual":

        st.subheader("✍️ Manual Feature Entry")

        feature_values = {}

        mean_features = [f for f in features if f.startswith("mean")]
        se_features = [f for f in features if f.endswith("error")]
        worst_features = [f for f in features if f.startswith("worst")]

        with st.expander("📊 Mean Measurements", expanded=True):
            for f in mean_features:
                feature_values[f] = st.number_input(f, value=0.0)

        with st.expander("📈 Standard Error Measurements"):
            for f in se_features:
                feature_values[f] = st.number_input(f, value=0.0)

        with st.expander("🔬 Worst Case Measurements"):
            for f in worst_features:
                feature_values[f] = st.number_input(f, value=0.0)

        input_values = [feature_values[f] for f in features]

    elif input_mode in ["Sample", "Random"]:

        if sample_data is None:
            st.info("Load sample from sidebar.")
            return []

        st.success("Sample Loaded Successfully ✅")

        st.dataframe(
            {features[i]: sample_data[i] for i in range(len(features))}
        )

        input_values = sample_data

    return input_values


# ============================================================
# BUILD FINAL INPUT (MERGE SLIDERS)
# ============================================================

def build_final_input(features, input_values, slider_values):

    feature_dict = dict(zip(features, input_values))

    for f in slider_values:
        if f in feature_dict:
            feature_dict[f] = slider_values[f]

    return [feature_dict[f] for f in features]


# ============================================================
# RUN PREDICTION
# ============================================================

def run_prediction(selected_model, input_values):

    result = predict({
        "method": selected_model,
        "features": input_values
    })

    if "error" in result:
        st.error(result["error"])
        return None

    return result


# ============================================================
# RENDER RESULTS
# ============================================================

def render_results(result, actual_label):

    render_prediction_card(result)

    render_probability_chart(
        result["probability_benign"],
        result["probability_malignant"]
    )

    if actual_label is not None:

        label_map = {0: "Malignant", 1: "Benign"}
        actual_label_text = label_map.get(actual_label, "Unknown")

        st.subheader("📌 Actual vs Predicted")

        col1, col2 = st.columns(2)

        col1.metric("Actual Label", actual_label_text)
        col2.metric("Predicted Label", result["label"])

        if actual_label_text == result["label"]:
            st.success("✅ Correct Prediction")
        else:
            st.error("❌ Incorrect Prediction")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def render_feature_importance(selected_model):

    st.subheader("📊 Feature Importance")

    importance_data = get_feature_importance(selected_model)

    if "error" not in importance_data:

        df_imp = pd.DataFrame(
            list(importance_data.items()),
            columns=["Feature", "Importance"]
        ).head(10)

        st.bar_chart(
            df_imp.set_index("Feature")
        )

