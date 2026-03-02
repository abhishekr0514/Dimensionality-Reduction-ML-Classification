import streamlit as st


def render_prediction_card(result: dict):
    """
    Displays prediction result in styled card format.

    Expected result format:
    {
        "label": "Benign",
        "prediction": 1,
        "probability_benign": 0.97,
        "probability_malignant": 0.03
    }
    """

    label = result.get("label")
    prob_benign = result.get("probability_benign")
    prob_malignant = result.get("probability_malignant")

    # Choose color based on prediction
    if label == "Benign":
        color = "#2ECC71"  # green
        emoji = "🟢"
        confidence = prob_benign
    else:
        color = "#E74C3C"  # red
        emoji = "🔴"
        confidence = prob_malignant

    st.markdown(
        f"""
        <div style="
            padding: 25px;
            border-radius: 12px;
            background-color: {color};
            color: white;
            text-align: center;
        ">
            <h2>{emoji} Prediction: {label}</h2>
            <h3>Confidence: {confidence:.4f}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display detailed probabilities
    st.markdown("### Detailed Probabilities")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Benign Probability", f"{prob_benign:.4f}")

    with col2:
        st.metric("Malignant Probability", f"{prob_malignant:.4f}")