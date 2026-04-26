import streamlit as st
from services.api_client import get_methods, get_sample, get_random_sample


def render_sidebar():
    """
    Renders sidebar and returns user selections.

    Returns:
        dict containing:
        {
            "method": selected_model,
            "input_mode": selected_mode,
            "sample_data": list or None,
            "actual_label": int or None
        }
    """

    st.sidebar.title("⚙️ Configuration")

    # ---------------------------------
    # Initialize session state
    # ---------------------------------
    if "sample_data" not in st.session_state:
        st.session_state["sample_data"] = None

    if "actual_label" not in st.session_state:
        st.session_state["actual_label"] = None

    # ---------------------------------
    # Model Selection
    # ---------------------------------
    methods = get_methods()

# 🔥 HANDLE EMPTY BACKEND RESPONSE
    if not methods:
        st.sidebar.error("⏳ Backend is waking up... please wait 30 seconds and refresh.")
        return {
            "method": None,
            "input_mode": None,
            "sample_data": None,
            "actual_label": None
        }

    selected_model = st.sidebar.selectbox(
        "Select Model",
        methods
    )

    # ---------------------------------
    # Input Mode Selection
    # ---------------------------------
    input_mode = st.sidebar.radio(
    "Input Mode",
    ["— Choose Input Mode —", "Manual", "Sample", "Random"]
    )
    if input_mode == "— Choose Input Mode —":
        input_mode = None

    # ---------------------------------
    # Manual Mode (Clear session data)
    # ---------------------------------
    if input_mode == "Manual":
        st.session_state["sample_data"] = None
        st.session_state["actual_label"] = None

    # ---------------------------------
    # Sample Mode
    # ---------------------------------
    if input_mode == "Sample":

        sample_index = st.sidebar.number_input(
            "Select Sample Index",
            min_value=0,
            value=0
        )

        if st.sidebar.button("Load Sample"):
            response = get_sample(sample_index)

            if "error" in response:
                st.sidebar.error(response["error"])
            else:
                st.session_state["sample_data"] = response["features"]
                st.session_state["actual_label"] = response["actual_label"]
                st.sidebar.success("Sample loaded successfully!")

    # ---------------------------------
    # Random Mode
    # ---------------------------------
    if input_mode == "Random":

        if st.sidebar.button("Generate Random Sample"):
            response = get_random_sample()

            if "error" in response:
                st.sidebar.error(response["error"])
            else:
                st.session_state["sample_data"] = response["features"]
                st.session_state["actual_label"] = response["actual_label"]
                st.sidebar.success("Random sample loaded!")

    return {
        "method": selected_model,
        "input_mode": input_mode,
        "sample_data": st.session_state.get("sample_data"),
        "actual_label": st.session_state.get("actual_label")
    }