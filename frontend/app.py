import streamlit as st
from views.prediction_page import render_prediction_page
from views.comparison_page import render_comparison_page
from views.dataset_explorer import render_dataset_explorer

st.set_page_config(layout="wide")

tab1, tab2, tab3 = st.tabs([
    "Prediction",
    "Model Comparison",
    "Dataset Explorer"
])

with tab1:
    render_prediction_page()

with tab2:
    render_comparison_page()

with tab3:
    render_dataset_explorer()