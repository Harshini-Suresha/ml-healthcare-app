"""ML for Healthcare & Predictive Analytics — Home page (multipage Streamlit app).

Run locally:  streamlit run app.py
Pages live in pages/: 1_Diabetes, 2_Breast_Cancer, 3_Heart_Disease, 4_Autism, 5_Compare.
"""
import streamlit as st

from ui_common import (
    AUTHOR_GITHUB, COLAB_URL, EMAIL, GITHUB,
    NB_RAW_URL, NBVIEWER_URL, SITE_URL, STREAMLIT_URL, sidebar_links,
)

st.set_page_config(page_title="ML for Healthcare & Predictive Analytics", layout="wide")

st.sidebar.title("ML for Healthcare")
st.sidebar.write("Pick a project from the **Pages** list above")
sidebar_links()

st.title("ML for Healthcare & Predictive Analytics")
st.write(
    "Interactive companions to four supervised-learning projects: heart disease, diabetes, "
    "breast cancer and autism spectrum disorder. Use the sidebar pages to change settings "
    "and watch the results update — or open the **Compare** page to diff two models side by side."
)

c = st.columns(4)
c[0].metric("Diabetes", "768 records", "KNN study")
c[1].metric("Breast cancer", "569 records", "KNN vs Logistic Regression")
c[2].metric("Heart disease", "13 attributes", "Neural network")
c[3].metric("Autism", "3 models", "ROC-AUC comparison")

st.subheader("Open the project")
l1, l2, l3, l4 = st.columns(4)
l1.link_button("Live Streamlit app", STREAMLIT_URL)
l2.link_button("Open in Colab", COLAB_URL)
l3.link_button("Notebook (.ipynb)", NB_RAW_URL)
l4.link_button("GitHub repo", GITHUB)
st.caption(f"Static site (GitHub Pages): {SITE_URL} · Notebook preview: {NBVIEWER_URL}")

st.subheader("What you can do here")
st.markdown(
    "- **Diabetes:** explore the real records, compare scaling and missing-value strategies, tune k, and score a new patient.\n"
    "- **Breast cancer:** see K chosen by cross-validation, Logistic Regression tuned by grid search, ROC curves and a record-by-record check.\n"
    "- **Heart disease:** train the 13-8-14-1 network yourself and watch the loss and accuracy curves.\n"
    "- **Autism:** run the cleaning, oversampling and three-model comparison, and inspect overfitting.\n"
    "- **Compare:** diff any two configurations (A vs B) and see the metric deltas."
)
st.subheader("Contact")
st.markdown(f"Questions or want to collaborate? Email [{EMAIL}](mailto:{EMAIL}) or visit [GitHub]({AUTHOR_GITHUB}).")
