"""Shared helpers for the multipage Streamlit app (Home + pages/1..5)."""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import ml_core as mc

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GITHUB = "https://github.com/Harshini-Suresha/ml-healthcare-app"
AUTHOR_GITHUB = "https://github.com/Harshini-Suresha"
EMAIL = "harshinisuresha7@gmail.com"
SITE_URL = "https://harshini-suresha.github.io/ml-healthcare-app/"
STREAMLIT_URL = "https://ml-healthcare-ai.streamlit.app"
COLAB_URL = "https://colab.research.google.com/github/Harshini-Suresha/ml-healthcare-app/blob/main/ML_Healthcare_Demo.ipynb"
NB_RAW_URL = "https://raw.githubusercontent.com/Harshini-Suresha/ml-healthcare-app/main/ML_Healthcare_Demo.ipynb"
NBVIEWER_URL = "https://nbviewer.org/github/Harshini-Suresha/ml-healthcare-app/blob/main/ML_Healthcare_Demo.ipynb"
DISCLAIMER = "Educational demonstration built from public datasets. Not a medical device and not for clinical decisions."


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def get_data(default_file, label, key):
    """Use the bundled CSV if present, otherwise ask for an upload."""
    path = os.path.join(DATA_DIR, default_file)
    if os.path.exists(path):
        return pd.read_csv(path)
    st.info(f"Add `data/{default_file}` to the repository, or upload the file here.")
    up = st.file_uploader(label, type="csv", key=key)
    return pd.read_csv(up) if up is not None else None


def pct(x):
    return f"{x * 100:.1f}%"


def sidebar_links():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"[GitHub repo]({GITHUB})  \n"
        f"[Website]({SITE_URL})  \n"
        f"[Live Streamlit app]({STREAMLIT_URL})  \n"
        f"[Open in Colab]({COLAB_URL})  \n"
        f"[Download notebook (.ipynb)]({NB_RAW_URL})  \n"
        f"[{EMAIL}](mailto:{EMAIL})"
    )


# ---------------------------------------------------------------- cached work
@st.cache_data(show_spinner=False)
def c_diab_df():
    return pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))


@st.cache_data(show_spinner=False)
def c_fit(df, zero_mode, scaler, k, test_size, seed):
    return mc.fit_knn(df, zero_mode, scaler, k, test_size, seed)


@st.cache_data(show_spinner=False)
def c_sweep(df, zero_mode, scaler, test_size, seed, kmax):
    return mc.knn_sweep(df, zero_mode, scaler, test_size, seed, kmax)


@st.cache_data(show_spinner=False)
def c_compare(df, k, test_size, seed):
    return mc.compare_preprocessing(df, k, test_size, seed)


@st.cache_data(show_spinner=False)
def c_bc(df, target, positive, test_size, seed, kmax):
    return mc.breast_cancer_study(df, target, positive, test_size, seed, kmax)


@st.cache_data(show_spinner=False)
def c_heart(df, test_size, seed, epochs, hidden):
    return mc.train_heart(df, test_size, seed, epochs, hidden)


@st.cache_data(show_spinner=False)
def c_autism(df, test_size, seed):
    return mc.autism_study(df, test_size, seed)
