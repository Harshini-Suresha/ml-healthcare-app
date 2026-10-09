"""Autism page: screening-based classification."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import ml_core as mc
from ui_common import DISCLAIMER, c_autism, get_data, show, sidebar_links

st.set_page_config(page_title="Autism — screening classification", page_icon="🫀", layout="wide")
st.sidebar.title("🫀 ML for Healthcare")
sidebar_links()

st.header("Autism spectrum disorder: screening-based classification")
st.caption(DISCLAIMER)
raw = get_data("autism_train.csv", "Autism screening CSV (Kaggle 'Autism Prediction' train.csv layout)", "asd_up")
if raw is None:
    st.stop()
a, b = st.columns(2)
test_size = a.slider("Validation share", 0.1, 0.4, 0.2, 0.05); seed = b.number_input("Random seed", 0, 999, 10)
try:
    with st.spinner("Cleaning, oversampling and training three models..."):
        s = c_autism(raw, float(test_size), int(seed))
except ValueError as e:
    st.error(str(e)); st.stop()
n = s["counts"]; c = st.columns(5)
c[0].metric("Raw records", n["raw"]); c[1].metric("After cleaning", n["clean"]); c[2].metric("Training", n["train"])
c[3].metric("Validation", n["val"]); c[4].metric("Oversampled training", n["oversampled"])
names = list(s["results"])
show(mc.plot_grouped(names, [s["results"][x]["train_auc"] for x in names], [s["results"][x]["val_auc"] for x in names],
                     "Train", "Validation", "ROC-AUC computed from predicted labels"))
st.caption("A big gap between train and validation signals overfitting.")
st.dataframe(pd.DataFrame({x: s["results"][x]["metrics"] for x in names}).round(3))
pick = st.selectbox("Model", names)
l, r = st.columns(2)
with l:
    show(mc.plot_confusion(s["results"][pick]["cm"], ("No ASD", "ASD"), pick))
with r:
    d = s["df"]; ct = pd.crosstab(d["sum_score"], d["Class/ASD"])
    fig, ax = plt.subplots(figsize=(5.6, 3.4)); idx = np.arange(len(ct))
    ax.bar(idx - .2, ct.get(0, 0), .4, color=mc.TEAL, label="No ASD"); ax.bar(idx + .2, ct.get(1, 0), .4, color=mc.RED, label="ASD")
    ax.set_xticks(idx, labels=ct.index); ax.legend(frameon=False); mc._style(ax, "Cases by screening sum score", "sum_score", "Records"); fig.tight_layout(); show(fig)
st.subheader("Check one validation record")
i = st.slider("Validation record", 0, len(s["yv"]) - 1, 0)
p = float(s["models"][pick].predict_proba(s["Xv_s"][[i]])[0, 1])
o = st.columns(3); o[0].metric("Actual", "ASD" if s["yv"][i] else "No ASD"); o[1].metric("Predicted probability", f"{p * 100:.0f}%")
o[2].metric("Prediction", "ASD" if p >= .5 else "No ASD")
