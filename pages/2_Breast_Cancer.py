"""Breast cancer page: KNN versus tuned Logistic Regression."""
import pandas as pd
import streamlit as st

import ml_core as mc
from ui_common import DISCLAIMER, c_bc, pct, show, sidebar_links

st.set_page_config(page_title="Breast cancer — KNN vs Logistic Regression", page_icon="🫀", layout="wide")
st.sidebar.title("🫀 ML for Healthcare")
sidebar_links()

st.header("Breast cancer: KNN versus tuned Logistic Regression")
st.caption(DISCLAIMER)
src = st.radio("Data", ["Built-in Wisconsin diagnostic dataset", "Upload my own CSV"], horizontal=True)
if src.startswith("Built-in"):
    df, target, positive = mc.load_breast_cancer_df(), "malignant", 1
else:
    up = st.file_uploader("CSV with a diagnosis column", type="csv", key="bc_up")
    if up is None:
        st.info("Upload a CSV to continue."); st.stop()
    df = pd.read_csv(up)
    target = st.selectbox("Target column", list(df.columns), index=len(df.columns) - 1)
    positive = st.selectbox("Positive (malignant) value", sorted(df[target].dropna().unique().tolist()))
c = st.columns(4)
c[0].metric("Records", len(df)); c[1].metric("Features", df.select_dtypes("number").shape[1] - (target in df.select_dtypes("number").columns))
c[2].metric("Positive", int((df[target] == positive).sum())); c[3].metric("Missing cells", int(df.isna().sum().sum()))
a, b, d = st.columns(3)
test_size = a.slider("Test share", 0.1, 0.5, 0.25, 0.05); seed = b.number_input("Random seed", 0, 999, 42); kmax = d.slider("Largest K to try", 5, 40, 25)
with st.spinner("Cross-validating K and grid-searching Logistic Regression..."):
    s = c_bc(df, target, positive, float(test_size), int(seed), int(kmax))
st.success(f"Best K by cross-validation: {s['best_k']}. Logistic Regression settings: C={s['best_params']['m__C']}, solver={s['best_params']['m__solver']} "
           f"(repeated stratified CV accuracy {pct(s['lr_cv'])}).")
l, r = st.columns(2)
with l:
    show(mc.plot_lines(pd.DataFrame({"K": s["ks"], "CV accuracy": s["cv_acc"]}), "K", ["CV accuracy"], [mc.TEAL], "Cross-validated accuracy by K", "K", "Accuracy"))
with r:
    show(mc.plot_roc([(n, *v["roc"], v["metrics"]["ROC-AUC"]) for n, v in s["results"].items()]))
tbl = pd.DataFrame({n: v["metrics"] for n, v in s["results"].items()}).round(3)
st.subheader("Held-out test results"); st.dataframe(tbl)
cols = st.columns(len(s["results"]))
for col, (n, v) in zip(cols, s["results"].items()):
    with col:
        show(mc.plot_confusion(v["cm"], ("Benign", "Malignant"), n))
st.subheader("Check one test record")
i = st.slider("Test record", 0, len(s["yte"]) - 1, 0)
row = s["Xte"].iloc[[i]]
truth = "malignant" if s["yte"].iloc[i] == 1 else "benign"
o = st.columns(len(s["models"]) + 1); o[0].metric("Actual", truth)
for col, (n, m) in zip(o[1:], s["models"].items()):
    p = float(m.predict_proba(row)[0, 1]); col.metric(n, f"{p * 100:.0f}% malignant", "malignant" if p >= .5 else "benign")
