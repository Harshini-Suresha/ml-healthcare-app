"""Diabetes page: K-Nearest Neighbors study."""
import streamlit as st

import ml_core as mc
from ui_common import DISCLAIMER, c_compare, c_diab_df, c_fit, c_sweep, pct, show, sidebar_links

st.set_page_config(page_title="Diabetes — KNN study", page_icon="🫀", layout="wide")
st.sidebar.title("🫀 ML for Healthcare")
sidebar_links()

st.header("Diabetes: K-Nearest Neighbors study")
st.caption(DISCLAIMER)
df = c_diab_df()
t1, t2, t3 = st.tabs(["Explore the data", "Model lab", "Score a patient"])

with t1:
    c = st.columns(4)
    c[0].metric("Records", len(df)); c[1].metric("Diabetic", int(df["Outcome"].sum()))
    c[2].metric("Non-diabetic", int((df["Outcome"] == 0).sum())); c[3].metric("Features", 8)
    feat = st.selectbox("Feature", mc.DIAB_FEATURES, index=1)
    l, r = st.columns(2)
    with l:
        show(mc.plot_hist_by_outcome(df, feat))
        if feat in mc.ZERO_COLS:
            st.caption(f"{int((df[feat] == 0).sum())} records with a zero (a missing value) are left out of this chart.")
    with r:
        show(mc.plot_corr(df))
    z = mc.zero_counts(df)
    show(mc.plot_bars(list(z.index), list(z.values), "Zeros that mean missing (records)", color=mc.RED, ylim=None, fmt="{:.0f}"))

with t2:
    a, b, c, d, e = st.columns(5)
    zero_mode = a.selectbox("Zeros", mc.ZERO_MODES, index=1)
    scaler = b.selectbox("Scaling", mc.SCALERS, index=1)
    k = c.slider("k", 1, 25, 5)
    test_size = d.slider("Test share", 0.1, 0.5, 0.3, 0.05)
    seed = e.number_input("Random seed", 0, 999, 42)
    fit = c_fit(df, zero_mode, scaler, int(k), float(test_size), int(seed))
    m = st.columns(4)
    m[0].metric("Train accuracy", pct(fit["train_acc"])); m[1].metric("Test accuracy", pct(fit["test_acc"]))
    m[2].metric("Recall (diabetic)", pct(fit["metrics"]["Recall"])); m[3].metric("Gap", f"{(fit['train_acc'] - fit['test_acc']) * 100:.1f} pts")
    l, r = st.columns(2)
    with l:
        show(mc.plot_confusion(fit["cm"], ("No", "Yes")))
    with r:
        sweep = c_sweep(df, zero_mode, scaler, float(test_size), int(seed), 25)
        show(mc.plot_lines(sweep, "k", ["Train accuracy", "Test accuracy"], [mc.GREY, mc.TEAL], "Accuracy by k", "k", "Accuracy"))
    st.dataframe(fit["report"].round(3))
    st.subheader("Every preprocessing combination at this k")
    st.dataframe(c_compare(df, int(k), float(test_size), int(seed)).round(3))
    st.caption("Tip: open the **Compare** page to diff two (zeros, scaling, k) settings side by side.")

with t3:
    st.write("Enter measurements. The model uses median-imputed zeros, StandardScaler and the k below.")
    k2 = st.slider("k for scoring", 1, 25, 7, key="k2")
    fit2 = c_fit(df, "Median impute", "StandardScaler", int(k2), 0.3, 42)
    med = df[mc.DIAB_FEATURES].median()
    with st.form("patient"):
        cols = st.columns(4); v = {}
        v["Pregnancies"] = cols[0].number_input("Pregnancies", 0, 20, int(med["Pregnancies"]))
        v["Glucose"] = cols[1].number_input("Glucose", 0, 250, int(med["Glucose"]))
        v["BloodPressure"] = cols[2].number_input("Blood pressure", 0, 140, int(med["BloodPressure"]))
        v["SkinThickness"] = cols[3].number_input("Skin thickness", 0, 100, int(med["SkinThickness"]))
        v["Insulin"] = cols[0].number_input("Insulin", 0, 900, int(med["Insulin"]))
        v["BMI"] = cols[1].number_input("BMI", 0.0, 70.0, float(med["BMI"]))
        v["DiabetesPedigreeFunction"] = cols[2].number_input("Pedigree function", 0.0, 2.5, float(med["DiabetesPedigreeFunction"]))
        v["Age"] = cols[3].number_input("Age", 18, 100, int(med["Age"]))
        go = st.form_submit_button("Score patient")
    if go:
        out = mc.predict_patient(fit2, v)
        (st.error if out["label"] else st.success)(
            f"Predicted: {'diabetic' if out['label'] else 'non-diabetic'} ({out['prob'] * 100:.0f}% of the {k2} nearest records are diabetic)")
        st.write("The records that voted:")
        st.dataframe(out["neighbours"])
