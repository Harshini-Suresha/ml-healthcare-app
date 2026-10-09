"""Heart disease page: artificial neural network."""
import streamlit as st

import ml_core as mc
from ui_common import c_heart, get_data, pct, show, sidebar_links

st.set_page_config(page_title="Heart disease — neural network", layout="wide")
st.sidebar.title("ML for Healthcare")
sidebar_links()

st.header("Heart disease: artificial neural network")
st.caption("This page trains a scikit-learn network with the same 13-8-14-1 layout as the Keras notebook.")
df = get_data("heart.csv", "Heart disease CSV (13 attributes + target)", "heart_up")
if df is None:
    st.stop()
try:
    mc.check_heart(df)
except ValueError as e:
    st.error(str(e)); st.stop()
c = st.columns(3); c[0].metric("Patients", len(df)); c[1].metric("With disease", int((df["target"] == 1).sum())); c[2].metric("Attributes", 13)
a, b, d, e = st.columns(4)
epochs = a.slider("Epochs", 10, 300, 100, 10); test_size = b.slider("Test share", 0.1, 0.5, 0.3, 0.05)
seed = d.number_input("Random seed", 0, 999, 0); layers = e.text_input("Hidden layers", "8,14")
try:
    hidden = tuple(int(x) for x in layers.split(",") if x.strip())
    assert hidden and all(0 < h <= 256 for h in hidden)
except Exception:
    st.error("Hidden layers must be positive whole numbers separated by commas, for example 8,14."); st.stop()
if st.button("Train network") or "heart_res" not in st.session_state:
    with st.spinner("Training..."):
        st.session_state["heart_res"] = c_heart(df, float(test_size), int(seed), int(epochs), hidden)
res = st.session_state["heart_res"]
m = res["metrics"]; k = st.columns(5)
for col, name in zip(k, ["Accuracy", "Precision", "Recall", "F1-score", "Specificity"]):
    col.metric(name, pct(m[name]))
h = res["history"]; l, r = st.columns(2)
with l:
    show(mc.plot_lines(h, "epoch", ["loss"], [mc.RED], "Training loss", "Epoch", "Loss"))
with r:
    show(mc.plot_lines(h, "epoch", ["accuracy"], [mc.TEAL], "Training accuracy", "Epoch", "Accuracy", ylim=(0, 1)))
show(mc.plot_confusion(res["cm"], ("No", "Yes"), f"Test set ({res['n_test']} patients)"))
st.subheader("Score a patient")
med = df[mc.HEART_FEATURES].median()
with st.form("heart_form"):
    cols = st.columns(4); v = {}
    for i, f in enumerate(mc.HEART_FEATURES):
        v[f] = cols[i % 4].number_input(f, value=float(med[f]))
    go = st.form_submit_button("Predict")
if go:
    p = mc.predict_heart(res, v)
    (st.error if p >= .5 else st.success)(f"Predicted probability of heart disease: {p * 100:.0f}%")
