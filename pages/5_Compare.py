"""Compare page: diff any two configurations (A vs B) side by side."""
import pandas as pd
import streamlit as st

import ml_core as mc
from ui_common import (
    c_autism, c_bc, c_diab_df, c_fit, c_heart,
    get_data, pct, show, sidebar_links,
)

st.set_page_config(page_title="Compare — diff two models", layout="wide")
st.sidebar.title("ML for Healthcare")
sidebar_links()

st.header("Compare: diff two configurations")
st.caption("Pick a project, set Config A and Config B, and see the metric deltas (B − A).")
tab_d, tab_b, tab_h, tab_a = st.tabs(["Diabetes diff", "Breast cancer diff", "Heart diff", "Autism diff"])

# ------------------------------------------------------------------ diabetes
with tab_d:
    st.subheader("Diabetes: preprocessing / k diff")
    df = c_diab_df()
    a, b = st.columns(2)
    with a:
        st.markdown("**Config A**")
        za = st.selectbox("Zeros (A)", mc.ZERO_MODES, index=1, key="d_za")
        sa = st.selectbox("Scaling (A)", mc.SCALERS, index=1, key="d_sa")
        ka = st.slider("k (A)", 1, 25, 5, key="d_ka")
    with b:
        st.markdown("**Config B**")
        zb = st.selectbox("Zeros (B)", mc.ZERO_MODES, index=1, key="d_zb")
        sb = st.selectbox("Scaling (B)", mc.SCALERS, index=2, key="d_sb")
        kb = st.slider("k (B)", 1, 25, 11, key="d_kb")
    test_size = st.slider("Test share", 0.1, 0.5, 0.3, 0.05, key="d_ts")
    seed = st.number_input("Random seed", 0, 999, 42, key="d_seed")
    fa = c_fit(df, za, sa, int(ka), float(test_size), int(seed))
    fb = c_fit(df, zb, sb, int(kb), float(test_size), int(seed))
    mcols = st.columns(4)
    for col, name in zip(mcols, ["Accuracy", "Precision", "Recall", "F1-score"]):
        va, vb = fa["metrics"][name], fb["metrics"][name]
        col.metric(name, f"A {pct(va)} → B {pct(vb)}", f"{(vb - va) * 100:+.1f} pts")
    diff = pd.DataFrame({"A": fa["metrics"], "B": fb["metrics"]})
    diff["Δ (B − A)"] = diff["B"] - diff["A"]
    st.dataframe(diff.round(3))
    l, r = st.columns(2)
    with l:
        show(mc.plot_confusion(fa["cm"], ("No", "Yes"), f"A: {za} / {sa} / k={ka}"))
    with r:
        show(mc.plot_confusion(fb["cm"], ("No", "Yes"), f"B: {zb} / {sb} / k={kb}"))

# -------------------------------------------------------------- breast cancer
with tab_b:
    st.subheader("Breast cancer: KNN vs Logistic Regression diff")
    st.caption("Same split for both models; delta = Logistic Regression − KNN.")
    df = mc.load_breast_cancer_df()
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test share", 0.1, 0.5, 0.25, 0.05, key="b_ts")
    seed = c2.number_input("Random seed", 0, 999, 42, key="b_seed")
    kmax = c3.slider("Largest K to try", 5, 40, 25, key="b_kmax")
    with st.spinner("Cross-validating K and grid-searching Logistic Regression..."):
        s = c_bc(df, "malignant", 1, float(test_size), int(seed), int(kmax))
    names = list(s["results"])
    tbl = pd.DataFrame({n: v["metrics"] for n, v in s["results"].items()}).round(3)
    tbl[f"Δ ({names[1]} − {names[0]})"] = tbl[names[1]] - tbl[names[0]]
    st.dataframe(tbl)
    st.success(f"Best K = {s['best_k']}; LR params C={s['best_params']['m__C']}, solver={s['best_params']['m__solver']}.")
    show(mc.plot_roc([(n, *v["roc"], v["metrics"]["ROC-AUC"]) for n, v in s["results"].items()]))

# --------------------------------------------------------------------- heart
with tab_h:
    st.subheader("Heart disease: training-run diff")
    df = get_data("heart.csv", "Heart disease CSV (13 attributes + target)", "heart_up_cmp")
    if df is None:
        st.stop()
    try:
        mc.check_heart(df)
    except ValueError as e:
        st.error(str(e)); st.stop()
    a, b = st.columns(2)
    with a:
        st.markdown("**Config A**")
        ea = st.slider("Epochs (A)", 10, 300, 50, 10, key="h_ea")
        ha = st.text_input("Hidden layers (A)", "8,14", key="h_ha")
    with b:
        st.markdown("**Config B**")
        eb = st.slider("Epochs (B)", 10, 300, 100, 10, key="h_eb")
        hb = st.text_input("Hidden layers (B)", "8,14", key="h_hb")
    c1, c2 = st.columns(2)
    test_size = c1.slider("Test share", 0.1, 0.5, 0.3, 0.05, key="h_ts")
    seed = c2.number_input("Random seed", 0, 999, 0, key="h_seed")
    try:
        hidden_a = tuple(int(x) for x in ha.split(",") if x.strip())
        hidden_b = tuple(int(x) for x in hb.split(",") if x.strip())
        assert hidden_a and hidden_b and all(0 < h <= 256 for h in (*hidden_a, *hidden_b))
    except Exception:
        st.error("Hidden layers must be like 8,14."); st.stop()
    with st.spinner("Training both networks..."):
        ra = c_heart(df, float(test_size), int(seed), int(ea), hidden_a)
        rb = c_heart(df, float(test_size), int(seed), int(eb), hidden_b)
    mcols = st.columns(5)
    for col, name in zip(mcols, ["Accuracy", "Precision", "Recall", "F1-score", "Specificity"]):
        va, vb = ra["metrics"][name], rb["metrics"][name]
        col.metric(name, f"A {pct(va)} → B {pct(vb)}", f"{(vb - va) * 100:+.1f} pts")
    diff = pd.DataFrame({"A": ra["metrics"], "B": rb["metrics"]})
    diff["Δ (B − A)"] = diff["B"] - diff["A"]
    st.dataframe(diff.round(3))
    l, r = st.columns(2)
    with l:
        show(mc.plot_lines(ra["history"], "epoch", ["loss"], [mc.RED], "A: training loss", "Epoch", "Loss"))
    with r:
        show(mc.plot_lines(rb["history"], "epoch", ["loss"], [mc.TEAL], "B: training loss", "Epoch", "Loss"))

# -------------------------------------------------------------------- autism
with tab_a:
    st.subheader("Autism: model-vs-model diff")
    raw = get_data("autism_train.csv", "Autism screening CSV (Kaggle 'Autism Prediction' train.csv layout)", "asd_up_cmp")
    if raw is None:
        st.stop()
    c1, c2 = st.columns(2)
    test_size = c1.slider("Validation share", 0.1, 0.4, 0.2, 0.05, key="a_ts")
    seed = c2.number_input("Random seed", 0, 999, 10, key="a_seed")
    try:
        with st.spinner("Cleaning, oversampling and training three models..."):
            s = c_autism(raw, float(test_size), int(seed))
    except ValueError as e:
        st.error(str(e)); st.stop()
    names = list(s["results"])
    c1, c2 = st.columns(2)
    ma = c1.selectbox("Model A", names, index=0, key="a_ma")
    mb = c2.selectbox("Model B", names, index=min(1, len(names) - 1), key="a_mb")
    tbl = pd.DataFrame({ma: s["results"][ma]["metrics"], mb: s["results"][mb]["metrics"]}).round(3)
    tbl[f"Δ ({mb} − {ma})"] = tbl[mb] - tbl[ma]
    st.dataframe(tbl)
    l, r = st.columns(2)
    with l:
        show(mc.plot_confusion(s["results"][ma]["cm"], ("No ASD", "ASD"), ma))
    with r:
        show(mc.plot_confusion(s["results"][mb]["cm"], ("No ASD", "ASD"), mb))
