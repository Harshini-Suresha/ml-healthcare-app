"""Core ML helpers for the ML for Healthcare & Predictive Analytics app.

Pure scikit-learn / pandas / matplotlib, so the same code can be tested without Streamlit.
Every function mirrors the pipeline in the matching notebook of the four projects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

TEAL, RED, GREY, INK = "#1d6b55", "#c8283b", "#8fa89a", "#13281d"


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    if title: ax.set_title(title, loc="left", fontsize=11, color=INK)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax


def metrics_dict(y_true, y_pred, y_score=None):
    out = {"Accuracy": accuracy_score(y_true, y_pred),
           "Precision": precision_score(y_true, y_pred, zero_division=0),
           "Recall": recall_score(y_true, y_pred, zero_division=0),
           "F1-score": f1_score(y_true, y_pred, zero_division=0)}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["Specificity"] = tn / (tn + fp) if (tn + fp) else 0.0
    if y_score is not None and len(set(y_true)) == 2:
        out["ROC-AUC"] = roc_auc_score(y_true, y_score)
    return out


# ----------------------------------------------------------------------------- plots
def plot_confusion(cm, labels=("No", "Yes"), title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i][j] > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], labels=[f"Pred {l}" for l in labels]); ax.set_yticks([0, 1], labels=[f"True {l}" for l in labels])
    ax.set_title(title, loc="left", fontsize=11, color=INK); fig.tight_layout()
    return fig


def plot_lines(df, x, ys, colors, title, xlabel, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for y, c in zip(ys, colors):
        ax.plot(df[x], df[y], color=c, lw=2, marker="o", ms=3, label=y)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(frameon=False); ax.grid(alpha=.25)
    _style(ax, title, xlabel, ylabel); fig.tight_layout()
    return fig


def plot_bars(labels, values, title, color=TEAL, ylim=(0, 1), fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.bar(labels, values, color=color)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    if ylim: ax.set_ylim(*ylim)
    _style(ax, title); fig.tight_layout()
    return fig


def plot_grouped(labels, a, b, a_label, b_label, title, ylim=(0, 1.08)):
    x = np.arange(len(labels)); fig, ax = plt.subplots(figsize=(6, 3.4))
    for off, vals, c, l in [(-.2, a, GREY, a_label), (.2, b, TEAL, b_label)]:
        bars = ax.bar(x + off, vals, .4, color=c, label=l)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels=labels); ax.set_ylim(*ylim); ax.legend(frameon=False)
    _style(ax, title); fig.tight_layout()
    return fig


def plot_roc(curves, title="ROC curves"):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for (name, fpr, tpr, auc), c in zip(curves, [TEAL, RED, GREY]):
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#b3cdb1"); ax.legend(frameon=False, loc="lower right", fontsize=8)
    _style(ax, title, "False positive rate", "True positive rate"); fig.tight_layout()
    return fig


def plot_corr(df):
    c = df.corr(numeric_only=True); fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(c, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), labels=c.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(c)), labels=c.columns, fontsize=8)
    if len(c) <= 16:
        for i in range(len(c)):
            for j in range(len(c)):
                ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=.8); ax.set_title("Correlation matrix", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- diabetes
DIAB_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                 "DiabetesPedigreeFunction", "Age"]
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ZERO_MODES = ["Keep zeros", "Median impute", "Drop rows"]
SCALERS = ["None", "StandardScaler", "MinMaxScaler"]


def plot_hist_by_outcome(df, feature, bins=20):
    d = df[df[feature] != 0] if feature in ZERO_COLS else df
    edges = np.histogram_bin_edges(d[feature], bins=bins); fig, ax = plt.subplots(figsize=(6, 3.2))
    for o, c, l in [(0, TEAL, "Non-diabetic"), (1, RED, "Diabetic")]:
        ax.hist(d.loc[d["Outcome"] == o, feature], bins=edges, alpha=.65, color=c, label=l)
    ax.legend(frameon=False); _style(ax, f"{feature} by outcome", feature, "Records")
    fig.tight_layout()
    return fig


def zero_counts(df):
    return pd.Series({c: int((df[c] == 0).sum()) for c in ZERO_COLS}).sort_values(ascending=False)


def prepare_diabetes(df, zero_mode="Median impute", scaler="StandardScaler", test_size=0.3, seed=42):
    d = df.copy()
    if zero_mode == "Drop rows":
        d = d[(d[ZERO_COLS] != 0).all(axis=1)]
    X, y = d[DIAB_FEATURES].astype(float), d["Outcome"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    Xtr, Xte = Xtr.copy(), Xte.copy()
    med = None
    if zero_mode == "Median impute":
        med = Xtr[ZERO_COLS].replace(0, np.nan).median()
        for part in (Xtr, Xte):
            part[ZERO_COLS] = part[ZERO_COLS].replace(0, np.nan).fillna(med)
    sc = {"None": None, "StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()}[scaler]
    if sc is not None:
        A, B = sc.fit_transform(Xtr), sc.transform(Xte)
    else:
        A, B = Xtr.values, Xte.values
    return dict(Xtr=A, Xte=B, ytr=ytr.values, yte=yte.values, scaler=sc, median=med, zero_mode=zero_mode)


def fit_knn(df, zero_mode, scaler, k, test_size=0.3, seed=42):
    P = prepare_diabetes(df, zero_mode, scaler, test_size, seed)
    m = KNeighborsClassifier(k).fit(P["Xtr"], P["ytr"]); pred = m.predict(P["Xte"])
    return dict(model=m, prep=P, train_acc=m.score(P["Xtr"], P["ytr"]), test_acc=m.score(P["Xte"], P["yte"]),
                cm=confusion_matrix(P["yte"], pred, labels=[0, 1]),
                report=pd.DataFrame(classification_report(P["yte"], pred, output_dict=True, zero_division=0)).T,
                metrics=metrics_dict(P["yte"], pred, m.predict_proba(P["Xte"])[:, 1]), k=k)


def knn_sweep(df, zero_mode, scaler, test_size=0.3, seed=42, kmax=20):
    P = prepare_diabetes(df, zero_mode, scaler, test_size, seed); rows = []
    for k in range(1, kmax + 1):
        m = KNeighborsClassifier(k).fit(P["Xtr"], P["ytr"])
        rows.append((k, m.score(P["Xtr"], P["ytr"]), m.score(P["Xte"], P["yte"])))
    return pd.DataFrame(rows, columns=["k", "Train accuracy", "Test accuracy"])


def compare_preprocessing(df, k=5, test_size=0.3, seed=42):
    rows = []
    for z in ZERO_MODES:
        for s in SCALERS:
            f = fit_knn(df, z, s, k, test_size, seed)
            rows.append((z, s, f["train_acc"], f["test_acc"], f["train_acc"] - f["test_acc"]))
    return pd.DataFrame(rows, columns=["Zeros", "Scaling", "Train", "Test", "Gap"]).sort_values("Test", ascending=False)


def predict_patient(fit, values: dict, n_neighbors=None):
    P = fit["prep"]; row = pd.DataFrame([values])[DIAB_FEATURES].astype(float)
    if P["zero_mode"] == "Median impute":
        row[ZERO_COLS] = row[ZERO_COLS].replace(0, np.nan).fillna(P["median"])
    x = P["scaler"].transform(row) if P["scaler"] is not None else row.values
    m = fit["model"]; prob = float(m.predict_proba(x)[0, 1]); label = int(m.predict(x)[0])
    dist, idx = m.kneighbors(x, n_neighbors=n_neighbors or m.n_neighbors)
    nb = pd.DataFrame({"Distance": dist[0].round(3), "Outcome": ["Diabetic" if v else "Non-diabetic" for v in P["ytr"][idx[0]]]})
    return dict(label=label, prob=prob, neighbours=nb)


# ----------------------------------------------------------------------------- breast cancer
def load_breast_cancer_df():
    """Wisconsin diagnostic data shipped with scikit-learn. Target 'malignant' = 1."""
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True); df = d.frame.copy()
    df["malignant"] = 1 - df.pop("target")
    return df


def _pipe(model):
    return Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler()), ("m", model)])


def breast_cancer_study(df, target="malignant", positive=1, test_size=0.25, seed=42, kmax=25, folds=5, repeats=2):
    X = df.drop(columns=[target]).select_dtypes("number"); y = (df[target] == positive).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    ks = list(range(1, kmax + 1)); cvacc = [cross_val_score(_pipe(KNeighborsClassifier(k)), Xtr, ytr, cv=cv).mean() for k in ks]
    best_k = ks[int(np.argmax(cvacc))]
    knn = _pipe(KNeighborsClassifier(best_k)).fit(Xtr, ytr)
    grid = GridSearchCV(_pipe(LogisticRegression(max_iter=3000)),
                        {"m__C": [0.01, 0.1, 1, 10, 100], "m__solver": ["lbfgs", "liblinear"]},
                        cv=RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed), scoring="accuracy")
    grid.fit(Xtr, ytr); lr = grid.best_estimator_
    res = {}
    for name, m in [(f"KNN (K={best_k})", knn), ("Logistic Regression", lr)]:
        pred, prob = m.predict(Xte), m.predict_proba(Xte)[:, 1]; fpr, tpr, _ = roc_curve(yte, prob)
        res[name] = dict(metrics=metrics_dict(yte, pred, prob), cm=confusion_matrix(yte, pred, labels=[0, 1]), roc=(fpr, tpr),
                         report=pd.DataFrame(classification_report(yte, pred, output_dict=True, zero_division=0)).T)
    return dict(ks=ks, cv_acc=cvacc, best_k=best_k, best_params=grid.best_params_, lr_cv=grid.best_score_,
                results=res, Xte=Xte, yte=yte, models={f"KNN (K={best_k})": knn, "Logistic Regression": lr})


# ----------------------------------------------------------------------------- heart
HEART_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def check_heart(df):
    miss = [c for c in HEART_FEATURES + ["target"] if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {', '.join(miss)}. Expected the 13 clinical attributes plus 'target'.")


def train_heart(df, test_size=0.3, seed=0, epochs=100, hidden=(8, 14)):
    check_heart(df)
    X, y = df[HEART_FEATURES].values.astype(float), df["target"].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    sc = StandardScaler(); Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
    clf = MLPClassifier(hidden_layer_sizes=tuple(hidden), activation="relu", solver="adam", batch_size=8, random_state=seed)
    # Keras "uniform" initialization (small weights, zero biases) so training behaves like the Keras notebook.
    clf.partial_fit(Xtr[:8], ytr[:8], classes=[0, 1])
    rng = np.random.RandomState(seed)
    clf.coefs_ = [rng.uniform(-0.05, 0.05, w.shape) for w in clf.coefs_]
    clf.intercepts_ = [np.zeros_like(v) for v in clf.intercepts_]
    hist = []
    for e in range(1, epochs + 1):
        clf.partial_fit(Xtr, ytr); hist.append((e, clf.loss_, clf.score(Xtr, ytr)))
    pred = clf.predict(Xte)
    return dict(model=clf, scaler=sc, history=pd.DataFrame(hist, columns=["epoch", "loss", "accuracy"]),
                metrics=metrics_dict(yte, pred, clf.predict_proba(Xte)[:, 1]), cm=confusion_matrix(yte, pred, labels=[0, 1]),
                n_train=len(ytr), n_test=len(yte))


def predict_heart(res, values: dict):
    x = res["scaler"].transform(pd.DataFrame([values])[HEART_FEATURES].astype(float))
    return float(res["model"].predict_proba(x)[0, 1])


# ----------------------------------------------------------------------------- autism
A_SCORES = [f"A{i}_Score" for i in range(1, 11)]


def _age_group(a):
    return "Toddler" if a < 4 else "Kid" if a < 12 else "Teenager" if a < 18 else "Young" if a < 40 else "Senior"


def autism_prepare(raw):
    df = raw.copy()
    if "Class/ASD" not in df.columns and "ASD" in df.columns:
        df = df.rename(columns={"ASD": "Class/ASD"})
    need = ["ID", "age", "gender", "ethnicity", "jaundice", "austim", "used_app_before", "result", "age_desc", "relation", "Class/ASD"] + A_SCORES
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {', '.join(miss)}. Expected the Kaggle 'Autism Prediction' train.csv layout.")
    df = df.replace({"yes": 1, "no": 0, "?": "Others", "others": "Others"})
    df = df[df["result"] > -5].copy()
    df["ageGroup"] = df["age"].apply(_age_group)
    for c in ["jaundice", "austim", "used_app_before"]:
        df[c] = df[c].astype(int)
    df["sum_score"] = df[A_SCORES].sum(axis=1)
    df["ind"] = df["austim"] + df["used_app_before"] + df["jaundice"]
    df["age"] = np.log(df["age"].astype(float))
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    X = df.drop(columns=["ID", "age_desc", "used_app_before", "austim", "Class/ASD"]); y = df["Class/ASD"].astype(int)
    return X, y, df


def oversample(X, y, seed=0):
    X, y = np.asarray(X), np.asarray(y); counts = np.bincount(y); minority = int(np.argmin(counts))
    idx_min, idx_maj = np.where(y == minority)[0], np.where(y != minority)[0]
    extra = resample(idx_min, replace=True, n_samples=len(idx_maj) - len(idx_min), random_state=seed)
    idx = np.concatenate([np.arange(len(y)), extra]); return X[idx], y[idx]


def _boosted():
    try:
        from xgboost import XGBClassifier
        return "XGBoost", XGBClassifier(eval_metric="logloss", random_state=0)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return "Gradient Boosting", HistGradientBoostingClassifier(random_state=0)


def autism_study(raw, test_size=0.2, seed=10):
    X, y, df = autism_prepare(raw)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=test_size, random_state=seed)
    Xo, yo = oversample(Xtr, ytr, seed=0)
    sc = StandardScaler(); Xo_s, Xv_s = sc.fit_transform(Xo), sc.transform(Xv)
    bname, boosted = _boosted()
    models = {"Logistic Regression": LogisticRegression(max_iter=2000), bname: boosted, "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=0)}
    res = {}
    for n, m in models.items():
        m.fit(Xo_s, yo); pv = m.predict(Xv_s)
        res[n] = dict(train_auc=roc_auc_score(yo, m.predict(Xo_s)), val_auc=roc_auc_score(yv, pv),
                      val_auc_prob=roc_auc_score(yv, m.predict_proba(Xv_s)[:, 1]), cm=confusion_matrix(yv, pv, labels=[0, 1]),
                      metrics=metrics_dict(yv, pv, m.predict_proba(Xv_s)[:, 1]))
    return dict(results=res, models=models, scaler=sc, Xv=Xv, Xv_s=Xv_s, yv=yv.values, features=list(X.columns), df=df,
                counts=dict(raw=len(raw), clean=len(df), train=len(ytr), val=len(yv), oversampled=len(yo),
                            pos_train=int(ytr.sum()), pos_val=int(yv.sum())))
