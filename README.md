# ML for Healthcare & Predictive Analytics: interactive app

[![Streamlit](https://img.shields.io/badge/Streamlit-live-1d6b55?logo=streamlit)](https://ml-healthcare-app.streamlit.app)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Harshini-Suresha/ml-healthcare-app/blob/main/ML_Healthcare_Demo.ipynb)
[![Website](https://img.shields.io/badge/website-github.io-blue)](https://harshini-suresha.github.io/ml-healthcare-app/)
[![nbviewer](https://img.shields.io/badge/view-nbviewer-orange)](https://nbviewer.org/github/Harshini-Suresha/ml-healthcare-app/blob/main/ML_Healthcare_Demo.ipynb)

Multipage Streamlit app + Colab notebook for four supervised-learning projects: diabetes (KNN), breast cancer (KNN vs Logistic Regression), heart disease (neural network) and autism spectrum disorder (three classifiers) — plus a **Compare page** that diffs any two configs A vs B.

> Educational demonstration built from public datasets. Not a medical device and not for clinical decisions.

## Links

| What | Link |
|---|---|
| 🚀 Live Streamlit app | https://ml-healthcare-app.streamlit.app |
| 🌐 Website (GitHub Pages) | https://harshini-suresha.github.io/ml-healthcare-app/ |
| ▶ Colab notebook | https://colab.research.google.com/github/Harshini-Suresha/ml-healthcare-app/blob/main/ML_Healthcare_Demo.ipynb |
| 📓 Notebook file (.ipynb) | [raw .ipynb](https://raw.githubusercontent.com/Harshini-Suresha/ml-healthcare-app/main/ML_Healthcare_Demo.ipynb) · [nbviewer preview](https://nbviewer.org/github/Harshini-Suresha/ml-healthcare-app/blob/main/ML_Healthcare_Demo.ipynb) |
| ⭐ Repo | https://github.com/Harshini-Suresha/ml-healthcare-app |

## What is in this folder

| Path | Purpose |
|---|---|
| `app.py` | Streamlit Home page (entry point) |
| `pages/1_Diabetes.py` | Diabetes: KNN study |
| `pages/2_Breast_Cancer.py` | Breast cancer: KNN vs Logistic Regression |
| `pages/3_Heart_Disease.py` | Heart disease: neural network |
| `pages/4_Autism.py` | Autism: screening-based classification |
| `pages/5_Compare.py` | **Diff view:** A-vs-B comparison with Δ (B − A) tables |
| `ui_common.py` | Shared Streamlit helpers + all live URLs |
| `ml_core.py` | The ML code the app uses (also importable on its own) |
| `ML_Healthcare_Demo.ipynb` | Notebook version for Google Colab / Jupyter |
| `index.html` | Static website served by GitHub Pages |
| `data/` | Datasets (see `data/README.md`; add `autism_train.csv`) |
| `.streamlit/config.toml` | Light green theme |
| `requirements.txt` | Python dependencies |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free)

**Streamlit Community Cloud:** go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**, pick repo `ml-healthcare-app`, branch `main`, main file `app.py`, then **Deploy**. Update `STREAMLIT_URL` in `ui_common.py` + `index.html` if your URL differs.

**GitHub Pages website:** repo Settings → Pages → Deploy from branch `main`, folder `/ (root)`. This serves `index.html` at `https://<user>.github.io/ml-healthcare-app/`.

## Colab

Open the notebook directly from GitHub (works once the repository is public):

`https://colab.research.google.com/github/Harshini-Suresha/ml-healthcare-app/blob/main/ML_Healthcare_Demo.ipynb`

## Notes on fidelity

- Diabetes, breast cancer, heart and autism pipelines follow the project notebooks (scaling, splits, oversampling, grid search). The heart page trains scikit-learn's `MLPClassifier` with the same 13-8-14-1 layout and Keras-style starting weights instead of Keras, to keep the deployed app light.
- Results depend on the data you provide and on random seeds, so numbers can differ slightly from the original notebooks.

## Contact

Harshini Suresha: [harshinisuresha7@gmail.com](mailto:harshinisuresha7@gmail.com) | [GitHub](https://github.com/Harshini-Suresha)
