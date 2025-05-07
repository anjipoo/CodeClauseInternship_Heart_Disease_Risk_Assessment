# 🫀 Heart Disease Risk Assessment

A Streamlit web application for predicting heart disease risk using the UCI Heart Disease Dataset.  
The app uses machine learning models Random Forest, XGBoost and Gradient Boosting to provide predictions with confidence scores and visualize dataset insights.

---

## Live Demo

🔗 Try the app: [Heart Disease Risk Assessment](https://heartdiseaseriskassessment-m7igrsy32lyddht.streamlit.app/)

---

## Features

- Patient Data Input: 
  Enter parameters like age, sex, cholesterol, blood pressure, and more.

- Heart Disease Prediction:  
  Receive predictions with confidence scores for heart disease likelihood.

- Data Visualizations:
  View target distribution, feature histograms, box plots, correlation heatmap, and feature importance plots.

- Model Performance:
  Display accuracy, precision, recall, and F1-score of the trained model.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/anjipoo/CodeClauseInternship_Heart_Disease_Risk_Assessment.git
cd CodeClauseInternship_Heart_Disease_Risk_Assessment
```
### 2. Create a Virtual Environment and Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the App Locally

```bash
streamlit run app.py
```

---

## Files

- `app.py`
- `preprocess.py`
- `data/heart_disease_uci.csv`
- `visualizations/`
- `models/`:`best_model.pkl`, `scaler.pkl`, `metrics.json`
- `requirements.txt`

---

## Requirements

- Python: 3.12

### Python Dependencies from `requirements.txt`:

- `streamlit`
- `pandas`
- `numpy`
- `joblib`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`

Install all dependencies using:

```bash
pip install -r requirements.txt
```
