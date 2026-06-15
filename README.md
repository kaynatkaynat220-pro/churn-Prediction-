# Telecom Churn Analytics & Customer Segmentation Platform

A machine learning dashboard for telecom customer churn prediction and customer segmentation using supervised and unsupervised learning.

## Dataset

- **Kaggle:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- The app auto-downloads the dataset on first run.

## Pages

- **Predict Churn** — Upload CSV and predict churn with selected model
- **Model Arena** — Compare all 5 classification algorithms
- **Data Insights** — EDA and customer behavior patterns
- **Customer Segments** — Compare clustering techniques
- **Cluster Explorer** — Analyze cluster behavior
- **How It Works** — View ML pipeline
- **Code View** — View source code

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
