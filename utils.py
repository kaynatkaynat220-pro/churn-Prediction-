import os
import urllib.request
import pandas as pd
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "telco.csv")

def ensure_data(uploaded_file=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    if uploaded_file is not None:
        with open(DATA_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return DATA_PATH
    if not os.path.exists(DATA_PATH):
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        except Exception as e:
            st.error(f"Failed to auto-download dataset: {e}")
            return None
    return DATA_PATH

def _load_data(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def load_data(path):
    return _load_data(path)