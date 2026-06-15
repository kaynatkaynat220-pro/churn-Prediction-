import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from utils import ensure_data, load_data
from train_model import train_and_save as train_supervised
from clustering import train_and_save_clusters

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Analytics & Customer Segmentation Platform",
    layout="wide",
    page_icon="🚀"
)

# ------------------------------------------------------------------
# Custom CSS Theme
# ------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');
        .hero-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 3rem;
            text-align: center;
            background: linear-gradient(90deg, #00C9FF, #92FE9D, #00C9FF);
            background-size: 200% auto;
            color: transparent;
            -webkit-background-clip: text;
            background-clip: text;
            animation: shine 3s linear infinite;
        }
        @keyframes shine {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .hero-subtitle {
            text-align: center;
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 20px;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">Telecom Churn Analytics & Customer Segmentation Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Predict churn, compare models, and segment customers with AI</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Navigation Buttons (Renamed)
# ------------------------------------------------------------------
PAGES = {
    "🔮 Predict Churn": "predict",
    "⚔️ Model Arena": "models",
    "📈 Data Insights": "eda",
    "🧬 Customer Segments": "unsupervised",
    "🗺️ Cluster Explorer": "clustering",
    "⚙️ How It Works": "pipeline",
    "📜 Code View": "source"
}

if "page" not in st.session_state:
    st.session_state.page = "predict"

cols = st.columns(len(PAGES))
for i, (label, key) in enumerate(PAGES.items()):
    if cols[i].button(label, use_container_width=True, key=f"nav_{key}"):
        st.session_state.page = key
        st.rerun()

st.divider()

# ------------------------------------------------------------------
# Data & Artifacts
# ------------------------------------------------------------------
data_path = ensure_data()
if not data_path:
    st.warning("Auto-download failed. Please upload the Telco Customer Churn CSV manually.")
    uploaded_fallback = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_fallback is not None:
        data_path = ensure_data(uploaded_fallback)
    else:
        st.stop()

df = load_data(data_path)

if not os.path.exists("artifacts/supervised_metrics.csv"):
    with st.spinner("Training classification models... please wait"):
        train_supervised(data_path)

if not os.path.exists("artifacts/cluster_scores.json"):
    with st.spinner("Training clustering algorithms... please wait"):
        train_and_save_clusters(data_path)

@st.cache_resource
def load_artifacts():
    metrics_df = pd.read_csv("artifacts/supervised_metrics.csv")
    best_name = joblib.load("artifacts/best_model_name.joblib")

    model_files = {
        "Logistic Regression": "artifacts/Logistic_Regression_pipeline.joblib",
        "Decision Tree": "artifacts/Decision_Tree_pipeline.joblib",
        "Random Forest": "artifacts/Random_Forest_pipeline.joblib",
        "K-Nearest Neighbors": "artifacts/K-Nearest_Neighbors_pipeline.joblib",
        "Naive Bayes": "artifacts/Naive_Bayes_pipeline.joblib"
    }
    models = {name: joblib.load(path) for name, path in model_files.items()}
    best_pipe = models[best_name]

    cms = joblib.load("artifacts/confusion_matrices.joblib")

    with open("artifacts/cluster_scores.json") as f:
        cluster_scores = json.load(f)
    labels_dict = joblib.load("artifacts/cluster_labels.joblib")
    X_pca = joblib.load("artifacts/X_pca.joblib")
    with open("artifacts/cluster_summaries.json") as f:
        summaries = json.load(f)

    return metrics_df, best_name, best_pipe, models, cms, cluster_scores, labels_dict, X_pca, summaries

metrics_df, best_name, best_pipe, models, cms, cluster_scores, labels_dict, X_pca, summaries = load_artifacts()

# ------------------------------------------------------------------
# Page: Predict Churn
# ------------------------------------------------------------------
def predict_page():
    st.header("🔮 Predict Customer Churn")
    st.success(f"**Top performing model:** {best_name}")

    selected_model = st.selectbox(
        "Choose a classifier to predict churn",
        list(models.keys()),
        index=list(models.keys()).index(best_name)
    )

    uploaded = st.file_uploader("Drop a customer CSV file here", type=["csv"])
    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
        if "TotalCharges" in input_df.columns:
            input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")

        X = input_df.drop(["Churn", "customerID"], axis=1, errors="ignore")

        pipe = models[selected_model]
        predictions = pipe.predict(X)
        probabilities = pipe.predict_proba(X)
        yes_idx = list(pipe.classes_).index("Yes")

        input_df["Churn Prediction"] = predictions
        input_df["Churn Probability"] = np.round(probabilities[:, yes_idx], 3)

        total = len(input_df)
        churn_count = int((predictions == "Yes").sum())
        churn_rate = churn_count / total * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", total)
        c2.metric("Churners", churn_count)
        c3.metric("Churn Rate", f"{churn_rate:.2f}%")
        c4.metric("Model Used", selected_model)

        tab1, tab2, tab3 = st.tabs(["Predictions", "Model Metrics", "Confusion Matrix"])

        with tab1:
            st.dataframe(input_df.head(50), use_container_width=True)
            fig = px.pie(
                names=["No Churn", "Churn"],
                values=[total - churn_count, churn_count],
                title="Churn Prediction Distribution",
                color=["No Churn", "Churn"],
                color_discrete_map={"No Churn": "#92FE9D", "Churn": "#FF6B6B"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            model_metrics = metrics_df[metrics_df["Model"] == selected_model].reset_index(drop=True)
            st.dataframe(model_metrics, use_container_width=True)
            fig = px.bar(
                model_metrics.T[1:].reset_index(),
                x="index",
                y=0,
                color="index",
                title=f"{selected_model} Metrics",
                labels={"index": "Metric", "0": "Score"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            cm = cms[selected_model]
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["No", "Yes"],
                y=["No", "Yes"],
                labels=dict(x="Predicted", y="Actual"),
                title=f"{selected_model} Confusion Matrix",
                color_continuous_scale="Greens"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        if "Churn" in input_df.columns:
            y_true = input_df["Churn"]
            acc = accuracy_score(y_true, predictions)
            prec = precision_score(y_true, predictions, pos_label="Yes", zero_division=0)
            rec = recall_score(y_true, predictions, pos_label="Yes", zero_division=0)
            f1 = f1_score(y_true, predictions, pos_label="Yes", zero_division=0)
            st.subheader("Evaluation on Uploaded Data")
            st.write({
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4)
            })

        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

# ------------------------------------------------------------------
# Page: Model Arena
# ------------------------------------------------------------------
def models_page():
    st.header("⚔️ Model Arena")
    st.write("Battle of the 5 supervised learning algorithms.")

    styled = metrics_df.style.highlight_max(
        subset=["Accuracy", "Precision", "Recall", "F1-Score"],
        color="#90EE90"
    )
    st.dataframe(styled, use_container_width=True)

    fig = px.bar(
        metrics_df,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-Score"],
        barmode="group",
        title="Algorithm Performance",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_radar = go.Figure()
    for _, row in metrics_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row["Accuracy"], row["Precision"], row["Recall"], row["F1-Score"]],
            theta=["Accuracy", "Precision", "Recall", "F1-Score"],
            fill="toself",
            name=row["Model"]
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Radar Comparison"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Confusion Matrices")
    cols = st.columns(2)
    for idx, (name, cm) in enumerate(cms.items()):
        with cols[idx % 2]:
            fig = px.imshow(
                cm,
                text_auto=True,
                x=["No", "Yes"],
                y=["No", "Yes"],
                title=f"{name}",
                color_continuous_scale="Greens"
            )
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Page: Data Insights (EDA)
# ------------------------------------------------------------------
def eda_page():
    st.header("📈 Data Insights")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", df.shape[0])
    c2.metric("Features", df.shape[1])
    c3.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.2f}%")

    st.dataframe(df.head(10), use_container_width=True)

    tabs = st.tabs(["Churn Overview", "Contract", "Charges", "Tenure", "Payment"])

    with tabs[0]:
        churn_counts = df["Churn"].value_counts()
        fig = px.pie(values=churn_counts.values, names=churn_counts.index, hole=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
        fig = px.bar(contract, barmode="stack", title="Churn by Contract")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        fig = px.histogram(df, x="MonthlyCharges", color="Churn", barmode="overlay", title="Monthly Charges Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        fig = px.box(df, x="Churn", y="tenure", color="Churn", title="Tenure vs Churn")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        payment = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100
        fig = px.bar(payment, barmode="group", title="Churn by Payment Method")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="viridis", ax=ax)
    st.pyplot(fig)

# ------------------------------------------------------------------
# Page: Customer Segments (Unsupervised)
# ------------------------------------------------------------------
def unsupervised_page():
    st.header("🧬 Customer Segments")
    st.markdown("PCA + clustering algorithms reveal hidden customer groups.")

    scores_df = pd.DataFrame([
        {
            "Algorithm": k,
            "Silhouette Score": v["silhouette"],
            "Clusters Found": v["n_clusters"]
        }
        for k, v in cluster_scores.items()
    ])
    st.dataframe(scores_df, use_container_width=True)

    fig = px.bar(
        scores_df,
        x="Algorithm",
        y="Silhouette Score",
        color="Algorithm",
        title="Clustering Algorithm Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

    pca = joblib.load("artifacts/pca.joblib")
    ev = pca.explained_variance_ratio_
    fig = px.pie(
        values=ev,
        names=["PC1", "PC2"],
        title="PCA Explained Variance"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2D Cluster Visualizations")
    for name, labels in labels_dict.items():
        vis_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": labels.astype(str)
        })
        fig = px.scatter(
            vis_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title=f"{name} Clustering",
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Page: Cluster Explorer
# ------------------------------------------------------------------
def clustering_page():
    st.header("🗺️ Cluster Explorer")

    algo = st.selectbox("Pick a clustering technique", list(labels_dict.keys()))
    labels = labels_dict[algo]

    df_c = df.copy()
    df_c["Cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    st.info(f"Detected clusters: {n_clusters}")

    c1, c2 = st.columns(2)
    with c1:
        dist = pd.Series(labels).value_counts().sort_index()
        fig = px.bar(
            x=dist.index.astype(str),
            y=dist.values,
            labels={"x": "Cluster", "y": "Count"},
            title="Cluster Sizes"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        summary_df = pd.DataFrame(summaries[algo])
        st.dataframe(
            summary_df.style.format({
                "Avg_Tenure": "{:.2f}",
                "Avg_MonthlyCharges": "${:.2f}",
                "Avg_TotalCharges": "${:.2f}",
                "Churn_Rate": "{:.2%}"
            }),
            use_container_width=True
        )

    st.subheader("Tenure vs Monthly Charges by Segment")
    fig = px.scatter(
        df_c,
        x="tenure",
        y="MonthlyCharges",
        color=df_c["Cluster"].astype(str),
        size="TotalCharges",
        title="Customer Segments",
        opacity=0.6
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Strategic Recommendations")
    st.markdown("""
    - **New High-Risk Users:** Low tenure + high monthly charges → retention discounts.
    - **Long-Term Loyal:** High tenure + low charges → premium upsell.
    - **Premium Stable:** High tenure + high charges → VIP support.
    - **At-Risk Group:** High churn rate clusters → immediate outreach campaigns.
    """)

# ------------------------------------------------------------------
# Page: How It Works
# ------------------------------------------------------------------
def pipeline_page():
    st.header("⚙️ How It Works")

    steps = [
        "📥 **Data Collection** — Load Telco Customer Churn dataset.",
        "🧹 **Cleaning** — Convert `TotalCharges`, drop missing values, remove `customerID`.",
        "🔧 **Preprocessing** — Scale numerics, encode categoricals with one-hot encoding.",
        "✂️ **Splitting** — 80/20 stratified train-test split.",
        "🤖 **Classification** — Train Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes.",
        "🏆 **Selection** — Best model chosen by F1-Score.",
        "🧬 **Clustering** — PCA + K-Means, Hierarchical, DBSCAN.",
        "📊 **Evaluation** — Silhouette scores and cluster summaries.",
        "🚀 **Deployment** — Interactive Streamlit dashboard."
    ]

    for step in steps:
        st.markdown(f"- {step}")

    st.subheader("Preprocessing Pipeline")
    code = '''
ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])
'''
    st.code(code, language="python")

# ------------------------------------------------------------------
# Page: Code View
# ------------------------------------------------------------------
def source_page():
    st.header("📜 Code View")

    files = {
        "app.py": "app.py",
        "train_model.py": "train_model.py",
        "clustering.py": "clustering.py",
        "utils.py": "utils.py",
        "requirements.txt": "requirements.txt"
    }

    for label, path in files.items():
        with st.expander(f"📄 {label}"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lang = "python" if path.endswith(".py") else "text"
                    st.code(f.read(), language=lang)
            except Exception as e:
                st.error(f"Could not read {path}: {e}")

# ------------------------------------------------------------------
# Route
# ------------------------------------------------------------------
page = st.session_state.page

if page == "predict":
    predict_page()
elif page == "models":
    models_page()
elif page == "eda":
    eda_page()
elif page == "unsupervised":
    unsupervised_page()
elif page == "clustering":
    clustering_page()
elif page == "pipeline":
    pipeline_page()
elif page == "source":
    source_page()

st.divider()
st.markdown(
    "<center><small>Developed for Machine Learning Open Ended Lab | AI-SP-24</small></center>",
    unsafe_allow_html=True
)