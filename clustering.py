import os
import json
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from utils import _load_data, DATA_PATH, ensure_data
from train_model import build_preprocessor

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def train_and_save_clusters(df_path=DATA_PATH):
    if not ensure_data():
        raise FileNotFoundError("Dataset could not be downloaded.")

    df = _load_data(df_path)
    X = df.drop(["Churn", "customerID"], axis=1)

    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)

    joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "cluster_preprocessor.joblib"))
    joblib.dump(X_processed, os.path.join(ARTIFACT_DIR, "cluster_features.joblib"))

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_processed)

    joblib.dump(pca, os.path.join(ARTIFACT_DIR, "pca.joblib"))
    joblib.dump(X_pca, os.path.join(ARTIFACT_DIR, "X_pca.joblib"))

    algorithms = {
        "K-Means": KMeans(n_clusters=3, random_state=42, n_init=10),
        "Hierarchical": AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward"),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=10)
    }

    labels_dict = {}
    scores = {}

    for name, algo in algorithms.items():
        labels = algo.fit_predict(X_pca)
        labels_dict[name] = labels

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters > 1:
            sil_score = silhouette_score(X_pca, labels)
        else:
            sil_score = -1.0

        scores[name] = {
            "silhouette": round(sil_score, 4),
            "n_clusters": n_clusters
        }

    joblib.dump(labels_dict, os.path.join(ARTIFACT_DIR, "cluster_labels.joblib"))

    with open(os.path.join(ARTIFACT_DIR, "cluster_scores.json"), "w") as f:
        json.dump(scores, f, indent=2)

    summaries = {}
    for name, labels in labels_dict.items():
        df_c = df.copy()
        df_c["Cluster"] = labels
        summary = df_c.groupby("Cluster").agg(
            Avg_Tenure=("tenure", "mean"),
            Avg_MonthlyCharges=("MonthlyCharges", "mean"),
            Avg_TotalCharges=("TotalCharges", "mean"),
            Churn_Rate=("Churn", lambda x: (x == "Yes").mean())
        ).reset_index()
        summaries[name] = summary.to_dict(orient="records")

    with open(os.path.join(ARTIFACT_DIR, "cluster_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    return scores, summaries

if __name__ == "__main__":
    print(train_and_save_clusters())