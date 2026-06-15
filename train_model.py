import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from utils import _load_data, DATA_PATH, ensure_data

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def build_preprocessor(X):
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features)
        ]
    )
    return preprocessor

def train_and_save(df_path=DATA_PATH):
    if not ensure_data():
        raise FileNotFoundError("Dataset could not be downloaded.")

    df = _load_data(df_path)
    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }

    metrics = []
    cms = {}

    for name, clf in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="Yes", zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label="Yes", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="Yes", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=["No", "Yes"])
        cms[name] = cm

        metrics.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4)
        })

        joblib.dump(
            pipe,
            os.path.join(ARTIFACT_DIR, f"{name.replace(' ', '_')}_pipeline.joblib")
        )

    metrics_df = pd.DataFrame(metrics).sort_values("F1-Score", ascending=False)
    best_model = metrics_df.iloc[0]["Model"]

    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "best_model_name.joblib"))
    metrics_df.to_csv(os.path.join(ARTIFACT_DIR, "supervised_metrics.csv"), index=False)

    with open(os.path.join(ARTIFACT_DIR, "supervised_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(cms, os.path.join(ARTIFACT_DIR, "confusion_matrices.joblib"))

    return metrics_df, best_model

if __name__ == "__main__":
    metrics_df, best = train_and_save()
    print("Best model:", best)
    print(metrics_df)