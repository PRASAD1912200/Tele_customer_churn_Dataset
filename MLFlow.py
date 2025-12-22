"""
MLflow Pipeline for Telco Customer Churn Prediction
5 Models: Logistic Regression, Random Forest, Decision Tree, XGBoost, LightGBM
All parameters, metrics, and results are logged to MLflow and saved as tables
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT_NAME = "Telco_Customer_Churn_Prediction"
RESULTS_DIR = Path("mlflow_results")
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def load_and_preprocess_data():
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)

    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"Dataset shape: {df.shape}")

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    if df['SeniorCitizen'].dtype == 'object':
        df['SeniorCitizen'] = df['SeniorCitizen'].map({'No': 0, 'Yes': 1})

    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    label_encoders = {}
    X_encoded = X.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1])
    }

    print(f"\n{model_name} Metrics:")
    for k, v in metrics.items():
        if v is not None:
            print(f"{k}: {v:.4f}")

    return model, metrics, y_pred, y_pred_proba

# =============================================================================
# MLFLOW LOGGING
# =============================================================================
def log_to_mlflow(model, model_name, params, metrics):
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(params)
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)

        if model_name in ["Logistic_Regression", "Random_Forest", "Decision_Tree"]:
            mlflow.sklearn.log_model(model, "model")
        elif model_name == "XGBoost":
            # Save the booster object directly to avoid `_estimator_type` error
            booster = model.get_booster()
            mlflow.xgboost.log_model(booster, "model")
        elif model_name == "LightGBM":
            mlflow.lightgbm.log_model(model, "model")

        mlflow.set_tag("dataset", "Telco Customer Churn")
        mlflow.set_tag("model", model_name)

        return mlflow.active_run().info.run_id

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 80)
    print("MLFLOW PIPELINE FOR TELCO CUSTOMER CHURN")
    print("=" * 80)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    results, metrics_list, predictions = [], [], []

    models = [
        ("Logistic_Regression",
         LogisticRegression(C=1.0, solver='liblinear', max_iter=1000, random_state=RANDOM_STATE)),
        ("Random_Forest",
         RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)),
        ("Decision_Tree",
         DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)),
        ("XGBoost",
         xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                           eval_metric='logloss', random_state=RANDOM_STATE)),
        ("LightGBM",
         lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE))
    ]

    for name, model in models:
        print("\n" + "=" * 80)
        print(f"TRAINING {name}")
        print("=" * 80)

        trained_model, metrics, y_pred, y_proba = train_and_evaluate_model(
            model, name, X_train, X_test, y_train, y_test
        )

        run_id = log_to_mlflow(trained_model, name, model.get_params(), metrics)

        results.append({"model": name, "run_id": run_id, **metrics})
        metrics_list.append({"model": name, "run_id": run_id, **metrics})
        predictions.append({
            "model": name,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_pred_proba": y_proba if y_proba is not None else [None]*len(y_test)
        })

    pd.DataFrame(results).to_csv(RESULTS_DIR / "all_models_results.csv", index=False)
    pd.DataFrame(metrics_list).to_csv(RESULTS_DIR / "all_models_metrics.csv", index=False)

    for p in predictions:
        pd.DataFrame({
            "y_true": p["y_true"],
            "y_pred": p["y_pred"],
            "y_pred_proba": p["y_pred_proba"]
        }).to_csv(RESULTS_DIR / f"{p['model']}_predictions.csv", index=False)

    print("\nPIPELINE COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {RESULTS_DIR}")
    print("Run `mlflow ui` and open http://localhost:5000")

if __name__ == "__main__":
    main()
