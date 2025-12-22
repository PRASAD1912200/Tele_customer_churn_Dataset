"""
Flask API for Telco Customer Churn EDA
Returns JSON data for all visualizations
"""

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# App Initialization
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Data cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def churn_crosstab(column):
    table = pd.crosstab(df[column], df["Churn"])
    return {
        "categories": table.index.tolist(),
        "series": [
            {"name": "No", "data": table.get("No", []).tolist()},
            {"name": "Yes", "data": table.get("Yes", []).tolist()}
        ]
    }

def churn_rate(column):
    rate = df.groupby(column)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    )
    return {
        "labels": rate.index.tolist(),
        "values": [round(v, 2) for v in rate.values.tolist()],
        "ylabel": "Churn Rate (%)"
    }

def histogram_data(series, bins=50):
    hist, edges = np.histogram(series.dropna(), bins=bins)
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
    return {
        "x": centers,
        "y": hist.tolist()
    }

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return jsonify({
        "message": "Telco Customer Churn EDA API",
        "status": "running"
    })

@app.route("/health")
def health():
    return jsonify({"status": "UP"})

# -----------------------------------------------------------------------------
# EDA Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/churn-distribution")
def churn_distribution():
    counts = df["Churn"].value_counts()
    return jsonify({
        "chart_type": "pie",
        "title": "Customer Churn Distribution",
        "data": {
            "labels": counts.index.tolist(),
            "values": counts.values.tolist()
        }
    })

@app.route("/api/churn-count")
def churn_count():
    counts = df["Churn"].value_counts()
    return jsonify({
        "chart_type": "bar",
        "title": "Churn Count",
        "data": {
            "labels": counts.index.tolist(),
            "values": counts.values.tolist()
        }
    })

@app.route("/api/gender-churn")
def gender_churn():
    return jsonify({
        "chart_type": "grouped_bar",
        "title": "Gender vs Churn",
        "data": churn_crosstab("gender")
    })

@app.route("/api/senior-citizen-churn")
def senior_churn():
    return jsonify({
        "chart_type": "grouped_bar",
        "title": "Senior Citizen vs Churn",
        "data": churn_crosstab("SeniorCitizen")
    })

@app.route("/api/partner-dependents")
def partner_dependents():
    return jsonify({
        "chart_type": "multi_grouped_bar",
        "title": "Partner & Dependents vs Churn",
        "data": {
            "partner": churn_crosstab("Partner"),
            "dependents": churn_crosstab("Dependents")
        }
    })

@app.route("/api/tenure-distribution")
def tenure_distribution():
    return jsonify({
        "chart_type": "multi_chart",
        "title": "Tenure Distribution",
        "data": {
            "histogram": histogram_data(df["tenure"]),
            "boxplot": {
                "labels": ["No Churn", "Churn"],
                "data": [
                    df[df["Churn"] == "No"]["tenure"].dropna().tolist(),
                    df[df["Churn"] == "Yes"]["tenure"].dropna().tolist()
                ]
            }
        }
    })

@app.route("/api/monthly-charges")
def monthly_charges():
    return jsonify({
        "chart_type": "multi_chart",
        "title": "Monthly Charges Distribution",
        "data": {
            "histogram": histogram_data(df["MonthlyCharges"]),
            "boxplot": {
                "labels": ["No Churn", "Churn"],
                "data": [
                    df[df["Churn"] == "No"]["MonthlyCharges"].dropna().tolist(),
                    df[df["Churn"] == "Yes"]["MonthlyCharges"].dropna().tolist()
                ]
            }
        }
    })

@app.route("/api/total-charges")
def total_charges():
    clean = df.dropna(subset=["TotalCharges"])
    return jsonify({
        "chart_type": "multi_chart",
        "title": "Total Charges Distribution",
        "data": {
            "histogram": histogram_data(clean["TotalCharges"]),
            "boxplot": {
                "labels": ["No Churn", "Churn"],
                "data": [
                    clean[clean["Churn"] == "No"]["TotalCharges"].tolist(),
                    clean[clean["Churn"] == "Yes"]["TotalCharges"].tolist()
                ]
            }
        }
    })

@app.route("/api/contract-churn")
def contract_churn():
    return jsonify({
        "chart_type": "grouped_bar",
        "title": "Contract Type vs Churn",
        "data": churn_crosstab("Contract")
    })

@app.route("/api/payment-method-churn")
def payment_churn():
    return jsonify({
        "chart_type": "grouped_bar",
        "title": "Payment Method vs Churn",
        "data": churn_crosstab("PaymentMethod")
    })

@app.route("/api/internet-service-churn")
def internet_churn():
    return jsonify({
        "chart_type": "grouped_bar",
        "title": "Internet Service vs Churn",
        "data": churn_crosstab("InternetService")
    })

@app.route("/api/service-features")
def service_features():
    features = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "PaperlessBilling"
    ]

    return jsonify({
        "chart_type": "multi_grouped_bar",
        "title": "Service Features vs Churn",
        "data": {f: churn_crosstab(f) for f in features}
    })

@app.route("/api/correlation-heatmap")
def correlation_heatmap():
    numeric_cols = df.select_dtypes(include=np.number)
    corr = numeric_cols.corr()
    return jsonify({
        "chart_type": "heatmap",
        "title": "Correlation Heatmap",
        "data": {
            "labels": corr.columns.tolist(),
            "values": corr.values.tolist()
        }
    })

@app.route("/api/tenure-vs-monthly-charges")
def tenure_vs_monthly():
    return jsonify({
        "chart_type": "scatter",
        "title": "Tenure vs Monthly Charges",
        "data": [
            {
                "name": "No Churn",
                "x": df[df["Churn"] == "No"]["tenure"].tolist(),
                "y": df[df["Churn"] == "No"]["MonthlyCharges"].tolist()
            },
            {
                "name": "Churn",
                "x": df[df["Churn"] == "Yes"]["tenure"].tolist(),
                "y": df[df["Churn"] == "Yes"]["MonthlyCharges"].tolist()
            }
        ]
    })

@app.route("/api/churn-rates")
def churn_rates():
    return jsonify({
        "chart_type": "multi_bar",
        "title": "Churn Rates",
        "data": {
            "contract": churn_rate("Contract"),
            "payment_method": churn_rate("PaymentMethod")
        }
    })

@app.route("/api/statistical-summary")
def statistical_summary():
    metrics = ["tenure", "MonthlyCharges", "TotalCharges"]
    summary = {}

    for m in metrics:
        summary[m] = {
            "No Churn": round(df[df["Churn"] == "No"][m].mean(), 2),
            "Churn": round(df[df["Churn"] == "Yes"][m].mean(), 2)
        }

    return jsonify({
        "chart_type": "summary",
        "title": "Statistical Summary",
        "data": summary
    })

@app.route("/api/dataset-info")
def dataset_info():
    return jsonify({
        "rows": df.shape[0],
        "columns": df.shape[1],
        "churn_yes": int((df["Churn"] == "Yes").sum()),
        "churn_no": int((df["Churn"] == "No").sum())
    })

# -----------------------------------------------------------------------------
# Error Handling
# -----------------------------------------------------------------------------
@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
# Run Server
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
