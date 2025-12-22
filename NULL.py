"""
Data Processing APIs for Telco Customer Churn Dataset
Includes: null value count, outlier count, and data balancing
(undersampling / oversampling)
"""

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# App Initialization
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# -----------------------------------------------------------------------------
# Load Dataset (Once at Startup)
# -----------------------------------------------------------------------------
print("Loading dataset for data processing APIs...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle TotalCharges - it might have empty strings
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Convert SeniorCitizen to string for consistency
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# -----------------------------------------------------------------------------
# APIs
# -----------------------------------------------------------------------------
@app.route('/api/null_value_count', methods=['GET'])
def null_value_count():
    """Count null/NaN values in the dataset"""
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100

    null_data = {}
    for col in df.columns:
        count = int(null_counts[col])
        if count > 0:
            null_data[col] = {
                "count": count,
                "percentage": round(null_percentages[col], 2)
            }

    return jsonify({
        "status": "success",
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns_with_nulls": len(null_data),
        "null_counts": null_data,
        "summary": {
            "total_null_values": int(null_counts.sum()),
            "columns_with_nulls": [col for col in df.columns if null_counts[col] > 0]
        }
    })


@app.route('/api/outlier_count', methods=['GET'])
def outlier_count():
    """Count outliers in numeric columns using IQR method"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if 'customerID' in numeric_cols:
        numeric_cols.remove('customerID')

    outlier_data = {}
    total_outliers = 0

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        count = len(outliers)
        total_outliers += count

        if count > 0:
            outlier_data[col] = {
                "count": int(count),
                "percentage": round((count / len(df)) * 100, 2),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
                "Q1": round(float(Q1), 2),
                "Q3": round(float(Q3), 2),
                "IQR": round(float(IQR), 2)
            }

    return jsonify({
        "status": "success",
        "method": "IQR (Interquartile Range)",
        "total_rows": int(len(df)),
        "numeric_columns_analyzed": numeric_cols,
        "outlier_counts": outlier_data,
        "summary": {
            "total_outliers": int(total_outliers),
            "columns_with_outliers": list(outlier_data.keys())
        }
    })


@app.route('/api/imbalancetobalance(undersampling)', methods=['GET'])
@app.route('/api/imbalancetobalance-undersampling', methods=['GET'])
def imbalance_to_balance_undersampling():
    """Balance imbalanced dataset using undersampling"""
    try:
        from imblearn.under_sampling import RandomUnderSampler

        if 'Churn' not in df.columns:
            return jsonify({
                "status": "error",
                "message": "Churn column not found in dataset"
            }), 400

        original_dist = df['Churn'].value_counts().to_dict()
        original_total = len(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'customerID' in numeric_cols:
            numeric_cols.remove('customerID')

        X = df[numeric_cols].fillna(0)
        y = df['Churn'].map({'Yes': 1, 'No': 0})

        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X, y)

        new_dist = pd.Series(y_resampled).map({1: 'Yes', 0: 'No'}).value_counts().to_dict()
        new_total = len(X_resampled)

        return jsonify({
            "status": "success",
            "method": "Random UnderSampling",
            "original_distribution": {
                "Yes": int(original_dist.get('Yes', 0)),
                "No": int(original_dist.get('No', 0)),
                "total": int(original_total),
                "churn_percentage": round((original_dist.get('Yes', 0) / original_total) * 100, 2),
                "no_churn_percentage": round((original_dist.get('No', 0) / original_total) * 100, 2)
            },
            "balanced_distribution": {
                "Yes": int(new_dist.get('Yes', 0)),
                "No": int(new_dist.get('No', 0)),
                "total": int(new_total),
                "churn_percentage": round((new_dist.get('Yes', 0) / new_total) * 100, 2),
                "no_churn_percentage": round((new_dist.get('No', 0) / new_total) * 100, 2)
            },
            "reduction": {
                "rows_removed": int(original_total - new_total),
                "reduction_percentage": round(((original_total - new_total) / original_total) * 100, 2)
            }
        })

    except ImportError:
        return jsonify({
            "status": "error",
            "message": "imbalanced-learn not installed. Run: pip install imbalanced-learn"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/imbalancetobalance (oversampling)', methods=['GET'])
@app.route('/api/imbalancetobalance-oversampling', methods=['GET'])
def imbalance_to_balance_oversampling():
    """Balance imbalanced dataset using oversampling"""
    try:
        from imblearn.over_sampling import RandomOverSampler

        if 'Churn' not in df.columns:
            return jsonify({
                "status": "error",
                "message": "Churn column not found in dataset"
            }), 400

        original_dist = df['Churn'].value_counts().to_dict()
        original_total = len(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'customerID' in numeric_cols:
            numeric_cols.remove('customerID')

        X = df[numeric_cols].fillna(0)
        y = df['Churn'].map({'Yes': 1, 'No': 0})

        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)

        new_dist = pd.Series(y_resampled).map({1: 'Yes', 0: 'No'}).value_counts().to_dict()
        new_total = len(X_resampled)

        return jsonify({
            "status": "success",
            "method": "Random OverSampling",
            "original_distribution": {
                "Yes": int(original_dist.get('Yes', 0)),
                "No": int(original_dist.get('No', 0)),
                "total": int(original_total),
                "churn_percentage": round((original_dist.get('Yes', 0) / original_total) * 100, 2),
                "no_churn_percentage": round((original_dist.get('No', 0) / original_total) * 100, 2)
            },
            "balanced_distribution": {
                "Yes": int(new_dist.get('Yes', 0)),
                "No": int(new_dist.get('No', 0)),
                "total": int(new_total),
                "churn_percentage": round((new_dist.get('Yes', 0) / new_total) * 100, 2),
                "no_churn_percentage": round((new_dist.get('No', 0) / new_total) * 100, 2)
            },
            "increase": {
                "rows_added": int(new_total - original_total),
                "increase_percentage": round(((new_total - original_total) / original_total) * 100, 2)
            }
        })

    except ImportError:
        return jsonify({
            "status": "error",
            "message": "imbalanced-learn not installed. Run: pip install imbalanced-learn"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# -----------------------------------------------------------------------------
# Run Server
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Starting Data Processing API server...")
    print("API available at http://localhost:5000")
    print("Endpoints:")
    print("  - GET /api/null_value_count")
    print("  - GET /api/outlier_count")
    print("  - GET /api/imbalancetobalance-undersampling")
    print("  - GET /api/imbalancetobalance-oversampling")
    app.run(debug=True, host='0.0.0.0', port=5000)
