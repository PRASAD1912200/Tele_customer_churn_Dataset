"""
Comprehensive Exploratory Data Analysis (EDA) Report
Telco Customer Churn Dataset
All visualizations will be saved as PNG files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create directory for saving graphs
output_dir = Path("eda_graphs")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - TELCO CUSTOMER CHURN DATASET")
print("=" * 80)

# Load the dataset
print("\n1. Loading dataset...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {df.shape[1]}")
print(f"   Rows: {df.shape[0]}")

# Basic information
print("\n2. Dataset Information:")
print(df.info())

# Display first few rows
print("\n3. First 5 rows:")
print(df.head())

# Check for missing values
print("\n4. Missing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")

# Check for empty strings (which might represent missing data)
print("\n5. Checking for empty strings...")
for col in df.columns:
    empty_count = (df[col] == ' ').sum() if df[col].dtype == 'object' else 0
    if empty_count > 0:
        print(f"   {col}: {empty_count} empty strings")

# Handle TotalCharges - it might have empty strings
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(f"\n   TotalCharges: {df['TotalCharges'].isnull().sum()} missing values after conversion")

# Basic statistics
print("\n6. Basic Statistics:")
print(df.describe())

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

# 1. Target Variable Distribution (Churn)
print("\n1. Creating Churn Distribution plot...")
plt.figure(figsize=(10, 6))
churn_counts = df['Churn'].value_counts()
colors = ['#2ecc71', '#e74c3c']
plt.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
plt.title('Customer Churn Distribution', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / '01_churn_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/01_churn_distribution.png")

# 2. Churn Count Bar Plot
print("\n2. Creating Churn Count Bar plot...")
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x='Churn', palette=['#2ecc71', '#e74c3c'])
plt.title('Customer Churn Count', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Churn', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / '02_churn_count.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/02_churn_count.png")

# 3. Gender Distribution by Churn
print("\n3. Creating Gender vs Churn plot...")
plt.figure(figsize=(12, 6))
churn_gender = pd.crosstab(df['gender'], df['Churn'])
churn_gender.plot(kind='bar', color=['#3498db', '#e74c3c'], edgecolor='black')
plt.title('Gender Distribution by Churn Status', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Gender', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.legend(title='Churn', fontsize=11)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / '03_gender_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/03_gender_churn.png")

# 4. Senior Citizen Distribution by Churn
print("\n4. Creating Senior Citizen vs Churn plot...")
plt.figure(figsize=(12, 6))
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
senior_churn = pd.crosstab(df['SeniorCitizen'], df['Churn'])
senior_churn.plot(kind='bar', color=['#3498db', '#e74c3c'], edgecolor='black')
plt.title('Senior Citizen Distribution by Churn Status', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Senior Citizen', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.legend(title='Churn', fontsize=11)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / '04_senior_citizen_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/04_senior_citizen_churn.png")

# 5. Partner and Dependents Analysis
print("\n5. Creating Partner & Dependents analysis...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Partner
partner_churn = pd.crosstab(df['Partner'], df['Churn'])
partner_churn.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'], edgecolor='black')
axes[0].set_title('Partner Status by Churn', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Partner', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[0].legend(title='Churn', fontsize=10)
axes[0].tick_params(axis='x', rotation=0)

# Dependents
dependents_churn = pd.crosstab(df['Dependents'], df['Churn'])
dependents_churn.plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'], edgecolor='black')
axes[1].set_title('Dependents Status by Churn', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Dependents', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[1].legend(title='Churn', fontsize=10)
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(output_dir / '05_partner_dependents.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/05_partner_dependents.png")

# 6. Tenure Distribution
print("\n6. Creating Tenure Distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
axes[0].hist(df['tenure'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
axes[0].set_title('Tenure Distribution (Histogram)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tenure (months)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Box plot by Churn
df.boxplot(column='tenure', by='Churn', ax=axes[1], grid=True)
axes[1].set_title('Tenure Distribution by Churn Status', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Churn', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Tenure (months)', fontsize=11, fontweight='bold')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig(output_dir / '06_tenure_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/06_tenure_distribution.png")

# 7. Monthly Charges Distribution
print("\n7. Creating Monthly Charges Distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
axes[0].hist(df['MonthlyCharges'], bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
axes[0].set_title('Monthly Charges Distribution (Histogram)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Monthly Charges ($)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Box plot by Churn
df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1], grid=True)
axes[1].set_title('Monthly Charges by Churn Status', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Churn', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Monthly Charges ($)', fontsize=11, fontweight='bold')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig(output_dir / '07_monthly_charges.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/07_monthly_charges.png")

# 8. Total Charges Distribution
print("\n8. Creating Total Charges Distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Remove NaN values for plotting
total_charges_clean = df['TotalCharges'].dropna()

# Histogram
axes[0].hist(total_charges_clean, bins=50, color='#f39c12', edgecolor='black', alpha=0.7)
axes[0].set_title('Total Charges Distribution (Histogram)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Total Charges ($)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Box plot by Churn
df_clean = df.dropna(subset=['TotalCharges'])
df_clean.boxplot(column='TotalCharges', by='Churn', ax=axes[1], grid=True)
axes[1].set_title('Total Charges by Churn Status', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Churn', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Total Charges ($)', fontsize=11, fontweight='bold')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig(output_dir / '08_total_charges.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/08_total_charges.png")

# 9. Contract Type Analysis
print("\n9. Creating Contract Type analysis...")
plt.figure(figsize=(14, 6))
contract_churn = pd.crosstab(df['Contract'], df['Churn'])
contract_churn.plot(kind='bar', color=['#3498db', '#e74c3c'], edgecolor='black')
plt.title('Contract Type Distribution by Churn Status', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Contract Type', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.legend(title='Churn', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / '09_contract_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/09_contract_churn.png")

# 10. Payment Method Analysis
print("\n10. Creating Payment Method analysis...")
plt.figure(figsize=(14, 6))
payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'])
payment_churn.plot(kind='bar', color=['#3498db', '#e74c3c'], edgecolor='black')
plt.title('Payment Method Distribution by Churn Status', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Payment Method', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.legend(title='Churn', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / '10_payment_method_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/10_payment_method_churn.png")

# 11. Internet Service Analysis
print("\n11. Creating Internet Service analysis...")
plt.figure(figsize=(14, 6))
internet_churn = pd.crosstab(df['InternetService'], df['Churn'])
internet_churn.plot(kind='bar', color=['#3498db', '#e74c3c'], edgecolor='black')
plt.title('Internet Service Type Distribution by Churn Status', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Internet Service', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.legend(title='Churn', fontsize=11)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / '11_internet_service_churn.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/11_internet_service_churn.png")

# 12. Service Features Analysis (Phone, Multiple Lines, etc.)
print("\n12. Creating Service Features analysis...")
service_features = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                    'PaperlessBilling']

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for idx, feature in enumerate(service_features):
    if idx < len(axes):
        feature_churn = pd.crosstab(df[feature], df['Churn'])
        feature_churn.plot(kind='bar', ax=axes[idx], color=['#3498db', '#e74c3c'], edgecolor='black')
        axes[idx].set_title(f'{feature} by Churn', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Count', fontsize=9)
        axes[idx].legend(title='Churn', fontsize=8)
        axes[idx].tick_params(axis='x', rotation=45, labelsize=8)

plt.tight_layout()
plt.savefig(output_dir / '12_service_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/12_service_features.png")

# 13. Correlation Heatmap
print("\n13. Creating Correlation Heatmap...")
# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'customerID' in numeric_cols:
    numeric_cols.remove('customerID')

corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Numeric Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / '13_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/13_correlation_heatmap.png")

# 14. Tenure vs Monthly Charges Scatter Plot
print("\n14. Creating Tenure vs Monthly Charges scatter plot...")
plt.figure(figsize=(12, 8))
churn_yes = df[df['Churn'] == 'Yes']
churn_no = df[df['Churn'] == 'No']

plt.scatter(churn_no['tenure'], churn_no['MonthlyCharges'], 
           alpha=0.5, color='#2ecc71', label='No Churn', s=30)
plt.scatter(churn_yes['tenure'], churn_yes['MonthlyCharges'], 
           alpha=0.5, color='#e74c3c', label='Churn', s=30)
plt.xlabel('Tenure (months)', fontsize=12, fontweight='bold')
plt.ylabel('Monthly Charges ($)', fontsize=12, fontweight='bold')
plt.title('Tenure vs Monthly Charges by Churn Status', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '14_tenure_vs_monthly_charges.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/14_tenure_vs_monthly_charges.png")

# 15. Churn Rate by Contract and Payment Method
print("\n15. Creating Churn Rate by Contract and Payment Method...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Contract
contract_churn_rate = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
contract_churn_rate.plot(kind='bar', ax=axes[0], color='#e74c3c', edgecolor='black')
axes[0].set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Contract Type', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Churn Rate (%)', fontsize=11, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(contract_churn_rate):
    axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)

# Payment Method
payment_churn_rate = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
payment_churn_rate.plot(kind='bar', ax=axes[1], color='#e74c3c', edgecolor='black')
axes[1].set_title('Churn Rate by Payment Method', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Payment Method', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Churn Rate (%)', fontsize=11, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(payment_churn_rate):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / '15_churn_rates.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/15_churn_rates.png")

# 16. Statistical Summary by Churn
print("\n16. Creating Statistical Summary by Churn...")
numeric_summary = df.groupby('Churn')[['tenure', 'MonthlyCharges', 'TotalCharges']].agg(['mean', 'median', 'std'])
print("\n   Statistical Summary by Churn:")
print(numeric_summary)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ['tenure', 'MonthlyCharges', 'TotalCharges']
titles = ['Average Tenure', 'Average Monthly Charges', 'Average Total Charges']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    churn_yes_mean = df[df['Churn'] == 'Yes'][metric].mean()
    churn_no_mean = df[df['Churn'] == 'No'][metric].mean()
    
    bars = axes[idx].bar(['No Churn', 'Churn'], [churn_no_mean, churn_yes_mean], 
                         color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Value', fontsize=10, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.2f}', ha='center', va='bottom', 
                      fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / '16_statistical_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/16_statistical_summary.png")

# 17. Distribution of all numeric features
print("\n17. Creating Distribution plots for all numeric features...")
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, feature in enumerate(numeric_features):
    churn_yes = df[df['Churn'] == 'Yes'][feature].dropna()
    churn_no = df[df['Churn'] == 'No'][feature].dropna()
    
    axes[idx].hist(churn_no, bins=30, alpha=0.6, label='No Churn', color='#2ecc71', edgecolor='black')
    axes[idx].hist(churn_yes, bins=30, alpha=0.6, label='Churn', color='#e74c3c', edgecolor='black')
    axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feature, fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Frequency', fontsize=10, fontweight='bold')
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '17_numeric_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("   [OK] Saved: eda_graphs/17_numeric_distributions.png")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING SUMMARY REPORT...")
print("=" * 80)

# Create summary statistics
summary_report = []
summary_report.append("=" * 80)
summary_report.append("EDA SUMMARY REPORT - TELCO CUSTOMER CHURN DATASET")
summary_report.append("=" * 80)
summary_report.append(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
summary_report.append(f"\nTarget Variable: Churn")
summary_report.append(f"\nChurn Distribution:")
summary_report.append(f"  - No Churn: {(df['Churn'] == 'No').sum()} ({(df['Churn'] == 'No').sum()/len(df)*100:.2f}%)")
summary_report.append(f"  - Churn: {(df['Churn'] == 'Yes').sum()} ({(df['Churn'] == 'Yes').sum()/len(df)*100:.2f}%)")

summary_report.append("\n" + "-" * 80)
summary_report.append("KEY INSIGHTS:")
summary_report.append("-" * 80)

# Churn by Contract
contract_insights = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
summary_report.append(f"\n1. Churn Rate by Contract Type:")
for contract, rate in contract_insights.items():
    summary_report.append(f"   - {contract}: {rate:.2f}%")

# Churn by Payment Method
payment_insights = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
summary_report.append(f"\n2. Churn Rate by Payment Method:")
for payment, rate in payment_insights.items():
    summary_report.append(f"   - {payment}: {rate:.2f}%")

# Churn by Internet Service
internet_insights = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
summary_report.append(f"\n3. Churn Rate by Internet Service:")
for service, rate in internet_insights.items():
    summary_report.append(f"   - {service}: {rate:.2f}%")

# Average values
summary_report.append(f"\n4. Average Values by Churn Status:")
summary_report.append(f"   - Average Tenure (No Churn): {df[df['Churn'] == 'No']['tenure'].mean():.2f} months")
summary_report.append(f"   - Average Tenure (Churn): {df[df['Churn'] == 'Yes']['tenure'].mean():.2f} months")
summary_report.append(f"   - Average Monthly Charges (No Churn): ${df[df['Churn'] == 'No']['MonthlyCharges'].mean():.2f}")
summary_report.append(f"   - Average Monthly Charges (Churn): ${df[df['Churn'] == 'Yes']['MonthlyCharges'].mean():.2f}")

total_charges_no = df[df['Churn'] == 'No']['TotalCharges'].mean()
total_charges_yes = df[df['Churn'] == 'Yes']['TotalCharges'].mean()
if not pd.isna(total_charges_no):
    summary_report.append(f"   - Average Total Charges (No Churn): ${total_charges_no:.2f}")
if not pd.isna(total_charges_yes):
    summary_report.append(f"   - Average Total Charges (Churn): ${total_charges_yes:.2f}")

summary_report.append("\n" + "-" * 80)
summary_report.append("VISUALIZATIONS GENERATED:")
summary_report.append("-" * 80)
summary_report.append("All graphs have been saved in the 'eda_graphs' directory as PNG files.")
summary_report.append("=" * 80)

# Save summary report
with open('eda_summary_report.txt', 'w') as f:
    f.write('\n'.join(summary_report))

# Print summary report
print('\n'.join(summary_report))

print("\n" + "=" * 80)
print("EDA REPORT COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n[OK] All visualizations saved in: {output_dir}/")
print(f"[OK] Summary report saved as: eda_summary_report.txt")
print(f"\nTotal graphs generated: 17 PNG files")