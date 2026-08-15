"""
credit_risk_analysis.py
=======================
End-to-end credit risk scoring pipeline for Nigerian digital lenders.

Run after generate_data.py:
    python credit_risk_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             confusion_matrix, precision_recall_curve, average_precision_score)

np.random.seed(2024)

# ============================================================
# 1. LOAD & CLEAN
# ============================================================
df = pd.read_csv('data/nigeria_loan_applicants.csv')

df['AvgMonthlyInflow'].fillna(df['AvgMonthlyInflow'].median(), inplace=True)
df['AvgMonthlyOutflow'].fillna(df['AvgMonthlyOutflow'].median(), inplace=True)
df['MobileWalletBalance'].fillna(df['MobileWalletBalance'].median(), inplace=True)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
df['DebtToIncome'] = (df['LoanAmount'] / np.maximum(df['AvgMonthlyInflow'], 1)).round(2)
df['DebtToIncome'] = df['DebtToIncome'].clip(0, 20)

df['NetCashFlow'] = (df['AvgMonthlyInflow'] - df['AvgMonthlyOutflow']).round(0)
df['PreviousDefaultRate'] = (df['PreviousDefaults'] / np.maximum(df['PreviousLoans'], 1)).round(2)

df['DigitalEngagementScore'] = (
    (df['AirtimeMonthly'] / 5000) * 0.3 +
    (df['DataBundleMonthly'] / 3000) * 0.2 +
    (df['AppUsageHours'] / 5) * 0.3 +
    (df['MobileWalletBalance'] / 50000) * 0.2
).round(2)

df['FinancialDepthScore'] = (
    df['BVNVerified'] * 0.3 +
    (df['LinkedBankAccounts'] / 5) * 0.3 +
    (df['SalaryCreditFrequency'] / 4) * 0.4
).round(2)

df['LoanTier'] = pd.cut(df['LoanAmount'], bins=[0, 20000, 100000, 500000],
                         labels=['Small (<20k)', 'Medium (20-100k)', 'Large (>100k)'])

# ============================================================
# 3. MODELING
# ============================================================
cat_cols = ['Gender', 'State', 'EmploymentType', 'EducationLevel', 'DevicePriceTier', 'ReferralSource']
df_enc = df.copy()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col + '_enc'] = le.fit_transform(df_enc[col])
    le_dict[col] = le

feature_cols = ['Age', 'LoanAmount', 'LoanTenureDays', 'InterestRate', 'PreviousLoans',
                'PreviousDefaults', 'DaysSinceLastLoan', 'AirtimeMonthly', 'DataBundleMonthly',
                'MobileWalletBalance', 'AppUsageHours', 'SalaryCreditFrequency',
                'AvgMonthlyInflow', 'AvgMonthlyOutflow', 'BVNVerified', 'LinkedBankAccounts',
                'ApplicationHour', 'DebtToIncome', 'NetCashFlow', 'PreviousDefaultRate',
                'DigitalEngagementScore', 'FinancialDepthScore']
for col in cat_cols:
    feature_cols.append(col + '_enc')

X = df_enc[feature_cols]
y = df_enc['Defaulted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
}

results = {}

for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    results[name] = {'model': model, 'y_pred': y_pred, 'y_proba': y_proba, 'auc': auc, 'ap': ap}
    print(f"\n{name}: AUC={auc:.4f}, AP={ap:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Repaid', 'Defaulted']))

# Cross-validation
print("\n5-Fold CV (AUC-ROC):")
for name, model in models.items():
    if name == 'Logistic Regression':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc')
    print(f"  {name}: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ============================================================
# 4. RISK SCORING & DECISIONS
# ============================================================
gb_model = results['Gradient Boosting']['model']
df['DefaultProbability'] = gb_model.predict_proba(X)[:, 1]

df['RiskGrade'] = pd.cut(df['DefaultProbability'],
                          bins=[0, 0.05, 0.15, 0.30, 0.50, 1.0],
                          labels=['A (Prime)', 'B (Near-Prime)', 'C (Subprime)', 'D (High Risk)', 'E (Deep Subprime)'])

df['ExpectedLoss'] = (df['DefaultProbability'] * df['LoanAmount']).round(0)

def recommend_rate(row):
    base = row['InterestRate']
    if row['RiskGrade'] == 'A (Prime)': return max(base - 5, 10)
    elif row['RiskGrade'] == 'B (Near-Prime)': return base
    elif row['RiskGrade'] == 'C (Subprime)': return base + 5
    elif row['RiskGrade'] == 'D (High Risk)': return base + 10
    else: return base + 15

df['RecommendedRate'] = df.apply(recommend_rate, axis=1).round(1)

def loan_decision(row):
    if row['RiskGrade'] in ['A (Prime)', 'B (Near-Prime)']: return 'Approve'
    elif row['RiskGrade'] == 'C (Subprime)' and row['BVNVerified'] == 1: return 'Approve with Conditions'
    elif row['RiskGrade'] == 'C (Subprime)': return 'Manual Review'
    elif row['RiskGrade'] == 'D (High Risk)' and row['PreviousDefaults'] == 0 and row['BVNVerified'] == 1: return 'Manual Review'
    else: return 'Decline'

df['LoanDecision'] = df.apply(loan_decision, axis=1)

df.to_csv('data/nigeria_loan_scored.csv', index=False)
print("\nScored dataset saved to: data/nigeria_loan_scored.csv")

# ============================================================
# 5. EXPORT CHARTS
# ============================================================
os.makedirs('outputs', exist_ok=True)

# EDA Chart
fig, axes = plt.subplots(3, 3, figsize=(17, 14))
fig.suptitle('Nigerian Digital Lending: Credit Risk Drivers', fontsize=16, fontweight='bold', y=1.02)

ax = axes[0, 0]
emp_default = df.groupby('EmploymentType')['Defaulted'].mean() * 100
emp_default.sort_values(ascending=False).plot(kind='bar', ax=ax, color='#e74c3c', alpha=0.85)
ax.set_title('Default Rate by Employment Type', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
ax.tick_params(axis='x', rotation=30)
ax.grid(axis='y', alpha=0.3)

ax = axes[0, 1]
state_default = df.groupby('State')['Defaulted'].mean() * 100
state_default.sort_values(ascending=False).plot(kind='bar', ax=ax, color='#3498db', alpha=0.85)
ax.set_title('Default Rate by State', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

ax = axes[0, 2]
bvn_default = df.groupby('BVNVerified')['Defaulted'].mean() * 100
bvn_default.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.85)
ax.set_title('Default Rate: BVN Verified vs Not', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
ax.set_xticklabels(['Not Verified', 'Verified'], rotation=0)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
tier_default = df.groupby('LoanTier')['Defaulted'].mean() * 100
tier_default.plot(kind='bar', ax=ax, color='#f39c12', alpha=0.85)
ax.set_title('Default Rate by Loan Tier', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 1]
prev_default = df.groupby('PreviousDefaults')['Defaulted'].mean() * 100
prev_default.plot(kind='bar', ax=ax, color='#9b59b6', alpha=0.85)
ax.set_title('Default Rate by Previous Defaults', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
ax.set_xlabel('Number of Previous Defaults')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 2]
dti_bins = pd.cut(df['DebtToIncome'], bins=[0, 1, 2, 3, 5, 20], labels=['<1x', '1-2x', '2-3x', '3-5x', '>5x'])
dti_default = df.groupby(dti_bins)['Defaulted'].mean() * 100
dti_default.plot(kind='bar', ax=ax, color='#e67e22', alpha=0.85)
ax.set_title('Default Rate by Debt-to-Income', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)

ax = axes[2, 0]
device_default = df.groupby('DevicePriceTier')['Defaulted'].mean() * 100
device_default.plot(kind='bar', ax=ax, color='#1abc9c', alpha=0.85)
ax.set_title('Default Rate by Device Tier', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
ax.tick_params(axis='x', rotation=15)
ax.grid(axis='y', alpha=0.3)

ax = axes[2, 1]
eng_bins = pd.cut(df['DigitalEngagementScore'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
eng_default = df.groupby(eng_bins)['Defaulted'].mean() * 100
eng_default.plot(kind='bar', ax=ax, color='#c0392b', alpha=0.85)
ax.set_title('Default Rate by Digital Engagement', fontweight='bold')
ax.set_ylabel('Default Rate (%)')
ax.tick_params(axis='x', rotation=15)
ax.grid(axis='y', alpha=0.3)

ax = axes[2, 2]
numeric_cols = ['Age', 'LoanAmount', 'LoanTenureDays', 'InterestRate', 'PreviousLoans',
                'PreviousDefaults', 'AirtimeMonthly', 'MobileWalletBalance', 'AppUsageHours',
                'SalaryCreditFrequency', 'AvgMonthlyInflow', 'BVNVerified', 'LinkedBankAccounts',
                'DebtToIncome', 'DigitalEngagementScore', 'FinancialDepthScore', 'Defaulted']
corr = df[numeric_cols].corr()
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels([c.replace('AvgMonthlyInflow', 'Income') for c in numeric_cols], rotation=45, ha='right', fontsize=7)
ax.set_yticklabels([c.replace('AvgMonthlyInflow', 'Income') for c in numeric_cols], fontsize=7)
ax.set_title('Feature Correlation Matrix', fontweight='bold')
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f'{corr.iloc[i,j]:.1f}', ha='center', va='center', fontsize=6,
                color='white' if abs(corr.iloc[i,j]) > 0.5 else 'black')

plt.tight_layout()
plt.savefig('outputs/nigeria_01_eda_risk_drivers.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: outputs/nigeria_01_eda_risk_drivers.png")

# Model Performance Chart
fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle('Credit Risk Model: Performance & Portfolio Impact', fontsize=16, fontweight='bold', y=1.02)

ax = axes[0, 0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)

ax = axes[0, 1]
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['y_proba'])
    ax.plot(recall, precision, label=f"{name} (AP={res['ap']:.3f})", linewidth=2)
ax.axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.5, label=f'Baseline ({y_test.mean():.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves', fontweight='bold')
ax.legend(loc='lower left', fontsize=9)
ax.grid(alpha=0.3)

ax = axes[0, 2]
cm = confusion_matrix(y_test, results['Gradient Boosting']['y_pred'])
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Repaid', 'Defaulted'])
ax.set_yticklabels(['Repaid', 'Defaulted'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (Gradient Boosting)', fontweight='bold')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14, fontweight='bold',
                color='white' if cm[i,j] > cm.max()/2 else 'black')

ax = axes[1, 0]
gb_model = results['Gradient Boosting']['model']
importance = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=True)
colors = ['#e74c3c' if 'Default' in i or 'DebtToIncome' in i or 'BVN' in i or 'Financial' in i or 'Digital' in i
          else '#3498db' for i in importance.index]
importance.tail(15).plot(kind='barh', ax=ax, color=colors[-15:])
ax.set_title('Top 15 Feature Importance (Gradient Boosting)', fontweight='bold')
ax.set_xlabel('Importance')
ax.grid(axis='x', alpha=0.3)

ax = axes[1, 1]
risk_counts = df['RiskGrade'].value_counts().reindex(['A (Prime)', 'B (Near-Prime)', 'C (Subprime)', 'D (High Risk)', 'E (Deep Subprime)'], fill_value=0)
colors_risk = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
bars = ax.bar(risk_counts.index, risk_counts.values, color=colors_risk, edgecolor='white', linewidth=0.5)
ax.set_ylabel('Number of Applicants')
ax.set_title('Portfolio Risk Grade Distribution', fontweight='bold')
for bar, val in zip(bars, risk_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(risk_counts.values)*0.02,
            f"{val:,}", ha='center', va='bottom', fontweight='bold', fontsize=9)
ax.tick_params(axis='x', rotation=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 2]
decision_counts = df['LoanDecision'].value_counts()
decision_colors = {'Approve': '#2ecc71', 'Approve with Conditions': '#f39c12', 'Manual Review': '#3498db', 'Decline': '#e74c3c'}
colors_dec = [decision_colors.get(d, '#95a5a6') for d in decision_counts.index]
wedges, texts, autotexts = ax.pie(decision_counts, labels=decision_counts.index, autopct='%1.1f%%',
                                   colors=colors_dec, startangle=90)
ax.set_title('Automated Loan Decision Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/nigeria_02_model_performance.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: outputs/nigeria_02_model_performance.png")

print("\n=== ALL OUTPUTS GENERATED ===")
