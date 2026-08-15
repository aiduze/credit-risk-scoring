import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Nigeria Credit Risk Dashboard",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    # Try scored data first (preferred)
    if os.path.exists('data/nigeria_loan_scored.csv'):
        df = pd.read_csv('data/nigeria_loan_scored.csv')
        # Ensure RiskGrade is categorical
        df['RiskGrade'] = pd.Categorical(df['RiskGrade'], categories=['A (Prime)', 'B (Near-Prime)', 'C (Subprime)', 'D (High Risk)', 'E (Deep Subprime)'])
        df['LoanTier'] = pd.Categorical(df['LoanTier'], categories=['Small (<20k)', 'Medium (20-100k)', 'Large (>100k)'])
        return df

    # Try raw data next
    if os.path.exists('data/nigeria_loan_applicants.csv'):
        df = pd.read_csv('data/nigeria_loan_applicants.csv')
    else:
        # Fallback: generate clean synthetic data
        np.random.seed(2024)
        n = 8000
        states = ['Lagos', 'Kano', 'Rivers', 'Kaduna', 'Oyo', 'FCT Abuja', 'Delta', 'Ogun', 'Anambra', 'Enugu']
        state_weights = [0.22, 0.10, 0.08, 0.07, 0.07, 0.08, 0.06, 0.07, 0.13, 0.12]

        prev_loans = np.random.poisson(1.5, n).clip(0, 10)
        prev_defaults = np.array([np.random.binomial(int(pl), 0.15) for pl in prev_loans])

        df = pd.DataFrame({
            'ApplicantID': [f'APP_{str(i).zfill(6)}' for i in range(1, n+1)],
            'Defaulted': np.random.binomial(1, 0.22, n),
            'Age': np.random.gamma(5, 5, n).clip(18, 65).astype(int),
            'Gender': np.random.choice(['Male', 'Female'], n, p=[0.58, 0.42]),
            'State': np.random.choice(states, n, p=state_weights),
            'EmploymentType': np.random.choice(['Salaried', 'Self-employed', 'Unemployed', 'Student', 'Trader'], n, p=[0.30, 0.25, 0.15, 0.10, 0.20]),
            'EducationLevel': np.random.choice(['No formal', 'Primary', 'Secondary', 'Tertiary', 'Postgraduate'], n, p=[0.08, 0.12, 0.30, 0.40, 0.10]),
            'LoanAmount': np.random.choice([5000, 10000, 20000, 50000, 100000, 200000, 500000], n, p=[0.05, 0.10, 0.20, 0.25, 0.20, 0.15, 0.05]),
            'LoanTenureDays': np.random.choice([7, 14, 30, 60, 90], n, p=[0.05, 0.10, 0.50, 0.25, 0.10]),
            'InterestRate': np.random.uniform(10, 40, n).round(1),
            'PreviousLoans': prev_loans,
            'PreviousDefaults': prev_defaults,
            'DaysSinceLastLoan': np.where(prev_loans == 0, 999, np.random.exponential(60, n).clip(0, 730).astype(int)),
            'AirtimeMonthly': np.random.exponential(2000, n).clip(100, 50000).round(0),
            'DataBundleMonthly': np.random.exponential(1500, n).clip(0, 20000).round(0),
            'MobileWalletBalance': np.random.exponential(5000, n).clip(0, 500000).round(0),
            'DevicePriceTier': np.random.choice(['Budget (<50k)', 'Mid-range (50-150k)', 'Premium (>150k)'], n, p=[0.50, 0.35, 0.15]),
            'AppUsageHours': np.random.gamma(2, 1, n).clip(0.1, 8).round(1),
            'SalaryCreditFrequency': np.random.choice([0, 1, 2, 3, 4], n, p=[0.35, 0.25, 0.20, 0.15, 0.05]),
            'AvgMonthlyInflow': np.random.exponential(80000, n).clip(0, 2000000).round(0),
            'AvgMonthlyOutflow': np.random.exponential(65000, n).clip(0, 1800000).round(0),
            'BVNVerified': np.random.choice([0, 1], n, p=[0.12, 0.88]),
            'LinkedBankAccounts': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05]),
            'ApplicationHour': np.random.choice(range(24), n),
            'ReferralSource': np.random.choice(['Organic', 'Referral', 'Facebook Ads', 'Google Ads', 'Agent'], n, p=[0.30, 0.20, 0.20, 0.15, 0.15]),
        })

        # Introduce and fill missing values
        for col in ['AvgMonthlyInflow', 'AvgMonthlyOutflow', 'MobileWalletBalance']:
            missing_idx = np.random.choice(df.index, size=int(0.03*len(df)), replace=False)
            df.loc[missing_idx, col] = np.nan
        df = df.fillna({'AvgMonthlyInflow': df['AvgMonthlyInflow'].median(),
                        'AvgMonthlyOutflow': df['AvgMonthlyOutflow'].median(),
                        'MobileWalletBalance': df['MobileWalletBalance'].median()})

    # Feature engineering
    df['DebtToIncome'] = (df['LoanAmount'] / df['AvgMonthlyInflow'].clip(lower=1)).round(2).clip(0, 20)
    df['NetCashFlow'] = (df['AvgMonthlyInflow'] - df['AvgMonthlyOutflow']).round(0)
    df['PreviousDefaultRate'] = (df['PreviousDefaults'] / df['PreviousLoans'].clip(lower=1)).round(2)
    df['DigitalEngagementScore'] = ((df['AirtimeMonthly'] / 5000) * 0.3 + (df['DataBundleMonthly'] / 3000) * 0.2 + (df['AppUsageHours'] / 5) * 0.3 + (df['MobileWalletBalance'] / 50000) * 0.2).round(2)
    df['FinancialDepthScore'] = (df['BVNVerified'] * 0.3 + (df['LinkedBankAccounts'] / 5) * 0.3 + (df['SalaryCreditFrequency'] / 4) * 0.4).round(2)
    df['LoanTier'] = pd.cut(df['LoanAmount'], bins=[0, 20000, 100000, 500000], labels=['Small (<20k)', 'Medium (20-100k)', 'Large (>100k)'])

    # Encode categoricals
    cat_cols = ['Gender', 'State', 'EmploymentType', 'EducationLevel', 'DevicePriceTier', 'ReferralSource']
    df_enc = df.copy()
    for col in cat_cols:
        df_enc[col + '_enc'] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    feature_cols = ['Age', 'LoanAmount', 'LoanTenureDays', 'InterestRate', 'PreviousLoans',
                    'PreviousDefaults', 'DaysSinceLastLoan', 'AirtimeMonthly', 'DataBundleMonthly',
                    'MobileWalletBalance', 'AppUsageHours', 'SalaryCreditFrequency',
                    'AvgMonthlyInflow', 'AvgMonthlyOutflow', 'BVNVerified', 'LinkedBankAccounts',
                    'ApplicationHour', 'DebtToIncome', 'NetCashFlow', 'PreviousDefaultRate',
                    'DigitalEngagementScore', 'FinancialDepthScore']
    for col in cat_cols:
        feature_cols.append(col + '_enc')

    X = df_enc[feature_cols].copy()
    y = df_enc['Defaulted'].copy()

    # CRITICAL: Ensure no NaN or inf before training
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    df['DefaultProbability'] = gb.predict_proba(X)[:, 1]

    df['RiskGrade'] = pd.cut(df['DefaultProbability'], bins=[0, 0.05, 0.15, 0.30, 0.50, 1.0],
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
    return df


df = load_data()

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.title("Filters")

risk_filter = st.sidebar.multiselect(
    "Risk grade",
    options=df['RiskGrade'].cat.categories.tolist(),
    default=df['RiskGrade'].cat.categories.tolist()
)

decision_filter = st.sidebar.multiselect(
    "Loan decision",
    options=sorted(df['LoanDecision'].unique()),
    default=sorted(df['LoanDecision'].unique())
)

state_filter = st.sidebar.multiselect(
    "State",
    options=sorted(df['State'].unique()),
    default=sorted(df['State'].unique())
)

employment_filter = st.sidebar.multiselect(
    "Employment type",
    options=sorted(df['EmploymentType'].unique()),
    default=sorted(df['EmploymentType'].unique())
)

loan_tier_filter = st.sidebar.multiselect(
    "Loan tier",
    options=df['LoanTier'].cat.categories.tolist(),
    default=df['LoanTier'].cat.categories.tolist()
)

bvn_filter = st.sidebar.multiselect(
    "BVN verified",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: 'Not Verified' if x == 0 else 'Verified'
)

filtered = df[
    (df['RiskGrade'].isin(risk_filter)) &
    (df['LoanDecision'].isin(decision_filter)) &
    (df['State'].isin(state_filter)) &
    (df['EmploymentType'].isin(employment_filter)) &
    (df['LoanTier'].isin(loan_tier_filter)) &
    (df['BVNVerified'].isin(bvn_filter))
]

# ============================================================
# HEADER
# ============================================================
st.title("Nigerian Digital Lending: Credit Risk Dashboard")
st.markdown("Score loan applicants, grade risk, and automate approval decisions.")

# ============================================================
# KPI ROW
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

total_loan_value = filtered['LoanAmount'].sum()
total_expected_loss = filtered['ExpectedLoss'].sum()

col1.metric("Total Applicants", f"{len(filtered):,}")
col2.metric("Total Loan Value", f"N{total_loan_value:,.0f}")
col3.metric("Expected Loss", f"N{total_expected_loss:,.0f}")
col4.metric("Portfolio Default Risk", f"{filtered['DefaultProbability'].mean()*100:.1f}%")
col5.metric("Avg Recommended Rate", f"{filtered['RecommendedRate'].mean():.1f}%")

st.divider()

# ============================================================
# CHARTS ROW 1
# ============================================================
left, right = st.columns(2)

with left:
    st.subheader("Risk Grade Distribution")
    risk_counts = filtered['RiskGrade'].value_counts().reindex(['A (Prime)', 'B (Near-Prime)', 'C (Subprime)', 'D (High Risk)', 'E (Deep Subprime)'], fill_value=0)
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("Applicants")
    ax.set_ylim(0, max(risk_counts.values) * 1.15)
    for bar, val in zip(bars, risk_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(risk_counts.values)*0.02,
                f"{val:,}", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.tick_params(axis='x', rotation=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

with right:
    st.subheader("Default Rate by Driver")
    driver = st.selectbox("Select driver", ['EmploymentType', 'State', 'LoanTier', 'DevicePriceTier', 'BVNVerified'], key="driver1")
    fig, ax = plt.subplots(figsize=(6, 4))
    if driver == 'BVNVerified':
        default_by = filtered.groupby(driver)['Defaulted'].mean() * 100
        default_by.index = ['Not Verified', 'Verified']
    else:
        default_by = filtered.groupby(driver)['Defaulted'].mean() * 100
        default_by = default_by.sort_values(ascending=False)
    bars = ax.bar(default_by.index.astype(str), default_by.values, color='#e74c3c', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("Default Rate (%)")
    ax.set_ylim(0, max(default_by.values) * 1.2 if len(default_by) > 0 else 10)
    ax.tick_params(axis='x', rotation=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

# ============================================================
# CHARTS ROW 2
# ============================================================
left2, right2 = st.columns(2)

with left2:
    st.subheader("Loan Decision Distribution")
    decision_counts = filtered['LoanDecision'].value_counts()
    decision_colors = {'Approve': '#2ecc71', 'Approve with Conditions': '#f39c12', 'Manual Review': '#3498db', 'Decline': '#e74c3c'}
    colors_dec = [decision_colors.get(d, '#95a5a6') for d in decision_counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(decision_counts, labels=decision_counts.index, autopct='%1.1f%%',
                                       colors=colors_dec, startangle=90)
    ax.set_title(f"{len(filtered):,} applicants")
    st.pyplot(fig)

with right2:
    st.subheader("Expected Loss by Risk Grade")
    fig, ax = plt.subplots(figsize=(6, 4))
    loss_by_grade = filtered.groupby('RiskGrade')['ExpectedLoss'].sum() / 1e6
    colors_loss = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
    bars = ax.bar(loss_by_grade.index, loss_by_grade.values, color=colors_loss[:len(loss_by_grade)], edgecolor='white', linewidth=0.5)
    ax.set_ylabel("Expected Loss (N Millions)")
    ax.tick_params(axis='x', rotation=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

st.divider()

# ============================================================
# APPLICANT TABLE
# ============================================================
st.subheader("Applicant Risk Register")

table_cols = ['ApplicantID', 'RiskGrade', 'DefaultProbability', 'LoanAmount', 'LoanTenureDays',
              'InterestRate', 'RecommendedRate', 'LoanDecision', 'EmploymentType', 'State',
              'BVNVerified', 'DebtToIncome', 'PreviousDefaultRate']

display_df = filtered[table_cols].copy()
display_df['DefaultProbability'] = (display_df['DefaultProbability'] * 100).round(1).astype(str) + '%'
display_df['BVNVerified'] = display_df['BVNVerified'].map({0: 'No', 1: 'Yes'})
display_df = display_df.sort_values('LoanAmount', ascending=False)

st.dataframe(display_df, use_container_width=True, hide_index=True)

csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download filtered applicant list",
    data=csv,
    file_name=f"nigeria_loan_applicants_{len(filtered)}.csv",
    mime="text/csv"
)

st.divider()

# ============================================================
# DECISION RULES
# ============================================================
st.subheader("Automated Decision Rules")

rules = {
    'A (Prime) + B (Near-Prime)': 'Auto-approve at standard or reduced rate',
    'C (Subprime) + BVN Verified': 'Approve with conditions (higher rate, shorter tenure)',
    'C (Subprime) + No BVN': 'Route to manual underwriting review',
    'D (High Risk) + No previous defaults + BVN': 'Manual review with additional documentation',
    'D (High Risk) + Previous defaults': 'Decline or offer micro-loan only',
    'E (Deep Subprime)': 'Auto-decline with referral to financial literacy program'
}

for rule, action in rules.items():
    with st.expander(f"{rule}"):
        st.markdown(f"- **Action:** {action}")

st.caption("Built with Streamlit — Data is synthetic for demonstration purposes")
