"""
generate_data.py
================
Generates synthetic Nigerian digital lending dataset.
Run first: python generate_data.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(2024)

N_APPLICANTS = 8000

applicant_id = [f'APP_{str(i).zfill(6)}' for i in range(1, N_APPLICANTS + 1)]

states = ['Lagos', 'Kano', 'Rivers', 'Kaduna', 'Oyo', 'FCT Abuja', 'Delta', 'Ogun', 'Anambra', 'Enugu']
state_weights = [0.22, 0.10, 0.08, 0.07, 0.07, 0.08, 0.06, 0.07, 0.13, 0.12]
applicant_state = np.random.choice(states, N_APPLICANTS, p=state_weights)

age = np.random.gamma(shape=5, scale=5, size=N_APPLICANTS).clip(18, 65).astype(int)
gender = np.random.choice(['Male', 'Female'], N_APPLICANTS, p=[0.58, 0.42])

employment = np.random.choice(
    ['Salaried', 'Self-employed', 'Unemployed', 'Student', 'Trader'],
    N_APPLICANTS, p=[0.30, 0.25, 0.15, 0.10, 0.20]
)

education = np.random.choice(
    ['No formal', 'Primary', 'Secondary', 'Tertiary', 'Postgraduate'],
    N_APPLICANTS, p=[0.08, 0.12, 0.30, 0.40, 0.10]
)

loan_amount = np.random.choice(
    [5000, 10000, 20000, 50000, 100000, 200000, 500000],
    N_APPLICANTS, p=[0.05, 0.10, 0.20, 0.25, 0.20, 0.15, 0.05]
)

loan_tenure_days = np.random.choice([7, 14, 30, 60, 90], N_APPLICANTS, p=[0.05, 0.10, 0.50, 0.25, 0.10])

interest_rate = np.where(loan_tenure_days <= 14, np.random.uniform(10, 20, N_APPLICANTS),
                np.where(loan_tenure_days <= 30, np.random.uniform(15, 30, N_APPLICANTS),
                         np.random.uniform(20, 40, N_APPLICANTS))).round(1)

previous_loans = np.random.poisson(lam=1.5, size=N_APPLICANTS).clip(0, 10)
previous_defaults = np.random.binomial(previous_loans, 0.15)
previous_defaults = np.minimum(previous_defaults, previous_loans)

days_since_last_loan = np.random.exponential(scale=60, size=N_APPLICANTS).clip(0, 730).astype(int)
days_since_last_loan = np.where(previous_loans == 0, 999, days_since_last_loan)

airtime_monthly = np.random.exponential(scale=2000, size=N_APPLICANTS).clip(100, 50000).round(0)
data_bundle_monthly = np.random.exponential(scale=1500, size=N_APPLICANTS).clip(0, 20000).round(0)
mobile_wallet_balance = np.random.exponential(scale=5000, size=N_APPLICANTS).clip(0, 500000).round(0)

device_price_tier = np.random.choice(['Budget (<50k)', 'Mid-range (50-150k)', 'Premium (>150k)'],
                                      N_APPLICANTS, p=[0.50, 0.35, 0.15])
app_usage_hours = np.random.gamma(shape=2, scale=1, size=N_APPLICANTS).clip(0.1, 8).round(1)

salary_credit_frequency = np.random.choice([0, 1, 2, 3, 4], N_APPLICANTS, p=[0.35, 0.25, 0.20, 0.15, 0.05])
avg_monthly_inflow = np.random.exponential(scale=80000, size=N_APPLICANTS).clip(0, 2000000).round(0)
avg_monthly_outflow = np.random.exponential(scale=65000, size=N_APPLICANTS).clip(0, 1800000).round(0)

bvn_verified = np.random.choice([0, 1], N_APPLICANTS, p=[0.12, 0.88])
linked_bank_accounts = np.random.choice([0, 1, 2, 3, 4, 5], N_APPLICANTS, p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05])

hour_probs = np.array([0.015]*6 + [0.04]*6 + [0.055]*6 + [0.04]*6)
hour_probs = hour_probs / hour_probs.sum()
application_hour = np.random.choice(range(24), N_APPLICANTS, p=hour_probs)

referral_source = np.random.choice(
    ['Organic', 'Referral', 'Facebook Ads', 'Google Ads', 'Agent'],
    N_APPLICANTS, p=[0.30, 0.20, 0.20, 0.15, 0.15]
)

# Default generation
default_score = (
    0.20 * (previous_defaults / np.maximum(previous_loans, 1)) +
    0.15 * (1 - bvn_verified) +
    0.12 * (employment == 'Unemployed').astype(float) +
    0.10 * (employment == 'Student').astype(float) +
    0.10 * (avg_monthly_inflow < 20000).astype(float) +
    0.08 * (mobile_wallet_balance < 1000).astype(float) +
    0.08 * (linked_bank_accounts == 0).astype(float) +
    0.07 * (loan_amount / np.maximum(avg_monthly_inflow, 1) > 3).astype(float) +
    0.05 * (airtime_monthly < 500).astype(float) +
    0.05 * (days_since_last_loan < 30).astype(float)
)

state_risk = {'Lagos': -0.05, 'Kano': 0.05, 'Rivers': 0.0, 'Kaduna': 0.03, 'Oyo': -0.02,
              'FCT Abuja': -0.08, 'Delta': 0.02, 'Ogun': -0.03, 'Anambra': -0.01, 'Enugu': 0.01}
for i, s in enumerate(applicant_state):
    default_score[i] += state_risk.get(s, 0)

default_score += np.random.normal(0, 0.08, N_APPLICANTS)
default_score = np.clip(default_score, 0, 1)
defaulted = (default_score > np.percentile(default_score, 78)).astype(int)

df = pd.DataFrame({
    'ApplicantID': applicant_id,
    'Defaulted': defaulted,
    'Age': age,
    'Gender': gender,
    'State': applicant_state,
    'EmploymentType': employment,
    'EducationLevel': education,
    'LoanAmount': loan_amount,
    'LoanTenureDays': loan_tenure_days,
    'InterestRate': interest_rate,
    'PreviousLoans': previous_loans,
    'PreviousDefaults': previous_defaults,
    'DaysSinceLastLoan': days_since_last_loan,
    'AirtimeMonthly': airtime_monthly,
    'DataBundleMonthly': data_bundle_monthly,
    'MobileWalletBalance': mobile_wallet_balance,
    'DevicePriceTier': device_price_tier,
    'AppUsageHours': app_usage_hours,
    'SalaryCreditFrequency': salary_credit_frequency,
    'AvgMonthlyInflow': avg_monthly_inflow,
    'AvgMonthlyOutflow': avg_monthly_outflow,
    'BVNVerified': bvn_verified,
    'LinkedBankAccounts': linked_bank_accounts,
    'ApplicationHour': application_hour,
    'ReferralSource': referral_source
})

for col in ['AvgMonthlyInflow', 'AvgMonthlyOutflow', 'MobileWalletBalance']:
    df.loc[np.random.choice(df.index, size=int(0.03 * len(df)), replace=False), col] = np.nan

os.makedirs('data', exist_ok=True)
df.to_csv('data/nigeria_loan_applicants.csv', index=False)

print(f"Dataset generated: {N_APPLICANTS} applicants x {df.shape[1]} features")
print(f"Default rate: {df['Defaulted'].mean()*100:.1f}%")
print(f"Saved to: data/nigeria_loan_applicants.csv")
