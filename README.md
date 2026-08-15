# Nigerian Digital Lending: Credit Risk Scoring

&gt; **End-to-end credit risk model** for Nigerian fintech/loan app institutions using alternative data (mobile behavior, BVN verification, transaction patterns).

---

## Problem Statement

Nigerian digital lenders face **~22% default rates** and lack traditional credit bureau coverage for most borrowers. This project builds an **alternative-data credit risk model** that scores every loan applicant using:

- Mobile wallet & airtime behavior
- BVN verification status
- Bank transaction patterns
- Device and app engagement signals
- Employment and demographic data

---

## Dataset

**Source:** Synthetic dataset engineered for Nigerian digital lending context.

| Property | Value |
|----------|-------|
| Applicants | 8,000 |
| Features | 24 (demographic, loan, alternative data, banking, behavioral) |
| Default Rate | ~22% |
| Missing Values | 3% (realistic) |

**Key features:**
- `BVNVerified` — Bank Verification Number (critical for Nigerian KYC)
- `PreviousDefaults` / `PreviousLoans` — credit history
- `DebtToIncome` — loan amount vs monthly income
- `DigitalEngagementScore` — composite from airtime, data, app usage, wallet balance
- `FinancialDepthScore` — composite from BVN, linked accounts, salary credits
- `LoanAmount`, `LoanTenureDays`, `InterestRate`

---

## Reproducibility

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate dataset
python generate_data.py

# 3. Run analysis
python credit_risk_analysis.py

# 4. Launch dashboard
streamlit run app.py
