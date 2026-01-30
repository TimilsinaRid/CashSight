#  CashSight

CashSight is a Streamlit-based cash flow forecasting MVP for small businesses, agencies, and consultants. It helps users predict future cash balances, spot low-cash risk days, detect recurring expenses (rent, payroll, SaaS, loan payments), and identify late-paying clients from invoice history.

## Features
- 90-day cash balance forecast from uploaded transactions
- Risk-day detection using a user-defined threshold
- Biggest cash-drop days (largest net expense days)
- Recurring expense detection from patterns in spending
- Late-paying client analysis (optional invoices upload)

## Input Files

### transactions.csv (required)
Columns:
- `date`
- `type` (income / expense)
- `amount`
- `category` (optional but recommended)
- `client_or_vendor` (optional but recommended)
- `notes` (optional)

### invoices.csv (optional)
Columns:
- `invoice_id`
- `client`
- `issue_date`
- `due_date`
- `paid_date` (blank if unpaid)
- `amount`

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
