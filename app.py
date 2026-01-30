import streamlit as st
import pandas as pd

# =========================
# Page config / branding
# =========================
st.set_page_config(
    page_title="RunwayRadar – Cash Flow Predictor",
    page_icon="💸",
    layout="wide",
)

st.markdown("### 💸 RunwayRadar")
st.markdown(
    "Forecast your cash balance, spot risk days, and understand which clients & expenses are stressing your cash flow."
)

st.markdown("---")

# =========================
# Sidebar – controls
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("**1. Upload your data**")
    transactions_file = st.file_uploader(
        "Transactions CSV",
        type=["csv"],
        help="Required columns: date, type, amount, category, client_or_vendor, notes",
    )

    st.markdown("**2. (Optional) Upload invoices**")
    invoices_file = st.file_uploader(
        "Invoices CSV (optional)",
        type=["csv"],
        help="Required columns: invoice_id, client, issue_date, due_date, paid_date, amount",
    )

    st.markdown("---")
    st.markdown("**3. Configure forecast**")

    starting_balance = st.number_input(
        "Starting cash balance",
        value=5000.0,
        step=100.0,
    )

    threshold = st.number_input(
        "Risk threshold (alert if balance below this)",
        value=1000.0,
        step=100.0,
    )

    st.markdown("---")
    st.caption("💡 Tip: Start with a small sample CSV and tune from there.")


# =========================
# Main logic
# =========================
if transactions_file is None:
    st.info("⬅️ Upload **transactions.csv** in the sidebar to see your forecast.")
    st.stop()

# ---- Load & clean transactions data ----
df = pd.read_csv(transactions_file)

required_cols = {"date", "type", "amount"}
if not required_cols.issubset(df.columns):
    st.error(f"❌ transactions.csv must contain at least these columns: {required_cols}")
    st.stop()

df["date"] = pd.to_datetime(df["date"])

df["signed_amount"] = df.apply(
    lambda row: -row["amount"] if str(row["type"]).lower() == "expense" else row["amount"],
    axis=1,
)

# ---- Build 90-day forecast ----
start_date = df["date"].min()
end_date = start_date + pd.Timedelta(days=89)
date_range = pd.date_range(start=start_date, end=end_date, freq="D")

forecast_df = pd.DataFrame({"date": date_range})
daily_net = df.groupby("date")["signed_amount"].sum().reindex(date_range, fill_value=0)
forecast_df["daily_net"] = daily_net.values
forecast_df["balance"] = starting_balance + forecast_df["daily_net"].cumsum()

# ---- Risk days ----
risk_days = forecast_df[forecast_df["balance"] < threshold]

# ---- Big expense days ----
expense_days = forecast_df[forecast_df["daily_net"] < 0].copy()
expense_days["abs_expense"] = expense_days["daily_net"].abs()
top_expense_days = (
    expense_days.sort_values("abs_expense", ascending=False)
    .head(10)
    .copy()
)

# ---- Recurring expenses detection ----
expenses_only = df[df["type"].str.lower() == "expense"].copy()
recurring_records = []

for vendor, group in expenses_only.groupby("client_or_vendor"):
    if pd.isna(vendor):
        continue
    if len(group) < 3:
        continue

    group = group.sort_values("date")
    dates = group["date"].tolist()
    if len(dates) < 3:
        continue

    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    if not gaps:
        continue

    avg_gap = sum(gaps) / len(gaps)

    # Only consider "reasonable" recurring windows (weekly-ish to monthly-ish)
    if avg_gap < 5 or avg_gap > 45:
        continue

    avg_amount = group["amount"].mean()
    last_date = dates[-1]
    next_date = last_date + pd.Timedelta(days=round(avg_gap))

    # Frequency label
    if 5 <= avg_gap <= 9:
        freq_label = "Weekly-ish"
    elif 10 <= avg_gap <= 20:
        freq_label = "Bi-weekly-ish"
    elif 25 <= avg_gap <= 35:
        freq_label = "Monthly-ish"
    else:
        freq_label = "Recurring"

    recurring_records.append(
        {
            "vendor": vendor,
            "avg_gap_days": round(avg_gap, 1),
            "frequency": freq_label,
            "avg_amount": round(float(avg_amount), 2),
            "last_payment_date": last_date.date(),
            "next_expected_date": next_date.date(),
        }
    )

if recurring_records:
    recurring_df = pd.DataFrame(recurring_records).sort_values(
        ["next_expected_date", "avg_amount"], ascending=[True, False]
    )
else:
    recurring_df = pd.DataFrame(
        columns=[
            "vendor",
            "avg_gap_days",
            "frequency",
            "avg_amount",
            "last_payment_date",
            "next_expected_date",
        ]
    )

# =========================
# Layout: Tabs
# =========================
overview_tab, forecast_tab, expenses_tab, recurring_tab, late_tab = st.tabs(
    ["🔍 Overview", "📈 Forecast", "💥 Big Expense Days", "🧾 Recurring Expenses", "🐢 Late-Paying Clients"]
)

# ---- OVERVIEW TAB ----
with overview_tab:
    st.subheader("🔍 Overview")

    lowest_row = forecast_df.sort_values("balance").iloc[0]
    lowest_date = lowest_row["date"].date()
    lowest_balance = float(lowest_row["balance"])
    num_risk_days = len(risk_days)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lowest projected balance", f"${lowest_balance:,.0f}", help="Across the next 90 days")
    with col2:
        st.metric("Risk days (balance < threshold)", num_risk_days)
    with col3:
        st.metric("Starting balance", f"${starting_balance:,.0f}")

    st.markdown("---")

    if num_risk_days == 0:
        st.success("✅ You stay above your risk threshold for all 90 days. Strong runway. 🔥")
    else:
        first_risk = risk_days.iloc[0]
        risk_date = first_risk["date"].date()
        risk_balance = float(first_risk["balance"])

        st.warning(
            f"⚠️ Your balance first drops below ${threshold:,.0f} on {risk_date} "
            f"with a projected balance of ${risk_balance:,.0f}."
        )

    st.markdown("#### Recent transaction snapshot")
    st.dataframe(
        df.sort_values("date", ascending=False).head(10),
        use_container_width=True,
    )


# ---- FORECAST TAB ----
with forecast_tab:
    st.subheader("📈 90-Day Cash Flow Forecast")

    st.markdown("**Forecast table (first 25 days)**")
    st.dataframe(forecast_df.head(25), use_container_width=True)

    st.markdown("**Balance over time**")
    st.line_chart(forecast_df.set_index("date")["balance"])

    st.markdown("**Daily net cash (income − expenses)**")
    st.bar_chart(forecast_df.set_index("date")["daily_net"])


# ---- BIG EXPENSE DAYS TAB ----
with expenses_tab:
    st.subheader("💥 Biggest Net Expense Days")

    if top_expense_days.empty:
        st.info("No large net expense days found in the forecast horizon.")
    else:
        show_df = top_expense_days[["date", "daily_net", "balance"]].copy()
        show_df.rename(columns={"daily_net": "net_expense"}, inplace=True)

        st.markdown(
            "These are the days where cash drops the most after income and expenses are combined."
        )
        st.dataframe(show_df, use_container_width=True)


# ---- RECURRING EXPENSES TAB ----
with recurring_tab:
    st.subheader("🧾 Recurring Expenses")

    if recurring_df.empty:
        st.info("No recurring expense patterns detected yet. Add more months of data to see patterns like salary, rent, EMIs, or subscriptions.")
    else:
        st.markdown(
            "Based on your past expenses, these vendors look like **recurring payments** "
            "(salary, rent, SaaS, EMIs, etc.)."
        )
        st.dataframe(recurring_df, use_container_width=True)


# ---- LATE-PAYING CLIENTS TAB ----
with late_tab:
    st.subheader("🐢 Late-Paying Clients")

    if invoices_file is None:
        st.info("Upload **invoices.csv** in the sidebar to analyze client payment delays.")
    else:
        inv = pd.read_csv(invoices_file)

        required_invoice_cols = {"client", "issue_date", "due_date", "paid_date", "amount"}
        if not required_invoice_cols.issubset(inv.columns):
            st.error(
                f"❌ invoices.csv must contain at least these columns: {required_invoice_cols}"
            )
        else:
            inv["issue_date"] = pd.to_datetime(inv["issue_date"])
            inv["due_date"] = pd.to_datetime(inv["due_date"])
            inv["paid_date"] = pd.to_datetime(inv["paid_date"], errors="coerce")

            paid = inv.dropna(subset=["paid_date"]).copy()
            if paid.empty:
                st.info("No paid invoices with `paid_date` found, so delays can’t be computed yet.")
            else:
                paid["delay_days"] = (paid["paid_date"] - paid["due_date"]).dt.days

                late_stats = (
                    paid.groupby("client")["delay_days"]
                    .mean()
                    .reset_index()
                    .sort_values("delay_days", ascending=False)
                )
                late_stats = late_stats[late_stats["delay_days"] > 0]

                if late_stats.empty:
                    st.success("✅ No clients consistently paying late based on your invoice data.")
                else:
                    late_stats["delay_days"] = late_stats["delay_days"].round(1)
                    st.markdown(
                        "Clients ordered by how many days after due date they usually pay:"
                    )
                    st.dataframe(late_stats, use_container_width=True)
