from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

from decision_engine import build_evaluated_scenarios, build_ranked_recommendations
from forecast_engine import (
    build_cashflow_model,
    money,
    read_invoices,
    read_recurring_obligations,
    read_transactions,
)
from scenario_engine import simulate_scenario


APP_DIR = Path(__file__).resolve().parent
SAMPLE_TRANSACTIONS_PATH = APP_DIR / "example_transactions.csv"
SAMPLE_INVOICES_PATH = APP_DIR / "example_Invoices.csv"
SAMPLE_OBLIGATIONS_PATH = APP_DIR / "example_recurring_obligations.csv"


st.set_page_config(page_title="CashSight", page_icon="💼", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .app-card {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.96));
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .eyebrow {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .hero-copy {
            color: #334155;
            max-width: 820px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_projection_chart(
    forecast_df: pd.DataFrame,
    threshold: float,
    scenario_df: pd.DataFrame | None = None,
) -> Any:
    if go is None:
        return None

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["projected_cash"],
            mode="lines",
            name="Baseline cash",
            line={"color": "#0f766e", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=[threshold] * len(forecast_df),
            mode="lines",
            name="Low cash threshold",
            line={"color": "#ef4444", "width": 2, "dash": "dash"},
        )
    )

    if scenario_df is not None:
        figure.add_trace(
            go.Scatter(
                x=scenario_df["date"],
                y=scenario_df["projected_cash"],
                mode="lines",
                name="Scenario cash",
                line={"color": "#2563eb", "width": 3},
            )
        )

    figure.update_layout(
        height=420,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_title="Date",
        yaxis_title="Projected cash",
    )
    figure.update_yaxes(tickprefix="$", gridcolor="rgba(148, 163, 184, 0.18)")
    figure.update_xaxes(gridcolor="rgba(148, 163, 184, 0.12)")
    return figure


def render_projection_view(
    forecast_df: pd.DataFrame,
    threshold: float,
    scenario_df: pd.DataFrame | None = None,
) -> None:
    figure = render_projection_chart(
        forecast_df=forecast_df,
        threshold=threshold,
        scenario_df=scenario_df,
    )
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
        return

    fallback_df = pd.DataFrame(
        {
            "date": forecast_df["date"],
            "Baseline cash": forecast_df["projected_cash"],
            "Low cash threshold": [threshold] * len(forecast_df),
        }
    )
    if scenario_df is not None:
        fallback_df["Scenario cash"] = scenario_df["projected_cash"].values

    st.caption("Plotly is unavailable in this environment, so Streamlit is using a built-in chart fallback.")
    st.line_chart(fallback_df.set_index("date"))


def event_options(events: pd.DataFrame) -> dict[str, str]:
    if events.empty:
        return {}
    options = {}
    for row in events.itertuples(index=False):
        label = f"{row.name} · {pd.Timestamp(row.date).date()} · {money(float(row.amount))}"
        options[label] = row.event_id
    return options


def default_forecast_start(use_sample_data: bool, transactions_file, invoices_file) -> pd.Timestamp:
    default_date = pd.Timestamp.today().normalize()
    if not use_sample_data or transactions_file is not None:
        return default_date

    sample_start = default_date
    if SAMPLE_TRANSACTIONS_PATH.exists():
        sample_transactions = pd.read_csv(SAMPLE_TRANSACTIONS_PATH, usecols=["date"])
        sample_transactions["date"] = pd.to_datetime(sample_transactions["date"], errors="coerce")
        sample_transactions = sample_transactions.dropna(subset=["date"])
        if not sample_transactions.empty:
            sample_start = sample_transactions["date"].max().normalize() + pd.Timedelta(days=1)

    if invoices_file is None and SAMPLE_INVOICES_PATH.exists():
        sample_invoices = pd.read_csv(SAMPLE_INVOICES_PATH)
        if "due_date" in sample_invoices.columns and "status" in sample_invoices.columns:
            sample_invoices["due_date"] = pd.to_datetime(sample_invoices["due_date"], errors="coerce")
            unpaid = sample_invoices[
                sample_invoices["status"].astype(str).str.lower().eq("unpaid")
            ].dropna(subset=["due_date"])
            if not unpaid.empty:
                sample_start = max(sample_start, unpaid["due_date"].min().normalize() - pd.Timedelta(days=2))
    return sample_start


st.markdown('<div class="eyebrow">CashSight</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Forecast cash. Surface risk. Test decisions.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">CashSight is a structured decision-support system for short-term cash forecasting and scenario simulation. It maps invoices into projected inflows, adds recurring obligations, estimates variable spend from history, flags shortfall risks, explains the likely causes, and lets you test timing changes before making a move.</div>',
    unsafe_allow_html=True,
)
st.markdown("")

with st.sidebar:
    st.header("Model settings")
    use_sample_data = st.toggle("Use bundled sample data", value=True)

    st.markdown("**Upload data**")
    transactions_file = st.file_uploader(
        "Transactions CSV",
        type=["csv"],
        help="Required: date, amount. Optional: type, category, description, client_or_vendor, is_recurring.",
    )
    invoices_file = st.file_uploader(
        "Invoices CSV",
        type=["csv"],
        help="Required: amount, issue_date, due_date. Optional: invoice_id, client_name or client, paid_date, status.",
    )
    obligations_file = st.file_uploader(
        "Recurring obligations CSV",
        type=["csv"],
        help="Required: name, amount, frequency, next_due_date.",
    )

    st.markdown("---")
    sidebar_default_start = default_forecast_start(use_sample_data, transactions_file, invoices_file)
    forecast_start = pd.Timestamp(
        st.date_input("Forecast start date", value=sidebar_default_start.date())
    )
    forecast_days = st.slider("Projection horizon", min_value=30, max_value=90, value=60, step=15)
    current_cash_balance = st.number_input("Current cash balance", value=7000.0, step=100.0)
    low_cash_threshold = st.number_input("Low cash threshold", value=2500.0, step=100.0)

    st.markdown("---")
    st.caption(
        "If you skip recurring obligations, CashSight will infer recurring expenses from historical transactions."
    )


transactions_source = SAMPLE_TRANSACTIONS_PATH if use_sample_data and transactions_file is None else transactions_file
invoices_source = SAMPLE_INVOICES_PATH if use_sample_data and invoices_file is None else invoices_file
obligations_source = SAMPLE_OBLIGATIONS_PATH if use_sample_data and obligations_file is None else obligations_file

if transactions_source is None:
    st.info("Upload a transactions CSV or enable the bundled sample data to start the model.")
    schema_col1, schema_col2, schema_col3 = st.columns(3)
    with schema_col1:
        st.markdown("**Transactions**")
        st.code("date,amount,category,description,is_recurring", language="text")
    with schema_col2:
        st.markdown("**Invoices**")
        st.code("client_name,amount,issue_date,due_date,paid_date,status", language="text")
    with schema_col3:
        st.markdown("**Recurring obligations**")
        st.code("name,amount,frequency,next_due_date", language="text")
    st.stop()

try:
    transactions = read_transactions(transactions_source)
except ValueError as exc:
    st.error(f"Could not load transactions data: {exc}")
    st.stop()

invoice_load_error = None
invoices = None
if invoices_source is not None:
    try:
        invoices = read_invoices(invoices_source)
    except ValueError as exc:
        invoice_load_error = str(exc)

obligation_load_error = None
obligations = None
if obligations_source is not None:
    try:
        obligations = read_recurring_obligations(obligations_source)
    except ValueError as exc:
        obligation_load_error = str(exc)

model = build_cashflow_model(
    transactions=transactions,
    invoices=invoices,
    obligations=obligations,
    current_cash_balance=current_cash_balance,
    forecast_start=forecast_start,
    forecast_days=forecast_days,
    low_cash_threshold=low_cash_threshold,
)

forecast_df = model["forecast"]
summary = model["summary"]
risks = model["risks"]
evaluated_scenarios = build_evaluated_scenarios(model)
recommendations = build_ranked_recommendations(model, limit=3)
invoice_delays = model["invoice_delay_by_client"]
recurring_obligations = model["recurring_obligations"]
scenario_expense_events = model["scenario_expense_events"]
scenario_revenue_events = model["scenario_revenue_events"]

first_alert = risks.iloc[0] if not risks.empty else None

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Current cash balance", money(summary["current_cash_balance"]))
metric_col2.metric(
    "Baseline minimum cash",
    money(summary["baseline_min_cash"]),
    delta=f"on {summary['baseline_min_date'].date()}",
)
metric_col3.metric("Negative cash days", summary["negative_cash_days"])
metric_col4.metric("Average payment delay", f"{summary['avg_payment_delay_days']:.1f} days")

chart_col, alert_col = st.columns([1.8, 1.1], gap="large")
with chart_col:
    st.markdown("### Cash projection graph")
    render_projection_view(forecast_df=forecast_df, threshold=low_cash_threshold)
    st.caption(
        "Daily confidence is attached to the forecast: confirmed events = high, estimated events = medium, inferred patterns = low."
    )

with alert_col:
    st.markdown("### Risk alerts")
    if invoice_load_error:
        st.warning(f"Invoices were not loaded: {invoice_load_error}")
    if obligation_load_error:
        st.warning(f"Recurring obligations were not loaded: {obligation_load_error}")

    if risks.empty:
        st.success("No risk events were detected in the current forecast window.")
    else:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(
            f"**{first_alert['severity']} priority**: {first_alert['risk_type']} on "
            f"{pd.Timestamp(first_alert['risk_date']).date()}"
        )
        st.write(first_alert["explanation"])
        st.markdown(
            f"Projected cash: {money(float(first_alert['projected_cash']))}  \n"
            f"Cause type: {first_alert['cause_type']}  \n"
            f"Largest expense: {first_alert['largest_expense']}  \n"
            f"Delayed inflow: {first_alert['delayed_inflow']}  \n"
            f"Timing mismatch: {first_alert['timing_mismatch']}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Cause explanations")
if risks.empty:
    st.info("There are no cause explanations to display because no risk events were triggered.")
else:
    display_risks = risks.copy()
    display_risks["risk_date"] = pd.to_datetime(display_risks["risk_date"]).dt.date
    st.dataframe(
        display_risks[
            [
                "severity",
                "risk_type",
                "cause_type",
                "risk_date",
                "projected_cash",
                "largest_expense",
                "delayed_inflow",
                "timing_mismatch",
                "explanation",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("### Scenario simulation")
scenario_col1, scenario_col2 = st.columns([1.15, 1.55], gap="large")

scenario_df = None
scenario_result = None

with scenario_col1:
    scenario_type = st.selectbox(
        "Scenario type",
        options=[
            ("delay_expense", "Delay an expense"),
            ("accelerate_invoice", "Accelerate an invoice"),
            ("change_revenue_timing", "Change revenue timing"),
            ("adjust_expense_amount", "Adjust expense amount"),
        ],
        format_func=lambda option: option[1],
    )[0]

    if scenario_type in {"delay_expense", "adjust_expense_amount"}:
        available_options = event_options(scenario_expense_events)
        if available_options:
            target_label = st.selectbox(
                "Expense to modify",
                options=list(available_options.keys()),
            )
        else:
            target_label = None
            st.info("No projected expense events are available for this scenario yet.")
    else:
        available_options = event_options(scenario_revenue_events)
        if available_options:
            target_label = st.selectbox(
                "Revenue event to modify",
                options=list(available_options.keys()),
            )
        else:
            target_label = None
            st.info("No projected revenue events are available for this scenario yet.")

    day_shift = 0
    amount_delta = 0.0
    if scenario_type == "delay_expense":
        day_shift = st.number_input("Delay by days", min_value=1, max_value=30, value=7)
    elif scenario_type == "accelerate_invoice":
        day_shift = st.number_input("Accelerate by days", min_value=1, max_value=30, value=7)
    elif scenario_type == "change_revenue_timing":
        day_shift = st.number_input("Shift revenue timing by days", min_value=-30, max_value=30, value=-5)
    elif scenario_type == "adjust_expense_amount":
        amount_delta = st.number_input(
            "Adjust expense amount by",
            value=-300.0,
            step=100.0,
            help="Negative values reduce the expense. Positive values increase it.",
        )

    if available_options and target_label is not None:
        scenario_df, scenario_result = simulate_scenario(
            model=model,
            scenario_type=scenario_type,
            target_event_id=available_options[target_label],
            day_shift=int(day_shift),
            amount_delta=float(amount_delta),
        )

with scenario_col2:
    if scenario_df is None or scenario_result is None:
        st.info("Add future invoices or recurring obligations to unlock scenario simulation.")
    else:
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Baseline min cash", money(scenario_result.baseline_min_cash))
        result_col2.metric(
            "Scenario min cash",
            money(scenario_result.scenario_min_cash),
            delta=money(scenario_result.impact),
        )
        result_col3.metric("Scenario low point", str(scenario_result.scenario_min_date.date()))

        st.code(
            "\n".join(
                [
                    f"Baseline min cash: {money(scenario_result.baseline_min_cash)}",
                    f"Scenario min cash: {money(scenario_result.scenario_min_cash)}",
                    f"Impact: {money(scenario_result.impact)}",
                ]
            ),
            language="text",
        )
        render_projection_view(
            forecast_df=forecast_df,
            threshold=low_cash_threshold,
            scenario_df=scenario_df,
        )

st.markdown("### Recommended actions (ranked)")
if recommendations.empty:
    st.info("No ranked scenarios are available yet because the model does not have enough future events to test.")
else:
    display_recommendations = recommendations[
        [
            "rank_label",
            "action",
            "impact_display",
            "new_min_cash_display",
            "feasibility_score",
            "confidence_score",
            "final_score",
            "explanation",
        ]
    ].rename(
        columns={
            "rank_label": "rank",
            "impact_display": "impact",
            "new_min_cash_display": "new_min_cash",
        }
    )
    st.dataframe(display_recommendations, use_container_width=True, hide_index=True)
    st.markdown("**Full scenario evaluation**")
    scenario_table = evaluated_scenarios[
        [
            "action",
            "impact_display",
            "new_min_cash_display",
            "feasibility_score",
            "confidence_score",
            "final_score",
            "explanation",
        ]
    ].rename(
        columns={
            "impact_display": "impact",
            "new_min_cash_display": "new_min_cash",
        }
    )
    st.dataframe(scenario_table, use_container_width=True, hide_index=True)

detail_tab1, detail_tab2, detail_tab3 = st.tabs(
    ["Forecast detail", "Projected events", "Data inputs"]
)

with detail_tab1:
    display_forecast = forecast_df.copy()
    display_forecast["date"] = pd.to_datetime(display_forecast["date"]).dt.date
    st.dataframe(display_forecast, use_container_width=True, hide_index=True)
    st.download_button(
        "Download forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="cashsight_forecast.csv",
        mime="text/csv",
    )

with detail_tab2:
    events_col1, events_col2 = st.columns(2)
    with events_col1:
        st.markdown("**Projected invoice inflows**")
        if scenario_revenue_events.empty:
            st.info("No unpaid invoices were projected into the selected horizon.")
        else:
            invoice_view = scenario_revenue_events.copy()
            invoice_view["date"] = pd.to_datetime(invoice_view["date"]).dt.date
            invoice_view["due_date"] = pd.to_datetime(invoice_view["due_date"]).dt.date
            st.dataframe(
                invoice_view[
                    ["name", "client_name", "amount", "due_date", "date", "delay_days"]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with events_col2:
        st.markdown("**Projected recurring outflows**")
        if scenario_expense_events.empty:
            st.info("No recurring obligations were projected into the selected horizon.")
        else:
            outflow_view = scenario_expense_events.copy()
            outflow_view["date"] = pd.to_datetime(outflow_view["date"]).dt.date
            st.dataframe(
                outflow_view[["name", "amount", "frequency", "date", "source"]],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("**Variable expense baseline**")
    st.dataframe(model["variable_profile"], use_container_width=True, hide_index=True)

with detail_tab3:
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        st.markdown("**Recent transactions**")
        tx_view = transactions.sort_values("date", ascending=False).head(15).copy()
        tx_view["date"] = pd.to_datetime(tx_view["date"]).dt.date
        st.dataframe(
            tx_view[
                ["date", "flow_type", "amount", "category", "counterparty", "description", "is_recurring"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Recurring obligations in model**")
        if recurring_obligations.empty:
            st.info("No recurring obligations were loaded or inferred.")
        else:
            obligation_view = recurring_obligations.copy()
            obligation_view["next_due_date"] = pd.to_datetime(obligation_view["next_due_date"]).dt.date
            st.dataframe(obligation_view, use_container_width=True, hide_index=True)

    with input_col2:
        st.markdown("**Invoice payment behavior**")
        if invoices is None or invoices.empty:
            st.info("No invoices were loaded.")
        else:
            invoice_history = invoices.copy()
            invoice_history["due_date"] = pd.to_datetime(invoice_history["due_date"]).dt.date
            invoice_history["paid_date"] = pd.to_datetime(invoice_history["paid_date"]).dt.date
            st.dataframe(invoice_history, use_container_width=True, hide_index=True)

        st.markdown("**Average delay by client**")
        if invoice_delays.empty:
            st.info("No paid invoices were available to calculate payment delay.")
        else:
            st.dataframe(invoice_delays, use_container_width=True, hide_index=True)
