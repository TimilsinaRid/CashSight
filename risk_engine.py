from __future__ import annotations

from typing import Any

import pandas as pd


def money(value: float) -> str:
    return f"${value:,.0f}"


def _get_value(risk_event: Any, key: str, default=None):
    if isinstance(risk_event, dict):
        return risk_event.get(key, default)
    if isinstance(risk_event, pd.Series):
        return risk_event.get(key, default)
    return getattr(risk_event, key, default)


def _as_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    return pd.Timestamp(value).normalize()


def _event_context(
    risk_date: pd.Timestamp,
    inflow_events: pd.DataFrame,
    outflow_events: pd.DataFrame,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "largest_expense_name": None,
        "largest_expense_amount": None,
        "largest_expense_date": None,
        "delayed_inflow_name": None,
        "delayed_inflow_amount": None,
        "delayed_inflow_date": None,
        "timing_mismatch": False,
    }

    if not outflow_events.empty:
        window_start = risk_date - pd.Timedelta(days=4)
        window_end = risk_date + pd.Timedelta(days=4)
        nearby_expenses = outflow_events[
            (outflow_events["date"] >= window_start) & (outflow_events["date"] <= window_end)
        ].copy()
        if nearby_expenses.empty:
            nearby_expenses = outflow_events[outflow_events["date"] <= risk_date].copy()
        if not nearby_expenses.empty:
            largest_expense = nearby_expenses.sort_values(
                ["amount", "date"], ascending=[False, True]
            ).iloc[0]
            context["largest_expense_name"] = largest_expense["name"]
            context["largest_expense_amount"] = float(largest_expense["amount"])
            context["largest_expense_date"] = pd.Timestamp(largest_expense["date"])

    if not inflow_events.empty:
        delayed = inflow_events.sort_values(["date", "amount"], ascending=[True, False]).copy()
        delayed = delayed[delayed["date"] >= risk_date].head(1)
        if delayed.empty and "due_date" in inflow_events.columns:
            delayed = inflow_events[inflow_events["due_date"] <= risk_date].sort_values(
                ["date", "amount"], ascending=[True, False]
            ).head(1)
        if not delayed.empty:
            inflow = delayed.iloc[0]
            context["delayed_inflow_name"] = inflow["name"]
            context["delayed_inflow_amount"] = float(inflow["amount"])
            context["delayed_inflow_date"] = pd.Timestamp(inflow["date"])

    expense_date = context["largest_expense_date"]
    inflow_date = context["delayed_inflow_date"]
    if expense_date is not None and inflow_date is not None and expense_date <= inflow_date:
        context["timing_mismatch"] = True

    return context


def classify_risk_cause(risk_event, forecast_df: pd.DataFrame) -> dict[str, str]:
    risk_date = _as_timestamp(_get_value(risk_event, "risk_date", forecast_df["date"].min()))
    expense_name = _get_value(risk_event, "largest_expense", "a large expense")
    inflow_name = _get_value(risk_event, "delayed_inflow", "a delayed inflow")
    expense_date = _as_timestamp(_get_value(risk_event, "largest_expense_date"))
    inflow_date = _as_timestamp(_get_value(risk_event, "delayed_inflow_date"))
    dominant_client_name = forecast_df.attrs.get("dominant_client_name") or "one client"
    dominant_client_share = float(forecast_df.attrs.get("dominant_client_share", 0.0))
    inflow_gap_std = float(forecast_df.attrs.get("inflow_gap_std", 0.0))
    inflow_amount_cv = float(forecast_df.attrs.get("inflow_amount_cv", 0.0))

    window_start = risk_date - pd.Timedelta(days=13)
    window = forecast_df[
        (forecast_df["date"] >= window_start) & (forecast_df["date"] <= risk_date)
    ].copy()
    negative_days = int((window["outflows"] > window["inflows"]).sum()) if not window.empty else 0
    structural_deficit = (
        not window.empty
        and float(window["outflows"].sum()) > float(window["inflows"].sum())
        and negative_days >= min(7, len(window))
    )
    timing_mismatch = expense_date is not None and inflow_date is not None and expense_date <= inflow_date
    concentration_risk = dominant_client_share > 0.5
    volatility_risk = inflow_gap_std >= 4 or inflow_amount_cv >= 0.75

    if timing_mismatch:
        return {
            "cause_type": "timing_mismatch",
            "explanation": (
                f"Cash pressure on {risk_date.date()} is a timing mismatch: {expense_name} lands before "
                f"{inflow_name}, so outflows hit before the offsetting inflow arrives."
            ),
        }
    if structural_deficit:
        return {
            "cause_type": "structural_deficit",
            "explanation": (
                f"Cash pressure on {risk_date.date()} reflects a structural deficit: outflows exceed inflows "
                f"across {negative_days} recent forecast days."
            ),
        }
    if concentration_risk:
        return {
            "cause_type": "concentration_risk",
            "explanation": (
                f"Cash risk is concentrated because {dominant_client_name} represents "
                f"{dominant_client_share:.0%} of modeled inflow value."
            ),
        }
    if volatility_risk:
        return {
            "cause_type": "volatility_risk",
            "explanation": (
                f"Cash risk is tied to inflow volatility: expected inflows arrive on an irregular cadence, "
                f"which increases the chance of temporary shortfalls."
            ),
        }
    return {
        "cause_type": "structural_deficit",
        "explanation": (
            f"Cash pressure on {risk_date.date()} is driven by modeled outflows staying ahead of inflows."
        ),
    }


def detect_risks(
    forecast_df: pd.DataFrame,
    inflow_events: pd.DataFrame,
    outflow_events: pd.DataFrame,
    invoices: pd.DataFrame,
    delay_by_client: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    events: list[dict[str, Any]] = []

    def add_grouped_risks(mask: pd.Series, risk_type: str, severity: str) -> None:
        subset = forecast_df[mask].copy()
        if subset.empty:
            return
        subset["group_id"] = (subset["date"].diff().dt.days.ne(1)).cumsum()
        for _, streak in subset.groupby("group_id"):
            focal = streak.loc[streak["projected_cash"].idxmin()]
            risk_date = pd.Timestamp(focal["date"])
            context = _event_context(risk_date, inflow_events, outflow_events)
            events.append(
                {
                    "risk_type": risk_type,
                    "severity": severity,
                    "risk_date": risk_date,
                    "projected_cash": round(float(focal["projected_cash"]), 2),
                    "largest_expense": context["largest_expense_name"] or "None identified",
                    "largest_expense_date": context["largest_expense_date"],
                    "largest_expense_amount": context["largest_expense_amount"],
                    "delayed_inflow": context["delayed_inflow_name"] or "None identified",
                    "delayed_inflow_date": context["delayed_inflow_date"],
                    "delayed_inflow_amount": context["delayed_inflow_amount"],
                    "timing_mismatch": "Yes" if context["timing_mismatch"] else "No",
                }
            )

    add_grouped_risks(forecast_df["projected_cash"] < 0, "Negative cash day", "High")
    add_grouped_risks(
        (forecast_df["projected_cash"] >= 0) & (forecast_df["projected_cash"] < threshold),
        "Low cash buffer",
        "Medium",
    )

    if not forecast_df.empty:
        outflow_mean = float(forecast_df["outflows"].mean())
        outflow_std = float(forecast_df["outflows"].std(ddof=0))
        spike_cutoff = outflow_mean + outflow_std
        spike_days = forecast_df[forecast_df["outflows"] > spike_cutoff].sort_values(
            "outflows", ascending=False
        )
        for row in spike_days.head(3).itertuples(index=False):
            risk_date = pd.Timestamp(row.date)
            context = _event_context(risk_date, inflow_events, outflow_events)
            events.append(
                {
                    "risk_type": "Expense spike",
                    "severity": "Medium",
                    "risk_date": risk_date,
                    "projected_cash": round(float(row.projected_cash), 2),
                    "largest_expense": context["largest_expense_name"] or "None identified",
                    "largest_expense_date": context["largest_expense_date"],
                    "largest_expense_amount": context["largest_expense_amount"],
                    "delayed_inflow": context["delayed_inflow_name"] or "None identified",
                    "delayed_inflow_date": context["delayed_inflow_date"],
                    "delayed_inflow_amount": context["delayed_inflow_amount"],
                    "timing_mismatch": "Yes" if context["timing_mismatch"] else "No",
                }
            )

    client_totals = pd.DataFrame(columns=["client_name", "share"])
    revenue_base = pd.DataFrame()
    if not inflow_events.empty and "client_name" in inflow_events.columns:
        revenue_base = inflow_events.groupby("client_name", as_index=False)["amount"].sum()
    elif not invoices.empty:
        revenue_base = invoices.groupby("client_name", as_index=False)["amount"].sum()

    if not revenue_base.empty and float(revenue_base["amount"].sum()) > 0:
        revenue_base["share"] = revenue_base["amount"] / revenue_base["amount"].sum()
        client_totals = revenue_base.sort_values("share", ascending=False)
        top_client = client_totals.iloc[0]
        if float(top_client["share"]) > 0.5:
            risk_date = pd.Timestamp(forecast_df["date"].min())
            context = _event_context(risk_date, inflow_events, outflow_events)
            events.append(
                {
                    "risk_type": "Revenue concentration risk",
                    "severity": "Medium",
                    "risk_date": risk_date,
                    "projected_cash": round(float(forecast_df.iloc[0]["projected_cash"]), 2),
                    "largest_expense": context["largest_expense_name"] or "None identified",
                    "largest_expense_date": context["largest_expense_date"],
                    "largest_expense_amount": context["largest_expense_amount"],
                    "delayed_inflow": context["delayed_inflow_name"] or "None identified",
                    "delayed_inflow_date": context["delayed_inflow_date"],
                    "delayed_inflow_amount": context["delayed_inflow_amount"],
                    "timing_mismatch": "Yes" if context["timing_mismatch"] else "No",
                }
            )

    if not delay_by_client.empty and not client_totals.empty:
        dependency = client_totals.merge(delay_by_client, on="client_name", how="left").fillna(
            {"avg_delay_days": 0.0}
        )
        dependency = dependency.sort_values(["share", "avg_delay_days"], ascending=[False, False])
        at_risk = dependency[
            (dependency["share"] >= 0.2) & (dependency["avg_delay_days"] >= 5)
        ].head(1)
        if not at_risk.empty:
            client_row = at_risk.iloc[0]
            risk_date = pd.Timestamp(forecast_df["date"].min())
            context = _event_context(risk_date, inflow_events, outflow_events)
            events.append(
                {
                    "risk_type": "Late-paying client dependency",
                    "severity": "High",
                    "risk_date": risk_date,
                    "projected_cash": round(float(forecast_df.iloc[0]["projected_cash"]), 2),
                    "largest_expense": context["largest_expense_name"] or "None identified",
                    "largest_expense_date": context["largest_expense_date"],
                    "largest_expense_amount": context["largest_expense_amount"],
                    "delayed_inflow": context["delayed_inflow_name"] or "None identified",
                    "delayed_inflow_date": context["delayed_inflow_date"],
                    "delayed_inflow_amount": context["delayed_inflow_amount"],
                    "timing_mismatch": "Yes" if context["timing_mismatch"] else "No",
                    "dependency_client": client_row["client_name"],
                    "dependency_share": round(float(client_row["share"]), 4),
                    "avg_delay_days": round(float(client_row["avg_delay_days"]), 1),
                }
            )

    if not events:
        return pd.DataFrame(
            columns=[
                "risk_type",
                "severity",
                "risk_date",
                "projected_cash",
                "cause_type",
                "largest_expense",
                "delayed_inflow",
                "timing_mismatch",
                "explanation",
            ]
        )

    risk_df = pd.DataFrame(events)
    cause_details = risk_df.apply(
        lambda row: pd.Series(classify_risk_cause(row, forecast_df)),
        axis=1,
    )
    risk_df[["cause_type", "explanation"]] = cause_details[["cause_type", "explanation"]]

    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    risk_df["severity_rank"] = risk_df["severity"].map(severity_order).fillna(9)
    risk_df = risk_df.sort_values(["severity_rank", "risk_date"]).drop(columns=["severity_rank"])
    return risk_df.reset_index(drop=True)
