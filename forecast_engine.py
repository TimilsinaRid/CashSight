from __future__ import annotations

from typing import Any

import pandas as pd


TRANSACTION_REQUIRED_COLS = {"date", "amount"}
INVOICE_REQUIRED_COLS = {"amount", "issue_date", "due_date"}
OBLIGATION_REQUIRED_COLS = {"name", "amount", "frequency", "next_due_date"}

CONFIDENCE_SCORES = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.35,
}


def money(value: float) -> str:
    return f"${value:,.0f}"


def _as_timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _coalesce_text(row: pd.Series, columns: list[str], fallback: str) -> str:
    for column in columns:
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).strip()
    return fallback


def _parse_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_transactions(file_obj) -> pd.DataFrame:
    df = pd.read_csv(file_obj).copy()
    missing = TRANSACTION_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Transactions file is missing columns: {', '.join(sorted(missing))}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"]).copy()
    if df.empty:
        raise ValueError("Transactions file does not contain valid dated amounts.")

    if "type" in df.columns:
        flow_map = {
            "income": "inflow",
            "inflow": "inflow",
            "revenue": "inflow",
            "expense": "outflow",
            "outflow": "outflow",
            "cost": "outflow",
        }
        df["flow_type"] = (
            df["type"].astype(str).str.strip().str.lower().map(flow_map).fillna("")
        )
    else:
        df["flow_type"] = ""

    inferred_flow = df["amount"].apply(lambda value: "inflow" if value >= 0 else "outflow")
    df.loc[df["flow_type"] == "", "flow_type"] = inferred_flow
    df["signed_amount"] = df.apply(
        lambda row: abs(row["amount"]) if row["flow_type"] == "inflow" else -abs(row["amount"]),
        axis=1,
    )
    df["amount"] = df["signed_amount"].abs()

    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    df["category"] = df["category"].fillna("Uncategorized").astype(str).str.strip()
    df.loc[df["category"] == "", "category"] = "Uncategorized"

    if "is_recurring" not in df.columns:
        df["is_recurring"] = False
    df["is_recurring"] = df["is_recurring"].apply(_parse_bool)

    df["description"] = df.apply(
        lambda row: _coalesce_text(
            row,
            ["description", "notes", "client_or_vendor", "counterparty"],
            fallback=row["category"],
        ),
        axis=1,
    )
    df["counterparty"] = df.apply(
        lambda row: _coalesce_text(
            row,
            ["client_or_vendor", "counterparty", "description"],
            fallback=row["category"],
        ),
        axis=1,
    )

    return df.sort_values("date").reset_index(drop=True)


def read_invoices(file_obj) -> pd.DataFrame:
    inv = pd.read_csv(file_obj).copy()
    missing = INVOICE_REQUIRED_COLS - set(inv.columns)
    if missing:
        raise ValueError(f"Invoices file is missing columns: {', '.join(sorted(missing))}")

    for column in ["issue_date", "due_date", "paid_date"]:
        if column not in inv.columns:
            inv[column] = pd.NaT
        inv[column] = pd.to_datetime(inv[column], errors="coerce")

    inv["amount"] = pd.to_numeric(inv["amount"], errors="coerce")
    inv = inv.dropna(subset=["amount", "due_date"]).copy()
    if inv.empty:
        return pd.DataFrame(
            columns=[
                "invoice_id",
                "client_name",
                "amount",
                "issue_date",
                "due_date",
                "paid_date",
                "status",
            ]
        )

    if "invoice_id" not in inv.columns:
        inv["invoice_id"] = [f"INV-{idx + 1:03d}" for idx in range(len(inv))]

    client_column = "client_name" if "client_name" in inv.columns else "client"
    if client_column not in inv.columns:
        inv[client_column] = "Unknown client"
    inv["client_name"] = inv[client_column].fillna("Unknown client").astype(str).str.strip()

    if "status" not in inv.columns:
        inv["status"] = inv["paid_date"].apply(lambda value: "paid" if pd.notna(value) else "unpaid")
    inv["status"] = inv["status"].fillna("unpaid").astype(str).str.strip().str.lower()
    inv.loc[inv["paid_date"].notna(), "status"] = "paid"
    inv.loc[inv["paid_date"].isna() & (inv["status"] == ""), "status"] = "unpaid"

    return inv[
        ["invoice_id", "client_name", "amount", "issue_date", "due_date", "paid_date", "status"]
    ].sort_values("due_date").reset_index(drop=True)


def read_recurring_obligations(file_obj) -> pd.DataFrame:
    obligations = pd.read_csv(file_obj).copy()
    missing = OBLIGATION_REQUIRED_COLS - set(obligations.columns)
    if missing:
        raise ValueError(
            f"Recurring obligations file is missing columns: {', '.join(sorted(missing))}"
        )

    obligations["amount"] = pd.to_numeric(obligations["amount"], errors="coerce")
    obligations["next_due_date"] = pd.to_datetime(obligations["next_due_date"], errors="coerce")
    obligations["frequency"] = obligations["frequency"].astype(str).str.strip().str.lower()
    obligations["name"] = obligations["name"].fillna("Recurring obligation").astype(str).str.strip()

    obligations = obligations.dropna(subset=["amount", "next_due_date"]).copy()
    obligations = obligations[obligations["frequency"].isin({"weekly", "monthly"})].copy()
    if obligations.empty:
        return pd.DataFrame(columns=["name", "amount", "frequency", "next_due_date", "source"])

    obligations["amount"] = obligations["amount"].abs()
    obligations["source"] = obligations.get("source", "uploaded")
    return obligations[
        ["name", "amount", "frequency", "next_due_date", "source"]
    ].sort_values("next_due_date").reset_index(drop=True)


def summarize_invoice_delays(invoices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if invoices.empty:
        empty_delay = pd.DataFrame(
            columns=["client_name", "avg_delay_days", "invoice_count", "amount"]
        )
        return invoices.copy(), empty_delay, 0.0

    paid = invoices.dropna(subset=["paid_date"]).copy()
    if paid.empty:
        empty_delay = pd.DataFrame(
            columns=["client_name", "avg_delay_days", "invoice_count", "amount"]
        )
        return paid, empty_delay, 0.0

    paid["delay_days"] = (paid["paid_date"] - paid["due_date"]).dt.days
    by_client = (
        paid.groupby("client_name", as_index=False)
        .agg(
            avg_delay_days=("delay_days", "mean"),
            invoice_count=("invoice_id", "count"),
            amount=("amount", "sum"),
        )
        .sort_values("avg_delay_days", ascending=False)
    )
    by_client["avg_delay_days"] = by_client["avg_delay_days"].round(1)
    by_client["amount"] = by_client["amount"].round(2)
    overall_delay = float(max(paid["delay_days"].mean(), 0.0))
    return paid.reset_index(drop=True), by_client.reset_index(drop=True), round(overall_delay, 1)


def infer_recurring_obligations(transactions: pd.DataFrame) -> pd.DataFrame:
    outflows = transactions[transactions["flow_type"] == "outflow"].copy()
    if outflows.empty:
        return pd.DataFrame(columns=["name", "amount", "frequency", "next_due_date", "source"])

    recurring_records: list[dict[str, Any]] = []
    grouped = outflows.groupby("counterparty", dropna=False)

    for counterparty, group in grouped:
        if not counterparty or len(group) < 2:
            continue

        group = group.sort_values("date")
        gaps = group["date"].diff().dropna().dt.days
        if gaps.empty:
            continue

        median_gap = float(gaps.median())
        if 5 <= median_gap <= 9:
            frequency = "weekly"
            next_due_date = group["date"].max() + pd.Timedelta(days=7)
        elif 23 <= median_gap <= 35:
            frequency = "monthly"
            next_due_date = group["date"].max() + pd.DateOffset(months=1)
        else:
            continue

        recurring_records.append(
            {
                "name": str(counterparty),
                "amount": round(float(group["amount"].mean()), 2),
                "frequency": frequency,
                "next_due_date": _as_timestamp(next_due_date),
                "source": "detected",
            }
        )

    if not recurring_records:
        return pd.DataFrame(columns=["name", "amount", "frequency", "next_due_date", "source"])

    recurring_df = pd.DataFrame(recurring_records).drop_duplicates(subset=["name", "frequency"])
    return recurring_df.sort_values(["next_due_date", "name"]).reset_index(drop=True)


def project_invoice_inflows(
    invoices: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    delay_by_client: pd.DataFrame,
    overall_delay_days: float,
) -> pd.DataFrame:
    empty_columns = [
        "event_id",
        "date",
        "amount",
        "event_type",
        "name",
        "client_name",
        "invoice_id",
        "due_date",
        "expected_date",
        "delay_days",
        "source",
    ]
    if invoices.empty:
        return pd.DataFrame(columns=empty_columns)

    open_invoices = invoices[invoices["status"] != "paid"].copy()
    if open_invoices.empty:
        return pd.DataFrame(columns=empty_columns)

    client_delay_map = delay_by_client.set_index("client_name")["avg_delay_days"].to_dict()
    records: list[dict[str, Any]] = []

    for invoice in open_invoices.itertuples(index=False):
        delay_days = float(client_delay_map.get(invoice.client_name, overall_delay_days))
        expected_date = invoice.due_date + pd.Timedelta(days=max(round(delay_days), 0))
        expected_date = max(_as_timestamp(expected_date), _as_timestamp(forecast_start))
        if expected_date > forecast_end:
            continue

        records.append(
            {
                "event_id": f"invoice::{invoice.invoice_id}",
                "date": _as_timestamp(expected_date),
                "amount": round(float(invoice.amount), 2),
                "event_type": "invoice_inflow",
                "name": f"{invoice.invoice_id} · {invoice.client_name}",
                "client_name": invoice.client_name,
                "invoice_id": invoice.invoice_id,
                "due_date": _as_timestamp(invoice.due_date),
                "expected_date": _as_timestamp(expected_date),
                "delay_days": round(delay_days, 1),
                "source": "confirmed",
            }
        )

    if not records:
        return pd.DataFrame(columns=empty_columns)

    return pd.DataFrame(records).sort_values(["date", "amount"], ascending=[True, False]).reset_index(drop=True)


def project_recurring_outflows(
    obligations: pd.DataFrame,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
) -> pd.DataFrame:
    empty_columns = [
        "event_id",
        "date",
        "amount",
        "event_type",
        "name",
        "frequency",
        "source",
    ]
    if obligations.empty:
        return pd.DataFrame(columns=empty_columns)

    records: list[dict[str, Any]] = []
    for row in obligations.itertuples(index=False):
        due_date = _as_timestamp(row.next_due_date)
        while due_date <= forecast_end:
            if due_date >= forecast_start:
                records.append(
                    {
                        "event_id": f"expense::{row.name}::{due_date.date()}",
                        "date": due_date,
                        "amount": round(float(row.amount), 2),
                        "event_type": "recurring_outflow",
                        "name": row.name,
                        "frequency": row.frequency,
                        "source": row.source,
                    }
                )
            if row.frequency == "weekly":
                due_date = due_date + pd.Timedelta(days=7)
            else:
                due_date = due_date + pd.DateOffset(months=1)

    if not records:
        return pd.DataFrame(columns=empty_columns)

    return pd.DataFrame(records).sort_values(["date", "amount"], ascending=[True, False]).reset_index(drop=True)


def estimate_variable_expense_events(
    transactions: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    forecast_start: pd.Timestamp,
    recurring_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_end = forecast_start - pd.Timedelta(days=1)
    history_start = history_end - pd.Timedelta(days=89)
    history = transactions[
        (transactions["date"] >= history_start)
        & (transactions["date"] <= history_end)
        & (transactions["flow_type"] == "outflow")
    ].copy()

    empty_events = pd.DataFrame(
        columns=["event_id", "date", "amount", "event_type", "name", "source"]
    )
    empty_profile = pd.DataFrame(columns=["weekday", "avg_variable_outflow"])
    if history.empty:
        return empty_events, empty_profile

    history["normalized_counterparty"] = history["counterparty"].map(_normalize_name)
    history["normalized_description"] = history["description"].map(_normalize_name)
    recurring_mask = history["is_recurring"]
    if recurring_names:
        recurring_mask = recurring_mask | history["normalized_counterparty"].isin(recurring_names)
        recurring_mask = recurring_mask | history["normalized_description"].isin(recurring_names)

    variable_history = history[~recurring_mask].copy()
    if variable_history.empty:
        variable_history = history.copy()

    daily = (
        variable_history.groupby(variable_history["date"].dt.normalize())["amount"]
        .sum()
        .sort_index()
    )
    history_dates = pd.date_range(start=history_start, end=history_end, freq="D")
    daily = daily.reindex(history_dates, fill_value=0.0)
    weekday_average = daily.groupby(daily.index.dayofweek).mean().round(2)

    profile = pd.DataFrame(
        {
            "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "avg_variable_outflow": [float(weekday_average.get(day, 0.0)) for day in range(7)],
        }
    )

    records: list[dict[str, Any]] = []
    for date in forecast_dates:
        amount = round(float(weekday_average.get(date.dayofweek, 0.0)), 2)
        if amount <= 0:
            continue
        records.append(
            {
                "event_id": f"variable::{date.date()}",
                "date": _as_timestamp(date),
                "amount": amount,
                "event_type": "variable_outflow",
                "name": "Estimated variable expenses",
                "source": "historical_average",
            }
        )

    if not records:
        return empty_events, profile

    return pd.DataFrame(records).reset_index(drop=True), profile


def _assign_event_confidence(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        copy = events.copy()
        copy["confidence_label"] = pd.Series(dtype=str)
        copy["confidence_basis"] = pd.Series(dtype=str)
        copy["confidence_score"] = pd.Series(dtype=float)
        return copy

    events = events.copy()

    def classify(row: pd.Series) -> tuple[str, str]:
        if row["event_type"] == "invoice_inflow":
            return "high", "confirmed"
        if row["event_type"] == "recurring_outflow":
            if str(row.get("source", "")).strip().lower() in {"uploaded", "confirmed", "manual"}:
                return "high", "confirmed"
            return "low", "inferred"
        return "medium", "estimated"

    labels = events.apply(classify, axis=1, result_type="expand")
    labels.columns = ["confidence_label", "confidence_basis"]
    events[["confidence_label", "confidence_basis"]] = labels
    events["confidence_score"] = events["confidence_label"].map(CONFIDENCE_SCORES).astype(float)
    return events


def _add_daily_confidence(
    forecast_df: pd.DataFrame,
    inflow_events: pd.DataFrame,
    outflow_events: pd.DataFrame,
) -> pd.DataFrame:
    all_events = []
    for events in (inflow_events, outflow_events):
        if not events.empty:
            all_events.append(
                events[
                    ["date", "amount", "confidence_label", "confidence_basis", "confidence_score"]
                ].copy()
            )

    if not all_events:
        forecast_df["confidence_score"] = 0.6
        forecast_df["confidence_label"] = "medium"
        forecast_df["confidence_basis"] = "estimated"
        return forecast_df

    combined = pd.concat(all_events, ignore_index=True)

    def confidence_summary(group: pd.DataFrame) -> pd.Series:
        weights = group["amount"].abs()
        total_weight = float(weights.sum())
        if total_weight <= 0:
            score = 0.6
        else:
            score = float((weights * group["confidence_score"]).sum() / total_weight)
        if score >= 0.75:
            label = "high"
        elif score >= 0.5:
            label = "medium"
        else:
            label = "low"
        basis = "/".join(sorted(set(group["confidence_basis"])))
        return pd.Series(
            {
                "confidence_score": round(score, 2),
                "confidence_label": label,
                "confidence_basis": basis,
            }
        )

    daily_confidence = combined.groupby("date").apply(confidence_summary)
    forecast_df["confidence_score"] = (
        forecast_df["date"].map(daily_confidence["confidence_score"]).fillna(0.6).round(2)
    )
    forecast_df["confidence_label"] = forecast_df["date"].map(daily_confidence["confidence_label"]).fillna("medium")
    forecast_df["confidence_basis"] = forecast_df["date"].map(daily_confidence["confidence_basis"]).fillna("estimated")
    return forecast_df


def assemble_daily_forecast(
    forecast_dates: pd.DatetimeIndex,
    current_cash_balance: float,
    inflow_events: pd.DataFrame,
    outflow_events: pd.DataFrame,
) -> pd.DataFrame:
    forecast_df = pd.DataFrame({"date": forecast_dates})

    def grouped_amounts(events: pd.DataFrame, event_type: str) -> pd.Series:
        if events.empty:
            return pd.Series(dtype=float)
        typed = events[events["event_type"] == event_type]
        if typed.empty:
            return pd.Series(dtype=float)
        return typed.groupby("date")["amount"].sum()

    invoice_inflows = grouped_amounts(inflow_events, "invoice_inflow")
    recurring_outflows = grouped_amounts(outflow_events, "recurring_outflow")
    variable_outflows = grouped_amounts(outflow_events, "variable_outflow")

    forecast_df["invoice_inflows"] = forecast_df["date"].map(invoice_inflows).fillna(0.0)
    forecast_df["recurring_outflows"] = forecast_df["date"].map(recurring_outflows).fillna(0.0)
    forecast_df["variable_outflows"] = forecast_df["date"].map(variable_outflows).fillna(0.0)
    forecast_df["inflows"] = forecast_df["invoice_inflows"]
    forecast_df["outflows"] = forecast_df["recurring_outflows"] + forecast_df["variable_outflows"]
    forecast_df["net_cash_flow"] = forecast_df["inflows"] - forecast_df["outflows"]
    forecast_df["projected_cash"] = current_cash_balance + forecast_df["net_cash_flow"].cumsum()
    return _add_daily_confidence(forecast_df, inflow_events, outflow_events)


def _calculate_forecast_meta(
    forecast_df: pd.DataFrame,
    inflow_events: pd.DataFrame,
    invoices: pd.DataFrame,
) -> dict[str, Any]:
    dominant_client_name = None
    dominant_client_share = 0.0
    revenue_source = pd.DataFrame()
    if not inflow_events.empty and "client_name" in inflow_events.columns:
        revenue_source = inflow_events.groupby("client_name", as_index=False)["amount"].sum()
    elif not invoices.empty:
        revenue_source = invoices.groupby("client_name", as_index=False)["amount"].sum()

    if not revenue_source.empty:
        client_totals = revenue_source.copy()
        total_invoices = float(client_totals["amount"].sum())
        if total_invoices > 0 and not client_totals.empty:
            client_totals["share"] = client_totals["amount"] / total_invoices
            top_client = client_totals.sort_values("share", ascending=False).iloc[0]
            dominant_client_name = str(top_client["client_name"])
            dominant_client_share = float(top_client["share"])

    inflow_days = forecast_df[forecast_df["inflows"] > 0][["date", "inflows"]].copy()
    if len(inflow_days) >= 2:
        gap_days = inflow_days["date"].sort_values().diff().dropna().dt.days
        inflow_gap_std = float(gap_days.std(ddof=0)) if not gap_days.empty else 0.0
    else:
        inflow_gap_std = 0.0

    inflow_amount_cv = 0.0
    if not inflow_days.empty and float(inflow_days["inflows"].mean()) > 0:
        inflow_amount_cv = float(inflow_days["inflows"].std(ddof=0) / inflow_days["inflows"].mean())

    meta = {
        "dominant_client_name": dominant_client_name,
        "dominant_client_share": round(dominant_client_share, 4),
        "inflow_gap_std": round(inflow_gap_std, 2),
        "inflow_amount_cv": round(inflow_amount_cv, 2),
        "confirmed_event_days": int(
            (forecast_df["confidence_basis"].fillna("").str.contains("confirmed")).sum()
        ),
        "estimated_event_days": int(
            (forecast_df["confidence_basis"].fillna("").str.contains("estimated")).sum()
        ),
        "inferred_event_days": int(
            (forecast_df["confidence_basis"].fillna("").str.contains("inferred")).sum()
        ),
    }
    forecast_df.attrs.update(meta)
    return meta


def build_cashflow_model(
    transactions: pd.DataFrame,
    invoices: pd.DataFrame | None,
    obligations: pd.DataFrame | None,
    current_cash_balance: float,
    forecast_start: pd.Timestamp,
    forecast_days: int,
    low_cash_threshold: float,
) -> dict[str, Any]:
    from risk_engine import detect_risks

    invoices = invoices.copy() if invoices is not None else pd.DataFrame()
    obligations = obligations.copy() if obligations is not None else pd.DataFrame()

    forecast_start = _as_timestamp(forecast_start)
    forecast_dates = pd.date_range(start=forecast_start, periods=forecast_days, freq="D")
    forecast_end = pd.Timestamp(forecast_dates[-1]).normalize()

    recurring_obligations = (
        infer_recurring_obligations(transactions) if obligations.empty else obligations.copy()
    )

    paid_invoices, delay_by_client, overall_delay_days = summarize_invoice_delays(invoices)
    invoice_events = project_invoice_inflows(
        invoices=invoices,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        delay_by_client=delay_by_client,
        overall_delay_days=overall_delay_days,
    )
    recurring_events = project_recurring_outflows(
        obligations=recurring_obligations,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )

    recurring_names = {
        _normalize_name(name) for name in recurring_obligations.get("name", pd.Series(dtype=str))
    }
    variable_events, variable_profile = estimate_variable_expense_events(
        transactions=transactions,
        forecast_dates=forecast_dates,
        forecast_start=forecast_start,
        recurring_names=recurring_names,
    )

    inflow_events = _assign_event_confidence(invoice_events.copy())
    outflow_events = _assign_event_confidence(
        pd.concat([recurring_events, variable_events], ignore_index=True)
    )

    forecast_df = assemble_daily_forecast(
        forecast_dates=forecast_dates,
        current_cash_balance=current_cash_balance,
        inflow_events=inflow_events,
        outflow_events=outflow_events,
    )
    forecast_meta = _calculate_forecast_meta(forecast_df, inflow_events, invoices)

    risk_df = detect_risks(
        forecast_df=forecast_df,
        inflow_events=inflow_events,
        outflow_events=outflow_events,
        invoices=invoices,
        delay_by_client=delay_by_client,
        threshold=low_cash_threshold,
    )

    min_row = forecast_df.loc[forecast_df["projected_cash"].idxmin()]
    summary = {
        "current_cash_balance": round(float(current_cash_balance), 2),
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "baseline_min_cash": round(float(min_row["projected_cash"]), 2),
        "baseline_min_date": pd.Timestamp(min_row["date"]),
        "negative_cash_days": int((forecast_df["projected_cash"] < 0).sum()),
        "low_buffer_days": int((forecast_df["projected_cash"] < low_cash_threshold).sum()),
        "avg_payment_delay_days": round(float(overall_delay_days), 1),
    }

    return {
        "forecast": forecast_df,
        "summary": summary,
        "risks": risk_df,
        "inflow_events": inflow_events,
        "outflow_events": outflow_events,
        "scenario_expense_events": recurring_events,
        "scenario_revenue_events": invoice_events,
        "invoice_delay_by_client": delay_by_client,
        "paid_invoices": paid_invoices,
        "recurring_obligations": recurring_obligations,
        "variable_profile": variable_profile,
        "forecast_meta": forecast_meta,
    }
