from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from forecast_engine import assemble_daily_forecast, money


@dataclass(frozen=True)
class ScenarioResult:
    baseline_min_cash: float
    scenario_min_cash: float
    impact: float
    baseline_min_date: pd.Timestamp
    scenario_min_date: pd.Timestamp
    summary: str


def simulate_scenario(
    model: dict[str, Any],
    scenario_type: str,
    target_event_id: str,
    day_shift: int = 0,
    amount_delta: float = 0.0,
) -> tuple[pd.DataFrame, ScenarioResult]:
    inflow_events = model["inflow_events"].copy()
    outflow_events = model["outflow_events"].copy()
    forecast_df = model["forecast"]
    forecast_start = pd.Timestamp(forecast_df["date"].min()).normalize()
    forecast_dates = pd.DatetimeIndex(forecast_df["date"])
    current_cash_balance = float(model["summary"]["current_cash_balance"])

    if scenario_type == "delay_expense":
        mask = outflow_events["event_id"] == target_event_id
        outflow_events.loc[mask, "date"] = outflow_events.loc[mask, "date"] + pd.Timedelta(days=day_shift)
    elif scenario_type == "accelerate_invoice":
        mask = inflow_events["event_id"] == target_event_id
        new_dates = inflow_events.loc[mask, "date"] - pd.Timedelta(days=abs(day_shift))
        inflow_events.loc[mask, "date"] = new_dates.clip(lower=forecast_start)
    elif scenario_type == "change_revenue_timing":
        mask = inflow_events["event_id"] == target_event_id
        new_dates = inflow_events.loc[mask, "date"] + pd.Timedelta(days=day_shift)
        inflow_events.loc[mask, "date"] = new_dates.clip(lower=forecast_start)
    elif scenario_type == "adjust_expense_amount":
        mask = outflow_events["event_id"] == target_event_id
        outflow_events.loc[mask, "amount"] = (
            outflow_events.loc[mask, "amount"] + amount_delta
        ).clip(lower=0.0)
    else:
        raise ValueError(f"Unsupported scenario type: {scenario_type}")

    scenario_forecast = assemble_daily_forecast(
        forecast_dates=forecast_dates,
        current_cash_balance=current_cash_balance,
        inflow_events=inflow_events,
        outflow_events=outflow_events,
    )
    scenario_forecast.attrs.update(forecast_df.attrs)

    baseline_min_row = forecast_df.loc[forecast_df["projected_cash"].idxmin()]
    scenario_min_row = scenario_forecast.loc[scenario_forecast["projected_cash"].idxmin()]
    baseline_min_cash = float(baseline_min_row["projected_cash"])
    scenario_min_cash = float(scenario_min_row["projected_cash"])
    impact = scenario_min_cash - baseline_min_cash

    result = ScenarioResult(
        baseline_min_cash=round(baseline_min_cash, 2),
        scenario_min_cash=round(scenario_min_cash, 2),
        impact=round(impact, 2),
        baseline_min_date=pd.Timestamp(baseline_min_row["date"]),
        scenario_min_date=pd.Timestamp(scenario_min_row["date"]),
        summary=(
            f"Baseline min cash: {money(baseline_min_cash)} | "
            f"Scenario min cash: {money(scenario_min_cash)} | "
            f"Impact: {money(impact)}"
        ),
    )
    return scenario_forecast, result


def _best_expense_candidates(model: dict[str, Any]) -> pd.DataFrame:
    explicit_expenses = model.get("scenario_expense_events", pd.DataFrame()).copy()
    if explicit_expenses.empty:
        explicit_expenses = model.get("outflow_events", pd.DataFrame()).copy()
    if explicit_expenses.empty:
        return explicit_expenses
    filtered = explicit_expenses[explicit_expenses["event_type"] != "variable_outflow"].copy()
    if filtered.empty:
        filtered = explicit_expenses.copy()
    return filtered.sort_values(["amount", "date"], ascending=[False, True]).reset_index(drop=True)


def _best_revenue_candidates(model: dict[str, Any]) -> pd.DataFrame:
    revenue_events = model.get("scenario_revenue_events", pd.DataFrame()).copy()
    if revenue_events.empty:
        revenue_events = model.get("inflow_events", pd.DataFrame()).copy()
    return revenue_events.sort_values(["amount", "date"], ascending=[False, True]).reset_index(drop=True)


def generate_candidate_scenarios(model: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    baseline_min_date = pd.Timestamp(model["summary"]["baseline_min_date"]).date()

    expense_candidates = _best_expense_candidates(model)
    if not expense_candidates.empty:
        largest_expense = expense_candidates.iloc[0]
        for delay_days in (3, 7):
            scenarios.append(
                {
                    "scenario_type": "delay_expense",
                    "target_event_id": largest_expense["event_id"],
                    "day_shift": delay_days,
                    "amount_delta": 0.0,
                    "action": f"Delay {largest_expense['name']} by {delay_days} days",
                    "explanation": (
                        f"Move {largest_expense['name']} out by {delay_days} days to relieve cash pressure "
                        f"around the baseline low point on {baseline_min_date}."
                    ),
                }
            )

        recurring_candidates = expense_candidates[
            expense_candidates["event_type"] == "recurring_outflow"
        ].copy()
        if not recurring_candidates.empty:
            top_recurring = recurring_candidates.iloc[0]
            reduction = round(float(top_recurring["amount"]) * 0.10, 2)
            scenarios.append(
                {
                    "scenario_type": "adjust_expense_amount",
                    "target_event_id": top_recurring["event_id"],
                    "day_shift": 0,
                    "amount_delta": -reduction,
                    "action": f"Reduce {top_recurring['name']} by 10%",
                    "explanation": (
                        f"Trim the largest recurring expense by {money(reduction)} to improve daily cash coverage."
                    ),
                }
            )

    revenue_candidates = _best_revenue_candidates(model)
    if not revenue_candidates.empty:
        largest_invoice = revenue_candidates.iloc[0]
        scenarios.append(
            {
                "scenario_type": "accelerate_invoice",
                "target_event_id": largest_invoice["event_id"],
                "day_shift": 7,
                "amount_delta": 0.0,
                "action": f"Accelerate {largest_invoice['name']} by 7 days",
                "explanation": (
                    f"Pull the largest invoice receipt forward by a week to lift the minimum cash position."
                ),
            }
        )

        timing_target = revenue_candidates.sort_values("date", ascending=True).iloc[0]
        scenarios.append(
            {
                "scenario_type": "change_revenue_timing",
                "target_event_id": timing_target["event_id"],
                "day_shift": -3,
                "amount_delta": 0.0,
                "action": f"Shift {timing_target['name']} 3 days earlier",
                "explanation": (
                    f"Bring a scheduled inflow forward to reduce short-term timing pressure in the forecast."
                ),
            }
        )

    deduped: list[dict[str, Any]] = []
    seen = set()
    for scenario in scenarios:
        scenario_key = (
            scenario["scenario_type"],
            scenario["target_event_id"],
            scenario["day_shift"],
            round(float(scenario["amount_delta"]), 2),
        )
        if scenario_key in seen:
            continue
        seen.add(scenario_key)
        deduped.append(scenario)
    return deduped


def evaluate_scenarios(model: dict[str, Any], scenarios: list[dict[str, Any]]) -> pd.DataFrame:
    if not scenarios:
        return pd.DataFrame(
            columns=[
                "action",
                "scenario_type",
                "target_event_id",
                "day_shift",
                "amount_delta",
                "new_min_cash",
                "impact",
                "feasibility_score",
                "confidence_score",
                "final_score",
                "explanation",
            ]
        )

    def get_target_row(scenario: dict[str, Any]) -> pd.Series | None:
        scenario_type = scenario["scenario_type"]
        if scenario_type in {"delay_expense", "adjust_expense_amount"}:
            events = model.get("outflow_events", pd.DataFrame())
        else:
            events = model.get("inflow_events", pd.DataFrame())
        if events.empty:
            return None
        matched = events[events["event_id"] == scenario["target_event_id"]]
        if matched.empty:
            return None
        return matched.iloc[0]

    def feasibility_for_scenario(scenario: dict[str, Any], target_row: pd.Series | None) -> tuple[float, str]:
        scenario_type = scenario["scenario_type"]
        shift_days = abs(int(scenario.get("day_shift", 0)))
        amount_delta = float(scenario.get("amount_delta", 0.0))

        if scenario_type == "delay_expense":
            if shift_days <= 3:
                return 0.85, "Short expense deferral is usually the easiest operational change."
            if shift_days <= 7:
                return 0.72, "A one-week expense delay is possible, but needs coordination."
            return 0.5, "Longer expense delays are harder to negotiate reliably."

        if scenario_type == "accelerate_invoice":
            if shift_days <= 3:
                return 0.75, "Pulling an invoice in by a few days is plausible with customer follow-up."
            if shift_days <= 7:
                return 0.58, "Accelerating invoice payment by a week is possible but less certain."
            return 0.38, "Large invoice acceleration requests are difficult to execute consistently."

        if scenario_type == "change_revenue_timing":
            if shift_days <= 3:
                return 0.62, "Small revenue timing adjustments are feasible when work milestones are flexible."
            if shift_days <= 7:
                return 0.48, "Medium revenue timing shifts are harder because client timing must move too."
            return 0.3, "Large revenue timing shifts are usually hard to control."

        if scenario_type == "adjust_expense_amount":
            base_amount = float(target_row["amount"]) if target_row is not None else 0.0
            reduction_ratio = abs(amount_delta) / base_amount if base_amount > 0 else 0.0
            if reduction_ratio <= 0.10:
                return 0.9, "A 10% trim to a recurring expense is usually practical."
            if reduction_ratio <= 0.20:
                return 0.72, "A moderate recurring expense reduction is feasible with some tradeoffs."
            return 0.45, "Large recurring expense cuts are harder to sustain immediately."

        return 0.5, "Feasibility is uncertain for this scenario type."

    def confidence_for_scenario(
        baseline_forecast: pd.DataFrame,
        scenario_forecast: pd.DataFrame,
    ) -> tuple[float, int, str]:
        projected_delta = (
            scenario_forecast["projected_cash"] - baseline_forecast["projected_cash"]
        ).abs()
        net_delta = (
            scenario_forecast["net_cash_flow"] - baseline_forecast["net_cash_flow"]
        ).abs()
        affected_mask = (projected_delta > 0.01) | (net_delta > 0.01)
        affected_days = scenario_forecast.loc[affected_mask].copy()

        if affected_days.empty:
            return 0.6, 0, "No forecast days changed materially, so confidence falls back to the baseline level."

        weights = projected_delta.loc[affected_mask].copy()
        if float(weights.sum()) <= 0:
            weights = pd.Series([1.0] * len(affected_days), index=affected_days.index)

        confidence_score = float(
            (affected_days["confidence_score"] * weights).sum() / weights.sum()
        )
        basis_counts = affected_days["confidence_label"].value_counts().to_dict()
        dominant_basis = max(basis_counts, key=basis_counts.get)
        explanation = (
            f"Confidence comes from {len(affected_days)} affected forecast days, mostly driven by {dominant_basis}-confidence inputs."
        )
        return round(confidence_score, 2), int(len(affected_days)), explanation

    results: list[dict[str, Any]] = []
    baseline_forecast = model["forecast"]
    for scenario in scenarios:
        target_row = get_target_row(scenario)
        scenario_forecast, scenario_result = simulate_scenario(
            model=model,
            scenario_type=scenario["scenario_type"],
            target_event_id=scenario["target_event_id"],
            day_shift=int(scenario.get("day_shift", 0)),
            amount_delta=float(scenario.get("amount_delta", 0.0)),
        )
        feasibility_score, feasibility_reason = feasibility_for_scenario(scenario, target_row)
        confidence_score, affected_days, confidence_reason = confidence_for_scenario(
            baseline_forecast=baseline_forecast,
            scenario_forecast=scenario_forecast,
        )
        final_score = round(
            float(scenario_result.impact) * feasibility_score * confidence_score,
            2,
        )
        results.append(
            {
                "action": scenario["action"],
                "scenario_type": scenario["scenario_type"],
                "target_event_id": scenario["target_event_id"],
                "day_shift": int(scenario.get("day_shift", 0)),
                "amount_delta": round(float(scenario.get("amount_delta", 0.0)), 2),
                "new_min_cash": scenario_result.scenario_min_cash,
                "impact": scenario_result.impact,
                "feasibility_score": round(feasibility_score, 2),
                "confidence_score": round(confidence_score, 2),
                "final_score": final_score,
                "affected_days": affected_days,
                "baseline_min_cash": scenario_result.baseline_min_cash,
                "baseline_min_date": scenario_result.baseline_min_date,
                "scenario_min_date": scenario_result.scenario_min_date,
                "summary": scenario_result.summary,
                "explanation": (
                    f"{scenario['explanation']} {feasibility_reason} {confidence_reason}"
                ),
                "feasibility_reason": feasibility_reason,
                "confidence_reason": confidence_reason,
                "scenario_forecast": scenario_forecast,
            }
        )

    ranked = pd.DataFrame(results).sort_values(
        ["final_score", "impact", "new_min_cash"], ascending=[False, False, False]
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked
