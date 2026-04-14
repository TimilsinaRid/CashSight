from __future__ import annotations

from pathlib import Path

import pandas as pd

from cashsight_engine import (
    build_cashflow_model,
    build_evaluated_scenarios,
    build_ranked_recommendations,
    money,
    read_invoices,
    read_recurring_obligations,
    read_transactions,
)


APP_DIR = Path(__file__).resolve().parent
TRANSACTIONS_PATH = APP_DIR / "example_transactions.csv"
INVOICES_PATH = APP_DIR / "example_Invoices.csv"
OBLIGATIONS_PATH = APP_DIR / "example_recurring_obligations.csv"
OUTPUT_PATH = APP_DIR / "forecast_output.csv"


def main() -> None:
    transactions = read_transactions(TRANSACTIONS_PATH)
    invoices = read_invoices(INVOICES_PATH) if INVOICES_PATH.exists() else pd.DataFrame()
    obligations = (
        read_recurring_obligations(OBLIGATIONS_PATH) if OBLIGATIONS_PATH.exists() else pd.DataFrame()
    )

    model = build_cashflow_model(
        transactions=transactions,
        invoices=invoices,
        obligations=obligations,
        current_cash_balance=7000.0,
        forecast_start=pd.Timestamp.today().normalize(),
        forecast_days=60,
        low_cash_threshold=2500.0,
    )

    forecast = model["forecast"].copy()
    forecast.to_csv(OUTPUT_PATH, index=False)

    summary = model["summary"]
    print("=== CashSight Forecast ===")
    print(f"Forecast window: {summary['forecast_start'].date()} to {summary['forecast_end'].date()}")
    print(f"Baseline minimum cash: {money(summary['baseline_min_cash'])} on {summary['baseline_min_date'].date()}")
    print(f"Negative cash days: {summary['negative_cash_days']}")
    print(f"Low buffer days: {summary['low_buffer_days']}")
    print(f"Average payment delay: {summary['avg_payment_delay_days']:.1f} days")
    print("")
    print("=== Top risk alerts ===")
    if model["risks"].empty:
        print("No risks detected in the baseline forecast.")
    else:
        risk_view = model["risks"][
            ["severity", "risk_type", "cause_type", "risk_date", "projected_cash", "explanation"]
        ]
        print(risk_view.head(5).to_string(index=False))
    print("")
    print("=== Ranked actions ===")
    recommendations = build_ranked_recommendations(model, limit=3)
    if recommendations.empty:
        print("No scenario recommendations are available.")
    else:
        recommendation_view = recommendations[["rank_label", "action", "impact_display", "explanation"]]
        print(recommendation_view.to_string(index=False))
    print("")
    print("=== Full scenario evaluation ===")
    evaluated_scenarios = build_evaluated_scenarios(model)
    if evaluated_scenarios.empty:
        print("No evaluated scenarios are available.")
    else:
        scenario_view = evaluated_scenarios[
            [
                "action",
                "impact_display",
                "new_min_cash_display",
                "feasibility_score",
                "confidence_score",
                "final_score",
                "explanation",
            ]
        ]
        print(scenario_view.to_string(index=False))
    print("")
    print(f"Saved detailed forecast to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
