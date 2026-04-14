from __future__ import annotations

import pandas as pd

from forecast_engine import money
from scenario_engine import evaluate_scenarios, generate_candidate_scenarios


def build_evaluated_scenarios(model: dict) -> pd.DataFrame:
    scenarios = generate_candidate_scenarios(model)
    evaluated = evaluate_scenarios(model, scenarios)
    if evaluated.empty:
        return pd.DataFrame(
            columns=[
                "action",
                "impact",
                "new_min_cash",
                "feasibility_score",
                "confidence_score",
                "final_score",
                "explanation",
            ]
        )

    evaluated = evaluated.copy()
    evaluated["impact_display"] = evaluated["impact"].apply(money)
    evaluated["new_min_cash_display"] = evaluated["new_min_cash"].apply(money)
    return evaluated.reset_index(drop=True)


def build_ranked_recommendations(model: dict, limit: int = 3) -> pd.DataFrame:
    ranked = build_evaluated_scenarios(model)
    if ranked.empty:
        return pd.DataFrame(
            columns=[
                "rank_label",
                "action",
                "impact",
                "impact_display",
                "new_min_cash",
                "new_min_cash_display",
                "feasibility_score",
                "confidence_score",
                "final_score",
                "explanation",
            ]
        )

    recommendations = ranked.head(limit).copy()
    rank_labels = ["Best action", "Second", "Third"]
    recommendations["rank_label"] = [
        rank_labels[index] if index < len(rank_labels) else f"Rank {index + 1}"
        for index in range(len(recommendations))
    ]
    recommendations["explanation"] = recommendations.apply(
        lambda row: (
            f"{row['explanation']} New minimum cash: {money(float(row['new_min_cash']))}. "
            f"Final score: {row['final_score']:.2f}."
        ),
        axis=1,
    )
    return recommendations.reset_index(drop=True)
