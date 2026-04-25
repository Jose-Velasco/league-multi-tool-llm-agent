from __future__ import annotations

from pathlib import Path

import pandas as pd

from league_multi_tool_llm_agent.evaluation.agents import EvalCaseResult


def save_eval_results(
    *,
    results: list[EvalCaseResult],
    output_dir: str,
) -> None:
    """
    Save evaluation results using pandas.

    Outputs:
    - detailed CSV (full results and judge scores)
    - manual grading CSV (for human annotation)
    - summary CSV (aggregated metrics per condition)
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    detailed_path = out / "recommendation_eval_detailed.csv"
    manual_path = out / "recommendation_eval_manual_grading.csv"
    summary_path = out / "recommendation_eval_summary.csv"

    df = pd.DataFrame([r.__dict__ for r in results])

    df.to_csv(detailed_path, index=False)

    # Manual Grading Sheet
    manual_df = df[
        [
            "condition",
            "model_name",
            "use_rag",
            "query",
            "answer",
            "retrieved_context",
        ]
    ].copy()

    # Add empty grading columns
    manual_df["manual_relevance_1_5"] = ""
    manual_df["manual_explanation_1_5"] = ""
    manual_df["manual_personalization_1_5"] = ""
    manual_df["manual_groundedness_1_5"] = ""
    manual_df["manual_notes"] = ""

    manual_df.to_csv(manual_path, index=False)

    summary_df = df.groupby(["condition", "model_name", "use_rag"], as_index=False).agg(
        n=("query", "count"),
        avg_latency_seconds=("latency_seconds", "mean"),
        avg_relevance=("relevance", "mean"),
        avg_explanation_quality=("explanation_quality", "mean"),
        avg_personalization=("personalization", "mean"),
        avg_groundedness=("groundedness", "mean"),
    )

    # Round for cleaner presentation
    summary_df = summary_df.round(3)

    summary_df.to_csv(summary_path, index=False)

    print("\nSUMMARY")
    print(summary_df)

    print(f"\nSaved detailed results to: {detailed_path}")
    print(f"Saved manual grading sheet to: {manual_path}")
    print(f"Saved summary results to: {summary_path}")


def checkpoint_results(
    *,
    results: list[EvalCaseResult],
    output_dir: str,
    label: str = "checkpoint",
) -> None:
    """Save partial evaluation results so crashes do not lose progress."""
    if not results:
        return

    checkpoint_dir = Path(output_dir) / label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_eval_results(
        results=results,
        output_dir=str(checkpoint_dir),
    )
