"""Metrics computation (Section 10).

All metrics are recomputed from raw logs — never stored as simulator
internal averages.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


_EPS = 1e-12


def high_quality_ratio(df: pd.DataFrame, q_threshold: float = 0.8) -> float:
    """HQR = |{assigned with quality >= threshold}| / |assigned|."""
    assigned = df[df["selected"] == True]  # noqa: E712
    if len(assigned) == 0:
        return float("nan")
    hq = (assigned["execution_quality"] >= q_threshold).sum()
    return hq / len(assigned)


def average_quality(df: pd.DataFrame) -> float:
    assigned = df[df["selected"] == True]  # noqa: E712
    if len(assigned) == 0:
        return float("nan")
    return float(assigned["execution_quality"].mean())


def completion_ratio(df: pd.DataFrame, total_tasks: int) -> float:
    if total_tasks == 0:
        return float("nan")
    return len(df[df["selected"] == True]) / total_tasks  # noqa: E712


def payment_per_assigned(df: pd.DataFrame) -> float:
    assigned = df[df["selected"] == True]  # noqa: E712
    if len(assigned) == 0:
        return float("nan")
    return float(assigned["expected_contract_cost"].sum() / len(assigned))


def payment_per_hq(df: pd.DataFrame, q_threshold: float = 0.8) -> float:
    assigned = df[df["selected"] == True]  # noqa: E712
    hq = assigned[assigned["execution_quality"] >= q_threshold]
    if len(hq) == 0:
        return float("nan")
    return float(hq["expected_contract_cost"].sum() / len(hq))


def platform_utility(df: pd.DataFrame) -> float:
    assigned = df[df["selected"] == True]  # noqa: E712
    return float(assigned["immediate_value"].sum())


def ir_violation_count(df: pd.DataFrame, U_out_col: str = "experienced_utility") -> int:
    """Count pairs where experienced utility < 0 (IR violation).

    Note: U_out should be subtracted beforehand or compared post-hoc.
    """
    return int((df["experienced_utility"] < -1e-8).sum())


def jain_fairness(values: np.ndarray) -> float:
    """Jain's fairness index: (sum x)^2 / (N * sum x^2)."""
    if len(values) == 0:
        return float("nan")
    s = values.sum()
    s2 = (values * values).sum()
    if s2 < _EPS:
        return float("nan")
    return float(s * s / (len(values) * s2))


def budget_utilization(df: pd.DataFrame, budgets: dict[int, float]) -> dict:
    """Compute budget utilization per slot."""
    utilization = {}
    for slot in df["slot"].unique():
        slot_cost = df[(df["slot"] == slot) & (df["selected"] == True)]["expected_contract_cost"].sum()  # noqa: E712
        budget = budgets.get(int(slot), float("nan"))
        utilization[int(slot)] = slot_cost / max(budget, _EPS)
    return utilization


def compute_summary(
    pair_log: pd.DataFrame,
    slot_log: pd.DataFrame,
    provider_log: pd.DataFrame,
) -> dict:
    """Compute comprehensive run summary from logs.

    Returns a dict suitable for JSON serialization.
    """
    assigned = pair_log[pair_log["selected"] == True]  # noqa: E712

    summary = {
        "total_slots": int(slot_log["slot"].max() + 1),
        "total_tasks": int(slot_log["num_tasks"].sum()) if "num_tasks" in slot_log.columns else 0,
        "total_assigned": len(assigned),
        "HQR": high_quality_ratio(pair_log),
        "average_quality": average_quality(pair_log),
        "completion_ratio": completion_ratio(pair_log, len(pair_log)),
        "cumulative_payment": float(assigned["expected_contract_cost"].sum()),
        "cost_per_assigned": payment_per_assigned(pair_log),
        "cost_per_HQ": payment_per_hq(pair_log),
        "platform_utility": platform_utility(pair_log),
        "provider_utility_mean": float(assigned["experienced_utility"].mean()) if len(assigned) > 0 else float("nan"),
    }

    # Stage ratios
    if "stage_before" in provider_log.columns:
        total_prov_slots = len(provider_log)
        if total_prov_slots > 0:
            summary["cultivation_ratio"] = float(
                (provider_log["stage_before"] == "cultivation").sum() / total_prov_slots
            )
            summary["maintenance_ratio"] = float(
                (provider_log["stage_before"] == "maintenance").sum() / total_prov_slots
            )

    return summary
