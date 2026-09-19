#!/usr/bin/env python3
"""Analyze the completed R72 canonical coverage-first Main-180 matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROUND = Path(__file__).resolve().parent
SCENARIOS = ("stationary", "burst", "dynamic")
METHODS = ("MOI", "PASI")


def dump(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_table(frame: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = frame.pivot(index=["scenario", "seed"], columns="method")
    out = pd.DataFrame(index=pivot.index).reset_index()
    for metric in (
        "design_reserve", "payment", "pps", "coverage",
        "qualified_coverage", "mean_quality", "assigned", "qualified",
    ):
        if metric in pivot.columns.get_level_values(0):
            out[f"{metric}_MOI"] = pivot[(metric, "MOI")].to_numpy()
            out[f"{metric}_PASI"] = pivot[(metric, "PASI")].to_numpy()
    out["payment_saving_percent"] = 100.0 * (out.payment_MOI - out.payment_PASI) / out.payment_MOI
    out["pps_saving_percent"] = 100.0 * (out.pps_MOI - out.pps_PASI) / out.pps_MOI
    if "design_reserve_MOI" in out:
        out["decision_fee_saving_percent"] = 100.0 * (
            out.design_reserve_MOI - out.design_reserve_PASI
        ) / out.design_reserve_MOI
        out["decision_fee_per_service_MOI"] = out.design_reserve_MOI / out.assigned_MOI
        out["decision_fee_per_service_PASI"] = out.design_reserve_PASI / out.assigned_PASI
        out["decision_fee_per_service_saving_percent"] = 100.0 * (
            out.decision_fee_per_service_MOI - out.decision_fee_per_service_PASI
        ) / out.decision_fee_per_service_MOI
    out["coverage_delta_pp"] = 100.0 * (out.coverage_PASI - out.coverage_MOI)
    if "qualified_coverage_MOI" in out:
        out["qualified_coverage_delta_pp"] = 100.0 * (
            out.qualified_coverage_PASI - out.qualified_coverage_MOI
        )
    if "mean_quality_MOI" in out:
        out["mean_quality_delta"] = out.mean_quality_PASI - out.mean_quality_MOI

    summaries = []
    metrics = [
        "payment_saving_percent",
        "pps_saving_percent",
        "coverage_delta_pp",
    ]
    if "decision_fee_saving_percent" in out:
        metrics = [
            "decision_fee_saving_percent",
            "decision_fee_per_service_saving_percent",
            *metrics,
        ]
    if "qualified_coverage_delta_pp" in out:
        metrics.append("qualified_coverage_delta_pp")
    if "mean_quality_delta" in out:
        metrics.append("mean_quality_delta")
    for scenario_index, scenario in enumerate(SCENARIOS):
        part = out[out.scenario == scenario]
        row: dict[str, object] = {"scenario": scenario, "n_pairs": int(len(part))}
        rng = np.random.default_rng(20260915 + scenario_index + (100 if prefix == "historical" else 0))
        for metric in metrics:
            values = part[metric].to_numpy(float)
            lo, hi = mean_ci(values, rng)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"ci95_low_{metric}"] = lo
            row[f"ci95_high_{metric}"] = hi
            row[f"std_{metric}"] = float(values.std(ddof=1))
            row[f"min_{metric}"] = float(values.min())
            row[f"max_{metric}"] = float(values.max())
            row[f"positive_{metric}_count"] = int((values > 0).sum())
        summaries.append(row)
    return out, pd.DataFrame(summaries)


def main() -> None:
    completion = json.loads((ROUND / "completion.json").read_text(encoding="utf-8"))
    if completion["completed"] != 180 or completion["errors"]:
        raise AssertionError("Do not summarize an incomplete Main-180 matrix")
    current = pd.read_csv(ROUND / "seed_results.csv")
    historical_raw = pd.read_csv(ROUND / "historical_run_metrics.csv")
    if len(current) != 180 or len(historical_raw) != 180:
        raise AssertionError("Expected exactly 180 rows in each protocol")

    historical = historical_raw.rename(
        columns={
            "mechanism": "method",
            "total_payment": "payment",
            "service_count": "assigned",
            "active_tasks": "tasks",
            "service_coverage": "coverage",
            "payment_per_service": "pps",
            "platform_utility_if_defined": "platform_utility",
        }
    ).copy()
    historical["qualified"] = historical["assigned"]
    historical["qualified_coverage"] = historical["coverage"]
    historical["mean_quality"] = np.nan

    merge_keys = ["scenario", "method", "seed"]
    comparison = current.merge(
        historical,
        on=merge_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_current", "_historical"),
        indicator=True,
    )
    if not (comparison._merge == "both").all():
        raise AssertionError("Current/historical cell keys do not match")
    comparison = comparison.drop(columns="_merge")
    for metric in ("payment", "assigned", "tasks", "coverage", "pps", "platform_utility"):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_current"] - comparison[f"{metric}_historical"]
    comparison["event_tape_hash_equal"] = (
        comparison.event_tape_hash_current == comparison.event_tape_hash_historical
    )
    comparison.to_csv(ROUND / "historical_current_comparison.csv", index=False)

    current_paired, current_summary = paired_table(current, "current")
    historical_paired, historical_summary = paired_table(historical, "historical")
    current_paired.to_csv(ROUND / "current_paired_seed_results.csv", index=False)
    current_summary.to_csv(ROUND / "current_scenario_summary.csv", index=False)
    historical_paired.to_csv(ROUND / "historical_paired_seed_results_recomputed.csv", index=False)
    historical_summary.to_csv(ROUND / "historical_scenario_summary_recomputed.csv", index=False)

    headline = current_summary.merge(
        historical_summary,
        on=["scenario", "n_pairs"],
        suffixes=("_current", "_historical"),
    )
    for metric in ("payment_saving_percent", "pps_saving_percent", "coverage_delta_pp"):
        headline[f"delta_mean_{metric}"] = (
            headline[f"mean_{metric}_current"] - headline[f"mean_{metric}_historical"]
        )
    headline.to_csv(ROUND / "headline_protocol_comparison.csv", index=False)

    negative_mask = (
        (current.qos_violations > 0)
        | (current.ir_violations > 0)
        | (current.target_violations > 0)
        | (current.reserve_over_budget_slots > 0)
        | (current.settlement_over_budget_slots > 0)
    )
    current.loc[negative_mask].to_csv(ROUND / "negative_outcomes.csv", index=False)

    cells = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((ROUND / "cells").glob("*.json"))]
    cell_frame = pd.DataFrame(
        {
            "scenario": row["scenario"],
            "method": row["method"],
            "seed": row["seed"],
            "tasks_fingerprint": row["input_fingerprints"]["tasks"],
            "providers_fingerprint": row["input_fingerprints"]["providers"],
            "provider_static_fingerprint": row["input_fingerprints"]["provider_static"],
            "event_tape_hash": row["event_tape_hash"],
            "candidate_graph_hash": row["candidate_graph_hash"],
            "selection_hash": row["selection_hash"],
        }
        for row in cells
    )
    paired_input_checks = []
    for (scenario, seed), group in cell_frame.groupby(["scenario", "seed"], sort=True):
        paired_input_checks.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "two_methods": len(group) == 2,
                "tasks_equal": group.tasks_fingerprint.nunique() == 1,
                "providers_equal": group.providers_fingerprint.nunique() == 1,
                "provider_static_equal": group.provider_static_fingerprint.nunique() == 1,
                "event_tape_equal": group.event_tape_hash.nunique() == 1,
            }
        )
    input_checks = pd.DataFrame(paired_input_checks)
    input_checks.to_csv(ROUND / "paired_input_checks.csv", index=False)

    domain_columns = [
        "scenario",
        "method",
        "seed",
        "candidate_rows",
        "legal_rows",
        "selected_rows",
        "legal_effort_below_a_min",
        "legal_effort_at_one",
        "legal_design_settlement_cost_diff",
        "max_legal_design_settlement_cost_diff",
        "zeta_above_one",
    ]
    current[domain_columns].to_csv(ROUND / "domain_activation_audit.csv", index=False)

    protocol_summary = comparison.groupby(["scenario", "method"], sort=True).agg(
        runs=("seed", "size"),
        exact_event_tapes=("event_tape_hash_equal", "sum"),
        exact_task_counts=("delta_tasks", lambda x: int((x == 0).sum())),
        exact_service_counts=("delta_assigned", lambda x: int((x == 0).sum())),
        mean_payment_delta=("delta_payment", "mean"),
        max_abs_payment_delta=("delta_payment", lambda x: float(np.abs(x).max())),
        mean_pps_delta=("delta_pps", "mean"),
        max_abs_coverage_delta=("delta_coverage", lambda x: float(np.abs(x).max())),
    ).reset_index()
    protocol_summary.to_csv(ROUND / "protocol_difference_summary.csv", index=False)

    audit = {
        "matrix_complete": len(current) == 180,
        "runtime_errors": len(completion["errors"]),
        "all_event_tapes_match_historical": bool(comparison.event_tape_hash_equal.all()),
        "historical_event_tapes_matched": int(comparison.event_tape_hash_equal.sum()),
        "all_paired_current_inputs_equal": bool(input_checks.drop(columns=["scenario", "seed"]).all().all()),
        "exact_service_count_cells": int((comparison.delta_assigned == 0).sum()),
        "changed_service_count_cells": int((comparison.delta_assigned != 0).sum()),
        "max_abs_service_count_delta": int(np.abs(comparison.delta_assigned).max()),
        "exact_payment_cells_at_1e_10": int((np.abs(comparison.delta_payment) <= 1e-10).sum()),
        "changed_payment_cells_at_1e_10": int((np.abs(comparison.delta_payment) > 1e-10).sum()),
        "max_abs_payment_delta": float(np.abs(comparison.delta_payment).max()),
        "qos_violations": int(current.qos_violations.sum()),
        "ir_violations": int(current.ir_violations.sum()),
        "target_violations": int(current.target_violations.sum()),
        "reserve_over_budget_slots": int(current.reserve_over_budget_slots.sum()),
        "settlement_over_budget_slots": int(current.settlement_over_budget_slots.sum()),
        "settlement_over_reserve_slots": int(current.settlement_over_reserve_slots.sum()),
        "negative_outcome_runs": int(negative_mask.sum()),
        "legal_effort_below_a_min": int(current.legal_effort_below_a_min.sum()),
        "legal_design_settlement_cost_diff": int(current.legal_design_settlement_cost_diff.sum()),
        "max_legal_design_settlement_cost_diff": float(current.max_legal_design_settlement_cost_diff.max()),
        "zeta_above_one": int(current.zeta_above_one.sum()),
        "all_current_pasi_payment_savings_positive": bool((current_paired.payment_saving_percent > 0).all()),
        "all_current_pasi_pps_savings_positive": bool((current_paired.pps_saving_percent > 0).all()),
        "all_current_pasi_decision_fee_savings_positive": bool((current_paired.decision_fee_saving_percent > 0).all()),
        "all_matching_objectives_declared_coverage_first": bool(
            all(row["matching_objective_declared"] == "coverage_first_payment_second" for row in cells)
        ),
        "all_matching_objective_types_observed_coverage_first": bool(
            all(row["observed_matching_objective_types"] == ["coverage_first_payment_second"] for row in cells)
        ),
        "coverage_first_matcher_calls": int(sum(row["coverage_first_matcher_calls"] for row in cells)),
        "coverage_first_solver_counts": {
            solver: int(sum(row["coverage_first_solver_counts"].get(solver, 0) for row in cells))
            for solver in sorted({
                solver
                for row in cells
                for solver in row["coverage_first_solver_counts"]
            })
        },
        "interpretation": {
            "historical_program_equivalence": False,
            "canonical_protocol_confirmation": True,
            "primary_optimization_metric": "design_reserve",
            "execution_settlement_metric": "payment",
            "reason": "The canonical protocol explicitly uses coverage-first matching and design-side decision cost; execution settlement is reported separately.",
        },
        "references_changed": False,
        "manuscript_changed": False,
        "hashes": {
            "seed_results.csv": sha256_file(ROUND / "seed_results.csv"),
            "historical_run_metrics.csv": sha256_file(ROUND / "historical_run_metrics.csv"),
            "runner": sha256_file(ROUND / "run_main180_coverage_first.py"),
            "analysis": sha256_file(Path(__file__)),
        },
    }
    dump(ROUND / "audit_summary.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
