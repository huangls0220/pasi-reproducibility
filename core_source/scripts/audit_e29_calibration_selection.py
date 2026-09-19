"""Audit E29 split integrity, frozen selection, and reported outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_e29_calibration_selection import (
    ALPHA,
    CALIBRATION_SEEDS,
    GUARDRAIL_WEIGHTS,
    SCENARIOS,
    TEST_SEEDS,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results

    calibration_path = root / "e29_calibration_seed_results.csv"
    bounds_path = root / "e29_calibration_cell_bounds.csv"
    selection_path = root / "e29_selection_frozen_before_test.json"
    test_path = root / "e29_test_seed_results_all_candidates.csv"
    summary_path = root / "e29_test_summary_all_candidates.csv"
    selected_path = root / "e29_test_summary_selected_policy.csv"
    metadata_path = root / "e29_metadata.json"

    calibration = pd.read_csv(calibration_path)
    bounds = pd.read_csv(bounds_path)
    test = pd.read_csv(test_path)
    summary = pd.read_csv(summary_path)
    selected_summary = pd.read_csv(selected_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected_cells = len(SCENARIOS) * len(GUARDRAIL_WEIGHTS)
    checks = {
        "calibration_rows_450": len(calibration) == 450,
        "test_rows_450": len(test) == 450,
        "bounds_cells_30": len(bounds) == expected_cells,
        "summary_cells_30": len(summary) == expected_cells,
        "calibration_odd_seeds_only": set(calibration.seed) == set(CALIBRATION_SEEDS),
        "test_even_seeds_only": set(test.seed) == set(TEST_SEEDS),
        "splits_disjoint": set(calibration.seed).isdisjoint(set(test.seed)),
        "all_candidates_present": (
            set(calibration.global_guardrail_weight) == set(GUARDRAIL_WEIGHTS)
            and set(test.global_guardrail_weight) == set(GUARDRAIL_WEIGHTS)),
        "all_scenarios_present": (
            set(calibration.scenario) == set(SCENARIOS)
            and set(test.scenario) == set(SCENARIOS)),
        "all_runs_ok": bool(calibration.status.isin(["ok", "completed"]).all()
                            and test.status.isin(["ok", "completed"]).all()),
        "all_within_global_envelope": bool(
            calibration.within_global_envelope.all()
            and test.within_global_envelope.all()),
        "no_test_side_lookahead": bool(
            not calibration.test_side_lookahead.any()
            and not test.test_side_lookahead.any()),
        "selection_did_not_use_test": bool(
            not selection["test_results_used_for_selection"]
            and not metadata["test_results_used_for_selection"]),
        "selection_written_before_test": bool(
            selection["selection_written_before_test_launch"]),
        "calibration_hash_matches_selection": (
            selection["calibration_results_sha256"] == digest(calibration_path)),
        "bounds_hash_matches_selection": (
            selection["calibration_bounds_sha256"] == digest(bounds_path)),
        "selection_hash_matches_metadata": (
            metadata["selection_file_sha256"] == digest(selection_path)),
        "test_hash_matches_metadata": (
            metadata["test_results_sha256"] == digest(test_path)),
        "summary_hash_matches_metadata": (
            metadata["test_summary_sha256"] == digest(summary_path)),
    }

    worst = bounds.groupby("global_guardrail_weight")[
        "simultaneous_cp_upper"].max().to_dict()
    eligible = [float(g) for g in GUARDRAIL_WEIGHTS
                if all(worst[float(h)] <= ALPHA
                       for h in GUARDRAIL_WEIGHTS if h >= g)]
    recomputed = min(eligible) if eligible else 1.0
    reported = float(selection["selected_global_guardrail_weight"])
    checks["selection_recomputes"] = bool(np.isclose(recomputed, reported))
    checks["selected_summary_only_selected_g"] = bool(
        set(selected_summary.global_guardrail_weight) == {reported}
        and len(selected_summary) == len(SCENARIOS))

    selected_test = summary[
        np.isclose(summary.global_guardrail_weight, reported)]
    fixed_test = summary[np.isclose(summary.global_guardrail_weight, 1.0)]
    merged = selected_test.merge(
        fixed_test[["scenario", "mean_service_coverage"]], on="scenario",
        suffixes=("_selected", "_fixed"))
    merged["coverage_gain_pp"] = 100.0 * (
        merged.mean_service_coverage_selected
        - merged.mean_service_coverage_fixed)
    audit = {
        "experiment": "E29 audit",
        "checks": checks,
        "passed": bool(all(checks.values())),
        "selected_g": reported,
        "calibration_worst_simultaneous_upper": float(worst[reported]),
        "alpha": ALPHA,
        "selected_test_worst_rate": float(
            selected_test.qos_violation_rate.max()),
        "selected_test_worst_cp_upper95": float(
            selected_test.qos_violation_cp_upper95.max()),
        "selected_test_total_violations": int(
            selected_test.qos_violations_total.sum()),
        "selected_test_total_assigned": int(selected_test.assigned_total.sum()),
        "coverage_gain_over_fixed_pp_by_scenario": {
            row.scenario: float(row.coverage_gain_pp)
            for row in merged.itertuples()},
        "negative_results_preserved": {
            row.scenario: {
                "violations": int(row.qos_violations_total),
                "assigned": int(row.assigned_total),
                "rate": float(row.qos_violation_rate),
                "cp_upper95": float(row.qos_violation_cp_upper95),
            }
            for row in selected_test.itertuples()
            if row.qos_violations_total > 0
        },
    }
    output = root / "e29_audit.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
