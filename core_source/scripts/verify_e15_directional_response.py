"""Fail-closed verifier for the formal E15 directional-response results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT.parent / "results" / "e15_directional_response"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    calibration = pd.read_csv(RESULTS / "e15_formal_calibration.csv")
    seeds = pd.read_csv(RESULTS / "e15_formal_seed_results.csv")
    summary = pd.read_csv(RESULTS / "e15_formal_paired_summary.csv")
    metadata = json.loads(
        (RESULTS / "e15_formal_metadata.json").read_text(encoding="utf-8"))

    require(len(calibration) == 300, "300 calibration-only scenario/seed rows")
    require(len(seeds) == 1200, "1200 test runs (4 profiles x 10 scenarios x 30 seeds)")
    require(len(summary) == 40, "40 paired profile/scenario summaries")
    require(metadata["temporal_split"] == "first 20% calibration; remaining 80% test",
            "pre-specified disjoint temporal split recorded")
    require(metadata["test_side_true_parameter_used_for_decision"] is False,
            "test-side true response is not used for decisions")
    require(metadata["test_side_ratio_used_after_execution_for_audit_only"] is True,
            "test-side ratio is explicitly audit-only")
    require(metadata["raw_or_derived_trace_redistributed"] is False,
            "no raw or derived GeoLife trace is redistributed")

    require(seeds["run_status"].isin(["ok", "completed"]).all(),
            "all test runs completed")
    require(seeds["interval_contains_test_ratio"].astype(bool).all(),
            "all frozen calibration intervals contain the held-out audit ratio")
    require((seeds[["qos_violation_rate", "target_miss_rate",
                    "under_incentive_rate"]].to_numpy(float) == 0.0).all(),
            "all measured safety violation rates are zero")

    directional = summary[
        (summary["envelope"] == "directional")
        & np.isclose(summary["reserve_multiplier"], 1.0)]
    require(len(directional) == 10, "directional x1 profile contains all ten scenarios")
    require(directional["gate_pass"].astype(bool).all(),
            "every directional x1 scenario passes the pre-specified gate")
    require(np.allclose(directional["delta_service_coverage"], 0.0,
                        atol=1e-12, rtol=0.0),
            "directional x1 preserves paired service coverage in every scenario")
    require((directional["delta_service_coverage_ci_low"] >= -0.01).all(),
            "all directional x1 coverage lower bounds exceed the -1 pp gate")

    corrected = seeds[seeds["envelope"] == "directional"]
    require((np.abs(corrected["corrected_response_bias"].to_numpy(float)) < 1e-8).all(),
            "calibration-only correction removes systematic design bias numerically")
    by_seed = seeds.groupby("seed")["event_tape_hash"].nunique()
    require((by_seed == 1).all(), "all methods and scenarios share one frozen tape per seed")
    require(not any(p.suffix.lower() in {".parquet", ".plt", ".csv.gz"}
                    for p in RESULTS.rglob("*")),
            "formal result directory contains no raw trace files")

    print("E15 formal verification complete: all checks passed.")


if __name__ == "__main__":
    main()
