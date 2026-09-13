"""Fail-closed verifier for E19 probability-weighting sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT.parent / "results" / "e19_probability_weighting"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    seeds = pd.read_csv(RESULTS / "e19_formal_seed_results.csv")
    summary = pd.read_csv(RESULTS / "e19_formal_paired_summary.csv")
    metadata = json.loads(
        (RESULTS / "e19_formal_metadata.json").read_text(encoding="utf-8"))

    require(len(seeds) == 720, "720 formal runs")
    require(len(summary) == 24, "24 workload/zeta/method summaries")
    require(set(np.round(seeds["zeta"], 2)) == {0.65, 0.80, 1.00, 1.20},
            "three nonlinear zeta values plus identity")
    require(set(seeds["workload"]) == {"low", "medium", "high"},
            "all three workloads present")
    require(set(seeds["method"]) == {"MOI", "PASI"},
            "matched MOI and PASI methods present")
    require(seeds["run_status"].isin(["ok", "completed"]).all(),
            "all runs completed")
    require((seeds[["qos_violation_rate", "under_incentive_rate",
                    "target_miss_rate"]].to_numpy(float) == 0.0).all(),
            "all measured safety violation rates are zero")
    require(seeds.groupby(["workload", "zeta", "seed"])["event_tape_hash"]
            .nunique().eq(1).all(), "MOI and PASI share each frozen event tape")
    require(seeds.groupby(["workload", "seed"])["event_tape_hash"]
            .nunique().eq(1).all(), "zeta settings reuse the same event tape")
    require(summary["n_pairs"].eq(30).all(), "every summary contains 30 paired seeds")
    require(metadata["changed_factor"] == "provider zeta only",
            "single-factor intervention recorded")
    require(metadata["raw_or_derived_trace_redistributed"] is False,
            "restricted trace data are not redistributed")
    require(not any(p.suffix.lower() in {".parquet", ".plt"}
                    for p in RESULTS.rglob("*")),
            "result directory contains no raw trace files")
    print("E19 formal verification complete: all checks passed.")


if __name__ == "__main__":
    main()
