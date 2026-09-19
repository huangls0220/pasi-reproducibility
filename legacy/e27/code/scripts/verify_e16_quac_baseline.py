"""Fail-closed verifier for the formal E16 QUAC-F baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT.parent / "results" / "e16_quac_public_baseline"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    seeds = pd.read_csv(RESULTS / "e16_formal_seed_results.csv")
    summary = pd.read_csv(RESULTS / "e16_formal_paired_summary.csv")
    metadata = json.loads(
        (RESULTS / "e16_formal_metadata.json").read_text(encoding="utf-8"))

    require(len(seeds) == 270, "270 runs (3 workloads x 3 methods x 30 seeds)")
    require(len(summary) == 9, "nine workload/method paired summaries")
    require(set(seeds["method"]) == {"PASI", "MOI", "QUAC-F"},
            "PASI, MOI, and public QUAC-F mapping are all present")
    require(set(seeds["workload"]) == {"low", "medium", "high"},
            "low, medium, and high workloads are present")
    require((seeds["run_status"] == "completed").all(), "all runs completed")
    require((seeds[["qos_violation_rate", "under_incentive_rate",
                    "target_miss_rate"]].to_numpy(float) == 0.0).all(),
            "all measured safety violation rates are zero")
    require((seeds["payment"] > 0.0).all(), "all realised total payments are positive")
    require((seeds.groupby(["workload", "seed"])["event_tape_hash"].nunique() == 1).all(),
            "all methods share one frozen event tape per workload and seed")

    quac = summary[summary["method"] == "QUAC-F"].set_index("workload")
    require(len(quac) == 3, "QUAC-F has 30-pair summaries for all three workloads")
    require(np.allclose(quac["delta_service_coverage"], 0.0,
                        atol=1e-12, rtol=0.0),
            "QUAC-F and PASI have identical paired service coverage at all loads")
    require((quac["delta_pps_ci_high"] < 0.0).all(),
            "QUAC-F PPS is significantly below PASI at all loads")
    require((seeds.loc[seeds["method"] == "QUAC-F",
                       "negative_intercept_rate"] == 1.0).all(),
            "faithful QUAC-F uses signed IR-binding intercepts in every run")

    require(metadata["source_doi"] == "10.1109/MASS.2017.45",
            "public source DOI is recorded")
    require("not author code" in metadata["implementation_status"],
            "independent-implementation status is disclosed")
    require("two-stage binary MILP" in metadata["matching_solver"],
            "high-load exact coverage-first MILP is recorded")
    require(metadata["raw_or_derived_trace_redistributed"] is False,
            "no raw or derived GeoLife trace is redistributed")
    require(not any(p.suffix.lower() in {".parquet", ".plt", ".csv.gz"}
                    for p in RESULTS.rglob("*")),
            "formal result directory contains no raw trace files")

    print("E16 formal verification complete: all checks passed.")


if __name__ == "__main__":
    main()
