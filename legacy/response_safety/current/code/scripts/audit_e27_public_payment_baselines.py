"""Fail-closed audit for E27 public payment baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORKLOADS = ("low", "medium", "high")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--suffix", choices=("pilot", "formal"), default="formal")
    args = parser.parse_args()
    frame = pd.read_csv(args.results / f"e27_{args.suffix}_seed_results.csv")
    summary = pd.read_csv(args.results / f"e27_{args.suffix}_summary.csv")
    metadata = json.loads(
        (args.results / f"e27_{args.suffix}_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    seeds = [1] if args.suffix == "pilot" else list(range(1, args.seeds + 1))
    expected = {(workload, seed) for workload in WORKLOADS for seed in seeds}
    actual = set(zip(frame["workload"], frame["seed"].astype(int)))
    finite = [
        "pasi_payment",
        "qim_payment",
        "csopt_payment",
        "pasi_pps",
        "qim_pps",
        "csopt_pps",
        "pasi_mean_quality",
        "qim_mean_quality",
        "csopt_mean_quality",
    ]
    checks = {
        "complete_matrix": actual == expected and len(frame) == len(expected),
        "unique_rows": not frame.duplicated(["workload", "seed"]).any(),
        "all_comparable_sets_nonempty": bool((frame["comparable_tasks"] > 0).all()),
        "candidate_competition_present": bool(
            (frame["candidate_bids"] >= 2 * frame["comparable_tasks"]).all()
        ),
        "task_partition_exact": bool(
            np.array_equal(
                frame["total_tasks"].to_numpy(),
                (
                    frame["comparable_tasks"]
                    + frame["excluded_lt_two_candidates"]
                    + frame["excluded_no_legal_candidate"]
                ).to_numpy(),
            )
        ),
        "qim_support_respected": bool(
            (frame["min_reserve_bid"] >= -1e-10).all()
            and (frame["max_reserve_bid"] <= 4.0 + 1e-10).all()
        ),
        "all_ir_checks_pass": bool(
            (frame[[
                "pasi_ir_violations",
                "qim_ir_violations",
                "csopt_ir_violations",
            ]] == 0).all().all()
        ),
        "all_qos_checks_pass": bool(
            (frame[[
                "pasi_qos_violations",
                "qim_qos_violations",
                "csopt_qos_violations",
            ]] == 0).all().all()
        ),
        "finite_metrics": bool(np.isfinite(frame[finite].to_numpy()).all()),
        "qim_csopt_winners_equivalent": bool(
            (frame["qim_csopt_equivalent_slots"] == frame["evaluated_slots"]).all()
        ),
        "qim_csopt_payments_equivalent": bool(
            (frame["qim_csopt_max_abs_payment_gap"] <= 1e-9).all()
        ),
        "summary_has_expected_workloads": set(summary["workload"]) == set(WORKLOADS),
        "metadata_discloses_direct_bids": "direct effective reserve bids"
        in metadata.get("information_difference", ""),
        "metadata_discloses_contract_difference": "critical/externality"
        in metadata.get("contract_difference", ""),
        "no_raw_trace_files": not any(
            path.name
            in {"providers.parquet", "tasks.parquet", "provider_static.parquet"}
            for path in args.results.rglob("*")
            if path.is_file()
        ),
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": len(frame),
        "expected_rows": len(expected),
        "pasi_lower_than_qim_total": int((frame["pasi_minus_qim_pps"] < 0).sum()),
        "pasi_lower_than_csopt_total": int((frame["pasi_minus_csopt_pps"] < 0).sum()),
    }
    output = args.results / f"e27_{args.suffix}_integrity_audit.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
