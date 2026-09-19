#!/usr/bin/env python3
"""Confirm the original four-scale, ten-seed experiment under R72."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from run_e1_current180 import ROUND, run_cell


OUT = ROUND / "scale_current80"
SCALES = ((25, 20), (50, 40), (100, 80), (200, 160))


def summarize(result: dict) -> dict:
    job = result["job"]
    slots = pd.DataFrame(result["slot_rows"])
    assigned = int(slots["num_assigned"].sum())
    tasks = int(slots["num_tasks"].sum())
    return {
        **job,
        "assigned": assigned,
        "tasks": tasks,
        "coverage": assigned / max(tasks, 1),
        "design_reserve": float(slots["budget_used"].sum()),
        "execution_settlement": float(slots["total_payment"].sum()),
        "objective_types": result["run_row"]["objective_types"],
        "coverage_first_calls": result["run_row"]["coverage_first_calls"],
        "ir_violations": result["run_row"]["ir_violations"],
        "target_violations": result["run_row"]["target_violations"],
        "event_fingerprints": result["run_row"]["event_fingerprints"],
    }


def analyse(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["N", "seed", "method"])
    frame.to_json(OUT / "seed_results.jsonl", orient="records", lines=True)
    paired = frame.pivot(
        index=["N", "M", "seed"], columns="method",
        values=["assigned", "tasks", "coverage", "design_reserve", "execution_settlement"],
    )
    paired.columns = [f"{metric}_{method}" for metric, method in paired.columns]
    paired = paired.reset_index()
    for basis in ("design_reserve", "execution_settlement"):
        paired[f"{basis}_saving_pct"] = 100.0 * (1.0 - paired[f"{basis}_PASI"] / paired[f"{basis}_MOI"])
    paired["coverage_diff_pp"] = 100.0 * (paired["coverage_PASI"] - paired["coverage_MOI"])
    paired.to_csv(OUT / "paired_seed_results.csv", index=False)
    summary = paired.groupby(["N", "M"], as_index=False).agg(
        n_seeds=("seed", "count"),
        design_saving_pct=("design_reserve_saving_pct", "mean"),
        settlement_saving_pct=("execution_settlement_saving_pct", "mean"),
        coverage_diff_pp=("coverage_diff_pp", "mean"),
        min_design_saving_pct=("design_reserve_saving_pct", "min"),
        max_abs_coverage_diff_pp=("coverage_diff_pp", lambda x: x.abs().max()),
    )
    summary.to_csv(OUT / "scale_summary.csv", index=False)
    audit = {
        "formal_runs": len(frame),
        "expected_runs": 80,
        "all_coverage_first": bool((frame["coverage_first_calls"] == 1000).all()),
        "ir_violations": int(frame["ir_violations"].sum()),
        "target_violations": int(frame["target_violations"].sum()),
        "reference_changes": False,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    jobs = [
        {"scenario": "stationary", "method": method, "seed": seed, "N": N, "M": M}
        for N, M in SCALES for seed in range(1, 11) for method in ("MOI", "PASI")
    ]
    if args.smoke:
        jobs = [job for job in jobs if job["seed"] == 1]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            row = summarize(future.result())
            rows.append(row)
            print(f"[{index:02d}/{len(jobs):02d}] {row['N']}x{row['M']} seed={row['seed']} {row['method']}", flush=True)
    analyse(rows)


if __name__ == "__main__":
    main()
