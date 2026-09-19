#!/usr/bin/env python3
"""Confirm the 40-run state-reset experiment under the R72 protocol."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from run_e1_current180 import ROUND, run_cell


OUT = ROUND / "state_reset_current40"


def summarize(result: dict) -> dict:
    job = result["job"]
    slots = pd.DataFrame(result["slot_rows"])
    assigned = int(slots["num_assigned"].sum())
    tasks = int(slots["num_tasks"].sum())
    audit = result["run_row"]
    return {
        **job,
        "assigned": assigned,
        "tasks": tasks,
        "coverage": assigned / max(tasks, 1),
        "design_reserve": float(slots["budget_used"].sum()),
        "execution_settlement": float(slots["total_payment"].sum()),
        "design_pps": float(slots["budget_used"].sum()) / max(assigned, 1),
        "settlement_pps": float(slots["total_payment"].sum()) / max(assigned, 1),
        "coverage_first_calls": audit["coverage_first_calls"],
        "ir_violations": audit["ir_violations"],
        "target_violations": audit["target_violations"],
        "event_fingerprints": json.dumps(audit["event_fingerprints"], sort_keys=True),
    }


def analyse(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["seed", "method", "condition"])
    frame.to_csv(OUT / "seed_results.csv", index=False)
    paired = frame.pivot(
        index=["seed", "method"], columns="condition",
        values=["assigned", "tasks", "coverage", "design_reserve", "execution_settlement", "design_pps", "settlement_pps"],
    )
    paired.columns = [f"{metric}_{condition}" for metric, condition in paired.columns]
    paired = paired.reset_index()
    paired["coverage_loss_pp"] = 100.0 * (paired["coverage_normal"] - paired["coverage_reset"])
    paired["design_pps_change_pct"] = 100.0 * (paired["design_pps_reset"] / paired["design_pps_normal"] - 1.0)
    paired["settlement_pps_change_pct"] = 100.0 * (paired["settlement_pps_reset"] / paired["settlement_pps_normal"] - 1.0)
    paired.to_csv(OUT / "paired_reset_by_seed.csv", index=False)
    summary = paired.groupby("method", as_index=False).agg(
        n=("seed", "count"),
        coverage_normal=("coverage_normal", "mean"),
        coverage_reset=("coverage_reset", "mean"),
        coverage_loss_pp=("coverage_loss_pp", "mean"),
        design_pps_normal=("design_pps_normal", "mean"),
        design_pps_reset=("design_pps_reset", "mean"),
        design_pps_change_pct=("design_pps_change_pct", "mean"),
        settlement_pps_change_pct=("settlement_pps_change_pct", "mean"),
    )
    summary.to_csv(OUT / "reset_summary.csv", index=False)
    audit = {
        "formal_runs": len(frame),
        "expected_runs": 40,
        "all_coverage_first": bool((frame["coverage_first_calls"] == 1000).all()),
        "ir_violations": int(frame["ir_violations"].sum()),
        "target_violations": int(frame["target_violations"].sum()),
        "normal_reset_raw_tapes_equal": bool(
            frame.groupby(["seed", "method"])["event_fingerprints"].nunique().eq(1).all()
        ),
        "observable_to_platform": True,
        "reference_changes": False,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = [601] if args.smoke else list(range(601, 611))
    jobs = [
        {"scenario": "stationary", "method": method, "seed": seed, "condition": condition}
        for seed in seeds for method in ("MOI", "PASI") for condition in ("normal", "reset")
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            row = summarize(future.result())
            rows.append(row)
            print(f"[{index:02d}/{len(jobs):02d}] seed={row['seed']} {row['method']} {row['condition']}", flush=True)
    analyse(rows)


if __name__ == "__main__":
    main()
