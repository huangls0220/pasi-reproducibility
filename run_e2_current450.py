#!/usr/bin/env python3
"""Confirm E2 component ablations under the R72 canonical protocol."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from run_e1_current180 import ROUND, run_cell


OUT = ROUND / "e2_current450"
VARIANTS = (
    "MOI",
    "FULL_PASI",
    "NO_POSITIVE_ACCUMULATION",
    "NO_NEGATIVE_FEEDBACK",
    "FROZEN_STATE",
)


def summarize(result: dict) -> dict:
    job = result["job"]
    slots = pd.DataFrame(result["slot_rows"])
    assigned = int(slots["num_assigned"].sum())
    tasks = int(slots["num_tasks"].sum())
    audit = result["run_row"]
    return {
        "scenario": job["scenario"],
        "seed": job["seed"],
        "variant": job["variant"],
        "assigned": assigned,
        "tasks": tasks,
        "coverage": assigned / max(tasks, 1),
        "design_reserve": float(slots["budget_used"].sum()),
        "execution_settlement": float(slots["total_payment"].sum()),
        "design_pps": float(slots["budget_used"].sum()) / max(assigned, 1),
        "settlement_pps": float(slots["total_payment"].sum()) / max(assigned, 1),
        "mean_H": float(pd.DataFrame(result["phase_rows"])["mean_selected_H"].mean()),
        "raw_event_fingerprints": json.dumps(audit["event_fingerprints"], sort_keys=True),
        "effective_event_fingerprints": json.dumps(audit["effective_event_fingerprints"], sort_keys=True),
        "coverage_first_calls": audit["coverage_first_calls"],
        "objective_types": audit["objective_types"],
        "ir_violations": audit["ir_violations"],
        "target_violations": audit["target_violations"],
    }


def analyse(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["scenario", "seed", "variant"])
    frame.to_json(OUT / "seed_results.jsonl", orient="records", lines=True)
    numeric = frame.drop(columns=["raw_event_fingerprints", "effective_event_fingerprints", "objective_types"])
    numeric.to_csv(OUT / "seed_results.csv", index=False)
    paired = numeric.pivot(
        index=["scenario", "seed"], columns="variant",
        values=["design_reserve", "execution_settlement", "design_pps", "settlement_pps", "coverage", "mean_H"],
    )
    paired.columns = [f"{metric}__{variant}" for metric, variant in paired.columns]
    paired = paired.reset_index()
    for variant in VARIANTS:
        if variant in {"MOI", "FULL_PASI"}:
            continue
        for metric in ("design_reserve", "execution_settlement", "design_pps", "settlement_pps"):
            paired[f"{metric}_degradation_pct__{variant}"] = 100.0 * (
                paired[f"{metric}__{variant}"] / paired[f"{metric}__FULL_PASI"] - 1.0
            )
        paired[f"coverage_diff_pp__{variant}"] = 100.0 * (
            paired[f"coverage__{variant}"] - paired["coverage__FULL_PASI"]
        )
    paired.to_csv(OUT / "paired_ablation_by_seed.csv", index=False)

    value_cols = [column for column in paired.columns if "degradation_pct" in column or "coverage_diff_pp" in column]
    summary = paired.groupby("scenario")[value_cols].mean().reset_index()
    summary.to_csv(OUT / "ablation_summary.csv", index=False)
    audit = {
        "formal_runs": len(frame),
        "expected_runs": 450,
        "all_coverage_first": bool((frame["coverage_first_calls"] == 1000).all()),
        "ir_violations": int(frame["ir_violations"].sum()),
        "target_violations": int(frame["target_violations"].sum()),
        "raw_tape_shared_within_scenario_seed": bool(
            frame.groupby(["scenario", "seed"])["raw_event_fingerprints"].nunique().eq(1).all()
        ),
        "reference_changes": False,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    jobs = []
    for scenario in ("stationary", "burst", "dynamic"):
        for seed in range(1, 31):
            for variant in VARIANTS:
                jobs.append({
                    "scenario": scenario,
                    "seed": seed,
                    "variant": variant,
                    "method": "MOI" if variant == "MOI" else "PASI",
                    "sim_method": "MOI" if variant == "MOI" else "PASI",
                })
    if args.smoke:
        jobs = [job for job in jobs if job["seed"] == 1]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            row = summarize(future.result())
            rows.append(row)
            print(f"[{index:03d}/{len(jobs):03d}] {row['scenario']} seed={row['seed']} {row['variant']}", flush=True)
    analyse(rows)


if __name__ == "__main__":
    main()
