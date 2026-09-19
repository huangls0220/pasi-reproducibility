#!/usr/bin/env python3
"""E1 confirmation under the R72 canonical protocol.

Runs the original 3 x 2 x 30 matrix with the repaired coverage-first core.
Only run/phase/slot aggregates are retained; candidate- and participant-level
logs remain transient and are not part of the public artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROUND = Path(__file__).resolve().parent
CODE = ROUND / "core_source"
OUT = ROUND / "e1_current180"
PHASES = {"Early": (0, 99), "Middle": (300, 499), "Late": (700, 999)}


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def cell_key(job: dict) -> str:
    return f"{job['scenario']}__seed_{job['seed']:03d}__{job['method']}"


def run_cell(job: dict) -> dict:
    sys.path.insert(0, str(CODE))
    from src.datasets.synthetic import generate_synthetic_episode
    import src.simulator as simmodule

    plan = load(ROUND / "preregistration.json")
    record = plan["config_records"][job["scenario"]]
    cfg = copy.deepcopy(record["config"])
    cfg.setdefault("simulation", {})["log_level"] = "selected"
    cfg["simulation"].setdefault("state_reset", {"enabled": False})
    if job.get("condition") == "reset":
        cfg["simulation"]["state_reset"] = {
            "enabled": True,
            "at_slot": 500,
            "fraction": 0.50,
            "observable_to_platform": True,
        }
    elif job.get("condition") == "normal":
        cfg["simulation"]["state_reset"] = {"enabled": False}
    cfg["dataset"] = {"pattern": job["scenario"]}
    if "N" in job:
        cfg["providers"]["N_mean"] = int(job["N"])
    if "M" in job:
        cfg["tasks"]["M_mean"] = int(job["M"])
    if cfg.get("matching", {}).get("objective") != "coverage_first_payment_second":
        raise AssertionError("E1 confirmation requires coverage-first matching")

    data = generate_synthetic_episode(cfg, seed=job["seed"], pattern=job["scenario"])
    raw_fingerprints = {
        name: frame_hash(value) for name, value in data.items() if isinstance(value, pd.DataFrame)
    }
    variant = job.get("variant", "FULL" if job["method"] == "PASI" else "MOI")
    if variant in {"NO_POSITIVE_ACCUMULATION", "NO_NEGATIVE_FEEDBACK", "FROZEN_STATE"}:
        data = {name: value.copy(deep=True) if isinstance(value, pd.DataFrame) else copy.deepcopy(value)
                for name, value in data.items()}
        if variant in {"NO_POSITIVE_ACCUMULATION", "FROZEN_STATE"}:
            data["provider_static"].loc[:, "xi"] = 0.0
        if variant in {"NO_NEGATIVE_FEEDBACK", "FROZEN_STATE"}:
            data["provider_static"].loc[:, "delta"] = 0.0
    effective_fingerprints = {
        name: frame_hash(value) for name, value in data.items() if isinstance(value, pd.DataFrame)
    }

    objective_types: set[str] = set()
    calls = 0
    matcher = simmodule.coverage_first_matching

    def audited_matcher(*args, **kwargs):
        nonlocal calls
        result = matcher(*args, **kwargs)
        calls += 1
        objective_types.add(result.get("objective_type"))
        return result

    simulator = simmodule.Simulator(
        cfg,
        data,
        method="SAMI" if job.get("sim_method", job["method"]) == "PASI" else job.get("sim_method", job["method"]),
        seed=job["seed"],
    )
    with patch.object(simmodule, "coverage_first_matching", side_effect=audited_matcher):
        result = simulator.run()

    if calls != int(result["summary"]["T"]):
        raise AssertionError("Coverage-first matcher was not called in every slot")
    if objective_types != {"coverage_first_payment_second"}:
        raise AssertionError(f"Unexpected objectives: {objective_types}")

    pairs = result["pair_log"]
    selected = pairs[pairs["selected"].astype(bool)].copy()
    providers = result["provider_log"]
    assigned_providers = providers[providers["assigned"].astype(bool)][
        ["slot", "provider_id", "H_before", "H_after"]
    ].copy()
    merged = selected.merge(assigned_providers, on=["slot", "provider_id"], how="left", validate="one_to_one")
    slots = result["slot_log"].copy()

    phase_rows = []
    for phase, (start, end) in PHASES.items():
        ps = slots[slots["slot"].between(start, end)]
        pm = merged[merged["slot"].between(start, end)]
        phase_rows.append(
            {
                **job,
                "phase": phase,
                "slot_start": start + 1,
                "slot_end": end + 1,
                "mean_selected_H": float(pm["H_after"].mean()),
                "median_selected_H": float(pm["H_after"].median()),
                "mean_selected_lambda": float(pm["lambda_required"].mean()),
                "design_reserve": float(ps["budget_used"].sum()),
                "execution_settlement": float(ps["total_payment"].sum()),
                "assigned": int(ps["num_assigned"].sum()),
                "tasks": int(ps["num_tasks"].sum()),
                "coverage": float(ps["num_assigned"].sum() / max(ps["num_tasks"].sum(), 1)),
            }
        )

    corr = float(merged[["H_after", "lambda_required"]].corr(method="spearman").iloc[0, 1])
    early = merged[merged["slot"].between(0, 99)].groupby("provider_id")["H_after"].mean()
    late = merged[merged["slot"].between(700, 999)].groupby("provider_id")["H_after"].mean()
    shared = early.index.intersection(late.index)
    same_delta = late.loc[shared] - early.loc[shared]

    slot_rows = slots[["slot", "num_tasks", "num_assigned", "budget_used", "total_payment"]].copy()
    slot_rows.insert(0, "seed", job["seed"])
    slot_rows.insert(0, "method", job["method"])
    slot_rows.insert(0, "scenario", job["scenario"])
    return {
        "job": job,
        "phase_rows": phase_rows,
        "slot_rows": slot_rows.to_dict(orient="records"),
        "run_row": {
            **job,
            "event_fingerprints": raw_fingerprints,
            "effective_event_fingerprints": effective_fingerprints,
            "selected_rows": int(len(merged)),
            "spearman_H_lambda": corr,
            "same_provider_n": int(len(shared)),
            "same_provider_mean_early_late_delta_H": float(same_delta.mean()),
            "same_provider_nonpositive_count": int((same_delta <= 0).sum()),
            "coverage_first_calls": calls,
            "objective_types": sorted(objective_types),
            "ir_violations": int(result["summary"]["ir_violations"]),
            "target_violations": int(result["summary"]["target_violations"]),
        },
    }


def analyse(results: list[dict]) -> None:
    phase = pd.DataFrame([row for result in results for row in result["phase_rows"]])
    slots = pd.DataFrame([row for result in results for row in result["slot_rows"]])
    runs = pd.DataFrame([result["run_row"] for result in results])
    OUT.mkdir(parents=True, exist_ok=True)
    phase.to_csv(OUT / "phase_by_seed.csv", index=False)
    slots.to_parquet(OUT / "slot_aggregates.parquet", index=False)
    runs.to_json(OUT / "run_audit.jsonl", orient="records", lines=True)

    paired_phase = phase.pivot(
        index=["scenario", "seed", "phase"], columns="method",
        values=["design_reserve", "execution_settlement", "assigned", "tasks", "coverage", "mean_selected_H", "mean_selected_lambda"],
    )
    paired_phase.columns = [f"{metric}_{method}" for metric, method in paired_phase.columns]
    paired_phase = paired_phase.reset_index()
    for basis in ("design_reserve", "execution_settlement"):
        paired_phase[f"{basis}_saving_pct"] = 100.0 * (
            1.0 - paired_phase[f"{basis}_PASI"] / paired_phase[f"{basis}_MOI"]
        )
    paired_phase["coverage_diff_pp"] = 100.0 * (
        paired_phase["coverage_PASI"] - paired_phase["coverage_MOI"]
    )
    paired_phase.to_csv(OUT / "paired_phase_by_seed.csv", index=False)

    paired_slots = slots.pivot(
        index=["scenario", "seed", "slot"], columns="method",
        values=["num_tasks", "num_assigned", "budget_used", "total_payment"],
    )
    paired_slots.columns = [f"{metric}_{method}" for metric, method in paired_slots.columns]
    paired_slots = paired_slots.reset_index().sort_values(["scenario", "seed", "slot"])
    timing = []
    for (scenario, seed), group in paired_slots.groupby(["scenario", "seed"], sort=True):
        cumulative_moi = group["budget_used_MOI"].cumsum()
        cumulative_pasi = group["budget_used_PASI"].cumsum()
        cumulative_assigned_moi = group["num_assigned_MOI"].cumsum()
        cumulative_assigned_pasi = group["num_assigned_PASI"].cumsum()
        pps_moi = cumulative_moi / cumulative_assigned_moi.clip(lower=1)
        pps_pasi = cumulative_pasi / cumulative_assigned_pasi.clip(lower=1)
        saving_rate = 100.0 * (1.0 - pps_pasi / pps_moi)
        final = float(saving_rate.iloc[-1])
        record = {"scenario": scenario, "seed": int(seed), "final_design_pps_saving_pct": final}
        for fraction in (0.5, 0.9):
            reached = group.loc[saving_rate >= fraction * final, "slot"] if final > 0 else pd.Series(dtype=float)
            record[f"T{int(fraction * 100)}"] = int(reached.iloc[0] + 1) if len(reached) else None
        timing.append(record)
    timing_df = pd.DataFrame(timing)
    timing_df.to_csv(OUT / "formation_timing_by_seed.csv", index=False)

    phase_summary = paired_phase.groupby(["scenario", "phase"], as_index=False).agg(
        mean_H=("mean_selected_H_PASI", "mean"),
        mean_lambda=("mean_selected_lambda_PASI", "mean"),
        design_saving_pct=("design_reserve_saving_pct", "mean"),
        settlement_saving_pct=("execution_settlement_saving_pct", "mean"),
        coverage_diff_pp=("coverage_diff_pp", "mean"),
    )
    phase_summary.to_csv(OUT / "phase_summary.csv", index=False)
    timing_summary = timing_df.groupby("scenario", as_index=False).agg(
        mean_T50=("T50", "mean"), mean_T90=("T90", "mean"),
        min_T90=("T90", "min"), max_T90=("T90", "max"),
    )
    timing_summary.to_csv(OUT / "timing_summary.csv", index=False)
    audit = {
        "formal_runs": len(runs),
        "expected_runs": 180,
        "all_coverage_first": bool((runs["coverage_first_calls"] == 1000).all()),
        "ir_violations": int(runs["ir_violations"].sum()),
        "target_violations": int(runs["target_violations"].sum()),
        "same_provider_nonpositive_total": int(runs["same_provider_nonpositive_count"].sum()),
        "reference_changes": False,
        "restricted_inputs_published": False,
    }
    dump(OUT / "audit_summary.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    jobs = [
        {"scenario": scenario, "method": method, "seed": seed}
        for scenario in ("stationary", "burst", "dynamic")
        for seed in range(1, 31)
        for method in ("MOI", "PASI")
    ]
    if args.smoke:
        jobs = [job for job in jobs if job["scenario"] == "stationary" and job["seed"] == 1]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_cell, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"[{index:03d}/{len(jobs):03d}] {cell_key(result['job'])}", flush=True)
    analyse(results)


if __name__ == "__main__":
    main()
