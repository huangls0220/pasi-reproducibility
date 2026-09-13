"""E17: post-calibration response drift without test-side response observation.

The original task response rate is frozen as the design-side value.  The true
response rate then follows a predeclared slot schedule in the held-out test
region.  Contract construction receives only either the frozen point estimate
or a fixed [0.70, 1.30] ratio envelope; realised ratios are retained solely for
the after-execution audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_e8_robustness import bootstrap_ci, configure, tape_hash
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator

BOUND_LOWER = 0.70
BOUND_UPPER = 1.30
PROFILES = ("frozen_point", "fixed_envelope")
SCENARIOS = (
    "stable", "down_step", "up_step", "down_ramp", "alternating",
    "hidden_blocks", "outside_low", "outside_high",
)


def slot_ratios(slots: np.ndarray, scenario: str, seed: int) -> np.ndarray:
    unique = np.sort(np.unique(slots))
    position = {int(slot): i for i, slot in enumerate(unique)}
    n = max(len(unique), 1)
    index = np.asarray([position[int(slot)] for slot in slots], dtype=int)
    fraction = index / max(n - 1, 1)
    if scenario == "stable":
        ratio = np.ones(len(slots))
    elif scenario == "down_step":
        ratio = np.where(fraction < 0.5, 1.0, 0.80)
    elif scenario == "up_step":
        ratio = np.where(fraction < 0.5, 1.0, 1.20)
    elif scenario == "down_ramp":
        ratio = 1.0 - 0.30 * fraction
    elif scenario == "alternating":
        ratio = np.where(index % 2 == 0, 0.75, 1.25)
    elif scenario == "hidden_blocks":
        rng = np.random.default_rng(900_000 + seed)
        block_values = rng.uniform(BOUND_LOWER, BOUND_UPPER,
                                   size=max(1, int(np.ceil(n / 8))))
        ratio = block_values[np.minimum(index // 8, len(block_values) - 1)]
    elif scenario == "outside_low":
        ratio = np.full(len(slots), 0.55)
    elif scenario == "outside_high":
        ratio = np.full(len(slots), 1.45)
    else:
        raise ValueError(scenario)
    return ratio.astype(float)


def build_config(folder: Path, n_providers: int, profile: str) -> dict:
    control = {"scenario": "control", "factor": "control", "level": 0.0}
    cfg = configure(control, n_providers)
    cfg["dataset"] = {"processed_dir": str(folder), "use_region": "test",
                      "service_radius_km": 3.0}
    cfg["simulation"]["log_level"] = "selected"
    cfg["matching"]["objective"] = "coverage_first_payment_second"
    if profile == "fixed_envelope":
        cfg["uncertainty_aware"] = {
            "enabled": True,
            "state_confidence_z": 3.290527,
            "cost_relative_bound": 0.0,
            "response_relative_bound": 0.0,
            "response_ratio_lower": BOUND_LOWER,
            "response_ratio_upper": BOUND_UPPER,
            "missing_state_floor_zero": True,
            "interpretation": (
                "fixed response envelope declared before the held-out test; "
                "no test-side response observation enters contract construction"),
        }
        cfg["contract"]["D_bar"] = 50.0
        cfg["matching"]["budget_ratio"] = 3.5
    elif profile != "frozen_point":
        raise ValueError(profile)
    return cfg


def run_one(episodes: Path, scenario: str, profile: str, seed: int) -> dict:
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    cfg = build_config(folder, int(meta["n_providers"]), profile)
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
    tasks = data["tasks"].copy()
    tasks["kappa_design"] = tasks["kappa"].to_numpy(float)
    ratios = slot_ratios(tasks["slot"].to_numpy(int), scenario, seed)
    tasks["kappa"] = tasks["kappa_design"].to_numpy(float) * ratios
    tasks["response_ratio_audit"] = ratios
    data["tasks"] = tasks

    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    elapsed = time.perf_counter() - started
    summary = result["summary"]
    slot = result["slot_log"]
    assigned = int(summary["num_assigned"])
    qualified = int(slot["num_qualified_completed"].sum())
    qos_violations = max(0, assigned - qualified)
    total_edges = float((slot["num_providers"] * slot["num_tasks"]).sum())
    feasible_edges = float(slot["num_feasible_pairs"].sum())
    within = bool(np.all((ratios >= BOUND_LOWER - 1e-12)
                         & (ratios <= BOUND_UPPER + 1e-12)))
    return {
        "run_id": f"e17-{scenario}-{profile}-seed-{seed:03d}",
        "scenario": scenario, "profile": profile, "seed": seed,
        "ratio_min_audit": float(ratios.min()),
        "ratio_max_audit": float(ratios.max()),
        "within_declared_envelope": within,
        "test_side_response_used_for_decision": False,
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "service_coverage": float(summary["assignment_ratio"]),
        "qos_violation_rate": qos_violations / max(assigned, 1),
        "under_incentive_rate": int(summary["ir_violations"]) / max(assigned, 1),
        "target_miss_rate": int(summary["target_violations"]) / max(assigned, 1),
        "legal_edge_rate": feasible_edges / max(total_edges, 1.0),
        "platform_utility": float(summary["platform_utility"]),
        "core_runtime_s": float(summary["total_runtime"]),
        "end_to_end_runtime_s": elapsed,
        "event_tape_hash": tape_hash(folder),
        "run_status": result["diagnostics"].get("status", "unknown"),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ("pps", "service_coverage", "qos_violation_rate",
               "under_incentive_rate", "target_miss_rate", "legal_edge_rate",
               "platform_utility")
    rows = []
    for scenario_index, (scenario, group) in enumerate(
            frame.groupby("scenario", sort=False), 1):
        reference = group[group["profile"] == "frozen_point"].set_index("seed")
        for profile_index, (profile, sub) in enumerate(
                group.groupby("profile", sort=False), 1):
            sub = sub.set_index("seed").sort_index()
            base = reference.loc[sub.index]
            row = {
                "scenario": scenario, "profile": profile, "n_pairs": len(sub),
                "within_declared_envelope": bool(sub["within_declared_envelope"].all()),
                "test_side_response_used_for_decision": bool(
                    sub["test_side_response_used_for_decision"].any()),
                "all_runs_ok": bool(sub["run_status"].isin(["ok", "completed"]).all()),
                "ratio_min_audit": float(sub["ratio_min_audit"].min()),
                "ratio_max_audit": float(sub["ratio_max_audit"].max()),
            }
            for metric_index, metric in enumerate(metrics, 1):
                values = sub[metric].to_numpy(float)
                delta = values - base[metric].to_numpy(float)
                lo, hi = bootstrap_ci(values, 31_000 + 100 * scenario_index
                                      + 10 * profile_index + metric_index)
                dlo, dhi = bootstrap_ci(delta, 41_000 + 100 * scenario_index
                                        + 10 * profile_index + metric_index)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
                row[f"delta_{metric}"] = float(delta.mean())
                row[f"delta_{metric}_ci_low"] = dlo
                row[f"delta_{metric}_ci_high"] = dhi
            row["safety_gate"] = bool(
                row["all_runs_ok"]
                and not row["test_side_response_used_for_decision"]
                and row["qos_violation_rate_ci_high"] <= 0.005
                and row["under_incentive_rate_ci_high"] <= 0.005
                and row["target_miss_rate_ci_high"] <= 0.005)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    jobs = [(args.episodes, scenario, profile, seed)
            for scenario in SCENARIOS for profile in PROFILES for seed in seeds]
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            _, scenario, profile, seed = futures[future]
            rows.append(future.result())
            print(f"[{index:04d}/{len(jobs)}] E17 {scenario} {profile} seed={seed}",
                  flush=True)
    frame = pd.DataFrame(rows).sort_values(["scenario", "profile", "seed"])
    summary = summarize(frame)
    out = ROOT.parent / "results" / "e17_response_drift"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    frame.to_csv(out / f"e17_{suffix}_seed_results.csv", index=False)
    summary.to_csv(out / f"e17_{suffix}_paired_summary.csv", index=False)
    metadata = {
        "experiment": "E17 post-calibration response drift and observability stress",
        "dataset": "GeoLife GPS Trajectories 1.3 / frozen medium test region",
        "design_response": "original task response rate frozen before test",
        "declared_response_ratio_envelope": [BOUND_LOWER, BOUND_UPPER],
        "test_side_response_used_for_decision": False,
        "audit_only_scenarios": ["outside_low", "outside_high"],
        "scenarios": list(SCENARIOS), "profiles": list(PROFILES),
        "seeds": seeds, "runs": len(frame),
        "elapsed_seconds": time.time() - started,
        "raw_or_derived_trace_redistributed": False,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / f"e17_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[["scenario", "profile", "within_declared_envelope",
                   "mean_service_coverage", "mean_pps", "mean_target_miss_rate",
                   "mean_qos_violation_rate", "safety_gate"]].to_string(index=False),
          flush=True)


if __name__ == "__main__":
    main()
