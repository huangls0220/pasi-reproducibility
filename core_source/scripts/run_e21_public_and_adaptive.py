"""E21: mapped public baselines and past-only adaptive response intervals.

This experiment is deliberately simulation-only.  It uses the same generated
event tape, information fields, QoS constraints, provider capacity, budget and
coverage-first/payment-second objective for every method.  QUAC-I-MAPPED and
QIM-E-MAPPED are independent protocol mappings, not author implementations.
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

from scripts.run_a1_3_complete import legacy_cfg
from scripts.run_e8_robustness import bootstrap_ci
from src.datasets.synthetic import generate_synthetic_episode
from src.simulator import Simulator

METHODS = ("PASI", "QUAC-F", "QUAC-I-MAPPED", "QIM-E-MAPPED")
WORKLOADS = {"low": 2, "medium": 4, "high": 6}
SCENARIOS = ("stable", "down_step", "down_ramp", "alternating", "hidden_blocks")
PROFILES = ("frozen_point", "fixed_envelope", "adaptive_interval")


def config(n_tasks: int) -> dict:
    cfg = legacy_cfg(T=180, N=36, M=n_tasks, pattern="stationary")
    cfg["simulation"]["log_level"] = "selected"
    cfg["matching"]["objective"] = "coverage_first_payment_second"
    cfg["matching"]["budget_ratio"] = 3.5
    cfg["contract"]["D_bar"] = 50.0
    return cfg


def ratios_for(tasks: pd.DataFrame, scenario: str, seed: int) -> np.ndarray:
    slots = tasks["slot"].to_numpy(int)
    unique = np.sort(np.unique(slots))
    pos = {int(value): index for index, value in enumerate(unique)}
    idx = np.asarray([pos[int(value)] for value in slots], dtype=int)
    frac = idx / max(len(unique) - 1, 1)
    if scenario == "stable":
        return np.ones(len(tasks))
    if scenario == "down_step":
        return np.where(frac < 0.5, 1.0, 0.70)
    if scenario == "down_ramp":
        return 1.0 - 0.30 * frac
    if scenario == "alternating":
        return np.where(idx % 2 == 0, 0.70, 1.30)
    if scenario == "hidden_blocks":
        rng = np.random.default_rng([seed, 21001])
        blocks = rng.uniform(0.70, 1.30, size=max(1, int(np.ceil(len(unique) / 8))))
        return blocks[np.minimum(idx // 8, len(blocks) - 1)]
    raise ValueError(scenario)


def metrics(result: dict) -> dict:
    summary = result["summary"]
    slot = result["slot_log"]
    assigned = int(summary["num_assigned"])
    qualified = int(slot["num_qualified_completed"].sum())
    return {
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "service_coverage": float(summary["assignment_ratio"]),
        "qos_violation_rate": max(0, assigned - qualified) / max(assigned, 1),
        "under_incentive_rate": int(summary["ir_violations"]) / max(assigned, 1),
        "target_miss_rate": int(summary["target_violations"]) / max(assigned, 1),
        "platform_utility": float(summary["platform_utility"]),
        "assigned": assigned,
        "status": result["diagnostics"].get("status", "unknown"),
    }


def baseline_one(workload: str, n_tasks: int, method: str, seed: int) -> dict:
    cfg = config(n_tasks)
    if method == "QUAC-I-MAPPED":
        cfg["uncertainty_aware"] = {
            "enabled": True, "state_confidence_z": 0.0,
            "cost_relative_bound": 0.20,
            "response_ratio_lower": 0.80, "response_ratio_upper": 1.20,
        }
    data = generate_synthetic_episode(cfg, seed=seed)
    started = time.perf_counter()
    result = Simulator(cfg, data, method=method, seed=seed).run()
    return {"family": "public_baseline", "workload": workload,
            "method": method, "seed": seed,
            "runtime_s": time.perf_counter() - started, **metrics(result)}


def adaptive_one(scenario: str, profile: str, seed: int) -> dict:
    cfg = config(WORKLOADS["medium"])
    data = generate_synthetic_episode(cfg, seed=seed)
    tasks = data["tasks"].copy()
    tasks["kappa_design"] = tasks["kappa"].to_numpy(float)
    ratio = ratios_for(tasks, scenario, seed)
    tasks["kappa"] = tasks["kappa_design"].to_numpy(float) * ratio
    tasks["response_ratio_audit"] = ratio
    data["tasks"] = tasks
    if profile != "frozen_point":
        cfg["uncertainty_aware"] = {
            "enabled": True, "state_confidence_z": 0.0,
            "cost_relative_bound": 0.0,
            "response_ratio_lower": 0.70, "response_ratio_upper": 1.30,
        }
    if profile == "adaptive_interval":
        cfg["uncertainty_aware"]["adaptive_response"] = {
            "enabled": True, "min_history": 24, "window": 96,
            "miscoverage_alpha": 0.05, "safety_margin": 0.02,
            "global_lower": 0.70, "global_upper": 1.30,
        }
    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    slot = result["slot_log"]
    coverage_loss = np.nan
    final_lo = slot["response_bound_lower"].iloc[-1]
    final_hi = slot["response_bound_upper"].iloc[-1]
    return {"family": "adaptive_response", "scenario": scenario,
            "profile": profile, "seed": seed, "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()), "test_side_lookahead": False,
            "final_bound_lower": float(final_lo) if final_lo is not None else np.nan,
            "final_bound_upper": float(final_hi) if final_hi is not None else np.nan,
            "runtime_s": time.perf_counter() - started,
            "coverage_loss": coverage_loss, **metrics(result)}


def paired_summary(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    sub = frame[frame["family"] == family].copy()
    group_cols = ["workload", "method"] if family == "public_baseline" else ["scenario", "profile"]
    ref_col = "method" if family == "public_baseline" else "profile"
    ref_name = "PASI" if family == "public_baseline" else "frozen_point"
    outer = group_cols[0]
    rows = []
    for outer_index, (outer_value, group) in enumerate(sub.groupby(outer, sort=False), 1):
        reference = group[group[ref_col] == ref_name].set_index("seed")
        for inner_index, (inner_value, cell) in enumerate(group.groupby(group_cols[1], sort=False), 1):
            cell = cell.set_index("seed").sort_index()
            base = reference.loc[cell.index]
            row = {outer: outer_value, group_cols[1]: inner_value,
                   "reference": ref_name, "n_pairs": len(cell),
                   "all_runs_ok": bool((cell["status"] == "ok").all())}
            for metric_index, metric in enumerate(("pps", "service_coverage",
                                                    "qos_violation_rate",
                                                    "under_incentive_rate",
                                                    "target_miss_rate",
                                                    "platform_utility"), 1):
                values = cell[metric].to_numpy(float)
                delta = values - base[metric].to_numpy(float)
                lo, hi = bootstrap_ci(values, 21000 + 100 * outer_index +
                                      10 * inner_index + metric_index)
                dlo, dhi = bootstrap_ci(delta, 31000 + 100 * outer_index +
                                        10 * inner_index + metric_index)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
                row[f"delta_{metric}"] = float(delta.mean())
                row[f"delta_{metric}_ci_low"] = dlo
                row[f"delta_{metric}_ci_high"] = dhi
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    jobs = []
    for workload, n_tasks in WORKLOADS.items():
        jobs.extend(("baseline", (workload, n_tasks, method, seed))
                    for method in METHODS for seed in seeds)
    for scenario in SCENARIOS:
        jobs.extend(("adaptive", (scenario, profile, seed))
                    for profile in PROFILES for seed in seeds)
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(baseline_one if kind == "baseline" else adaptive_one,
                               *job): (kind, job) for kind, job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 20 == 0 or index == len(jobs):
                print(f"[{index}/{len(jobs)}] E21", flush=True)
    frame = pd.DataFrame(rows)
    out = ROOT / "results" / "e21_public_and_adaptive"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    frame.to_csv(out / f"e21_{suffix}_seed_results.csv", index=False)
    baseline = paired_summary(frame, "public_baseline")
    adaptive = paired_summary(frame, "adaptive_response")
    baseline.to_csv(out / f"e21_{suffix}_public_baseline_summary.csv", index=False)
    adaptive.to_csv(out / f"e21_{suffix}_adaptive_summary.csv", index=False)
    metadata = {
        "experiment": "E21 mapped public baselines and adaptive response interval",
        "evidence_scope": "simulation-only; no real-participant evidence",
        "event_tapes": "same deterministic generated tape per seed across methods/profiles",
        "public_mappings": {
            "QUAC-I-MAPPED": "independent affine-contract mapping with frozen type envelope",
            "QIM-E-MAPPED": "independent minimum-QoS procurement mapping using modeled cost bids",
        },
        "not_author_code": True,
        "same_protocol": ["information fields", "QoS", "capacity", "budget",
                          "coverage-first/payment-second objective", "seeds"],
        "adaptive_observations": ["completed-task response ratio", "one-slot causal update"],
        "test_side_lookahead": False,
        "seeds": seeds, "runs": len(frame), "elapsed_seconds": time.time() - started,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / f"e21_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(baseline[["workload", "method", "mean_service_coverage", "mean_pps",
                    "mean_qos_violation_rate"]].to_string(index=False))
    print(adaptive[["scenario", "profile", "mean_service_coverage", "mean_pps",
                    "mean_qos_violation_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
