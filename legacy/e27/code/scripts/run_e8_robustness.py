"""Run the preregistered E8 single-factor robustness matrix on GeoLife medium."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_a1_3_complete import legacy_cfg
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator


def scenarios() -> list[dict]:
    rows = [{"scenario": "control", "factor": "control", "level": 0.0}]
    rows += [{"scenario": f"state_noise_{v:g}", "factor": "state_noise", "level": v}
             for v in (0.02, 0.05, 0.10, 0.20)]
    rows += [{"scenario": f"state_delay_{v}", "factor": "state_delay", "level": v}
             for v in (1, 3, 5)]
    rows += [{"scenario": f"state_missing_{v:g}", "factor": "state_missing", "level": v}
             for v in (0.05, 0.10, 0.20, 0.40)]
    rows += [{"scenario": f"response_error_{v:+g}", "factor": "response_error", "level": v}
             for v in (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30)]
    rows += [{"scenario": f"cost_error_{v:+g}", "factor": "cost_error", "level": v}
             for v in (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30)]
    return rows


def joint_scenario() -> dict:
    """Worst-direction combination selected after the single-factor pilot."""
    return {
        "scenario": "joint_worst", "factor": "joint", "level": 1.0,
        "state_noise": 0.20, "state_delay": 5, "state_missing": 0.40,
        "response_error": 0.30, "cost_error": -0.30,
    }


def tape_hash(folder: Path) -> str:
    h = hashlib.sha256()
    for name in ("tasks.parquet", "providers.parquet", "provider_static.parquet", "meta.json"):
        h.update((folder / name).read_bytes())
    return h.hexdigest()


def configure(scenario: dict, n_providers: int) -> dict:
    cfg = legacy_cfg(T=1000, N=n_providers, M=4, pattern="stationary")
    cfg["simulation"]["log_level"] = "selected"
    factor, level = scenario["factor"], float(scenario["level"])
    if factor == "joint":
        noise = float(scenario["state_noise"])
        delay = int(scenario["state_delay"])
        missing = float(scenario["state_missing"])
        response = float(scenario["response_error"])
        cost = float(scenario["cost_error"])
    else:
        noise = level if factor == "state_noise" else 0.0
        delay = int(level) if factor == "state_delay" else 0
        missing = level if factor == "state_missing" else 0.0
        response = level if factor == "response_error" else 0.0
        cost = level if factor == "cost_error" else 0.0
    cfg["state_observation"] = {
        "noise_std": noise,
        "delay_slots": delay,
        "missing_rate": missing,
        "missing_strategy": "last_observation_carried_forward",
    }
    cfg["model_error"] = {
        "response_bias": response,
        "cost_bias": cost,
    }
    return cfg


def run_one(episodes: Path, scenario: dict, seed: int) -> dict:
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    cfg = configure(scenario, int(meta["n_providers"]))
    cfg["dataset"] = {"processed_dir": str(folder), "use_region": "all",
                      "service_radius_km": 3.0}
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
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
    scenario_json = json.dumps(scenario, sort_keys=True, separators=(",", ":"))
    return {
        "run_id": f"e8-{scenario['scenario']}-seed-{seed:03d}",
        "dataset": "geolife", "mechanism": "PASI", "variant": scenario["scenario"],
        "workload": "medium", "seed": seed, "N": int(meta["n_providers"]), "M": 4,
        "mismatch_direction": "none", "mismatch_magnitude": 0.0,
        "mismatch_duration": 0,
        "state_noise": float(cfg["state_observation"]["noise_std"]),
        "state_delay": int(cfg["state_observation"]["delay_slots"]),
        "state_missing": float(cfg["state_observation"]["missing_rate"]),
        "response_error": float(cfg["model_error"]["response_bias"]),
        "cost_error": float(cfg["model_error"]["cost_bias"]),
        "factor": scenario["factor"], "level": float(scenario["level"]),
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "service_coverage": float(summary["assignment_ratio"]),
        "qos_violations": qos_violations,
        "qos_violation_rate": qos_violations / max(assigned, 1),
        "under_incentive_rate": int(summary["ir_violations"]) / max(assigned, 1),
        "legal_edge_rate": feasible_edges / max(total_edges, 1.0),
        "correction_rate": math.nan,
        "target_miss_rate": int(summary["target_violations"]) / max(assigned, 1),
        "platform_utility": float(summary["platform_utility"]),
        "core_runtime_s": float(summary["total_runtime"]),
        "end_to_end_runtime_s": elapsed,
        "peak_rss_mb": math.nan, "incremental_rss_mb": math.nan,
        "config_hash": hashlib.sha256(scenario_json.encode()).hexdigest(),
        "event_tape_hash": tape_hash(folder),
        "code_commit": "e8-derived-from-b635f664c68ffdef54ebe4c55a7d57990479b7fd",
        "run_status": result["diagnostics"].get("status", "unknown"),
    }


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng([20260814, seed])
    n = len(values)
    means = values[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    control = frame[frame["variant"] == "control"].set_index("seed")
    out = []
    metrics = ["pps", "service_coverage", "qos_violation_rate", "target_miss_rate",
               "under_incentive_rate", "legal_edge_rate", "platform_utility"]
    for idx, (variant, sub) in enumerate(frame.groupby("variant", sort=False), 1):
        sub = sub.set_index("seed").sort_index()
        base = control.loc[sub.index]
        row = {"variant": variant, "factor": sub["factor"].iloc[0],
               "level": sub["level"].iloc[0], "n_pairs": len(sub),
               "all_runs_ok": bool(sub["run_status"].isin(["ok", "completed"]).all())}
        for metric in metrics:
            values = sub[metric].to_numpy(dtype=float)
            delta = values - base[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(delta, idx)
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"delta_{metric}"] = float(np.mean(delta))
            row[f"delta_{metric}_ci_low"] = lo
            row[f"delta_{metric}_ci_high"] = hi
        row["safety_gate_pass"] = bool(
            row["mean_under_incentive_rate"] <= 1e-12
            and row["mean_qos_violation_rate"] <= 0.005
            and row["mean_target_miss_rate"] <= 0.005
            and row["delta_qos_violation_rate_ci_high"] <= 0.005
            and row["delta_target_miss_rate_ci_high"] <= 0.005
            and row["delta_service_coverage_ci_low"] >= -0.01
        )
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--joint", action="store_true")
    args = parser.parse_args()
    scens = ([scenarios()[0], joint_scenario()] if args.joint else scenarios())
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    jobs = [(s, seed) for s in scens for seed in seeds]
    out_dir = ROOT.parent / "results" / "e8"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, args.episodes, s, seed): (s["scenario"], seed)
                   for s, seed in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            variant, seed = futures[future]
            rows.append(future.result())
            print(f"[{done:04d}/{len(jobs)}] {variant} seed={seed}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["factor", "level", "seed"])
    suffix = "joint" if args.joint else ("pilot" if args.pilot else "formal")
    frame.to_csv(out_dir / f"e8_{suffix}_seed_results.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(out_dir / f"e8_{suffix}_summary.csv", index=False)
    metadata = {
        "dataset": "GeoLife GPS Trajectories 1.3 / medium workload",
        "scenario_count": len(scens), "seeds": seeds, "runs": len(frame),
        "missing_strategy": "last observation carried forward",
        "decision_side": "perturbed state/model", "evaluation_side": "true state/model",
        "elapsed_seconds": time.time() - started,
        "raw_or_derived_trace_redistributed": False,
    }
    (out_dir / f"e8_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[["variant", "mean_pps", "mean_service_coverage",
                   "mean_qos_violation_rate", "mean_target_miss_rate",
                   "mean_under_incentive_rate",
                   "safety_gate_pass"]].to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
