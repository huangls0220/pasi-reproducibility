"""E13: response-error safety-reserve frontier for PASI-U.

This experiment does not relax PASI-U's uncertainty envelope.  It varies two
transparent realizability resources: the maximum bonus and the matching
budget.  Cap-only, budget-only, and joint reserves separate contract-edge
recovery from downstream budget rationing.  Every realised outcome continues
to be evaluated with the frozen true-side E8 simulator.
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

from scripts.run_e8_robustness import (bootstrap_ci, joint_scenario, scenarios,
                                       tape_hash)
from scripts.run_e11_pasi_u import configure_uncertainty
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator


BASE_D_BAR = 10.0
BASE_BUDGET_RATIO = 0.70


def response_scenarios() -> list[dict]:
    return [s for s in scenarios() if s["factor"] in {"control", "response_error"}] + [joint_scenario()]


def configure_reserve(scenario: dict, n_providers: int, mode: str,
                      multiplier: float) -> dict:
    cfg = configure_uncertainty(scenario, n_providers)
    cap_mult = multiplier if mode in {"cap", "joint"} else 1.0
    budget_mult = multiplier if mode in {"budget", "joint"} else 1.0
    cfg["contract"]["D_bar"] = BASE_D_BAR * cap_mult
    cfg["matching"]["budget_ratio"] = BASE_BUDGET_RATIO * budget_mult
    cfg["matching"]["objective"] = "coverage_first_payment_second"
    cfg["safety_reserve"] = {
        "mode": mode,
        "multiplier": multiplier,
        "base_D_bar": BASE_D_BAR,
        "effective_D_bar": cfg["contract"]["D_bar"],
        "base_budget_ratio": BASE_BUDGET_RATIO,
        "effective_budget_ratio": cfg["matching"]["budget_ratio"],
        "interpretation": (
            "explicit uncertainty reserve; robust response/cost/state bounds "
            "remain unchanged"
        ),
    }
    return cfg


def run_one_reserve(episodes: Path, scenario: dict, seed: int, mode: str,
                    multiplier: float) -> dict:
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    cfg = configure_reserve(scenario, int(meta["n_providers"]), mode, multiplier)
    cfg["dataset"] = {"processed_dir": str(folder), "use_region": "all",
                      "service_radius_km": 3.0}
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    elapsed = time.perf_counter() - started
    summary = result["summary"]
    slot = result["slot_log"]
    pair = result["pair_log"]
    selected = pair[pair["selected"].astype(bool)] if not pair.empty else pair
    assigned = int(summary["num_assigned"])
    qualified = int(slot["num_qualified_completed"].sum())
    total_edges = float((slot["num_providers"] * slot["num_tasks"]).sum())
    feasible_edges = float(slot["num_feasible_pairs"].sum())
    d_star = selected["D_star"].to_numpy(float) if len(selected) else np.array([], dtype=float)
    scenario_json = json.dumps({"scenario": scenario, "mode": mode,
                                "multiplier": multiplier}, sort_keys=True,
                               separators=(",", ":"))
    return {
        "run_id": f"e13-{scenario['scenario']}-{mode}-{multiplier:g}-seed-{seed:03d}",
        "dataset": "geolife", "mechanism": "PASI-U-R",
        "variant": scenario["scenario"], "factor": scenario["factor"],
        "level": float(scenario["level"]), "seed": seed,
        "reserve_mode": mode, "reserve_multiplier": float(multiplier),
        "D_bar": float(cfg["contract"]["D_bar"]),
        "budget_ratio": float(cfg["matching"]["budget_ratio"]),
        "response_error": float(cfg["model_error"]["response_bias"]),
        "cost_error": float(cfg["model_error"]["cost_bias"]),
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "service_coverage": float(summary["assignment_ratio"]),
        "qos_violation_rate": max(0, assigned - qualified) / max(assigned, 1),
        "target_miss_rate": int(summary["target_violations"]) / max(assigned, 1),
        "under_incentive_rate": int(summary["ir_violations"]) / max(assigned, 1),
        "legal_edge_rate": feasible_edges / max(total_edges, 1.0),
        "platform_utility": float(summary["platform_utility"]),
        "reserve_activation_rate": float(np.mean(d_star > BASE_D_BAR + 1e-9)) if len(d_star) else 0.0,
        "mean_selected_D": float(np.mean(d_star)) if len(d_star) else 0.0,
        "max_selected_D": float(np.max(d_star)) if len(d_star) else 0.0,
        "mean_selected_cap_topup": float(np.mean(np.maximum(d_star - BASE_D_BAR, 0.0))) if len(d_star) else 0.0,
        "core_runtime_s": float(summary["total_runtime"]),
        "end_to_end_runtime_s": elapsed,
        "event_tape_hash": tape_hash(folder),
        "config_hash": hashlib.sha256(scenario_json.encode()).hexdigest(),
        "run_status": result["diagnostics"].get("status", "unknown"),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["pps", "service_coverage", "qos_violation_rate",
               "target_miss_rate", "under_incentive_rate", "legal_edge_rate",
               "platform_utility", "reserve_activation_rate",
               "mean_selected_cap_topup"]
    rows: list[dict] = []
    base_candidates = frame[frame["reserve_multiplier"] == 1.0]
    if base_candidates.empty:
        raise ValueError("summary requires at least one multiplier-1 reserve profile")
    preferred_mode = "cap" if "cap" in set(base_candidates["reserve_mode"]) else base_candidates["reserve_mode"].iloc[0]
    base = base_candidates[base_candidates["reserve_mode"] == preferred_mode]
    for idx, ((variant, mode, mult), sub) in enumerate(
            frame.groupby(["variant", "reserve_mode", "reserve_multiplier"], sort=False), 1):
        ref = base[base["variant"] == variant]
        merged = sub.merge(ref, on="seed", suffixes=("", "_base"), validate="one_to_one")
        row = {
            "variant": variant, "factor": sub["factor"].iloc[0],
            "level": float(sub["level"].iloc[0]), "reserve_mode": mode,
            "reserve_multiplier": float(mult), "D_bar": float(sub["D_bar"].iloc[0]),
            "budget_ratio": float(sub["budget_ratio"].iloc[0]), "n_pairs": len(merged),
            "all_runs_ok": bool(sub["run_status"].isin(["ok", "completed"]).all()),
        }
        for j, metric in enumerate(metrics, 1):
            values = merged[metric].to_numpy(float)
            deltas = values - merged[f"{metric}_base"].to_numpy(float)
            lo, hi = bootstrap_ci(deltas, seed=idx * 100 + j)
            abs_lo, abs_hi = bootstrap_ci(values, seed=idx * 1000 + j)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"delta_vs_base_{metric}"] = float(deltas.mean())
            row[f"delta_vs_base_{metric}_ci_low"] = lo
            row[f"delta_vs_base_{metric}_ci_high"] = hi
            row[f"{metric}_ci_low"] = abs_lo
            row[f"{metric}_ci_high"] = abs_hi
        row["safety_pass"] = bool(
            row["under_incentive_rate_ci_high"] <= 0.005
            and row["qos_violation_rate_ci_high"] <= 0.005
            and row["target_miss_rate_ci_high"] <= 0.005
        )
        row["coverage_recovery_pass"] = bool(
            row["delta_vs_base_service_coverage_ci_low"] >= -1e-12
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--modes", nargs="+", default=["cap", "budget", "joint"],
                        choices=["cap", "budget", "joint"])
    parser.add_argument("--multipliers", nargs="+", type=float,
                        default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument(
        "--profiles", nargs="+",
        help="Explicit MODE:MULTIPLIER pairs; overrides --modes/--multipliers",
    )
    args = parser.parse_args()
    if args.profiles:
        profiles = []
        for item in args.profiles:
            mode, raw_mult = item.split(":", 1)
            if mode not in {"cap", "budget", "joint"}:
                raise ValueError(f"unknown reserve mode in profile: {mode}")
            profiles.append((mode, float(raw_mult)))
    else:
        profiles = [(mode, mult) for mode in args.modes for mult in args.multipliers]
    if any(mult < 1.0 for _, mult in profiles):
        raise ValueError("reserve multipliers must be >= 1")
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    jobs = [(s, seed, mode, mult) for s in response_scenarios()
            for seed in seeds for mode, mult in profiles]
    out_dir = ROOT.parent / "results" / "e13_response_reserve"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one_reserve, args.episodes, s, seed, mode, mult):
                   (s["scenario"], seed, mode, mult)
                   for s, seed, mode, mult in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            variant, seed, mode, mult = futures[future]
            rows.append(future.result())
            print(f"[{done:04d}/{len(jobs)}] {variant} seed={seed} {mode} x{mult:g}", flush=True)
    frame = pd.DataFrame(rows).sort_values(
        ["factor", "level", "reserve_mode", "reserve_multiplier", "seed"])
    suffix = "pilot" if args.pilot else "formal"
    frame.to_csv(out_dir / f"e13_{suffix}_seed_results.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(out_dir / f"e13_{suffix}_frontier.csv", index=False)
    metadata = {
        "experiment": "E13 PASI-U response-error safety-reserve frontier",
        "dataset": "GeoLife GPS Trajectories 1.3 / frozen E8 medium workload",
        "response_bound": "unchanged scenario-magnitude symmetric PASI-U envelope",
        "reserve_profiles": [{"mode": mode, "multiplier": mult}
                             for mode, mult in profiles],
        "seeds": seeds, "runs": len(frame), "elapsed_seconds": time.time() - started,
        "raw_or_derived_trace_redistributed": False,
    }
    (out_dir / f"e13_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    cols = ["variant", "reserve_mode", "reserve_multiplier", "D_bar", "budget_ratio",
            "mean_service_coverage", "delta_vs_base_service_coverage", "mean_pps",
            "delta_vs_base_pps", "mean_reserve_activation_rate", "safety_pass"]
    print(summary[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
