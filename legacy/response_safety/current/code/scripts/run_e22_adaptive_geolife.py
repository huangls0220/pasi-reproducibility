"""E22: past-only adaptive response intervals on the frozen E20 GeoLife tapes.

The runner reuses E20's medium-load episodes, six response trajectories,
global [0.70, 1.30] envelope, budget, contract domain, and 30 paired seeds.
Frozen-point, fixed-envelope, and adaptive contracts therefore differ only in
the response information supplied to contract construction. Realised ratios
enter the adaptive history only after the current slot has been assigned.
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
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.run_e20_envelope_tradeoff import (
    SCENARIOS,
    build_config,
    slot_ratios,
)
from scripts.run_e8_robustness import bootstrap_ci, tape_hash
from src.datasets.mobility import positions_to_episode
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator

HALF_WIDTH = 0.30
LOWER = 1.0 - HALF_WIDTH
UPPER = 1.0 + HALF_WIDTH
PROFILES = ("frozen_point", "fixed_envelope", "adaptive_interval")


def ensure_episodes(cache: Path, episodes: Path, seeds: list[int]) -> None:
    """Rebuild missing medium-load tapes with the frozen E5/E20 adapter."""
    for seed in seeds:
        folder = episodes / f"seed_{seed:03d}"
        required = (
            folder / "tasks.parquet",
            folder / "providers.parquet",
            folder / "provider_static.parquet",
            folder / "meta.json",
        )
        if not all(path.exists() for path in required):
            positions_to_episode(cache, folder, "medium", seed,
                                 service_radius_km=3.0)


def metrics(result: dict) -> dict:
    summary = result["summary"]
    slot = result["slot_log"]
    assigned = int(summary["num_assigned"])
    qualified = int(slot["num_qualified_completed"].sum())
    qos_violations = max(0, assigned - qualified)
    fallback_candidates = int(
        slot.get("response_global_fallback_candidates",
                 pd.Series(dtype=int)).sum())
    fallback_selected = int(
        slot.get("response_global_fallback_selected",
                 pd.Series(dtype=int)).sum())
    adaptive_slots = int(slot["response_bound_mode"].astype(str)
                         .str.startswith("adaptive_").sum())
    global_slots = int(slot["response_bound_mode"].astype(str)
                       .str.startswith("global_").sum())
    alarm_slots = int((slot["response_bound_mode"]
                       == "global_regime_alarm").sum())
    final_lower = pd.to_numeric(
        slot["response_bound_lower"], errors="coerce").dropna()
    final_upper = pd.to_numeric(
        slot["response_bound_upper"], errors="coerce").dropna()
    return {
        "payment": float(summary["cumulative_payment"]),
        "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
        "service_coverage": float(summary["assignment_ratio"]),
        "assigned": assigned,
        "qos_violations": qos_violations,
        "qos_violation_rate": qos_violations / max(assigned, 1),
        "under_incentive_rate": int(summary["ir_violations"]) / max(assigned, 1),
        "target_miss_rate": int(summary["target_violations"]) / max(assigned, 1),
        "fallback_candidates": fallback_candidates,
        "fallback_selected": fallback_selected,
        "fallback_selected_rate": fallback_selected / max(assigned, 1),
        "adaptive_slots": adaptive_slots,
        "global_slots": global_slots,
        "regime_alarm_slots": alarm_slots,
        "slot_count": int(len(slot)),
        "final_bound_lower": (float(final_lower.iloc[-1])
                              if len(final_lower) else np.nan),
        "final_bound_upper": (float(final_upper.iloc[-1])
                              if len(final_upper) else np.nan),
        "status": result["diagnostics"].get("status", "unknown"),
    }


def run_one(episodes: Path, scenario: str, profile: str, seed: int) -> dict:
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    base_profile = "frozen_point" if profile == "frozen_point" else "fixed_envelope"
    cfg = build_config(folder, int(meta["n_providers"]), base_profile,
                       LOWER, UPPER)
    if profile == "adaptive_interval":
        cfg["uncertainty_aware"]["adaptive_response"] = {
            "enabled": True,
            "min_history": 24,
            "window": 96,
            "miscoverage_alpha": 0.05,
            "safety_margin": 0.02,
            "global_lower": LOWER,
            "global_upper": UPPER,
        }
    data = load_trace_episode("geolife", cfg, seed=seed,
                              processed_dir=folder)
    tasks = data["tasks"].copy()
    tasks["kappa_design"] = tasks["kappa"].to_numpy(float)
    ratios = slot_ratios(tasks["slot"].to_numpy(int), scenario, seed,
                         LOWER, UPPER)
    tasks["kappa"] = tasks["kappa_design"].to_numpy(float) * ratios
    tasks["response_ratio_audit"] = ratios
    data["tasks"] = tasks

    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    return {
        "run_id": f"e22-{scenario}-{profile}-seed-{seed:03d}",
        "scenario": scenario,
        "profile": profile,
        "seed": seed,
        "ratio_min_audit": float(ratios.min()),
        "ratio_max_audit": float(ratios.max()),
        "within_global_envelope": bool(
            np.all((ratios >= LOWER - 1e-12)
                   & (ratios <= UPPER + 1e-12))),
        "test_side_lookahead": False,
        "runtime_s": time.perf_counter() - started,
        "event_tape_hash": tape_hash(folder),
        **metrics(result),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics_to_summarize = (
        "pps", "service_coverage", "qos_violation_rate",
        "under_incentive_rate", "target_miss_rate",
        "fallback_selected_rate",
    )
    rows = []
    for scenario_index, (scenario, group) in enumerate(
            frame.groupby("scenario", sort=False), 1):
        reference = group[group["profile"] == "frozen_point"].set_index("seed")
        fixed = group[group["profile"] == "fixed_envelope"].set_index("seed")
        for profile_index, (profile, cell) in enumerate(
                group.groupby("profile", sort=False), 1):
            cell = cell.set_index("seed").sort_index()
            base = reference.loc[cell.index]
            fixed_base = fixed.loc[cell.index]
            row = {
                "scenario": scenario,
                "profile": profile,
                "n_pairs": len(cell),
                "all_runs_ok": bool(cell["status"].isin(["ok", "completed"]).all()),
                "within_global_envelope": bool(cell["within_global_envelope"].all()),
                "test_side_lookahead": bool(cell["test_side_lookahead"].any()),
                "assigned_total": int(cell["assigned"].sum()),
                "qos_violations_total": int(cell["qos_violations"].sum()),
                "fallback_candidates_total": int(cell["fallback_candidates"].sum()),
                "fallback_selected_total": int(cell["fallback_selected"].sum()),
                "adaptive_slot_fraction": float(
                    cell["adaptive_slots"].sum() / max(cell["slot_count"].sum(), 1)),
                "global_slot_fraction": float(
                    cell["global_slots"].sum() / max(cell["slot_count"].sum(), 1)),
                "regime_alarm_slot_fraction": float(
                    cell["regime_alarm_slots"].sum()
                    / max(cell["slot_count"].sum(), 1)),
            }
            for metric_index, metric in enumerate(metrics_to_summarize, 1):
                values = cell[metric].to_numpy(float)
                delta_frozen = values - base[metric].to_numpy(float)
                delta_fixed = values - fixed_base[metric].to_numpy(float)
                seed_base = 52_000 + 100 * scenario_index + 10 * profile_index
                lo, hi = bootstrap_ci(values, seed_base + metric_index)
                flo, fhi = bootstrap_ci(delta_frozen,
                                         seed_base + 1000 + metric_index)
                xlo, xhi = bootstrap_ci(delta_fixed,
                                         seed_base + 2000 + metric_index)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
                row[f"delta_frozen_{metric}"] = float(delta_frozen.mean())
                row[f"delta_frozen_{metric}_ci_low"] = flo
                row[f"delta_frozen_{metric}_ci_high"] = fhi
                row[f"delta_fixed_{metric}"] = float(delta_fixed.mean())
                row[f"delta_fixed_{metric}_ci_low"] = xlo
                row[f"delta_fixed_{metric}_ci_high"] = xhi
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    ensure_episodes(args.cache, args.episodes, seeds)
    jobs = [(args.episodes, scenario, profile, seed)
            for scenario in SCENARIOS for profile in PROFILES for seed in seeds]
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            _, scenario, profile, seed = futures[future]
            rows.append(future.result())
            print(f"[{index:04d}/{len(jobs)}] E22 {scenario} {profile} seed={seed}",
                  flush=True)

    frame = pd.DataFrame(rows).sort_values(["scenario", "profile", "seed"])
    summary = summarize(frame)
    out = PROJECT / "results" / "e22_adaptive_geolife"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    frame.to_csv(out / f"e22_{suffix}_seed_results.csv", index=False)
    summary.to_csv(out / f"e22_{suffix}_paired_summary.csv", index=False)
    metadata = {
        "experiment": "E22 past-only adaptive interval on frozen E20 GeoLife tapes",
        "dataset": "GeoLife GPS Trajectories 1.3 / frozen medium episodes",
        "e20_protocol_match": [
            "episodes", "event-tape hashes", "six trajectories", "30 paired seeds",
            "global [0.70,1.30] envelope", "budget", "contract domain",
            "coverage-first/payment-second objective",
        ],
        "profiles": list(PROFILES),
        "scenarios": list(SCENARIOS),
        "test_side_lookahead": False,
        "adaptive_observations": "completed-task response ratios, appended after assignment",
        "seeds": seeds,
        "runs": len(frame),
        "elapsed_seconds": time.time() - started,
        "position_cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "raw_or_derived_trace_redistributed": False,
    }
    (out / f"e22_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[[
        "scenario", "profile", "mean_service_coverage",
        "delta_frozen_service_coverage", "delta_fixed_service_coverage",
        "mean_pps", "qos_violations_total", "mean_qos_violation_rate",
        "fallback_selected_total", "adaptive_slot_fraction",
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
