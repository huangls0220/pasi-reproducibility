"""E23: preregistered past-only response-risk/coverage frontier.

The five guardrail weights are fixed before the formal run.  Weight zero
recovers E22's adaptive interval, while weight one expands every adaptive
interval to E20's global [0.70, 1.30] envelope.  All intermediate profiles use
the same frozen GeoLife episodes, trajectories, contract domain, budget, and
post-assignment observation protocol.  No current-slot realised response is
used for contract construction.
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

from scripts.run_e20_envelope_tradeoff import SCENARIOS, build_config, slot_ratios
from scripts.run_e22_adaptive_geolife import ensure_episodes, metrics
from scripts.run_e8_robustness import bootstrap_ci, tape_hash
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator

LOWER = 0.70
UPPER = 1.30
GUARDRAIL_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)
PROFILES = tuple(f"guardrail_{int(100 * weight):03d}" for weight in GUARDRAIL_WEIGHTS)


def wilson_upper(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    """Two-sided 95% Wilson score interval's upper endpoint."""
    if trials <= 0:
        return float("nan")
    rate = successes / trials
    z2 = z * z
    centre = rate + z2 / (2.0 * trials)
    radius = z * np.sqrt(rate * (1.0 - rate) / trials
                         + z2 / (4.0 * trials * trials))
    return float((centre + radius) / (1.0 + z2 / trials))


def profile_weight(profile: str) -> float:
    return int(profile.rsplit("_", 1)[1]) / 100.0


def run_one(episodes: Path, scenario: str, profile: str, seed: int) -> dict:
    weight = profile_weight(profile)
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    cfg = build_config(folder, int(meta["n_providers"]), "fixed_envelope",
                       LOWER, UPPER)
    cfg["uncertainty_aware"]["adaptive_response"] = {
        "enabled": True,
        "min_history": 24,
        "window": 96,
        "miscoverage_alpha": 0.05,
        "safety_margin": 0.02,
        "global_lower": LOWER,
        "global_upper": UPPER,
        "global_guardrail_weight": weight,
    }
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
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
        "run_id": f"e23-{scenario}-{profile}-seed-{seed:03d}",
        "scenario": scenario,
        "profile": profile,
        "global_guardrail_weight": weight,
        "seed": seed,
        "ratio_min_audit": float(ratios.min()),
        "ratio_max_audit": float(ratios.max()),
        "within_global_envelope": bool(
            np.all((ratios >= LOWER - 1e-12) & (ratios <= UPPER + 1e-12))),
        "test_side_lookahead": False,
        "runtime_s": time.perf_counter() - started,
        "event_tape_hash": tape_hash(folder),
        **metrics(result),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metric_names = (
        "pps", "service_coverage", "qos_violation_rate",
        "under_incentive_rate", "target_miss_rate", "fallback_selected_rate",
    )
    rows: list[dict] = []
    for scenario_index, (scenario, group) in enumerate(
            frame.groupby("scenario", sort=False), 1):
        e22 = group[group["global_guardrail_weight"] == 0.0].set_index("seed")
        e20 = group[group["global_guardrail_weight"] == 1.0].set_index("seed")
        for profile_index, (profile, cell) in enumerate(
                group.groupby("profile", sort=False), 1):
            cell = cell.set_index("seed").sort_index()
            ref_e22 = e22.loc[cell.index]
            ref_e20 = e20.loc[cell.index]
            assigned_total = int(cell["assigned"].sum())
            violation_total = int(cell["qos_violations"].sum())
            row = {
                "scenario": scenario,
                "profile": profile,
                "global_guardrail_weight": float(
                    cell["global_guardrail_weight"].iloc[0]),
                "n_pairs": len(cell),
                "all_runs_ok": bool(
                    cell["status"].isin(["ok", "completed"]).all()),
                "within_global_envelope": bool(
                    cell["within_global_envelope"].all()),
                "test_side_lookahead": bool(cell["test_side_lookahead"].any()),
                "assigned_total": assigned_total,
                "qos_violations_total": violation_total,
                "pooled_qos_violation_rate": (
                    violation_total / max(assigned_total, 1)),
                "qos_violation_wilson_upper_95": wilson_upper(
                    violation_total, assigned_total),
                "fallback_candidates_total": int(
                    cell["fallback_candidates"].sum()),
                "fallback_selected_total": int(cell["fallback_selected"].sum()),
                "adaptive_slot_fraction": float(
                    cell["adaptive_slots"].sum()
                    / max(cell["slot_count"].sum(), 1)),
                "global_slot_fraction": float(
                    cell["global_slots"].sum()
                    / max(cell["slot_count"].sum(), 1)),
                "regime_alarm_slot_fraction": float(
                    cell["regime_alarm_slots"].sum()
                    / max(cell["slot_count"].sum(), 1)),
            }
            for metric_index, metric in enumerate(metric_names, 1):
                values = cell[metric].to_numpy(float)
                delta_e22 = values - ref_e22[metric].to_numpy(float)
                delta_e20 = values - ref_e20[metric].to_numpy(float)
                seed_base = 73_000 + 100 * scenario_index + 10 * profile_index
                lo, hi = bootstrap_ci(values, seed_base + metric_index)
                e22_lo, e22_hi = bootstrap_ci(
                    delta_e22, seed_base + 1000 + metric_index)
                e20_lo, e20_hi = bootstrap_ci(
                    delta_e20, seed_base + 2000 + metric_index)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
                row[f"delta_e22_{metric}"] = float(delta_e22.mean())
                row[f"delta_e22_{metric}_ci_low"] = e22_lo
                row[f"delta_e22_{metric}_ci_high"] = e22_hi
                row[f"delta_e20_{metric}"] = float(delta_e20.mean())
                row[f"delta_e20_{metric}_ci_low"] = e20_lo
                row[f"delta_e20_{metric}_ci_high"] = e20_hi
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
    rows: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            _, scenario, profile, seed = futures[future]
            rows.append(future.result())
            print(f"[{index:04d}/{len(jobs)}] E23 {scenario} {profile} "
                  f"seed={seed}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["scenario", "profile", "seed"])
    summary = summarize(frame)
    out = PROJECT / "results" / "e23_guardrail_frontier"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    frame.to_csv(out / f"e23_{suffix}_seed_results.csv", index=False)
    summary.to_csv(out / f"e23_{suffix}_paired_summary.csv", index=False)
    metadata = {
        "experiment": "E23 preregistered past-only guardrail frontier",
        "dataset": "GeoLife GPS Trajectories 1.3 / frozen medium episodes",
        "e20_e22_protocol_match": [
            "episodes", "event-tape hashes", "six trajectories", "30 paired seeds",
            "global [0.70,1.30] envelope", "budget", "contract domain",
            "coverage-first/payment-second objective",
        ],
        "guardrail_weights_predeclared": list(GUARDRAIL_WEIGHTS),
        "selection_rule": "report all weights; no formal-seed tuning or winner selection",
        "guarantee_scope": (
            "only weight 1.0 inherits the fixed global-envelope guarantee; "
            "weights below 1.0 are empirical risk-coverage points"),
        "test_side_lookahead": False,
        "adaptive_observations": (
            "current-slot response audit enters history only after current-slot "
            "contract construction and assignment; no pre-decision use"),
        "seeds": seeds,
        "runs": len(frame),
        "elapsed_seconds": time.time() - started,
        "position_cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "simulator_hash": hashlib.sha256(
            (ROOT / "src" / "simulator.py").read_bytes()).hexdigest(),
        "raw_or_derived_trace_redistributed": False,
    }
    (out / f"e23_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[[
        "scenario", "global_guardrail_weight", "mean_service_coverage",
        "delta_e20_service_coverage", "qos_violations_total",
        "pooled_qos_violation_rate", "qos_violation_wilson_upper_95",
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
