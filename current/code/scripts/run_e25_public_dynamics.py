"""E25: partial external validation with public participant dynamics.

ExtraSensory affects currently observed provider availability and network
state only.  NetEaseCrowd affects the unobserved execution-response stress
trace only.  GeoLife supplies the frozen mobility/candidate graph.  Requests,
costs, capacities, contracts, and payments remain model-generated.
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

from scripts.run_e20_envelope_tradeoff import build_config
from scripts.run_e22_adaptive_geolife import metrics
from scripts.run_e23_guardrail_frontier import wilson_upper
from scripts.run_e8_robustness import bootstrap_ci, tape_hash
from src.datasets.public_dynamics import (
    load_extrasensory_drivers,
    load_netease_response_drivers,
    resample_trace,
    sha256_file,
)
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator

LOWER = 0.70
UPPER = 1.30
NETEASE_PROFILES = ("frozen_point", "guardrail_075", "fixed_envelope")
EXTRA_METHODS = ("PASI", "MOI")


def _adaptive_config(cfg: dict, weight: float = 0.75) -> None:
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


def run_extra_one(episodes: Path, driver: dict, method: str, seed: int) -> dict:
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    cfg = build_config(folder, int(meta["n_providers"]), "frozen_point", LOWER, UPPER)
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
    providers = data["providers"].copy()
    slots = np.sort(providers["slot"].unique())
    availability = resample_trace(driver["availability_ratio"], len(slots))
    network = resample_trace(driver["network_ratio"], len(slots))
    slot_pos = {int(slot): index for index, slot in enumerate(slots)}
    index = providers["slot"].map(slot_pos).to_numpy(int)
    providers["availability_duration"] *= availability[index]
    providers["communication_rate"] *= network[index]
    data["providers"] = providers
    started = time.perf_counter()
    result = Simulator(cfg, data, method=method, seed=seed).run()
    return {
        "branch": "extrasensory_observed_state",
        "profile": method,
        "seed": seed,
        "source_token": driver["source_token"],
        "availability_ratio_min": float(availability.min()),
        "availability_ratio_max": float(availability.max()),
        "network_ratio_min": float(network.min()),
        "network_ratio_max": float(network.max()),
        "current_state_observed_before_decision": True,
        "test_side_response_used_for_decision": False,
        "event_tape_hash": tape_hash(folder),
        "runtime_s": time.perf_counter() - started,
        **metrics(result),
    }


def run_netease_one(episodes: Path, driver: dict, profile: str, seed: int) -> dict:
    folder = episodes / f"seed_{seed:03d}"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    base = "frozen_point" if profile == "frozen_point" else "fixed_envelope"
    cfg = build_config(folder, int(meta["n_providers"]), base, LOWER, UPPER)
    if profile == "guardrail_075":
        _adaptive_config(cfg, 0.75)
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=folder)
    tasks = data["tasks"].copy()
    slots = np.sort(tasks["slot"].unique())
    ratios_by_slot = resample_trace(driver["response_ratio"], len(slots))
    slot_pos = {int(slot): index for index, slot in enumerate(slots)}
    ratios = ratios_by_slot[tasks["slot"].map(slot_pos).to_numpy(int)]
    tasks["kappa_design"] = tasks["kappa"].to_numpy(float)
    tasks["kappa"] = tasks["kappa_design"].to_numpy(float) * ratios
    tasks["response_ratio_audit"] = ratios
    data["tasks"] = tasks
    started = time.perf_counter()
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    return {
        "branch": "netease_completed_quality",
        "profile": profile,
        "seed": seed,
        "source_token": driver["source_token"],
        "response_ratio_min": float(ratios.min()),
        "response_ratio_max": float(ratios.max()),
        "within_declared_envelope": bool(
            np.all((ratios >= LOWER - 1e-12) & (ratios <= UPPER + 1e-12))),
        "current_response_observed_before_decision": False,
        "observed_completion_only": True,
        "event_tape_hash": tape_hash(folder),
        "runtime_s": time.perf_counter() - started,
        **metrics(result),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for branch, group in frame.groupby("branch", sort=False):
        if branch == "extrasensory_observed_state":
            reference_name = "MOI"
        else:
            reference_name = "fixed_envelope"
        reference = group[group["profile"] == reference_name].set_index("seed")
        for profile_index, (profile, cell) in enumerate(group.groupby("profile", sort=False), 1):
            cell = cell.set_index("seed").sort_index()
            ref = reference.loc[cell.index]
            assigned = int(cell["assigned"].sum())
            violations = int(cell["qos_violations"].sum())
            row = {
                "branch": branch,
                "profile": profile,
                "reference_profile": reference_name,
                "n_pairs": int(len(cell)),
                "all_runs_ok": bool(cell["status"].isin(["ok", "completed"]).all()),
                "assigned_total": assigned,
                "qos_violations_total": violations,
                "pooled_qos_violation_rate": violations / max(assigned, 1),
                "qos_violation_wilson_upper_95": wilson_upper(violations, assigned),
            }
            for metric_index, metric in enumerate(("service_coverage", "pps", "qos_violation_rate"), 1):
                values = cell[metric].to_numpy(float)
                delta = values - ref[metric].to_numpy(float)
                base = 125_000 + 100 * profile_index + metric_index
                lo, hi = bootstrap_ci(values, base)
                dlo, dhi = bootstrap_ci(delta, base + 10_000)
                row[f"mean_{metric}"] = float(values.mean())
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
                row[f"delta_reference_{metric}"] = float(delta.mean())
                row[f"delta_reference_{metric}_ci_low"] = dlo
                row[f"delta_reference_{metric}_ci_high"] = dhi
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--extrasensory-dir", type=Path, required=True)
    parser.add_argument("--netease-csv", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT / "results" / "e25_public_dynamics")
    args = parser.parse_args()
    if args.seeds != 30:
        raise ValueError("E25 is preregistered for exactly 30 paired seeds")
    extra, extra_manifest = load_extrasensory_drivers(args.extrasensory_dir, args.seeds)
    netease, netease_manifest = load_netease_response_drivers(args.netease_csv, args.seeds)
    seeds = [1] if args.pilot else list(range(1, args.seeds + 1))
    jobs = []
    for seed in seeds:
        jobs.extend(("extra", args.episodes, extra[seed], method, seed)
                    for method in EXTRA_METHODS)
        jobs.extend(("netease", args.episodes, netease[seed], profile, seed)
                    for profile in NETEASE_PROFILES)
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for kind, episodes, driver, profile, seed in jobs:
            fn = run_extra_one if kind == "extra" else run_netease_one
            futures[pool.submit(fn, episodes, driver, profile, seed)] = (kind, profile, seed)
        for index, future in enumerate(as_completed(futures), 1):
            kind, profile, seed = futures[future]
            rows.append(future.result())
            print(f"[{index:03d}/{len(jobs)}] E25 {kind} {profile} seed={seed}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["branch", "profile", "seed"])
    summary = summarize(frame)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "pilot" if args.pilot else "formal"
    frame.to_csv(args.out / f"e25_{suffix}_seed_results.csv", index=False)
    summary.to_csv(args.out / f"e25_{suffix}_summary.csv", index=False)
    pd.DataFrame(extra_manifest + netease_manifest).to_csv(
        args.out / "e25_local_source_manifest.csv", index=False)
    metadata = {
        "experiment": "E25 partial external validation with public participant dynamics",
        "seeds": seeds,
        "runs": int(len(frame)),
        "elapsed_seconds": time.time() - started,
        "extrasensory_role": "current observed sensor/network availability multipliers only",
        "netease_role": "completed-worker correctness dynamics transformed with calibration-prefix quantiles only",
        "geolife_role": "frozen mobility and candidate graph",
        "model_generated_fields": ["requests", "costs", "capacities", "contract response mapping", "payments"],
        "not_claimed": ["real bids", "real private costs", "causal payment response", "field deployment"],
        "netease_bias": "top-activity convenience sample and completed-annotation records only",
        "test_side_lookahead": False,
        "raw_participant_data_redistributed": False,
        "extrasensory_mirror_commit": "167e0a899cb73c6e20398e8af886517cc9357c3e",
        "netease_repository_commit": "2c3d461d8cfac70623da86110282876f6555121c",
        "netease_csv_sha256": sha256_file(args.netease_csv),
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "public_dynamics_module_hash": hashlib.sha256(
            (ROOT / "src" / "datasets" / "public_dynamics.py").read_bytes()).hexdigest(),
        "simulator_hash": hashlib.sha256((ROOT / "src" / "simulator.py").read_bytes()).hexdigest(),
    }
    (args.out / f"e25_{suffix}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
