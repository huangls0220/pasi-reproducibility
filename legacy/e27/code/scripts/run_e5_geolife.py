"""Run the preregistered GeoLife E5 matrix with paired event tapes."""

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
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.run_a1_3_complete import legacy_cfg
from src.datasets.mobility import positions_to_episode
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pair(cache: Path, restricted_root: Path, workload: str, seed: int) -> list[dict]:
    processed = restricted_root / workload / f"seed_{seed:03d}"
    meta = positions_to_episode(cache, processed, workload, seed, service_radius_km=3.0)
    cfg = legacy_cfg(T=1000, N=meta["n_providers"], M=4, pattern="stationary")
    cfg["simulation"]["log_level"] = "selected"
    cfg["dataset"] = {
        "processed_dir": str(processed), "use_region": "all",
        "service_radius_km": 3.0,
    }
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=processed)

    rows = []
    for method in ("MOI", "PASI"):
        result = Simulator(cfg, data, method=method, seed=seed).run()
        summary = result["summary"]
        providers = result["provider_log"]
        assigned = int(summary["num_assigned"])
        row = {
            "dataset": "geolife", "workload": workload, "seed": seed,
            "method": method, "total_tasks": int(summary["total_tasks"]),
            "assigned_tasks": assigned,
            "service_coverage": float(summary["assignment_ratio"]),
            "high_quality_rate": float(summary["HQR"]),
            "average_quality": float(summary["average_quality"]),
            "payment": float(summary["cumulative_payment"]),
            "pps": float(summary["cumulative_payment"]) / max(assigned, 1),
            "platform_utility": float(summary["platform_utility"]),
            "ir_violations": int(summary["ir_violations"]),
            "target_violations": int(summary["target_violations"]),
            "qos_violation_rate": int(summary["target_violations"]) / max(assigned, 1),
            "mean_runtime_state": float(providers["H_before"].mean()),
            "positive_runtime_state_rate": float((providers["H_before"] > 1e-12).mean()),
            "core_runtime_s": float(summary["total_runtime"]),
            "run_status": result["diagnostics"].get("status", "unknown"),
            "position_cache_sha256": meta["position_cache_sha256"],
        }
        rows.append(row)
    return rows


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10000):
    n = len(values)
    means = values[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for workload in ("low", "medium", "high"):
        sub = frame[frame["workload"] == workload]
        moi = sub[sub["method"] == "MOI"].sort_values("seed")
        pasi = sub[sub["method"] == "PASI"].sort_values("seed")
        if not np.array_equal(moi["seed"].to_numpy(), pasi["seed"].to_numpy()):
            raise ValueError(f"Unpaired seeds in {workload}")
        rng = np.random.default_rng([20260814, {"low": 1, "medium": 2, "high": 3}[workload]])
        delta_pps = pasi["pps"].to_numpy() - moi["pps"].to_numpy()
        delta_cov = (pasi["service_coverage"].to_numpy()
                     - moi["service_coverage"].to_numpy())
        delta_qos = (pasi["qos_violation_rate"].to_numpy()
                      - moi["qos_violation_rate"].to_numpy())
        pps_ci = bootstrap_ci(delta_pps, rng)
        cov_ci = bootstrap_ci(delta_cov, rng)
        qos_ci = bootstrap_ci(delta_qos, rng)
        output.append({
            "dataset": "geolife", "workload": workload, "n_pairs": len(moi),
            "moi_mean_pps": moi["pps"].mean(), "pasi_mean_pps": pasi["pps"].mean(),
            "delta_pps_mean": delta_pps.mean(),
            "delta_pps_ci_low": pps_ci[0], "delta_pps_ci_high": pps_ci[1],
            "pps_improved_direction_count": int((delta_pps < 0).sum()),
            "moi_mean_coverage": moi["service_coverage"].mean(),
            "pasi_mean_coverage": pasi["service_coverage"].mean(),
            "delta_coverage_mean": delta_cov.mean(),
            "delta_coverage_ci_low": cov_ci[0], "delta_coverage_ci_high": cov_ci[1],
            "delta_qos_mean": delta_qos.mean(),
            "delta_qos_ci_low": qos_ci[0], "delta_qos_ci_high": qos_ci[1],
            "qos_gate_pass": bool((moi["qos_violation_rate"].eq(0).all()
                                   and pasi["qos_violation_rate"].eq(0).all())
                                  or qos_ci[1] <= 0.005),
            "coverage_gate_pass": bool(cov_ci[0] >= -0.01),
            "pps_evidence_pass": bool(pps_ci[1] < 0),
            "all_runs_ok": bool(sub["run_status"].isin(["ok", "completed"]).all()),
        })
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()

    restricted = PROJECT / "restricted" / "geolife" / "episodes"
    out = PROJECT / "results" / "geolife"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(w, s) for w in ("low", "medium", "high") for s in range(1, args.seeds + 1)]
    rows = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair, args.cache, restricted, w, s): (w, s)
                   for w, s in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            w, s = futures[future]
            rows.extend(future.result())
            print(f"[{done:02d}/{len(jobs)}] {w} seed={s}", flush=True)

    frame = pd.DataFrame(rows).sort_values(["workload", "seed", "method"])
    frame.to_csv(out / "e5_geolife_seed_results.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(out / "e5_geolife_summary.csv", index=False)
    metadata = {
        "dataset": "GeoLife GPS Trajectories 1.3",
        "official_url": "https://www.microsoft.com/en-us/download/details.aspx?id=52367",
        "raw_sha256": "1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6",
        "license": "MSR-LA non-commercial; no redistribution of data or derivative datasets",
        "trace_fields": ["provider identity", "UTC slot", "latitude", "longitude"],
        "model_generated_fields": ["tasks", "cost/response", "capacity", "communication", "behaviour"],
        "source_tag": "prime-exp-route-a-a1-8-stage2bvr-v1.0",
        "source_commit": "b635f664c68ffdef54ebe4c55a7d57990479b7fd",
        "e5_simulator_sha256": file_hash(ROOT / "src" / "simulator.py"),
        "e5_mobility_sha256": file_hash(ROOT / "src" / "datasets" / "mobility.py"),
        "runs": len(frame), "paired_seeds_per_workload": args.seeds,
        "elapsed_seconds": time.time() - started,
        "redistribution_note": "Raw and processed trajectory-derived files are excluded from release packages.",
    }
    (out / "e5_geolife_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
