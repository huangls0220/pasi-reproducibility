#!/usr/bin/env python3
"""Baseline tuning on the VALIDATION region (Section 13).

Protocol (13.1):
  - validation uses the first 20% of the timeline (T_val = 0.2*T) and a
    DISJOINT seed range (101..) from the formal test seeds (1..30);
  - >= 5 seeds per candidate;
  - results frozen into configs/tuned/{same_budget,quality_matched}.yaml;
  - the test phase never modifies them.

Same-Budget (13.2): budget is enforced by the matching for every method;
the winner maximises mean HQR (tie-break: platform utility).

Quality-Matched (13.3): target = PRIME's validation HQR (or --target);
per baseline, sweep `incentive_scale` on top of the same-budget winner and
pick the LOWEST-payment config with |HQR - target| <= 0.01.  If no
candidate reaches the band, the baseline is reported quality_infeasible.

Usage:
  python scripts/tune_baselines.py --config configs/experiments/overall.yaml \
      --dataset synthetic --protocol both --parallel 14
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VAL_SEEDS = [101, 102, 103, 104, 105]
SCALE_GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

GRIDS = {
    "FR": {
        "base_payment": [0.0, 0.05, 0.1],
        "p": [0.2, 0.4, 0.6],
        "D": [2.0, 3.0, 5.0],
    },
    "DP": {"p": [0.2, 0.3, 0.4, 0.6]},
    "RAI": {"p": [0.2, 0.4, 0.6], "ewma_alpha": [0.1, 0.3]},
    "LDI": {
        "p_start": [0.6, 0.8],
        "p_end": [0.05, 0.1, 0.2],
        "decay_interactions": [20, 40],
        "D": [2.0, 3.0],
    },
}
CFG_KEY = {"FR": "fixed_reward", "DP": "dynamic_pricing",
           "RAI": "reputation_aware", "LDI": "linear_decay"}


def eval_candidate(job: dict) -> dict:
    """Run one (method, params, seed) validation episode."""
    from src.datasets.synthetic import generate_synthetic_episode
    from src.io_utils import deep_merge
    from src.simulator import Simulator

    cfg = deep_merge(job["cfg"], {"baselines": {job["cfg_key"]: job["params"]}}) \
        if job["cfg_key"] else job["cfg"]
    data = generate_synthetic_episode(cfg, seed=job["seed"], pattern="stationary")
    sim = Simulator(cfg, data, method=job["method"], seed=job["seed"])
    res = sim.run()
    s = res["summary"]
    return {
        "method": job["method"], "cand_id": job["cand_id"], "seed": job["seed"],
        "params": job["params"], "HQR": s["HQR"],
        "average_quality": s["average_quality"],
        "cumulative_payment": s["cumulative_payment"],
        "platform_utility": s["platform_utility"],
        "num_assigned": s["num_assigned"],
        "diag": res["diagnostics"]["status"],
    }


def _grid_candidates(method: str) -> list[dict]:
    grid = GRIDS[method]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _run_jobs(jobs: list[dict], parallel: int) -> list[dict]:
    if parallel > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=parallel) as ex:
            return list(ex.map(eval_candidate, jobs, chunksize=2))
    return [eval_candidate(j) for j in jobs]


def main() -> None:
    import numpy as np
    import pandas as pd
    import yaml

    from src.io_utils import deep_merge, load_config

    ap = argparse.ArgumentParser(description="Tune baselines on validation region")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--protocol", default="both",
                    choices=["same_budget", "quality_matched", "both"])
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--target", type=float, default=None,
                    help="Quality-matched HQR target (default: PRIME validation HQR)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    exp_cfg = load_config(args.config)
    exp_cfg.pop("sweep", None)
    default_cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg = deep_merge(default_cfg, exp_cfg)
    T_full = int(cfg.get("simulation", {}).get("T", 1000))
    cfg = deep_merge(cfg, {"simulation": {
        "T": max(50, int(T_full * args.val_frac)),
        "save_logs": ["slot"], "log_level": "selected"}})
    print(f"Validation region: T={cfg['simulation']['T']} seeds={VAL_SEEDS}")

    out_dir = ROOT / "results" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    tuned_dir = ROOT / "configs" / "tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    # ── PRIME reference (for quality-matched target) ─────────
    t0 = time.perf_counter()
    prime_jobs = [{"cfg": cfg, "cfg_key": None, "params": {}, "method": "PRIME",
                   "cand_id": "PRIME_ref", "seed": s} for s in VAL_SEEDS]
    prime_rows = _run_jobs(prime_jobs, args.parallel)
    all_rows += prime_rows
    prime_hqr = float(np.nanmean([r["HQR"] for r in prime_rows]))
    target = args.target if args.target is not None else prime_hqr
    print(f"PRIME validation HQR = {prime_hqr:.4f} -> quality target {target:.4f} "
          f"({time.perf_counter() - t0:.0f}s)")

    # ── Same-budget grid ─────────────────────────────────────
    winners: dict[str, dict] = {}
    for method in ["FR", "DP", "RAI", "LDI"]:
        cands = _grid_candidates(method)
        jobs = [{"cfg": cfg, "cfg_key": CFG_KEY[method], "params": p,
                 "method": method, "cand_id": f"{method}_{i}", "seed": s}
                for i, p in enumerate(cands) for s in VAL_SEEDS]
        rows = _run_jobs(jobs, args.parallel)
        all_rows += rows
        df = pd.DataFrame(rows)
        agg = df.groupby("cand_id").agg(
            HQR=("HQR", "mean"), utility=("platform_utility", "mean"),
            payment=("cumulative_payment", "mean"),
            assigned=("num_assigned", "mean")).reset_index()
        agg = agg.sort_values(["HQR", "utility"], ascending=False)
        best_id = agg.iloc[0]["cand_id"]
        winners[method] = cands[int(best_id.split("_")[-1])]
        print(f"[same-budget] {method}: best {winners[method]} "
              f"HQR={agg.iloc[0]['HQR']:.4f} pay={agg.iloc[0]['payment']:.0f}")

    same_budget_cfg = {"baselines": {CFG_KEY[m]: winners[m] for m in winners}}
    with open(tuned_dir / "same_budget.yaml", "w", encoding="utf-8") as fh:
        yaml.dump(same_budget_cfg, fh)

    # ── Quality-matched: incentive_scale sweep on winners ────
    qm_result: dict[str, dict] = {}
    if args.protocol in ("quality_matched", "both"):
        for method in ["FR", "DP", "RAI", "LDI"]:
            jobs = []
            for i, scale in enumerate(SCALE_GRID):
                params = dict(winners[method], incentive_scale=scale)
                jobs += [{"cfg": cfg, "cfg_key": CFG_KEY[method], "params": params,
                          "method": method, "cand_id": f"{method}_qm_{i}", "seed": s}
                         for s in VAL_SEEDS]
            rows = _run_jobs(jobs, args.parallel)
            all_rows += rows
            df = pd.DataFrame(rows)
            agg = df.groupby("cand_id").agg(
                HQR=("HQR", "mean"), payment=("cumulative_payment", "mean")).reset_index()
            agg["scale"] = agg["cand_id"].str.split("_").str[-1].astype(int) \
                .map(dict(enumerate(SCALE_GRID)))
            band = agg[(agg["HQR"] - target).abs() <= 0.01]
            if len(band):
                pick = band.sort_values("payment").iloc[0]
                qm_result[method] = dict(winners[method],
                                         incentive_scale=float(pick["scale"]))
                print(f"[quality-matched] {method}: scale={pick['scale']} "
                      f"HQR={pick['HQR']:.4f} pay={pick['payment']:.0f}")
            else:
                closest = agg.iloc[(agg["HQR"] - target).abs().argmin()]
                qm_result[method] = {"quality_infeasible": True,
                                     "closest_scale": float(closest["scale"]),
                                     "closest_HQR": float(closest["HQR"])}
                print(f"[quality-matched] {method}: INFEASIBLE "
                      f"(closest HQR={closest['HQR']:.4f} at scale={closest['scale']})")

        qm_baselines = {CFG_KEY[m]: qm_result[m] for m in qm_result
                        if not qm_result[m].get("quality_infeasible")}
        qm_cfg = {"baselines": qm_baselines}
        with open(tuned_dir / "quality_matched.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(qm_cfg, fh)
        # Full report (incl. infeasible baselines) kept separately so the
        # tuned config stays directly include-able by run_sweep.
        report = {"quality_target": float(target), "tolerance": 0.01,
                  "results": {m: qm_result[m] for m in qm_result}}
        with open(out_dir / "quality_matched_report.json", "w", encoding="utf-8") as fh:
            import json
            json.dump(report, fh, indent=2)

    flat = []
    for r in all_rows:
        row = {k: v for k, v in r.items() if k != "params"}
        row["params"] = str(r["params"])
        flat.append(row)
    pd.DataFrame(flat).to_csv(out_dir / "tuning_results.csv", index=False)
    print(f"\nSaved: {out_dir / 'tuning_results.csv'}, {tuned_dir}/same_budget.yaml"
          + (", quality_matched.yaml" if qm_result else ""))


if __name__ == "__main__":
    main()
