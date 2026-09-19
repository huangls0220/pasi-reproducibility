#!/usr/bin/env python3
"""Aggregate experiment results across variations × methods × seeds.

Reads results/<experiment>/<dataset>/<variation>/<method>/seed_*/summary.json
and writes:
  results/summaries/<experiment>_<dataset>_by_seed.csv
  results/summaries/<experiment>_<dataset>_summary.csv  (mean ± 95% CI)

--with-fairness additionally computes Jain fairness indices per run from
provider logs (assignment-count based) and pair logs (utility based,
when available).

Usage:
  python scripts/aggregate_results.py --experiment overall
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stats_utils import mean_ci  # noqa: E402

KEY_METRICS = ["HQR", "average_quality", "completion_ratio", "assignment_ratio",
               "cumulative_payment", "payment_per_assigned", "payment_per_HQ",
               "platform_utility", "provider_utility_mean", "budget_utilization",
               "maintenance_entries", "recultivation_events", "ir_violations",
               "target_violations", "mean_optimality_gap", "total_runtime",
               "jain_assignments", "jain_utilities"]


def _jain(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or (x * x).sum() < 1e-12:
        return float("nan")
    return float(x.sum() ** 2 / (len(x) * (x * x).sum()))


def collect(exp_dir: Path, with_fairness: bool) -> pd.DataFrame:
    rows = []
    for f in sorted(exp_dir.rglob("summary.json")):
        parts = f.relative_to(exp_dir).parts
        if len(parts) != 4:
            continue
        variation, method_dir, seed_dir = parts[0], parts[1], parts[2]
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        diag_path = f.parent / "diagnostics.json"
        if diag_path.exists():
            with open(diag_path, encoding="utf-8") as fh:
                s["diag_status"] = json.load(fh).get("status", "unknown")
        s["variation"] = variation
        s.setdefault("method", method_dir)

        if with_fairness:
            ppath = f.parent / "provider_log.parquet"
            if ppath.exists():
                pl = pd.read_parquet(ppath, columns=["provider_id", "assigned"])
                counts = pl.groupby("provider_id")["assigned"].sum().to_numpy()
                s["jain_assignments"] = _jain(counts)
            gpath = f.parent / "pair_log.parquet"
            if gpath.exists():
                pr = pd.read_parquet(
                    gpath, columns=["provider_id", "selected", "experienced_utility"])
                sel = pr[pr["selected"]]
                if len(sel):
                    util = sel.groupby("provider_id")["experienced_utility"].sum().to_numpy()
                    s["jain_utilities"] = _jain(util)
        rows.append(s)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (var, method), g in df.groupby(["variation", "method"]):
        row = {"variation": var, "method": method, "n_seeds": len(g),
               "diag_failed": int((g.get("diag_status", "ok") != "ok").sum())
               if "diag_status" in g.columns else 0}
        for m in KEY_METRICS:
            if m not in g.columns:
                continue
            vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy()
            if len(vals):
                ci = mean_ci(vals)
                row[f"{m}_mean"] = ci["mean"]
                row[f"{m}_ci_lo"] = ci["ci_lower"]
                row[f"{m}_ci_hi"] = ci["ci_upper"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variation", "method"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate experiment results")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--with-fairness", action="store_true",
                    help="Compute Jain indices from provider/pair logs")
    args = ap.parse_args()

    exp_dir = ROOT / "results" / args.experiment / args.dataset
    if not exp_dir.exists():
        sys.exit(f"No results at {exp_dir}")

    df = collect(exp_dir, args.with_fairness)
    if df.empty:
        sys.exit("No summaries found")
    summary = summarize(df)

    out_dir = ROOT / "results" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{args.experiment}_{args.dataset}"
    df.to_csv(f"{base}_by_seed.csv", index=False)
    summary.to_csv(f"{base}_summary.csv", index=False)
    print(f"{len(df)} runs -> {base}_by_seed.csv")

    show = [c for c in ["variation", "method", "n_seeds", "diag_failed",
                        "HQR_mean", "average_quality_mean", "cumulative_payment_mean",
                        "platform_utility_mean"] if c in summary.columns]
    pd.set_option("display.width", 200)
    print(summary[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
