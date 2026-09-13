#!/usr/bin/env python3
"""Statistical tests for an experiment (Section 14).

Pairs by (variation, seed); compares every method to the reference with
paired t-test (Shapiro-normal differences) or Wilcoxon signed-rank,
Holm-Bonferroni corrected, with effect sizes and 95% CIs.

Usage:
  python scripts/statistical_tests.py --experiment overall --reference PRIME
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stats_utils import compare_methods  # noqa: E402

METRICS = ["HQR", "average_quality", "completion_ratio", "cumulative_payment",
           "payment_per_assigned", "payment_per_HQ", "platform_utility",
           "provider_utility_mean", "budget_utilization"]


def load_summaries(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for f in exp_dir.rglob("summary.json"):
        parts = f.relative_to(exp_dir).parts  # variation/method/seed_x/summary.json
        if len(parts) != 4:
            continue
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        s["variation"], s["method_dir"] = parts[0], parts[1]
        rows.append(s)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run statistical tests")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--reference", default="PRIME")
    ap.add_argument("--variation", default=None,
                    help="Restrict to one variation (default: each separately)")
    args = ap.parse_args()

    exp_dir = ROOT / "results" / args.experiment / args.dataset
    if not exp_dir.exists():
        sys.exit(f"No results at {exp_dir}")
    df = load_summaries(exp_dir)
    if df.empty:
        sys.exit("No summaries found")

    out_dir = ROOT / "results" / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    variations = [args.variation] if args.variation else sorted(df["variation"].unique())
    for var in variations:
        sub = df[df["variation"] == var]
        methods = sorted(sub["method"].unique())
        if args.reference not in methods or len(methods) < 2:
            continue
        # Pair by seed: keep seeds present for ALL methods
        seeds_per = [set(sub[sub["method"] == m]["seed"]) for m in methods]
        common = sorted(set.intersection(*seeds_per))
        results = {}
        for m in methods:
            mrows = sub[(sub["method"] == m) & (sub["seed"].isin(common))] \
                .sort_values("seed")
            results[m] = {k: mrows[k].tolist() for k in METRICS if k in mrows.columns}
        rows = compare_methods(results, reference=args.reference)
        for r in rows:
            r["variation"] = var
            r["experiment"] = args.experiment
        all_rows += rows
        print(f"{var}: {len(rows)} comparisons over {len(common)} paired seeds")

    out = pd.DataFrame(all_rows)
    path = out_dir / f"{args.experiment}_tests.csv"
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} rows -> {path}")

    # Console digest: significant HQR/payment comparisons
    if len(out):
        key = out[out["metric"].isin(["HQR", "cumulative_payment", "platform_utility"])]
        for _, r in key.iterrows():
            sig = "***" if r["adjusted_p"] < 0.001 else \
                  "**" if r["adjusted_p"] < 0.01 else \
                  "*" if r["adjusted_p"] < 0.05 else "ns"
            print(f"  [{r['variation']}] {r['metric']}: {r['method_a']} vs "
                  f"{r['method_b']}: diff={r['mean_diff']:+.4g} "
                  f"[{r['ci_lower']:.4g},{r['ci_upper']:.4g}] "
                  f"p_adj={r['adjusted_p']:.2e} {sig} ({r['test_name']}, "
                  f"eff={r['effect_size']:.2f})")


if __name__ == "__main__":
    main()
