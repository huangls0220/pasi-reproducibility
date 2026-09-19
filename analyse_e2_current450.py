#!/usr/bin/env python3
"""Add paired bootstrap intervals and compact variant summaries to E2."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "e2_current450"
VARIANTS = (
    "NO_POSITIVE_ACCUMULATION",
    "NO_NEGATIVE_FEEDBACK",
    "FROZEN_STATE",
)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    paired = pd.read_csv(OUT / "paired_ablation_by_seed.csv")
    records: list[dict] = []
    for s_idx, (scenario, group) in enumerate(paired.groupby("scenario", sort=True)):
        for v_idx, variant in enumerate(VARIANTS):
            degradation = group[f"design_pps_degradation_pct__{variant}"].to_numpy(float)
            coverage = group[f"coverage_diff_pp__{variant}"].to_numpy(float)
            low, high = bootstrap_ci(degradation, seed=7200 + 10 * s_idx + v_idx)
            records.append({
                "scenario": scenario,
                "variant": variant,
                "n_pairs": len(group),
                "mean_design_pps_degradation_pct": float(degradation.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "positive_count": int((degradation > 0).sum()),
                "mean_coverage_diff_pp": float(coverage.mean()),
                "max_abs_coverage_diff_pp": float(np.abs(coverage).max()),
                "mean_selected_H": float(group[f"mean_H__{variant}"].mean()),
                "full_pasi_mean_selected_H": float(group["mean_H__FULL_PASI"].mean()),
            })
    pd.DataFrame(records).to_csv(OUT / "ablation_inference.csv", index=False)


if __name__ == "__main__":
    main()
