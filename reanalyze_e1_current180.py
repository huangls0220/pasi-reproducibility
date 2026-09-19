#!/usr/bin/env python3
"""Recompute E1 formation timing from retained aggregate slot metrics."""

from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent / "e1_current180"
slots = pd.read_parquet(OUT / "slot_aggregates.parquet")
paired = slots.pivot(
    index=["scenario", "seed", "slot"], columns="method",
    values=["num_assigned", "budget_used"],
)
paired.columns = [f"{metric}_{method}" for metric, method in paired.columns]
paired = paired.reset_index().sort_values(["scenario", "seed", "slot"])
rows = []
for (scenario, seed), group in paired.groupby(["scenario", "seed"], sort=True):
    pps_moi = group["budget_used_MOI"].cumsum() / group["num_assigned_MOI"].cumsum().clip(lower=1)
    pps_pasi = group["budget_used_PASI"].cumsum() / group["num_assigned_PASI"].cumsum().clip(lower=1)
    rate = 100.0 * (1.0 - pps_pasi / pps_moi)
    final = float(rate.iloc[-1])
    row = {"scenario": scenario, "seed": int(seed), "final_design_pps_saving_pct": final}
    for fraction in (0.5, 0.9):
        reached = group.loc[rate >= fraction * final, "slot"]
        row[f"T{int(fraction * 100)}"] = int(reached.iloc[0] + 1) if len(reached) else None
    rows.append(row)
timing = pd.DataFrame(rows)
timing.to_csv(OUT / "formation_timing_by_seed.csv", index=False)
summary = timing.groupby("scenario", as_index=False).agg(
    mean_T50=("T50", "mean"), mean_T90=("T90", "mean"),
    min_T90=("T90", "min"), max_T90=("T90", "max"),
)
summary.to_csv(OUT / "timing_summary.csv", index=False)
print(summary.to_string(index=False))
