"""Paired-seed E13 summary against each reserve profile's control arm.

This post-processing step keeps the experimental runs unchanged.  For every
scenario/profile pair it joins the 30 scenario seeds to the 30 control seeds
from the same reserve profile, then bootstraps the paired seed differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def paired_bootstrap_ci(values: np.ndarray, seed: int,
                        draws: int = 20_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["service_coverage", "pps", "legal_edge_rate",
               "reserve_activation_rate", "qos_violation_rate",
               "target_miss_rate", "under_incentive_rate"]
    keys = ["reserve_mode", "reserve_multiplier"]
    controls = frame[frame["variant"] == "control"]
    rows: list[dict] = []
    grouped = frame[frame["variant"] != "control"].groupby(
        ["variant", *keys], sort=True)
    for row_no, ((variant, mode, multiplier), scenario) in enumerate(grouped, 1):
        control = controls[
            (controls["reserve_mode"] == mode)
            & (controls["reserve_multiplier"] == multiplier)
        ]
        paired = scenario.merge(control, on="seed", suffixes=("", "_control"),
                                validate="one_to_one")
        if len(paired) != len(scenario):
            raise ValueError(f"incomplete control pairing for {variant}/{mode}/{multiplier}")
        row = {
            "variant": variant,
            "factor": scenario["factor"].iloc[0],
            "level": float(scenario["level"].iloc[0]),
            "reserve_mode": mode,
            "reserve_multiplier": float(multiplier),
            "n_pairs": len(paired),
        }
        for metric_no, metric in enumerate(metrics, 1):
            scenario_values = paired[metric].to_numpy(float)
            control_values = paired[f"{metric}_control"].to_numpy(float)
            delta = scenario_values - control_values
            lo, hi = paired_bootstrap_ci(delta, seed=row_no * 100 + metric_no)
            row[f"mean_{metric}"] = float(scenario_values.mean())
            row[f"control_{metric}"] = float(control_values.mean())
            row[f"paired_delta_{metric}"] = float(delta.mean())
            row[f"paired_delta_{metric}_ci_low"] = lo
            row[f"paired_delta_{metric}_ci_high"] = hi
        row["safety_pass"] = bool(
            row["mean_qos_violation_rate"] == 0.0
            and row["mean_target_miss_rate"] == 0.0
            and row["mean_under_incentive_rate"] == 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decomposition", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    summary = summarize(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    headline = summary[
        summary["variant"].isin(["response_error_+0.2", "response_error_+0.3",
                                  "joint_worst"])
    ].copy()
    keep = ["variant", "reserve_mode", "reserve_multiplier", "n_pairs",
            "mean_service_coverage", "control_service_coverage",
            "paired_delta_service_coverage",
            "paired_delta_service_coverage_ci_low",
            "paired_delta_service_coverage_ci_high", "mean_pps",
            "paired_delta_pps", "paired_delta_pps_ci_low",
            "paired_delta_pps_ci_high", "mean_reserve_activation_rate",
            "safety_pass"]
    headline[keep].to_csv(args.decomposition, index=False)
    print(headline[keep].to_string(index=False))


if __name__ == "__main__":
    main()
