"""E18: real mobile-sensing compensation schedule as a post-hoc anchor.

TILES-2018 reports actual participant compensation for a ten-week mobile and
wearable sensing study.  The schedule is preserved verbatim at the category
level.  The $1 value of a ten-point activity follows exactly from each listed
weekly threshold (100/$10, 150/$15, 200/$20, 250/$25).  Mapping it to the
paper's normalized unit-effort cost is a sensitivity conversion, not a claim
that TILES reports bids, PASI costs, or per-service payments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MEDIAN_NORMALIZED_UNIT_EFFORT_COST = 0.09300718055798415
SOURCE_DOI = "10.1038/s41597-020-00655-3"
SOURCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7567859/"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stage_schedule = pd.DataFrame([
        ("Enrollment session", "fixed stage payment", 75.0),
        ("Baseline survey (part II)", "fixed stage payment", 75.0),
        ("Data collection", "weekly maximum", 25.0),
        ("Post-study survey", "fixed stage payment", 75.0),
    ], columns=["activity", "payment_type", "usd"])
    weekly = pd.DataFrame([
        (100, 10.0), (150, 15.0), (200, 20.0), (250, 25.0),
    ], columns=["minimum_points", "gift_card_usd"])
    weekly["usd_per_point"] = weekly["gift_card_usd"] / weekly["minimum_points"]
    if not np.allclose(weekly["usd_per_point"], 0.1, atol=1e-12):
        raise AssertionError("TILES weekly cutoffs are not linear at USD 0.10/point")
    ten_point_activity_usd = 1.0
    scale = ten_point_activity_usd / MEDIAN_NORMALIZED_UNIT_EFFORT_COST

    seeds = pd.read_csv(args.seed_results)
    required = {"workload", "method", "seed", "payment", "pps",
                "service_coverage"}
    if missing := required - set(seeds.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    mapped = seeds.copy()
    mapped["anchor"] = "TILES-2018 ten-point sensing/survey activity"
    mapped["usd_per_normalized_unit"] = scale
    mapped["payment_usd_sensitivity"] = mapped["payment"] * scale
    mapped["pps_usd_sensitivity"] = mapped["pps"] * scale
    mapped.to_csv(args.out / "e18_tiles_mapped_seed_results.csv", index=False)
    summary = (mapped.groupby(["workload", "method"], as_index=False)
               .agg(mean_pps_usd_sensitivity=("pps_usd_sensitivity", "mean"),
                    mean_service_coverage=("service_coverage", "mean"),
                    n_runs=("seed", "size")))
    summary.to_csv(args.out / "e18_tiles_anchor_summary.csv", index=False)
    stage_schedule.to_csv(args.out / "e18_tiles_stage_schedule.csv", index=False)
    weekly.to_csv(args.out / "e18_tiles_weekly_cutoffs.csv", index=False)
    metadata = {
        "experiment": "E18 TILES-2018 real mobile-sensing compensation anchor",
        "source_doi": SOURCE_DOI, "source_url": SOURCE_URL,
        "study_scope": "212 hospital workers; 10-week mobile/wearable sensing study",
        "observed_or_reported_quantity": "published participant compensation schedule",
        "not_claimed": ["participant bids", "provider private costs",
                        "record-level payment observations", "PASI field deployment"],
        "weekly_cutoff_identity": "USD 0.10 per point at every published cutoff",
        "ten_point_activity_usd": ten_point_activity_usd,
        "median_normalized_unit_effort_cost": MEDIAN_NORMALIZED_UNIT_EFFORT_COST,
        "mapping_type": "post-hoc schedule sensitivity; assignments unchanged",
        "seed_results_sha256": sha256(args.seed_results),
        "script_sha256": sha256(Path(__file__)),
    }
    (args.out / "e18_tiles_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(stage_schedule.to_string(index=False))
    print(weekly.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
