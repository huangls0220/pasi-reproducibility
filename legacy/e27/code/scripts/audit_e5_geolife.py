"""Integrity and paired-statistics audit for the completed GeoLife E5 matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
RESULTS = PROJECT / "results" / "geolife"
RESTRICTED = PROJECT / "restricted" / "geolife" / "episodes"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 20000):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    frame = pd.read_csv(RESULTS / "e5_geolife_seed_results.csv")
    expected = pd.MultiIndex.from_product(
        [["low", "medium", "high"], range(1, 31), ["MOI", "PASI"]],
        names=["workload", "seed", "method"],
    )
    actual = pd.MultiIndex.from_frame(frame[["workload", "seed", "method"]])
    missing = expected.difference(actual)
    extra = actual.difference(expected)

    registry = []
    for workload in ("low", "medium", "high"):
        for seed in range(1, 31):
            directory = RESTRICTED / workload / f"seed_{seed:03d}"
            registry.append({
                "dataset": "geolife", "workload": workload, "seed": seed,
                "tasks_sha256": digest(directory / "tasks.parquet"),
                "providers_sha256": digest(directory / "providers.parquet"),
                "provider_static_sha256": digest(directory / "provider_static.parquet"),
                "meta_sha256": digest(directory / "meta.json"),
            })
    pd.DataFrame(registry).to_csv(RESULTS / "e5_geolife_event_tape_registry.csv", index=False)

    stats_rows = []
    for index, workload in enumerate(("low", "medium", "high"), start=1):
        sub = frame[frame["workload"] == workload]
        moi = sub[sub["method"] == "MOI"].sort_values("seed")
        pasi = sub[sub["method"] == "PASI"].sort_values("seed")
        pps_delta = pasi["pps"].to_numpy() - moi["pps"].to_numpy()
        relative_saving = 100.0 * (-pps_delta) / moi["pps"].to_numpy()
        coverage_delta = (pasi["service_coverage"].to_numpy()
                          - moi["service_coverage"].to_numpy())
        qos_delta = (pasi["qos_violation_rate"].to_numpy()
                     - moi["qos_violation_rate"].to_numpy())
        test = stats.ttest_rel(pasi["pps"], moi["pps"])
        stats_rows.append({
            "dataset": "geolife", "workload": workload, "n_pairs": len(moi),
            "relative_pps_saving_mean_pct": relative_saving.mean(),
            "relative_pps_saving_ci_low_pct": bootstrap_ci(relative_saving, 8140 + index)[0],
            "relative_pps_saving_ci_high_pct": bootstrap_ci(relative_saving, 8140 + index)[1],
            "relative_pps_saving_min_pct": relative_saving.min(),
            "relative_pps_saving_max_pct": relative_saving.max(),
            "paired_t": test.statistic, "paired_t_pvalue": test.pvalue,
            "cohens_dz_for_pasi_minus_moi": pps_delta.mean() / pps_delta.std(ddof=1),
            "pps_improved_count": int((pps_delta < 0).sum()),
            "coverage_delta_mean": coverage_delta.mean(),
            "coverage_delta_ci_low": bootstrap_ci(coverage_delta, 8240 + index)[0],
            "coverage_delta_ci_high": bootstrap_ci(coverage_delta, 8240 + index)[1],
            "qos_delta_mean": qos_delta.mean(),
            "qos_delta_ci_low": bootstrap_ci(qos_delta, 8340 + index)[0],
            "qos_delta_ci_high": bootstrap_ci(qos_delta, 8340 + index)[1],
        })
    pd.DataFrame(stats_rows).to_csv(
        RESULTS / "e5_geolife_paired_statistics.csv", index=False
    )

    audit = {
        "expected_rows": len(expected), "actual_rows": len(frame),
        "duplicate_keys": int(frame.duplicated(["workload", "seed", "method"]).sum()),
        "missing_keys": [list(x) for x in missing], "extra_keys": [list(x) for x in extra],
        "nan_cells": int(frame.isna().sum().sum()),
        "non_ok_runs": int((~frame["run_status"].isin(["ok", "completed"])).sum()),
        "ir_violations": int(frame["ir_violations"].sum()),
        "target_violations": int(frame["target_violations"].sum()),
        "paired_coverage_mismatches": int(sum(
            not np.array_equal(
                frame[(frame.workload == w) & (frame.method == "MOI")]
                    .sort_values("seed")["service_coverage"].to_numpy(),
                frame[(frame.workload == w) & (frame.method == "PASI")]
                    .sort_values("seed")["service_coverage"].to_numpy(),
            ) for w in ("low", "medium", "high")
        )),
        "event_tapes_registered": len(registry),
        "pass": bool(len(frame) == len(expected) and not len(missing) and not len(extra)
                     and not frame.duplicated(["workload", "seed", "method"]).any()
                     and not frame.isna().any().any()
                     and frame["run_status"].isin(["ok", "completed"]).all()),
    }
    (RESULTS / "e5_geolife_integrity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    print(pd.DataFrame(stats_rows).to_string(index=False))


if __name__ == "__main__":
    main()
