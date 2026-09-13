"""Preprocessing for E25 public-participant dynamics.

The functions in this module deliberately expose only normalized, aggregate
time-series drivers.  ExtraSensory contributes observed sensor/network
availability; NetEaseCrowd contributes time-ordered answer correctness among
completed annotations.  Neither source contains PASI bids, private costs, or
randomized payments, and this module must not label its outputs as such.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXTRA_MODALITY_COLUMNS = (
    "raw_acc:magnitude_stats:mean",
    "proc_gyro:magnitude_stats:mean",
    "raw_magnet:magnitude_stats:mean",
    "watch_acceleration:magnitude_stats:mean",
    "watch_heading:mean_cos",
    "location_quick_features:std_lat",
    "audio_naive:mfcc0:mean",
)
EXTRA_NETWORK_COLUMNS = (
    "discrete:wifi_status:is_reachable_via_wifi",
    "discrete:wifi_status:is_reachable_via_wwan",
    "discrete:wifi_status:is_not_reachable",
    "discrete:wifi_status:missing",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def calibration_only_scale(
    values: Iterable[float],
    calibration_fraction: float = 0.20,
    lower: float = 0.70,
    upper: float = 1.30,
) -> tuple[np.ndarray, dict]:
    """Scale the held-out suffix using quantiles from the prefix only."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) < 10:
        raise ValueError("a public dynamics trace needs at least 10 finite values")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0,1)")
    split = max(1, min(len(array) - 1, int(np.floor(len(array) * calibration_fraction))))
    calibration = array[:split]
    test = array[split:]
    q05, q95 = np.quantile(calibration, [0.05, 0.95])
    if q95 - q05 <= 1e-12:
        q05, q95 = float(np.min(calibration)), float(np.max(calibration))
    if q95 - q05 <= 1e-12:
        scaled = np.ones(len(test), dtype=float)
        below = above = 0
        degenerate = True
    else:
        unit = (test - q05) / (q95 - q05)
        below = int(np.sum(unit < 0.0))
        above = int(np.sum(unit > 1.0))
        scaled = lower + (upper - lower) * np.clip(unit, 0.0, 1.0)
        degenerate = False
    return scaled.astype(float), {
        "n_total": int(len(array)),
        "n_calibration": int(len(calibration)),
        "n_test": int(len(test)),
        "calibration_q05": float(q05),
        "calibration_q95": float(q95),
        "test_below_calibration_q05": below,
        "test_above_calibration_q95": above,
        "degenerate_calibration": degenerate,
        "scaled_lower": float(lower),
        "scaled_upper": float(upper),
    }


def resample_trace(values: Iterable[float], n_slots: int) -> np.ndarray:
    """Time-warp a frozen trace to a requested number of slots."""
    array = np.asarray(list(values), dtype=float)
    if n_slots < 1 or len(array) < 1:
        raise ValueError("trace and slot count must both be non-empty")
    index = np.floor(np.linspace(0, len(array) - 1, n_slots)).astype(int)
    return array[index]


def load_extrasensory_drivers(root: Path, seeds: int = 30) -> tuple[dict[int, dict], list[dict]]:
    """Return per-user observed availability/network drivers.

    Selection is outcome-independent: files are ordered by SHA-256 of their
    anonymized filename and the first ``seeds`` are used.
    """
    files = sorted(root.glob("*.csv.gz"), key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
    if len(files) < seeds:
        raise FileNotFoundError(f"need {seeds} ExtraSensory user files, found {len(files)}")
    selection_rule = (
        f"all_{seeds}_available_files_ordered_by_sha256_of_anonymized_filename"
        if len(files) == seeds
        else f"first_{seeds}_by_sha256_of_anonymized_filename"
    )
    usecols = ["timestamp", *EXTRA_MODALITY_COLUMNS, *EXTRA_NETWORK_COLUMNS]
    traces: dict[int, dict] = {}
    manifest: list[dict] = []
    for seed, path in enumerate(files[:seeds], 1):
        frame = pd.read_csv(path, usecols=usecols).sort_values("timestamp")
        frame = frame.drop_duplicates("timestamp", keep="last")
        sensor_yield = frame.loc[:, EXTRA_MODALITY_COLUMNS].notna().mean(axis=1).to_numpy(float)
        wifi = frame[EXTRA_NETWORK_COLUMNS[0]].fillna(0.0).to_numpy(float)
        wwan = frame[EXTRA_NETWORK_COLUMNS[1]].fillna(0.0).to_numpy(float)
        offline = frame[EXTRA_NETWORK_COLUMNS[2]].fillna(0.0).to_numpy(float)
        network_raw = np.where(wifi >= 0.5, 1.0,
                      np.where(wwan >= 0.5, 0.75,
                      np.where(offline >= 0.5, 0.25, 0.50)))
        availability, availability_meta = calibration_only_scale(sensor_yield)
        network, network_meta = calibration_only_scale(network_raw)
        token = _source_token(f"ExtraSensory:{path.name}")
        traces[seed] = {
            "availability_ratio": availability,
            "network_ratio": network,
            "source_token": token,
        }
        manifest.append({
            "dataset": "ExtraSensory",
            "seed": seed,
            "source_token": token,
            "source_file_sha256": sha256_file(path),
            "selection_rule": selection_rule,
            **{f"availability_{key}": value for key, value in availability_meta.items()},
            **{f"network_{key}": value for key, value in network_meta.items()},
        })
    return traces, manifest


def load_netease_response_drivers(
    csv_path: Path,
    seeds: int = 30,
    block_size: int = 8,
) -> tuple[dict[int, dict], list[dict]]:
    """Return real completed-worker accuracy dynamics from NetEaseCrowd.

    The most active workers are used to guarantee enough longitudinal data;
    this convenience-sample bias is recorded in the manifest.  Accuracy is
    aggregated in non-overlapping completion blocks before the 20/80 split.
    """
    frame = pd.read_csv(csv_path)
    required = {"workerId", "answer", "truth", "completeTime", "capability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"NetEaseCrowd columns missing: {sorted(missing)}")
    frame = frame.copy()
    frame["correct"] = (frame["answer"] == frame["truth"]).astype(float)
    counts = frame.groupby("workerId").size().rename("n").reset_index()
    selected = counts.sort_values(["n", "workerId"], ascending=[False, True]).head(seeds)
    if len(selected) < seeds:
        raise ValueError(f"need {seeds} NetEaseCrowd workers, found {len(selected)}")
    traces: dict[int, dict] = {}
    manifest: list[dict] = []
    for seed, row in enumerate(selected.itertuples(index=False), 1):
        worker_id = row.workerId
        worker = frame[frame["workerId"] == worker_id].sort_values("completeTime")
        block = np.arange(len(worker), dtype=int) // int(block_size)
        accuracy = worker.groupby(block, sort=True)["correct"].mean().to_numpy(float)
        response, response_meta = calibration_only_scale(accuracy)
        token = _source_token(f"NetEaseCrowd:{worker_id}")
        traces[seed] = {"response_ratio": response, "source_token": token}
        manifest.append({
            "dataset": "NetEaseCrowd",
            "seed": seed,
            "source_token": token,
            "source_file_sha256": sha256_file(csv_path),
            "selection_rule": "top_30_by_completion_count_then_worker_id",
            "completion_count": int(len(worker)),
            "capability_count": int(worker["capability"].nunique()),
            "block_size": int(block_size),
            "observed_completion_only": True,
            **{f"response_{key}": value for key, value in response_meta.items()},
        })
    return traces, manifest


def load_netease_stratified_response_drivers(
    csv_path: Path,
    per_stratum: int = 30,
    min_completions: int = 80,
    block_size: int = 8,
) -> tuple[dict[int, dict], list[dict]]:
    """Return an activity-stratified NetEaseCrowd worker panel.

    Workers with at least ``min_completions`` completed annotations are sorted
    by completion count and split into three equal-rank activity strata.  The
    requested number is then selected within each stratum by a stable hash of
    the worker identifier.  Selection never uses answers, truth, capability,
    or derived accuracy.  The returned manifest contains only pseudonymous
    tokens, not raw worker identifiers.
    """
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    if min_completions < 10:
        raise ValueError("min_completions must be at least 10")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    frame = pd.read_csv(csv_path)
    required = {"workerId", "answer", "truth", "completeTime", "capability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"NetEaseCrowd columns missing: {sorted(missing)}")
    frame = frame.copy()
    frame["correct"] = (frame["answer"] == frame["truth"]).astype(float)
    counts = frame.groupby("workerId").size().rename("n").reset_index()
    eligible = counts[counts["n"] >= int(min_completions)].copy()
    if len(eligible) < 3 * per_stratum:
        raise ValueError(
            f"need at least {3 * per_stratum} eligible workers, found {len(eligible)}")

    eligible = eligible.sort_values(["n", "workerId"], ascending=[True, True]).reset_index(drop=True)
    stratum_indices = np.array_split(np.arange(len(eligible)), 3)
    strata = [eligible.iloc[index].copy() for index in stratum_indices]
    labels = ("low", "medium", "high")
    selected_frames: list[pd.DataFrame] = []
    for label, stratum in zip(labels, strata):
        if len(stratum) < per_stratum:
            raise ValueError(f"activity stratum {label} has only {len(stratum)} workers")
        stratum = stratum.copy()
        stratum["activity_stratum"] = label
        stratum["selection_hash"] = stratum["workerId"].map(
            lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest())
        selected_frames.append(stratum.sort_values("selection_hash").head(per_stratum))
    selected = pd.concat(selected_frames, ignore_index=True)

    traces: dict[int, dict] = {}
    manifest: list[dict] = []
    source_sha256 = sha256_file(csv_path)
    for source_index, row in enumerate(selected.itertuples(index=False), 1):
        worker_id = row.workerId
        worker = frame[frame["workerId"] == worker_id].sort_values("completeTime")
        block = np.arange(len(worker), dtype=int) // int(block_size)
        accuracy = worker.groupby(block, sort=True)["correct"].mean().to_numpy(float)
        response, response_meta = calibration_only_scale(accuracy)
        token = _source_token(f"NetEaseCrowd:{worker_id}")
        traces[source_index] = {
            "response_ratio": response,
            "source_token": token,
            "activity_stratum": row.activity_stratum,
        }
        manifest.append({
            "dataset": "NetEaseCrowd",
            "source_index": source_index,
            "source_token": token,
            "source_file_sha256": source_sha256,
            "selection_rule": (
                "completion_count_ge_threshold_then_equal_rank_activity_tertiles_"
                "then_sha256_of_worker_id_within_stratum"),
            "eligible_worker_count": int(len(eligible)),
            "activity_stratum": row.activity_stratum,
            "activity_stratum_population": int(len(strata[labels.index(row.activity_stratum)])),
            "completion_count": int(len(worker)),
            "capability_count": int(worker["capability"].nunique()),
            "min_completions": int(min_completions),
            "block_size": int(block_size),
            "observed_completion_only": True,
            **{f"response_{key}": value for key, value in response_meta.items()},
        })
    return traces, manifest
