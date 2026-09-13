from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.public_dynamics import (
    calibration_only_scale,
    load_netease_response_drivers,
    load_netease_stratified_response_drivers,
    resample_trace,
)


def test_calibration_only_scale_never_reads_test_quantiles():
    values = np.r_[np.linspace(0.2, 0.8, 20), np.full(80, 10.0)]
    scaled, meta = calibration_only_scale(values)
    assert meta["n_calibration"] == 20
    assert meta["test_above_calibration_q95"] == 80
    assert np.allclose(scaled, 1.30)


def test_resample_trace_preserves_endpoints():
    out = resample_trace([0.7, 0.9, 1.3], 7)
    assert len(out) == 7
    assert out[0] == 0.7
    assert out[-1] == 1.3


def test_netease_selection_uses_activity_not_accuracy(tmp_path: Path):
    rows = []
    for worker, count, answer in [(10, 100, 0), (20, 120, 1), (30, 110, 0)]:
        for index in range(count):
            rows.append({"workerId": worker, "answer": answer, "truth": 1,
                         "completeTime": index, "capability": 1})
    path = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    traces, manifest = load_netease_response_drivers(path, seeds=2, block_size=4)
    assert set(traces) == {1, 2}
    assert manifest[0]["completion_count"] == 120
    assert manifest[1]["completion_count"] == 110
    assert all(row["observed_completion_only"] for row in manifest)


def test_netease_stratified_selection_is_balanced_and_pseudonymous(tmp_path: Path):
    rows = []
    for worker in range(12):
        count = 40 + 8 * worker
        for index in range(count):
            rows.append({
                "workerId": f"worker-{worker}",
                "answer": (worker + index) % 2,
                "truth": index % 2,
                "completeTime": index,
                "capability": worker % 3,
            })
    path = tmp_path / "stratified.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    traces, manifest = load_netease_stratified_response_drivers(
        path, per_stratum=2, min_completions=40, block_size=4)
    assert len(traces) == 6
    assert pd.Series([row["activity_stratum"] for row in manifest]).value_counts().to_dict() == {
        "low": 2, "medium": 2, "high": 2}
    assert all("worker_id" not in row for row in manifest)
    assert all(len(row["source_token"]) == 16 for row in manifest)
    assert all(row["eligible_worker_count"] == 12 for row in manifest)
