"""Trace adapter tests (Section 8 / 11).

Uses TINY FABRICATED raw files as PARSER FIXTURES ONLY — they exercise the
preprocessing/schema path and are never presented as experiment results
(instruction §C.2: no fake Google/Alibaba results).
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.datasets.trace_common import events_to_episode
from src.datasets.trace_loader import load_trace_episode


@pytest.fixture()
def processed_dir(tmp_path):
    """Build a small processed dataset from a synthetic event fixture."""
    rng = np.random.default_rng(0)
    n = 400
    start = rng.uniform(0, 3600 * 4, n)
    events = pd.DataFrame({
        "machine_id": rng.integers(0, 12, n).astype(str),
        "start_time": start,
        "end_time": start + rng.uniform(30, 900, n),
        "cpu_request": rng.uniform(0.1, 4.0, n),
    })
    ds_cfg = {"slot_seconds": 300, "max_slots": 60}
    out = tmp_path / "processed"
    meta = events_to_episode(events, ds_cfg, "fixture", {"fixture.csv": "n/a"}, out)
    assert meta["n_tasks"] > 0
    return out


class TestTraceCommon:
    def test_schema_files_written(self, processed_dir):
        for f in ["tasks.parquet", "providers.parquet",
                  "provider_static.parquet", "meta.json"]:
            assert (processed_dir / f).exists()

    def test_task_ranges(self, processed_dir):
        tasks = pd.read_parquet(processed_dir / "tasks.parquet")
        assert (tasks["L"] >= 0.1 - 1e-9).all() and (tasks["L"] <= 1.0 + 1e-9).all()
        assert (tasks["min_quality"] < tasks["q_bar"]).all()
        assert (tasks["deadline"] > 0).all()

    def test_meta_records_provenance(self, processed_dir):
        with open(processed_dir / "meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert "raw_checksums" in meta and "preprocess_seed" in meta


class TestTraceLoader:
    def test_load_and_behavior_attach(self, processed_dir):
        cfg = {"dataset": {"processed_dir": str(processed_dir), "use_region": "all"},
               "providers": {"behavioral_fraction": 0.7}}
        ep = load_trace_episode("fixture", cfg, seed=1, processed_dir=processed_dir)
        st = ep["provider_static"]
        for col in ["alpha", "beta", "zeta", "omega", "xi", "delta", "outside_option"]:
            assert col in st.columns
        # non-behavioural providers: omega=0, zeta=1
        nb = st[~st["behavioral"]]
        if len(nb):
            assert np.allclose(nb["omega"], 0.0) and np.allclose(nb["zeta"], 1.0)

    def test_same_seed_same_behavior(self, processed_dir):
        cfg = {"dataset": {"use_region": "all"}}
        a = load_trace_episode("fixture", cfg, seed=9, processed_dir=processed_dir)
        b = load_trace_episode("fixture", cfg, seed=9, processed_dir=processed_dir)
        pd.testing.assert_frame_equal(a["provider_static"], b["provider_static"])

    def test_missing_files_friendly_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="preprocess"):
            load_trace_episode("google", {}, seed=1, processed_dir=tmp_path / "nope")

    def test_region_split(self, processed_dir):
        cfg_v = {"dataset": {"use_region": "validation"}}
        cfg_t = {"dataset": {"use_region": "test"}}
        val = load_trace_episode("fixture", cfg_v, seed=1, processed_dir=processed_dir)
        tst = load_trace_episode("fixture", cfg_t, seed=1, processed_dir=processed_dir)
        assert len(val["tasks"]) + len(tst["tasks"]) > 0
        if len(tst["tasks"]):
            assert tst["tasks"]["slot"].min() == 0  # re-based
