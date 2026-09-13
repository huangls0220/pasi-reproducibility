from pathlib import Path

import pandas as pd

from src.datasets.mobility import build_geolife_position_cache, positions_to_episode
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator
from scripts.run_a1_3_complete import legacy_cfg


def _write_plt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(["h"] * 6)
    rows = [
        "39.9000,116.3000,0,0,0,2009-02-14,00:00:00",
        "39.9010,116.3010,0,0,0,2009-02-14,00:05:00",
        "39.9020,116.3020,0,0,0,2009-02-14,00:10:00",
    ]
    path.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")


def test_geolife_fixture_and_episode(tmp_path):
    raw = tmp_path / "Data"
    _write_plt(raw / "000" / "Trajectory" / "20090214000000.plt")
    _write_plt(raw / "001" / "Trajectory" / "20090214000000.plt")
    cache = tmp_path / "restricted" / "positions.parquet"
    meta = build_geolife_position_cache(raw, cache)
    assert meta["providers"] == 2
    out = tmp_path / "episode"
    episode = positions_to_episode(cache, out, "medium", seed=1)
    assert episode["n_tasks"] > 0
    tasks = pd.read_parquet(out / "tasks.parquet")
    providers = pd.read_parquet(out / "providers.parquet")
    assert {"latitude", "longitude"}.issubset(tasks.columns)
    assert {"latitude", "longitude"}.issubset(providers.columns)
    assert episode["redistributable"] is False


def test_spatial_radius_rejects_all_distant_pairs(tmp_path):
    raw = tmp_path / "Data"
    _write_plt(raw / "000" / "Trajectory" / "20090214000000.plt")
    _write_plt(raw / "001" / "Trajectory" / "20090214000000.plt")
    cache = tmp_path / "restricted" / "positions.parquet"
    build_geolife_position_cache(raw, cache)
    out = tmp_path / "episode"
    positions_to_episode(cache, out, "medium", seed=1)
    tasks = pd.read_parquet(out / "tasks.parquet")
    tasks["latitude"] = 0.0
    tasks["longitude"] = 0.0
    tasks.to_parquet(out / "tasks.parquet", index=False)

    cfg = legacy_cfg(T=10, N=2, M=2)
    cfg["simulation"]["log_level"] = "full"
    cfg["dataset"] = {
        "processed_dir": str(out), "use_region": "all", "service_radius_km": 1.0,
    }
    data = load_trace_episode("geolife", cfg, seed=1, processed_dir=out)
    result = Simulator(cfg, data, method="PASI", seed=1).run()
    assert result["summary"]["num_assigned"] == 0
    assert (result["pair_log"]["infeasible_reason"] == "spatial_out_of_range").any()
