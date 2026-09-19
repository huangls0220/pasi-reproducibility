"""Mobility-trace preprocessing for PASI trace-driven experiments.

The trace controls provider presence and location.  Task attributes,
communication rates, compute load, capacity, and behavioural parameters are
seeded model quantities, not fields observed in GeoLife or T-Drive.

GeoLife is licensed for non-commercial use and forbids redistribution of the
data and derivative datasets.  Position caches and processed episodes created
by this module are local restricted artifacts and must not be packaged.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BEIJING_BBOX = (39.4, 41.1, 115.4, 117.6)  # lat_min, lat_max, lon_min, lon_max


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def build_geolife_position_cache(
    data_root: Path,
    out_file: Path,
    start: str = "2009-02-14",
    end: str = "2009-02-21",
    slot_minutes: int = 10,
    bbox: tuple[float, float, float, float] = BEIJING_BBOX,
    max_speed_kmh: float = 180.0,
) -> dict:
    """Create one median provider position per user and UTC slot."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    lat_min, lat_max, lon_min, lon_max = bbox
    frames = []
    files_read = 0
    raw_points = 0

    for user_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        trajectory_dir = user_dir / "Trajectory"
        if not trajectory_dir.exists():
            continue
        user_frames = []
        for path in sorted(trajectory_dir.glob("*.plt")):
            # Filenames encode trajectory start. Include the previous date for
            # a trajectory crossing into the frozen window.
            try:
                file_day = pd.Timestamp(path.stem[:8])
            except ValueError:
                continue
            if file_day < start_ts.tz_localize(None) - pd.Timedelta(days=1):
                continue
            if file_day >= end_ts.tz_localize(None):
                continue
            frame = pd.read_csv(
                path, skiprows=6, header=None, usecols=[0, 1, 5, 6],
                names=["latitude", "longitude", "date", "time"],
            )
            files_read += 1
            raw_points += len(frame)
            frame["timestamp"] = pd.to_datetime(
                frame["date"].astype(str) + " " + frame["time"].astype(str),
                utc=True, errors="coerce",
            )
            frame = frame.dropna(subset=["timestamp", "latitude", "longitude"])
            frame = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] < end_ts)]
            frame = frame[
                frame["latitude"].between(lat_min, lat_max)
                & frame["longitude"].between(lon_min, lon_max)
            ]
            if len(frame):
                user_frames.append(frame[["timestamp", "latitude", "longitude"]])
        if not user_frames:
            continue

        user = pd.concat(user_frames, ignore_index=True).sort_values("timestamp")
        dt_h = user["timestamp"].diff().dt.total_seconds().to_numpy() / 3600.0
        distance = _haversine_km(
            user["latitude"].shift().to_numpy(), user["longitude"].shift().to_numpy(),
            user["latitude"].to_numpy(), user["longitude"].to_numpy(),
        )
        speed = np.divide(distance, dt_h, out=np.zeros_like(distance), where=dt_h > 0)
        keep = ~np.isfinite(speed) | (speed <= max_speed_kmh)
        user = user[keep].copy()
        user["provider_id"] = f"G{user_dir.name}"
        frames.append(user)

    if not frames:
        raise ValueError("No GeoLife points survived the frozen window and filters")

    points = pd.concat(frames, ignore_index=True)
    slot_seconds = slot_minutes * 60
    points["slot"] = ((points["timestamp"] - start_ts).dt.total_seconds()
                      // slot_seconds).astype(int)
    positions = (points.groupby(["provider_id", "slot"], as_index=False)
                 .agg(latitude=("latitude", "median"),
                      longitude=("longitude", "median"),
                      observed_points=("timestamp", "size")))
    positions["timestamp"] = start_ts + pd.to_timedelta(
        positions["slot"] * slot_seconds, unit="s"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    positions.to_parquet(out_file, index=False)

    meta = {
        "source": "GeoLife GPS Trajectories 1.3",
        "window_utc": [start, end],
        "slot_minutes": slot_minutes,
        "bbox": list(bbox),
        "max_speed_kmh": max_speed_kmh,
        "files_read": files_read,
        "raw_points_in_candidate_files": raw_points,
        "positions": int(len(positions)),
        "providers": int(positions["provider_id"].nunique()),
        "active_slots": int(positions["slot"].nunique()),
        "license": "Microsoft Research License Agreement; non-commercial; no redistribution",
        "redistributable": False,
    }
    (out_file.parent / "position_cache_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def positions_to_episode(
    positions_file: Path,
    out_dir: Path,
    workload: str,
    seed: int,
    service_radius_km: float = 3.0,
) -> dict:
    """Generate one paired event tape from a frozen mobility position cache."""
    intensity = {"low": 0.35, "medium": 0.70, "high": 1.05}[workload]
    rng = np.random.default_rng([seed, 5150])
    positions = pd.read_parquet(positions_file)
    slots = sorted(positions["slot"].unique())

    provider_rows = []
    task_rows = []
    task_counter = 0
    for slot in slots:
        active = positions[positions["slot"] == slot].copy()
        if len(active) < 2:
            continue
        n_tasks = max(1, int(round(intensity * len(active))))
        anchors = active.iloc[rng.integers(0, len(active), n_tasks)]
        # Small spatial jitter avoids placing every task exactly on a provider.
        task_lat = anchors["latitude"].to_numpy() + rng.normal(0, 0.004, n_tasks)
        task_lon = anchors["longitude"].to_numpy() + rng.normal(0, 0.005, n_tasks)
        for idx in range(n_tasks):
            L = rng.uniform(0.1, 1.0)
            rho = rng.uniform(1.2, 2.5)
            task_rows.append({
                "episode_id": f"geolife_{workload}_s{seed}",
                "slot": int(slot), "task_id": f"T{task_counter:07d}",
                "latitude": float(task_lat[idx]), "longitude": float(task_lon[idx]),
                "L": L, "input_size": rng.uniform(0.1, 2.0),
                "output_size": rng.uniform(0.05, 1.0), "deadline": rho,
                "min_quality": rng.uniform(0.65, 0.85),
                "q_bar": rng.uniform(0.90, 1.00),
                "kappa": np.clip(2.5 / (L + 0.5) * rng.uniform(0.8, 1.2), 1.0, 5.0),
                "value": 1.0 + 3.0 * L + 1.0 / rho, "raw_duration": 600.0,
            })
            task_counter += 1
        for row in active.itertuples(index=False):
            provider_rows.append({
                "episode_id": f"geolife_{workload}_s{seed}",
                "slot": int(slot), "provider_id": row.provider_id,
                "latitude": float(row.latitude), "longitude": float(row.longitude),
                "load_ratio": rng.uniform(0.0, 0.5),
                "communication_rate": rng.uniform(5.0, 20.0),
                "availability_duration": rng.uniform(2.0, 10.0),
            })

    tasks = pd.DataFrame(task_rows)
    providers = pd.DataFrame(provider_rows).drop_duplicates(["slot", "provider_id"])
    provider_ids = sorted(providers["provider_id"].unique())
    static = pd.DataFrame({
        "episode_id": f"geolife_{workload}_s{seed}",
        "provider_id": provider_ids,
        "max_processing_rate": rng.uniform(5.0, 20.0, len(provider_ids)),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(out_dir / "tasks.parquet", index=False)
    providers.to_parquet(out_dir / "providers.parquet", index=False)
    static.to_parquet(out_dir / "provider_static.parquet", index=False)
    meta = {
        "source": "geolife",
        "trace_driven_fields": ["provider_id", "slot", "latitude", "longitude"],
        "model_generated_fields": [
            "tasks", "load_ratio", "communication_rate", "availability_duration",
            "max_processing_rate", "task_attributes", "behavioural_parameters",
        ],
        "workload": workload, "seed": seed,
        "service_radius_km": service_radius_km,
        "T": int(tasks["slot"].nunique()),
        "n_tasks": int(len(tasks)), "n_providers": int(len(provider_ids)),
        "n_provider_slot_rows": int(len(providers)),
        "position_cache_sha256": sha256(positions_file),
        "redistributable": False,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
