"""Shared trace preprocessing core (Sections 8.3–8.4).

Turns a normalized event table (one row per task instance with
[machine_id, start_time, end_time, cpu_request]) into the unified
episode schema used by src/datasets/trace_loader.py.

The mapping (§8.4):
    raw_L = cpu_request * duration
    L     = L_min + (L_max-L_min) * clip((raw_L - P5)/(P95 - P5), 0, 1)
    deadline    = rho * duration,  rho ~ U[1.2, 2.5]
    min_quality ~ U[0.65, 0.85];  q_bar ~ U[0.90, 1.00]
    kappa = clip(kappa_0 * capacity / difficulty, kappa_min, kappa_max)
    value = v0 + v1 * L_norm + v2 * urgency
All random mappings are driven by a fixed preprocessing seed recorded in
meta.json, so preprocessing is reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            if fh.tell() > max_bytes:
                h.update(b"__TRUNCATED__")
                break
    return h.hexdigest()


def events_to_episode(
    events: pd.DataFrame,
    ds_cfg: dict,
    source_name: str,
    raw_checksums: dict[str, str],
    out_dir: Path,
    seed: int = 20260701,
) -> dict:
    """events columns required: machine_id, start_time, end_time, cpu_request.

    Times in seconds.  Writes tasks/providers/provider_static parquet +
    meta.json into out_dir; returns meta.
    """
    rng = np.random.default_rng(seed)
    slot_s = float(ds_cfg.get("slot_seconds", 300))
    max_slots = int(ds_cfg.get("max_slots", 1000))

    ev = events.copy()
    ev = ev.dropna(subset=["machine_id", "start_time", "end_time", "cpu_request"])
    ev = ev[ev["end_time"] > ev["start_time"]]
    ev["duration"] = ev["end_time"] - ev["start_time"]

    qlo, qhi = ds_cfg.get("duration_clip_quantiles", [0.01, 0.99])
    lo, hi = ev["duration"].quantile([qlo, qhi])
    ev = ev[(ev["duration"] >= lo) & (ev["duration"] <= hi)]
    if len(ev) == 0:
        raise ValueError("No events survive cleaning — check raw files")

    t0 = float(ev["start_time"].min())
    ev["slot"] = ((ev["start_time"] - t0) / slot_s).astype(int)
    ev = ev[ev["slot"] < max_slots]

    # ── Robust L normalization (P5/P95) ──────────────────────
    raw_L = ev["cpu_request"].to_numpy() * ev["duration"].to_numpy()
    p5, p95 = np.percentile(raw_L, ds_cfg.get("L_norm_quantiles", [5, 95])
                            if max(ds_cfg.get("L_norm_quantiles", [0.05, 0.95])) > 1
                            else [q * 100 for q in ds_cfg.get("L_norm_quantiles", [0.05, 0.95])])
    L_min, L_max = ds_cfg.get("L_range", [0.1, 1.0])
    L = L_min + (L_max - L_min) * np.clip((raw_L - p5) / max(p95 - p5, 1e-12), 0, 1)

    n = len(ev)
    rho = rng.uniform(*ds_cfg.get("deadline_rho", [1.2, 2.5]), n)
    dur_norm = ev["duration"].to_numpy() / max(float(ev["duration"].median()), 1e-9)
    urgency = 1.0 / np.maximum(rho, 1e-9)
    vc = ds_cfg.get("value_coef", {"v0": 1.0, "v1": 3.0, "v2": 1.0})
    L_unit = (L - L_min) / max(L_max - L_min, 1e-12)

    kappa0 = float(ds_cfg.get("kappa_0", 2.5))
    kmin, kmax = ds_cfg.get("kappa_clip", [1.0, 5.0])
    difficulty = np.maximum(L_unit + 0.5, 0.5)
    kappa = np.clip(kappa0 / difficulty * rng.uniform(0.8, 1.2, n), kmin, kmax)

    tasks = pd.DataFrame({
        "episode_id": f"{source_name}_ep1",
        "slot": ev["slot"].to_numpy(),
        "task_id": [f"T{i:07d}" for i in range(n)],
        "L": L,
        "input_size": rng.uniform(0.1, 2.0, n),
        "output_size": rng.uniform(0.05, 1.0, n),
        "deadline": rho * np.clip(dur_norm, 0.3, 3.0),
        "min_quality": rng.uniform(*ds_cfg.get("min_quality", [0.65, 0.85]), n),
        "q_bar": rng.uniform(*ds_cfg.get("q_bar", [0.90, 1.00]), n),
        "kappa": kappa,
        "value": vc["v0"] + vc["v1"] * L_unit + vc["v2"] * urgency,
        "raw_duration": ev["duration"].to_numpy(),
        "raw_cpu_request": ev["cpu_request"].to_numpy(),
    })

    # ── Providers: machines online in slots where they run tasks ──
    machines = sorted(ev["machine_id"].astype(str).unique())
    mid_map = {m: f"P{i:05d}" for i, m in enumerate(machines)}
    ev["provider_id"] = ev["machine_id"].astype(str).map(mid_map)

    slot_load = (ev.groupby(["provider_id", "slot"])["cpu_request"].sum()
                 .clip(0, None).rename("load_raw").reset_index())
    p95_load = max(float(slot_load["load_raw"].quantile(0.95)), 1e-9)
    slot_load["load_ratio"] = (slot_load["load_raw"] / p95_load * 0.5).clip(0, 0.5)
    rng2 = np.random.default_rng(seed + 1)
    m = len(slot_load)
    providers = pd.DataFrame({
        "episode_id": f"{source_name}_ep1",
        "slot": slot_load["slot"],
        "provider_id": slot_load["provider_id"],
        "load_ratio": slot_load["load_ratio"],
        "communication_rate": rng2.uniform(5.0, 20.0, m),
        "availability_duration": rng2.uniform(2.0, 10.0, m),
    })

    static = pd.DataFrame({
        "episode_id": f"{source_name}_ep1",
        "provider_id": list(mid_map.values()),
        "max_processing_rate": np.random.default_rng(seed + 2)
            .uniform(5.0, 20.0, len(mid_map)),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(out_dir / "tasks.parquet", index=False)
    providers.to_parquet(out_dir / "providers.parquet", index=False)
    static.to_parquet(out_dir / "provider_static.parquet", index=False)

    meta = {
        "source": source_name,
        "raw_checksums": raw_checksums,
        "preprocess_seed": seed,
        "slot_seconds": slot_s,
        "n_tasks": int(len(tasks)),
        "n_machines": len(machines),
        "n_slots": int(tasks["slot"].max()) + 1,
        "L_p5_p95": [float(p5), float(p95)],
        "duration_quantiles": {str(q): float(ev["duration"].quantile(q))
                               for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
        "time_range": [float(ev["start_time"].min()), float(ev["end_time"].max())],
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta
