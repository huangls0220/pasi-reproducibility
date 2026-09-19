"""Unified trace episode loader (Section 8.3).

Loads PREPROCESSED trace data (Google / Alibaba) from data/processed/<name>/
and attaches model-generated behavioural parameters (seeded, documented —
these are NOT trace fields; the paper must call this trace-driven
simulation, §8/appendix D).

Required processed files (produced by scripts/preprocess_*.py):
    tasks.parquet, providers.parquet, provider_static.parquet, meta.json

If the files are missing, raises FileNotFoundError with download and
preprocessing instructions — synthetic data is NEVER silently substituted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent

_INSTRUCTIONS = {
    "google": (
        "Google ClusterData2019 not found.\n"
        "  1. Fetch instance/machine events for one cell (2-4 days) from\n"
        "     https://github.com/google/cluster-data (ClusterData2019 v3).\n"
        "  2. Put raw files under data/raw/google/.\n"
        "  3. python scripts/preprocess_google.py --raw data/raw/google "
        "--out data/processed/google\n"
    ),
    "alibaba": (
        "Alibaba cluster-trace-v2018 not found.\n"
        "  1. Download batch_task.csv (+ machine_meta.csv) from\n"
        "     https://github.com/alibaba/clusterdata (cluster-trace-v2018).\n"
        "  2. Put raw files under data/raw/alibaba/.\n"
        "  3. python scripts/preprocess_alibaba.py --raw data/raw/alibaba "
        "--out data/processed/alibaba\n"
    ),
}

TASK_SCHEMA = ["episode_id", "slot", "task_id", "L", "input_size", "output_size",
               "deadline", "min_quality", "q_bar", "kappa", "value", "raw_duration"]
PROV_SCHEMA = ["episode_id", "slot", "provider_id", "load_ratio",
               "communication_rate", "availability_duration"]
STATIC_SCHEMA = ["episode_id", "provider_id", "max_processing_rate"]


def validate_schema(df: pd.DataFrame, required: list[str], name: str) -> list[str]:
    """Return list of problems (empty = ok)."""
    problems = [f"{name}: missing column '{c}'" for c in required if c not in df.columns]
    if not problems and df.isna().any().any():
        na_cols = df.columns[df.isna().any()].tolist()
        problems.append(f"{name}: NaN values in {na_cols}")
    return problems


def load_trace_episode(
    name: str,
    cfg: dict[str, Any],
    seed: int,
    processed_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a processed trace episode and attach behavioural parameters."""
    ds_cfg = cfg.get("dataset", {}) or {}
    pdir = Path(processed_dir or ds_cfg.get("processed_dir", _ROOT / "data" / "processed" / name))
    if not pdir.is_absolute():
        pdir = _ROOT / pdir

    needed = ["tasks.parquet", "providers.parquet", "provider_static.parquet", "meta.json"]
    missing = [f for f in needed if not (pdir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Processed {name} trace incomplete at {pdir} (missing: {missing}).\n"
            + _INSTRUCTIONS.get(name, "")
        )

    tasks = pd.read_parquet(pdir / "tasks.parquet")
    providers = pd.read_parquet(pdir / "providers.parquet")
    static = pd.read_parquet(pdir / "provider_static.parquet")
    with open(pdir / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)

    problems = (validate_schema(tasks, TASK_SCHEMA, "tasks")
                + validate_schema(providers, PROV_SCHEMA, "providers")
                + validate_schema(static, STATIC_SCHEMA, "provider_static"))
    if problems:
        raise ValueError(f"{name} schema validation failed: {problems}")

    # Region split (§8.4): first 20% of slots = validation, rest = test
    region = ds_cfg.get("use_region", "test")
    max_slot = int(tasks["slot"].max())
    cut = int(max_slot * 0.2)
    if region == "validation":
        keep = tasks["slot"] <= cut
        tasks = tasks[keep]
        providers = providers[providers["slot"] <= cut]
    elif region == "test":
        tasks = tasks[tasks["slot"] > cut]
        providers = providers[providers["slot"] > cut]
        # re-base slots to 0
        base = int(tasks["slot"].min()) if len(tasks) else 0
        tasks = tasks.assign(slot=tasks["slot"] - base)
        providers = providers.assign(slot=providers["slot"] - base)

    # Model-generated behavioural parameters (seeded; NOT trace fields)
    prov_cfg = cfg.get("providers", {}) or {}
    rng = np.random.default_rng([seed, 4242])
    n = len(static)
    rho_B = float(prov_cfg.get("behavioral_fraction", 0.70))
    behavioral = rng.random(n) < rho_B

    def u(key, lo, hi):
        a, b = prov_cfg.get(key, [lo, hi])
        return rng.uniform(a, b, n)

    static = static.copy()
    static["alpha"] = u("alpha", 0.05, 0.20)
    static["beta"] = u("beta", 0.02, 0.10)
    static["zeta"] = np.where(behavioral, u("zeta", 0.65, 0.95), 1.0)
    static["omega"] = np.where(behavioral, u("omega", 0.05, 0.25), 0.0)
    static["xi"] = u("xi", 0.05, 0.12)
    static["delta"] = u("delta", 0.01, 0.05)
    static["outside_option"] = float(prov_cfg.get("outside_option", 0.01))
    static["behavioral"] = behavioral

    meta.update({"behavior_seed": seed, "behavioral_fraction": rho_B,
                 "region": region, "T": int(tasks["slot"].max()) + 1 if len(tasks) else 0})

    return {"tasks": tasks, "providers": providers,
            "provider_static": static, "meta": meta}
