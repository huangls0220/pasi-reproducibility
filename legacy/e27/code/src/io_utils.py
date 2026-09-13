"""I/O utilities: config loading, logging, Parquet save/load."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml


# ── Config ──────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def save_config_snapshot(cfg: dict, output_path: str | Path) -> None:
    """Save a config dict as YAML snapshot."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)


# ── Environment info ────────────────────────────────────────────

def get_env_info() -> dict[str, str]:
    """Capture Python version, platform, and key package versions."""
    import platform

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
    }
    for pkg in ["numpy", "scipy", "pandas", "matplotlib"]:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[f"{pkg}_version"] = "not_installed"
    return info


# ── Parquet helpers ─────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to Parquet, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load DataFrame from Parquet."""
    return pd.read_parquet(path)


# ── JSON helpers ────────────────────────────────────────────────

def save_json(obj: Any, path: str | Path) -> None:
    """Save object as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Result path helpers ─────────────────────────────────────────

def make_run_dir(
    results_root: str | Path,
    experiment: str,
    dataset: str,
    method: str,
    seed: int,
) -> Path:
    """Create and return a unique output directory for one run."""
    run_dir = Path(results_root) / experiment / dataset / method / f"seed_{seed:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.json"


def summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def diagnostics_path(run_dir: Path) -> Path:
    return run_dir / "diagnostics.json"
