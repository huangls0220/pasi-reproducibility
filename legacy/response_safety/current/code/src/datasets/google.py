"""Google ClusterData2019 adapter (Trace A, Section 8.1/3.1).

Parses instance-event exports into the normalized event table consumed by
trace_common.events_to_episode.  Accepts CSV (or CSV.GZ) with at least:
    machine_id, start_time, end_time, and a CPU request column
    (cpu_request | requested_cpu | resource_request.cpus).
Times may be in microseconds (Google native) — auto-detected and
converted to seconds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_CPU_CANDIDATES = ["cpu_request", "requested_cpu", "resource_request.cpus", "cpus"]
_START_CANDIDATES = ["start_time", "start", "time_start"]
_END_CANDIDATES = ["end_time", "end", "time_end"]


def _pick(df: pd.DataFrame, candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Google raw file lacks a {what} column (looked for {candidates}); "
                     f"available: {list(df.columns)}")


def load_events(raw_dir: Path) -> pd.DataFrame:
    files = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.csv.gz")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {raw_dir}. Download ClusterData2019 instance "
            "events for one cell first (see configs/datasets/google_trace.yaml)."
        )
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    cpu = _pick(df, _CPU_CANDIDATES, "CPU request")
    st = _pick(df, _START_CANDIDATES, "start time")
    en = _pick(df, _END_CANDIDATES, "end time")
    if "machine_id" not in df.columns:
        raise ValueError("Google raw file lacks machine_id")

    out = df[["machine_id", st, en, cpu]].rename(
        columns={st: "start_time", en: "end_time", cpu: "cpu_request"})
    # Google native timestamps are microseconds since trace start
    if out["start_time"].abs().max() > 1e12:
        out["start_time"] /= 1e6
        out["end_time"] /= 1e6
    return out
