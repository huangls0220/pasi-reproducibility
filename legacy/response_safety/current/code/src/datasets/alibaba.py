"""Alibaba cluster-trace-v2018 adapter (Trace B, Section 8.1/3.2).

Parses batch_task.csv into the normalized event table consumed by
trace_common.events_to_episode.  v2018 batch_task columns (no header):
    task_name, instance_num, job_name, task_type, status,
    start_time, end_time, plan_cpu, plan_mem
Only Terminated tasks with valid times are kept.  machine_id is not in
batch_task; if machine-level files are absent we shard jobs onto a
configurable pseudo-machine pool derived from job hash — documented in
meta.json as "machine_source": "job_hash" (an approximation; prefer
container/machine tables when available).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

V2018_COLS = ["task_name", "instance_num", "job_name", "task_type", "status",
              "start_time", "end_time", "plan_cpu", "plan_mem"]


def load_events(raw_dir: Path, machine_subset: int = 1000) -> tuple[pd.DataFrame, str]:
    f = raw_dir / "batch_task.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found. Download cluster-trace-v2018 batch_task.csv "
            "first (see configs/datasets/alibaba_trace.yaml)."
        )
    df = pd.read_csv(f, header=None, names=V2018_COLS)
    df = df[df["status"].astype(str).str.lower().str.startswith("term")]
    df = df.dropna(subset=["start_time", "end_time", "plan_cpu"])
    df = df[df["end_time"] > df["start_time"]]

    machine_src = "machine_column"
    if "machine_id" not in df.columns:
        # Pseudo-machine assignment by job hash (documented approximation)
        df["machine_id"] = ("M" + (df["job_name"].astype(str).apply(hash).abs()
                                   % machine_subset).astype(str))
        machine_src = "job_hash"

    out = df[["machine_id", "start_time", "end_time", "plan_cpu"]].rename(
        columns={"plan_cpu": "cpu_request"})
    out["cpu_request"] = out["cpu_request"] / 100.0  # plan_cpu is in % of cores
    return out, machine_src
