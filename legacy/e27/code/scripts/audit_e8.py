"""Integrity audit for E8 seed-level and summary outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT.parent / "results" / "e8"
E5 = ROOT.parent.parent / "experiment-e5" / "results" / "geolife" / "e5_geolife_seed_results.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    formal = pd.read_csv(RESULTS / "e8_formal_seed_results.csv")
    joint = pd.read_csv(RESULTS / "e8_joint_seed_results.csv")
    e5 = pd.read_csv(E5)
    checks: dict[str, bool | int | float | str] = {}
    checks["formal_rows_840"] = len(formal) == 840
    checks["formal_variants_28"] = formal["variant"].nunique() == 28
    checks["formal_30_each"] = bool(formal.groupby("variant").size().eq(30).all())
    checks["formal_unique_run_id"] = not formal["run_id"].duplicated().any()
    checks["formal_no_missing_required"] = not formal.drop(
        columns=["correction_rate", "peak_rss_mb", "incremental_rss_mb"]
    ).isna().any().any()
    checks["same_tape_within_seed"] = bool(formal.groupby("seed")["event_tape_hash"].nunique().eq(1).all())
    checks["one_config_hash_per_variant"] = bool(formal.groupby("variant")["config_hash"].nunique().eq(1).all())
    checks["joint_rows_60"] = len(joint) == 60
    checks["joint_30_each"] = bool(joint.groupby("variant").size().eq(30).all())
    checks["joint_unique_run_id"] = not joint["run_id"].duplicated().any()

    control = formal[formal["variant"] == "control"].sort_values("seed")
    e5_pasi = e5[(e5["workload"] == "medium") & (e5["method"] == "PASI")].sort_values("seed")
    checks["control_seed_alignment_with_e5"] = np.array_equal(
        control["seed"].to_numpy(), e5_pasi["seed"].to_numpy())
    checks["control_pps_exact_e5"] = bool(np.array_equal(
        control["pps"].to_numpy(), e5_pasi["pps"].to_numpy()))
    checks["control_coverage_exact_e5"] = bool(np.array_equal(
        control["service_coverage"].to_numpy(), e5_pasi["service_coverage"].to_numpy()))
    joint_control = joint[joint["variant"] == "control"].sort_values("seed")
    checks["joint_control_exact_formal"] = bool(
        np.array_equal(joint_control["pps"].to_numpy(), control["pps"].to_numpy())
        and np.array_equal(joint_control["service_coverage"].to_numpy(),
                           control["service_coverage"].to_numpy()))

    unsafe_status = formal[~formal["run_status"].isin(["ok", "completed"])]
    checks["diagnostic_unsafe_rows"] = int(len(unsafe_status))
    checks["unsafe_status_explained_by_safety_metrics"] = bool(
        ((unsafe_status["qos_violation_rate"] > 0)
         | (unsafe_status["target_miss_rate"] > 0)
         | (unsafe_status["under_incentive_rate"] > 0)).all())
    checks["simulator_sha256"] = sha(ROOT / "src" / "simulator.py")
    checks["pair_eval_sha256"] = sha(ROOT / "src" / "pair_eval.py")
    checks["runner_sha256"] = sha(ROOT / "scripts" / "run_e8_robustness.py")
    pass_keys = [k for k, v in checks.items() if isinstance(v, bool)]
    report = {"all_boolean_checks_pass": all(checks[k] for k in pass_keys), "checks": checks}
    (RESULTS / "e8_integrity_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_boolean_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
