"""Prespecified R73 GeoLife temporal-window replay. Keep all generated data outside this package."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "legacy" / "response_safety" / "current" / "code"
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scripts.run_e20_envelope_tradeoff import slot_ratios
from src.datasets.mobility import build_geolife_position_cache, positions_to_episode
from src.datasets.trace_loader import load_trace_episode
from src.simulator import Simulator

PLAN_PATH = Path(__file__).with_name("preregistration.json")
NAMES = ("tasks.parquet", "providers.parquet", "provider_static.parquet", "meta.json")
SOURCE_FILES = ("src/simulator.py", "src/pair_eval.py", "src/matching.py", "src/datasets/mobility.py", "src/datasets/trace_loader.py", "scripts/run_e20_envelope_tradeoff.py")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def plan() -> dict:
    return read(PLAN_PATH)


def outside_package(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT.resolve() or resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("Restricted inputs/results must stay outside the public package")
    return resolved


def sources() -> dict:
    result = {name: sha(CODE / name) for name in SOURCE_FILES}
    result["R56/configurations.json"] = sha(ROOT / "legacy" / "response_safety" / "current" / "configurations.json")
    return result


def source_check(metadata: dict) -> None:
    if metadata["preregistration_sha256"] != sha(PLAN_PATH) or metadata["runner_sha256"] != sha(Path(__file__)) or metadata["source_sha256"] != sources():
        raise ValueError("Preregistered plan, runner, or core changed after preparation")


def prepare(raw: Path, output: Path) -> None:
    p = plan()
    output = outside_package(output)
    if output.exists():
        raise FileExistsError("Choose a new restricted output directory")
    if not raw.is_dir():
        raise FileNotFoundError("--raw must point to the authorized GeoLife Data directory")
    output.mkdir(parents=True)
    metadata = {"preregistration_sha256": sha(PLAN_PATH), "runner_sha256": sha(Path(__file__)), "source_sha256": sources(), "windows": {}, "errors": []}
    write(output / "input_manifest.json", metadata)
    for idx, (start, end) in enumerate(p["windows_utc"], start=1):
        wid = f"w{idx:02d}"
        folder = output / wid
        cache = folder / "positions.parquet"
        try:
            cache_meta = build_geolife_position_cache(raw, cache, start=start, end=end)
            episodes = {}
            for seed in p["episode_builder"]["model_seeds"]:
                target = folder / "episodes" / f"seed_{seed:03d}"
                episode_meta = positions_to_episode(cache, target, "medium", seed)
                episodes[str(seed)] = {"meta": episode_meta, "file_sha256": {name: sha(target / name) for name in NAMES}}
            metadata["windows"][wid] = {"start": start, "end": end, "cache_meta": cache_meta, "cache_sha256": sha(cache), "episodes": episodes}
            print(json.dumps({"prepared": wid, "providers": cache_meta["providers"], "active_slots": cache_meta["active_slots"], "episodes": len(episodes)}), flush=True)
        except Exception:
            metadata["errors"].append({"window": wid, "error": traceback.format_exc()})
            print(json.dumps({"failed": wid}), flush=True)
        write(output / "input_manifest.json", metadata)
    if metadata["errors"]:
        raise RuntimeError("Some preregistered windows failed preprocessing; failures retained in input_manifest.json")


def verify_inputs(output: Path) -> dict:
    output = outside_package(output)
    metadata = read(output / "input_manifest.json")
    source_check(metadata)
    if len(metadata["windows"]) != len(plan()["windows_utc"]):
        raise ValueError("Incomplete prespecified windows")
    for wid, entry in metadata["windows"].items():
        if sha(output / wid / "positions.parquet") != entry["cache_sha256"]:
            raise ValueError(f"Changed cache: {wid}")
        for seed, ep in entry["episodes"].items():
            folder = output / wid / "episodes" / f"seed_{int(seed):03d}"
            if {name: sha(folder / name) for name in NAMES} != ep["file_sha256"]:
                raise ValueError(f"Changed episode: {wid}/{seed}")
    return metadata


def cell(output: Path, wid: str, seed: int, scenario: str, profile: str) -> dict:
    manifest = read(output / "input_manifest.json")
    entry = manifest["windows"][wid]["episodes"][str(seed)]
    episode = output / wid / "episodes" / f"seed_{seed:03d}"
    if {name: sha(episode / name) for name in NAMES} != entry["file_sha256"]:
        raise ValueError("Input episode changed")
    n = int(entry["meta"]["n_providers"])
    cfg = copy.deepcopy(read(ROOT / "legacy" / "response_safety" / "current" / "configurations.json")[str(seed)][profile])
    cfg["dataset"]["processed_dir"] = str(episode)
    cfg["providers"]["N_mean"] = n
    if cfg["matching"]["objective"] != "coverage_first_payment_second" or cfg["contract"]["D_bar"] != 50 or cfg["matching"]["budget_ratio"] != 3.5:
        raise ValueError("Formal R56 protocol mismatch")
    data = load_trace_episode("geolife", cfg, seed=seed, processed_dir=episode)
    if data["meta"]["source"] != "geolife" or data["meta"]["redistributable"] is not False:
        raise ValueError("Source or redistribution metadata mismatch")
    tasks = data["tasks"].copy()
    tasks["kappa_design"] = tasks["kappa"].to_numpy(float)
    ratios = slot_ratios(tasks["slot"].to_numpy(int), scenario, seed, 0.7, 1.3)
    if not np.all((ratios >= 0.7 - 1e-12) & (ratios <= 1.3 + 1e-12)):
        raise ValueError("Stress outside frozen envelope")
    tasks["kappa"] = tasks["kappa_design"].to_numpy(float) * ratios
    tasks["response_ratio_audit"] = ratios
    data["tasks"] = tasks
    result = Simulator(cfg, data, method="PASI", seed=seed).run()
    summary, slots = result["summary"], result["slot_log"]
    assigned = int(summary["num_assigned"])
    qualified = int(slots["num_qualified_completed"].sum())
    total = int(summary["total_tasks"])
    reserved = float(slots["budget_used"].sum())
    payment = float(summary["cumulative_payment"])
    return {"window": wid, "seed": seed, "scenario": scenario, "profile": profile,
            "tasks": total, "assigned": assigned, "qualified": qualified,
            "coverage": assigned / max(total, 1), "qualified_coverage": qualified / max(total, 1),
            "qos_violations": assigned - qualified, "qos_violation_rate": (assigned - qualified) / max(assigned, 1),
            "ir_violations": int(summary["ir_violations"]), "target_violations": int(summary["target_violations"]),
            "design_reserve": reserved, "execution_payment": payment, "pps": payment / max(assigned, 1),
            "settlement_over_reserve_slots": int((slots["total_payment"] > slots["budget_used"] + 1e-7).sum()),
            "reserve_over_budget_slots": int((slots["budget_used"] > slots["budget"] + 1e-7).sum()),
            "settlement_over_budget_slots": int((slots["total_payment"] > slots["budget"] + 1e-7).sum()),
            "run_status": result["diagnostics"].get("status"),
            "source": data["meta"]["source"], "redistributable": False,
            "episode_sha256": entry["file_sha256"], "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "preregistration_sha256": manifest["preregistration_sha256"], "runner_sha256": manifest["runner_sha256"]}


def run(output: Path, smoke: bool) -> None:
    output = outside_package(output)
    metadata = verify_inputs(output)
    p = plan()
    jobs = [(f"w{i:02d}", seed, scenario, profile)
            for i in range(1, len(p["windows_utc"]) + 1)
            for seed in p["episode_builder"]["model_seeds"]
            for scenario in p["mechanism"]["scenarios"]
            for profile in p["mechanism"]["profiles"]]
    if smoke:
        jobs = jobs[:2]
    else:
        smoke_report = read(output / "smoke_passed.json")
        if not smoke_report["passed"] or smoke_report["input_manifest_sha256"] != sha(output / "input_manifest.json"):
            raise RuntimeError("Run a successful --action smoke with these frozen inputs first")
    result_dir = output / "results"
    result_dir.mkdir(exist_ok=True)
    for wid, seed, scenario, profile in jobs:
        name = f"{wid}__seed{seed:03d}__{scenario}__{profile}"
        target = result_dir / f"{name}.json"
        if target.exists():
            if smoke or not read(target).get("error"):
                continue
            raise FileExistsError(f"Existing failure retained: {target}")
        try:
            row = cell(output, wid, seed, scenario, profile)
        except Exception:
            row = {"window": wid, "seed": seed, "scenario": scenario, "profile": profile, "error": traceback.format_exc()}
        write(target, row)
        print(json.dumps({"cell": name, "error": "error" in row, "coverage": row.get("coverage"), "qos_violations": row.get("qos_violations")}), flush=True)
    if smoke:
        rows = [read(result_dir / f"w01__seed{p['episode_builder']['model_seeds'][0]:03d}__stable__{profile}.json") for profile in p["mechanism"]["profiles"]]
        passed = all("error" not in row and row["source"] == "geolife" and row["redistributable"] is False for row in rows)
        write(output / "smoke_passed.json", {"passed": passed, "cells": len(rows), "input_manifest_sha256": sha(output / "input_manifest.json"), "not_new_independent_mobility_sample": True})
        if not passed:
            raise RuntimeError("Smoke failed; do not start full matrix")


def summarize(output: Path) -> None:
    output = outside_package(output)
    metadata = verify_inputs(output)
    p = plan()
    jobs = len(p["windows_utc"]) * len(p["episode_builder"]["model_seeds"]) * len(p["mechanism"]["scenarios"]) * len(p["mechanism"]["profiles"])
    rows = [read(path) for path in sorted((output / "results").glob("*.json"))]
    failures = [row for row in rows if "error" in row]
    ok = [row for row in rows if "error" not in row]
    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in ok]
    pd.DataFrame(flat).to_csv(output / "seed_results.csv", index=False)
    if ok:
        df = pd.DataFrame(flat)
        grouped = df.groupby(["window", "scenario", "profile"], as_index=False).agg(
            seeds=("seed", "nunique"), coverage=("coverage", "mean"), qualified_coverage=("qualified_coverage", "mean"),
            tasks=("tasks", "sum"), assigned=("assigned", "sum"), qualified=("qualified", "sum"),
            qos_violations=("qos_violations", "sum"), ir_violations=("ir_violations", "sum"),
            target_violations=("target_violations", "sum"), pps=("pps", "mean"),
            design_reserve=("design_reserve", "mean"), execution_payment=("execution_payment", "mean"),
            settlement_over_reserve_slots=("settlement_over_reserve_slots", "sum"),
            reserve_over_budget_slots=("reserve_over_budget_slots", "sum"),
            settlement_over_budget_slots=("settlement_over_budget_slots", "sum"))
        grouped["qos_violation_rate"] = grouped["qos_violations"] / grouped["assigned"].clip(lower=1)
        grouped.to_csv(output / "window_summary.csv", index=False)
        pair = grouped.pivot(index=["window", "scenario"], columns="profile", values=["coverage", "qualified_coverage", "qos_violation_rate", "pps"])
        pair.columns = [f"{a}__{b}" for a, b in pair.columns]
        pair = pair.reset_index()
        for name in ("coverage", "qualified_coverage", "qos_violation_rate", "pps"):
            pair[f"{name}__envelope_minus_point"] = pair[f"{name}__fixed_envelope"] - pair[f"{name}__frozen_point"]
        pair.to_csv(output / "window_paired.csv", index=False)
    write(output / "completion.json", {"expected_cells": jobs, "completed_cells": len(ok), "failed_cells": len(failures),
          "failures": failures, "windows": len(metadata["windows"]), "model_seeds_per_window": len(p["episode_builder"]["model_seeds"]),
          "statistical_unit": "temporal mobility window; nested model seeds", "risk_certification": False,
          "references_changed": False, "preregistration_sha256": metadata["preregistration_sha256"],
          "runner_sha256": metadata["runner_sha256"], "source_sha256": metadata["source_sha256"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "smoke", "run", "summarize"))
    parser.add_argument("--raw", type=Path, help="Authorized GeoLife Data directory; only needed for prepare")
    parser.add_argument("--out", required=True, type=Path, help="Restricted output directory outside this package")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.raw is None:
            parser.error("prepare requires --raw")
        prepare(args.raw, args.out)
    elif args.action == "check":
        print(json.dumps({"windows": len(verify_inputs(args.out)["windows"]), "passed": True}))
    elif args.action == "smoke":
        run(args.out, True)
    elif args.action == "run":
        run(args.out, False)
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()
