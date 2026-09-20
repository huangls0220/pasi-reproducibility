"""Legally rebuild and verify all frozen E5/E7/E16 GeoLife episodes outside the public package."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "legacy" / "response_safety" / "current" / "code"
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

from src.datasets.mobility import build_geolife_position_cache, positions_to_episode


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(raw: Path, output: Path, workloads: list[str], seeds: list[int]) -> None:
    output = output.resolve()
    if output == ROOT.resolve() or output.is_relative_to(ROOT.resolve()):
        raise ValueError("Restricted positions and episodes must stay outside this package")
    if output.exists():
        raise FileExistsError("Choose a new restricted output directory")
    if not raw.is_dir():
        raise FileNotFoundError("--raw must point to an authorized GeoLife Data directory")
    plan = json.loads((ROOT / "evidence_runners" / "e5" / "preregistration.json").read_text(encoding="utf-8"))
    cache = output / "cache" / "positions.parquet"
    meta = build_geolife_position_cache(raw, cache)
    expected_cache = "ab4ce644cbd4ce689f7f829710933a619e38c121dc9054af72e0b175ecbedb80"
    if sha(cache) != expected_cache:
        raise ValueError("Rebuilt position cache differs from frozen published input")
    report = {"cache_sha256": sha(cache), "cache_meta": meta, "episode_count": 0,
              "workloads": workloads, "seeds": seeds, "checks": [], "all_passed": False,
              "restricted_data_not_exported": True}
    output.mkdir(parents=True, exist_ok=True)
    for workload in workloads:
        for seed in seeds:
            episode = output / "episodes" / workload / f"seed_{seed:03d}"
            positions_to_episode(cache, episode, workload, seed, service_radius_km=3.0)
            expected = plan["entries"][f"{workload}/{seed}"]["episode_sha256"]
            actual = {name: sha(episode / name) for name in expected}
            report["checks"].append({"workload": workload, "seed": seed,
                                     "matching_files": sum(actual[name] == expected[name] for name in expected),
                                     "file_count": len(expected), "pass": actual == expected})
            report["episode_count"] += 1
            (output / "reconstruction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if actual != expected:
                raise ValueError(f"Episode hash mismatch: {workload}/{seed}; retained in restricted output")
            print(json.dumps({"reconstructed": f"{workload}/{seed}", "files_matched": len(expected)}), flush=True)
    report["all_passed"] = all(item["pass"] for item in report["checks"])
    (output / "reconstruction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Authorized GeoLife Data directory")
    parser.add_argument("--out", type=Path, required=True, help="New restricted directory outside package")
    parser.add_argument("--workloads", nargs="+", choices=("low", "medium", "high"), default=["low", "medium", "high"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 31)))
    args = parser.parse_args()
    run(args.raw, args.out, args.workloads, args.seeds)
