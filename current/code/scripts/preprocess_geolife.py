"""Build the restricted GeoLife position cache and one processed event tape."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.mobility import build_geolife_position_cache, positions_to_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--restricted-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workload", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    cache = args.restricted_cache / "positions.parquet"
    if args.rebuild_cache or not cache.exists():
        cache_meta = build_geolife_position_cache(args.raw, cache)
        print(json.dumps(cache_meta, indent=2))
    episode_meta = positions_to_episode(cache, args.out, args.workload, args.seed)
    print(json.dumps(episode_meta, indent=2))


if __name__ == "__main__":
    main()
