"""Rebuild the original GeoLife cache/seed 001 and compare inputs, not performance.

All generated trajectories remain in an external restricted scratch directory.
No Simulator.run or formal experiment runner is called.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.datasets.mobility import build_geolife_position_cache, positions_to_episode, sha256
from src.datasets.trace_loader import load_trace_episode
from scripts.run_e20_envelope_tradeoff import build_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--reference-episode', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    if scratch.is_relative_to(ROOT.parent.resolve()):
        raise ValueError('Restricted scratch must be outside the distributable artifact.')
    if scratch.exists():
        raise FileExistsError('Choose a new empty scratch location; existing data are preserved.')
    if not args.raw.is_dir() or not args.reference_cache.is_file() or not args.reference_episode.is_dir():
        raise FileNotFoundError('Raw data and archived reference inputs must exist locally.')
    scratch.mkdir(parents=True)
    cache = scratch / 'cache/positions.parquet'
    episode = scratch / 'episodes/medium/seed_001'
    cache_meta = build_geolife_position_cache(args.raw, cache)
    meta = positions_to_episode(cache, episode, 'medium', 1)
    checks = {'source_is_geolife': cache_meta['source'] == 'GeoLife GPS Trajectories 1.3',
              'restricted': cache_meta['redistributable'] is False and meta['redistributable'] is False,
              'old_window': cache_meta['window_utc'] == ['2009-02-14', '2009-02-21']}
    byte_identity = {'positions': sha256(cache) == sha256(args.reference_cache)}
    for name, fresh, archived in [('positions', cache, args.reference_cache)] + [
        (name, episode / name, args.reference_episode / name)
        for name in ['tasks.parquet', 'providers.parquet', 'provider_static.parquet']]:
        pd.testing.assert_frame_equal(pd.read_parquet(fresh), pd.read_parquet(archived), check_exact=True)
        checks[name + '_values_exact'] = True
        byte_identity[name] = sha256(fresh) == sha256(archived)
    old_meta = json.loads((args.reference_episode / 'meta.json').read_text(encoding='utf-8'))
    checks['episode_metadata_exact'] = meta == old_meta
    cfg = build_config(episode, int(meta['n_providers']), 'fixed_envelope', .7, 1.3)
    data = load_trace_episode('geolife', cfg, seed=1, processed_dir=episode)
    old_cfg = build_config(args.reference_episode, int(old_meta['n_providers']), 'fixed_envelope', .7, 1.3)
    old_data = load_trace_episode('geolife', old_cfg, seed=1, processed_dir=args.reference_episode)
    loaded_rows = {}
    for key, frame in data.items():
        if isinstance(frame, pd.DataFrame):
            pd.testing.assert_frame_equal(frame, old_data[key], check_exact=True)
            checks['loader_' + key + '_exact'] = True
            loaded_rows[key] = len(frame)
    out = {'checks': checks, 'all_pass': all(checks.values()), 'parquet_byte_identity': byte_identity,
           'window_utc': cache_meta['window_utc'], 'positions': cache_meta['positions'],
           'providers': cache_meta['providers'], 'seed': 1, 'workload': 'medium',
           'loaded_row_counts': loaded_rows, 'source_rebuild_only': True,
           'formal_experiment_launched': False, 'restricted_data_redistributed': False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
    if not out['all_pass']:
        raise AssertionError('Reconstruction mismatch; inspect the report without changing archived inputs.')


if __name__ == '__main__':
    main()
