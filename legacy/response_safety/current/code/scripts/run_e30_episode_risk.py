"""E30: prospective unused-window replay with full-episode risk control.

Protocol: ../round47-protocol.md. Restricted mobility stays outside outputs.
Bounds concern independent simulator replicates conditional on fixed mobility,
not independent people, assignments, or transfer across calendar windows.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_e29_calibration_selection import run_one
from scripts.run_e20_envelope_tradeoff import SCENARIOS
from scripts.run_e23_guardrail_frontier import PROFILES, profile_weight
from scripts.run_e8_robustness import tape_hash
from src.datasets.mobility import build_geolife_position_cache, positions_to_episode

ALPHA, DELTA = 0.005, 0.05
CAL_SEEDS = tuple(range(47001, 47016))
TEST_SEEDS = tuple(range(48001, 48016))
WINDOWS = {'calibration': ('2009-03-01', '2009-03-08'),
           'test': ('2009-03-15', '2009-03-22')}
RAW_HASH = '1107c5ac064d0a23c8d021a8736a77e53abc75b227062e6260342c6a8d86bdb6'
OLD_HASH = 'ab4ce644cbd4ce689f7f829710933a619e38c121dc9054af72e0b175ecbedb80'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def kl(x, q):
    if x == q:
        return 0.0
    if q <= 0 or q >= 1:
        return math.inf
    a = 0.0 if x == 0 else x * math.log(x / q)
    b = 0.0 if x == 1 else (1-x) * math.log((1-x)/(1-q))
    return a+b


def bounded_kl_upper(losses, tail):
    """Chernoff upper CI for a mean of iid [0,1] episode losses."""
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError('at least one episode loss required')
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('losses must be finite in [0,1]')
    if not 0 < tail < 1:
        raise ValueError('tail must be in (0,1)')
    mean = float(values.mean())
    if mean == 1:
        return 1.0
    if mean == 0:
        return -math.expm1(math.log(tail)/len(values))
    target = math.log(1/tail)/len(values)
    lo, hi = mean, 1.0
    for _ in range(100):
        mid = (lo+hi)/2
        if kl(mean, mid) <= target:
            lo = mid
        else:
            hi = mid
    return hi


def selection_from_bounds(frame):
    if len(frame) != len(SCENARIOS)*len(PROFILES):
        raise ValueError('incomplete calibration family')
    if frame.duplicated(['scenario', 'profile']).any():
        raise ValueError('duplicate calibration cell')
    worst = frame.groupby('global_guardrail_weight')['episode_risk_upper'].max()
    eligible = [float(g) for g, u in worst.items() if u <= ALPHA]
    return {'selected_g': min(eligible) if eligible else 1.0,
            'statistical_pass': bool(eligible),
            'selection_basis': 'episode_KL' if eligible else 'structural_fallback',
            'alpha': ALPHA, 'delta': DELTA,
            'worst_episode_upper_by_g': {str(k): float(v) for k,v in worst.items()},
            'test_outcomes_used': False,
            'cross_window_statistical_transfer_claimed': False}


def one(episodes, split, scenario, profile, seed):
    row = run_one(episodes, split, scenario, profile, seed)
    row['run_id'] = f'e30-{split}-{scenario}-{profile}-{seed}'
    a, v = int(row['assigned']), int(row['qos_violations'])
    if a < 0 or not 0 <= v <= a:
        raise ValueError('invalid episode violation counts')
    row['episode_loss'] = v/max(a, 1)
    row['zero_service_episode'] = a == 0
    if row['status'] not in ('ok', 'completed') or not row['within_global_envelope']:
        raise RuntimeError('failed full-method run; do not publish partial results')
    return row


def grid(episodes, split, seeds, profiles, workers, out):
    jobs = [(episodes, split, s, p, seed) for s in SCENARIOS
            for p in profiles for seed in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, *j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result()
            rows.append(row)
            # Incremental actual result recovery, never substitutes for full grid.
            with (out/f'{split}_running.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
            if i % 15 == 0 or i == len(jobs):
                print(f'E30 {split} {i}/{len(jobs)}', flush=True)
    frame = pd.DataFrame(rows).sort_values(['scenario', 'profile', 'seed'])
    frame.to_csv(out/f'e30_{split}_seed_results.csv', index=False)
    return frame


def summarize(frame, tail):
    rows = []
    # A separate simultaneous family for two-sided coverage CIs.
    coverage_tail = DELTA/(2*frame.groupby(['scenario','profile']).ngroups)
    for (scenario, profile), cell in frame.groupby(['scenario', 'profile']):
        n = len(cell)
        cov = float(cell.service_coverage.mean())
        radius = math.sqrt(math.log(1/coverage_tail)/(2*n))
        rows.append({'scenario': scenario, 'profile': profile,
                     'global_guardrail_weight': float(cell.global_guardrail_weight.iloc[0]),
                     'n_episodes': n, 'assigned': int(cell.assigned.sum()),
                     'qos_violations': int(cell.qos_violations.sum()),
                     'pooled_violation_rate_descriptive': float(cell.qos_violations.sum()/max(cell.assigned.sum(),1)),
                     'mean_episode_loss': float(cell.episode_loss.mean()),
                     'episode_risk_upper': bounded_kl_upper(cell.episode_loss, tail),
                     'risk_cell_tail': tail, 'coverage_cell_tail': coverage_tail,
                     'mean_coverage': cov,
                     'coverage_lower': max(0.0, cov-radius),
                     'coverage_upper': min(1.0, cov+radius),
                     'mean_pps': float(cell.pps.mean()),
                     'zero_service_episodes': int(cell.zero_service_episode.sum())})
    return pd.DataFrame(rows)


def prepare(data_root, restricted, split, seeds, old_positions):
    folder = restricted/split
    folder.mkdir(parents=True, exist_ok=False)
    cache = folder/'positions.parquet'
    start, end = WINDOWS[split]
    meta = build_geolife_position_cache(data_root, cache, start=start, end=end)
    pos = pd.read_parquet(cache)
    if pos.timestamp.min() < pd.Timestamp(start, tz='UTC') or pos.timestamp.max() >= pd.Timestamp(end, tz='UTC'):
        raise ValueError('cache timestamp outside frozen window')
    keys = ['provider_id','timestamp']
    if len(pos.merge(old_positions[keys], on=keys)):
        raise ValueError('old and new provider-time observations overlap')
    episodes = folder/'episodes'
    manifests = []
    for seed in seeds:
        episode = episodes/f'seed_{seed:03d}'
        em = positions_to_episode(cache, episode, 'medium', seed, service_radius_km=3.0)
        if em['n_tasks'] <= 0:
            raise ValueError('empty fixed window; do not replace dates post hoc')
        manifests.append({'seed': seed, 'event_tape_hash': tape_hash(episode),
                          'n_tasks': em['n_tasks'], 'n_providers': em['n_providers']})
    write_json(folder/'episode_manifest.json', manifests)
    audit = {'window_utc': [start,end], 'cache_hash': digest(cache),
             'old_provider_time_overlap': 0, 'positions': len(pos),
             'same_provider_count_as_old': len(set(pos.provider_id)&set(old_positions.provider_id)),
             'source_cache_meta': meta, 'episode_manifest': manifests,
             'redistributable': False}
    return episodes, audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--raw-zip', type=Path, required=True)
    p.add_argument('--old-cache', type=Path, required=True)
    p.add_argument('--old-episodes', type=Path, required=True)
    p.add_argument('--restricted', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        row = one(args.old_episodes, 'pilot_old_data', 'hidden_blocks',
                  next(p for p in PROFILES if profile_weight(p)==.25), 1)
        write_json(args.out/'e30_pilot_old_seed001.json', row)
        print('Pilot full old episode passed', flush=True)
        return
    if (args.out/'e30_frozen.json').exists():
        raise FileExistsError('Existing prospective run; do not overwrite or rerun selection')
    started = time.time()
    if digest(args.raw_zip) != RAW_HASH or digest(args.old_cache) != OLD_HASH:
        raise ValueError('raw archive or old cache identity mismatch')
    protocol = ROOT.parent/'round47-protocol.md'
    frozen = {'protocol_hash': digest(protocol), 'runner_hash': digest(__file__),
              'simulator_hash': digest(ROOT/'src/simulator.py'),
              'windows': WINDOWS, 'calibration_seeds': CAL_SEEDS, 'test_seeds': TEST_SEEDS,
              'candidate_profiles': PROFILES, 'scenarios': SCENARIOS,
              'alpha': ALPHA, 'delta': DELTA, 'frozen_unix_time': time.time(),
              'statistical_unit': 'whole independently randomized model episode conditional on fixed mobility',
              'risk': 'E[V/max(A,1)]', 'raw_data_redistributed': False}
    write_json(args.out/'e30_frozen.json', frozen)
    old_pos = pd.read_parquet(args.old_cache)
    cal_episodes, cal_audit = prepare(args.data_root, args.restricted, 'calibration', CAL_SEEDS, old_pos)
    write_json(args.out/'e30_calibration_source_audit.json', cal_audit)
    cal = grid(cal_episodes, 'calibration', CAL_SEEDS, PROFILES, args.workers, args.out)
    bounds = summarize(cal, DELTA/(len(SCENARIOS)*len(PROFILES)))
    bounds.to_csv(args.out/'e30_calibration_summary.csv', index=False)
    selection = selection_from_bounds(bounds)
    selection['written_unix_time'] = time.time()
    selection['calibration_results_hash'] = digest(args.out/'e30_calibration_seed_results.csv')
    write_json(args.out/'e30_selection_before_test.json', selection)
    print('Selection frozen: '+json.dumps(selection), flush=True)
    # First construction/read of final-test trajectory outcomes occurs here.
    test_episodes, test_audit = prepare(args.data_root, args.restricted, 'test', TEST_SEEDS, old_pos)
    cal_pos = pd.read_parquet(args.restricted/'calibration/positions.parquet')
    test_pos = pd.read_parquet(args.restricted/'test/positions.parquet')
    if len(cal_pos.merge(test_pos[['provider_id','timestamp']], on=['provider_id','timestamp'])):
        raise ValueError('calibration/test provider-time overlap')
    test_audit['calibration_provider_time_overlap'] = 0
    test_audit['test_preparation_after_selection'] = True
    write_json(args.out/'e30_test_source_audit.json', test_audit)
    weights = {selection['selected_g'], .25, 1.0}
    test_profiles = tuple(p for p in PROFILES if profile_weight(p) in weights)
    test = grid(test_episodes, 'test', TEST_SEEDS, test_profiles, args.workers, args.out)
    summary = summarize(test, DELTA/(len(SCENARIOS)*len(test_profiles)))
    summary.to_csv(args.out/'e30_test_summary.csv', index=False)
    gains = []
    for s, cell in test.groupby('scenario'):
        ref = cell[cell.global_guardrail_weight==1].set_index('seed')
        comp = cell[cell.global_guardrail_weight==.25].set_index('seed')
        d = (comp.service_coverage-ref.service_coverage).to_numpy()
        radius = math.sqrt(2*math.log(2*len(SCENARIOS)/DELTA)/len(d))
        gains.append({'scenario':s, 'comparator_minus_fixed_coverage_pp': float(d.mean()*100),
                      'gain_ci_lower_pp': max(-1,float(d.mean())-radius)*100,
                      'gain_ci_upper_pp': min(1,float(d.mean())+radius)*100})
    pd.DataFrame(gains).to_csv(args.out/'e30_test_paired_coverage.csv', index=False)
    # Original E29 data remain immutable; derive a clearly retrospective unit audit.
    old = pd.read_csv(ROOT.parent/'results/e29_calibration_selection/e29_calibration_seed_results.csv')
    old['episode_loss'] = old.qos_violations/old.assigned.clip(lower=1)
    old['zero_service_episode'] = old.assigned==0
    summarize(old, DELTA/30).to_csv(args.out/'e29_retrospective_episode_bounds.csv', index=False)
    write_json(args.out/'e30_completion.json', {'runs':len(cal)+len(test), 'elapsed_seconds':time.time()-started,
        'selected_g':selection['selected_g'], 'statistical_pass':selection['statistical_pass'],
        'test_profiles':test_profiles, 'runner_hash':digest(__file__),
        'frozen_runner_unchanged':digest(__file__)==frozen['runner_hash'],
        'all_status_ok':bool(test.status.isin(['ok','completed']).all()),
        'minimum_calibration_upper_at_zero_loss':bounded_kl_upper(np.zeros(len(CAL_SEEDS)),DELTA/30)})
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
