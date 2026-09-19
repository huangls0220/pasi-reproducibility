"""Portable R56 input reconstruction and outcome replay; no downloads or uploads."""
from pathlib import Path
import argparse, copy, csv, hashlib, json, sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
CODE=ROOT/'current/code'
sys.path.insert(0,str(CODE))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def outside(path):
    p=path.resolve()
    if p.is_relative_to(ROOT.resolve()):raise ValueError('Keep generated/restricted data outside the distributable repository')
    return p
def check(episodes,seeds):
    expected=read(ROOT/'current/episode_fingerprints.json');checks=[]
    for seed in seeds:
        folder=episodes/f'seed_{seed:03d}'
        for name,digest in expected[str(seed)]['episode_file_sha256'].items():
            actual=sha(folder/name)
            checks.append({'seed':seed,'file':name,'pass':actual==digest})
        if not all(x['pass'] for x in checks):raise ValueError(f'Input hash mismatch: seed {seed}')
    return checks
def rebuild(args):
    from src.datasets.mobility import build_geolife_position_cache, positions_to_episode
    target=outside(args.out)
    if target.exists():raise FileExistsError('Choose a new restricted output folder')
    if args.cache:
        cache=args.cache.resolve()
    else:
        if not args.raw:raise ValueError('Supply --raw Data-directory or --cache')
        cache=target/'cache/positions.parquet'
        build_geolife_position_cache(args.raw,cache)
    expected='ab4ce644cbd4ce689f7f829710933a619e38c121dc9054af72e0b175ecbedb80'
    if sha(cache)!=expected:raise ValueError('Position cache differs from the frozen original')
    for seed in args.seeds:positions_to_episode(cache,target/'episodes/medium'/f'seed_{seed:03d}','medium',seed,service_radius_km=3.0)
    checks=check(target/'episodes/medium',args.seeds)
    print(json.dumps({'action':'rebuild','checks':len(checks),'passed':True,'simulator_runs':0}))
def replay(args):
    import numpy as np
    from scripts.run_e20_envelope_tradeoff import slot_ratios
    from src.datasets.trace_loader import load_trace_episode
    from src.simulator import Simulator
    # This wrapper replays scientific outcome columns, not historical timings,
    # the original shadow-information audit, or the MILP audit instrumentation.
    check(args.episodes,args.seeds)
    out=outside(args.out)
    if out.exists():raise FileExistsError('Preserve previous runs; choose a new output folder')
    out.mkdir(parents=True)
    configs=read(ROOT/'current/configurations.json');rows=[];comparisons=[]
    with (ROOT/'current/results/seed_results.csv').open(encoding='utf-8-sig',newline='') as f:
        refs={(int(r['seed']),r['scenario'],r['profile']):r for r in csv.DictReader(f)}
    for seed in args.seeds:
        folder=args.episodes/f'seed_{seed:03d}'
        for scenario in ('stable','down_step','up_step'):
            for profile in ('frozen_point','fixed_envelope'):
                cfg=copy.deepcopy(configs[str(seed)][profile]);cfg['dataset']['processed_dir']=str(folder.resolve())
                data=load_trace_episode('geolife',cfg,seed=seed,processed_dir=folder)
                data['tasks']=data['tasks'].copy()
                data['tasks']['kappa_design']=data['tasks']['kappa'].to_numpy(float)
                ratios=slot_ratios(data['tasks']['slot'].to_numpy(int),scenario,seed,.7,1.3)
                data['tasks']['kappa']=data['tasks']['kappa_design'].to_numpy(float)*ratios
                data['tasks']['response_ratio_audit']=ratios
                result=Simulator(cfg,data,method='PASI',seed=seed).run()
                s=result['summary'];slot=result['slot_log'];n=int(s['num_assigned']);t=int(s['total_tasks'])
                qualified=int(slot['num_qualified_completed'].sum());payment=float(s['cumulative_payment'])
                row={'seed':seed,'scenario':scenario,'profile':profile,'tasks':t,'assigned':n,'qualified':qualified,
                    'qos_violations':n-qualified,'ir_violations':int(s['ir_violations']),
                    'target_violations':int(s['target_violations']),'coverage':n/max(t,1),'qualified_coverage':qualified/max(t,1),
                    'qos_violation_rate':(n-qualified)/max(n,1),'payment':payment,'pps':payment/max(n,1),
                    'reserved_budget':float(slot.budget_used.sum()),'available_budget':float(slot.budget.sum()),
                    'run_status':result['diagnostics'].get('status')}
                rows.append(row)
                ref=refs.get((seed,scenario,profile))
                if ref:
                    for key,value in row.items():
                        if key in ('seed','scenario','profile','run_status'):continue
                        comparisons.append({'seed':seed,'scenario':scenario,'profile':profile,'field':key,
                            'pass':bool(np.isclose(value,float(ref[key]),rtol=0,atol=1e-8))})
                # Incremental outcomes retain negative QoS/IR results and failures.
                with (out/'outcomes.csv').open('w',encoding='utf-8',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                print(json.dumps(row),flush=True)
    report={'cells':len(rows),'comparisons':comparisons,'passed':all(r['pass'] for r in comparisons),
        'not_reproduced':['historical timing','original shadow audit','MILP certificate instrumentation'],
        'new_population_sample':False,'wrapper_sha256':sha(Path(__file__))}
    (out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['passed']:raise AssertionError('Differences retained; do not replace archived reference data')
def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('verify-package')
    a=sub.add_parser('rebuild');a.add_argument('--raw',type=Path);a.add_argument('--cache',type=Path)
    a.add_argument('--out',type=Path,required=True);a.add_argument('--seeds',type=int,nargs='+',default=list(range(1,31)))
    for command in ('check-inputs','replay'):
        a=sub.add_parser(command);a.add_argument('--episodes',type=Path,required=True)
        a.add_argument('--seeds',type=int,nargs='+',default=list(range(2,31)))
        if command=='replay':a.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.action=='verify-package':
        hashes=read(ROOT/'manifest.json');bad=[n for n,h in hashes.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
        if bad:raise ValueError(bad)
        print(json.dumps({'files_checked':len(hashes),'passed':True,'simulator_runs':0}))
    elif args.action=='check-inputs':
        checks=check(args.episodes,args.seeds);print(json.dumps({'checks':len(checks),'passed':True,'simulator_runs':0}))
    elif args.action=='rebuild':rebuild(args)
    else:replay(args)
if __name__=='__main__':main()
