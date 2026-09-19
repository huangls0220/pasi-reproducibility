"""E5 original 3 x 30 x 2 matrix, repaired core and coverage-first objective."""
from pathlib import Path
import argparse, copy, hashlib, importlib, json, os, shutil, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from unittest.mock import patch
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
CODE=ROOT/'code'
SOURCE=ROOT.parent/'round56-confirmation/code'
R63=ROOT.parent/'round63-e5-diagnostic'
EPISODES=Path('C:/Users/huang/Documents/Codex/2026-08-13/anz/work/experiment-e5/restricted/geolife/episodes')
OLDRESULT=Path('C:/Users/huang/Documents/Codex/2026-08-13/anz/work/experiment-e5/results/geolife/e5_geolife_seed_results.csv')
FILES=('tasks.parquet','providers.parquet','provider_static.parquet','meta.json')
WORKLOADS=('low','medium','high')
METHODS=('MOI','PASI')

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def frame_hash(df):return hashlib.sha256(pd.util.hash_pandas_object(df,index=False).to_numpy().tobytes()).hexdigest()
def code_hashes():return {p.relative_to(CODE).as_posix():sha(p) for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
def key(job):return f"{job['workload']}__seed_{job['seed']:03d}__{job['method']}"

def prepare():
    if (ROOT/'preregistration.json').exists():raise FileExistsError('Do not overwrite freeze')
    assert OLDRESULT.is_file()
    for p in SOURCE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.toml','.yaml','.yml','.json','.txt'):
            dst=CODE/p.relative_to(SOURCE);dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():assert sha(dst)==sha(p),'Existing pre-freeze copy differs'
            else:shutil.copy2(p,dst)
    base=load(R63/'preregistration.json')['config']
    base['matching']['objective']='coverage_first_payment_second'
    assert base['simulation']['log_level']=='full'
    entries={}
    for workload in WORKLOADS:
        for seed in range(1,31):
            folder=EPISODES/workload/f'seed_{seed:03d}';meta=load(folder/'meta.json')
            assert meta['redistributable'] is False and int(meta['n_providers'])==base['providers']['N_mean']
            cfg=copy.deepcopy(base);cfg['dataset']['processed_dir']=str(folder)
            entries[f'{workload}/{seed}']={'episode_path':str(folder),'episode_sha256':{n:sha(folder/n) for n in FILES},
                'config':cfg,'config_sha256':digest(cfg)}
    for method in METHODS:assert load(R63/f'original__{method}.json')['archived_summary_matches']
    jobs=[{'workload':w,'seed':s,'method':m} for w in WORKLOADS for s in range(1,31) for m in METHODS]
    plan={'created_utc':utc(),'round':64,'jobs':jobs,'workers':4,'entries':entries,
        'core_hashes':code_hashes(),'runner_sha256':sha(Path(__file__)),
        'old_result_path':str(OLDRESULT),'old_result_sha256':sha(OLDRESULT),
        'r63_reference_sha256':{m:sha(R63/f'repaired_coverage__{m}.json') for m in METHODS},
        'primary_comparison':'paired within-workload mean PPS saving 100*(MOI-PASI)/MOI; 30 seeds',
        'secondary':'coverage and qualified coverage differences in pp; mean quality; settlement payment and design reserve; target/QoS/IR failures',
        'interval':'10000 paired-seed bootstrap resamples; percentile 95%; RNG seed 20260914 + workload index',
        'interval_scope':'model-seed variability conditional on reused mobility corpus; not participant population or virgin holdout',
        'qos_tolerance':1e-9,'qos_event':'execution_quality < q_min-tol OR total_delay > deadline+tol',
        'failure_policy':'persist errors and all completed cells; do not tune/filter unsuccessful cases',
        'smoke_formal_cells':[j for j in jobs if j['workload']=='medium' and j['seed']==1],
        'candidate_logs_redistributable':False,'references_changed':False,'manuscript_changed':False}
    dump(ROOT/'preregistration.json',plan);print(json.dumps({'frozen':True,'episodes':len(entries),'jobs':len(jobs),'runner':plan['runner_sha256']}))

def verify(plan):
    assert sha(Path(__file__))==plan['runner_sha256']
    assert code_hashes()==plan['core_hashes']
    assert sha(OLDRESULT)==plan['old_result_sha256']
    for e in plan['entries'].values():
        assert digest(e['config'])==e['config_sha256']
        assert {n:sha(Path(e['episode_path'])/n) for n in FILES}==e['episode_sha256']

def run_one(job):
    plan=load(ROOT/'preregistration.json');name=key(job)
    folder=ROOT/'restricted'/name;folder.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(CODE))
    from src.datasets.trace_loader import load_trace_episode
    import src.simulator as simmodule
    entry=plan['entries'][f"{job['workload']}/{job['seed']}"]
    ep=Path(entry['episode_path']);cfg=copy.deepcopy(entry['config'])
    assert {n:sha(ep/n) for n in FILES}==entry['episode_sha256']
    data=load_trace_episode('geolife',cfg,seed=job['seed'],processed_dir=ep)
    fps={k:frame_hash(v) for k,v in data.items() if isinstance(v,pd.DataFrame)}
    simulator=simmodule.Simulator(cfg,data,method=job['method'],seed=job['seed'])
    evaluator=simmodule.evaluate_pairs;original_log=simulator._log_pairs;state={};contexts=[]
    def evaluation(mech,ctx):
        state['ctx']={k:np.array(v,copy=True) if isinstance(v,np.ndarray) else v for k,v in ctx.items()}
        return evaluator(mech,ctx)
    def log(slot,slot_provs,slot_tasks,P,Tk,ev,selected):
        n=len(P);frame={'slot':np.full(n,slot),'provider_id':slot_provs.provider_id.to_numpy()[P],
            'task_id':slot_tasks.task_id.to_numpy()[Tk],'selected':selected.copy()}
        for k,v in state['ctx'].items():
            if np.asarray(v).ndim==0 or np.asarray(v).shape==(n,):frame['ctx_'+k]=np.broadcast_to(v,(n,)).copy()
        contexts.append(pd.DataFrame(frame))
        return original_log(slot,slot_provs,slot_tasks,P,Tk,ev,selected)
    simulator._log_pairs=log;started=time.perf_counter()
    with patch.object(simmodule,'evaluate_pairs',side_effect=evaluation):result=simulator.run()
    elapsed=time.perf_counter()-started
    ctx=pd.concat(contexts,ignore_index=True);ctx.to_parquet(folder/'candidate_contexts.parquet',index=False)
    for k in ('pair_log','slot_log','provider_log'):result[k].to_parquet(folder/f'{k}.parquet',index=False)
    pair=result['pair_log'];slot=result['slot_log'];summary=result['summary'];sel=pair[pair.selected]
    actual=sel.merge(ctx[ctx.selected][['slot','provider_id','task_id','ctx_deadline']],on=['slot','provider_id','task_id'],validate='one_to_one')
    quality_bad=actual.execution_quality<actual.task_min_quality-plan['qos_tolerance']
    deadline_bad=actual.total_delay>actual.ctx_deadline+plan['qos_tolerance']
    qos_bad=quality_bad|deadline_bad;assigned=int(summary['num_assigned']);tasks=int(summary['total_tasks'])
    graph=pair[['slot','provider_id','task_id','spatial_candidate']]
    row={**job,'tasks':tasks,'assigned':assigned,'qualified':assigned-int(qos_bad.sum()),
        'coverage':assigned/max(tasks,1),'qualified_coverage':(assigned-int(qos_bad.sum()))/max(tasks,1),
        'payment':float(summary['cumulative_payment']),'pps':float(summary['cumulative_payment'])/max(assigned,1),
        'mean_quality':float(summary['average_quality']),'high_quality_rate':float(summary['HQR']),
        'platform_utility':float(summary['platform_utility']),
        'quality_violations':int(quality_bad.sum()),'deadline_violations':int(deadline_bad.sum()),'qos_violations':int(qos_bad.sum()),
        'qos_violation_rate':int(qos_bad.sum())/max(assigned,1),
        'target_violations':int(summary['target_violations']),'ir_violations':int(summary['ir_violations']),
        'design_reserve':float(slot.budget_used.sum()),'available_budget':float(slot.budget.sum()),
        'slots_reserve_exceeds_budget':int((slot.budget_used>slot.budget+1e-7).sum()),
        'slots_settlement_exceeds_reserve':int((slot.total_payment>slot.budget_used+1e-7).sum()),
        'slots_settlement_exceeds_budget':int((slot.total_payment>slot.budget+1e-7).sum()),
        'candidate_rows':len(pair),'spatial_candidate_rows':int(pair.spatial_candidate.sum()),
        'legal_rows':int(pair.feasible_contract.sum()),'input_fingerprints':fps,'spatial_graph_sha256':frame_hash(graph),
        'episode_sha256':entry['episode_sha256'],'config_sha256':entry['config_sha256'],
        'mean_runtime_state':float(result['provider_log'].H_before.mean()),
        'diagnostic_status':result['diagnostics'].get('status'),'diagnostics':result['diagnostics'],
        'instrumented_runtime_s':elapsed,'core_protocol_version':summary.get('contract_protocol_version'),
        'decision_cost_basis':summary.get('decision_cost_basis'),
        'restricted_log_sha256':{p.name:sha(p) for p in folder.glob('*.parquet')}}
    (ROOT/'cells').mkdir(exist_ok=True)
    dump(ROOT/'cells'/f'{name}.json',row)
    assert {n:sha(ep/n) for n in FILES}==entry['episode_sha256']
    return row

def smoke():
    plan=load(ROOT/'preregistration.json');verify(plan);checks=[]
    for job in plan['smoke_formal_cells']:
        row=run_one(job);ref=load(R63/f"repaired_coverage__{job['method']}.json")
        assert sha(R63/f"repaired_coverage__{job['method']}.json")==plan['r63_reference_sha256'][job['method']]
        for k in ('tasks','assigned','coverage','payment','pps','mean_quality','target_violations','ir_violations','candidate_rows','spatial_candidate_rows'):
            checks.append({'method':job['method'],'field':k,'pass':bool(np.isclose(row[k],ref[k],rtol=0,atol=1e-8))})
        checks.append({'method':job['method'],'field':'input_fingerprints','pass':row['input_fingerprints']==ref['input_fingerprints']})
        checks.append({'method':job['method'],'field':'QoS','pass':row['qos_violations']==ref['qos_violations_tol1e9']})
    dump(ROOT/'smoke_checks.json',{'passed':all(x['pass'] for x in checks),'checks':checks,'included_in_formal_180':True})
    verify(plan);assert all(x['pass'] for x in checks)
    print(json.dumps({'smoke_passed':True,'checks':len(checks),'formal_cells_completed':2}))

def execute():
    plan=load(ROOT/'preregistration.json');verify(plan)
    assert load(ROOT/'smoke_checks.json')['passed']
    with (ROOT/'execution_started.json').open('x',encoding='utf-8') as f:json.dump({'started':utc(),'plan_sha256':sha(ROOT/'preregistration.json')},f)
    smoke_ids={key(j) for j in plan['smoke_formal_cells']}
    jobs=[j for j in plan['jobs'] if key(j) not in smoke_ids]
    rows=[load(ROOT/'cells'/f'{i}.json') for i in sorted(smoke_ids)];errors=[]
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        futures={pool.submit(run_one,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:rows.append(future.result())
            except Exception:errors.append({'job':job,'error':traceback.format_exc()})
            dump(ROOT/'execution_progress.json',{'completed':len(rows),'expected':180,'errors':errors})
            if len(rows)%10==0 or errors:print(json.dumps({'completed':len(rows),'expected':180,'errors':len(errors)}),flush=True)
    verify(plan)
    fields=[k for k,v in rows[0].items() if not isinstance(v,(dict,list))]
    pd.DataFrame([{k:r[k] for k in fields} for r in rows]).sort_values(['workload','seed','method']).to_csv(ROOT/'seed_results.csv',index=False)
    result={'completed':len(rows),'expected':180,'errors':errors,'finished':utc(),'core_and_inputs_unchanged':True,
        'new_workloads_or_seeds':False,'references_changed':False,'manuscript_changed':False}
    dump(ROOT/'completion.json',result);print(json.dumps(result),flush=True)
    assert len(rows)==180 and not errors,'Keep incomplete/failed evidence, do not claim full confirmation'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','execute']);args=p.parse_args()
    {'prepare':prepare,'smoke':smoke,'execute':execute}[args.action]()
