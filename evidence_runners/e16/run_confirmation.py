"""Exact original E16 matrix under the repaired core; full local audit logs."""
from pathlib import Path
import argparse, copy, hashlib, json, os, shutil, sys, time, traceback
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[v]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;CODE=ROOT/'code';SOURCE=ROOT.parent/'round56-confirmation/code'
EPISODES=Path('C:/Users/huang/Documents/Codex/2026-08-13/anz/work/experiment-e5/restricted/geolife/episodes')
HIST=Path('C:/Users/huang/Documents/Codex/2026-08-13/anz/work/round20-scientific-boundaries/results/e16_quac_public_baseline/e16_formal_seed_results.csv')
OLDROOT=Path('C:/Users/huang/Documents/Codex/2026-08-13/anz/work/round20-scientific-boundaries/experiment-repo')
FILES=('tasks.parquet','providers.parquet','provider_static.parquet','meta.json');WORKLOADS=('low','medium','high');METHODS=('PASI','MOI','QUAC-F')
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def framehash(df):return hashlib.sha256(pd.util.hash_pandas_object(df,index=False).to_numpy().tobytes()).hexdigest()
def codehashes():return {p.relative_to(CODE).as_posix():sha(p) for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
def key(j):return f"{j['workload']}__seed_{j['seed']:03d}__{j['method']}"

def prepare():
    if (ROOT/'formal_preregistration.json').exists():raise FileExistsError('Preserve formal freeze')
    diagnostic=load(ROOT/'preregistration.json')
    assert load(ROOT/'diagnostic_old_completion.json')['completed']==9 and load(ROOT/'diagnostic_current_completion.json')['completed']==9
    for p in SOURCE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.toml','.yaml','.yml','.json','.txt'):
            t=CODE/p.relative_to(SOURCE);t.parent.mkdir(parents=True,exist_ok=True)
            if t.exists():assert sha(t)==sha(p)
            else:shutil.copy2(p,t)
    sys.path.insert(0,str(CODE));from scripts.run_e8_robustness import configure,tape_hash
    control={'scenario':'control','factor':'control','level':0.0};entries={}
    for w in WORKLOADS:
        for s in range(1,31):
            folder=EPISODES/w/f'seed_{s:03d}';meta=load(folder/'meta.json');assert meta['redistributable'] is False
            cfg=configure(control,int(meta['n_providers']));cfg['dataset']={'processed_dir':str(folder),'use_region':'all','service_radius_km':3.0}
            cfg['matching']['objective']='coverage_first_payment_second';cfg['simulation']['log_level']='full'
            entries[f'{w}/{s}']={'path':str(folder),'episode_sha256':{n:sha(folder/n) for n in FILES},'tape_hash':tape_hash(folder),'config':cfg,'config_sha256':digest(cfg)}
    hist=pd.read_csv(HIST);assert len(hist)==270 and not hist.duplicated(['workload','seed','method']).any()
    jobs=[{'workload':w,'seed':s,'method':m} for w in WORKLOADS for s in range(1,31) for m in METHODS]
    assert set(map(tuple,hist[['workload','seed','method']].to_numpy()))==set((j['workload'],j['seed'],j['method']) for j in jobs)
    plan={'round':70,'created_utc':utc(),'jobs':jobs,'workers':4,'entries':entries,'core_hashes':codehashes(),
          'runner_sha256':sha(Path(__file__)),'historical_result_path':str(HIST),'historical_result_sha256':sha(HIST),
          'old_e16_runner_sha256':diagnostic['old_e16_runner_sha256'],'current_e16_runner_sha256':diagnostic['current_e16_runner_sha256'],
          'gate':'seed001 exact equality but no all-seed proof because historical candidate logs absent and decision-cost semantics changed',
          'primary':'exact paired current-vs-historical non-runtime outcome compatibility across original270 keys',
          'secondary':'current full-log outcome reconstruction; design-vs-settlement cost; free-effort/QoS; paired method comparisons',
          'interval':'same historical bootstrap function/seeds for direct summary comparison plus explicit paired differences',
          'failure_policy':'retain runtime exceptions and all service QoS/IR/target/budget failures; no retuning/filtering',
          'smoke_jobs':[{'workload':'medium','seed':1,'method':m} for m in METHODS],
          'scope':'reused GeoLife mobility and modeled economic variables; not new participants or risk certification',
          'candidate_logs_redistributable':False,'new_cells_or_seeds':False,'references_changed':False,'manuscript_changed':False,'github_changed':False}
    dump(ROOT/'formal_preregistration.json',plan);print(json.dumps({'formal_jobs':270,'episodes':90,'workers':4,'new_cells_or_seeds':False}))

def verify(plan):
    assert sha(Path(__file__))==plan['runner_sha256'] and codehashes()==plan['core_hashes'] and sha(HIST)==plan['historical_result_sha256']
    for e in plan['entries'].values():assert {n:sha(Path(e['path'])/n) for n in FILES}==e['episode_sha256'] and digest(e['config'])==e['config_sha256']

def run_one(job):
    plan=load(ROOT/'formal_preregistration.json');name=key(job);folder=ROOT/'restricted'/name;folder.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(CODE));from src.datasets.trace_loader import load_trace_episode;import src.simulator as sm
    entry=plan['entries'][f"{job['workload']}/{job['seed']}"];ep=Path(entry['path']);cfg=copy.deepcopy(entry['config'])
    data=load_trace_episode('geolife',cfg,seed=job['seed'],processed_dir=ep);fps={k:framehash(v) for k,v in data.items() if isinstance(v,pd.DataFrame)}
    simulator=sm.Simulator(cfg,data,method=job['method'],seed=job['seed']);evaluator=sm.evaluate_pairs;original_log=simulator._log_pairs;state={};contexts=[]
    def evaluation(mech,ctx):state['ctx']={k:np.array(v,copy=True) if isinstance(v,np.ndarray) else v for k,v in ctx.items()};return evaluator(mech,ctx)
    def log_pairs(slot,slot_provs,slot_tasks,P,Tk,ev,selected):
        n=len(P);f={'slot':np.full(n,slot),'provider_id':slot_provs.provider_id.to_numpy()[P],'task_id':slot_tasks.task_id.to_numpy()[Tk],'selected':selected.copy()}
        for k,v in state['ctx'].items():
            a=np.asarray(v)
            if a.ndim==0 or a.shape==(n,):f['ctx_'+k]=np.broadcast_to(v,(n,)).copy()
        contexts.append(pd.DataFrame(f));return original_log(slot,slot_provs,slot_tasks,P,Tk,ev,selected)
    simulator._log_pairs=log_pairs;started=time.perf_counter()
    with patch.object(sm,'evaluate_pairs',side_effect=evaluation):result=simulator.run()
    elapsed=time.perf_counter()-started;ctx=pd.concat(contexts,ignore_index=True)
    ctx.to_parquet(folder/'candidate_contexts.parquet',index=False)
    for n in ('pair_log','slot_log','provider_log'):result[n].to_parquet(folder/f'{n}.parquet',index=False)
    pair=result['pair_log'];slot=result['slot_log'];summary=result['summary'];sel=pair[pair.selected];assigned=int(summary['num_assigned']);qualified=int(slot.num_qualified_completed.sum())
    total_edges=float((slot.num_providers*slot.num_tasks).sum());feasible_edges=float(slot.num_feasible_pairs.sum())
    row={**job,'N':int(load(ep/'meta.json')['n_providers']),'M':4,'tasks':int(summary['total_tasks']),'assigned':assigned,'qualified':qualified,
         'payment':float(summary['cumulative_payment']),'pps':float(summary['cumulative_payment'])/max(assigned,1),'service_coverage':float(summary['assignment_ratio']),
         'mean_quality':float(slot.mean_quality.dropna().mean()),'qos_violations':assigned-qualified,'qos_violation_rate':(assigned-qualified)/max(assigned,1),
         'ir_violations':int(summary['ir_violations']),'under_incentive_rate':int(summary['ir_violations'])/max(assigned,1),
         'target_violations':int(summary['target_violations']),'target_miss_rate':int(summary['target_violations'])/max(assigned,1),
         'legal_edge_rate':feasible_edges/max(total_edges,1.0),'platform_utility':float(summary['platform_utility']),
         'mean_base_payment':float(sel.base_payment.mean()) if len(sel) else float('nan'),'negative_intercept_rate':float((sel.base_payment<0).mean()) if len(sel) else float('nan'),
         'core_runtime_s':float(summary['total_runtime']),'instrumented_runtime_s':elapsed,'event_tape_hash':entry['tape_hash'],
         'diagnostic_status':result['diagnostics'].get('status','unknown'),'execution_status':result['diagnostics'].get('execution_status','completed'),
         'decision_cost_basis':summary.get('decision_cost_basis'),'contract_protocol_version':summary.get('contract_protocol_version'),
         'design_reserve':float(slot.budget_used.sum()),'available_budget':float(slot.budget.sum()),
         'reserve_over_budget_slots':int((slot.budget_used>slot.budget+1e-7).sum()),'settlement_over_reserve_slots':int((slot.total_payment>slot.budget_used+1e-7).sum()),
         'settlement_over_budget_slots':int((slot.total_payment>slot.budget+1e-7).sum()),'candidate_rows':len(pair),'legal_rows':int(pair.feasible_contract.sum()),
         'input_fingerprints':fps,'candidate_graph_sha256':framehash(pair[['slot','provider_id','task_id','spatial_candidate']]),
         'config_sha256':entry['config_sha256'],'episode_sha256':entry['episode_sha256'],'diagnostics':result['diagnostics']}
    row['restricted_log_sha256']={p.name:sha(p) for p in folder.glob('*.parquet')};dump(ROOT/'cells'/f'{name}.json',row);return row

def smoke():
    plan=load(ROOT/'formal_preregistration.json');verify(plan);ref=pd.read_csv(ROOT/'diagnostic_current_seed001.csv').set_index(['workload','seed','method']);checks=[]
    metrics=('payment','pps','service_coverage','mean_quality','qos_violation_rate','under_incentive_rate','target_miss_rate','legal_edge_rate','platform_utility','mean_base_payment','negative_intercept_rate')
    for j in plan['smoke_jobs']:
        r=run_one(j);old=ref.loc[(j['workload'],j['seed'],j['method'])]
        for m in metrics:checks.append({'method':j['method'],'metric':m,'passed':bool(np.isclose(r[m],old[m],rtol=0,atol=1e-10,equal_nan=True))})
    dump(ROOT/'smoke_checks.json',{'checks':checks,'all_passed':all(x['passed'] for x in checks),'included_in_formal270':True});verify(plan);assert all(x['passed'] for x in checks)
    print(json.dumps({'smoke_jobs':3,'checks':len(checks),'all_passed':True}))

def execute():
    plan=load(ROOT/'formal_preregistration.json');verify(plan);assert load(ROOT/'smoke_checks.json')['all_passed']
    with (ROOT/'execution_started.json').open('x',encoding='utf-8') as f:json.dump({'started_utc':utc(),'plan_sha256':sha(ROOT/'formal_preregistration.json')},f)
    done={key(j) for j in plan['smoke_jobs']};rows=[load(ROOT/'cells'/f'{n}.json') for n in sorted(done)];errors=[]
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        futures={pool.submit(run_one,j):j for j in plan['jobs'] if key(j) not in done}
        for fut in as_completed(futures):
            try:rows.append(fut.result())
            except Exception:errors.append({'job':futures[fut],'error':traceback.format_exc()})
            dump(ROOT/'execution_progress.json',{'completed':len(rows),'expected':270,'errors':errors})
            if len(rows)%10==0 or errors:print(json.dumps({'completed':len(rows),'expected':270,'errors':len(errors)}),flush=True)
    verify(plan);fields=[k for k,v in rows[0].items() if not isinstance(v,(dict,list))]
    pd.DataFrame([{k:r[k] for k in fields} for r in rows]).sort_values(['workload','seed','method']).to_csv(ROOT/'seed_results.csv',index=False)
    result={'completed':len(rows),'expected':270,'errors':errors,'finished_utc':utc(),'code_and_inputs_unchanged':True,'new_cells_or_seeds':False,'references_changed':False,'manuscript_changed':False}
    dump(ROOT/'completion.json',result);print(json.dumps(result),flush=True);assert len(rows)==270 and not errors

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','execute']);a=p.parse_args();{'prepare':prepare,'smoke':smoke,'execute':execute}[a.action]()
