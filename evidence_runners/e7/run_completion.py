"""Complete original E7 remainder, reuse audited seed001 once, preserve provenance."""
from pathlib import Path
import os
for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[var]='1'
import argparse,copy,hashlib,importlib.util,json,shutil,sys,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;R67=ROOT.parent/'round67-e7-confirmation';R68=ROOT.parent/'round68-e7-equivalence'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('priorrunner',R67/'run_confirmation.py')
batch=importlib.util.module_from_spec(spec);sys.modules[spec.name]=batch;spec.loader.exec_module(batch);batch.ROOT=ROOT
adapter=batch.adapter
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def key(j):return batch.key(j)
def utc():return datetime.now(timezone.utc).isoformat()

def prepare():
    if (ROOT/'preregistration.json').exists():raise FileExistsError('Preserve freeze')
    p67=load(R67/'preregistration.json');p68=load(R68/'preregistration.json')
    assert adapter.hashes(adapter.CURRENT)==p67['current_core']==p68['current_core']
    assert adapter.hashes(adapter.OLD)==p67['old_source']==p68['old_source']
    assert sha(R67/'run_confirmation.py')==p68['r67_runner_sha256']
    assert sha(ROOT.parent/'round66-e7-adapter/run_adapter.py')==p68['adapter_sha256']
    frozen=pd.read_csv(R68/'next_original_confirmation_jobs.csv').fillna('')
    jobs=frozen[['cell_id','seed','method','original_stage']].to_dict('records');assert len(jobs)==1680
    configurations=copy.deepcopy(p68['configs']);episodes=copy.deepcopy(p67['episodes'])
    checks=[];reuse={};cache={}
    for _,entry in frozen[frozen.seed==1].iterrows():
        job={'cell_id':entry.cell_id,'seed':1,'method':entry.method};name=key(job)
        original=R68/'current/cells'/f'{name}.json';row=load(original);cell=configurations[entry.cell_id]['cell']
        ep=episodes[f"{cell['workload']}/1"];folder=Path(ep['path']);cfg=copy.deepcopy(configurations[entry.cell_id]['config'])
        cfg['dataset']={'processed_dir':str(folder),'use_region':'all','service_radius_km':3.}
        if entry.cell_id not in cache:
            data=adapter.oldrunner.load_trace_episode('geolife',cfg,seed=1,processed_dir=folder)
            data['tasks']=adapter.oldrunner.apply_qos(data['tasks'],cell['qos'])
            fps={k:batch.framehash(v) for k,v in data.items() if isinstance(v,pd.DataFrame)}
            phase=np.random.default_rng([1,9707]).integers(0,5*cell['duration'],size=len(data['provider_static']))
            cache[entry.cell_id]=(fps,hashlib.sha256(phase.tobytes()).hexdigest())
        fps,phasehash=cache[entry.cell_id];logs=R68/'current/restricted'/name
        pair=pd.read_parquet(logs/'pair_log.parquet')
        c={'result_matches_R68_frozen_reuse_hash':sha(original)==entry.prior_result_sha256,
            'configuration':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==row['config_sha256'],
            'regenerated_inputs':fps==row['input_fingerprints'],
            'window_phases':phasehash==row['window_phase_sha256'],
            'graph':batch.framehash(pair[['slot','provider_id','task_id','spatial_candidate']])==row['candidate_graph_sha256'],
            'log_hashes':all(sha(logs/n)==h for n,h in row['restricted_log_sha256'].items()),
            'execution_completed':row['execution_status']=='completed'}
        checks.append({**job,**c,'all_passed':all(c.values())})
        reuse[name]={'original_result':str(original),'result_sha256':sha(original),'restricted_logs':str(logs)}
    pd.DataFrame(checks).to_csv(ROOT/'reuse_checks.csv',index=False)
    assert len(checks)==120 and all(c['all_passed'] for c in checks)
    jobsframe=pd.DataFrame(jobs);jobsframe['execution_source']=np.where(jobsframe.seed==1,'R68_verified_reuse','R69_new_run')
    jobsframe.to_csv(ROOT/'frozen_jobs.csv',index=False)
    plan={'round':69,'created':utc(),'jobs':jobs,'configs':configurations,'episodes':episodes,'reuse':reuse,'workers':4,
        'runner_sha256':sha(Path(__file__)),'r67_runner_sha256':sha(R67/'run_confirmation.py'),
        'adapter_sha256':p68['adapter_sha256'],'current_core':p68['current_core'],'old_source':p68['old_source'],
        'original_files':p67['original_files'],'r67_downward_results_sha256':sha(R67/'seed_results.csv'),
        'r68_next_jobs_sha256':sha(R68/'next_original_confirmation_jobs.csv'),
        'reuse_checks_sha256':sha(ROOT/'reuse_checks.csv'),'jobs_sha256':sha(ROOT/'frozen_jobs.csv'),
        'analysis':'same R67 per-cell paired PPS/payment/SC/QSC/QoS descriptive comparisons;10000 paired-seed bootstrap; seed20260915+cell numeric ID',
        'interval_scope':'conditional reused-trace model-seed variation, not real-participant risk or an independent new validation sample',
        'failure_policy':'retain every outcome including failed service diagnostics; no retuning/filtering',
        'references_changed':False,'manuscript_changed':False,'new_cells_or_seeds':False}
    dump(ROOT/'preregistration.json',plan)
    for name,e in reuse.items():
        dst=ROOT/'cells'/f'{name}.json';dst.parent.mkdir(exist_ok=True);shutil.copy2(e['original_result'],dst)
        assert sha(dst)==e['result_sha256']
    verify(plan)
    print(json.dumps({'jobs':1680,'reuse_verified':120,'reuse_checks':840,'new_runs':1560}),flush=True)

def verify(plan):
    assert sha(Path(__file__))==plan['runner_sha256']
    assert sha(R67/'run_confirmation.py')==plan['r67_runner_sha256']
    assert sha(ROOT.parent/'round66-e7-adapter/run_adapter.py')==plan['adapter_sha256']
    assert adapter.hashes(adapter.CURRENT)==plan['current_core'] and adapter.hashes(adapter.OLD)==plan['old_source']
    assert sha(R67/'seed_results.csv')==plan['r67_downward_results_sha256']
    assert all(sha(batch.OLD/n)==h for n,h in plan['original_files'].items())
    for e in plan['episodes'].values():assert {p.name:sha(p) for p in Path(e['path']).iterdir() if p.is_file()}==e['sha256']
    for name,e in plan['reuse'].items():
        assert sha(Path(e['original_result']))==e['result_sha256']==sha(ROOT/'cells'/f'{name}.json')

def execute():
    plan=load(ROOT/'preregistration.json');verify(plan)
    with (ROOT/'execution_started.json').open('x',encoding='utf-8') as f:json.dump({'started':utc(),'plan_sha256':sha(ROOT/'preregistration.json')},f)
    rows=[load(ROOT/'cells'/f'{name}.json') for name in sorted(plan['reuse'])];errors=[]
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        futures={pool.submit(batch.run_one,j):j for j in plan['jobs'] if key(j) not in plan['reuse']}
        for future in as_completed(futures):
            job=futures[future]
            try:rows.append(future.result())
            except Exception:errors.append({'job':job,'error':traceback.format_exc()})
            dump(ROOT/'execution_progress.json',{'completed_total':len(rows),'reused':120,'new_completed':len(rows)-120,'expected_total':1680,'errors':errors})
            if len(rows)%50==0 or errors:print(json.dumps({'completed_total':len(rows),'new_completed':len(rows)-120,'expected_total':1680,'errors':len(errors)}),flush=True)
    verify(plan)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]).sort_values(['cell_id','seed','method']).to_csv(ROOT/'seed_results.csv',index=False)
    dump(ROOT/'completion.json',{'completed':len(rows),'expected':1680,'reused':120,'new_completed':len(rows)-120,'errors':errors,'finished':utc(),
        'inputs_and_cores_unchanged':True,'references_changed':False,'manuscript_changed':False})
    assert len(rows)==1680 and not errors

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','execute']);a=p.parse_args();{'prepare':prepare,'execute':execute}[a.action]()
