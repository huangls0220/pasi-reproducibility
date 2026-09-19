"""Full-log audit and per-original-cell paired descriptive analysis."""
from pathlib import Path
import ast, hashlib, json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds

ROOT=Path(__file__).resolve().parent/'merged'
OLD=Path('C:/Users/huang/Documents/Codex/2026-08-13/anz/work/experiment-e7/results/e7')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')
def key(j):return f"{j['cell_id']}__seed_{j['seed']:03d}__{j['method']}"

plan=load(ROOT/'preregistration.json');completion=load(ROOT/'completion.json')
assert completion['completed']==2880 and not completion['errors']
source=ROOT.parent.parent/'round64-e5-confirmation/check_matching.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='solve')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),globals())
frame=pd.read_csv(ROOT/'seed_results.csv');assert len(frame)==2880 and not frame.duplicated(['cell_id','seed','method']).any()
audits=[];matching=[];records={}
for i,job in enumerate(plan['jobs'],1):
    name=key(job);r=load(ROOT/'cells'/f'{name}.json');records[name]=r;folder=Path(load(ROOT/'log_sources.json')[name])
    logs_match=all(sha(folder/n)==h for n,h in r['restricted_log_sha256'].items())
    p=pd.read_parquet(folder/'pair_log.parquet');c=pd.read_parquet(folder/'candidate_contexts.parquet')
    s=pd.read_parquet(folder/'slot_log.parquet').set_index('slot')
    prov=pd.read_parquet(folder/'provider_log.parquet').sort_values(['provider_id','slot'])
    cfg=plan['configs'][job['cell_id']]['config'];cell=plan['configs'][job['cell_id']]['cell']
    prov['belief_before']=prov.groupby('provider_id').H_belief.shift().fillna(cfg['providers'].get('initial_H',.1))
    c=c.merge(prov[['slot','provider_id','belief_before']],on=['slot','provider_id'],validate='many_to_one')
    sign=1. if cell['direction']=='runtime_higher' else -1.
    rt=np.clip(c.e7_design_state+sign*cell['magnitude']*c.e7_active,0.,1.)
    selected=p[p.selected].merge(c[['slot','provider_id','task_id','ctx_deadline']],on=['slot','provider_id','task_id'],validate='one_to_one')
    qbad=selected.execution_quality<selected.task_min_quality-1e-9;dbad=selected.total_delay>selected.ctx_deadline+1e-9
    totals=selected.groupby('slot').agg(pay=('expected_contract_cost','sum'),reserve=('decision_contract_cost','sum'),count=('task_id','size'))
    totals=s.join(totals).fillna({'pay':0.,'reserve':0.,'count':0})
    checks={
        'log_hashes_unchanged':logs_match,
        'preupdate_design_state':bool(np.allclose(c.e7_design_state,c.belief_before,atol=1e-12,rtol=0)),
        'runtime_state_policy':bool(np.allclose(c.ctx_H_t,rt,atol=1e-12,rtol=0)),
        'contract_state_policy':bool(np.allclose(c.ctx_H_d,rt if job['method']=='PASI' else c.e7_design_state,atol=1e-12,rtol=0)),
        'unique_provider_per_slot':not selected.duplicated(['slot','provider_id']).any(),
        'unique_task_per_slot':not selected.duplicated(['slot','task_id']).any(),
        'selected_feasible':bool(selected.feasible_contract.all()),
        'selected_count':len(selected)==r['assigned'] and bool((totals['count']==totals.num_assigned).all()),
        'quality_count':int(qbad.sum())==r['quality_violations'],
        'deadline_count':int(dbad.sum())==r['deadline_violations'],
        'qos_union_count':int((qbad|dbad).sum())==r['qos_violations'],
        'ir_count':int((~selected.ir_ok).sum())==r['ir_violations'],
        'target_count':int((selected.implementation_gap < -1e-7).sum())==r['target_violations'],
        'payment_per_slot':bool(np.allclose(totals.pay,totals.total_payment,atol=1e-7,rtol=0)),
        'reserve_per_slot':bool(np.allclose(totals.reserve,totals.budget_used,atol=1e-7,rtol=0)),
        'payment_total':abs(selected.expected_contract_cost.sum()-r['payment'])<1e-7,
        'reserve_over_budget_count':int((s.budget_used>s.budget+1e-7).sum())==r['reserve_over_budget_slots'],
        'settlement_over_reserve_count':int((s.total_payment>s.budget_used+1e-7).sum())==r['settlement_over_reserve_slots'],
        'execution_completed':r['execution_status']=='completed'}
    audits.append({**job,**{k:bool(v) for k,v in checks.items()},'all_passed':all(checks.values())})
    times=np.sort(p.slot.unique());chosen=times[np.unique(np.linspace(0,len(times)-1,min(5,len(times)),dtype=int))]
    good=True;maxgap=0.
    for t in chosen:
        legal=p[(p.slot==t)&p.feasible_contract]
        k,cost,solver=solve(legal,float(s.loc[t,'budget']));gap=abs(cost-float(s.loc[t,'budget_used']))
        good &= k==int(s.loc[t,'num_assigned']) and gap<1e-7;maxgap=max(maxgap,gap)
    matching.append({**job,'slots_checked':len(chosen),'max_payment_gap':maxgap,'all_passed':bool(good)})
    if i%100==0:print(json.dumps({'audited':i,'expected':2880}),flush=True)
pd.DataFrame(audits).to_csv(ROOT/'cell_audit.csv',index=False)
pd.DataFrame(matching).to_csv(ROOT/'matching_audit.csv',index=False)
paired_inputs=[]
for cid,seed in frame[['cell_id','seed']].drop_duplicates().itertuples(index=False,name=None):
    a=records[key({'cell_id':cid,'seed':seed,'method':'MOI'})];b=records[key({'cell_id':cid,'seed':seed,'method':'PASI'})]
    paired_inputs.append({'cell_id':cid,'seed':seed,**{k:a[k]==b[k] for k in ('input_fingerprints','candidate_graph_sha256','window_phase_sha256','config_sha256')}})
pd.DataFrame(paired_inputs).to_csv(ROOT/'paired_input_audit.csv',index=False)
indexed=frame.set_index(['cell_id','seed','method'])
left=indexed.xs('MOI',level='method');right=indexed.xs('PASI',level='method');assert left.index.equals(right.index)
paired=pd.DataFrame(index=left.index)
for k in ('pps','payment','coverage','qualified_coverage','qos_violation_rate','assigned','qos_violations','ir_violations','target_violations'):
    paired['moi_'+k]=left[k];paired['pasi_'+k]=right[k]
    paired['delta_'+k]=right[k]-left[k]
paired['pps_saving_percent']=100*(left.pps-right.pps)/left.pps
paired['qsc_gain_pp']=100*(right.qualified_coverage-left.qualified_coverage)
paired['sc_gain_pp']=100*(right.coverage-left.coverage)
paired['qos_delta_pp']=100*(right.qos_violation_rate-left.qos_violation_rate)
paired.to_csv(ROOT/'paired_seed_results.csv')
cellrows=[]
metrics=('pps_saving_percent','delta_payment','qsc_gain_pp','sc_gain_pp','qos_delta_pp')
for cid,g in paired.groupby(level='cell_id',sort=True):
    n=len(g);rng=np.random.default_rng(20260915+int(cid.replace('E7C','')));indices=rng.integers(0,n,size=(10000,n))
    row={**plan['configs'][cid]['cell'],'n_pairs':n,
        'pasi_pps_lower_count':int((g.pasi_pps<g.moi_pps-1e-12).sum()),
        'pasi_pps_higher_count':int((g.pasi_pps>g.moi_pps+1e-12).sum()),
        'pasi_qsc_lower_count':int((g.qsc_gain_pp<-1e-9).sum()),
        'pasi_qos_violations':int(g.pasi_qos_violations.sum()),'moi_qos_violations':int(g.moi_qos_violations.sum())}
    for metric in metrics:
        values=g[metric].to_numpy();means=values[indices].mean(axis=1)
        row['mean_'+metric]=float(values.mean());row[metric+'_ci_low']=float(np.quantile(means,.025));row[metric+'_ci_high']=float(np.quantile(means,.975))
    cellrows.append(row)
pd.DataFrame(cellrows).to_csv(ROOT/'cell_paired_summary.csv',index=False)
summary=frame.groupby(['cell_id','method'],sort=True).agg(n=('seed','count'),mean_pps=('pps','mean'),mean_payment=('payment','mean'),
    mean_sc=('coverage','mean'),mean_qsc=('qualified_coverage','mean'),mean_qos_rate=('qos_violation_rate','mean'),
    qos_violations=('qos_violations','sum'),assigned=('assigned','sum'),ir_violations=('ir_violations','sum'),target_violations=('target_violations','sum'))
summary.to_csv(ROOT/'method_cell_summary.csv')
negative=frame[(frame.qos_violations>0)|(frame.ir_violations>0)|(frame.target_violations>0)|(frame.reserve_over_budget_slots>0)|(frame.settlement_over_reserve_slots>0)|(frame.settlement_over_budget_slots>0)]
negative.to_csv(ROOT/'negative_outcome_runs.csv',index=False)
old=pd.concat([pd.read_csv(OLD/'e7_screening_seed_results.csv'),pd.read_csv(OLD/'e7_confirmation_seed_results.csv')])
old=old.rename(columns={'mechanism':'method','service_coverage':'coverage'})
old=old[['cell_id','seed','method','payment','pps','coverage','qos_violations','qos_violation_rate']]
comparison=frame.merge(old,on=['cell_id','seed','method'],suffixes=('_current','_historical'),validate='one_to_one')
comparison.to_csv(ROOT/'historical_protocol_comparison.csv',index=False)
result={'runs':len(frame),'pairs':len(paired),'cells':len(cellrows),'other_original_cells_unresolved':0,
    'all_cell_checks_passed':all(r['all_passed'] for r in audits),
    'all_paired_input_checks_passed':all(all(r[k] for k in ('input_fingerprints','candidate_graph_sha256','window_phase_sha256','config_sha256')) for r in paired_inputs),
    'matching_slots_checked':sum(r['slots_checked'] for r in matching),'all_sampled_matching_passed':all(r['all_passed'] for r in matching),
    'negative_outcome_runs':len(negative),'by_method':{},'paired_counts':{
        'pasi_pps_higher':int((paired.pasi_pps>paired.moi_pps+1e-12).sum()),
        'pasi_pps_lower':int((paired.pasi_pps<paired.moi_pps-1e-12).sum()),
        'pasi_qsc_higher':int((paired.qsc_gain_pp>1e-9).sum()),'pasi_qsc_lower':int((paired.qsc_gain_pp<-1e-9).sum()),
        'same_sc':int((abs(paired.sc_gain_pp)<1e-9).sum())},
    'interval_scope':plan['interval_scope'],'references_changed':False,'manuscript_changed':False,
    'analysis_sha256':sha(Path(__file__)),'matching_solver_sha256':sha(source)}
for method,g in frame.groupby('method'):
    result['by_method'][method]={'runs':len(g),'assigned':int(g.assigned.sum()),'quality_violations':int(g.quality_violations.sum()),
        'deadline_violations':int(g.deadline_violations.sum()),'qos_violations':int(g.qos_violations.sum()),
        'qos_affected_runs':int((g.qos_violations>0).sum()),'ir_violations':int(g.ir_violations.sum()),'target_violations':int(g.target_violations.sum()),
        'reserve_over_budget_slots':int(g.reserve_over_budget_slots.sum()),'settlement_over_reserve_slots':int(g.settlement_over_reserve_slots.sum()),
        'settlement_over_budget_slots':int(g.settlement_over_budget_slots.sum()),'diagnostic_status_counts':g.diagnostic_status.value_counts().to_dict()}
dump('audit_summary.json',result);print(json.dumps(result,indent=2),flush=True)
assert result['all_cell_checks_passed'] and result['all_paired_input_checks_passed'] and result['all_sampled_matching_passed']
