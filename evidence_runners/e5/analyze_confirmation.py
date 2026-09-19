"""Audit all completed cells and summarize paired model-seed variability."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=load(ROOT/'preregistration.json');done=load(ROOT/'completion.json')
assert done['completed']==180 and not done['errors']
frame=pd.read_csv(ROOT/'seed_results.csv')
assert len(frame)==180 and not frame.duplicated(['workload','seed','method']).any()
cells={};audits=[];endpoints=[]
for job in plan['jobs']:
    name=f"{job['workload']}__seed_{job['seed']:03d}__{job['method']}"
    r=load(ROOT/'cells'/f'{name}.json');cells[(job['workload'],job['seed'],job['method'])]=r
    folder=ROOT/'restricted'/name
    assert {n:sha(folder/n) for n in r['restricted_log_sha256']}==r['restricted_log_sha256']
    pair=pd.read_parquet(folder/'pair_log.parquet');slot=pd.read_parquet(folder/'slot_log.parquet')
    ctx=pd.read_parquet(folder/'candidate_contexts.parquet');sel=pair[pair.selected]
    slot_selected=sel.groupby('slot').agg(n=('task_id','size'),pay=('expected_contract_cost','sum'),reserve=('decision_contract_cost','sum'))
    s=slot.set_index('slot').join(slot_selected).fillna({'n':0,'pay':0.,'reserve':0.})
    checks={
        'input_episode_frozen':r['episode_sha256']==plan['entries'][f"{job['workload']}/{job['seed']}"]['episode_sha256'],
        'selected_edges_legal':bool(sel.feasible_contract.all()),
        'selected_edges_in_spatial_graph':bool(sel.spatial_candidate.all()),
        'provider_capacity_one':not sel.duplicated(['slot','provider_id']).any(),
        'task_capacity_one':not sel.duplicated(['slot','task_id']).any(),
        'per_slot_count_recomputes':bool((s.n==s.num_assigned).all()),
        'per_slot_settlement_recomputes':bool(np.allclose(s.pay,s.total_payment,rtol=0,atol=1e-8)),
        'per_slot_design_cost_recomputes':bool(np.allclose(s.reserve,s.budget_used,rtol=0,atol=1e-8)),
        'payment_total_recomputes':bool(np.isclose(sel.expected_contract_cost.sum(),r['payment'],rtol=0,atol=1e-8)),
        'selected_count_recomputes':len(sel)==r['assigned'],
        'count_consistency':0<=r['qualified']<=r['assigned']<=r['tasks'],
    }
    audits.append({**job,**checks})
    # Observed endpoint support, not a certificate for unobserved contexts.
    legal=pair.feasible_contract.to_numpy(bool)
    assert ctx[['slot','provider_id','task_id']].equals(pair[['slot','provider_id','task_id']])
    margin=ctx.ctx_V*ctx.ctx_q_bar*ctx.ctx_kappa*np.exp(-ctx.ctx_kappa)-ctx.ctx_L*(ctx.ctx_alpha+2*ctx.ctx_beta)
    endpoints.append({**job,'legal_rows':int(legal.sum()),
        'effort_one_rows':int((np.abs(pair.a_star.to_numpy()[legal]-1)<=1e-9).sum()),
        'target_one_rows':int((np.abs(pair.a_target.to_numpy()[legal]-1)<=1e-9).sum()),
        'min_system_derivative_at_one_legal':float(margin.to_numpy()[legal].min())})
assert all(all(v for k,v in r.items() if k not in ('workload','seed','method')) for r in audits)
pair_checks=[]
for w in ('low','medium','high'):
    for seed in range(1,31):
        a=cells[(w,seed,'MOI')];b=cells[(w,seed,'PASI')]
        checks={'workload':w,'seed':seed,'loaded_inputs_equal':a['input_fingerprints']==b['input_fingerprints'],
            'spatial_graph_equal':a['spatial_graph_sha256']==b['spatial_graph_sha256'],
            'config_equal':a['config_sha256']==b['config_sha256']}
        pair_checks.append(checks)
assert all(r['loaded_inputs_equal'] and r['spatial_graph_equal'] and r['config_equal'] for r in pair_checks)

means=[];paired=[];contrasts=[]
metrics=('pps','payment','coverage','qualified_coverage','mean_quality','qos_violation_rate')
for wi,w in enumerate(('low','medium','high')):
    work=frame[frame.workload==w];moi=work[work.method=='MOI'].set_index('seed').sort_index();pasi=work[work.method=='PASI'].set_index('seed').sort_index()
    assert list(moi.index)==list(range(1,31)) and moi.index.equals(pasi.index)
    indices=np.random.default_rng(20260914+wi).integers(0,30,(10000,30))
    def ci(v):return np.quantile(np.asarray(v)[indices].mean(axis=1),[.025,.975]).tolist()
    for method,sub in [('MOI',moi),('PASI',pasi)]:
        row={'workload':w,'method':method,'n':30}
        for metric in metrics:
            v=sub[metric].to_numpy(float);row['mean_'+metric]=float(v.mean());row[metric+'_ci_low'],row[metric+'_ci_high']=ci(v)
        for metric in ('tasks','assigned','qualified','quality_violations','deadline_violations','qos_violations','ir_violations','target_violations','slots_reserve_exceeds_budget','slots_settlement_exceeds_reserve','slots_settlement_exceeds_budget'):
            row['total_'+metric]=int(sub[metric].sum())
        row['pooled_qos_violation_rate']=row['total_qos_violations']/max(row['total_assigned'],1)
        means.append(row)
    arrays={'pps_saving_percent':100*(moi.pps.to_numpy()-pasi.pps.to_numpy())/moi.pps.to_numpy(),
        'payment_saving_percent':100*(moi.payment.to_numpy()-pasi.payment.to_numpy())/moi.payment.to_numpy(),
        'coverage_delta_pp':100*(pasi.coverage.to_numpy()-moi.coverage.to_numpy()),
        'qualified_coverage_delta_pp':100*(pasi.qualified_coverage.to_numpy()-moi.qualified_coverage.to_numpy()),
        'quality_delta':pasi.mean_quality.to_numpy()-moi.mean_quality.to_numpy(),
        'qos_delta_pp':100*(pasi.qos_violation_rate.to_numpy()-moi.qos_violation_rate.to_numpy())}
    for i,seed in enumerate(range(1,31)):paired.append({'workload':w,'seed':seed,**{k:float(v[i]) for k,v in arrays.items()}})
    row={'workload':w,'n_pairs':30,'pasi_pps_lower_count':int((pasi.pps<moi.pps).sum()),'pasi_coverage_lower_count':int((pasi.coverage<moi.coverage-1e-12).sum())}
    for k,v in arrays.items():row['mean_'+k]=float(v.mean());row[k+'_ci_low'],row[k+'_ci_high']=ci(v)
    contrasts.append(row)

old=pd.read_csv(plan['old_result_path'])
merged=frame.merge(old,on=['workload','seed','method'],suffixes=('_current','_legacy'),validate='one_to_one')
historical=[]
for r in merged.to_dict('records'):
    historical.append({'workload':r['workload'],'seed':r['seed'],'method':r['method'],
        'payment_legacy':r['payment_legacy'],'payment_current':r['payment_current'],
        'payment_current_minus_legacy':r['payment_current']-r['payment_legacy'],
        'pps_legacy':r['pps_legacy'],'pps_current':r['pps_current'],
        'pps_current_minus_legacy':r['pps_current']-r['pps_legacy'],
        'coverage_legacy':r['service_coverage'],'coverage_current':r['coverage'],
        'coverage_current_minus_legacy_pp':100*(r['coverage']-r['service_coverage']),
        'target_legacy':r['target_violations_legacy'],'target_current':r['target_violations_current'],
        'ir_legacy':r['ir_violations_legacy'],'ir_current':r['ir_violations_current'],
        'qos_current':r['qos_violations']})
for name,data in [('cell_audit',audits),('paired_input_audit',pair_checks),('endpoint_audit',endpoints),('summary',means),('paired_seed_results',paired),('paired_summary',contrasts),('legacy_protocol_comparison',historical)]:
    pd.DataFrame(data).to_csv(ROOT/f'{name}.csv',index=False)
negative=frame[(frame.qos_violations>0)|(frame.ir_violations>0)|(frame.target_violations>0)|(frame.slots_settlement_exceeds_budget>0)|(frame.slots_reserve_exceeds_budget>0)]
negative.to_csv(ROOT/'negative_cells.csv',index=False)
summary={'cells':180,'pairs':90,'all_cell_consistency_checks_pass':True,'all_paired_input_graph_config_checks_pass':True,
    'negative_cells':len(negative),'total_qos_violations':int(frame.qos_violations.sum()),'total_ir_violations':int(frame.ir_violations.sum()),
    'total_target_violations':int(frame.target_violations.sum()),'diagnostic_status_counts':frame.diagnostic_status.value_counts().to_dict(),
    'observed_all_legal_efforts_at_one':all(r['legal_rows']==r['effort_one_rows'] for r in endpoints),
    'interval_scope':plan['interval_scope'],'references_changed':False,'manuscript_changed':False,
    'analysis_sha256':sha(Path(__file__)),'preregistration_sha256':sha(ROOT/'preregistration.json')}
dump(ROOT/'audit_summary.json',summary)
print(json.dumps(summary,indent=2));print(pd.DataFrame(contrasts).to_string(index=False))
