"""Full-log/current-vs-historical integrity audit for R70 E16."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;CODE=ROOT/'code'
HISTROOT=Path('C:/Users/huang/Documents/Codex/2026-08-13/anz/work/round20-scientific-boundaries/results/e16_quac_public_baseline')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)),encoding='utf-8')

plan=load(ROOT/'formal_preregistration.json');done=load(ROOT/'completion.json');assert done['completed']==270 and not done['errors']
frame=pd.read_csv(ROOT/'seed_results.csv');assert len(frame)==270 and not frame.duplicated(['workload','seed','method']).any()
hist=pd.read_csv(HISTROOT/'e16_formal_seed_results.csv');assert len(hist)==270
assert set(map(tuple,frame[['workload','seed','method']].to_numpy()))==set(map(tuple,hist[['workload','seed','method']].to_numpy()))
records={};audits=[];domains=[]
for i,j in enumerate(plan['jobs'],1):
    name=f"{j['workload']}__seed_{j['seed']:03d}__{j['method']}";r=load(ROOT/'cells'/f'{name}.json');records[name]=r;folder=ROOT/'restricted'/name
    assert {n:sha(folder/n) for n in r['restricted_log_sha256']}==r['restricted_log_sha256']
    pair=pd.read_parquet(folder/'pair_log.parquet');slot=pd.read_parquet(folder/'slot_log.parquet');ctx=pd.read_parquet(folder/'candidate_contexts.parquet')
    assert pair[['slot','provider_id','task_id']].equals(ctx[['slot','provider_id','task_id']]);sel=pair[pair.selected]
    totals=sel.groupby('slot').agg(count=('task_id','size'),payment=('expected_contract_cost','sum'),reserve=('decision_contract_cost','sum'))
    s=slot.set_index('slot').join(totals).fillna({'count':0,'payment':0.,'reserve':0.})
    checks={'log_hashes':True,'input_episode':r['episode_sha256']==plan['entries'][f"{j['workload']}/{j['seed']}"]['episode_sha256'],
      'input_config':r['config_sha256']==plan['entries'][f"{j['workload']}/{j['seed']}"]['config_sha256'],
      'selected_legal':bool(sel.feasible_contract.all()),'selected_spatial':bool(sel.spatial_candidate.all()),
      'provider_capacity':not sel.duplicated(['slot','provider_id']).any(),'task_capacity':not sel.duplicated(['slot','task_id']).any(),
      'selected_count':len(sel)==r['assigned'] and bool((s['count']==s.num_assigned).all()),
      'payment_per_slot':bool(np.allclose(s.payment,s.total_payment,rtol=0,atol=1e-8)),
      'reserve_per_slot':bool(np.allclose(s.reserve,s.budget_used,rtol=0,atol=1e-8)),
      'payment_total':bool(np.isclose(sel.expected_contract_cost.sum(),r['payment'],rtol=0,atol=1e-8)),
      'qualified_count':int(slot.num_qualified_completed.sum())==r['qualified'],
      'qos_count':r['assigned']-r['qualified']==r['qos_violations'],
      'ir_count':int((~sel.ir_ok).sum())==r['ir_violations'],'target_count':int((sel.implementation_gap<-1e-7).sum())==r['target_violations'],
      'budget_counts':int((slot.budget_used>slot.budget+1e-7).sum())==r['reserve_over_budget_slots'] and int((slot.total_payment>slot.budget_used+1e-7).sum())==r['settlement_over_reserve_slots'] and int((slot.total_payment>slot.budget+1e-7).sum())==r['settlement_over_budget_slots'],
      'execution_completed':r['execution_status']=='completed'}
    audits.append({**j,**checks,'all_passed':all(checks.values())})
    legal=pair.feasible_contract.to_numpy(bool);costdiff=np.abs(pair.decision_contract_cost-pair.expected_contract_cost).to_numpy()
    domains.append({**j,'candidate_rows':len(pair),'legal_rows':int(legal.sum()),'selected_rows':len(sel),
      'legal_effort_below_a_min':int(((pair.a_star<pair.a_min-1e-9).to_numpy()&legal).sum()),
      'legal_effort_at_one':int(((np.abs(pair.a_star-1)<=1e-9).to_numpy()&legal).sum()),
      'legal_target_at_one':int(((np.abs(pair.a_target-1)<=1e-9).to_numpy()&legal).sum()),
      'legal_design_settlement_cost_diff':int(((costdiff>1e-9)&legal).sum()),
      'max_legal_design_settlement_cost_diff':float(costdiff[legal].max()) if legal.any() else 0.,
      'zeta_above_one':int((ctx.ctx_zeta_t>1+1e-12).sum()),'robust_enabled_rows':int(ctx.ctx_robust_enabled.astype(bool).sum())})
    if i%30==0:print(json.dumps({'audited':i,'expected':270}),flush=True)
auditframe=pd.DataFrame(audits);domain=pd.DataFrame(domains);auditframe.to_csv(ROOT/'cell_audit.csv',index=False);domain.to_csv(ROOT/'domain_activation_audit.csv',index=False)
assert auditframe.all_passed.all()

# Exact paired input/graph identities across all three methods.
paired=[]
for w in ('low','medium','high'):
  for seed in range(1,31):
    rr=[records[f'{w}__seed_{seed:03d}__{m}'] for m in ('PASI','MOI','QUAC-F')]
    paired.append({'workload':w,'seed':seed,'input_fingerprints_equal':rr[0]['input_fingerprints']==rr[1]['input_fingerprints']==rr[2]['input_fingerprints'],
      'candidate_graph_equal':rr[0]['candidate_graph_sha256']==rr[1]['candidate_graph_sha256']==rr[2]['candidate_graph_sha256'],
      'config_equal':rr[0]['config_sha256']==rr[1]['config_sha256']==rr[2]['config_sha256'],
      'event_tape_equal':rr[0]['event_tape_hash']==rr[1]['event_tape_hash']==rr[2]['event_tape_hash']})
pairframe=pd.DataFrame(paired);pairframe.to_csv(ROOT/'paired_input_audit.csv',index=False);assert pairframe.drop(columns=['workload','seed']).to_numpy(bool).all()

# Historical/current compatibility excluding inherently variable runtimes and renamed status fields.
keys=['workload','seed','method'];merged=hist.merge(frame,on=keys,suffixes=('_historical','_current'),validate='one_to_one')
metrics=('N','M','payment','pps','service_coverage','mean_quality','qos_violation_rate','under_incentive_rate','target_miss_rate','legal_edge_rate','platform_utility','mean_base_payment','negative_intercept_rate')
compat=[]
for m in metrics:
    a=merged[m+'_historical'].to_numpy(float);b=merged[m+'_current'].to_numpy(float);diff=np.abs(a-b)
    compat.append({'metric':m,'rows':len(diff),'changed_gt_1e12':int((diff>1e-12).sum()),'max_abs_diff':float(np.nanmax(diff))})
pd.DataFrame(compat).to_csv(ROOT/'historical_metric_compatibility.csv',index=False)
merged[keys+sum(([m+'_historical',m+'_current'] for m in metrics),[])].to_csv(ROOT/'historical_protocol_comparison.csv',index=False)

# Rebuild the exact E16 reporting summary with the unchanged historical function.
sys.path.insert(0,str(CODE));from scripts.run_e16_quac_baseline import summarize
rebuilt=summarize(frame.rename(columns={'execution_status':'run_status'}));rebuilt.to_csv(ROOT/'paired_summary.csv',index=False)
historical_summary=pd.read_csv(HISTROOT/'e16_formal_paired_summary.csv');sk=['workload','method'];cs=historical_summary.merge(rebuilt,on=sk,suffixes=('_historical','_current'),validate='one_to_one')
summarychecks=[]
for c in historical_summary.columns:
    if c in sk or c in ('mean_core_runtime_s','core_runtime_s_ci_low','core_runtime_s_ci_high','delta_core_runtime_s','delta_core_runtime_s_ci_low','delta_core_runtime_s_ci_high'):continue
    if not pd.api.types.is_numeric_dtype(historical_summary[c]):ok=(cs[c+'_historical']==cs[c+'_current']).all();gap=0.
    else:
        gap=float(np.nanmax(np.abs(cs[c+'_historical'].to_numpy(float)-cs[c+'_current'].to_numpy(float))));ok=gap<=1e-12
    summarychecks.append({'field':c,'passed':bool(ok),'max_abs_diff':gap})
pd.DataFrame(summarychecks).to_csv(ROOT/'summary_compatibility.csv',index=False)

negative=frame[(frame.qos_violations>0)|(frame.ir_violations>0)|(frame.target_violations>0)|(frame.reserve_over_budget_slots>0)|(frame.settlement_over_budget_slots>0)]
negative.to_csv(ROOT/'negative_outcomes.csv',index=False)
result={'runs':270,'pairs_per_external_comparison':90,'all_log_and_outcome_checks_passed':bool(auditframe.all_passed.all()),
 'all_paired_input_graph_config_checks_passed':bool(pairframe.drop(columns=['workload','seed']).to_numpy(bool).all()),
 'historical_scientific_metrics_exact_at_1e12':all(r['changed_gt_1e12']==0 for r in compat),
 'historical_reporting_summary_exact_at_1e12':all(r['passed'] for r in summarychecks),
 'free_effort_change_inactive_on_legal_domain':int(domain.legal_effort_below_a_min.sum())==0,
 'design_settlement_cost_change_inactive_on_legal_domain':int(domain.legal_design_settlement_cost_diff.sum())==0,
 'prelec_interior_change_inactive':int(domain.zeta_above_one.sum())==0,'robust_changes_inactive':int(domain.robust_enabled_rows.sum())==0,
 'legal_candidate_rows':int(domain.legal_rows.sum()),'all_legal_efforts_at_one':int(domain.legal_effort_at_one.sum())==int(domain.legal_rows.sum()),
 'all_legal_targets_at_one':int(domain.legal_target_at_one.sum())==int(domain.legal_rows.sum()),
 'max_legal_design_settlement_cost_diff':float(domain.max_legal_design_settlement_cost_diff.max()),
 'negative_outcome_runs':len(negative),'qos_violations':int(frame.qos_violations.sum()),'ir_violations':int(frame.ir_violations.sum()),'target_violations':int(frame.target_violations.sum()),
 'compatibility_conclusion':'R20 E16 numerical outcomes are compatible with current core on exact original270 jobs; preserve values, correct allocation wording to design-side decision cost',
 'scope':plan['scope'],'references_changed':False,'manuscript_changed':False,'github_changed':False,'analysis_sha256':sha(Path(__file__))}
dump(ROOT/'audit_summary.json',result);print(json.dumps(result,indent=2),flush=True)
assert all([result['historical_scientific_metrics_exact_at_1e12'],result['historical_reporting_summary_exact_at_1e12'],result['free_effort_change_inactive_on_legal_domain'],result['design_settlement_cost_change_inactive_on_legal_domain'],result['prelec_interior_change_inactive'],result['robust_changes_inactive']])
