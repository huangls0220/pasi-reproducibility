"""Independent post-run optimality QA on five deterministic slots per cell."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds

ROOT=Path(__file__).resolve().parent
def solve(frame,budget):
    if len(frame)==0:return 0,0.,'empty'
    P,ps=pd.factorize(frame.provider_id);T,ts=pd.factorize(frame.task_id)
    cost=frame.decision_contract_cost.to_numpy(float)
    assert np.all(cost>=-1e-10)
    n=max(len(ps),len(ts));weight=np.zeros((n,n));valid=np.zeros((n,n),bool)
    penalty=1.0+float(np.abs(cost).sum())
    weight[P,T]=cost-penalty;valid[P,T]=True
    pi,ti=linear_sum_assignment(weight);chosen=valid[pi,ti]
    indices={(int(p),int(t)):i for i,(p,t) in enumerate(zip(P,T))}
    picked=np.array([indices[(int(p),int(t))] for p,t in zip(pi[chosen],ti[chosen])],int)
    k=len(picked);payment=float(cost[picked].sum())
    if payment<=budget+1e-8:return k,payment,'hungarian_budget_slack'
    A=np.zeros((len(ps)+len(ts)+1,len(frame)))
    A[P,np.arange(len(frame))]=1;A[len(ps)+T,np.arange(len(frame))]=1;A[-1]=cost
    upper=np.r_[np.ones(len(ps)+len(ts)),budget]
    constraint=LinearConstraint(A,-np.inf,upper)
    kwargs={'integrality':np.ones(len(frame)),'bounds':Bounds(0,1),'options':{'time_limit':20.0}}
    first=milp(-np.ones(len(frame)),constraints=constraint,**kwargs)
    if first.status!=0:raise RuntimeError('Independent cardinality MILP did not certify optimality')
    k=int(np.rint(first.x).sum())
    second=milp(cost,constraints=[constraint,LinearConstraint(np.ones((1,len(frame))),k,k)],**kwargs)
    if second.status!=0:raise RuntimeError('Independent payment MILP did not certify optimality')
    return k,float(cost[np.rint(second.x).astype(bool)].sum()),'two_stage_milp'

plan=json.loads((ROOT/'preregistration.json').read_text(encoding='utf-8'));rows=[]
for job in plan['jobs']:
    name=f"{job['workload']}__seed_{job['seed']:03d}__{job['method']}"
    path=ROOT/'restricted'/name;pair=pd.read_parquet(path/'pair_log.parquet');slot=pd.read_parquet(path/'slot_log.parquet').set_index('slot')
    candidates=np.sort(pair.slot.unique());chosen=candidates[np.unique(np.linspace(0,len(candidates)-1,min(5,len(candidates)),dtype=int))]
    checks=[];cost_gap=0.;counts={}
    for t in chosen:
        f=pair[(pair.slot==t)&pair.feasible_contract];k,c,solver=solve(f,float(slot.loc[t,'budget']))
        gap=abs(c-float(slot.loc[t,'budget_used']));cost_gap=max(cost_gap,gap)
        checks.append(k==int(slot.loc[t,'num_assigned']) and gap<=1e-7)
        counts[solver]=counts.get(solver,0)+1
    rows.append({**job,'slots_checked':len(chosen),'passed':all(checks),'max_payment_gap':cost_gap,**counts})
pd.DataFrame(rows).fillna(0).to_csv(ROOT/'independent_matching_audit.csv',index=False)
report={'post_run_QA':True,'cells':len(rows),'slots_checked':sum(r['slots_checked'] for r in rows),
    'all_passed':all(r['passed'] for r in rows),'max_payment_gap':max(r['max_payment_gap'] for r in rows),
    'selection':'5 evenly spaced distinct nonempty slot IDs per cell, fixed by log domain, not by outcomes',
    'scope':'sampled-slot exact objective check, not a claim of checking all slots; no Simulator or outcome changes'}
(ROOT/'matching_audit_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report));assert report['all_passed']
