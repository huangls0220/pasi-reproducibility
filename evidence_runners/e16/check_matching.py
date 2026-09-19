"""Independent five-slot matching audit for the current E16 confirmation."""
from pathlib import Path
import json,ast
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment,milp,LinearConstraint,Bounds

ROOT=Path(__file__).resolve().parent
source=ROOT.parent/'round64-e5-confirmation/check_matching.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='solve')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),globals())
plan=json.loads((ROOT/'formal_preregistration.json').read_text());rows=[]
for i,j in enumerate(plan['jobs'],1):
    name=f"{j['workload']}__seed_{j['seed']:03d}__{j['method']}";folder=ROOT/'restricted'/name
    pair=pd.read_parquet(folder/'pair_log.parquet');slot=pd.read_parquet(folder/'slot_log.parquet').set_index('slot')
    ids=np.sort(pair.slot.unique());chosen=ids[np.unique(np.linspace(0,len(ids)-1,min(5,len(ids)),dtype=int))];good=[];gap=0.
    for t in chosen:
        f=pair[(pair.slot==t)&pair.feasible_contract];k,c,_=solve(f,float(slot.loc[t,'budget']));d=abs(c-float(slot.loc[t,'budget_used']));gap=max(gap,d);good.append(k==int(slot.loc[t,'num_assigned']) and d<=1e-7)
    rows.append({**j,'slots_checked':len(chosen),'max_payment_gap':gap,'passed':all(good)})
    if i%30==0:print(json.dumps({'matching_audited':i,'expected':270}),flush=True)
pd.DataFrame(rows).to_csv(ROOT/'independent_matching_audit.csv',index=False)
summary={'runs':270,'slots_checked':sum(x['slots_checked'] for x in rows),'all_passed':all(x['passed'] for x in rows),'max_payment_gap':max(x['max_payment_gap'] for x in rows),
 'selection':'five deterministic evenly spaced nonempty pair-log slots per run','scope':'sampled exact objective audit, not exhaustive all-slot certificate'}
(ROOT/'matching_audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary));assert summary['all_passed']
