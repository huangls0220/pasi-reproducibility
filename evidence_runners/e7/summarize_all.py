"""Direction-separated complete original E7 summaries; no universal payment claim."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;M=ROOT/'merged'
plan=json.loads((M/'preregistration.json').read_text());audit=json.loads((M/'audit_summary.json').read_text())
assert audit['runs']==2880 and audit['all_cell_checks_passed'] and audit['all_paired_input_checks_passed'] and audit['all_sampled_matching_passed']
frame=pd.read_csv(M/'seed_results.csv');paired=pd.read_csv(M/'paired_seed_results.csv');cells=pd.read_csv(M/'cell_paired_summary.csv')
group={cid:('zero' if e['cell']['magnitude']==0 else ('up' if e['cell']['direction']=='runtime_higher' else 'down')) for cid,e in plan['configs'].items()}
for f in (frame,paired,cells):f['branch']=f.cell_id.map(group)
branches=[]
for branch,g in paired.groupby('branch'):
    c=cells[cells.branch==branch];r=frame[frame.branch==branch]
    row={'branch':branch,'cells':len(c),'pairs':len(g),'runs':len(r),
        'equal_cell_mean_pps_saving_percent':float(c.mean_pps_saving_percent.mean()),
        'cell_mean_pps_saving_min_percent':float(c.mean_pps_saving_percent.min()),
        'cell_mean_pps_saving_max_percent':float(c.mean_pps_saving_percent.max()),
        'equal_cell_mean_qsc_gain_pp':float(c.mean_qsc_gain_pp.mean()),
        'pasi_pps_lower_pairs':int((g.pasi_pps<g.moi_pps-1e-12).sum()),
        'pasi_pps_higher_pairs':int((g.pasi_pps>g.moi_pps+1e-12).sum()),
        'same_sc_pairs':int((abs(g.sc_gain_pp)<1e-9).sum()),
        'pasi_qsc_higher_pairs':int((g.qsc_gain_pp>1e-9).sum()),
        'pasi_qsc_lower_pairs':int((g.qsc_gain_pp<-1e-9).sum())}
    for method in ('MOI','PASI'):
        f=r[r.method==method]
        for k in ('assigned','qos_violations','quality_violations','deadline_violations','ir_violations','target_violations'):
            row[method.lower()+'_'+k]=int(f[k].sum())
    branches.append(row)
pd.DataFrame(branches).to_csv(M/'direction_summary.csv',index=False)
paths=json.loads((M/'log_sources.json').read_text());identity=[]
cols=['slot','provider_id','task_id','selected','a_target','p_star','D_star','base_payment','a_star','expected_contract_cost']
for cid,seed in paired[paired.branch=='zero'][['cell_id','seed']].itertuples(index=False,name=None):
    a=pd.read_parquet(Path(paths[f'{cid}__seed_{seed:03d}__MOI'])/'pair_log.parquet',columns=cols)
    b=pd.read_parquet(Path(paths[f'{cid}__seed_{seed:03d}__PASI'])/'pair_log.parquet',columns=cols)
    identity.append({'cell_id':cid,'seed':seed,'exact_MOI_PASI_identity':a.equals(b)})
pd.DataFrame(identity).to_csv(M/'zero_mismatch_full_identity.csv',index=False)
originaldir=Path('C:/Users/huang/Documents/Codex/2026-08-13/anz/work/experiment-e7/results/e7')
original=pd.concat([pd.read_csv(originaldir/'e7_screening_seed_results.csv'),pd.read_csv(originaldir/'e7_confirmation_seed_results.csv')]).rename(columns={'mechanism':'method'})
keys=['cell_id','seed','method'];membership=frame[keys].merge(original[keys],on=keys,how='outer',indicator=True,validate='one_to_one')
assert len(membership)==2880 and (membership['_merge']=='both').all()
reference=pd.read_csv(ROOT.parent/'round67-e7-confirmation/cell_paired_summary.csv')
down=cells[cells.branch=='down'].drop(columns='branch')
pd.testing.assert_frame_equal(down.sort_values('cell_id').reset_index(drop=True),reference.sort_values('cell_id').reset_index(drop=True),check_exact=False,rtol=1e-12,atol=1e-12)
report={'complete_original_membership':True,'runs':2880,'cells':96,'pairs':1440,
    'source_counts':json.loads((M/'completion.json').read_text())['source_counts'],
    'directions':branches,'zero_identity_pairs':len(identity),'zero_identity_all_passed':all(x['exact_MOI_PASI_identity'] for x in identity),
    'downward_R67_summary_unchanged':True,'all_outcome_and_matching_audits_passed':True,
    'matching_sampled_slots':audit['matching_slots_checked'],'new_samples':False,'references_changed':False,'manuscript_changed':False}
(M/'final_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
assert report['zero_identity_all_passed']
print(json.dumps(report,indent=2))
