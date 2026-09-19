"""Verify A1.6 Stage 1C — reads all result files, checks all requirements."""
import sys, json, hashlib
from pathlib import Path
import pandas as pd, numpy as np

R = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('results/route_a/a1_6_stage1c')
p = 0; f = 0; CS = []

def ck(n, ok):
    global p, f
    if ok: p += 1; CS.append((n, True))
    else: f += 1; CS.append((n, False)); print(f"  FAIL: {n}")

# Base runs
base = pd.read_csv(R/'base_runs'/'base_run_summary.csv')
ck('base_2_rows', len(base)==2)
ck('base_ok', all(base['status']=='ok'))
ck('base_vio_0', base['violations'].sum()==0)

# Assignment traces
for sc in ['stationary','state_reset']:
    axf = R/'assignment_trace'/f'assignment_trace_{sc}_seed_601.csv'
    ck(f'ax_{sc}_exists', axf.exists())
    if axf.exists():
        df = pd.read_csv(axf)
        ck(f'ax_{sc}_keys_ok', df['assignment_key'].nunique()==len(df))
        for col in ['omega','Lambda_at_quote','objective_payment','H_at_quote']:
            if col in df.columns: ck(f'ax_{sc}_{col}_nz', df[col].max()>1e-10)
        if all(c in df.columns for c in ['state_credit_at_quote','omega','H_at_quote']):
            ck(f'ax_{sc}_sc_id', abs(df['state_credit_at_quote']-df['omega']*df['H_at_quote']).max()<1e-10)

# Provider traces
for sc in ['stationary','state_reset']:
    pxf = R/'provider_trace'/f'provider_trace_{sc}_seed_601.parquet'
    ck(f'px_{sc}_exists', pxf.exists())
    if pxf.exists():
        df = pd.read_parquet(pxf)
        ck(f'px_{sc}_100k', len(df)==100000)
        ck(f'px_{sc}_lam_ref', df['Lambda_reference'].max()>0.01)
        ck(f'px_{sc}_pay_ref', df['payment_quote_reference'].max()>0.01)
        if 'state_credit_before' in df.columns and 'omega' in df.columns:
            ck(f'px_{sc}_sc_id', abs(df['state_credit_before']-df['omega']*df['H_before_slot']).max()<1e-10)

# Reset event
re = pd.read_csv(R/'reset_event'/'reset_event_seed_601.csv')
ck('reset_50', len(re)==50)
ck('reset_no_nan', not re.isna().any().any())
ck('reset_all_before_0', (re['H_before_reset'].abs()<1e-8).sum()==50)
ctrl = pd.read_csv(R/'reset_event'/'control_group_event_seed_601.csv')
ck('control_50', len(ctrl)==50)
ck('control_pass', ctrl['pass'].sum()==50)

# Environment
env = pd.read_csv(R/'environment_consumption'/'shared_environment_audit_seed_601.csv')
ck('env_shared', env['equal'].sum()==1000)

# No stage 2
for d in ['four_world','same_pair','recovery']:
    ck(f'no_{d}', not (R/d).exists())

# Input frozen (episode generation not called)
ia = json.loads((R/'audit'/'input_freeze_audit.json').read_text())
ck('no_ep_gen', ia.get('episode_generation_called')==False)
ck('no_rid_resample', ia.get('reset_id_resampling_called')==False)

all_ok = f == 0
res = {'passed': p, 'failed': f, 'all_pass': all_ok, 'checks': [{'name':n,'pass':v} for n,v in CS],
       'stage1c_status': 'PASS' if all_ok else 'FAIL'}
(R/'audit'/'stage1c_verification.json').write_text(json.dumps(res, indent=2))
(R/'audit'/'stage1c_verification.txt').write_text(json.dumps(res, indent=2))
print(f'\nVERIFY: {"PASS" if all_ok else "FAIL"} — {p} passed, {f} failed')
sys.exit(0 if all_ok else 1)
