"""A1.6 Stage 1B verification — reads ALL result files, no hardcoded PASS."""
import sys, json
from pathlib import Path
import pandas as pd, numpy as np

R = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('results/route_a/a1_6_stage1b')
passed = 0; failed = 0; checks = []

def chk(name, ok):
    global passed, failed
    if ok: passed += 1; checks.append({'name': name, 'pass': True})
    else: failed += 1; checks.append({'name': name, 'pass': False}); print(f"  FAIL: {name}")

# 1. Base runs
base = pd.read_csv(R/'base_runs'/'base_run_summary.csv')
chk('base_2_rows', len(base)==2)
chk('base_status_ok', all(base['status']=='ok'))
chk('base_shared_tape_hash', len(set(base['event_tape_sha256']))==1)
chk('base_shared_rid_hash', len(set(base['reset_ids_sha256']))==1)
chk('base_violations_zero', base['violations'].sum()==0)
chk('base_only_seed_601', all(base['seed']==601))

# 2. Assignment traces
for sc in ['stationary','state_reset']:
    f = R/'assignment_trace'/f'assignment_trace_{sc}_seed_601.csv'
    chk(f'ax_{sc}_exists', f.exists())
    if not f.exists(): continue
    df = pd.read_csv(f)
    chk(f'ax_{sc}_keys_no_null', df['assignment_key'].notna().all())
    chk(f'ax_{sc}_keys_unique', df['assignment_key'].nunique()==len(df))
    for col in ['omega','H_at_quote','Lambda_at_quote','objective_payment','expected_bonus']:
        if col in df.columns:
            chk(f'ax_{sc}_{col}_not_all_zero', df[col].max()>1e-12)
    if 'state_credit_at_quote' in df.columns and 'omega' in df.columns and 'H_at_quote' in df.columns:
        err = abs(df['state_credit_at_quote'] - df['omega']*df['H_at_quote']).max()
        chk(f'ax_{sc}_sc_identity', err<1e-10)
    chk(f'ax_{sc}_tape_hash_consistent', all(df['event_tape_sha256']==df['event_tape_sha256'].iloc[0]))

# 3. Payment reconciliation
rec = pd.read_csv(R/'audit'/'assignment_payment_reconciliation.csv')
chk('pay_reconcile_both', rec['pass'].sum()==2)

# 4. Provider traces
for sc in ['stationary','state_reset']:
    f = R/'provider_trace'/f'provider_trace_{sc}_seed_601.parquet'
    chk(f'px_{sc}_exists', f.exists())
    if not f.exists(): continue
    df = pd.read_parquet(f)
    chk(f'px_{sc}_100k', len(df)==100000)
    chk(f'px_{sc}_100_providers', df['provider_id'].nunique()==100)
    chk(f'px_{sc}_1000_slots', df['slot'].nunique()==1000)
    for slot in [0,100,500,999]:
        sl_df = df[df['slot']==slot]
        chk(f'px_{sc}_slot{slot}_100_providers', len(sl_df)==100)
        chk(f'px_{sc}_slot{slot}_50_reset', sl_df['provider_is_reset'].sum()==50)
    if 'state_credit_before' in df.columns and 'omega' in df.columns and 'H_before_slot' in df.columns:
        err = abs(df['state_credit_before'] - df['omega']*df['H_before_slot']).max()
        chk(f'px_{sc}_sc_identity', err<1e-10)

# 5. Reset event
re = pd.read_csv(R/'reset_event'/'reset_event_seed_601.csv')
chk('reset_50_rows', len(re)==50)
chk('reset_no_nan', not re.isna().any().any())
chk('reset_H_before_zero', (re['H_before_reset'].abs()<0.01).sum()==50)
chk('reset_most_succeeded', re['reset_success'].sum()>=45)
ctrl = pd.read_csv(R/'reset_event'/'control_group_reset_audit_seed_601.csv')
chk('control_50_rows', len(ctrl)==50)
chk('control_no_violations', ctrl['pass'].sum()==50)

# 6. Environment
env = pd.read_csv(R/'environment_audit'/'shared_environment_audit_seed_601.csv')
chk('env_1000_rows', len(env)==1000)
chk('env_all_equal', env['equal'].sum()==1000)

# 7. No Stage 2
for s2 in ['four_world','same_pair','recovery']:
    chk(f'no_{s2}', not (R/s2).exists())

# Final
all_pass = failed == 0
out = {'passed': passed, 'failed': failed, 'all_pass': all_pass, 'checks': checks,
       'stage1b_status': 'PASS' if all_pass else 'FAIL'}
(R/'audit'/'stage1b_verification.json').write_text(json.dumps(out, indent=2))
(R/'audit'/'stage1b_verification.txt').write_text(json.dumps(out, indent=2))
print(f'\nVERIFICATION: {"PASS" if all_pass else "FAIL"} — {passed} passed, {failed} failed')
sys.exit(0 if all_pass else 1)
