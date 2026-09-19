"""A1.6 Stage 1 verification — reads result files, checks all required conditions."""
import sys, json, hashlib
from pathlib import Path
import pandas as pd

RDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('results/route_a/a1_6_stage1')
passed = 0; failed = 0; results = []

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1; results.append({'check': name, 'status': 'PASS', 'detail': detail})
    else:
        failed += 1; results.append({'check': name, 'status': 'FAIL', 'detail': detail})
        print(f"  FAIL: {name} — {detail}")

# 1. Event Tape exists
tape_dir = RDIR/'event_tape'
for f in ['tasks.parquet','providers.parquet','provider_static.parquet']:
    check(f'event_tape_{f}_exists', (tape_dir/f).exists())

# 2. Event Tape content hash
meta = json.loads((tape_dir/'event_tape_metadata.json').read_text())
tape_hash = meta['content_sha256']
tape_bytes = b''
for f in ['tasks.parquet','providers.parquet','provider_static.parquet']:
    tape_bytes += (tape_dir/f).read_bytes()
tape_hash2 = hashlib.sha256(tape_bytes).hexdigest()[:16]
check('event_tape_hash_reproducible', tape_hash == tape_hash2, f'{tape_hash} == {tape_hash2}')

# 3. Two base runs
base = pd.read_csv(RDIR/'base_runs'/'base_run_summary.csv')
check('base_runs_count', len(base) == 2, f'Got {len(base)}')
check('both_runs_completed', all(base['status'] == 'ok'))
check('shared_event_tape_hash', len(set(base['event_tape_sha256'])) == 1)
check('shared_reset_ids_hash', len(set(base['reset_ids_sha256'])) == 1)
check('violations_zero', base['violations'].sum() == 0)

# 4. Assignment traces
for sc in ['stationary', 'state_reset']:
    f = RDIR/'assignment_trace'/f'assignment_trace_{sc}_seed_601.csv'
    check(f'ax_{sc}_exists', f.exists())
    if f.exists():
        df = pd.read_csv(f)
        check(f'ax_{sc}_nonempty', len(df) > 0, f'{len(df)} rows')
        check(f'ax_{sc}_keys_unique', df['assignment_key'].nunique() == len(df))
        check(f'ax_{sc}_provider_is_reset_not_all_false', df['provider_is_reset'].any() if 'provider_is_reset' in df.columns else False)
        check(f'ax_{sc}_H_not_zero', df['H_at_quote'].max() > 0.01)
        check(f'ax_{sc}_tape_hash_consistent', all(df['event_tape_sha256'] == tape_hash))

# 5. Provider traces
for sc in ['stationary', 'state_reset']:
    f = RDIR/'provider_trace'/f'provider_trace_{sc}_seed_601.parquet'
    check(f'px_{sc}_exists', f.exists())
    if f.exists():
        df = pd.read_parquet(f)
        check(f'px_{sc}_nonempty', len(df) > 0, f'{len(df)} rows')
        check(f'px_{sc}_100_providers', df['provider_id'].nunique() == 100)
        check(f'px_{sc}_1000_slots', df['slot'].nunique() == 1000)

# 6. Reset IDs
rid = pd.read_csv(RDIR/'config'/'reset_ids_seed_601.csv')
check('reset_ids_50', rid['is_reset'].sum() == 50)
check('reset_ids_canonical', all(rid['provider_id'].str.match(r'^P\d{5}$')))

# 7. Reset event
ra = pd.read_csv(RDIR/'audit'/'reset_event_audit.csv')
check('reset_success', len(ra[(ra['provider_is_reset']==True)&(ra['reset_group_not_zero_violation']==False)]) == 50)
check('control_not_reset', len(ra[(ra['provider_is_reset']==False)&(ra['control_reset_violation']==True)]) == 0)

# 8. Label audit
la = pd.read_csv(RDIR/'audit'/'provider_reset_label_audit.csv')
check('label_mismatches_zero', la['mismatch_count'].sum() == 0)

# 9. No stage 2 outputs
for stage2_dir in ['four_world','same_pair','recovery']:
    check(f'no_{stage2_dir}', not (RDIR/stage2_dir).exists())

# 10. Verify JSON
v = json.loads((RDIR/'audit'/'stage1_verification.json').read_text())
check('verification_passes', v['stage1_status'] == 'PASS')

# Summary
all_pass = failed == 0
print(f"\n{'='*60}")
print(f"VERIFICATION: {'PASS' if all_pass else 'FAIL'}")
print(f"  {passed} passed, {failed} failed")
print(f"{'='*60}")

(RDIR/'audit'/'stage1_verification.txt').write_text(
    json.dumps({'passed': passed, 'failed': failed, 'all_pass': all_pass, 'checks': results}, indent=2))

sys.exit(0 if all_pass else 1)
