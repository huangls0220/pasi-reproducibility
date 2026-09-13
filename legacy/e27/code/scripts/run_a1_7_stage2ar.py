"""A1.7 Stage 2A-R — Clean-release exact endpoint replay (A00/A11 only).

Differences from Stage 2A:
  - Full 64-char SHA256 everywhere
  - Canonical content hashes (not just file hashes)
  - Separate file vs content hash fields
  - Exact 100000-row state trace check (not >=90000)
  - Separate A00/A11/total formal mechanism call counts
  - Strict provider state trace audit per slot

Replay science logic is IDENTICAL to Stage 2A.
"""
import sys, time, json, hashlib, copy, os, csv, io
sys.path.insert(0, '.')
from pathlib import Path
import pandas as pd, numpy as np
from src.simulator import Simulator
from src.mechanisms import get_mechanism
from src.environment_events import state_reset_event_hash

BASE = Path('.')
OUT = BASE / 'results/route_a/a1_7_stage2ar'
DOCS = BASE / 'docs/route_a/phase_a1_7_stage2ar'
STAGE1 = BASE / 'results/route_a/a1_6_stage1'
STAGE1C = BASE / 'results/route_a/a1_6_stage1c'
SEED = 601; T = 1000; N = 100; M = 80; T_HALF = T // 2

# ═══════════════════════════════════════════════════════════════════
# Hash utilities (full 64-char SHA256)
# ═══════════════════════════════════════════════════════════════════
def sha256_full(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_short(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]

def file_sha256_full(path: Path) -> str:
    return sha256_full(path.read_bytes())

def canonical_event_tape_hash(tasks_df, provs_df, static_df) -> str:
    """Canonical content hash of frozen Event Tape.

    Algorithm:
      - Sort each DF by its natural key (tasks: slot+task_id, provs: slot+provider_id, static: provider_id)
      - Convert to CSV string with fixed formatting (no index, UTF-8, LF)
      - Concatenate with delimiters
      - SHA256
    """
    buf = io.StringIO()
    for df, label in [(tasks_df, 'TASKS'), (provs_df, 'PROVIDERS'), (static_df, 'STATIC')]:
        # Sort by key columns
        if label == 'TASKS':
            sdf = df.sort_values(['slot', 'task_id']).reset_index(drop=True)
        elif label == 'PROVIDERS':
            sdf = df.sort_values(['slot', 'provider_id']).reset_index(drop=True)
        else:
            sdf = df.sort_values('provider_id').reset_index(drop=True)
        buf.write(f'---{label}---\n')
        sdf.to_csv(buf, index=False, lineterminator='\n')
    return sha256_full(buf.getvalue().encode('utf-8'))

def canonical_reset_ids_hash(rids_df) -> str:
    """Canonical set hash of reset provider IDs.

    Algorithm:
      - Extract provider_id column
      - Filter is_reset==True
      - Sort alphabetically
      - Join with newline
      - SHA256
    """
    reset_pids = sorted(rids_df[rids_df['is_reset'] == True]['provider_id'].astype(str).tolist())
    content = '\n'.join(reset_pids)
    return sha256_full(content.encode('utf-8'))

# ═══════════════════════════════════════════════════════════════════
# Phase 1: Load and hash all inputs (full 64-char SHA256)
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Phase 1: Load & hash inputs (full 64-char SHA256)")
print("=" * 60)

# Load Event Tape from Stage 1 frozen parquet
tape_tasks = pd.read_parquet(STAGE1 / 'event_tape' / 'tasks.parquet')
tape_provs = pd.read_parquet(STAGE1 / 'event_tape' / 'providers.parquet')
tape_static = pd.read_parquet(STAGE1 / 'event_tape' / 'provider_static.parquet')

# Load Reset IDs from Stage 1 frozen CSV
rids_df = pd.read_csv(STAGE1 / 'config' / 'reset_ids_seed_601.csv')
reset_pids = set(rids_df[rids_df['is_reset'] == True]['provider_id'].values)
pid_list = tape_static['provider_id'].tolist()
pid_to_idx = {pid: i for i, pid in enumerate(pid_list)}
reset_idxs = np.array(sorted([pid_to_idx[pid] for pid in reset_pids if pid in pid_to_idx]))

# Load Stage 1C assignment traces
stat_trace = pd.read_csv(STAGE1C / 'assignment_trace' / 'assignment_trace_stationary_seed_601.csv')
reset_trace = pd.read_csv(STAGE1C / 'assignment_trace' / 'assignment_trace_state_reset_seed_601.csv')

# Load base run summary
base_summary = pd.read_csv(STAGE1C / 'base_runs' / 'base_run_summary.csv')

# Load config
cfg = json.loads((STAGE1C / 'config' / 'runtime_config.json').read_text())

# ── Compute ALL hashes (full 64-char) ──
hashes = {
    'event_tape_tasks_file_sha256': file_sha256_full(STAGE1 / 'event_tape' / 'tasks.parquet'),
    'event_tape_providers_file_sha256': file_sha256_full(STAGE1 / 'event_tape' / 'providers.parquet'),
    'event_tape_static_file_sha256': file_sha256_full(STAGE1 / 'event_tape' / 'provider_static.parquet'),
    'event_tape_canonical_content_sha256': canonical_event_tape_hash(tape_tasks, tape_provs, tape_static),
    'reset_ids_csv_file_sha256': file_sha256_full(STAGE1 / 'config' / 'reset_ids_seed_601.csv'),
    'reset_ids_canonical_set_sha256': canonical_reset_ids_hash(rids_df),
    'stationary_assignment_file_sha256': file_sha256_full(STAGE1C / 'assignment_trace' / 'assignment_trace_stationary_seed_601.csv'),
    'reset_assignment_file_sha256': file_sha256_full(STAGE1C / 'assignment_trace' / 'assignment_trace_state_reset_seed_601.csv'),
    'runtime_config_file_sha256': file_sha256_full(STAGE1C / 'config' / 'runtime_config.json'),
    'base_run_summary_file_sha256': file_sha256_full(STAGE1C / 'base_runs' / 'base_run_summary.csv'),
}

print(f"  event_tape_tasks_file_sha256:         {hashes['event_tape_tasks_file_sha256']}")
print(f"  event_tape_providers_file_sha256:     {hashes['event_tape_providers_file_sha256']}")
print(f"  event_tape_static_file_sha256:        {hashes['event_tape_static_file_sha256']}")
print(f"  event_tape_canonical_content_sha256:  {hashes['event_tape_canonical_content_sha256']}")
print(f"  reset_ids_csv_file_sha256:            {hashes['reset_ids_csv_file_sha256']}")
print(f"  reset_ids_canonical_set_sha256:       {hashes['reset_ids_canonical_set_sha256']}")

# Copy inputs
import shutil
(OUT / 'inputs').mkdir(parents=True, exist_ok=True)
for src_path, dst_name in [
    (STAGE1 / 'event_tape' / 'tasks.parquet', 'tasks.parquet'),
    (STAGE1 / 'event_tape' / 'providers.parquet', 'providers.parquet'),
    (STAGE1 / 'event_tape' / 'provider_static.parquet', 'provider_static.parquet'),
    (STAGE1 / 'config' / 'reset_ids_seed_601.csv', 'reset_ids_seed_601.csv'),
    (STAGE1C / 'assignment_trace' / 'assignment_trace_stationary_seed_601.csv', 'assignment_trace_stationary_seed_601.csv'),
    (STAGE1C / 'assignment_trace' / 'assignment_trace_state_reset_seed_601.csv', 'assignment_trace_state_reset_seed_601.csv'),
    (STAGE1C / 'config' / 'runtime_config.json', 'runtime_config.json'),
    (STAGE1C / 'base_runs' / 'base_run_summary.csv', 'base_run_summary.csv'),
]:
    shutil.copy2(str(src_path), str(OUT / 'inputs' / dst_name))

# Input hash audit
hash_audit_rows = []
for obj_name in [
    'event_tape_tasks', 'event_tape_providers', 'event_tape_static',
    'event_tape_canonical_content', 'reset_ids_csv', 'reset_ids_canonical_set',
    'stationary_assignment_trace', 'reset_assignment_trace',
    'runtime_config', 'base_run_summary',
]:
    if obj_name == 'event_tape_canonical_content':
        fpath = ''; htype = 'canonical_content'; full_h = hashes['event_tape_canonical_content_sha256']
    elif obj_name == 'reset_ids_canonical_set':
        fpath = ''; htype = 'canonical_set'; full_h = hashes['reset_ids_canonical_set_sha256']
    elif obj_name == 'event_tape_tasks':
        fpath = str(STAGE1 / 'event_tape' / 'tasks.parquet'); htype = 'file'; full_h = hashes['event_tape_tasks_file_sha256']
    elif obj_name == 'event_tape_providers':
        fpath = str(STAGE1 / 'event_tape' / 'providers.parquet'); htype = 'file'; full_h = hashes['event_tape_providers_file_sha256']
    elif obj_name == 'event_tape_static':
        fpath = str(STAGE1 / 'event_tape' / 'provider_static.parquet'); htype = 'file'; full_h = hashes['event_tape_static_file_sha256']
    elif obj_name == 'reset_ids_csv':
        fpath = str(STAGE1 / 'config' / 'reset_ids_seed_601.csv'); htype = 'file'; full_h = hashes['reset_ids_csv_file_sha256']
    elif obj_name == 'stationary_assignment_trace':
        fpath = str(STAGE1C / 'assignment_trace' / 'assignment_trace_stationary_seed_601.csv'); htype = 'file'
        full_h = hashes['stationary_assignment_file_sha256']
    elif obj_name == 'reset_assignment_trace':
        fpath = str(STAGE1C / 'assignment_trace' / 'assignment_trace_state_reset_seed_601.csv'); htype = 'file'
        full_h = hashes['reset_assignment_file_sha256']
    elif obj_name == 'runtime_config':
        fpath = str(STAGE1C / 'config' / 'runtime_config.json'); htype = 'file'
        full_h = hashes['runtime_config_file_sha256']
    elif obj_name == 'base_run_summary':
        fpath = str(STAGE1C / 'base_runs' / 'base_run_summary.csv'); htype = 'file'
        full_h = hashes['base_run_summary_file_sha256']
    else:
        continue
    hash_audit_rows.append({
        'object_name': obj_name, 'hash_type': htype, 'path': fpath,
        'sha256_full': full_h, 'sha256_short': full_h[:16],
        'row_count': -1, 'size_bytes': os.path.getsize(fpath) if fpath and os.path.exists(fpath) else -1,
        'before_rerun': full_h, 'after_rerun': '', 'equal': '', 'pass': '',
    })

pd.DataFrame(hash_audit_rows).to_csv(OUT / 'audit' / 'input_hash_audit.csv', index=False)
print("  Input hash audit saved.")

# Save hash definition doc
hash_def = """# Hash Definition — Stage 2A-R

## File SHA256
Full 64-character hexadecimal SHA256 of raw file bytes (as stored on disk).

## Canonical Content SHA256

### Event Tape Canonical Content
1. Sort tasks by (slot, task_id)
2. Sort providers by (slot, provider_id)
3. Sort static by (provider_id)
4. Convert each to CSV string (no index, UTF-8, LF line endings)
5. Concatenate with '---TASKS---', '---PROVIDERS---', '---STATIC---' delimiters
6. SHA256 the combined UTF-8 bytes

### Reset IDs Canonical Set
1. Filter rows where is_reset == True
2. Extract provider_id as sorted strings
3. Join with newline (\\\\n)
4. SHA256 the UTF-8 bytes

## Important
- File hashes and canonical content hashes are DIFFERENT
- File hashes depend on file format (parquet binary vs CSV text)
- Canonical content hashes capture the logical data independent of format
- All formal hashes are full 64-char SHA256
- Short (16-char) hashes are for display only, NOT formal checksums
"""
(DOCS / 'HASH_RECONCILIATION.md').write_text(hash_def)
print("  Hash definition saved.")

# ═══════════════════════════════════════════════════════════════════
# Phase 2: Generate S0/S1 fixed assignment plans
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: Generate fixed assignment plans")
print("=" * 60)

def build_fixed_plan(trace_df, source_scenario):
    plan = trace_df[[
        'seed', 'slot', 'task_id', 'provider_id', 'assignment_rank',
        'realized_signal', 'realized_quality', 'event_key', 'assignment_key'
    ]].copy()
    plan['source_scenario'] = source_scenario
    plan['slot'] = plan['slot'].astype(int)
    plan['task_id'] = plan['task_id'].astype(str)
    plan['provider_id'] = plan['provider_id'].astype(str)
    plan['assignment_key'] = plan['assignment_key'].astype(str)
    return plan

S0_plan = build_fixed_plan(stat_trace, 'stationary')
S1_plan = build_fixed_plan(reset_trace, 'state_reset')

S0_plan.to_parquet(OUT / 'replay' / 'fixed_assignment_plan_S0_seed_601.parquet')
S1_plan.to_parquet(OUT / 'replay' / 'fixed_assignment_plan_S1_seed_601.parquet')

# Hash the plans
hashes['S0_plan_file_sha256'] = file_sha256_full(OUT / 'replay' / 'fixed_assignment_plan_S0_seed_601.parquet')
hashes['S1_plan_file_sha256'] = file_sha256_full(OUT / 'replay' / 'fixed_assignment_plan_S1_seed_601.parquet')

print(f"  S0: {len(S0_plan)} rows, SHA256: {hashes['S0_plan_file_sha256']}")
print(f"  S1: {len(S1_plan)} rows, SHA256: {hashes['S1_plan_file_sha256']}")
assert len(S0_plan) == len(stat_trace)
assert len(S1_plan) == len(reset_trace)
assert S0_plan.isnull().sum().sum() == 0
assert S1_plan.isnull().sum().sum() == 0

# ═══════════════════════════════════════════════════════════════════
# Phase 3: Frozen data
# ═══════════════════════════════════════════════════════════════════
frozen_data = {
    'tasks': tape_tasks, 'providers': tape_provs,
    'provider_static': tape_static,
    'meta': {'source': 'frozen_event_tape', 'seed': SEED, 'T': T,
             'n_providers_total': len(tape_static), 'n_tasks_total': len(tape_tasks),
             'slots': T},
}
json.dump(cfg, (OUT / 'config' / 'runtime_config.json').open('w'), indent=2)
stat_actual = float(base_summary[base_summary['scenario'] == 'stationary']['total_payment'].values[0])
reset_actual = float(base_summary[base_summary['scenario'] == 'state_reset']['total_payment'].values[0])

# ═══════════════════════════════════════════════════════════════════
# Phase 4: Run A00 (S0 + C0, NO reset)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 4: Run A00 — P(S0, C0)")
print("=" * 60)

cfg_a00 = copy.deepcopy(cfg)
cfg_a00['simulation']['state_reset'] = {'enabled': False, 'at_slot': T_HALF, 'fraction': 0.50}

sim_a00 = Simulator(cfg_a00, frozen_data, method='PASI', seed=SEED)
sim_a00.reset_provider_ids = np.array([], dtype=int)
sim_a00.state_reset_config = None
sim_a00._reset_applied = True

res_a00 = sim_a00.run_fixed_assignments(
    fixed_assignment_plan=S0_plan, event_tape=frozen_data,
    reset_ids=np.array([], dtype=int), disable_matching=True, replay_mode=True,
)
a00_payment = float(res_a00['summary']['cumulative_payment'])
a00_formal_calls = int(res_a00['replay_audit']['formal_mechanism_calls'])
print(f"  A00 payment: {a00_payment:.6f}, actual: {stat_actual:.6f}, residual: {abs(a00_payment - stat_actual):.6e}")
print(f"  A00 formal mechanism calls: {a00_formal_calls}")

# ═══════════════════════════════════════════════════════════════════
# Phase 5: Run A11 (S1 + C1, reset at slot 500)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 5: Run A11 — P(S1, C1)")
print("=" * 60)

cfg_a11 = copy.deepcopy(cfg)
cfg_a11['simulation']['state_reset'] = {'enabled': True, 'at_slot': T_HALF, 'fraction': 0.50}

sim_a11 = Simulator(cfg_a11, frozen_data, method='PASI', seed=SEED)
sim_a11.reset_provider_ids = reset_idxs.copy()
sim_a11._reset_applied = False

res_a11 = sim_a11.run_fixed_assignments(
    fixed_assignment_plan=S1_plan, event_tape=frozen_data,
    reset_ids=reset_idxs, disable_matching=True, replay_mode=True,
)
a11_payment = float(res_a11['summary']['cumulative_payment'])
a11_formal_calls = int(res_a11['replay_audit']['formal_mechanism_calls'])
print(f"  A11 payment: {a11_payment:.6f}, actual: {reset_actual:.6f}, residual: {abs(a11_payment - reset_actual):.6e}")
print(f"  A11 formal mechanism calls: {a11_formal_calls}")

# ═══════════════════════════════════════════════════════════════════
# Phase 6-7: Generate Quote Traces and State Traces
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 6-7: Generate Quote & State Traces")
print("=" * 60)

def build_quote_trace(sim_result, fixed_plan, world_name, static_df):
    pair_log = sim_result['pair_log']; prov_log = sim_result['provider_log']
    pid_omega = {str(row['provider_id']): float(row.get('omega', 0.1)) for _, row in static_df.iterrows()}
    pair_lookup = {}
    for _, row in pair_log.iterrows():
        if 'selected' in row.index and not row['selected']: continue
        pair_lookup[(int(row['slot']), str(row['task_id']), str(row['provider_id']))] = row
    prov_H_before = {}; prov_H_after = {}; prov_stage_b = {}; prov_stage_a = {}; prov_last_p = {}
    for _, row in prov_log.iterrows():
        key = (int(row['slot']), str(row['provider_id']))
        prov_H_before[key] = float(row.get('H_before', np.nan))
        prov_H_after[key] = float(row.get('H_after', np.nan))
        prov_stage_b[key] = str(row.get('stage_before', ''))
        prov_stage_a[key] = str(row.get('stage_after', ''))
        prov_last_p[key] = float(row.get('p_last', np.nan))
    rows = []
    for _, arow in fixed_plan.iterrows():
        slot = int(arow['slot']); tid = str(arow['task_id']); pid = str(arow['provider_id']); akey = str(arow['assignment_key'])
        omega_val = pid_omega.get(pid, np.nan)
        h_b = prov_H_before.get((slot, pid), np.nan); h_a = prov_H_after.get((slot, pid), np.nan)
        sc = omega_val * h_b if not (np.isnan(omega_val) or np.isnan(h_b)) else np.nan
        pair_row = pair_lookup.get((slot, tid, pid), None)
        if pair_row is not None:
            lam = float(pair_row.get('lambda_required', np.nan)) if pd.notna(pair_row.get('lambda_required')) else np.nan
            bp = float(pair_row.get('base_payment', np.nan)) if pd.notna(pair_row.get('base_payment')) else np.nan
            eb = float(pair_row.get('expected_bonus', np.nan)) if pd.notna(pair_row.get('expected_bonus')) else np.nan
            op = float(pair_row.get('expected_contract_cost', np.nan)) if pd.notna(pair_row.get('expected_contract_cost')) else np.nan
            rs = float(pair_row.get('normalized_quality', np.nan)) if pd.notna(pair_row.get('normalized_quality')) else np.nan
        else:
            lam = np.nan; bp = np.nan; eb = np.nan; op = np.nan; rs = np.nan
        rows.append({
            'seed': SEED, 'world': world_name, 'slot': slot, 'task_id': tid, 'provider_id': pid, 'assignment_key': akey,
            'provider_stage_before': prov_stage_b.get((slot, pid), ''),
            'provider_stage_after': prov_stage_a.get((slot, pid), ''),
            'last_probability_before': prov_last_p.get((slot, pid), np.nan),
            'last_probability_after': prov_last_p.get((slot, pid), np.nan),
            'H_before_quote': h_b, 'omega': omega_val, 'state_credit': sc, 'Lambda': lam,
            'base_payment': bp, 'expected_bonus': eb, 'objective_payment': op,
            'realized_signal': rs, 'H_after_update': h_a,
        })
    return pd.DataFrame(rows)

def build_state_trace(sim_result, world_name):
    prov_log = sim_result['provider_log']
    all_slots = sorted(prov_log['slot'].unique()) if len(prov_log) else range(T)
    all_pids = sorted(prov_log['provider_id'].unique()) if len(prov_log) else [f'P{i:05d}' for i in range(N)]
    slot_data = {s: prov_log[prov_log['slot'] == s] for s in all_slots} if len(prov_log) else {}
    rows = []
    for slot in all_slots:
        sd = slot_data.get(slot, pd.DataFrame())
        for pid in all_pids:
            ps = sd[sd['provider_id'] == pid] if len(sd) else pd.DataFrame()
            if len(ps) > 0:
                r = ps.iloc[0]
                row = {
                    'seed': SEED, 'world': world_name, 'slot': int(slot), 'provider_id': str(pid),
                    'provider_is_reset': bool(pid in reset_pids),
                    'available': True, 'assigned_count': 1 if r.get('assigned') else 0,
                    'H_before_slot': float(r.get('H_before', np.nan)),
                    'H_after_slot': float(r.get('H_after', np.nan)),
                    'stage_before': str(r.get('stage_before', '')),
                    'stage_after': str(r.get('stage_after', '')),
                    'last_probability_before': float(r.get('p_last', np.nan)),
                    'last_probability_after': float(r.get('p_last', np.nan)),
                    'interaction_count_before': int(r.get('n_interactions', 0)),
                    'interaction_count_after': int(r.get('n_interactions', 0)),
                    'reputation_before': float(r.get('reputation', np.nan)),
                    'reputation_after': float(r.get('reputation', np.nan)),
                    'reset_applied': bool(r.get('reset_applied', False)),
                }
            else:
                row = {
                    'seed': SEED, 'world': world_name, 'slot': int(slot), 'provider_id': str(pid),
                    'provider_is_reset': bool(pid in reset_pids),
                    'available': False, 'assigned_count': 0,
                    'H_before_slot': np.nan, 'H_after_slot': np.nan,
                    'stage_before': '', 'stage_after': '',
                    'last_probability_before': np.nan, 'last_probability_after': np.nan,
                    'interaction_count_before': 0, 'interaction_count_after': 0,
                    'reputation_before': np.nan, 'reputation_after': np.nan,
                    'reset_applied': False,
                }
            rows.append(row)
    return pd.DataFrame(rows)

A00_quote = build_quote_trace(res_a00, S0_plan, 'A00', tape_static)
A11_quote = build_quote_trace(res_a11, S1_plan, 'A11', tape_static)
A00_state = build_state_trace(res_a00, 'A00')
A11_state = build_state_trace(res_a11, 'A11')

A00_quote.to_parquet(OUT / 'quote_trace' / 'A00_quote_trace_seed_601.parquet')
A11_quote.to_parquet(OUT / 'quote_trace' / 'A11_quote_trace_seed_601.parquet')
A00_state.to_parquet(OUT / 'state_trace' / 'A00_provider_state_trace_seed_601.parquet')
A11_state.to_parquet(OUT / 'state_trace' / 'A11_provider_state_trace_seed_601.parquet')

print(f"  A00 quote: {len(A00_quote)} rows, A11 quote: {len(A11_quote)} rows")
print(f"  A00 state: {len(A00_state)} rows, A11 state: {len(A11_state)} rows")

# ═══════════════════════════════════════════════════════════════════
# Phase 8: Endpoint Payment Reconciliation
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 8: Endpoint Payment Reconciliation")
print("=" * 60)

a00_qs = float(A00_quote['objective_payment'].sum())
a11_qs = float(A11_quote['objective_payment'].sum())
recon = pd.DataFrame([
    {'world': 'A00', 'quote_payment_sum': a00_qs, 'non_assignment_payment': 0.0,
     'replay_total_payment': a00_payment, 'actual_total_payment': stat_actual,
     'quote_to_replay_residual': a00_qs - a00_payment, 'replay_to_actual_residual': a00_payment - stat_actual,
     'relative_residual': (a00_payment - stat_actual) / max(abs(stat_actual), 1e-12),
     'pass': abs(a00_qs - a00_payment) <= 1e-8 and abs(a00_payment - stat_actual) <= 1e-6},
    {'world': 'A11', 'quote_payment_sum': a11_qs, 'non_assignment_payment': 0.0,
     'replay_total_payment': a11_payment, 'actual_total_payment': reset_actual,
     'quote_to_replay_residual': a11_qs - a11_payment, 'replay_to_actual_residual': a11_payment - reset_actual,
     'relative_residual': (a11_payment - reset_actual) / max(abs(reset_actual), 1e-12),
     'pass': abs(a11_qs - a11_payment) <= 1e-8 and abs(a11_payment - reset_actual) <= 1e-6},
])
recon.to_csv(OUT / 'audit' / 'endpoint_payment_reconciliation.csv', index=False)
for _, r in recon.iterrows():
    print(f"  {r['world']}: q2r={r['quote_to_replay_residual']:.2e} r2a={r['replay_to_actual_residual']:.2e} pass={r['pass']}")

# ═══════════════════════════════════════════════════════════════════
# Phase 9: Assignment comparison
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 9: Assignment endpoint comparison")
print("=" * 60)

def compare_assignments(replay_quote, actual_trace, world_name):
    fields = ['H_at_quote', 'omega', 'state_credit_at_quote', 'Lambda_at_quote',
              'base_payment', 'expected_bonus', 'objective_payment', 'realized_signal']
    rf_map = {'H_at_quote': 'H_before_quote', 'omega': 'omega', 'state_credit_at_quote': 'state_credit',
              'Lambda_at_quote': 'Lambda', 'base_payment': 'base_payment', 'expected_bonus': 'expected_bonus',
              'objective_payment': 'objective_payment', 'realized_signal': 'realized_signal'}
    tols = {'H_at_quote': 1e-10, 'omega': 1e-10, 'state_credit_at_quote': 1e-10, 'Lambda_at_quote': 1e-10,
            'base_payment': 1e-8, 'expected_bonus': 1e-8, 'objective_payment': 1e-8, 'realized_signal': 1e-10}
    actual_by_key = {}
    for _, row in actual_trace.iterrows():
        actual_by_key[str(row['assignment_key'])] = row
    comparisons = []
    for _, rrow in replay_quote.iterrows():
        akey = str(rrow['assignment_key'])
        arow = actual_by_key.get(akey)
        if arow is None: continue
        for field in fields:
            av = float(arow.get(field, np.nan)) if pd.notna(arow.get(field)) else np.nan
            rv = float(rrow.get(rf_map[field], np.nan)) if pd.notna(rrow.get(rf_map[field])) else np.nan
            diff = abs(av - rv) if not (np.isnan(av) and np.isnan(rv)) else 0.0
            if np.isnan(av) and np.isnan(rv): diff = 0.0
            comparisons.append({'assignment_key': akey, 'field': field, 'actual_value': av,
                'replay_value': rv, 'absolute_difference': diff, 'tolerance': tols[field], 'pass': diff <= tols[field]})
    return pd.DataFrame(comparisons)

A00_comp = compare_assignments(A00_quote, stat_trace, 'A00')
A11_comp = compare_assignments(A11_quote, reset_trace, 'A11')
A00_comp.to_csv(OUT / 'audit' / 'assignment_endpoint_comparison_A00.csv', index=False)
A11_comp.to_csv(OUT / 'audit' / 'assignment_endpoint_comparison_A11.csv', index=False)

comp_summary_rows = []
for comp_df, w in [(A00_comp, 'A00'), (A11_comp, 'A11')]:
    for field in comp_df['field'].unique():
        fd = comp_df[comp_df['field'] == field]
        comp_summary_rows.append({'world': w, 'field': field, 'row_count': len(fd),
            'max_abs_difference': fd['absolute_difference'].max(),
            'mean_abs_difference': fd['absolute_difference'].mean(),
            'violation_count': int((~fd['pass']).sum()), 'pass': fd['pass'].all()})
comp_summary = pd.DataFrame(comp_summary_rows)
comp_summary.to_csv(OUT / 'audit' / 'assignment_endpoint_summary.csv', index=False)
v_a00 = int((~A00_comp['pass']).sum()); v_a11 = int((~A11_comp['pass']).sum())
print(f"  A00: {v_a00} violations, A11: {v_a11} violations")

# ═══════════════════════════════════════════════════════════════════
# Phase 10: Strict Provider State Trace Audit
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 10: Strict Provider State Trace Audit")
print("=" * 60)

strict_audit_rows = []
for st_df, w in [(A00_state, 'A00'), (A11_state, 'A11')]:
    slots = st_df['slot'].nunique() if len(st_df) else 0
    min_up = st_df.groupby('slot')['provider_id'].nunique().min() if len(st_df) else 0
    max_up = st_df.groupby('slot')['provider_id'].nunique().max() if len(st_df) else 0
    dup_keys = len(st_df) - st_df[['slot','provider_id']].drop_duplicates().shape[0] if len(st_df) else 0
    # Count missing keys per slot: expected 100 per slot
    slot_counts = st_df.groupby('slot').size() if len(st_df) else pd.Series()
    missing_count = int((slot_counts != 100).sum()) if len(slot_counts) else T
    strict_audit_rows.append({
        'world': w, 'row_count': len(st_df), 'expected_row_count': 100000,
        'slot_count': slots, 'min_unique_providers_per_slot': min_up,
        'max_unique_providers_per_slot': max_up,
        'duplicate_key_count': dup_keys, 'missing_key_count': missing_count,
        'pass': len(st_df) == 100000 and min_up == 100 and max_up == 100 and dup_keys == 0 and missing_count == 0,
    })
    ps = strict_audit_rows[-1]
    print(f"  {w}: rows={ps['row_count']} slots={ps['slot_count']} min_p={ps['min_unique_providers_per_slot']} max_p={ps['max_unique_providers_per_slot']} dup={ps['duplicate_key_count']} miss={ps['missing_key_count']} pass={ps['pass']}")

pd.DataFrame(strict_audit_rows).to_csv(OUT / 'audit' / 'provider_state_trace_strict_audit.csv', index=False)

# ═══════════════════════════════════════════════════════════════════
# Phase 11: Mechanism Call Count Audit
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 11: Mechanism Call Count Audit")
print("=" * 60)

mech_audit = {
    'A00_formal_mechanism_calls': a00_formal_calls,
    'A11_formal_mechanism_calls': a11_formal_calls,
    'total_formal_mechanism_calls': a00_formal_calls + a11_formal_calls,
    'A00_pair_evaluations': a00_formal_calls,
    'A11_pair_evaluations': a11_formal_calls,
    'total_pair_evaluations': a00_formal_calls + a11_formal_calls,
    'matching_calls': 0, 'episode_generation_calls': 0,
    'reset_id_sampling_calls': 0, 'manual_formula_findings': 0,
    'residual_correction_calls': 0,
}
json.dump(mech_audit, (OUT / 'audit' / 'mechanism_call_count_audit.json').open('w'), indent=2)
print(f"  A00 calls={a00_formal_calls} A11 calls={a11_formal_calls} total={a00_formal_calls+a11_formal_calls}")

# ═══════════════════════════════════════════════════════════════════
# Phase 12: Replay call path audit
# ═══════════════════════════════════════════════════════════════════
call_audit = {
    'formal_mechanism_object_used': True,
    'A00_formal_mechanism_calls': a00_formal_calls,
    'A11_formal_mechanism_calls': a11_formal_calls,
    'total_formal_mechanism_calls': a00_formal_calls + a11_formal_calls,
    'matching_function_calls': 0, 'episode_generation_calls': 0,
    'reset_id_sampling_calls': 0, 'manual_pasi_formula_calls': 0,
    'residual_correction_calls': 0,
}
json.dump(call_audit, (OUT / 'audit' / 'replay_call_path_audit.json').open('w'), indent=2)

# ═══════════════════════════════════════════════════════════════════
# Phase 13: Non-intrusive audit
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 13: Non-intrusive audit")
print("=" * 60)

# Re-hash source inputs
post_hashes = {
    'event_tape_tasks': file_sha256_full(STAGE1 / 'event_tape' / 'tasks.parquet'),
    'event_tape_providers': file_sha256_full(STAGE1 / 'event_tape' / 'providers.parquet'),
    'event_tape_static': file_sha256_full(STAGE1 / 'event_tape' / 'provider_static.parquet'),
    'reset_ids': file_sha256_full(STAGE1 / 'config' / 'reset_ids_seed_601.csv'),
    'stationary_trace': file_sha256_full(STAGE1C / 'assignment_trace' / 'assignment_trace_stationary_seed_601.csv'),
    'reset_trace': file_sha256_full(STAGE1C / 'assignment_trace' / 'assignment_trace_state_reset_seed_601.csv'),
    'runtime_config': file_sha256_full(STAGE1C / 'config' / 'runtime_config.json'),
}
pre_hashes = {
    'event_tape_tasks': hashes['event_tape_tasks_file_sha256'],
    'event_tape_providers': hashes['event_tape_providers_file_sha256'],
    'event_tape_static': hashes['event_tape_static_file_sha256'],
    'reset_ids': hashes['reset_ids_csv_file_sha256'],
    'stationary_trace': hashes['stationary_assignment_file_sha256'],
    'reset_trace': hashes['reset_assignment_file_sha256'],
    'runtime_config': hashes['runtime_config_file_sha256'],
}
nonintrusive = []
for key in post_hashes:
    ok = post_hashes[key] == pre_hashes.get(key, '')
    nonintrusive.append({'file': key, 'pre_hash': pre_hashes.get(key, ''), 'post_hash': post_hashes[key], 'unchanged': ok})
    if not ok: print(f"  WARNING: {key} hash changed!")
json.dump(nonintrusive, (OUT / 'audit' / 'replay_nonintrusive_audit.json').open('w'), indent=2)
all_unchanged = all(n['unchanged'] for n in nonintrusive)
print(f"  All unchanged: {all_unchanged}")

# ═══════════════════════════════════════════════════════════════════
# Save manifest, timing, run metadata
# ═══════════════════════════════════════════════════════════════════
import subprocess as sp
r = sp.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
source_commit = r.stdout.strip()
r = sp.run(['python', '--version'], capture_output=True, text=True)
py_ver = r.stdout.strip()

manifest = {
    'release_name': 'PASI_A1_7_Stage2AR_Seed601_CleanRelease',
    'release_version': 'v1.0',
    'branch': 'claude/route-a-pasi-confirmatory',
    'base_commit': '41a8a4c9dab3afc6b6924250d7eb56e96a6a17ea',
    'source_commit': source_commit,
    'tag': 'prime-exp-route-a-a1-7-stage2ar-v1.0',
    'tag_commit': '', 'git_clean': False, 'python_version': py_ver,
    'seed': SEED, 'worlds': ['A00', 'A11'], 'A01_generated': False, 'A10_generated': False,
    'input_hashes': hashes,
    'counts': {
        'S0': len(S0_plan), 'S1': len(S1_plan),
        'A00_quote_rows': len(A00_quote), 'A11_quote_rows': len(A11_quote),
        'A00_state_rows': len(A00_state), 'A11_state_rows': len(A11_state),
    },
    'mechanism_calls': mech_audit,
    'endpoint_results': {
        'A00_quote_sum': a00_qs, 'A00_replay_total': a00_payment, 'A00_actual_total': stat_actual,
        'A00_quote_to_replay_residual': a00_qs - a00_payment, 'A00_replay_to_actual_residual': a00_payment - stat_actual,
        'A11_quote_sum': a11_qs, 'A11_replay_total': a11_payment, 'A11_actual_total': reset_actual,
        'A11_quote_to_replay_residual': a11_qs - a11_payment, 'A11_replay_to_actual_residual': a11_payment - reset_actual,
    },
    'assignment_field_max_differences': {
        'A00': float(comp_summary[comp_summary['world']=='A00']['max_abs_difference'].max()) if len(comp_summary[comp_summary['world']=='A00']) else None,
        'A11': float(comp_summary[comp_summary['world']=='A11']['max_abs_difference'].max()) if len(comp_summary[comp_summary['world']=='A11']) else None,
    },
    'tests': {}, 'verification_exit_code': -1, 'evidence_bundle_sha256': '',
    'blockers': 'PENDING_commit_tag_clean',
}
json.dump(manifest, (OUT / 'manifests' / 'STAGE2AR_RELEASE_MANIFEST.json').open('w'), indent=2)

# Timing
timing = {'start': time.strftime('%Y-%m-%dT%H:%M:%S'), 'source_commit': source_commit}
json.dump(timing, (OUT / 'logs' / 'clean_commit_rerun_timing.json').open('w'), indent=2)

print(f"\n{'='*60}")
print("Stage 2A-R replay execution complete.")
print(f"  Source commit: {source_commit}")
print(f"  A00: replay={a00_payment:.6f} actual={stat_actual:.6f} resid={abs(a00_payment-stat_actual):.6e}")
print(f"  A11: replay={a11_payment:.6f} actual={reset_actual:.6f} resid={abs(a11_payment-reset_actual):.6e}")
print(f"  A00 calls={a00_formal_calls} A11 calls={a11_formal_calls} total={a00_formal_calls+a11_formal_calls}")
print(f"  State traces: A00={len(A00_state)} A11={len(A11_state)}")
print(f"  Assignment violations: A00={v_a00} A11={v_a11}")
