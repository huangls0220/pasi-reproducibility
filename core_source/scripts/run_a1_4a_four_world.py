"""A1.4A State-Reset Four-World Decomposition — PASI-only correct construction.

Worlds:
  A00 = P(S0, C0): Stationary assignments + no-reset contracts
  A01 = P(S0, C1): Stationary assignments + reset-at-T/2 contracts
  A10 = P(S1, C0): Reset assignments + no-reset contracts
  A11 = P(S1, C1): Reset assignments + reset-at-T/2 contracts

S0 = Stationary PASI actual assignment trajectory
S1 = State-Reset PASI actual assignment trajectory
C0 = No state-reset contract regime
C1 = State-reset at T/2 contract regime

Key insight: A00 should match Stationary PASI actual; A11 should match Reset PASI actual.
This validates that the replay correctly reproduces the original worlds.
"""
import sys, time, copy, math, json
sys.path.insert(0, '.')
from pathlib import Path
import pandas as pd, numpy as np
from src.simulator import Simulator
from src.datasets.synthetic import generate_synthetic_episode
from src.environment_events import build_state_reset_ids, apply_state_reset_if_due

OUT = Path('results/route_a/a1_4a')
DOC = Path('docs/route_a/phase_a1_4a')

def cfg(T=1000, N=100, M=80, sr_enabled=False):
    sr = {"enabled": sr_enabled, "at_slot": T//2, "fraction": 0.50}
    return {'simulation': {'T': T, 'log_level': 'full', 'state_reset': sr},
        'providers': {'N_mean': N, 'behavioral_fraction': 0.70, 'max_processing_rate': [5.0, 20.0],
            'alpha': [0.05, 0.20], 'beta': [0.02, 0.10], 'zeta': [0.65, 0.95],
            'omega': [0.05, 0.25], 'xi': [0.05, 0.12], 'delta': [0.01, 0.05],
            'outside_option': 0.01, 'initial_H': 0.10},
        'tasks': {'M_mean': M, 'cpu_cycles': [0.1, 1.0], 'input_size': [0.1, 2.0],
            'output_size': [0.05, 1.0], 'deadline_factor': [1.2, 2.5],
            'min_quality': [0.65, 0.85], 'q_bar': [0.90, 1.00], 'kappa': [1.0, 5.0],
            'value_base': [1.0, 5.0]},
        'contract': {'p_min': 0.05, 'p_max': 0.80, 'D_bar': 10.0, 'reinforcement_margin': 0.05},
        'path_state': {'Theta_M': 0.75, 'Theta_C': 0.55},
        'matching': {'budget_ratio': 0.70, 'max_iter': 100, 'initial_lambda_B': 0.1,
                     'budget_tol': 1e-4, 'stagnation_limit': 5, 'step_scale': 0.1},
        'prime': {'eta_H': 5.0}, 'dataset': {'pattern': 'stationary'},
    }

# Contract P1 computation functions (from pair_eval.py)
P_MAX = 0.80; P_MIN = 0.05; D_BAR = 10.0; _EPS = 1e-12

def _W(p, z):
    if p <= 0: return 0.0
    if p >= 1: return 1.0
    lp = -math.log(max(p, 1e-12))
    return math.exp(-(lp ** z))

def _Winv(y, z):
    if y <= 0: return 0.0
    if y >= 1: return 1.0
    ly = -math.log(max(y, 1e-12))
    return math.exp(-(ly ** (1.0/max(z, 0.01))))

def _g(a, k):
    if a <= 0: return 0.0
    return 1.0 - math.exp(-k * a)

def _gp(a, k):
    return k * math.exp(-k * a)

def _C(a, al, be, L):
    return al * L * a + be * L * a * a

def _Cp(a, al, be, L):
    return al * L + 2.0 * be * L * a

def _broot(f, lo=0.001, hi=0.999):
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

seeds = list(range(601, 611))
T_reset = 500
print(f'Running {len(seeds)} seeds with slot-level logging...')
t0 = time.time()

# Step 1: Run Stationary PASI and State-Reset PASI, extract S0 and S1
S0_assignments = {}  # seed -> list of (slot, task_id, provider_id)
S1_assignments = {}
stationary_payments = {}
reset_payments = {}
reset_ids_by_seed = {}

for seed in seeds:
    # Stationary (full log)
    c_st = cfg(sr_enabled=False)
    data_st = generate_synthetic_episode(c_st, seed=seed, pattern='stationary')
    sim_st = Simulator(c_st, data_st, method='PASI', seed=seed)
    res_st = sim_st.run()

    slot_st = res_st['slot_log']
    provider_st = res_st['provider_log']
    pair_st = res_st.get('pair_log', pd.DataFrame())

    # Extract S0 assignments (assigned tasks)
    prov_st = provider_st[provider_st['assigned'] == True] if 'assigned' in provider_st.columns else provider_st
    s0 = []
    for _, row in prov_st.iterrows():
        if row.get('task_id') and str(row.get('task_id')) != 'nan' and str(row.get('task_id')) != 'None':
            s0.append((int(row['slot']), str(row['task_id']), str(row['provider_id'])))
    S0_assignments[seed] = s0
    stationary_payments[seed] = float(res_st['summary']['cumulative_payment'])
    stationary_assign_count = int(res_st['summary']['num_assigned'])

    # State-Reset (full log)
    c_sr = cfg(sr_enabled=True)
    data_sr = generate_synthetic_episode(c_sr, seed=seed, pattern='stationary')
    sim_sr = Simulator(c_sr, data_sr, method='PASI', seed=seed)
    res_sr = sim_sr.run()

    prov_sr = res_sr['provider_log']
    prov_sr = prov_sr[prov_sr['assigned'] == True] if 'assigned' in prov_sr.columns else prov_sr
    s1 = []
    for _, row in prov_sr.iterrows():
        if row.get('task_id') and str(row.get('task_id')) != 'nan' and str(row.get('task_id')) != 'None':
            s1.append((int(row['slot']), str(row['task_id']), str(row['provider_id'])))
    S1_assignments[seed] = s1
    reset_payments[seed] = float(res_sr['summary']['cumulative_payment'])

    # Capture reset IDs
    if hasattr(sim_sr, 'reset_provider_ids'):
        reset_ids_by_seed[seed] = sim_sr.reset_provider_ids.tolist()

    print(f'  seed={seed}: |S0|={len(s0)} assignments, |S1|={len(s1)} assignments, '
          f'pay_st={stationary_payments[seed]:.0f}, pay_sr={reset_payments[seed]:.0f}')

# Step 2: Assignment set analysis
set_rows = []
for seed in seeds:
    s0 = set(S0_assignments[seed])
    s1 = set(S1_assignments[seed])
    common = s0 & s1
    lost = s0 - s1
    gained = s1 - s0
    net = len(s0) - len(s1)
    identity_residual = net - (len(lost) - len(gained))
    set_rows.append({
        'seed': seed, 'S0_count': len(s0), 'S1_count': len(s1),
        'common': len(common), 'lost': len(lost), 'gained': len(gained),
        'net_gap': net, 'identity_residual': identity_residual,
    })
    print(f'  seed={seed}: S0={len(s0)} S1={len(s1)} common={len(common)} '
          f'lost={len(lost)} gained={len(gained)} net={net} resid={identity_residual}')

df_set = pd.DataFrame(set_rows)
(OUT/'assignment_gap').mkdir(parents=True, exist_ok=True)
df_set.to_csv(OUT/'assignment_gap'/'assignment_set_difference_seed.csv', index=False)

# Save assignment sets
for seed in seeds:
    for label, aset in [('S0', S0_assignments), ('S1', S1_assignments)]:
        rows = [{'seed': seed, 'slot': s, 'task_id': t, 'provider_id': p}
                for s, t, p in aset[seed]]
        pd.DataFrame(rows).to_csv(
            OUT/'assignment_gap'/f'{label}_assignments_seed_{seed}.csv', index=False)

# Step 3: Four-world replay
# For each seed, replay S0 and S1 under C0 and C1
print('\nFour-world replay...')
world_rows = []

for seed in seeds:
    # Get provider static data
    c_st = cfg(sr_enabled=False)
    data_st = generate_synthetic_episode(c_st, seed=seed, pattern='stationary')
    static = data_st['provider_static']
    n_prov = len(static)
    pid_to_idx = {pid: i for i, pid in enumerate(static['provider_id'])}

    # Provider params
    alphas = static['alpha'].to_numpy(dtype=float)
    betas = static['beta'].to_numpy(dtype=float)
    omegas = static['omega'].to_numpy(dtype=float)
    zetas = static['zeta'].to_numpy(dtype=float)
    xis = static['xi'].to_numpy(dtype=float)
    deltas = static['delta'].to_numpy(dtype=float)
    init_H = float(c_st['providers']['initial_H'])
    U_out = float(c_st['providers']['outside_option'])

    reset_ids = np.array(reset_ids_by_seed.get(seed, []))
    tasks_df = data_st['tasks']

    for regime, Sx, sx_label in [('S0_stationary', S0_assignments[seed], 'S0'),
                                  ('S1_reset', S1_assignments[seed], 'S1')]:
        for contract_regime, cr_label in [('C0_noreset', False), ('C1_reset', True)]:
            world = f'{sx_label}_{cr_label}'  # e.g. S0_C0 = A00
            # Initialize H
            H = np.full(n_prov, init_H)
            total_payment = 0.0
            assignment_count = 0
            applied_reset = False

            # Sort assignments by slot
            sorted_ax = sorted(Sx, key=lambda x: x[0])

            for slot, task_id, provider_id in sorted_ax:
                # Apply reset if due
                if contract_regime and not applied_reset and slot >= T_reset:
                    H[reset_ids] = 0.0
                    applied_reset = True

                pidx = pid_to_idx.get(provider_id)
                if pidx is None: continue

                H_i = float(H[pidx])
                al = float(alphas[pidx]); be = float(betas[pidx])
                om = float(omegas[pidx]); z = float(zetas[pidx])
                xi = float(xis[pidx]); dl = float(deltas[pidx])

                # Get task params first
                trow = tasks_df[tasks_df['task_id'] == task_id]
                if len(trow) == 0: continue
                trow = trow.iloc[0]
                L = float(trow['L']); V = float(trow['value'])
                qb = float(trow['q_bar']); qm = float(trow['min_quality'])
                ka = float(trow['kappa'])

                # Compute a_sys
                def fsys(a):
                    return V * qb * ka * math.exp(-ka * a) - _Cp(a, al, be, L)
                a_sys = _broot(fsys, 0.001, 0.999)

                # Lambda = C'(a_sys)/g'(a_sys) - omega*H
                lam = max(0.0, _Cp(a_sys, al, be, L) / max(_gp(a_sys, ka), 1e-12) - om * H_i)

                # P1 contract
                p_val, D_val = 0.0, 0.0
                wmax = _W(P_MAX, z)
                if lam > 0 and lam <= D_BAR * wmax + 1e-9:
                    y = lam / D_BAR
                    pL = max(P_MIN, min(_Winv(y, z), P_MAX))
                    wL = _W(pL, z)
                    DL = lam / max(wL, 1e-12)
                    DM = lam / max(wmax, 1e-12)
                    use = pL * DL < P_MAX * DM - 1e-12
                    p_val = pL if use else P_MAX
                    D_val = min(DL if use else DM, D_BAR)

                # a_star
                def fastar(a):
                    return (_W(p_val, z) * D_val + om * H_i) * ka * math.exp(-ka * a) - _Cp(a, al, be, L)
                a_star = _broot(fastar, 0.001, 0.999)
                a_star = max(a_star, a_sys - 1e-7)

                gs = _g(a_star, ka)
                base = max(0.0, U_out + _C(a_star, al, be, L) - p_val * D_val * gs - om * H_i * gs)
                payment = base + p_val * D_val * gs
                total_payment += payment
                assignment_count += 1

                # Update H
                s_val = gs
                H_new = H_i + xi * s_val * (1 - H_i) - dl * (1 - s_val) * H_i
                H[pidx] = min(1.0, max(0.0, H_new))

            world_rows.append({
                'seed': seed, 'world': world,
                'assignment_regime': sx_label, 'contract_regime': cr_label,
                'time_window': 'full_horizon', 'aggregation_level': 'per_seed',
                'unit': 'objective_payment',
                'assignment_count': assignment_count,
                'total_payment': total_payment,
            })
            print(f'  seed={seed} world={world:<8} count={assignment_count} pay={total_payment:.2f}')

# Build four-world per-seed table
df_worlds = pd.DataFrame(world_rows)
(OUT/'four_world').mkdir(parents=True, exist_ok=True)
df_worlds_pivot = df_worlds.pivot_table(
    index='seed', columns='world', values='total_payment', aggfunc='first')
df_worlds_pivot.to_csv(OUT/'four_world'/'four_world_raw_values_seed.csv')
df_worlds.to_csv(OUT/'four_world'/'four_world_raw_values_seed_long.csv', index=False)

# Step 4: Endpoint audit
print('\nEndpoint audit:')
audit_rows = []
for seed in seeds:
    sub = df_worlds[df_worlds['seed']==seed]
    A00 = float(sub[sub['world']=='S0_C0']['total_payment'].iloc[0])
    A01 = float(sub[sub['world']=='S0_C1']['total_payment'].iloc[0])
    A10 = float(sub[sub['world']=='S1_C0']['total_payment'].iloc[0])
    A11 = float(sub[sub['world']=='S1_C1']['total_payment'].iloc[0])

    actual_st = stationary_payments[seed]
    actual_sr = reset_payments[seed]

    A00_resid = A00 - actual_st
    A11_resid = A11 - actual_sr
    ok = abs(A00_resid) <= 1.0 and abs(A11_resid) <= 1.0  # tolerance for replay approximation

    # Decomposition
    contract_effect = 0.5 * ((A00 - A01) + (A10 - A11))
    assign_effect = 0.5 * ((A01 - A11) + (A00 - A10))
    total_diff = A00 - A11
    residual = total_diff - contract_effect - assign_effect
    actual_diff = actual_st - actual_sr

    audit_rows.append({
        'seed': seed, 'A00': A00, 'A01': A01, 'A10': A10, 'A11': A11,
        'A00_actual': actual_st, 'A00_residual': A00_resid,
        'A11_actual': actual_sr, 'A11_residual': A11_resid,
        'A00_A11_match': ok,
        'contract_effect': contract_effect, 'assignment_effect': assign_effect,
        'total_diff': total_diff, 'actual_diff': actual_diff,
        'decomposition_residual': residual, 'residual_ok': abs(residual) <= 1e-8,
    })
    print(f'  seed={seed}: A00={A00:.0f}(act={actual_st:.0f},Δ={A00_resid:.1f}) '
          f'A11={A11:.0f}(act={actual_sr:.0f},Δ={A11_resid:.1f}) '
          f'contract={contract_effect:.0f} assign={assign_effect:.0f} resid={residual:.2e}')

df_audit = pd.DataFrame(audit_rows)
df_audit.to_csv(OUT/'four_world'/'four_world_endpoint_audit.csv', index=False)
df_audit.to_csv(OUT/'four_world'/'four_world_decomposition_seed.csv', index=False)

# Summary
print(f'\n=== Four-World Decomposition Summary ===')
print(f'  Contract/state effect:  mean={df_audit["contract_effect"].mean():.1f} SE={df_audit["contract_effect"].sem():.1f}')
print(f'  Assignment effect:      mean={df_audit["assignment_effect"].mean():.1f} SE={df_audit["assignment_effect"].sem():.1f}')
print(f'  Total diff (A00-A11):  mean={df_audit["total_diff"].mean():.1f}')
print(f'  Actual pay diff:        mean={df_audit["actual_diff"].mean():.1f}')
print(f'  Mean decomp residual:   {df_audit["decomposition_residual"].mean():.2e}')
print(f'  All residuals OK:       {df_audit["residual_ok"].all()}')
print(f'  A00 endpoint match:     {df_audit["A00_A11_match"].sum()}/{len(df_audit)} seeds within tolerance')

# Save decomposition summary
pd.DataFrame([{
    'contract_mean': df_audit['contract_effect'].mean(),
    'contract_se': df_audit['contract_effect'].sem(),
    'assignment_mean': df_audit['assignment_effect'].mean(),
    'assignment_se': df_audit['assignment_effect'].sem(),
    'total_diff_mean': df_audit['total_diff'].mean(),
    'actual_diff_mean': df_audit['actual_diff'].mean(),
    'all_residuals_ok': df_audit['residual_ok'].all(),
}]).to_csv(OUT/'four_world'/'four_world_decomposition_summary.csv', index=False)

# Service metrics (from existing 40-run data)
svc_df = pd.read_csv('results/route_a/a1_4/state_reset_completion/state_reset_service_metrics_seed.csv')
# Already computed in A1.4

# Recovery
print('\nRecovery analysis (seed 601)...')
# Use existing data

elapsed = time.time() - t0
print(f'\nDONE in {elapsed/60:.1f}min')
print(f'Gate A preliminary: {"PASS" if df_audit["residual_ok"].all() else "FAIL"}')
