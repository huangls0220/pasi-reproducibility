"""A1.6 Stage 2 — Four-world replay using Simulator scalar evaluate_pair."""
import sys, time, copy, json, hashlib
sys.path.insert(0, '.')
from pathlib import Path
import pandas as pd, numpy as np
from src.simulator import Simulator
from src.datasets.synthetic import generate_synthetic_episode
from src.contracts import evaluate_pair
from src.mechanisms import get_mechanism

SEED = 601; T = 1000; N = 100; M = 80; TH = T//2
OUT = Path('results/route_a/a1_6_stage2')
for d in ['four_world','audit','manifests']: (OUT/d).mkdir(parents=True, exist_ok=True)

# Config + mechanism
cfg = {'contract': {'p_min': 0.05, 'p_max': 0.80, 'D_bar': 10.0, 'reinforcement_margin': 0.05},
       'path_state': {'Theta_M': 0.75, 'Theta_C': 0.55}, 'prime': {'eta_H': 5.0}}
mech = get_mechanism('PASI', cfg)

print("Phase 1: Run base worlds")
c_full = {
    'simulation': {'T': T, 'log_level': 'full', 'state_reset': {'enabled': False, 'at_slot': TH, 'fraction': 0.50}},
    'providers': {'N_mean': N, 'behavioral_fraction': 0.70, 'max_processing_rate': [5.0, 20.0],
        'alpha': [0.05, 0.20], 'beta': [0.02, 0.10], 'zeta': [0.65, 0.95], 'omega': [0.05, 0.25],
        'xi': [0.05, 0.12], 'delta': [0.01, 0.05], 'outside_option': 0.01, 'initial_H': 0.10},
    'tasks': {'M_mean': M, 'cpu_cycles': [0.1, 1.0], 'input_size': [0.1, 2.0], 'output_size': [0.05, 1.0],
        'deadline_factor': [1.2, 2.5], 'min_quality': [0.65, 0.85], 'q_bar': [0.90, 1.00], 'kappa': [1.0, 5.0],
        'value_base': [1.0, 5.0]},
    'contract': cfg['contract'], 'path_state': cfg['path_state'], 'prime': cfg['prime'],
    'matching': {'budget_ratio': 0.70, 'max_iter': 100, 'initial_lambda_B': 0.1, 'budget_tol': 1e-4, 'stagnation_limit': 5, 'step_scale': 0.1},
    'dataset': {'pattern': 'stationary'},
}
STAGE1 = Path('results/route_a/a1_6_stage1')
static = pd.read_parquet(STAGE1/'event_tape'/'provider_static.parquet')
tasks_df = pd.read_parquet(STAGE1/'event_tape'/'tasks.parquet')
rids_df = pd.read_csv(STAGE1/'config'/'reset_ids_seed_601.csv')
reset_pids = set(rids_df[rids_df['is_reset']==True]['provider_id'].values)
pid_to_idx = {pid: i for i, pid in enumerate(static['provider_id'])}
n_prov = len(static)

base_pay = {}; S_traj = {}
for scenario, sr in [('stationary', False), ('state_reset', True)]:
    c = copy.deepcopy(c_full); c['simulation']['state_reset']['enabled'] = sr
    data = generate_synthetic_episode(c, seed=SEED, pattern='stationary')
    sim = Simulator(c, data, method='PASI', seed=SEED)
    res = sim.run()
    base_pay[scenario] = float(res['summary']['cumulative_payment'])
    pl = res['provider_log']
    ax = []; assn = pl[pl['assigned']==True] if 'assigned' in pl.columns else pd.DataFrame()
    for _, r in assn.iterrows():
        tid = r.get('task_id','')
        if not tid or str(tid)=='nan': continue
        ax.append({'slot':int(r['slot']),'task_id':str(tid),'provider_id':str(r['provider_id']),
                   'H_before':float(r.get('H_before',0)),'H_after':float(r.get('H_after',0))})
    S_traj[scenario] = sorted(ax, key=lambda a:a['slot'])
    print(f"  {scenario}: {len(ax)} assignments, pay={base_pay[scenario]:.2f}")

S0 = S_traj['stationary']; S1 = S_traj['state_reset']

# Phase 2: Four-world replay
print("\nPhase 2: Four-world replay (scalar evaluate_pair)")
P_MIN, P_MAX, D_BAR = 0.05, 0.80, 10.0
R_MARGIN, T_M, ETA_H, DP_MAX = 0.05, 0.75, 5.0, 0.05

def replay_world(assignments, apply_reset):
    H = np.full(n_prov, 0.10)
    tot = 0.0; cnt = 0; applied = False
    for a in assignments:
        slot = a['slot']; tid = a['task_id']; pid = a['provider_id']
        if apply_reset and not applied and slot >= TH:
            for ri in [pid_to_idx[p] for p in reset_pids if p in pid_to_idx]: H[ri] = 0.0
            applied = True
        pidx = pid_to_idx.get(pid)
        if pidx is None: continue
        Hi = float(H[pidx])
        trow = tasks_df[tasks_df['task_id']==tid]
        if len(trow)==0: continue
        tr = trow.iloc[0]
        sr = static.iloc[pidx]

        result = evaluate_pair(
            L=float(tr['L']), V=float(tr['value']), q_bar=float(tr['q_bar']), q_min=float(tr['min_quality']),
            kappa=float(tr['kappa']), alpha=float(sr['alpha']), beta=float(sr['beta']),
            omega=float(sr['omega']), zeta=float(sr['zeta']), xi=float(sr['xi']), delta=float(sr['delta']),
            H=Hi, stage='cultivation', U_out=0.01, p_min=P_MIN, p_max=P_MAX, D_bar=D_BAR,
            reinforcement_margin=R_MARGIN, Theta_M=T_M, eta_H=ETA_H,
            last_probability=0.0, delta_p_max=DP_MAX,
            F_i_t=float(sr['max_processing_rate']), effective_deadline=float(tr['deadline']),
            D_tr=(float(tr['input_size'])+float(tr['output_size']))/10.0,
            input_size=float(tr['input_size']), output_size=float(tr['output_size']), communication_rate=10.0,
        )
        pay = float(result['expected_contract_cost']) if result.get('feasible',True) else 0.0
        tot += pay; cnt += 1
        gs = float(result['normalized_quality']) if result.get('feasible',True) else 0.7
        H[pidx] = min(1.0, max(0.0, Hi + float(sr['xi'])*gs*(1-Hi) - float(sr['delta'])*(1-gs)*Hi))
    return tot, cnt

print("  A00 (S0+C0)..."); A00, c00 = replay_world(S0, False)
print("  A01 (S0+C1)..."); A01, c01 = replay_world(S0, True)
print("  A10 (S1+C0)..."); A10, c10 = replay_world(S1, False)
print("  A11 (S1+C1)..."); A11, c11 = replay_world(S1, True)

contract = 0.5*((A00-A01)+(A10-A11)); assign = 0.5*((A01-A11)+(A00-A10))
total = A00-A11; resid = total-contract-assign

print(f"\nA00={A00:.2f} A01={A01:.2f} A10={A10:.2f} A11={A11:.2f}")
print(f"A00 resid: {A00-base_pay['stationary']:.2f}  A11 resid: {A11-base_pay['state_reset']:.2f}")
print(f"Contract: {contract:.2f}  Assignment: {assign:.2f}  Total: {total:.2f}  Residual: {resid:.2e}")

ah0 = hashlib.sha256(str(sorted([(a['slot'],a['task_id'],a['provider_id']) for a in S0])).encode()).hexdigest()[:16]
ah1 = hashlib.sha256(str(sorted([(a['slot'],a['task_id'],a['provider_id']) for a in S1])).encode()).hexdigest()[:16]
ok = abs(A00-base_pay['stationary'])<50 and abs(A11-base_pay['state_reset'])<50

pd.DataFrame([{'seed':SEED,'world':w,'total_payment':p} for w,p in [('A00',A00),('A01',A01),('A10',A10),('A11',A11)]]).to_csv(OUT/'four_world'/'four_world_raw_seed.csv', index=False)
pd.DataFrame([{'seed':SEED,'A00':A00,'A01':A01,'A10':A10,'A11':A11,'contract_effect':contract,'assignment_effect':assign,'total_diff':total,'residual':resid}]).to_csv(OUT/'four_world'/'four_world_decomposition_seed.csv', index=False)
pd.DataFrame([{'seed':SEED,'world':'A00','replay':A00,'actual':base_pay['stationary'],'residual':A00-base_pay['stationary'],'pass':abs(A00-base_pay['stationary'])<50},{'seed':SEED,'world':'A11','replay':A11,'actual':base_pay['state_reset'],'residual':A11-base_pay['state_reset'],'pass':abs(A11-base_pay['state_reset'])<50}]).to_csv(OUT/'four_world'/'four_world_endpoint_audit.csv', index=False)
pd.DataFrame([{'seed':SEED,'A00_A01_hash':ah0,'A10_A11_hash':ah1,'S0_S1_different':ah0!=ah1}]).to_csv(OUT/'four_world'/'four_world_assignment_hash_audit.csv', index=False)
json.dump({'phase':'A1.6-Stage2','stage2':'PASS' if ok else 'PARTIAL','A00':A00,'A01':A01,'A10':A10,'A11':A11,'contract':contract,'assignment':assign},open(OUT/'manifests'/'STAGE2_RELEASE_MANIFEST.json','w'),indent=2)
print(f"\nStage 2: {'PASS' if ok else 'PARTIAL'} — scalar evaluate_pair replay")
