"""Run cumulative bridge C0-C11 with telescope identity verification."""
import sys, time, math, copy
sys.path.insert(0, '.')
from pathlib import Path
import pandas as pd, numpy as np
from src.simulator import Simulator
from src.datasets.synthetic import generate_synthetic_episode

def bcfg(T=1000, N=100, M=80):
    return {"simulation": {"T": T, "log_level": "summary", "state_reset": {"enabled": False}},
        "providers": {"N_mean": N, "behavioral_fraction": 0.70, "max_processing_rate": [5.0, 20.0],
            "alpha": [0.05, 0.20], "beta": [0.02, 0.10], "zeta": [0.65, 0.95],
            "omega": [0.05, 0.25], "xi": [0.05, 0.12], "delta": [0.01, 0.05],
            "outside_option": 0.01, "initial_H": 0.10},
        "tasks": {"M_mean": M, "cpu_cycles": [0.1, 1.0], "input_size": [0.1, 2.0],
            "output_size": [0.05, 1.0], "deadline_factor": [1.2, 2.5],
            "min_quality": [0.65, 0.85], "q_bar": [0.90, 1.00], "kappa": [1.0, 5.0],
            "value_base": [1.0, 5.0]},
        "contract": {"p_min": 0.05, "p_max": 0.80, "D_bar": 10.0, "reinforcement_margin": 0.05},
        "path_state": {"Theta_M": 0.75, "Theta_C": 0.55},
        "matching": {"budget_ratio": 0.70, "max_iter": 100, "initial_lambda_B": 0.1,
                     "budget_tol": 1e-4, "stagnation_limit": 5, "step_scale": 0.1},
        "prime": {"eta_H": 5.0}, "dataset": {"pattern": "stationary"},
    }

def m(cfg, sect, key, val):
    n = copy.deepcopy(cfg); n[sect][key] = val; return n

# Cumulative chain
cs = {}
cs["C0_legacy"] = bcfg()
cs["C1_initH"] = m(cs["C0_legacy"], "providers", "initial_H", 0.20)
cs["C2_omega"] = m(cs["C1_initH"], "providers", "omega", [0.08, 0.15])
cs["C3_Dbar"] = m(cs["C2_omega"], "contract", "D_bar", 8.0)
cs["C4_kappa"] = m(cs["C3_Dbar"], "tasks", "kappa", [2.0, 3.5])
cs["C5_value"] = m(cs["C4_kappa"], "tasks", "value_base", [1.5, 3.5])
cs["C6_alpha"] = m(cs["C5_value"], "providers", "alpha", [0.08, 0.12])
cs["C7_beta"] = m(cs["C6_alpha"], "providers", "beta", [0.03, 0.06])
cs["C8_zeta"] = m(cs["C7_beta"], "providers", "zeta", [0.75, 0.85])
cs["C9_T"] = m(cs["C8_zeta"], "simulation", "T", 500)
c10 = copy.deepcopy(cs["C9_T"])
c10["providers"]["N_mean"] = 50; c10["tasks"]["M_mean"] = 40
cs["C10_scale"] = c10
cs["C11_smoke"] = {"simulation": {"T": 500, "log_level": "summary", "state_reset": {"enabled": False}},
    "providers": {"N_mean": 50, "behavioral_fraction": 1.0, "max_processing_rate": [10.0, 15.0],
        "alpha": [0.08, 0.12], "beta": [0.03, 0.06], "zeta": [0.75, 0.85],
        "omega": [0.08, 0.15], "xi": [0.06, 0.10], "delta": [0.02, 0.04],
        "outside_option": 0.01, "initial_H": 0.20},
    "tasks": {"M_mean": 40, "cpu_cycles": [0.2, 0.8], "input_size": [0.1, 1.0],
        "output_size": [0.05, 0.5], "deadline_factor": [1.5, 2.5],
        "min_quality": [0.60, 0.80], "q_bar": [0.90, 0.95], "kappa": [2.0, 3.5],
        "value_base": [1.5, 3.5]},
    "contract": {"p_min": 0.05, "p_max": 0.80, "D_bar": 8.0, "reinforcement_margin": 0.05},
    "path_state": {"Theta_M": 0.75, "Theta_C": 0.55},
    "matching": {"budget_ratio": 0.70, "max_iter": 50},
    "prime": {"eta_H": 5.0}, "dataset": {"pattern": "stationary"},
}

order = list(cs.keys())
seeds = list(range(1, 11))
total = len(order) * 2 * len(seeds)
t0 = time.time()
rows = []
pay_data = {}

for cid in order:
    cfg = cs[cid]
    pat = cfg.get("dataset", {}).get("pattern", "stationary")
    for seed in seeds:
        data = generate_synthetic_episode(cfg, seed=seed, pattern=pat)
        for method in ["MOI", "PASI"]:
            sim = Simulator(cfg, data, method=method, seed=seed)
            res = sim.run()
            s = res["summary"]
            pay = float(s["cumulative_payment"])
            rows.append({"cum_id": cid, "method": method, "seed": seed, "payment": pay})
            pay_data[(cid, seed, method)] = pay
            done = len(rows)
            if done % 24 == 0 or done == total:
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                print(f"[{done}/{total}] {cid} s{seed} {method} pay={pay:.1f} eta={eta/60:.0f}m")

df = pd.DataFrame(rows)
outdir = Path("results/route_a/a1_3_final/cumulative_bridge")
outdir.mkdir(parents=True, exist_ok=True)
df.to_csv(outdir / "cumulative_bridge_all_runs.csv", index=False)

# Seed-level relative savings
rel_data = {}
for cid in order:
    for seed in seeds:
        m = pay_data[(cid, seed, "MOI")]
        p = pay_data[(cid, seed, "PASI")]
        rel_data[(cid, seed)] = (m - p) / m

rows2 = []
for cid in order:
    for seed in seeds:
        rows2.append({"cum_id": cid, "seed": seed,
                      "pay_moi": pay_data[(cid, seed, "MOI")],
                      "pay_pasi": pay_data[(cid, seed, "PASI")],
                      "rel": rel_data[(cid, seed)]})
pd.DataFrame(rows2).to_csv(outdir / "cumulative_bridge_seed_level.csv", index=False)

# Telescope identity
step_vals = []
prev = None
for cid in order:
    curr = np.array([rel_data[(cid, s)] for s in seeds])
    if prev is not None:
        diffs = curr - prev
        step_vals.append(float(np.mean(diffs)))
    prev = curr

R_C0 = float(np.mean([rel_data[(order[0], s)] for s in seeds]))
R_C11 = float(np.mean([rel_data[(order[-1], s)] for s in seeds]))
Gap = R_C11 - R_C0
Sum_step = sum(step_vals)
Cum_res = Gap - Sum_step

print()
for cid in order:
    r = np.mean([rel_data[(cid, s)] for s in seeds])
    print(f"  {cid:<14} rel={r*100:.6f}%")

print(f"\n  R_C0: {R_C0*100:.6f}%    R_C11: {R_C11*100:.6f}%")
print(f"  Gap_total: {Gap*100:.6f} pp")
print(f"  Sum_step:  {Sum_step*100:.6f} pp")
print(f"  Cumulative_residual: {Cum_res*100:.8f} pp")
print(f"  Telescope identity: <=1e-10 = {abs(Cum_res) <= 1e-10}")

# Arithmetic audit
sumr = [{"metric": "R_C0", "value": R_C0 * 100},
        {"metric": "R_C11", "value": R_C11 * 100},
        {"metric": "Gap_total", "value": Gap * 100},
        {"metric": "Sum_step", "value": Sum_step * 100},
        {"metric": "Cumulative_residual", "value": float(Cum_res * 100)}]
for i, sv in enumerate(step_vals):
    sumr.append({"metric": f"Step_C{i}_to_C{i+1}", "value": sv * 100})
pd.DataFrame(sumr).to_csv(outdir / "cumulative_bridge_arithmetic_audit.csv", index=False)

# B10 audit
c9 = cs["C9_T"]
c10c = cs["C10_scale"]
changed = []
for sect in ["providers", "tasks", "contract", "simulation"]:
    d9 = c9.get(sect, {}) or {}
    d10 = c10c.get(sect, {}) or {}
    for k in d9:
        v9 = str(d9.get(k)); v10 = str(d10.get(k))
        if v9 != v10: changed.append(f"{sect}.{k}")
print(f"\n  C9->C10 changed: {changed}")
print(f"  T unchanged: {c9['simulation']['T']} == {c10c['simulation']['T']}")

elapsed = time.time() - t0
print(f"\n  DONE: {len(df)} runs in {elapsed/60:.1f}min")
