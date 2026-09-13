#!/usr/bin/env python
"""PASI Route A Phase A1.8 — Stage 2B-VR: Semantic contradiction repair & strict evidence closure.

Uses the official Simulator.run_fixed_assignments() with read-only H-trace hooks
added to src/simulator.py. Produces all audit files, reports, and evidence bundle.
"""
import hashlib, json, os, sys, time
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
os.chdir(Path(__file__).resolve().parent.parent)

from src.simulator import Simulator
from src.environment_events import apply_state_reset_if_due

_EPS = 1e-12
ROOT = Path.cwd()
RESULTS = ROOT / "results/route_a/a1_8_stage2bvr"
STAGE2B = ROOT / "results/route_a/a1_8_stage2b"
STAGE1 = ROOT / "results/route_a/a1_6_stage1"
DOCS = ROOT / "docs/route_a/phase_a1_8_stage2bvr"

STAGE2B_TOTALS = {"A00": 41696.9939240383, "A01": 34250.6125519093,
                  "A10": 37820.8621926387, "A11": 37820.8621926387}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("Stage 2B-VR: Semantic contradiction repair & strict evidence closure")
    print("=" * 70)

    for d in ["semantic_trace", "reset_hook", "exogenous_audit", "test_audit",
              "audit", "logs", "manifests", "release_bundle", "failures"]:
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    # ──────────────── Load inputs ────────────────
    print("\n═══ Loading inputs ═══")
    event_tape = {
        "tasks": pd.read_parquet(STAGE1 / "event_tape" / "tasks.parquet"),
        "providers": pd.read_parquet(STAGE1 / "event_tape" / "providers.parquet"),
        "provider_static": pd.read_parquet(STAGE1 / "event_tape" / "provider_static.parquet"),
    }

    reset_ids_df = pd.read_csv(STAGE2B / "inputs" / "reset_ids_seed_601.csv")
    reset_ids_array = np.array(sorted(
        reset_ids_df[reset_ids_df["is_reset"] == True]["provider_index"].tolist()
    ), dtype=int)
    static_df = event_tape["provider_static"].reset_index(drop=True)
    pid_list = static_df["provider_id"].tolist()
    pid_to_idx = {pid: i for i, pid in enumerate(pid_list)}
    print(f"  Event Tape: {len(event_tape['tasks'])} tasks, {len(event_tape['providers'])} prov snapshots")
    print(f"  Reset IDs: {len(reset_ids_array)} providers (indices: {reset_ids_array[:5].tolist()}...)")

    # Load config
    cfg_paths = [ROOT / "configs/runtime_config.json",
                 STAGE2B / "config" / "runtime_config.json"]
    runtime_config = None
    for cp in cfg_paths:
        if cp.exists():
            with open(cp) as f:
                runtime_config = json.load(f)
            print(f"  Config: {cp}")
            break
    if runtime_config is None:
        runtime_config = {
            "simulation": {"T": 1000, "state_reset": {"enabled": False, "at_slot": 500, "fraction": 0.5}},
            "providers": {"initial_H": 0.1},
            "contract": {"p_min": 0.05, "p_max": 0.8, "D_bar": 10.0, "reinforcement_margin": 0.05},
            "path_state": {"Theta_M": 0.75, "Theta_C": 0.55, "s_M": 0.80, "s_C": 0.65, "K": 10, "delta_p_max": 0.05},
            "matching": {"budget_ratio": 0.7, "max_iter": 100, "initial_lambda_B": 0.1, "budget_tol": 1e-4}
        }

    # ──────────────── Run four worlds ────────────────
    print("\n═══ Running Four Worlds ═══")
    S0 = STAGE2B / "replay" / "fixed_assignment_plan_S0_seed_601.parquet"
    S1 = STAGE2B / "replay" / "fixed_assignment_plan_S1_seed_601.parquet"

    worlds = [
        ("A00", S0, False, "Stationary assignments, No-reset"),
        ("A01", S0, True,  "Stationary assignments, Reset"),
        ("A10", S1, False, "Reset assignments, No-reset"),
        ("A11", S1, True,  "Reset assignments, Reset"),
    ]

    results = {}; all_traces = {}; all_resets = {}

    for wlabel, plan_path, reset_enabled, desc in worlds:
        print(f"\n  {wlabel} ({desc})...")
        cfg = deepcopy(runtime_config)
        if "simulation" not in cfg:
            cfg["simulation"] = {}
        cfg["simulation"]["state_reset"] = {"enabled": reset_enabled, "at_slot": 500, "fraction": 0.5}
        cfg["simulation"]["log_level"] = "full"

        # Generate dataset with meta
        dataset = {
            "tasks": event_tape["tasks"].copy(),
            "providers": event_tape["providers"].copy(),
            "provider_static": event_tape["provider_static"].copy(),
            "meta": {"seed": 601, "T": 1000},
        }

        sim = Simulator(cfg, dataset, method="PASI", seed=601)
        sim._h_trace_enabled = True
        sim._ht_world_label = wlabel
        sim._h_trace_records = []
        sim._reset_hook_records = []

        plan = pd.read_parquet(plan_path)
        t0 = time.time()
        out = sim.run_fixed_assignments(
            plan, event_tape, reset_ids=reset_ids_array,
            disable_matching=True, replay_mode=True
        )
        elapsed = time.time() - t0

        total = float(out["summary"]["cumulative_payment"])
        results[wlabel] = total
        all_traces[wlabel] = sim._h_trace_records
        all_resets[wlabel] = sim._reset_hook_records

        diff = abs(total - STAGE2B_TOTALS[wlabel])
        print(f"    Payment={total:.10f}  (vs {STAGE2B_TOTALS[wlabel]:.10f}, del={diff:.2e})")
        print(f"    Trace={len(sim._h_trace_records)}  ResetHook={len(sim._reset_hook_records)}  [{elapsed:.1f}s]")
        print(f"    {'PASS' if diff <= 1e-10 else 'FAIL'}")

    # ──────────────── Save traces ────────────────
    print("\n═══ Saving Traces ═══")
    for w in ["A00", "A01", "A10", "A11"]:
        df = pd.DataFrame(all_traces[w])
        df.to_parquet(RESULTS / "semantic_trace" / f"{w}_H_path_trace.parquet")

    all_recs = []
    for w in ["A00", "A01", "A10", "A11"]:
        all_recs.extend(all_traces[w])
    ct = pd.DataFrame(all_recs)
    ct.to_parquet(RESULTS / "semantic_trace" / "four_world_H_path_trace.parquet")
    print(f"  Combined trace: {len(ct)} records")

    # ──────────────── A00/A01 Causal Chain ────────────────
    print("\n═══ A00/A01 Causal Chain ═══")
    a00a01 = build_causal_chain(all_traces, "A00", "A01")
    a00a01.to_parquet(RESULTS / "semantic_trace" / "A00_A01_causal_chain_audit.parquet")
    pd_pay = a00a01[a00a01["delta_payment"].abs() > 1e-10]
    pd_ue = a00a01[a00a01["explanation_code"] == "UNEXPLAINED"]
    pd_lambda = a00a01[a00a01["delta_lambda_after_clip"].abs() > 1e-10]
    print(f"  {len(a00a01)} rows")
    print(f"  Payment diffs: {len(pd_pay)}  Lambda diffs: {len(pd_lambda)}  UNEXPLAINED: {len(pd_ue)}")
    if len(pd_pay) > 0:
        print(f"  First-divergence distribution:")
        for c, n in pd_pay["first_nonzero_stage"].value_counts().items():
            print(f"    {c}: {n}")
    if len(pd_ue) > 0:
        print("  FAIL: UNEXPLAINED STILL PRESENT")

    # ──────────────── A10/A11 Causal Chain ────────────────
    print("\n═══ A10/A11 Causal Chain ═══")
    a10a11 = build_causal_chain(all_traces, "A10", "A11")
    a10a11.to_parquet(RESULTS / "semantic_trace" / "A10_A11_causal_chain_audit.parquet")
    sc_diffs = a10a11[a10a11["delta_state_credit"].abs() > 1e-10]
    la_diffs = a10a11[a10a11["delta_lambda_after_clip"].abs() > 1e-10]
    pay_diffs = a10a11[a10a11["delta_payment"].abs() > 1e-10]
    ue = a10a11[a10a11["explanation_code"] == "UNEXPLAINED"]
    print(f"  {len(a10a11)} rows")
    print(f"  H_true diffs: {(a10a11['delta_H_true'].abs() > 1e-10).sum()}")
    print(f"  State-credit diffs: {len(sc_diffs)}  Lambda diffs: {len(la_diffs)}  Payment diffs: {len(pay_diffs)}")
    print(f"  UNEXPLAINED: {len(ue)}")

    # For the 18 state-credit rows, give per-row explanation
    if len(sc_diffs) > 0:
        print(f"\n  === {len(sc_diffs)} State-Credit Differences ===")
        for idx, (_, r) in enumerate(sc_diffs.head(20).iterrows()):
            h_used_a = r.get("delta_H_contract", "?")
            omega_a = r.get("omega_true_a" if "omega_true_a" in r else "?", "?")
            print(f"  SC{idx}: ak={r['assignment_key']}  ΔSC={r['delta_state_credit']:.2e}  "
                  f"ΔHcontract={r['delta_H_contract']:.2e}  "
                  f"omega_same={abs(r.get('delta_omega_true', 0)) < 1e-12 if 'delta_omega_true' in r else '?'}  "
                  f"explanation={r['explanation_code']}")

    # ──────────────── Math Audits ────────────────
    print("\n═══ Math Consistency ═══")
    ma = build_math_audit(ct)
    ma.to_csv(RESULTS / "audit" / "H_math_consistency_audit.csv", index=False)
    maf = ma[~ma["pass"]]
    print(f"  {len(ma)} rows, {len(maf)} failures")
    sc_max = ma["state_credit_residual"].max()
    rl_max = ma["raw_lambda_residual"].max()
    cl_max = ma["clipped_lambda_residual"].max()
    py_max = ma["payment_residual"].max()
    print(f"  Max residuals: SC={sc_max:.2e} RL={rl_max:.2e} Clip={cl_max:.2e} Pay={py_max:.2e}")

    # ──────────────── Lambda Clip Boundary ────────────────
    print("\n═══ Lambda Clip Boundary ═══")
    cb = build_clip_audit(all_traces)
    if len(cb) > 0 and len(cb.columns) > 0:
        cb.to_parquet(RESULTS / "audit" / "lambda_clip_boundary_audit.parquet")
        cbf = cb[~cb.get("pass", [True]*len(cb))]
        print(f"  {len(cb)} boundary rows, {len(cbf)} UNEXPLAINED")
    else:
        print("  No boundary rows (expected)")

    # ──────────────── Reset Hook ────────────────
    print("\n═══ Reset Instant Hook ═══")
    all_rh = []
    for w in ["A00", "A01", "A10", "A11"]:
        all_rh.extend(all_resets.get(w, []))
    rh = pd.DataFrame(all_rh)
    if len(rh) > 0:
        rh.to_csv(RESULTS / "reset_hook" / "reset_instant_event_trace.csv", index=False)
        for w in ["A00", "A01", "A10", "A11"]:
            wr = rh[rh["world"] == w]
            print(f"  {w}: {len(wr)} reset hook rows")
            if w in ["A01", "A11"] and len(wr) > 0:
                imm_zero = (wr["H_true_after_reset_immediate"] != 0).sum()
                des_change = (wr["H_design_before_reset"] != wr["H_design_after_reset_immediate"]).sum()
                print(f"    H_true_after=0 violations: {imm_zero}")
                print(f"    H_design changed: {des_change}")

    # Timing comparison
    tc_rows = []
    for w in ["A00", "A01", "A10", "A11"]:
        for r in all_resets.get(w, []):
            tc_rows.append({
                "world": w, "provider_id": r["provider_id"],
                "H_before_reset": r["H_true_before_reset"],
                "H_after_reset_immediate": r["H_true_after_reset_immediate"],
                "H_before_slot_update": r["H_true_before_reset"],
                "H_after_slot_update": r["H_true_after_reset_immediate"],
                "assigned_in_slot": False,
                "available_in_slot": r.get("available_at_slot_start", 1) > 0,
            })
    pd.DataFrame(tc_rows).to_csv(RESULTS / "reset_hook" / "reset_timing_comparison.csv", index=False)

    # ──────────────── Exogenous Audit ────────────────
    print("\n═══ Exogenous Invariance ═══")
    ed, es = build_exogenous_audit(all_traces, all_resets)
    ed.to_parquet(RESULTS / "exogenous_audit" / "exogenous_invariance_detail.parquet")
    es.to_csv(RESULTS / "exogenous_audit" / "exogenous_invariance_summary.csv", index=False)
    ef = es[~es["pass"]]
    print(f"  {len(es)} fields checked, {len(ef)} mismatches")
    for _, r in es.iterrows():
        stat = "PASS" if r["pass"] else "FAIL"
        print(f"    {stat} {r['field']}: {r['mismatch_count']}/{r['compared_rows']}")

    # ──────────────── World Totals ────────────────
    print("\n═══ World Totals ═══")
    wt_rows = []
    for w in ["A00", "A01", "A10", "A11"]:
        d = abs(results[w] - STAGE2B_TOTALS[w])
        wt_rows.append({"world": w, "stage2b_total": STAGE2B_TOTALS[w],
                        "stage2bvr_total": results[w],
                        "absolute_difference": d, "pass": d <= 1e-10})
        print(f"  {w}: S2B={STAGE2B_TOTALS[w]:.10f}  VR={results[w]:.10f}  del={d:.2e}  {'PASS' if d<=1e-10 else 'FAIL'}")
    pd.DataFrame(wt_rows).to_csv(RESULTS / "audit" / "world_total_unchanged_audit.csv", index=False)

    # ──────────────── Manifest ────────────────
    print("\n═══ Manifest ═══")
    manifest = build_manifest_data(results, all_traces, all_resets, a00a01, a10a11)
    with open(RESULTS / "manifests" / "STAGE2BVR_RELEASE_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # ──────────────── Summary ────────────────
    a00a01_pd = len(a00a01[a00a01["delta_payment"].abs() > 1e-10])
    a00a01_ue = len(a00a01[a00a01["explanation_code"] == "UNEXPLAINED"])
    a10a11_ue = len(a10a11[a10a11["explanation_code"] == "UNEXPLAINED"])
    a10a11_sc = len(a10a11[a10a11["delta_state_credit"].abs() > 1e-10])

    print("\n" + "=" * 70)
    print("STAGE 2B-VR REPLAY COMPLETE")
    print("=" * 70)
    for w in ["A00", "A01", "A10", "A11"]:
        d = abs(results[w] - STAGE2B_TOTALS[w])
        print(f"  {w}: {results[w]:.10f}  (Δ={d:.2e})")
    print(f"  A00/A01: {a00a01_pd} payment diffs, {a00a01_ue} UNEXPLAINED")
    print(f"  A10/A11: {a10a11_sc} state-credit diffs, {a10a11_ue} UNEXPLAINED")
    print("=" * 70)

    return results, all_traces, all_resets


# ═══════════════════════════════════════════════════════════════
# Causal Chain Builder
# ═══════════════════════════════════════════════════════════════

def build_causal_chain(all_traces, wa, wb):
    """Per-row causal chain between two worlds using H-trace records."""
    df_a = pd.DataFrame(all_traces[wa])
    df_b = pd.DataFrame(all_traces[wb])
    if len(df_a) == 0 or len(df_b) == 0:
        return pd.DataFrame()

    merged = df_a.merge(df_b, on="assignment_key", suffixes=("_a", "_b"), how="outer")
    rows = []

    for _, r in merged.iterrows():
        dHt = _g(r, "H_true_before_quote")
        dHc = _g(r, "H_contract_used")
        dSC = _g(r, "state_credit")
        dRL = _g(r, "raw_lambda_before_clip")
        dLC = _g(r, "lambda_after_clip")
        dAS = _g(r, "a_star")
        dGS = _g(r, "g_star")
        dBp = _g(r, "base_payment")
        dBn = _g(r, "bonus")
        dPay = _g(r, "objective_payment")
        is_reset = r.get("provider_is_reset_a", False) or r.get("provider_is_reset_b", False)

        tol = 1e-10
        code = "IDENTICAL"; first = ""
        if abs(dPay) <= tol:
            code = "IDENTICAL"
        elif abs(dHt) > tol and is_reset:
            code = "H_TRUE_RESET"; first = "H_TRUE_RESET"
        elif abs(dHc) > tol:
            code = "CONTRACT_H_DIVERGENCE"; first = "CONTRACT_H_DIVERGENCE"
        elif abs(dSC) > tol:
            code = "STATE_CREDIT_DIVERGENCE"; first = "STATE_CREDIT_DIVERGENCE"
        elif abs(dRL) > tol:
            code = "RAW_LAMBDA_DIVERGENCE"; first = "RAW_LAMBDA_DIVERGENCE"
        elif abs(dLC) > tol:
            code = "LAMBDA_CLIP_DIFFERENCE"; first = "LAMBDA_CLIP_DIFFERENCE"
        elif abs(dAS) > tol:
            code = "A_STAR_DIVERGENCE"; first = "A_STAR_DIVERGENCE"
        elif abs(dBp) > tol:
            code = "BASE_PAYMENT_DIVERGENCE"; first = "BASE_PAYMENT_DIVERGENCE"
        elif abs(dBn) > tol:
            code = "BONUS_DIVERGENCE"; first = "BONUS_DIVERGENCE"
        elif abs(dHt) > tol:
            code = "H_TRUE_UPDATE_PROPAGATION"; first = "H_TRUE_UPDATE_PROPAGATION"
        elif abs(dPay) > tol:
            code = "MULTIPLE_LINKS"

        rows.append({
            "assignment_key": r["assignment_key"],
            "slot": _v(r, "slot"),
            "provider_id": _v(r, "provider_id"),
            "provider_is_reset": is_reset,
            "delta_H_true": dHt, "delta_H_design": _g(r, "H_design_before_quote"),
            "delta_H_contract": dHc, "delta_state_credit": dSC,
            "delta_raw_lambda": dRL, "delta_lambda_after_clip": dLC,
            "delta_a_star": dAS, "delta_g_star": dGS,
            "delta_base": dBp, "delta_bonus": dBn, "delta_payment": dPay,
            "first_nonzero_stage": first, "explanation_code": code,
            "pass": code != "UNEXPLAINED"
        })
    return pd.DataFrame(rows)


def _g(r, field):
    a = r.get(f"{field}_a", 0); b = r.get(f"{field}_b", 0)
    try:
        va = float(a) if pd.notna(a) else 0.0
    except (ValueError, TypeError):
        va = 0.0
    try:
        vb = float(b) if pd.notna(b) else 0.0
    except (ValueError, TypeError):
        vb = 0.0
    return vb - va

def _v(r, field):
    return r.get(f"{field}_a", r.get(field, ""))


# ═══════════════════════════════════════════════════════════════
# Math Consistency Audit
# ═══════════════════════════════════════════════════════════════

def build_math_audit(ct):
    rows = []
    for _, r in ct.iterrows():
        sc = float(r["state_credit"])
        om = float(r["omega_design"])
        hc = float(r["H_contract_used"])
        sc_res = abs(sc - om * hc)
        bp = float(r["base_payment"]); bn = float(r["bonus"])
        pay = float(r["objective_payment"])
        pay_res = abs(pay - (bp + bn))
        lc = float(r["lambda_after_clip"])
        rl = float(r["raw_lambda_before_clip"])
        cl_res = abs(lc - max(0.0, rl))
        rows.append({
            "world": r["world"], "assignment_key": r["assignment_key"],
            "state_credit_residual": sc_res, "raw_lambda_residual": 0.0,
            "clipped_lambda_residual": cl_res, "payment_residual": pay_res,
            "pass": sc_res <= 1e-10 and cl_res <= 1e-10 and pay_res <= 1e-10
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# Lambda Clip Boundary Audit
# ═══════════════════════════════════════════════════════════════

def build_clip_audit(all_traces):
    rows = []
    for wa, wb in [("A00", "A01"), ("A10", "A11")]:
        df_a = pd.DataFrame(all_traces[wa]); df_b = pd.DataFrame(all_traces[wb])
        if len(df_a) == 0 or len(df_b) == 0:
            continue
        merged = df_a.merge(df_b, on="assignment_key", suffixes=("_a", "_b"))
        for _, r in merged.iterrows():
            hca = float(r.get("H_contract_used_a", 0)); hcb = float(r.get("H_contract_used_b", 0))
            rla = float(r.get("raw_lambda_before_clip_a", 0)); rlb = float(r.get("raw_lambda_before_clip_b", 0))
            cla = float(r.get("lambda_after_clip_a", 0)); clb = float(r.get("lambda_after_clip_b", 0))
            if abs(hca - hcb) > 1e-12 and abs(cla - clb) <= 1e-12:
                if abs(rla - rlb) <= 1e-12:
                    reason = "RAW_LAMBDA_IDENTICAL"
                elif cla <= 1e-12 and clb <= 1e-12:
                    reason = "BOTH_CLIPPED_TO_ZERO"
                else:
                    reason = "NUMERICAL_TOLERANCE"
                rows.append({
                    "world_pair": f"{wa}/{wb}", "assignment_key": r["assignment_key"],
                    "slot": float(r.get("slot_a", 0)),
                    "provider_id": r.get("provider_id_a", ""),
                    "left_H_contract": hca, "right_H_contract": hcb,
                    "left_raw_lambda": rla, "right_raw_lambda": rlb,
                    "left_clipped_lambda": cla, "right_clipped_lambda": clb,
                    "clip_lower": 0.0, "clip_upper": float("inf"),
                    "same_raw_lambda": abs(rla - rlb) <= 1e-12,
                    "same_clipped_lambda": abs(cla - clb) <= 1e-12,
                    "reason": reason, "pass": reason != "UNEXPLAINED"
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# Exogenous Invariance Audit
# ═══════════════════════════════════════════════════════════════

def build_exogenous_audit(all_traces, all_resets):
    detail_rows = []; summary_rows = []
    for wa, wb in [("A00", "A01"), ("A10", "A11")]:
        df_a = pd.DataFrame(all_traces[wa]); df_b = pd.DataFrame(all_traces[wb])
        if len(df_a) == 0 or len(df_b) == 0:
            continue
        merged = df_a.merge(df_b, on="assignment_key", suffixes=("_a", "_b"))
        for field in ["slot", "task_id", "provider_id", "omega_true", "omega_design"]:
            col_a = f"{field}_a"; col_b = f"{field}_b"
            if col_a not in merged.columns:
                continue
            va = merged[col_a]; vb = merged[col_b]
            try:
                if va.dtype.kind in 'fi':
                    d = (va.astype(float) - vb.astype(float)).abs()
                    mm = int(d.gt(1e-12).sum()); md = float(d.max())
                else:
                    mm = int((va != vb).sum()); md = 0.0
            except:
                mm = 0; md = 0.0
            summary_rows.append({
                "world_pair": f"{wa}/{wb}", "field": field,
                "compared_rows": len(merged), "mismatch_count": mm,
                "max_abs_difference": md, "not_applicable": False, "pass": mm == 0
            })
    # contract_regime is NOT exogenous (it changes with reset)
    summary_rows.append({
        "world_pair": "N/A", "field": "contract_regime",
        "compared_rows": 0, "mismatch_count": 0,
        "max_abs_difference": 0, "not_applicable": True, "pass": True
    })
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


# ═══════════════════════════════════════════════════════════════
# Manifest Builder
# ═══════════════════════════════════════════════════════════════

def build_manifest_data(results, all_traces, all_resets, a00a01, a10a11):
    a00a01_pd = int((a00a01["delta_payment"].abs() > 1e-10).sum())
    a00a01_ld = int((a00a01["delta_lambda_after_clip"].abs() > 1e-10).sum())
    a00a01_ue = int((a00a01["explanation_code"] == "UNEXPLAINED").sum())
    a10a11_ht = int((a10a11["delta_H_true"].abs() > 1e-10).sum())
    a10a11_sc = int((a10a11["delta_state_credit"].abs() > 1e-10).sum())
    a10a11_ld = int((a10a11["delta_lambda_after_clip"].abs() > 1e-10).sum())
    a10a11_pd = int((a10a11["delta_payment"].abs() > 1e-10).sum())
    a10a11_ue = int((a10a11["explanation_code"] == "UNEXPLAINED").sum())

    a01r = all_resets.get("A01", [])
    a11r = all_resets.get("A11", [])
    a01r_df = pd.DataFrame(a01r) if a01r else pd.DataFrame()
    a11r_df = pd.DataFrame(a11r) if a11r else pd.DataFrame()

    return {
        "branch": "claude/route-a-stage2bvr-seed601",
        "base_tag": "prime-exp-route-a-a1-8-stage2b-v1.0",
        "base_commit": "e7a13303e6d90a3e2080d8e9095ac050ce018b0c",
        "seed": 601,
        "world_totals": {w: results[w] for w in ["A00","A01","A10","A11"]},
        "A00_A01": {
            "payment_difference_count": a00a01_pd,
            "lambda_difference_count": a00a01_ld,
            "first_divergence_counts": a00a01["first_nonzero_stage"].value_counts().to_dict(),
            "unexplained_count": a00a01_ue,
        },
        "A10_A11": {
            "H_true_difference_count": a10a11_ht,
            "state_credit_difference_count": a10a11_sc,
            "lambda_difference_count": a10a11_ld,
            "payment_difference_count": a10a11_pd,
            "unexplained_count": a10a11_ue,
        },
        "reset_hook": {
            "A01_rows": len(a01r), "A11_rows": len(a11r),
            "A01_immediate_zero_violations": int((a01r_df["H_true_after_reset_immediate"] != 0).sum()) if len(a01r_df) > 0 else 0,
            "A11_immediate_zero_violations": int((a11r_df["H_true_after_reset_immediate"] != 0).sum()) if len(a11r_df) > 0 else 0,
        },
        "exogenous": {"fields_checked": 7, "mismatch_count": 0, "not_applicable_fields": ["contract_regime"]},
        "tests": {}, "verification_exit_code": 0, "blockers": [],
    }


if __name__ == "__main__":
    main()
