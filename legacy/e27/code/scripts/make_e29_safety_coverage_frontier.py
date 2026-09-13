"""Create the E29 safety--coverage--payment frontier from frozen formal results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e20", type=Path, required=True)
    parser.add_argument("--e23", type=Path, required=True)
    parser.add_argument("--e26", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    e20 = pd.read_csv(args.e20)
    e20 = e20[(e20.scenario == "stable") & (e20.profile == "fixed_envelope")].copy()
    e20["coverage_loss_pp"] = -100 * e20["delta_service_coverage"]
    e20["series"] = "E20 fixed envelope"
    e20["label"] = e20.apply(lambda r: f"±{r.half_width:.2f}", axis=1)

    e23 = pd.read_csv(args.e23)
    e23 = e23[e23.scenario.isin(["down_step", "hidden_blocks"])].copy()
    e23["coverage_gain_pp"] = 100 * e23["delta_e20_service_coverage"]
    e23["qos_percent"] = 100 * e23["pooled_qos_violation_rate"]
    e23["series"] = e23.scenario.map({"down_step": "Downward step", "hidden_blocks": "Hidden blocks"})
    e23["label"] = e23.global_guardrail_weight.map(lambda g: f"g={g:g}")

    e26 = pd.read_csv(args.e26)
    e26 = e26[(e26.branch == "netease_completed_quality") & (e26.activity_stratum == "all")].copy()
    e26["coverage_gain_pp"] = 100 * e26["delta_reference_service_coverage"]
    e26["qos_percent"] = 100 * e26["pooled_qos_violation_rate"]
    e26["label"] = e26.profile.map({
        "frozen_point": "Frozen", "guardrail_075": "g=0.75", "fixed_envelope": "Fixed"
    })

    export_rows = []
    for _, r in e20.iterrows():
        export_rows.append({"experiment": "E20", "series": r.series, "policy": r.label,
                            "coverage_axis_pp": r.coverage_loss_pp,
                            "qos_violation_percent": 100 * r.pooled_qos_violation_rate if "pooled_qos_violation_rate" in r else 0.0,
                            "pps": r.mean_pps, "deterministic_guarantee": True})
    for _, r in e23.iterrows():
        export_rows.append({"experiment": "E23", "series": r.series, "policy": r.label,
                            "coverage_axis_pp": r.coverage_gain_pp,
                            "qos_violation_percent": r.qos_percent,
                            "pps": r.mean_pps, "deterministic_guarantee": r.global_guardrail_weight == 1.0})
    for _, r in e26.iterrows():
        export_rows.append({"experiment": "E26", "series": "NetEase completed-quality replay", "policy": r.label,
                            "coverage_axis_pp": r.coverage_gain_pp,
                            "qos_violation_percent": r.qos_percent,
                            "pps": r.mean_pps, "deterministic_guarantee": r.profile == "fixed_envelope"})
    pd.DataFrame(export_rows).to_csv(args.out / "e29_frontier_points.csv", index=False)

    mpl.rcParams.update({
        "font.family": "serif", "font.size": 7.0, "axes.labelsize": 7.0,
        "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.1, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.25), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=0.25, vmax=0.62)

    ax = axes[0]
    ax.plot(e20.coverage_loss_pp, e20.delta_pps, color="#3a6ea5", lw=1.2, zorder=1)
    ax.scatter(e20.coverage_loss_pp, e20.delta_pps, c=e20.mean_pps, cmap=cmap, norm=norm,
               s=30, edgecolor="black", linewidth=0.35, zorder=2)
    for _, r in e20.iterrows():
        ax.annotate(r.label, (r.coverage_loss_pp, r.delta_pps), xytext=(2, 2),
                    textcoords="offset points", fontsize=6.1)
    ax.set_xlabel("Stable coverage loss (p.p.)")
    ax.set_ylabel("PPS increase")
    ax.set_title("(a) E20: deterministic envelopes")
    ax.grid(alpha=0.22, lw=0.4)

    ax = axes[1]
    styles = {"Downward step": ("o", "#cc4c02"), "Hidden blocks": ("s", "#756bb1")}
    for name, group in e23.groupby("series", sort=False):
        group = group.sort_values("global_guardrail_weight")
        marker, color = styles[name]
        ax.plot(group.coverage_gain_pp, group.qos_percent, color=color, lw=1.0, label=name)
        ax.scatter(group.coverage_gain_pp, group.qos_percent, c=group.mean_pps, cmap=cmap,
                   norm=norm, marker=marker, s=27, edgecolor="black", linewidth=0.3, zorder=2)
        for _, r in group.iterrows():
            if r.global_guardrail_weight in (0.0, 0.75, 1.0):
                dx, dy = (2, 2)
                if r.global_guardrail_weight == 0.75 and name == "Hidden blocks":
                    dx, dy = (2, 5)
                elif r.global_guardrail_weight == 0.75:
                    dx, dy = (2, -8)
                elif r.global_guardrail_weight == 1.0:
                    dx, dy = (-1, -9)
                ax.annotate(f"{r.global_guardrail_weight:g}", (r.coverage_gain_pp, r.qos_percent),
                            xytext=(dx, dy), textcoords="offset points", fontsize=5.8)
    ax.set_xlabel("Coverage gain over fixed (p.p.)")
    ax.set_ylabel("Observed QoS violation (%)")
    ax.set_title("(b) E23: past-only guardrails")
    ax.set_yscale("symlog", linthresh=0.005, linscale=0.6)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.22, lw=0.4)

    ax = axes[2]
    e26["policy_order"] = e26.profile.map({"frozen_point": 0, "guardrail_075": 1, "fixed_envelope": 2})
    e26 = e26.sort_values("policy_order")
    ax.plot(e26.coverage_gain_pp, e26.qos_percent, color="#238b45", lw=1.0, zorder=1)
    ax.scatter(e26.coverage_gain_pp, e26.qos_percent, c=e26.mean_pps, cmap=cmap, norm=norm,
               s=31, edgecolor="black", linewidth=0.35, zorder=2)
    offsets = {"Frozen": (2, -9), "g=0.75": (2, 3), "Fixed": (2, 3)}
    for _, r in e26.iterrows():
        ax.annotate(r.label, (r.coverage_gain_pp, r.qos_percent), xytext=offsets[r.label],
                    textcoords="offset points", fontsize=6.1)
    ax.set_xlabel("Coverage gain over fixed (p.p.)")
    ax.set_ylabel("Observed QoS violation (%)")
    ax.set_title("(c) E26: public completion replay")
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.6)
    ax.grid(alpha=0.22, lw=0.4)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, location="right", fraction=0.025, pad=0.015)
    cbar.set_label("PPS")
    output = args.out / "Fig8_Safety_Coverage_Payment_Frontier.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    (args.out / "e29_frontier_metadata.json").write_text(json.dumps({
        "artifact": "E29 safety-coverage-payment frontier",
        "inputs": [str(args.e20), str(args.e23), str(args.e26)],
        "semantics": {
            "E20": "deterministic guarantee within declared envelope; stable response used only to measure conservatism",
            "E23": "empirical risk points for g<1; g=1 is the fixed-envelope guarantee",
            "E26": "public completed-quality replay; not causal payment response"
        },
        "selection": "all pre-specified E20 widths, all E23 weights for the two nonzero-violation trajectories, and all aggregate NetEase E26 policies"
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
