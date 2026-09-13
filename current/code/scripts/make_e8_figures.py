"""Create publication-neutral E8 robustness figures from frozen summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT.parent / "results" / "e8"
FIGS = RESULTS / "figures"


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RESULTS / "e8_formal_summary.csv")
    factors = ["state_noise", "state_delay", "state_missing", "response_error", "cost_error"]
    labels = ["State noise (std)", "State delay (slots)", "State missing rate",
              "Response-model bias", "Cost-model bias"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)
    for ax, factor, label in zip(axes.flat, factors, labels):
        sub = data[data["factor"] == factor].sort_values("level")
        ax.plot(sub["level"], 100 * sub["mean_target_miss_rate"], "o-", label="Target miss")
        ax.plot(sub["level"], 100 * sub["mean_qos_violation_rate"], "s-", label="QoS violation")
        ax.axhline(0.5, color="black", lw=0.8, ls="--", label="0.5% gate")
        ax.set_xlabel(label)
        ax.set_ylabel("Rate (%)")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    joint = pd.read_csv(RESULTS / "e8_joint_summary.csv").set_index("variant")
    metrics = ["mean_qos_violation_rate", "mean_target_miss_rate", "mean_under_incentive_rate"]
    vals = [100 * joint.loc["joint_worst", m] for m in metrics]
    axes.flat[-1].bar(["QoS", "Target", "Under-\nincentive"], vals, color=["#d95f02", "#7570b3", "#1b9e77"])
    axes.flat[-1].set_ylabel("Joint worst-case rate (%)")
    axes.flat[-1].grid(axis="y", alpha=0.25)
    fig.savefig(FIGS / "e8_safety_degradation.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for factor, label in zip(factors, labels):
        sub = data[data["factor"] == factor].sort_values("level")
        ax.plot(sub["level"], sub["delta_pps"], marker="o", label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Perturbation level (factor-specific scale)")
    ax.set_ylabel("Paired change in payment per service")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(FIGS / "e8_payment_sensitivity.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
