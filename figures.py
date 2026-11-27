"""Benchmark/ablation bar charts generated from the summary table."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [("throughput_per_h", "Throughput (candidates/h)", True),
          ("stability_rate", "Stability rate", False),
          ("effective_cost_per_stable", "Effective DFT cost (CPU-h per stable candidate)", True)]


def plot_summary(summary, out_path, highlight="full", synthetic=False):
    fig, axs = plt.subplots(1, len(PANELS), figsize=(10, 3.6))
    names = summary["variant"].tolist()
    colors = ["#0F6E56" if n == highlight else "#B4B2A9" for n in names]
    for ax, (key, title, log) in zip(axs, PANELS):
        means = summary[f"{key}_mean"].tolist()
        sds = summary[f"{key}_sd"].fillna(0).tolist()
        ax.bar(range(len(names)), means, yerr=sds, color=colors, edgecolor="#2C2C2A", linewidth=0.6, capsize=3)
        if log:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC TEST DATA - NOT RESULTS", ha="center", va="center", fontsize=26,
                 color="red", alpha=0.25, rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path
