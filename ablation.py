"""Aggregate run logs into ablation tables with paired statistics."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .metrics import METRIC_COLUMNS, load_log, run_metrics
from .stats import holm, min_wilcoxon_p, paired_comparison, summarize

# For these metrics a LOWER value is better; used only for the direction column in the report.
LOWER_IS_BETTER = {"dft_cost_per_candidate", "effective_cost_per_stable", "tc_mae_k"}


def discover_logs(directory: str | Path) -> list[dict]:
    logs = []
    for p in sorted(Path(directory).rglob("*.json")):
        if p.name.endswith(".schema.json"):
            continue
        try:
            logs.append(load_log(p))
        except (ValueError, KeyError):
            continue
    if not logs:
        raise FileNotFoundError(f"no valid run logs found under {directory}")
    return logs


def per_run_table(logs, stability_threshold=50.0, tc_within_k=10.0) -> pd.DataFrame:
    rows = [run_metrics(l, stability_threshold, tc_within_k) for l in logs]
    df = pd.DataFrame(rows)
    dup = df.duplicated(subset=["variant", "seed"], keep=False)
    if dup.any():
        bad = df.loc[dup, ["variant", "seed"]].drop_duplicates().values.tolist()
        raise ValueError(f"duplicate (variant, seed) logs: {bad}")
    return df.sort_values(["variant", "seed"]).reset_index(drop=True)


def summary_table(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in per_run.groupby("variant", sort=False):
        row = {"variant": variant, "n_seeds": int(len(g))}
        for m in METRIC_COLUMNS:
            s = summarize(g[m].tolist())
            row[f"{m}_mean"], row[f"{m}_sd"] = s["mean"], s["sd"]
            row[f"{m}_ci95_low"], row[f"{m}_ci95_high"] = s["ci95_low"], s["ci95_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(per_run: pd.DataFrame, reference: str) -> tuple[pd.DataFrame, list[str]]:
    """Paired comparison of every variant against `reference`, matched by seed; Holm-adjusted per metric."""
    warnings: list[str] = []
    if reference not in set(per_run["variant"]):
        raise ValueError(f"reference variant '{reference}' not found in logs")
    ref = per_run[per_run["variant"] == reference].set_index("seed")
    rows = []
    for variant in per_run["variant"].unique():
        if variant == reference:
            continue
        oth = per_run[per_run["variant"] == variant].set_index("seed")
        seeds = sorted(set(ref.index) & set(oth.index))
        if set(ref.index) != set(oth.index):
            warnings.append(f"{variant}: seeds differ from reference ({sorted(set(ref.index) ^ set(oth.index))}); "
                            f"using the {len(seeds)} shared seeds")
        if len(seeds) < 6:
            warnings.append(f"{variant}: only {len(seeds)} paired seeds; the smallest attainable two-sided exact "
                            f"Wilcoxon p is {min_wilcoxon_p(len(seeds)):.4g}")
        for m in METRIC_COLUMNS:
            a = ref.loc[seeds, m].astype(float)
            b = oth.loc[seeds, m].astype(float)
            ok = ~(a.isna() | b.isna())
            res = paired_comparison(a[ok].values, b[ok].values)
            res.update({"variant": variant, "metric": m, "reference": reference,
                        "better_when": "lower" if m in LOWER_IS_BETTER else "higher"})
            rows.append(res)
    df = pd.DataFrame(rows)
    if len(df):
        df["t_p_holm"] = float("nan")
        for m, idx in df.groupby("metric").groups.items():
            ps = df.loc[idx, "t_p"].fillna(1.0).tolist()
            df.loc[idx, "t_p_holm"] = holm(ps)
    return df, warnings


def _fmt(mean, sd, digits=2):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


TEX_COLUMNS = [("throughput_per_h", "Throughput (cand./h)", 1), ("stability_rate", "Stability rate", 3),
               ("space_group_retention", "Space-group retention", 3), ("dft_cost_per_candidate", "DFT cost (CPU-h/cand.)", 2),
               ("effective_cost_per_stable", "Eff. cost (CPU-h/stable)", 2), ("tc_mae_k", "$T_c$ MAE (K)", 2)]


def latex_table(summary: pd.DataFrame, caption: str, label: str, synthetic: bool) -> str:
    head = " & ".join(["Variant"] + [c[1] for c in TEX_COLUMNS]) + " \\\\"
    lines = ["\\begin{table}[ht]", "\\centering", "\\small",
             "\\begin{tabular}{l" + "c" * len(TEX_COLUMNS) + "}", "\\hline", head, "\\hline"]
    for _, r in summary.iterrows():
        cells = [r["variant"].replace("_", "\\_")]
        for key, _, d in TEX_COLUMNS:
            cells.append(_fmt(r[f"{key}_mean"], r[f"{key}_sd"], d))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    cap = ("SYNTHETIC TEST DATA - NOT RESULTS. " if synthetic else "") + caption
    lines += [f"\\caption{{{cap}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines)
