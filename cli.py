"""Command-line interface: summarize | make-synthetic | split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import ablation
from .figures import plot_summary
from .split import split_csv
from .synthetic import make_synthetic_logs


def _summarize(args) -> int:
    cfg = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    logs = ablation.discover_logs(args.logs)
    synthetic = any(l.get("_synthetic", False) for l in logs)
    if synthetic and not args.allow_synthetic:
        print("ERROR: synthetic test logs found. Outputs from synthetic data must never be reported as results. "
              "Re-run with --allow-synthetic only for pipeline testing.", file=sys.stderr)
        return 2
    prefix = "SYNTHETIC_" if synthetic else ""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thr = float(cfg.get("stability_threshold_mev_per_atom", 50.0))
    tol = float(cfg.get("tc_within_k", 10.0))
    ref = cfg.get("reference_variant", "full")
    per_run = ablation.per_run_table(logs, thr, tol)
    summary = ablation.summary_table(per_run)
    comp, warns = ablation.comparison_table(per_run, ref)
    per_run.to_csv(out / f"{prefix}per_run.csv", index=False)
    summary.to_csv(out / f"{prefix}summary.csv", index=False)
    comp.to_csv(out / f"{prefix}comparisons.csv", index=False)
    (out / f"{prefix}ablation_table.tex").write_text(
        ablation.latex_table(summary, "Ablation study: mean $\\pm$ SD over seeds.", "tab:ablation", synthetic), encoding="utf-8")
    plot_summary(summary, out / f"{prefix}ablation_figure.png", highlight=ref, synthetic=synthetic)
    report = ["# Ablation report", ""]
    if synthetic:
        report += ["**SYNTHETIC TEST DATA - NOT RESULTS.**", ""]
    report += [f"Reference variant: `{ref}`; variants: {', '.join(summary['variant'])}", ""]
    report += ["## Warnings", *([f"- {w}" for w in warns] or ["- none"]), ""]
    (out / f"{prefix}report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote results to {out}")
    for w in warns:
        print("WARNING:", w, file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="materiverse-eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize", help="aggregate run logs into ablation tables")
    s.add_argument("--logs", required=True)
    s.add_argument("--out", default="results")
    s.add_argument("--config", default="configs/ablation.yaml")
    s.add_argument("--allow-synthetic", action="store_true")
    m = sub.add_parser("make-synthetic", help="write synthetic logs for testing the pipeline")
    m.add_argument("--out", default="runs_synthetic")
    sp = sub.add_parser("split", help="group-aware train/validation/test split of a CSV")
    sp.add_argument("--input", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.cmd == "summarize":
        return _summarize(a)
    if a.cmd == "make-synthetic":
        paths = make_synthetic_logs(a.out)
        print(f"wrote {len(paths)} SYNTHETIC logs to {a.out}")
        return 0
    if a.cmd == "split":
        assign = split_csv(a.input, a.out, seed=a.seed)
        print(f"assigned {len(assign)} rows")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
