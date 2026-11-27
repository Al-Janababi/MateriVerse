"""Group-aware, deterministic train/validation/test split (Section 3.2).

All members of a group (for example the same reduced formula and space group) are assigned
to the same subset, so that no structure family appears in more than one subset.
"""
from __future__ import annotations

import csv
import hashlib


def _bucket(group: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def group_split(ids, groups, fractions=(0.8, 0.1, 0.1), seed=0, names=("train", "validation", "test")) -> dict:
    if len(ids) != len(groups):
        raise ValueError("ids and groups must have equal length")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    edges, acc = [], 0.0
    for f in fractions:
        acc += f
        edges.append(acc)
    out = {}
    for i, g in zip(ids, groups):
        u = _bucket(str(g), seed)
        for name, edge in zip(names, edges):
            if u < edge:
                out[i] = name
                break
        else:
            out[i] = names[-1]
    return out


def split_csv(in_path, out_path, id_col="id", group_col="structure_group", fractions=(0.8, 0.1, 0.1), seed=0):
    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assign = group_split([r[id_col] for r in rows], [r[group_col] for r in rows], fractions, seed)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([id_col, group_col, "split"])
        for r in rows:
            w.writerow([r[id_col], r[group_col], assign[r[id_col]]])
    return assign
