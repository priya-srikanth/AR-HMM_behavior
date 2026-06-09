#!/usr/bin/env python
"""Fit the pooled pre-stroke design (standardize + PCA) for a model family.

Reads the cached per-session feature tables (``scripts/assemble_features.py``),
fits standardization + PCA on the **pooled pre-stroke** sessions only, and
projects EVERY session (pre + post) through that frozen transform — so
post-stroke behavior is read out in the pre-stroke reference frame. Model A and
Model B differ only in which columns are present (Model A has no ``fr_c*``); the
present subset of :data:`CONTINUOUS` is selected automatically.

Outputs (to ``data_local/``):
  - ``design_<family>.npz``         : the frozen transform (mu, sd, components,
                                      z_mean, columns, rate_hz, var_ratio).
  - ``design_<family>/<animal>_<date>.npy`` : per-session PCA sequence (T, n_pca).
  - ``design_<family>/sequences.csv``       : manifest (animal, date, rel_day,
                                              epoch, T, path).

Usage:
    python scripts/build_design.py --family B
    python scripts/build_design.py --family A
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import pandas as pd

from arhmm_behavior.paths import PathResolver
from arhmm_behavior.model.design import CONTINUOUS, build_design


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["A", "B"], required=True)
    ap.add_argument("--pca-var", type=float, default=0.90)
    args = ap.parse_args()

    resolver = PathResolver()
    feat_dir = resolver.local_work() / "features" / args.family
    manifest = pd.read_csv(feat_dir / "manifest.csv")

    # Columns present in this family's tables, in CONTINUOUS order.
    sample = pd.read_parquet(feat_dir / manifest.iloc[0]["animal"] / f"{manifest.iloc[0]['date']}.parquet")
    cols = [c for c in CONTINUOUS if c in sample.columns]
    print(f"family {args.family}: {len(cols)} continuous columns -> {cols}")

    def _load(row) -> pd.DataFrame:
        return pd.read_parquet(feat_dir / row["animal"] / f"{row['date']}.parquet")

    pre = manifest[manifest["epoch"] == "pre"]
    pre_feats = [_load(r)[cols].to_numpy(np.float32) for _, r in pre.iterrows()]
    design = build_design(pre_feats, pca_var=args.pca_var, columns=cols)
    print(f"PCA: {len(cols)} features -> {design.components.shape[0]} comps "
          f"({100*design.var_ratio.sum():.0f}% var), fit on {len(pre_feats)} pre-stroke sessions")

    out_dir = resolver.local_work() / f"design_{args.family}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(resolver.local_work() / f"design_{args.family}.npz",
             mu=design.mu, sd=design.sd, components=design.components,
             z_mean=design.z_mean, var_ratio=design.var_ratio,
             columns=np.array(design.columns), rate_hz=_rate(resolver))

    rows = []
    for _, r in manifest.iterrows():
        seq = design.transform(_load(r)[cols].to_numpy(np.float32))
        path = out_dir / f"{r['animal']}_{r['date']}.npy"
        np.save(path, seq.astype(np.float32))
        rows.append({"animal": r["animal"], "date": r["date"], "rel_day": r["rel_day"],
                     "epoch": r["epoch"], "T": seq.shape[0], "path": path.name})
    with open(out_dir / "sequences.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["animal", "date", "rel_day", "epoch", "T", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"projected {len(rows)} sessions -> {out_dir}")


def _rate(resolver: PathResolver) -> float:
    import yaml
    with open(resolver.config_path.parent / "defaults.yaml") as f:
        return float(yaml.safe_load(f)["model"]["sampling_rate_hz"])


if __name__ == "__main__":
    main()
