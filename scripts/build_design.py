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

# Lateralized-lick axis: up-weight so PCA keeps it (at unit weight licking is ~2%
# of frames, so PCA discards tongue_x_mean ~95% and the model merges ipsi/contra
# licks into one syllable — see DECISIONS 2026-06-09).
SIDE_FEATURES = ("tongue_x_mean", "tongue_angle_mean", "fr_c2", "fr_c3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["A", "B"], required=True)
    ap.add_argument("--pca-var", type=float, default=0.95)
    ap.add_argument("--side-weight", type=float, default=2.0,
                    help="multiplier on the lateralized-lick features "
                         f"({', '.join(SIDE_FEATURES)}) after standardization, so PCA "
                         "keeps the side axis and the model can carve ipsi/contra licks. "
                         "1.0 = off (the old behavior that merged sides).")
    ap.add_argument("--rate", type=float, default=None,
                    help="model-grid rate (Hz); default from configs/defaults.yaml. "
                         "Must match the rate the features were assembled at.")
    ap.add_argument("--side-fit", choices=["off", "present", "present-cov"], default="off",
                    help="estimate the PCA on tongue-PRESENT bins so the sparse side "
                         "axis (NaN ~86%% of bins, imputed to 0) is not pruned. "
                         "'present' = standardize + PCA on present bins; "
                         "'present-cov' = standardize on all bins, PCA directions from "
                         "present-bin covariance; 'off' = original (all bins). Sessions "
                         "with no licks (severe stroke) are simply all-absent and excluded "
                         "from the fit; they still project with tongue imputed neutral.")
    args = ap.parse_args()

    resolver = PathResolver()
    rate = args.rate if args.rate else _rate(resolver)
    tag = f"{args.family}_{int(round(rate))}hz"
    feat_dir = resolver.local_work() / "features" / tag
    manifest = pd.read_csv(feat_dir / "manifest.csv")

    # Columns present in this family's tables, in CONTINUOUS order.
    sample = pd.read_parquet(feat_dir / manifest.iloc[0]["animal"] / f"{manifest.iloc[0]['date']}.parquet")
    cols = [c for c in CONTINUOUS if c in sample.columns]
    print(f"family {args.family}: {len(cols)} continuous columns -> {cols}")

    def _load(row) -> pd.DataFrame:
        return pd.read_parquet(feat_dir / row["animal"] / f"{row['date']}.parquet")

    weights = np.array([args.side_weight if c in SIDE_FEATURES else 1.0 for c in cols])
    print(f"side-feature weight = {args.side_weight} on "
          f"{[c for c in cols if c in SIDE_FEATURES]}")
    pre = manifest[manifest["epoch"] == "pre"]
    pre_feats = [_load(r)[cols].to_numpy(np.float32) for _, r in pre.iterrows()]

    # Tongue-present mask per session (a bin is present when tongue_x_mean is not
    # NaN; NaN == no tongue-out frame in that bin). Drives --side-fit so the
    # sparse side axis keeps its variance through PCA. No-lick sessions are
    # all-False and drop out of the fit (handled in build_design).
    fit_mask = standardize_on_mask = None
    if args.side_fit != "off":
        xi = cols.index("tongue_x_mean")
        fit_mask = [~np.isnan(f[:, xi]) for f in pre_feats]
        standardize_on_mask = (args.side_fit == "present")
        present_frac = np.concatenate(fit_mask).mean()
        n_empty = sum(int(not mk.any()) for mk in fit_mask)
        print(f"side-fit={args.side_fit}: fitting on tongue-present bins "
              f"({100*present_frac:.1f}% of bins); {n_empty}/{len(fit_mask)} sessions fully tongue-absent")

    design = build_design(pre_feats, pca_var=args.pca_var, columns=cols, weights=weights,
                          fit_mask=fit_mask, standardize_on_mask=standardize_on_mask)
    print(f"PCA: {len(cols)} features -> {design.components.shape[0]} comps "
          f"({100*design.var_ratio.sum():.0f}% var), fit on {len(pre_feats)} pre-stroke sessions")
    ret = (design.components ** 2).sum(0)
    print("retained-in-PCA for side features: "
          + ", ".join(f"{c}={ret[cols.index(c)]:.2f}" for c in cols if c in SIDE_FEATURES))

    out_dir = resolver.local_work() / f"design_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(resolver.local_work() / f"design_{tag}.npz",
             mu=design.mu, sd=design.sd, weights=design.weights,
             components=design.components, z_mean=design.z_mean,
             var_ratio=design.var_ratio, columns=np.array(design.columns), rate_hz=rate)

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
