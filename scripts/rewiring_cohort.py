#!/usr/bin/env python
"""Does the PS46 rewiring pattern replicate across PS46-50? (Model B)

State labels are global (one model), so the PS46 phenomena are testable per animal.
For each animal with acute data we compute pre->acute changes in 4 interpretable,
activity-robust-ish metrics of the among-survivors (non-contra) sequence:

  - Δ entropy           : mean per-source transition entropy (≈0 = not disorganized)
  - Δ same-cat routing  : fraction of transitions whose source & destination share a
                          behavior category (↑ = tighter like-with-like chaining / perseveration)
  - Δ inflow to S12      : share of transitions entering the 'near-lick posture' state
                          (↓ in PS46 = avoidance)
  - Δ inflow to ipsi-lick: share of transitions entering ipsi-lick states (↑ = compensatory)

Prints a per-animal table and writes figures/cohort_rewiring_metrics.png.
Usage: python scripts/rewiring_cohort.py [--tag B_33hz]
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from arhmm_behavior.paths import PathResolver
from arhmm_behavior.model import moseq
from arhmm_behavior.model.scoring import lick_side_labels, state_side_counts

_FEATS = ["tongue_out_frac", "tongue_angle_mean", "jaw_motion_energy",
          "treadmill_speed_mm_s", "tongue_motion_energy"]
_TO_3TIER = {"moderate-severe": "moderate", "mod-severe": "moderate"}
_SEV_COLOR = {"mild": "#4575b4", "moderate": "#f46d43", "severe": "#d73027"}
POSTURE = 12  # S12 = tongue-engaged near-lick posture (PS46)


def _epoch(rel_day, eb):
    if rel_day <= 0:
        return "pre"
    if rel_day <= eb["acute_max_rel_day"]:
        return "acute"
    if rel_day <= eb["subacute_max_rel_day"]:
        return "sub_acute"
    if rel_day >= eb["chronic_min_rel_day"]:
        return "chronic"
    return None


def _decode(design_dir, model, path):
    return moseq.decode(moseq.make_data([np.load(design_dir / path)], truncate=False), model)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="B_33hz")
    args = ap.parse_args()
    r = PathResolver(); lw = r.local_work()
    blob = pickle.load(open(lw / f"fit_{args.tag}" / "best_model.pkl", "rb"))
    model = blob["model"]; nlags = int(blob["config"]["nlags"])
    K = int(np.asarray(model["params"]["Ab"]).shape[0])
    design_dir = lw / f"design_{args.tag}"; feat_dir = lw / "features" / args.tag
    seqs = pd.read_csv(design_dir / "sequences.csv")
    meta = yaml.safe_load(open(r.animals_yaml()))

    # global side identity + global feature profile (pooled pre-stroke, all animals)
    tot = np.zeros((K, 2), int)
    fsum = np.zeros((K, len(_FEATS))); fcnt = np.zeros(K)
    for _, row in seqs[seqs["epoch"] == "pre"].iterrows():
        z = _decode(design_dir, model, row["path"])
        tbl = pd.read_parquet(feat_dir / row["animal"] / f"{row['date']}.parquet")
        side = lick_side_labels(tbl, nlags=nlags)[:len(z)]
        tot += state_side_counts(z, side, K)
        F = tbl[_FEATS].to_numpy()[nlags:nlags + len(z)].copy(); F[:, 1] = np.abs(F[:, 1])
        for k in range(K):
            m = z == k; fsum[k] += np.nan_to_num(F[m]).sum(0); fcnt[k] += m.sum()
    licks = tot.sum(1); sf = np.divide(tot[:, 0], licks, out=np.full(K, np.nan), where=licks > 0)
    thr = 0.02 * licks.sum()
    contra = [k for k in range(K) if licks[k] >= thr and sf[k] <= 0.2]
    ipsi = [k for k in range(K) if licks[k] >= thr and sf[k] >= 0.8]
    survivors = [k for k in range(K) if k not in contra]
    prof = fsum / np.maximum(fcnt[:, None], 1); zsc = (prof - prof.mean(0)) / (prof.std(0) + 1e-9)

    def label(k):
        if k in contra: return "contra-lick"
        if k in ipsi: return "ipsi-lick"
        TO, ANG, JAW, RUN, TM = zsc[k]
        if RUN > 1.0: return "run"
        if JAW > 0.7: return "jaw"
        if TO > 0.7: return "tongue"
        if TM > 0.7: return "face"
        return "quiescent"
    lab = {k: label(k) for k in range(K)}
    si = {s: i for i, s in enumerate(survivors)}
    cat = np.array([lab[s] for s in survivors])
    same = cat[:, None] == cat[None, :]
    pidx = si[POSTURE]; iidx = [si[k] for k in ipsi]

    def counts(zc):
        T = np.zeros((K, K)); np.add.at(T, (zc[:-1], zc[1:]), 1); np.fill_diagonal(T, 0)
        return T[np.ix_(survivors, survivors)]

    def metrics(zc):
        C = counts(zc); tot_ = C.sum()
        rs = C.sum(1, keepdims=True); P = np.divide(C, rs, out=np.zeros_like(C), where=rs > 0)
        H = [-(p[p > 0] * np.log(p[p > 0])).sum() for p in P if p.sum() > 0]
        return {"entropy": float(np.mean(H)),
                "same_cat": float(C[same].sum() / tot_),
                "inflow_posture": float(C[:, pidx].sum() / tot_),
                "inflow_ipsi": float(C[:, iidx].sum() / tot_)}

    rows = []
    for a in sorted(seqs["animal"].unique()):
        eb = meta[a]["epoch_boundaries"]
        sev = _TO_3TIER.get(meta[a].get("recovery_phenotype"), meta[a].get("recovery_phenotype"))
        zb = {"pre": [], "acute": []}
        for _, row in seqs[seqs["animal"] == a].iterrows():
            e = _epoch(int(row["rel_day"]), eb)
            if e in zb:
                zb[e].append(_decode(design_dir, model, row["path"]))
        if not zb["pre"] or not zb["acute"]:
            continue
        mp, ma = metrics(np.concatenate(zb["pre"])), metrics(np.concatenate(zb["acute"]))
        rows.append({"animal": a, "severity": sev,
                     **{f"d_{k}": round(ma[k] - mp[k], 3) for k in mp}})
    df = pd.DataFrame(rows)
    print(f"contra={contra} ipsi={ipsi} posture=S{POSTURE} ({lab[POSTURE]})\n")
    print("=== pre->acute change per animal (among-survivors sequence) ===")
    print(df.to_string(index=False))

    mets = [("d_entropy", "Δ entropy\n(expect ≈0)"),
            ("d_same_cat", "Δ same-category routing\n(expect ↑ perseveration)"),
            ("d_inflow_posture", "Δ inflow to S12 posture\n(expect ↓ avoidance)"),
            ("d_inflow_ipsi", "Δ inflow to ipsi-lick\n(expect ↑ compensatory)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (col, title) in zip(axes, mets):
        ax.bar(df["animal"], df[col], color=[_SEV_COLOR.get(s, "#333") for s in df["severity"]])
        ax.axhline(0, color="k", lw=0.8); ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("pre → acute change")
    fig.suptitle("PS46–50: does the PS46 rewiring pattern replicate? "
                 "(orange=moderate, blue=mild)", y=1.04)
    fig.tight_layout()
    out = r.outputs() / "cohort_rewiring_metrics.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
