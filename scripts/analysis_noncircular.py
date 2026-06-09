#!/usr/bin/env python
"""Two NON-circular post-stroke readouts (PS46): structure beyond the lick count.

A. AMONG-SURVIVORS conditional transitions. Restrict transitions to the
   non-contra (surviving) states and renormalize each SOURCE row over surviving
   destinations only. Conditional routing P(next | current) among survivors is
   occupancy-independent — it is NOT mechanically forced by contra becoming rare
   — so a change here is genuine sequence reorganization. We compare pre→acute
   and pre→chronic (weighted mean total-variation over source rows) against a
   pre split-half distance = the sampling-noise floor. If pre→acute ≈ floor, the
   lesion is a SELECTIVE deletion of contra licking that leaves the surviving
   subnetwork intact (a clean result counts alone cannot establish).

B. LICK-EXCLUDED biomarker. The AR-LL manifold biomarker averaged over NON-lick
   frames only (vs lick-only and all-frames), per epoch. Tells whether the
   quality of the SURVIVING movement dynamics (whisk/jaw/rest) departs baseline
   independent of whether licks occur — something spout behavior cannot measure.

Usage: python scripts/analysis_noncircular.py [--animal PS46] [--tag B_33hz]
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
import yaml

from arhmm_behavior.paths import PathResolver
from arhmm_behavior.model import moseq
from arhmm_behavior.model.scoring import lick_side_labels, state_side_counts, transition_matrix


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


def _ar_loglik_perframe(seq, z, params, nlags):
    """Per-frame AR log-lik array (len = len(seq)-nlags), aligned to z and to
    lick_side_labels(tbl, nlags)[:len(z)]."""
    Ab = np.asarray(params["Ab"]); Q = np.asarray(params["Q"])
    Qinv = np.linalg.inv(Q); logdet = np.linalg.slogdet(Q)[1]
    D = seq.shape[1]
    const = -0.5 * (D * np.log(2 * np.pi) + logdet)
    out = np.empty(len(seq) - nlags)
    for t in range(nlags, len(seq)):
        k = int(z[t - nlags])
        lag = seq[t - nlags:t][::-1].reshape(-1)
        resid = seq[t] - Ab[k] @ np.concatenate([lag, [1.0]])
        out[t - nlags] = const[k] - 0.5 * resid @ Qinv[k] @ resid
    return out


def _survivor_T(zc, survivors, K):
    """Row-normalized transition matrix among `survivors` (self removed),
    each source row renormalized over surviving destinations only."""
    T = transition_matrix(zc, K, drop_self=True)
    sub = T[np.ix_(survivors, survivors)]
    rs = sub.sum(1, keepdims=True)
    return np.divide(sub, rs, out=np.zeros_like(sub), where=rs > 0)


def _row_tv(P, Q, w):
    """Weighted mean total-variation distance over source rows."""
    tv = 0.5 * np.abs(P - Q).sum(1)
    return float(np.average(tv, weights=w)) if w.sum() > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="PS46")
    ap.add_argument("--tag", default="B_33hz")
    args = ap.parse_args()

    r = PathResolver(); lw = r.local_work()
    blob = pickle.load(open(lw / f"fit_{args.tag}" / "best_model.pkl", "rb"))
    model, cfg = blob["model"], blob["config"]
    nlags = int(cfg["nlags"]); K = int(np.asarray(model["params"]["Ab"]).shape[0])
    design_dir = lw / f"design_{args.tag}"; feat_dir = lw / "features" / args.tag
    seqs = pd.read_csv(design_dir / "sequences.csv")
    animals = yaml.safe_load(open(r.animals_yaml()))

    # side identity (pooled pre-stroke lick cross-tab)
    tot = np.zeros((K, 2), int)
    for _, row in seqs[seqs["epoch"] == "pre"].head(20).iterrows():
        z = moseq.decode(moseq.make_data([np.load(design_dir / row["path"])], truncate=False), model)[0]
        side = lick_side_labels(pd.read_parquet(feat_dir / row["animal"] / f"{row['date']}.parquet"),
                                nlags=nlags)[:len(z)]
        tot += state_side_counts(z, side, K)
    licks = tot.sum(1); frac = np.divide(tot[:, 0], licks, out=np.full(K, np.nan), where=licks > 0)
    thr = 0.02 * licks.sum()
    contra = [k for k in range(K) if licks[k] >= thr and frac[k] <= 0.2]
    survivors = [k for k in range(K) if k not in contra]
    print(f"contra (excluded) = {contra}; survivors = {survivors}")

    # decode this animal, pool z + per-frame LL + lick mask by epoch
    a = args.animal; eb = animals[a]["epoch_boundaries"]
    z_by = {"pre": [], "acute": [], "chronic": []}
    ll_by = {e: {"all": [], "nolick": [], "lick": []} for e in z_by}
    for _, row in seqs[seqs["animal"] == a].iterrows():
        e = _epoch(int(row["rel_day"]), eb)
        if e not in z_by:
            continue
        seq = np.load(design_dir / row["path"])
        z = moseq.decode(moseq.make_data([seq], truncate=False), model)[0]
        tbl = pd.read_parquet(feat_dir / a / f"{row['date']}.parquet")
        side = lick_side_labels(tbl, nlags=nlags)[:len(z)]
        ll = _ar_loglik_perframe(seq, z, model["params"], nlags)[:len(side)]
        z_by[e].append(z)
        nolick = side < 0
        ll_by[e]["all"].append(ll)
        ll_by[e]["nolick"].append(ll[nolick])
        ll_by[e]["lick"].append(ll[~nolick])

    # ---------- A. among-survivors conditional-transition reorganization ----------
    pre_zs = z_by["pre"]
    pre_cat = np.concatenate(pre_zs)
    Tpre = _survivor_T(pre_cat, survivors, K)
    # source weights = visit counts of each survivor in pre (so rare states don't dominate)
    leaves_pre = np.array([(np.diff(pre_cat) != 0)[pre_cat[:-1] == i].sum() for i in survivors], float)
    # noise floor: pre split-half (even/odd sessions)
    h1 = np.concatenate(pre_zs[0::2]); h2 = np.concatenate(pre_zs[1::2])
    floor = _row_tv(_survivor_T(h1, survivors, K), _survivor_T(h2, survivors, K), leaves_pre)
    print("\n--- A. among-survivors conditional transitions (weighted mean TV vs pre) ---")
    print(f"  pre split-half (noise floor): {floor:.3f}")
    for e in ["acute", "chronic"]:
        Te = _survivor_T(np.concatenate(z_by[e]), survivors, K)
        d = _row_tv(Tpre, Te, leaves_pre)
        verdict = "REORGANIZED (> noise)" if d > 2 * floor else "≈ noise (selective deletion)"
        print(f"  pre -> {e:7s}: {d:.3f}   [{verdict}]")

    # ---------- B. lick-excluded biomarker ----------
    print("\n--- B. AR-LL by epoch (mean per frame), all / non-lick / lick-only ---")
    base = {kind: np.concatenate(ll_by["pre"][kind]).mean() for kind in ("all", "nolick", "lick")}
    print(f"  {'epoch':9s} {'all':>10s} {'non-lick':>10s} {'lick-only':>10s}   (Δ vs pre-mean)")
    for e in ["pre", "acute", "chronic"]:
        vals = {kind: np.concatenate(ll_by[e][kind]).mean() for kind in ("all", "nolick", "lick")}
        print(f"  {e:9s} " + " ".join(f"{vals[k]-base[k]:>+10.1f}" for k in ("all", "nolick", "lick")))
    print("\nReading: if NON-LICK Δ dips acutely too -> surviving (whisk/jaw/rest) movement "
          "dynamics distort, not just licking. If non-lick stays ~0 -> distortion is lick-driven.")


if __name__ == "__main__":
    main()
