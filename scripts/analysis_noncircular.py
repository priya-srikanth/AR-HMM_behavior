#!/usr/bin/env python
"""Two NON-circular post-stroke readouts, replicated across the cohort (PS46-50).

A. AMONG-SURVIVORS conditional transitions. Restrict transitions to the
   non-contra (surviving) states and renormalize each SOURCE row over surviving
   destinations only. Conditional routing P(next | current) among survivors is
   occupancy-independent — NOT mechanically forced by contra becoming rare — so a
   change is genuine sequence reorganization. Significance: a BOOTSTRAP null
   (resample pre sessions with replacement into two independent draws, many
   times) gives the pre-vs-pre distance distribution; pre->acute / pre->chronic
   are reported as z-scores and p (fraction of null >= observed). If pre->acute
   ~ null, the lesion is a SELECTIVE deletion of contra licking that leaves the
   surviving subnetwork intact.

B. LICK-EXCLUDED biomarker. AR-LL manifold biomarker averaged over NON-lick
   frames only (vs lick-only / all), per epoch — does the quality of the
   surviving (whisk/jaw/rest) movement dynamics depart baseline independent of
   whether licks occur?

Usage: python scripts/analysis_noncircular.py [--tag B_33hz] [--n-boot 300]
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

_TO_3TIER = {"moderate-severe": "moderate", "mod-severe": "moderate"}


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
    T = transition_matrix(zc, K, drop_self=True)
    sub = T[np.ix_(survivors, survivors)]
    rs = sub.sum(1, keepdims=True)
    return np.divide(sub, rs, out=np.zeros_like(sub), where=rs > 0)


def _row_tv(P, Q, w):
    tv = 0.5 * np.abs(P - Q).sum(1)
    return float(np.average(tv, weights=w)) if w.sum() > 0 else np.nan


def _boot_null(pre_zs, survivors, K, w, rng, n):
    """pre-vs-pre distance null: two independent bootstrap resamples of pre sessions."""
    nS = len(pre_zs)
    out = []
    for _ in range(n):
        a = rng.integers(0, nS, nS); b = rng.integers(0, nS, nS)
        Ta = _survivor_T(np.concatenate([pre_zs[i] for i in a]), survivors, K)
        Tb = _survivor_T(np.concatenate([pre_zs[i] for i in b]), survivors, K)
        out.append(_row_tv(Ta, Tb, w))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="B_33hz")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    r = PathResolver(); lw = r.local_work()
    blob = pickle.load(open(lw / f"fit_{args.tag}" / "best_model.pkl", "rb"))
    model, cfg = blob["model"], blob["config"]
    nlags = int(cfg["nlags"]); K = int(np.asarray(model["params"]["Ab"]).shape[0])
    design_dir = lw / f"design_{args.tag}"; feat_dir = lw / "features" / args.tag
    seqs = pd.read_csv(design_dir / "sequences.csv")
    animals_meta = yaml.safe_load(open(r.animals_yaml()))

    # global side identity (pooled pre-stroke lick cross-tab)
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
    print(f"contra(excluded)={contra}  survivors={survivors}  n_boot={args.n_boot}\n")

    rows = []
    for a in sorted(seqs["animal"].unique()):
        eb = animals_meta[a]["epoch_boundaries"]
        sev = _TO_3TIER.get(animals_meta[a].get("recovery_phenotype"), animals_meta[a].get("recovery_phenotype"))
        z_by = {"pre": [], "acute": [], "chronic": []}
        ll_by = {e: {"all": [], "nolick": []} for e in z_by}
        for _, row in seqs[seqs["animal"] == a].iterrows():
            e = _epoch(int(row["rel_day"]), eb)
            if e not in z_by:
                continue
            seq = np.load(design_dir / row["path"])
            z = moseq.decode(moseq.make_data([seq], truncate=False), model)[0]
            side = lick_side_labels(pd.read_parquet(feat_dir / a / f"{row['date']}.parquet"),
                                    nlags=nlags)[:len(z)]
            ll = _ar_loglik_perframe(seq, z, model["params"], nlags)[:len(side)]
            z_by[e].append(z)
            ll_by[e]["all"].append(ll); ll_by[e]["nolick"].append(ll[side < 0])
        if len(z_by["pre"]) < 2:
            continue
        pre_cat = np.concatenate(z_by["pre"])
        Tpre = _survivor_T(pre_cat, survivors, K)
        w = np.array([(np.diff(pre_cat) != 0)[pre_cat[:-1] == i].sum() for i in survivors], float)
        null = _boot_null(z_by["pre"], survivors, K, w, rng, args.n_boot)
        rec = {"animal": a, "severity": sev}
        for e in ("acute", "chronic"):
            if z_by[e]:
                d = _row_tv(Tpre, _survivor_T(np.concatenate(z_by[e]), survivors, K), w)
                rec[f"Tdist_{e}"] = round(d, 3)
                rec[f"z_{e}"] = round((d - null.mean()) / (null.std() + 1e-9), 1)
                rec[f"p_{e}"] = round(float((null >= d).mean()), 3)
        # lick-excluded biomarker deltas vs pre
        base = np.concatenate(ll_by["pre"]["nolick"]).mean()
        for e in ("acute", "chronic"):
            if ll_by[e]["nolick"]:
                rec[f"nolickLL_{e}"] = round(float(np.concatenate(ll_by[e]["nolick"]).mean() - base), 1)
        rec["boot_floor"] = round(float(null.mean()), 3)
        rows.append(rec)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== A. among-survivors conditional-transition reorganization (TV vs pre; z,p vs bootstrap null) ===")
    print(df[["animal", "severity", "boot_floor", "Tdist_acute", "z_acute", "p_acute",
              "Tdist_chronic", "z_chronic", "p_chronic"]].to_string(index=False))
    print("\n=== B. lick-excluded biomarker (non-lick AR-LL, Δ vs pre) ===")
    print(df[["animal", "severity", "nolickLL_acute", "nolickLL_chronic"]].to_string(index=False))
    out = lw / f"analysis_{args.tag}" / "noncircular_cohort.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
