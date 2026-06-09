#!/usr/bin/env python
"""WHAT is the post-stroke 'rewiring'? Characterize the among-survivors transition
change in behavioral terms (PS46).

1. LABEL each syllable by its mean raw-feature profile (tongue-out, |tongue angle|,
   jaw motion, run speed) + its ipsi/contra lick identity -> a short behavioral tag.
2. Among the surviving (non-contra) states, list the TRANSITIONS that change most
   pre->acute (and ->chronic), with both states' tags and P(next|current) before/after.
3. SUMMARIZE the direction: routing mass into behavior CATEGORIES per epoch, and the
   mean per-source transition ENTROPY (more deterministic vs more random sequencing).

Usage: python scripts/characterize_rewiring.py [--animal PS46] [--tag B_33hz]
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

_FEATS = ["tongue_out_frac", "tongue_angle_mean", "jaw_motion_energy",
          "treadmill_speed_mm_s", "tongue_motion_energy"]


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


def _survivor_T(zc, survivors, K):
    T = transition_matrix(zc, K, drop_self=True)
    sub = T[np.ix_(survivors, survivors)]
    rs = sub.sum(1, keepdims=True)
    return np.divide(sub, rs, out=np.zeros_like(sub), where=rs > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="PS46")
    ap.add_argument("--tag", default="B_33hz")
    args = ap.parse_args()
    a = args.animal

    r = PathResolver(); lw = r.local_work()
    blob = pickle.load(open(lw / f"fit_{args.tag}" / "best_model.pkl", "rb"))
    model = blob["model"]; nlags = int(blob["config"]["nlags"])
    K = int(np.asarray(model["params"]["Ab"]).shape[0])
    design_dir = lw / f"design_{args.tag}"; feat_dir = lw / "features" / args.tag
    seqs = pd.read_csv(design_dir / "sequences.csv")
    eb = yaml.safe_load(open(r.animals_yaml()))[a]["epoch_boundaries"]

    # global side identity
    tot = np.zeros((K, 2), int)
    for _, row in seqs[seqs["epoch"] == "pre"].head(20).iterrows():
        z = moseq.decode(moseq.make_data([np.load(design_dir / row["path"])], truncate=False), model)[0]
        side = lick_side_labels(pd.read_parquet(feat_dir / row["animal"] / f"{row['date']}.parquet"),
                                nlags=nlags)[:len(z)]
        tot += state_side_counts(z, side, K)
    licks = tot.sum(1); sidefrac = np.divide(tot[:, 0], licks, out=np.full(K, np.nan), where=licks > 0)
    thr = 0.02 * licks.sum()
    contra = [k for k in range(K) if licks[k] >= thr and sidefrac[k] <= 0.2]
    ipsi = [k for k in range(K) if licks[k] >= thr and sidefrac[k] >= 0.8]
    survivors = [k for k in range(K) if k not in contra]

    # decode animal: z + per-frame raw features by epoch; pool feature sums per state (all epochs)
    z_by = {"pre": [], "acute": [], "chronic": []}
    fsum = np.zeros((K, len(_FEATS))); fcnt = np.zeros(K)
    for _, row in seqs[seqs["animal"] == a].iterrows():
        e = _epoch(int(row["rel_day"]), eb)
        if e not in z_by:
            continue
        seq = np.load(design_dir / row["path"])
        z = moseq.decode(moseq.make_data([seq], truncate=False), model)[0]
        tbl = pd.read_parquet(feat_dir / a / f"{row['date']}.parquet")
        F = tbl[_FEATS].to_numpy()[nlags:nlags + len(z)]
        F = np.abs(F) if False else F.copy()
        F[:, 1] = np.abs(F[:, 1])   # |tongue angle| (laterality magnitude)
        for k in range(K):
            m = z == k
            fsum[k] += np.nan_to_num(F[m]).sum(0); fcnt[k] += m.sum()
        z_by[e].append(z)
    prof = fsum / np.maximum(fcnt[:, None], 1)               # per-state mean raw features
    zsc = (prof - prof.mean(0)) / (prof.std(0) + 1e-9)        # z-scored across states

    def label(k):
        if k in contra: return "contra-lick"
        if k in ipsi:   return "ipsi-lick"
        TO, ANG, JAW, RUN, TM = zsc[k]
        if RUN > 1.0:   return "run"
        if JAW > 0.7:   return "jaw/chew"
        if TO > 0.7:    return "tongue-other"
        if TM > 0.7:    return "face-motion"
        return "quiescent"

    lab = {k: label(k) for k in range(K)}
    pre_use = np.bincount(np.concatenate(z_by["pre"]), minlength=K) / len(np.concatenate(z_by["pre"]))

    print(f"{a}: contra={contra} ipsi={ipsi}\n")
    print("=== syllable behavioral profiles (mean raw features; * = lick state) ===")
    print(f"{'St':>3} {'label':12s} {'tongOut':>7} {'|ang|':>6} {'jawMot':>7} {'run':>6} {'use_pre':>7}")
    for k in sorted(range(K), key=lambda k: -pre_use[k]):
        print(f"{k:>3} {lab[k]:12s} {prof[k,0]:7.3f} {prof[k,1]:6.1f} {prof[k,2]:7.3f} "
              f"{prof[k,3]:6.1f} {pre_use[k]:7.3f}")

    # among-survivors transitions, top changes pre->acute
    Tpre = _survivor_T(np.concatenate(z_by["pre"]), survivors, K)
    Tac = _survivor_T(np.concatenate(z_by["acute"]), survivors, K)
    dT = Tac - Tpre
    si = {s: i for i, s in enumerate(survivors)}
    print("\n=== top among-survivors transition CHANGES pre->acute (P(next|current)) ===")
    flat = sorted(((abs(dT[si[i], si[j]]), i, j) for i in survivors for j in survivors if i != j),
                  reverse=True)[:14]
    print(f"{'from':>20} -> {'to':<20} {'P_pre':>6} {'P_acute':>7} {'Δ':>6}")
    for _, i, j in flat:
        print(f"{('S%d '%i+lab[i]):>20} -> {('S%d '%j+lab[j]):<20} "
              f"{Tpre[si[i],si[j]]:6.2f} {Tac[si[i],si[j]]:7.2f} {dT[si[i],si[j]]:+6.2f}")

    # summary: routing mass into behavior categories, + transition entropy
    cats = sorted(set(lab[s] for s in survivors))
    def cat_inflow(T):
        # weight source rows by pre usage among survivors -> destination category mass
        w = np.array([pre_use[s] for s in survivors]); w = w / w.sum()
        dest = (w[:, None] * T).sum(0)
        return {c: float(sum(dest[si[s]] for s in survivors if lab[s] == c)) for c in cats}
    def mean_entropy(T):
        H = []
        for s in survivors:
            p = T[si[s]]; p = p[p > 0]
            if p.size: H.append(-(p * np.log(p)).sum())
        return float(np.mean(H))
    print("\n=== routing mass into behavior categories (survivor destinations) ===")
    ip, ia = cat_inflow(Tpre), cat_inflow(Tac)
    for c in cats:
        print(f"  {c:14s} pre {ip[c]:.3f} -> acute {ia[c]:.3f}  ({ia[c]-ip[c]:+.3f})")
    Hpre, Hac = mean_entropy(Tpre), mean_entropy(Tac)
    print(f"\nmean per-source transition entropy: pre {Hpre:.2f} -> "
          f"acute {Hac:.2f}  (lower = more stereotyped/deterministic sequencing)")

    # ---------------- summary figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = [(i, j, dT[si[i], si[j]]) for _, i, j in flat]
    labels = [f"S{i} {lab[i]} → S{j} {lab[j]}" for i, j, _ in items]
    vals = [d for _, _, d in items]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={"width_ratios": [1.7, 1]})
    y = np.arange(len(items))[::-1]
    axL.barh(y, vals, color=["#2ca02c" if v > 0 else "#d62728" for v in vals])
    axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=8)
    axL.axvline(0, color="k", lw=0.8)
    axL.set_xlabel("Δ P(next | current), pre → acute")
    axL.set_title(f"{a}: which surviving transitions rewire after stroke\n"
                  "green = more likely, red = less likely (self-transitions removed)")
    dcat = sorted(((ia[c] - ip[c], c) for c in cats), reverse=True)
    axR.barh([c for _, c in dcat][::-1], [v for v, _ in dcat][::-1],
             color=["#2ca02c" if v > 0 else "#d62728" for v, _ in dcat][::-1])
    axR.axvline(0, color="k", lw=0.8)
    axR.set_xlabel("Δ routing mass into category, pre → acute")
    axR.set_title(f"coarse picture: destination-category mass\n"
                  f"(sequencing entropy {Hpre:.2f}→{Hac:.2f}, ~unchanged)")
    fig.tight_layout()
    out = r.outputs() / f"{a.lower()}_rewiring_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
