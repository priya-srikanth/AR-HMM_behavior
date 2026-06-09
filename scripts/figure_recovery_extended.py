#!/usr/bin/env python
"""Two readouts the usage/biomarker figures don't show:

(1) TRANSITION matrices (PS46): pooled syllable transitions (self removed,
    row-normalized) for pre / acute / chronic, states ordered ipsi|contra|other,
    plus "fraction of transitions ENTERING a contra syllable" by epoch.
    CAVEAT (important — do not over-read): entries-into-contra ≈ contra usage ÷
    dwell, and dwell is ~constant here, so this routing metric is CONFOUNDED with
    contra usage / the raw contra lick count (which spout behavior already gives).
    It re-expresses the lick deficit; it does NOT independently demonstrate
    sequence reorganization. The non-circular test (TODO) is the CONDITIONAL
    structure at matched occupancy: restrict+renormalize transitions over the
    non-contra (surviving) states and ask whether that subnetwork rewires — if it
    doesn't, the lesion is a *selective* deletion of contra licking, which counts
    alone can't establish.
(2) SEVERITY-STRATIFIED COHORT (PS46-50): contra-syllable usage and the AR-LL
    manifold biomarker across days-from-stroke, colored by 3-tier severity — does
    the contra collapse/recovery scale with lesion size?

State side-identity is read from the model itself (pre-stroke lick cross-tab),
not hardcoded. Usage/biomarker come from the cached per-session CSV; transitions
re-decode PS46 only.

Usage: python scripts/figure_recovery_extended.py [--tag B_33hz]
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
from arhmm_behavior.model.scoring import lick_side_labels, state_side_counts, transition_matrix

_SEV_COLOR = {"mild": "#4575b4", "moderate": "#f46d43", "severe": "#d73027"}
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="B_33hz")
    ap.add_argument("--animal", default="PS46")
    args = ap.parse_args()

    r = PathResolver()
    lw = r.local_work()
    blob = pickle.load(open(lw / f"fit_{args.tag}" / "best_model.pkl", "rb"))
    model, cfg = blob["model"], blob["config"]
    nlags = int(cfg["nlags"])
    K = int(np.asarray(model["params"]["Ab"]).shape[0])
    design_dir = lw / f"design_{args.tag}"
    feat_dir = lw / "features" / args.tag
    seqs = pd.read_csv(design_dir / "sequences.csv")
    animals = yaml.safe_load(open(r.animals_yaml()))

    # --- side identity per state (pooled pre-stroke lick cross-tab) ---
    tot = np.zeros((K, 2), int)
    for _, row in seqs[seqs["epoch"] == "pre"].head(20).iterrows():
        z = moseq.decode(moseq.make_data([np.load(design_dir / row["path"])], truncate=False), model)[0]
        side = lick_side_labels(pd.read_parquet(feat_dir / row["animal"] / f"{row['date']}.parquet"),
                                nlags=nlags)[:len(z)]
        tot += state_side_counts(z, side, K)
    licks = tot.sum(1)
    frac = np.divide(tot[:, 0], licks, out=np.full(K, np.nan), where=licks > 0)
    thr = 0.02 * licks.sum()
    ipsi = [k for k in range(K) if licks[k] >= thr and frac[k] >= 0.8]
    contra = [k for k in range(K) if licks[k] >= thr and frac[k] <= 0.2]
    other = [k for k in range(K) if k not in ipsi and k not in contra]
    order = ipsi + contra + other
    print(f"ipsi={ipsi} contra={contra} other={other}")

    # ============ (1) TRANSITION REORGANIZATION (PS46) ============
    a = args.animal
    eb = animals[a]["epoch_boundaries"]
    z_by = {"pre": [], "acute": [], "chronic": []}
    for _, row in seqs[seqs["animal"] == a].iterrows():
        e = _epoch(int(row["rel_day"]), eb)
        if e in z_by:
            z = moseq.decode(moseq.make_data([np.load(design_dir / row["path"])], truncate=False), model)[0]
            z_by[e].append(z)
    inflow = {}   # fraction of (non-self) transitions whose destination is a contra syllable
    Tm = {}
    for e, zs in z_by.items():
        zc = np.concatenate(zs)
        T = transition_matrix(zc, K, drop_self=True)          # row-stochastic, no diagonal
        Tm[e] = T[np.ix_(order, order)]
        # weight rows by how often each source state is left, to get true inflow share
        leaves = np.array([(np.diff(zc) != 0)[zc[:-1] == i].sum() for i in range(K)], float)
        dest = (leaves[:, None] * T).sum(0)                   # expected transitions into each state
        inflow[e] = dest[contra].sum() / dest.sum() if dest.sum() else np.nan

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, 0.7]})
    nb = len(ipsi)
    for ax, e in zip(axes[:3], ["pre", "acute", "chronic"]):
        im = ax.imshow(Tm[e], cmap="magma", vmin=0, vmax=min(0.6, np.nanmax([Tm[x].max() for x in Tm])))
        ax.set_title(f"{a} {e}\nP(next | current), self removed")
        ax.set_xticks(range(K)); ax.set_xticklabels([f"S{k}" for k in order], fontsize=5, rotation=90)
        ax.set_yticks(range(K)); ax.set_yticklabels([f"S{k}" for k in order], fontsize=5)
        # box the contra destination columns (transitions INTO contra)
        ax.add_patch(plt.Rectangle((nb - 0.5, -0.5), len(contra), K, fill=False, ec="cyan", lw=1.8))
        ax.set_xlabel("destination (cyan = contra)")
    axes[0].set_ylabel("source syllable")
    axes[3].bar(list(inflow.keys()), list(inflow.values()),
                color=["#888", "#f46d43", "#66c2a5"])
    axes[3].set_title("routing INTO contra\n(share of transitions)")
    axes[3].set_ylabel("fraction of transitions → contra")
    fig.tight_layout()
    f1 = r.outputs() / f"{a.lower()}_transition_reorg.png"
    fig.savefig(f1, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"inflow-to-contra by epoch: { {k: round(v,3) for k,v in inflow.items()} }")
    print(f"wrote {f1}")

    # ============ (2) SEVERITY-STRATIFIED COHORT ============
    d = pd.read_csv(lw / f"analysis_{args.tag}" / "per_session_syllables.csv")
    d["sev3"] = d["severity_4tier"].map(lambda s: _TO_3TIER.get(s, s))
    d["CONTRA"] = d[[f"use_S{k}" for k in contra]].sum(axis=1)
    pre_ll = d[d["epoch"] == "pre"].groupby("animal")["ar_loglik"].transform("mean")
    d["ar_rel"] = d["ar_loglik"] - d.groupby("animal")["ar_loglik"].transform(
        lambda s: s[d.loc[s.index, "epoch"] == "pre"].mean())

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)
    for ax in (axA, axB):
        ax.axvspan(d["rel_day"].min() - 1, 0, color="0.93"); ax.axvline(0, color="k", lw=1)
    for an, g in d.groupby("animal"):
        g = g.sort_values("rel_day")
        sev = g["sev3"].iloc[0]
        col = _SEV_COLOR.get(sev, "#333")
        axA.plot(g["rel_day"], g["CONTRA"], "o-", ms=3, lw=1.3, color=col, alpha=0.8, label=f"{an} ({sev})")
        axB.plot(g["rel_day"], g["ar_rel"], "o-", ms=3, lw=1.3, color=col, alpha=0.8)
    axA.set_ylabel("contra-syllable usage"); axA.legend(fontsize=7, ncol=2)
    axA.set_title("PS46–50: contralesional lick-syllable usage across stroke, by 3-tier severity")
    axB.axhline(0, color="gray", ls=":", lw=0.8)
    axB.set_ylabel("AR log-lik − pre\n(manifold distortion)"); axB.set_xlabel("days from stroke")
    fig.tight_layout()
    f2 = r.outputs() / "cohort_recovery_by_severity.png"
    fig.savefig(f2, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {f2}")


if __name__ == "__main__":
    main()
