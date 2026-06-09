#!/usr/bin/env python
"""PS46 recovery figure: lateralized lick-syllable usage + manifold biomarker across stroke.

Reuses the per-session readout (``syllable_analysis.py`` ->
``analysis_<tag>/per_session_syllables.csv``) and labels each syllable's SIDE
identity from the model itself (decode pre-stroke sessions, cross-tab ipsi-L vs
contra-R lick bins) rather than hardcoding state indices. Two panels:
  (top) per-syllable usage across days-from-stroke, with the contra-specific
        (red) and ipsi-specific (blue) lick syllables highlighted + their sums;
  (bottom) the AR-LL manifold-distortion biomarker (relative to pre-stroke mean).

Usage: python scripts/figure_ps46_recovery.py [--animal PS46] [--tag B_33hz]
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arhmm_behavior.paths import PathResolver
from arhmm_behavior.model import moseq
from arhmm_behavior.model.scoring import lick_side_labels, state_side_counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="PS46")
    ap.add_argument("--tag", default="B_33hz")
    ap.add_argument("--n-id-sessions", type=int, default=20,
                    help="pre-stroke sessions (all animals) used to label each "
                         "state's ipsi/contra identity")
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

    # --- side identity per state, from pooled pre-stroke decodes ---
    pre = seqs[seqs["epoch"] == "pre"].head(args.n_id_sessions)
    tot = np.zeros((K, 2), int)
    for _, row in pre.iterrows():
        z = moseq.decode(moseq.make_data([np.load(design_dir / row["path"])], truncate=False), model)[0]
        tbl = pd.read_parquet(feat_dir / row["animal"] / f"{row['date']}.parquet")
        side = lick_side_labels(tbl, nlags=nlags)[:len(z)]
        tot += state_side_counts(z, side, K)
    licks = tot.sum(1)
    frac = np.divide(tot[:, 0], licks, out=np.full(K, np.nan), where=licks > 0)  # ipsi fraction
    thresh = 0.02 * licks.sum()
    ipsi = [k for k in range(K) if licks[k] >= thresh and frac[k] >= 0.8]
    contra = [k for k in range(K) if licks[k] >= thresh and frac[k] <= 0.2]
    print(f"ipsi states {ipsi}  contra states {contra}  (of K={K}, by pre-stroke lick cross-tab)")

    # --- PS46 per-session usage + biomarker ---
    d = pd.read_csv(lw / f"analysis_{args.tag}" / "per_session_syllables.csv")
    p = d[d["animal"] == args.animal].sort_values("rel_day").reset_index(drop=True)
    x = p["rel_day"].to_numpy()
    pre_ll = p[p["epoch"] == "pre"]["ar_loglik"].mean()

    reds = plt.cm.Reds(np.linspace(0.45, 0.9, max(len(contra), 1)))
    blues = plt.cm.Blues(np.linspace(0.45, 0.9, max(len(ipsi), 1)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})

    for ax in (ax1, ax2):
        ax.axvspan(x.min() - 1, 0, color="0.93", zorder=0)  # pre-stroke shade
        ax.axvline(0, color="k", lw=1.0, ls="-")

    # individual lick states (thin) + bold sums
    for c, k in zip(reds, contra):
        ax1.plot(x, p[f"use_S{k}"], "-", color=c, lw=1.0, alpha=0.7, label=f"S{k} contra")
    for c, k in zip(blues, ipsi):
        ax1.plot(x, p[f"use_S{k}"], "-", color=c, lw=1.0, alpha=0.7, label=f"S{k} ipsi")
    ax1.plot(x, p[[f"use_S{k}" for k in contra]].sum(axis=1), "o-", color="#c0392b",
             lw=2.6, ms=5, label="CONTRA total", zorder=5)
    ax1.plot(x, p[[f"use_S{k}" for k in ipsi]].sum(axis=1), "s-", color="#2471a3",
             lw=2.6, ms=5, label="IPSI total", zorder=5)
    ax1.set_ylabel("syllable usage (frac. of frames)")
    ax1.set_title(f"{args.animal} ({cfg.get('rate_hz')} Hz, MI={cfg.get('mi_lick_side')}): "
                  f"lateralized lick-syllable usage across stroke\n"
                  f"contra (mouse-R = contralesional) collapses acutely, then recovers/overshoots")
    ax1.legend(fontsize=7, ncol=2, loc="upper left")

    ax2.plot(x, p["ar_loglik"] - pre_ll, "o-", color="#6c3483", lw=2.2, ms=5)
    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_ylabel("AR log-lik − pre mean\n(manifold distortion)")
    ax2.set_xlabel("days from stroke")

    # epoch tick annotations
    for _, row in p.iterrows():
        pass
    fig.tight_layout()
    out = r.outputs() / "ps46_recovery_arhmm.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
