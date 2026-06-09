#!/usr/bin/env python
"""Does the fitted AR-HMM SPLIT ipsi vs contra licks, or MERGE them?

Lick-side MI (scripts/fit_arhmm.py) says *whether* syllables carry side
information; this says *how*. It decodes a few pre-stroke sessions with the
fitted best model and, for every syllable, counts ipsi-L vs contra-R lick bins
(``lick_side_labels``: 0 = ipsi-L, 1 = contra-R). It prints the lick-bearing
states sorted by total licks, with each state's ipsi-fraction:

  - a CLEAN lateralized model has some states ~all ipsi and others ~all contra;
  - a MERGED model has its lick states holding both sides near the base rate.

Reads the canonical rate-tagged caches written by build_design.py / fit_arhmm.py
(``design_<tag>``, ``fit_<tag>/best_model.pkl``, ``features/<tag>``).

Usage:
    python scripts/diagnose_lick_split.py --tag B_33hz --n-sessions 6
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from arhmm_behavior.paths import PathResolver
from arhmm_behavior.model import moseq
from arhmm_behavior.model.scoring import (
    lick_side_labels,
    state_side_counts,
    normalized_mi,
)


def _num_states(model) -> int:
    try:
        return int(model["hypparams"]["trans_hypparams"]["num_states"])
    except (KeyError, TypeError):
        return int(np.asarray(model["params"]["betas"]).shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="rate-tagged family, e.g. B_33hz")
    ap.add_argument("--n-sessions", type=int, default=6,
                    help="number of pre-stroke sessions to decode and pool")
    args = ap.parse_args()

    r = PathResolver()
    lw = r.local_work()
    design_dir = lw / f"design_{args.tag}"
    feat_dir = lw / "features" / args.tag
    blob = pickle.load(open(lw / f"fit_{args.tag}" / "best_model.pkl", "rb"))
    model, cfg = blob["model"], blob["config"]
    nlags = int(cfg["nlags"])
    num_states = _num_states(model)
    print(f"[{args.tag}] best config: {cfg}")
    print(f"decoding {args.n_sessions} pre-stroke sessions; num_states={num_states}, nlags={nlags}")

    seqs = pd.read_csv(design_dir / "sequences.csv")
    pre = seqs[seqs["epoch"] == "pre"].head(args.n_sessions)

    total = np.zeros((num_states, 2), dtype=int)
    z_all, side_all = [], []
    for _, row in pre.iterrows():
        seq = np.load(design_dir / row["path"])
        z = moseq.decode(moseq.make_data([seq], truncate=False), model)[0]
        tbl = pd.read_parquet(feat_dir / row["animal"] / f"{row['date']}.parquet")
        side = lick_side_labels(tbl, nlags=nlags)[:len(z)]
        total += state_side_counts(z, side, num_states)
        z_all.append(z)
        side_all.append(side)

    z_cat = np.concatenate(z_all)
    side_cat = np.concatenate(side_all)
    n_L, n_R = int((side_cat == 0).sum()), int((side_cat == 1).sum())
    base_ipsi = n_L / (n_L + n_R) if (n_L + n_R) else float("nan")
    mi = normalized_mi(z_cat, side_cat)
    print(f"\npooled lick bins: ipsi-L={n_L}  contra-R={n_R}  "
          f"base ipsi-fraction={base_ipsi:.2f}  |  lick-side MI (these sessions)={mi:.3f}")

    # Lick-bearing states, most-licked first. ipsi_frac near base = merged;
    # near 1.0 = ipsi-specific; near 0.0 = contra-specific.
    print("\nstate   ipsi-L  contra-R   total   ipsi_frac   verdict")
    order = np.argsort(-total.sum(1))
    for s in order:
        nl, nr = int(total[s, 0]), int(total[s, 1])
        tot = nl + nr
        if tot == 0:
            continue
        frac = nl / tot
        verdict = ("ipsi-specific" if frac >= 0.80 else
                   "contra-specific" if frac <= 0.20 else "MERGED (both sides)")
        print(f"  {s:>3}   {nl:>6}  {nr:>7}   {tot:>6}     {frac:>5.2f}     {verdict}")

    # One-line summary: are there both an ipsi-dominant and a contra-dominant
    # state among those carrying a meaningful share of licks?
    licky = [s for s in range(num_states) if total[s].sum() >= 0.05 * (n_L + n_R)]
    has_ipsi = any(total[s, 0] / total[s].sum() >= 0.80 for s in licky)
    has_contra = any(total[s, 0] / total[s].sum() <= 0.20 for s in licky)
    print(f"\nSUMMARY: {'CLEAN SPLIT' if (has_ipsi and has_contra) else 'NO CLEAN SPLIT — sides merged'} "
          f"(ipsi-specific state present={has_ipsi}, contra-specific state present={has_contra})")


if __name__ == "__main__":
    main()
