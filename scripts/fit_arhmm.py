#!/usr/bin/env python
"""Fit the sticky AR-HMM and sweep kappa/nlags, scoring by lick-side MI + duration.

Canonical engine: the jax_moseq sticky HDP-AR-HMM (Keypoint-MoSeq core), fit on
the pooled pre-stroke PCA design (``scripts/build_design.py``). For each
(kappa, nlags) in the sweep we:
  1. fit on a spread subset of pre-stroke sessions (Gibbs),
  2. decode every pre-stroke session at full length,
  3. score the syllables by:
       - median syllable DURATION (s) — target the ~0.5–0.7 s behavioral range;
       - normalized lick-side ipsi/contra MI and running MI vs INDEPENDENT labels
         (the model never sees the task/treadmill channels as observations).
The (kappa, nlags) with duration in range and the highest lick-side MI is the
pick. Run once per sampling rate (33 vs 50 Hz caches) to choose the rate too.

This is the heavy step — run on a GPU (Apple-Silicon: `pip install jax-metal`;
the script uses whatever JAX device is available). Results + the chosen model are
written to ``data_local/fit_<family>/``.

Usage:
    python scripts/fit_arhmm.py --family B --kappas 1e7 1e8 1e9 --nlags 3
    python scripts/fit_arhmm.py --family A --kappas 1e7 1e8 1e9 --nlags 2 3
"""
from __future__ import annotations

import argparse
import csv
import pickle

import numpy as np
import pandas as pd

from arhmm_behavior.paths import PathResolver
from arhmm_behavior.model import moseq
from arhmm_behavior.model.scoring import (
    lick_side_labels,
    locomotion_labels,
    normalized_mi,
    durations_s,
)


def _spread_subset(rows: pd.DataFrame, n: int) -> pd.DataFrame:
    """Evenly-spaced subset of sessions to fit on (keeps cross-animal spread)."""
    if len(rows) <= n:
        return rows
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    return rows.iloc[np.unique(idx)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["A", "B"], required=True)
    ap.add_argument("--kappas", type=float, nargs="+", default=[1e7, 1e8, 1e9])
    ap.add_argument("--nlags", type=int, nargs="+", default=[3])
    ap.add_argument("--num-states", type=int, default=16)
    ap.add_argument("--n-fit-sessions", type=int, default=12)
    ap.add_argument("--num-iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    resolver = PathResolver()
    design_dir = resolver.local_work() / f"design_{args.family}"
    seqs_manifest = pd.read_csv(design_dir / "sequences.csv")
    rate = float(np.load(resolver.local_work() / f"design_{args.family}.npz")["rate_hz"])
    feat_dir = resolver.local_work() / "features" / args.family

    pre = seqs_manifest[seqs_manifest["epoch"] == "pre"].reset_index(drop=True)
    fit_rows = _spread_subset(pre, args.n_fit_sessions)
    fit_seqs = [np.load(design_dir / r["path"]) for _, r in fit_rows.iterrows()]
    print(f"family {args.family} @ {rate} Hz: fit on {len(fit_seqs)} pre-stroke sessions, "
          f"score on {len(pre)}")

    out_dir = resolver.local_work() / f"fit_{args.family}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results, best = [], None
    for nlags in args.nlags:
        for kappa in args.kappas:
            data = moseq.make_data(fit_seqs, truncate=True)
            model, _ = moseq.fit(data, num_states=args.num_states, kappa=kappa,
                                 nlags=nlags, num_iters=args.num_iters, seed=args.seed)
            # Decode every pre-stroke session at full length and pool scores.
            all_dur, z_all, lick_all, run_all = [], [], [], []
            for _, r in pre.iterrows():
                seq = np.load(design_dir / r["path"])
                z = moseq.decode(moseq.make_data([seq], truncate=False), model)[0]
                tbl = pd.read_parquet(feat_dir / r["animal"] / f"{r['date']}.parquet")
                all_dur.append(durations_s(z, rate))
                z_all.append(z)
                lick_all.append(lick_side_labels(tbl, nlags=nlags)[:len(z)])
                run_all.append(locomotion_labels(tbl, nlags=nlags)[:len(z)])
            z_cat = np.concatenate(z_all)
            med = float(np.median(np.concatenate(all_dur)))
            mi_lick = normalized_mi(z_cat, np.concatenate(lick_all))
            mi_run = normalized_mi(z_cat, np.concatenate(run_all))
            rec = {"nlags": nlags, "kappa": kappa, "rate_hz": rate,
                   "median_dur_s": round(med, 3), "mi_lick_side": round(mi_lick, 3),
                   "mi_running": round(mi_run, 3)}
            results.append(rec)
            print(f"  nlags={nlags} kappa={kappa:.0e}: dur={med:.2f}s "
                  f"lick-side MI={mi_lick:.3f} running MI={mi_run:.3f}")
            in_range = 0.3 <= med <= 0.9
            score = (in_range, mi_lick)
            if best is None or score > best[0]:
                best = (score, rec, model)

    with open(out_dir / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    with open(out_dir / "best_model.pkl", "wb") as f:
        pickle.dump({"model": best[2], "config": best[1]}, f)
    print(f"\nBEST: {best[1]}  -> wrote sweep_results.csv + best_model.pkl to {out_dir}")


if __name__ == "__main__":
    main()
