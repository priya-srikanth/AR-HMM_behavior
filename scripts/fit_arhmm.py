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
    ap.add_argument("--num-iters", type=int, default=200,
                    help="Gibbs passes. 50 under-converges (degenerate fits / low MI); "
                         "use ~200-300, more at higher rates (longer sequences).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="random seeds; the best non-collapsed seed per (kappa,nlags) is "
                         "kept, so a single unlucky seed doesn't drive a collapse.")
    ap.add_argument("--rate", type=float, default=None,
                    help="model-grid rate (Hz); default from configs/defaults.yaml. "
                         "Selects the rate-namespaced design/features to fit. Run at "
                         "33 and 50 Hz and compare sweep_results.csv to pick the rate.")
    args = ap.parse_args()

    resolver = PathResolver()
    import yaml
    rate_sel = args.rate if args.rate else float(yaml.safe_load(
        open(resolver.config_path.parent / "defaults.yaml"))["model"]["sampling_rate_hz"])
    tag = f"{args.family}_{int(round(rate_sel))}hz"
    design_dir = resolver.local_work() / f"design_{tag}"
    seqs_manifest = pd.read_csv(design_dir / "sequences.csv")
    rate = float(np.load(resolver.local_work() / f"design_{tag}.npz")["rate_hz"])
    feat_dir = resolver.local_work() / "features" / tag

    pre = seqs_manifest[seqs_manifest["epoch"] == "pre"].reset_index(drop=True)
    fit_rows = _spread_subset(pre, args.n_fit_sessions)
    fit_seqs = [np.load(design_dir / r["path"]) for _, r in fit_rows.iterrows()]
    print(f"family {args.family} @ {rate} Hz: fit on {len(fit_seqs)} pre-stroke sessions, "
          f"score on {len(pre)}")

    out_dir = resolver.local_work() / f"fit_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = moseq.make_data(fit_seqs, truncate=True)

    def _score(model, nlags):
        """Decode every pre-stroke session at full length and pool scores.

        Returns (median duration s, lick-side MI, running MI, #states used).
        ``#states used`` < ~2 flags a degenerate collapse to a single syllable.
        """
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
        return (med, normalized_mi(z_cat, np.concatenate(lick_all)),
                normalized_mi(z_cat, np.concatenate(run_all)), int(len(np.unique(z_cat))))

    results, best = [], None
    for nlags in args.nlags:
        for kappa in args.kappas:
            # Fit each seed; keep the best NON-COLLAPSED one (so a single unlucky
            # seed landing in a 1-state / runaway-duration mode isn't reported).
            seed_best = None
            for seed in args.seeds:
                model, _ = moseq.fit(data, num_states=args.num_states, kappa=kappa,
                                     nlags=nlags, num_iters=args.num_iters, seed=seed)
                med, mi_lick, mi_run, n_used = _score(model, nlags)
                valid = (n_used >= 2) and (0.05 <= med <= 5.0)
                rec = {"nlags": nlags, "kappa": kappa, "rate_hz": rate, "seed": seed,
                       "median_dur_s": round(med, 3), "n_states_used": n_used,
                       "mi_lick_side": round(mi_lick, 3), "mi_running": round(mi_run, 3)}
                key = (valid, mi_lick)  # prefer non-collapsed, then high lick-side MI
                if seed_best is None or key > seed_best[0]:
                    seed_best = (key, rec, model)
            rec = seed_best[1]
            results.append(rec)
            print(f"  nlags={nlags} kappa={kappa:.0e} (best of {len(args.seeds)} seeds, "
                  f"seed={rec['seed']}): dur={rec['median_dur_s']}s "
                  f"states={rec['n_states_used']} lick-MI={rec['mi_lick_side']} "
                  f"run-MI={rec['mi_running']}")
            in_range = 0.3 <= rec["median_dur_s"] <= 0.9
            gscore = (in_range, rec["mi_lick_side"])
            if best is None or gscore > best[0]:
                best = (gscore, rec, seed_best[2])

    with open(out_dir / "sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    with open(out_dir / "best_model.pkl", "wb") as f:
        pickle.dump({"model": best[2], "config": best[1]}, f)
    print(f"\nBEST: {best[1]}  -> wrote sweep_results.csv + best_model.pkl to {out_dir}")


if __name__ == "__main__":
    main()
