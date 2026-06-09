#!/usr/bin/env python
"""Epoch- and severity-stratified syllable analysis — the scientific readout.

Decodes every session (pre + post) through the FROZEN pre-stroke AR-HMM
(``scripts/fit_arhmm.py`` → ``best_model.pkl``) and asks how behavioral structure
breaks and recovers after stroke. Per session it computes:

  - **syllable usage** : fraction of frames in each of the K syllables;
  - **median dwell**   : median syllable duration (s);
  - **AR-LL biomarker**: mean per-frame autoregressive log-likelihood of the
    session under the frozen pre-stroke model — a continuous "distance from the
    pre-stroke behavioral manifold" (drops post-stroke, recovers with healing;
    FINDINGS F9).

Sessions are tagged with their recovery EPOCH (pre/acute/sub_acute/chronic, from
the per-animal boundaries in the upstream ``animals.yaml``) and SEVERITY group
under both the 4-tier (``recovery_phenotype``) and 3-tier (mod-severe→moderate)
schemes. Outputs a per-session CSV plus figures: usage-by-epoch, the AR-LL
recovery trajectory stratified by severity, and pre-vs-chronic transition
matrices.

NOTE: the AR log-likelihood reads the jax_moseq AR params (``Ab``, ``Q``)
directly; if your installed jax_moseq names them differently, adjust
``_ar_loglik_per_frame`` (flagged inline). Validate on a real fit.

Usage:
    python scripts/syllable_analysis.py --family B
    python scripts/syllable_analysis.py --family A
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
from arhmm_behavior.model.scoring import usage, durations_s, transition_matrix

_EPOCH_COLOR = {"pre": "#888888", "acute": "#fb8072", "sub_acute": "#ffd92f", "chronic": "#66c2a5"}
_SEV_COLOR = {"mild": "#4575b4", "moderate": "#fdae61", "mod-severe": "#f46d43", "severe": "#d73027"}
_TO_3TIER = {"moderate-severe": "moderate", "mod-severe": "moderate"}


def _epoch_label(rel_day: int, eb: dict) -> str | None:
    """Map a session's rel_day to its epoch using the per-animal boundaries."""
    if rel_day <= 0:
        return "pre"
    if rel_day <= eb["acute_max_rel_day"]:
        return "acute"
    if rel_day <= eb["subacute_max_rel_day"]:
        return "sub_acute"
    if rel_day >= eb["chronic_min_rel_day"]:
        return "chronic"
    return None  # transitional gap (none in the current cohort)


def _ar_loglik_per_frame(seq: np.ndarray, z: np.ndarray, params: dict, nlags: int) -> float:
    """Mean per-frame AR log-likelihood of ``seq`` under each frame's decoded state.

    For frame t (>= nlags) in state k: predict x_t from the lag stack via the
    state's AR matrix ``Ab[k]`` (last column = bias) and score the residual under
    a Gaussian with covariance ``Q[k]``. Lower mean = further from the pre-stroke
    manifold. (jax_moseq stores ``Ab``/``Q`` under ``params['ar']`` in recent
    versions; falls back to top-level keys — adjust here if your version differs.)
    """
    ar = params.get("ar", params)
    Ab = np.asarray(ar["Ab"])              # (K, D, D*nlags + 1)
    Q = np.asarray(ar["Q"])                # (K, D, D)
    D = seq.shape[1]
    lls = []
    Qinv = np.linalg.inv(Q)
    logdet = np.linalg.slogdet(Q)[1]
    const = -0.5 * (D * np.log(2 * np.pi) + logdet)   # (K,)
    for t in range(nlags, len(seq)):
        k = int(z[t - nlags])
        lag = seq[t - nlags:t][::-1].reshape(-1)      # most-recent-first lag stack
        x_in = np.concatenate([lag, [1.0]])           # + bias
        resid = seq[t] - Ab[k] @ x_in
        lls.append(const[k] - 0.5 * resid @ Qinv[k] @ resid)
    return float(np.mean(lls)) if lls else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["A", "B"], required=True)
    ap.add_argument("--rate", type=float, default=None,
                    help="model-grid rate (Hz); default from configs/defaults.yaml. "
                         "Selects the rate-namespaced fit/design to analyze.")
    args = ap.parse_args()

    resolver = PathResolver()
    lw = resolver.local_work()
    import yaml
    rate_sel = args.rate if args.rate else float(yaml.safe_load(
        open(resolver.config_path.parent / "defaults.yaml"))["model"]["sampling_rate_hz"])
    tag = f"{args.family}_{int(round(rate_sel))}hz"
    with open(lw / f"fit_{tag}" / "best_model.pkl", "rb") as f:
        blob = pickle.load(f)
    model, cfg = blob["model"], blob["config"]
    nlags, rate, K = cfg["nlags"], cfg["rate_hz"], int(cfg["num_states"]) if "num_states" in cfg else None
    design_dir = lw / f"design_{tag}"
    seqs = pd.read_csv(design_dir / "sequences.csv")

    animals = yaml.safe_load(open(resolver.animals_yaml()))
    # infer K from the model if not in config
    if K is None:
        K = int(np.asarray(model["params"].get("ar", model["params"])["Ab"]).shape[0])

    rows, usage_by = [], {}
    for _, r in seqs.iterrows():
        seq = np.load(design_dir / r["path"])
        z = moseq.decode(moseq.make_data([seq], truncate=False), model)[0]
        info = animals[r["animal"]]
        eb = info["epoch_boundaries"]
        epoch = _epoch_label(int(r["rel_day"]), eb)
        pheno4 = info.get("recovery_phenotype")
        pheno3 = _TO_3TIER.get(pheno4, pheno4)
        u = usage(z, K)
        usage_by[(r["animal"], r["date"])] = u
        rows.append({
            "animal": r["animal"], "date": r["date"], "rel_day": int(r["rel_day"]),
            "epoch": epoch, "severity_4tier": pheno4, "severity_3tier": pheno3,
            "median_dwell_s": round(float(np.median(durations_s(z, rate))), 3),
            "ar_loglik": round(_ar_loglik_per_frame(seq, z, model["params"], nlags), 3),
            **{f"use_S{k}": round(float(u[k]), 4) for k in range(K)},
        })
    df = pd.DataFrame(rows)
    out = lw / f"analysis_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_session_syllables.csv", index=False)

    # --- Figure 1: AR-LL manifold-distortion biomarker vs rel_day, by severity ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pre_mean = df[df["epoch"] == "pre"]["ar_loglik"].mean()
    for sev, g in df.groupby("severity_3tier"):
        g = g.sort_values("rel_day")
        ax.plot(g["rel_day"], g["ar_loglik"] - pre_mean, "o-", ms=4,
                color=_SEV_COLOR.get(sev, "#333"), label=sev, alpha=0.8)
    ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("days from stroke"); ax.set_ylabel("AR log-lik − pre-stroke mean\n(manifold distortion)")
    ax.set_title(f"Model {args.family}: behavioral-manifold distortion across recovery (by 3-tier severity)")
    ax.legend(title="severity")
    fig.tight_layout(); fig.savefig(out / "ar_loglik_biomarker.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: syllable usage by epoch (cohort mean) ---
    order = ["pre", "acute", "sub_acute", "chronic"]
    present = [e for e in order if (df["epoch"] == e).any()]
    M = np.vstack([df[df["epoch"] == e][[f"use_S{k}" for k in range(K)]].mean().to_numpy()
                   for e in present])
    fig, ax = plt.subplots(figsize=(max(8, K * 0.5), 3.5))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(present))); ax.set_yticklabels(present)
    ax.set_xticks(range(K)); ax.set_xticklabels([f"S{k}" for k in range(K)], fontsize=7)
    ax.set_xlabel("syllable"); ax.set_title(f"Model {args.family}: mean syllable usage by epoch")
    plt.colorbar(im, ax=ax, label="usage fraction")
    fig.tight_layout(); fig.savefig(out / "usage_by_epoch.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(df.groupby(["severity_3tier", "epoch"])["ar_loglik"].mean().round(2).to_string())
    print(f"\nwrote per_session_syllables.csv + 2 figures to {out}")


if __name__ == "__main__":
    main()
