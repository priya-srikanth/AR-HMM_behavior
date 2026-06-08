"""Canonical AR-HMM: the jax_moseq sticky HDP-AR-HMM (the Keypoint-MoSeq engine).

Run directly on our PCA design (``model.design.build_design``). Unlike dynamax's
MAP-EM, the Gibbs sampler gives **smooth, monotonic** control of syllable duration
via the stickiness ``kappa`` (scan kappa to a target median duration — the KPMS
workflow). jax_moseq is an optional heavy dep (``pip install jax-moseq``); imported
lazily so the rest of the package works without it.

Locked working config (see docs/DECISIONS.md): kappa=1e8, num_states=16, nlags=3
→ ~0.7 s median syllables. Note jax_moseq drops the first ``nlags`` frames, so the
returned state sequence has length ``T - nlags``.
"""
from __future__ import annotations

import numpy as np

FS_HZ = 12.5


def default_hypparams(latent_dim: int, num_states: int = 16, kappa: float = 1e8, nlags: int = 3):
    """MoSeq-style transition + AR hyperparameters."""
    trans = {"num_states": num_states, "alpha": 5.7, "gamma": 1e3, "kappa": float(kappa)}
    ar = {"latent_dim": latent_dim, "nlags": nlags, "S_0_scale": 0.01, "K_0_scale": 10.0}
    return trans, ar


def make_data(sequences: list[np.ndarray], truncate: bool = True):
    """Stack PCA sequences into jax_moseq's (N, T, D) data dict with a mask."""
    import jax.numpy as jnp
    Lmin = min(s.shape[0] for s in sequences)
    if truncate:
        x = np.stack([s[:Lmin] for s in sequences]).astype(np.float32)
        mask = np.ones((len(sequences), Lmin), np.float32)
    else:
        Lmax = max(s.shape[0] for s in sequences)
        x = np.zeros((len(sequences), Lmax, sequences[0].shape[1]), np.float32)
        mask = np.zeros((len(sequences), Lmax), np.float32)
        for i, s in enumerate(sequences):
            x[i, : len(s)] = s
            mask[i, : len(s)] = 1.0
    return {"x": jnp.asarray(x), "mask": jnp.asarray(mask)}


def fit(data, num_states: int = 16, kappa: float = 1e8, nlags: int = 3,
        num_iters: int = 50, seed: int = 0):
    """Fit the Gibbs AR-HMM; returns (model_dict, state_seq (N, T-nlags))."""
    import jax.random as jr
    from jax_moseq.models import arhmm
    P = int(data["x"].shape[-1])
    trans, ar = default_hypparams(P, num_states, kappa, nlags)
    model = arhmm.init_model(data=data, trans_hypparams=trans, ar_hypparams=ar, seed=jr.PRNGKey(seed))
    for _ in range(num_iters):
        model = arhmm.resample_model(data, **model)
    return model, np.asarray(model["states"]["z"])


def decode(data, model, num_passes: int = 5):
    """Decode states for new sessions using fitted params (states-only resampling)."""
    import jax.random as jr
    from jax_moseq.models import arhmm
    states = arhmm.init_states(jr.PRNGKey(0), data["x"], data["mask"], model["params"])
    m = {"seed": jr.PRNGKey(1), "states": states, "params": model["params"], "hypparams": model["hypparams"]}
    for _ in range(num_passes):
        m = arhmm.resample_model(data, **m, states_only=True)
    return np.asarray(m["states"]["z"])


def syllable_durations(state_seq: np.ndarray) -> np.ndarray:
    """Pooled run-length durations (frames) across rows of a (N, T) state array."""
    out = []
    for s in np.atleast_2d(state_seq):
        ch = np.where(np.diff(s) != 0)[0]
        b = np.concatenate([[-1], ch, [len(s) - 1]])
        out.append(np.diff(b))
    return np.concatenate(out)
