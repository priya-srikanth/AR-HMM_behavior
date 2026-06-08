"""Fit and decode a baseline AR-HMM over the PCA design (via dynamax).

dynamax (JAX) is an optional, heavy dependency — install with ``pip install
".[model]"``. It is imported lazily inside the functions so the rest of the
package works without it.

Notes / known limitations of this baseline (see docs/FINDINGS.md):
  - The default transition prior is NOT sticky, so syllables are short
    (~2 frames). A sticky / HDP-HMM prior (MoSeq-style) or a duration model is
    the obvious next refinement to get behaviorally-plausible ~0.3–0.5 s syllables.
  - Batched EM materializes a (sessions, T, K, K) tensor; on limited RAM, fit on
    a spread subset of sessions and decode the rest (low-memory, per-session).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ARHMMFit:
    params: object
    states: list[np.ndarray]      # decoded state sequence per session
    usage: np.ndarray             # fraction of time per state
    state_feature_means: np.ndarray  # (K, n_features) z-scored signature
    log_likelihoods: np.ndarray


def fit_arhmm(
    fit_sequences: list[np.ndarray],
    num_states: int = 16,
    num_lags: int = 1,
    num_iters: int = 60,
    seed: int = 0,
):
    """Fit a LinearAutoregressiveHMM on equal-length sequences (truncate first)."""
    import jax, jax.numpy as jnp, jax.random as jr
    from dynamax.hidden_markov_model import LinearAutoregressiveHMM

    jax.config.update("jax_platform_name", "cpu")
    P = fit_sequences[0].shape[1]
    Lmin = min(s.shape[0] for s in fit_sequences)
    EM = np.stack([s[:Lmin] for s in fit_sequences]).astype(np.float32)
    model = LinearAutoregressiveHMM(num_states, P, num_lags=num_lags)
    inp = np.stack([np.asarray(model.compute_inputs(jnp.asarray(e))) for e in EM])
    params, props = model.initialize(key=jr.PRNGKey(seed), method="prior")
    params, lls = model.fit_em(
        params, props, jnp.asarray(EM), inputs=jnp.asarray(inp),
        num_iters=num_iters, verbose=False,
    )
    return model, params, np.asarray(lls)


def decode(model, params, sequences: list[np.ndarray]) -> list[np.ndarray]:
    """Most-likely states (smoothed-posterior argmax) per session.

    Decode at a common length to avoid per-length JIT recompiles; the smoother
    is low-memory per sequence (unlike batched EM).
    """
    import jax, jax.numpy as jnp
    Lmin = min(s.shape[0] for s in sequences)
    fn = jax.jit(lambda e, i: jnp.argmax(model.smoother(params, e, inputs=i).smoothed_probs, axis=-1))
    out = []
    for s in sequences:
        e = jnp.asarray(s[:Lmin].astype(np.float32))
        i = model.compute_inputs(e)
        out.append(np.asarray(fn(e, i)).astype(np.int16))
    return out


def state_diagnostics(states: list[np.ndarray], K: int) -> dict:
    """State usage and pooled syllable-duration run lengths (in frames)."""
    allst = np.concatenate([np.asarray(s) for s in states])
    usage = np.bincount(allst, minlength=K) / allst.size

    def runs(seq):
        ch = np.where(np.diff(seq) != 0)[0]
        b = np.concatenate([[-1], ch, [len(seq) - 1]])
        return np.diff(b)

    dur = np.concatenate([runs(np.asarray(s)) for s in states])
    return {"usage": usage, "durations": dur}
