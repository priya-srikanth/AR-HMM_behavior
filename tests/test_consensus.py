"""Unit tests for the consensus-basis math (synthetic factors, no real data)."""
import numpy as np

from arhmm_behavior.facerhythm.io import TCAFactors
from arhmm_behavior.facerhythm.consensus import (
    build_consensus,
    match_to_reference,
    project_lowrank,
)


def _make_session(rng, shared_spatial, shared_freq, perm, n_latent=30):
    """A session whose factors are the shared atoms, column-permuted + scaled."""
    K = shared_spatial.shape[1]
    P = shared_spatial[:, perm] * rng.uniform(0.5, 2.0, K)
    F = shared_freq[:, perm] * rng.uniform(0.5, 2.0, K)
    T = rng.random((n_latent, K)).astype(np.float32)
    return TCAFactors(spatial=P.astype(np.float32), frequency=F.astype(np.float32), time=T)


def test_match_recovers_permutation():
    rng = np.random.default_rng(0)
    n_pts, n_freq, K = 5, 4, 3
    S = rng.random((2 * n_pts, K)); Fq = rng.random((n_freq, K))
    ref = TCAFactors(S.astype(np.float32), Fq.astype(np.float32), rng.random((10, K)).astype(np.float32))
    perm = np.array([2, 0, 1])
    sess = _make_session(rng, S, Fq, perm)
    order, sims = match_to_reference(sess, ref)
    # ref slot i should map to the session column holding ref's atom i, i.e. perm^-1
    inv = np.argsort(perm)
    assert np.array_equal(order, inv)
    assert np.all(sims > 0.99)


def test_build_consensus_reliability_high():
    rng = np.random.default_rng(1)
    n_pts, n_freq, K = 6, 5, 4
    S = rng.random((2 * n_pts, K)); Fq = rng.random((n_freq, K))
    ref = TCAFactors(S.astype(np.float32), Fq.astype(np.float32), rng.random((12, K)).astype(np.float32))
    sessions = [_make_session(rng, S, Fq, rng.permutation(K)) for _ in range(8)]
    basis = build_consensus(sessions, ref)
    assert basis.n_sessions == 8
    assert np.all(basis.reliability > 0.95)  # all sessions share the same atoms


def test_project_lowrank_recovers_reference_time():
    rng = np.random.default_rng(2)
    n_pts, n_freq, K = 5, 4, 3
    S = rng.random((2 * n_pts, K)); Fq = rng.random((n_freq, K))
    ref = TCAFactors(S.astype(np.float32), Fq.astype(np.float32), rng.random((20, K)).astype(np.float32))
    basis = build_consensus([ref], ref)
    loadings = project_lowrank(ref, basis)  # (K, n_latent)
    # Projecting the reference onto a basis built from itself recovers its time
    # factors up to per-component scale; check correlation per component.
    for k in range(K):
        r = np.corrcoef(loadings[k], ref.time[:, k])[0, 1]
        assert r > 0.99
