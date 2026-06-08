"""Build a FaceRhythm consensus basis shared across sessions/animals.

Per-session TCA is fit independently, so component *k* is not comparable across
sessions. We match each session's components to a reference (spatial⊗frequency
cosine + Hungarian assignment), average the matched factors into consensus
spatial/frequency atoms, and express any session in the shared basis by
projecting its (low-rank) spectrogram onto the consensus atoms.

Point correspondence across sessions/animals (the 1093-point registered grid)
is what makes the spatial averaging meaningful — verified for the core
components; slow/low-power components are less reliable (see docs/FINDINGS.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .io import TCAFactors


def component_atoms(spatial: np.ndarray, frequency: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Outer-product atoms vec(spatialₖ ⊗ frequencyₖ), shape (K, n_space*n_freq)."""
    K = spatial.shape[1]
    A = np.zeros((K, spatial.shape[0] * frequency.shape[0]), dtype=np.float32)
    for k in range(K):
        o = np.outer(spatial[:, k], frequency[:, k]).ravel()
        A[k] = o / (np.linalg.norm(o) + 1e-9) if normalize else o
    return A


def match_to_reference(
    fac: TCAFactors, ref: TCAFactors
) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian-match ``fac`` components to ``ref`` by spatial⊗freq cosine.

    Returns ``(order, sims)`` where ``order[i]`` is the index in ``fac`` matched
    to reference slot ``i``, and ``sims[i]`` the cosine similarity of that match.
    """
    Aref = component_atoms(ref.spatial, ref.frequency)
    A = component_atoms(fac.spatial, fac.frequency)
    S = Aref @ A.T
    rows, cols = linear_sum_assignment(-S)
    order = np.empty(ref.rank, dtype=int)
    sims = np.empty(ref.rank, dtype=float)
    for r, c in zip(rows, cols):
        order[r] = c
        sims[r] = S[r, c]
    return order, sims


@dataclass
class ConsensusBasis:
    """Shared spatial/frequency atoms with per-component reliability."""

    spatial: np.ndarray    # (2*n_points, K)
    frequency: np.ndarray  # (n_freq, K)
    reliability: np.ndarray  # (K,) mean cross-session match cosine
    n_sessions: int

    def peak_freqs(self, frequencies: np.ndarray) -> np.ndarray:
        return frequencies[np.argmax(self.frequency, axis=0)]


def build_consensus(sessions: list[TCAFactors], reference: TCAFactors) -> ConsensusBasis:
    """Average matched, unit-normalized factors across sessions into a basis."""
    K = reference.rank
    sumP = np.zeros_like(reference.spatial)
    sumF = np.zeros_like(reference.frequency)
    rel = np.zeros(K)
    for fac in sessions:
        order, sims = match_to_reference(fac, reference)
        for slot, c in enumerate(order):
            p = fac.spatial[:, c]
            f = fac.frequency[:, c]
            sumP[:, slot] += p / (np.linalg.norm(p) + 1e-9)
            sumF[:, slot] += f / (np.linalg.norm(f) + 1e-9)
            rel[slot] += sims[slot]
    n = len(sessions)
    Pc = sumP / n
    Fc = sumF / n
    Pc /= np.linalg.norm(Pc, axis=0, keepdims=True) + 1e-9
    Fc /= np.linalg.norm(Fc, axis=0, keepdims=True) + 1e-9
    return ConsensusBasis(spatial=Pc, frequency=Fc, reliability=rel / n, n_sessions=n)


def project_lowrank(fac: TCAFactors, basis: ConsensusBasis) -> np.ndarray:
    """Loadings of a session in the consensus basis, via its own rank-K TCA.

    A session's spectrogram is well-approximated by its rank-K TCA reconstruction
    ``X ≈ Σ_j (spatialⱼ ⊗ freqⱼ) timeⱼ``. Projecting that reconstruction onto the
    consensus atoms is exact for the low-rank approximation and needs only the
    small factor files (no multi-GB VQT read). Returns ``(K, n_latent)``.
    """
    D = component_atoms(basis.spatial, basis.frequency).T  # (n_space*n_freq, K)
    # Unnormalized session atoms preserve loading amplitude.
    nsf = fac.spatial.shape[0] * fac.frequency.shape[0]
    B = np.empty((nsf, fac.rank), dtype=np.float32)
    for j in range(fac.rank):
        B[:, j] = np.outer(fac.spatial[:, j], fac.frequency[:, j]).ravel()
    G = np.linalg.pinv(D) @ B          # (K, rank): session atoms in consensus basis
    return G @ fac.time.T              # (K, n_latent)
