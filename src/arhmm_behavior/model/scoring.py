"""Score and summarize AR-HMM syllable sequences.

Two uses:
  - **Model selection** (``scripts/fit_arhmm.py``): rather than likelihood alone,
    we judge a fit by whether its syllables carry behavior we did NOT feed the
    model as observations — the normalized mutual information between the
    syllable sequence and INDEPENDENT labels (lick side from the task channels,
    locomotion from the treadmill). The headline metric is **lick-side
    ipsi/contra MI**: an unsupervised model that genuinely carves licking by side
    scores high, one that doesn't scores ~0 (see FINDINGS F8).
  - **Downstream analysis** (``scripts/...syllable analysis``): per-syllable
    usage, dwell-time, and transition-matrix summaries to compare across epochs
    and severity groups.

Everything operates on a per-grid-bin integer syllable sequence ``z`` (length T)
aligned to a session's feature table (same model grid).
"""
from __future__ import annotations

import numpy as np


def normalized_mi(z: np.ndarray, labels: np.ndarray) -> float:
    """Normalized mutual information between syllables ``z`` and ``labels``.

    Both are integer arrays of equal length. Returns ``I(z;labels) /
    sqrt(H(z)·H(labels))`` (geometric-mean normalization) in [0, 1]; 0 when the
    syllables carry no information about the label, 1 when they determine it.
    Bins where ``labels < 0`` (e.g. "no event") are dropped.
    """
    z = np.asarray(z)
    labels = np.asarray(labels)
    m = labels >= 0
    z, labels = z[m], labels[m]
    if z.size == 0:
        return 0.0
    zc = np.unique(z, return_inverse=True)[1]
    lc = np.unique(labels, return_inverse=True)[1]
    joint = np.zeros((zc.max() + 1, lc.max() + 1))
    np.add.at(joint, (zc, lc), 1.0)
    joint /= joint.sum()
    pz = joint.sum(1, keepdims=True)
    pl = joint.sum(0, keepdims=True)

    def _H(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (pz @ pl)[nz])).sum())
    Hz, Hl = _H(pz), _H(pl)
    denom = np.sqrt(Hz * Hl)
    return mi / denom if denom > 0 else 0.0


def lick_side_labels(table, nlags: int = 0) -> np.ndarray:
    """Per-bin lick-side label from the task channels: 0 ipsi-L, 1 contra-R, -1 none.

    Uses the mouse-frame ``lick_l`` / ``lick_r`` event-count columns. For this
    left-VLS cohort mouse-L = ipsilesional and mouse-R = contralesional. Bins with
    no lick are -1 (dropped by :func:`normalized_mi`). ``nlags`` drops the leading
    frames that jax_moseq removes, so the label aligns to the state sequence.
    """
    l = table["lick_l"].to_numpy() > 0
    r = table["lick_r"].to_numpy() > 0
    lab = np.full(len(l), -1, dtype=int)
    lab[l] = 0
    lab[r] = 1
    lab[l & r] = -1  # ambiguous bin (both sides) — drop
    return lab[nlags:]


def locomotion_labels(table, nlags: int = 0, thresh_mm_s: float = 5.0) -> np.ndarray:
    """Per-bin running label: 1 if treadmill speed > ``thresh``, else 0."""
    spd = table["treadmill_speed_mm_s"].to_numpy()
    return (spd[nlags:] > thresh_mm_s).astype(int)


def durations_s(z: np.ndarray, rate_hz: float) -> np.ndarray:
    """Syllable run-length durations (seconds) for a 1-D state sequence."""
    z = np.asarray(z)
    ch = np.where(np.diff(z) != 0)[0]
    edges = np.concatenate([[-1], ch, [len(z) - 1]])
    return np.diff(edges) / rate_hz


def usage(z: np.ndarray, num_states: int) -> np.ndarray:
    """Fraction of frames spent in each syllable (length ``num_states``)."""
    return np.bincount(np.asarray(z), minlength=num_states) / len(z)


def transition_matrix(z: np.ndarray, num_states: int, drop_self: bool = True) -> np.ndarray:
    """Row-stochastic syllable transition matrix.

    With ``drop_self`` the diagonal (self-transitions, which dominate at high
    stickiness) is zeroed before row-normalizing, giving the *sequence* structure
    (which syllable follows which) rather than dwell.
    """
    z = np.asarray(z)
    T = np.zeros((num_states, num_states))
    np.add.at(T, (z[:-1], z[1:]), 1.0)
    if drop_self:
        np.fill_diagonal(T, 0.0)
    rs = T.sum(1, keepdims=True)
    return np.divide(T, rs, out=np.zeros_like(T), where=rs > 0)
