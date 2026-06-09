"""Turn per-session feature tables into a model design: pooled, standardized, PCA.

Standardization and PCA are fit on the **pooled pre-stroke** data so the latent
coordinate system is shared across sessions/animals (post-stroke sessions are
later projected through the *same* transform, never refit, so changes are
measured against the pre-stroke reference frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Continuous observation columns (task event channels are inputs, excluded
# here). Two families: Model A drops the fr_c* columns (severe animals have no
# FaceRhythm); Model B uses all. The builder selects the subset that is present.
CONTINUOUS = [
    "fr_c0", "fr_c1", "fr_c2", "fr_c3", "fr_c4", "fr_c5", "fr_c6", "fr_c7", "fr_c8", "fr_c9",
    "tongue_x_mean", "tongue_y_mean", "tongue_out_frac", "tongue_motion_energy",
    "tongue_angle_mean", "tongue_angle_speed",
    "jaw_y_mean", "jaw_motion_energy", "treadmill_speed_mm_s", "treadmill_accel",
]


@dataclass
class Design:
    """Pooled PCA design + the transform needed to project new sessions."""

    sequences: list[np.ndarray]   # per-session (T_i, n_pca) PCA scores
    mu: np.ndarray                # feature mean (standardization)
    sd: np.ndarray                # feature std
    components: np.ndarray        # PCA components (n_pca, n_features), rows orthonormal
    z_mean: np.ndarray            # mean of standardized features (pre-PCA centering)
    var_ratio: np.ndarray
    columns: list[str]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project raw (T, n_features) features into the fitted PCA space.

        Standardize first, then impute missing values (NaN — e.g. tongue
        position when the tongue is absent) to 0 in standardized space, i.e. to
        the pre-stroke feature mean (neutral). This must happen AFTER
        standardization so the fill is the column mean, not a raw 0.
        """
        z = np.nan_to_num(np.clip((X - self.mu) / self.sd, -8, 8))
        return (z - self.z_mean) @ self.components.T


def build_design(
    session_features: list[np.ndarray],
    pca_var: float = 0.90,
    columns: list[str] | None = None,
) -> Design:
    """Fit standardization + PCA on pooled sessions; return per-session scores.

    Parameters
    ----------
    session_features : list of (T_i, n_features) arrays
        Raw continuous features per session, columns ordered as ``columns``.
    pca_var : float
        Cumulative variance to retain when choosing the number of components.
    columns : list[str], optional
        Names of the feature columns actually used, in array order. Defaults to
        the full :data:`CONTINUOUS` list; pass the present subset for Model A
        (which has no ``fr_c*`` columns) so ``Design.columns`` is accurate.
    """
    columns = columns or list(CONTINUOUS)
    # Keep NaN through the moment estimates so missing values (e.g. tongue
    # position when absent) do not bias the mean/std; impute to the column mean
    # only AFTER standardization (NaN -> 0 in standardized space).
    pooled = np.vstack(session_features)
    mu = np.nanmean(pooled, 0)
    sd = np.nanstd(pooled, 0) + 1e-9
    Z = np.nan_to_num(np.clip((pooled - mu) / sd, -8, 8))
    z_mean = Z.mean(0)
    Zc = Z - z_mean
    cov = (Zc.T @ Zc) / len(Zc)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    var = w / w.sum()
    n = int(np.searchsorted(np.cumsum(var), pca_var)) + 1
    comps = V[:, :n].T
    seqs = []
    for s in session_features:
        z = np.nan_to_num(np.clip((s - mu) / sd, -8, 8)) - z_mean
        seqs.append((z @ comps.T).astype(np.float32))
    return Design(seqs, mu, sd, comps, z_mean, var[:n], list(columns))
