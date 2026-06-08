"""Load FaceRhythm TCA factors and VQT metadata from the per-session HDF5 files.

TCA factor convention (rank-K, single-session dict element "0"):
    (xy points): (2*n_points, K)   spatial loading, xy-major: rows [x(n), y(n)]
    frequency:   (n_freq, K)
    time:        (n_latent, K)     the 12.5 Hz behavioral latent — the model input
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class TCAFactors:
    """Rank-K TCA factors for one session."""

    spatial: np.ndarray   # (2*n_points, K)
    frequency: np.ndarray  # (n_freq, K)
    time: np.ndarray       # (n_latent, K)

    @property
    def rank(self) -> int:
        return self.spatial.shape[1]

    @property
    def n_points(self) -> int:
        return self.spatial.shape[0] // 2

    def peak_freqs(self, frequencies: np.ndarray) -> np.ndarray:
        """Peak frequency (Hz) of each component."""
        return frequencies[np.argmax(self.frequency, axis=0)]


def load_tca_factors(path: str | Path, element: str = "0") -> TCAFactors:
    """Read TCA factors from a ``TCA.h5`` file."""
    with h5py.File(path, "r") as f:
        g = f["factors"][element]
        return TCAFactors(
            spatial=g["(xy points)"][:].astype(np.float32),
            frequency=g["frequency"][:].astype(np.float32),
            time=g["time"][:].astype(np.float32),
        )


def load_vqt_meta(path: str | Path, element: str = "0") -> dict[str, np.ndarray]:
    """Read the small VQT metadata needed for analysis/alignment.

    Returns ``frequencies`` (n_freq,), ``point_positions`` (2*n_points,), and
    ``x_axis`` (n_latent,) — the video-frame index of each latent sample. Only
    these small datasets are read; the multi-GB spectrogram is not loaded.
    """
    with h5py.File(path, "r") as f:
        return {
            "frequencies": f["frequencies"][:],
            "point_positions": f["point_positions"][:],
            "x_axis": f["x_axis"][element][:].astype(np.int64),
        }


def point_xy(point_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the flat ``point_positions`` (2*n_points,) into (x, y) per point.

    Stored xy-major: first half is one axis, second half the other. Returns the
    wider-range axis as ``x`` (image columns) and the other as ``y``.
    """
    n = point_positions.shape[0] // 2
    a, b = point_positions[:n], point_positions[n:]
    if np.ptp(a) >= np.ptp(b):
        return a, b
    return b, a
