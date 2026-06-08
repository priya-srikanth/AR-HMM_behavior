"""Align per-frame (250 fps video-clock) signals to the 12.5 Hz FaceRhythm clock.

FaceRhythm and DLC share the identical video clock; the VQT ``x_axis`` gives the
video-frame index of each latent sample. We bin a per-frame signal into the
latent windows it spans and summarize each window (movement *energy* = std, and
mean level).
"""
from __future__ import annotations

import numpy as np


def frame_to_latent_bins(x_axis: np.ndarray, n_frames: int, n_latent: int) -> np.ndarray:
    """Assign each video frame index to its latent-sample bin via ``x_axis``."""
    return np.clip(np.searchsorted(x_axis, np.arange(n_frames)), 0, n_latent - 1)


def bin_to_latent(
    signal: np.ndarray, bin_idx: np.ndarray, n_latent: int
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize a per-frame ``signal`` into latent windows.

    Returns ``(energy, mean)`` per latent sample, where ``energy`` is the
    within-window standard deviation (movement) and ``mean`` the average level.
    """
    s1 = np.bincount(bin_idx, weights=signal, minlength=n_latent)
    s2 = np.bincount(bin_idx, weights=signal * signal, minlength=n_latent)
    cnt = np.bincount(bin_idx, minlength=n_latent).clip(1)
    mean = s1 / cnt
    var = (s2 / cnt) - mean**2
    return np.sqrt(var.clip(0)), mean


def motion_energy(
    signal: np.ndarray, x_axis: np.ndarray, n_latent: int
) -> np.ndarray:
    """Convenience: per-latent-window movement energy of a per-frame signal."""
    bins = frame_to_latent_bins(x_axis, len(signal), n_latent)
    energy, _ = bin_to_latent(signal, bins, n_latent)
    return energy
