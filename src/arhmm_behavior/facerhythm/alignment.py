"""Align signals onto a common model clock.

Two clocks matter. FaceRhythm and DLC share the **video clock** (250 fps); the
VQT ``x_axis`` gives the video-frame index of each FaceRhythm latent sample.

Originally everything was binned onto FaceRhythm's native 12.5 Hz latent clock
(``frame_to_latent_bins`` below). We now instead build an explicit **model grid**
at a configurable rate (``configs/defaults.yaml`` ``model.sampling_rate_hz``,
e.g. 33 or 50 Hz) defined on the video timeline, because 12.5 Hz is too coarse
to resolve a ~50–120 ms tongue protrusion (see docs/FINDINGS.md). Onto this grid:
DLC per-frame signals are binned (250 fps → grid), wavesurfer events/treadmill
are mapped (ws → camera frame → grid), and the 12.5 Hz FaceRhythm latents are
*interpolated* up (``interp_to_grid``).
"""
from __future__ import annotations

import numpy as np


def frame_to_latent_bins(x_axis: np.ndarray, n_frames: int, n_latent: int) -> np.ndarray:
    """Assign each video frame index to its latent-sample bin via ``x_axis``.

    Legacy helper for the native-12.5 Hz FaceRhythm clock. New code uses the
    explicit model grid (:func:`frames_to_grid_bins`) instead.
    """
    return np.clip(np.searchsorted(x_axis, np.arange(n_frames)), 0, n_latent - 1)


def n_grid_bins(n_video_frames: int, video_fps: float, rate_hz: float) -> int:
    """Number of model-grid bins spanning a session of ``n_video_frames``."""
    return int(np.ceil(n_video_frames * rate_hz / video_fps))


def frames_to_grid_bins(
    frame_idx: np.ndarray, video_fps: float, rate_hz: float, n_bins: int
) -> np.ndarray:
    """Map video-frame indices to model-grid bins.

    Bin width is ``video_fps / rate_hz`` frames (e.g. 250/50 = 5 frames at
    50 Hz). Used both for DLC frames and, after ws→camera mapping, for
    wavesurfer-derived signals.
    """
    bins = np.floor(np.asarray(frame_idx) * rate_hz / video_fps).astype(int)
    return np.clip(bins, 0, n_bins - 1)


def grid_center_frames(n_bins: int, video_fps: float, rate_hz: float) -> np.ndarray:
    """Video-frame index at the center of each model-grid bin (for interpolation)."""
    width = video_fps / rate_hz
    return (np.arange(n_bins) + 0.5) * width


def interp_to_grid(
    values: np.ndarray, src_frames: np.ndarray, center_frames: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a signal sampled at ``src_frames`` onto grid centers.

    ``values`` is ``(T,)`` or ``(K, T)`` sampled at video-frame positions
    ``src_frames`` (e.g. the FaceRhythm latents at the VQT ``x_axis`` frames);
    returns the signal resampled at ``center_frames`` (the model-grid bin
    centers). FaceRhythm latents are smooth amplitude envelopes, so linear
    interpolation introduces no spurious structure. Endpoints are held constant.
    """
    src = np.asarray(src_frames, dtype=float)
    if values.ndim == 1:
        return np.interp(center_frames, src, values)
    return np.vstack([np.interp(center_frames, src, row) for row in values])


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
