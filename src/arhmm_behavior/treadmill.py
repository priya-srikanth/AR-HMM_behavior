"""Load the upstream-preprocessed treadmill (locomotion) signal.

The stroke pipeline already calibrates and smooths the raw wavesurfer treadmill
voltage into speed (mm/s) and writes it per session as
``spout_behavior/treadmill_signals_preprocessed/<animal>/<date>.npz``
(``Treadmill_smoothed`` at the wavesurfer sample rate ``Fs``). We consume that
output rather than re-deriving it. Available cohort-wide (all PS46–55), so it
feeds both model families (FaceRhythm+DLC on PS46–50, and DLC-only across the
whole cohort). Aligned to the FaceRhythm latent clock by
:func:`arhmm_behavior.features.assemble.treadmill_block`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_treadmill_speed(path: str | Path) -> tuple[np.ndarray, float]:
    """Return ``(speed_mm_s, fs_hz)`` from a preprocessed treadmill NPZ.

    ``speed_mm_s`` is the smoothed, calibrated locomotion speed at the
    wavesurfer sample rate ``fs_hz`` (typically 1000 Hz).
    """
    d = np.load(path, allow_pickle=True)
    return d["Treadmill_smoothed"].astype(np.float64), float(d["Fs"])
