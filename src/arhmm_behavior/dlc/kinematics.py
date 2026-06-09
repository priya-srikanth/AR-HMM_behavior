"""Load DLC cleaned traces and lick events, and derive tongue/jaw signals.

Cleaned-trace parquet columns: ``frame_idx, x_final, y_final, likelihood_raw,
is_interp_fill, is_baseline_fill, fill_method``. When a part is not visible the
trace is baseline-filled to (0, 0); ``is_baseline_fill`` flags those frames.

Side convention (matches the upstream ``lick_events`` ``side`` labels):
    L  <->  x_final > 0 ,   R  <->  x_final < 0   (image/tongue-deviation frame).
This is NOT the mouse-anatomical frame; translate via the upstream
``lr_convention`` before any ipsi-/contralesional interpretation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_cleaned_trace(path: str | Path) -> pd.DataFrame:
    """Load a cleaned tongue/jaw trace (one row per video frame)."""
    return pd.read_parquet(path)


def load_lick_events(path: str | Path) -> pd.DataFrame:
    """Load side-labeled DLC lick events (one row per detected lick)."""
    return pd.read_parquet(path)


def load_tongue_angle(path: str | Path) -> np.ndarray:
    """Per-frame eye→spout-referenced signed tongue angle (degrees), 250 fps.

    Computed upstream (``dlc_kinematics/_angle.py``, smoothed); one row per video
    frame, ~0 at rest and deviating ± during lateral protrusions. This is the
    biologically meaningful angle (matches the per-lick ``peak_angle_deg``); we
    use it as a within-lick kinematic feature at native rate, binned onto the
    model grid.
    """
    return pd.read_parquet(path)["angle_deg_smoothed"].to_numpy()


def protrusion(df: pd.DataFrame) -> np.ndarray:
    """Per-frame protrusion magnitude ``hypot(x, y)``; 0 where baseline-filled."""
    present = (~df["is_baseline_fill"].to_numpy()).astype(float)
    return np.hypot(df["x_final"].to_numpy(), df["y_final"].to_numpy()) * present


def side_split(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Split per-frame protrusion into L (x>0) and R (x<0) signals.

    Returns ``{"L": ..., "R": ..., "both": ...}`` per-frame arrays. Labels are in
    the image/tongue-x frame (see module docstring).
    """
    x = df["x_final"].to_numpy()
    present = ~df["is_baseline_fill"].to_numpy()
    prot = protrusion(df)
    return {
        "L": prot * ((x > 0) & present),
        "R": prot * ((x < 0) & present),
        "both": prot,
    }
