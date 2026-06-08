"""Build the unified, time-aligned feature matrix the AR-HMM consumes.

Everything is resampled to the 12.5 Hz FaceRhythm latent clock (the coarsest
clock; DLC and task events are binned/placed onto it). The result is a tidy
per-session table of shape (n_latent, D) plus column-role metadata.

Three blocks:
  - facerhythm : consensus-basis loadings (one column per shared component).
  - dlc        : binned tongue/jaw summaries (position, occupancy, motion energy).
  - task_input : per-bin event counts (tone/lick per mouse-side), to be used as
                 inputs/covariates rather than AR observations if desired.

Values are stored RAW. Standardization (z-scoring) is deliberately deferred to
model-assembly time so the scale can be fit pooled across pre-stroke sessions
rather than per-session. Side columns are mouse-frame (see lr_convention); for
this left-VLS cohort, mouse-L = ipsilesional, mouse-R = contralesional.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from arhmm_behavior.facerhythm.io import TCAFactors
from arhmm_behavior.facerhythm.consensus import ConsensusBasis, project_lowrank
from arhmm_behavior.facerhythm.alignment import frame_to_latent_bins, bin_to_latent
from arhmm_behavior.dlc.kinematics import protrusion
from arhmm_behavior.task_events import ws_samples_to_latent

FS_HZ = 12.5


@dataclass
class FeatureMatrix:
    """Per-session feature table + metadata."""

    table: pd.DataFrame                    # (n_latent, 2 + D): latent_idx, session_time_s, features...
    roles: dict[str, str]                  # column -> {"facerhythm","dlc","task_input"}
    meta: dict = field(default_factory=dict)


def facerhythm_block(fac: TCAFactors, basis: ConsensusBasis) -> dict[str, np.ndarray]:
    """Consensus-basis loadings, one column per shared component."""
    load = project_lowrank(fac, basis)     # (K, n_latent)
    return {f"fr_c{k}": load[k] for k in range(load.shape[0])}


def _binned(values: np.ndarray, bins: np.ndarray, n_latent: int, *, kind: str) -> np.ndarray:
    """Per-latent-window mean / fraction / energy(std) of a per-frame signal."""
    cnt = np.bincount(bins, minlength=n_latent).clip(1)
    if kind == "frac":
        return np.bincount(bins, weights=values.astype(float), minlength=n_latent) / cnt
    mean = np.bincount(bins, weights=values, minlength=n_latent) / cnt
    if kind == "mean":
        return mean
    if kind == "energy":
        m2 = np.bincount(bins, weights=values * values, minlength=n_latent) / cnt
        return np.sqrt((m2 - mean * mean).clip(0))
    raise ValueError(kind)


def dlc_block(
    tongue: pd.DataFrame, jaw: pd.DataFrame, x_axis: np.ndarray, n_latent: int
) -> dict[str, np.ndarray]:
    """Binned tongue/jaw summaries on the latent clock.

    tongue_x_mean is signed lateral position (sign encodes side; for this cohort
    x>0 = mouse-L = ipsilesional). tongue_out_frac is occupancy (tongue visible).
    """
    n = len(tongue)
    bins = frame_to_latent_bins(x_axis, n, n_latent)
    t_present = (~tongue["is_baseline_fill"].to_numpy()).astype(float)
    out = {
        "tongue_x_mean": _binned(tongue["x_final"].to_numpy() * t_present, bins, n_latent, kind="mean"),
        "tongue_y_mean": _binned(tongue["y_final"].to_numpy() * t_present, bins, n_latent, kind="mean"),
        "tongue_out_frac": _binned(t_present, bins, n_latent, kind="frac"),
        "tongue_motion_energy": _binned(protrusion(tongue), bins, n_latent, kind="energy"),
        "jaw_y_mean": _binned(jaw["y_final"].to_numpy(), bins, n_latent, kind="mean"),
        "jaw_motion_energy": _binned(jaw["y_final"].to_numpy(), bins, n_latent, kind="energy"),
    }
    return out


def task_block(
    events: dict[str, np.ndarray],
    template: dict[str, np.ndarray],
    x_axis: np.ndarray,
    n_latent: int,
    keys: tuple[str, ...] = ("Tone_L", "Tone_R", "Lick_L", "Lick_R"),
) -> dict[str, np.ndarray]:
    """Per-bin event counts on the latent clock (mouse-frame event keys)."""
    out = {}
    for key in keys:
        chan = np.zeros(n_latent)
        idx = ws_samples_to_latent(events[key], template, x_axis, n_latent)
        np.add.at(chan, idx, 1.0)
        out[key.lower()] = chan
    return out


def assemble(
    fac: TCAFactors,
    basis: ConsensusBasis,
    tongue: pd.DataFrame,
    jaw: pd.DataFrame,
    events: dict[str, np.ndarray],
    template: dict[str, np.ndarray],
    x_axis: np.ndarray,
    meta: dict | None = None,
) -> FeatureMatrix:
    """Assemble one session's full feature matrix from already-loaded inputs."""
    n_latent = fac.time.shape[0]
    fr = facerhythm_block(fac, basis)
    dlc = dlc_block(tongue, jaw, x_axis, n_latent)
    task = task_block(events, template, x_axis, n_latent)

    roles = {**{k: "facerhythm" for k in fr},
             **{k: "dlc" for k in dlc},
             **{k: "task_input" for k in task}}
    cols = {**fr, **dlc, **task}
    table = pd.DataFrame(cols).astype(np.float32)
    table.insert(0, "session_time_s", np.arange(n_latent) / FS_HZ)
    table.insert(0, "latent_idx", np.arange(n_latent))
    return FeatureMatrix(table=table, roles=roles, meta={"fs_hz": FS_HZ, **(meta or {})})
