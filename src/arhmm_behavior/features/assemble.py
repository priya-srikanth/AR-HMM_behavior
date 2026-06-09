"""Build the unified, time-aligned feature matrix the AR-HMM consumes.

All signals are placed on one **model grid** at a configurable rate
(``configs/defaults.yaml`` ``model.sampling_rate_hz`` — e.g. 33 or 50 Hz),
defined on the video timeline. We do NOT use FaceRhythm's native 12.5 Hz clock
as the grid, because a tongue protrusion is only ~50–120 ms and 12.5 Hz (80 ms
bins) barely samples a lick (see docs/FINDINGS.md). On the grid:

  - DLC (250 fps) per-frame signals are **binned** (250/rate frames per bin).
  - wavesurfer events / treadmill (1 kHz) are **mapped** (ws → camera frame →
    grid bin) via the alignment template.
  - the native-12.5 Hz FaceRhythm latents are **interpolated** up onto the grid
    (smooth amplitude envelopes, so interpolation adds no spurious structure).

The result is a tidy per-session table of shape (n_grid, 2 + D) plus per-column
role metadata. Four blocks:

  - facerhythm : consensus-basis loadings, one column per shared component
                 (PS46–50 only — the severe animals have no FaceRhythm).
  - dlc        : tongue/jaw summaries — position (present-frames only), occupancy,
                 motion energy, and the eye→spout tongue ANGLE (mean + angular
                 speed) which carries within-lick kinematics.
  - treadmill  : locomotion speed (mm/s) + acceleration (cohort-wide).
  - task_input : tone/lick event counts per mouse-side — kept as COVARIATES, not
                 AR observations, so lateralized licking must emerge from
                 FaceRhythm + DLC rather than being read off the labels.

Values are stored RAW; standardization/PCA are deferred to ``model/design.py``
and fit on pooled pre-stroke data. Side columns are mouse-frame (see
lr_convention); for this left-VLS cohort mouse-L = ipsilesional, mouse-R =
contralesional.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from arhmm_behavior.facerhythm.io import TCAFactors
from arhmm_behavior.facerhythm.consensus import ConsensusBasis, project_lowrank
from arhmm_behavior.facerhythm.alignment import (
    frames_to_grid_bins,
    grid_center_frames,
    interp_to_grid,
    n_grid_bins,
)
from arhmm_behavior.dlc.kinematics import protrusion

#: Default model-grid rate (Hz). The canonical value lives in
#: configs/defaults.yaml (model.sampling_rate_hz); orchestration code should pass
#: it explicitly. This constant is only a fallback for ad-hoc calls.
DEFAULT_RATE_HZ = 33.0
VIDEO_FPS = 250.0


@dataclass
class FeatureMatrix:
    """Per-session feature table + metadata."""

    table: pd.DataFrame          # (n_grid, 2 + D): grid_idx, session_time_s, features...
    roles: dict[str, str]        # column -> {"facerhythm","dlc","treadmill","task_input"}
    meta: dict = field(default_factory=dict)


def facerhythm_block(
    fac: TCAFactors, basis: ConsensusBasis, x_axis: np.ndarray, center_frames: np.ndarray
) -> dict[str, np.ndarray]:
    """Consensus-basis loadings interpolated onto the model grid.

    ``project_lowrank`` gives the per-component loadings at FaceRhythm's native
    12.5 Hz sampling (one value per latent sample); latent sample ``i`` sits at
    video frame ``x_axis[i]``. We interpolate each component onto the grid-bin
    center frames so FaceRhythm shares the (faster) model clock with DLC.
    """
    load = project_lowrank(fac, basis)                  # (K, n_latent) @ 12.5 Hz
    grid = interp_to_grid(load, x_axis, center_frames)  # (K, n_grid)
    return {f"fr_c{k}": grid[k] for k in range(grid.shape[0])}


def _binned(values: np.ndarray, bins: np.ndarray, n_bins: int, *, kind: str) -> np.ndarray:
    """Per-bin mean / fraction / energy(std) of a per-frame signal."""
    cnt = np.bincount(bins, minlength=n_bins).clip(1)
    if kind == "frac":
        return np.bincount(bins, weights=values.astype(float), minlength=n_bins) / cnt
    mean = np.bincount(bins, weights=values, minlength=n_bins) / cnt
    if kind == "mean":
        return mean
    if kind == "energy":
        m2 = np.bincount(bins, weights=values * values, minlength=n_bins) / cnt
        return np.sqrt((m2 - mean * mean).clip(0))
    raise ValueError(kind)


def _binned_present_mean(
    values: np.ndarray, present: np.ndarray, bins: np.ndarray, n_bins: int
) -> np.ndarray:
    """Per-bin mean over PRESENT frames only; NaN where a bin has no present frame.

    Tongue position: when the tongue is retracted there is no real position, so
    absent frames are excluded (not zeroed, which would conflate "absent" with
    "midline"); a fully-absent bin is left NaN for neutral imputation at design
    time. Occupancy (``tongue_out_frac``) carries the absence separately.
    """
    w = present.astype(float)
    num = np.bincount(bins, weights=values.astype(float) * w, minlength=n_bins)
    den = np.bincount(bins, weights=w, minlength=n_bins)
    out = np.full(n_bins, np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def dlc_block(
    tongue: pd.DataFrame,
    jaw: pd.DataFrame,
    n_bins: int,
    *,
    rate_hz: float,
    video_fps: float = VIDEO_FPS,
    tongue_angle: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Binned tongue/jaw summaries on the model grid.

    ``tongue``/``jaw`` are per-video-frame cleaned traces; ``tongue_angle`` is the
    optional per-frame eye→spout signed angle (deg). Each frame is assigned to a
    grid bin (250 fps → rate). Position uses present-frames-only means; angle and
    angular speed capture within-lick kinematics (computed at native 250 fps,
    then binned).
    """
    frame_idx = (tongue["frame_idx"].to_numpy()
                 if "frame_idx" in tongue.columns else np.arange(len(tongue)))
    bins = frames_to_grid_bins(frame_idx, video_fps, rate_hz, n_bins)
    present = ~tongue["is_baseline_fill"].to_numpy()
    out = {
        "tongue_x_mean": _binned_present_mean(tongue["x_final"].to_numpy(), present, bins, n_bins),
        "tongue_y_mean": _binned_present_mean(tongue["y_final"].to_numpy(), present, bins, n_bins),
        "tongue_out_frac": _binned(present.astype(float), bins, n_bins, kind="frac"),
        "tongue_motion_energy": _binned(protrusion(tongue), bins, n_bins, kind="energy"),
        "jaw_y_mean": _binned(jaw["y_final"].to_numpy(), bins, n_bins, kind="mean"),
        "jaw_motion_energy": _binned(jaw["y_final"].to_numpy(), bins, n_bins, kind="energy"),
    }
    if tongue_angle is not None:
        # Eye→spout signed angle: continuous (~0 at rest, ± during lateral
        # protrusions). Per-bin mean angle + mean angular SPEED (|dθ/dt|, deg/s),
        # the latter capturing how fast the tongue sweeps within a lick.
        ang = np.nan_to_num(np.asarray(tongue_angle, dtype=float))
        ang = ang[:len(frame_idx)] if len(ang) >= len(frame_idx) else np.pad(ang, (0, len(frame_idx) - len(ang)))
        ang_speed = np.abs(np.gradient(ang)) * video_fps
        out["tongue_angle_mean"] = _binned(ang, bins, n_bins, kind="mean")
        out["tongue_angle_speed"] = _binned(ang_speed, bins, n_bins, kind="mean")
    return out


def treadmill_block(
    speed_ws: np.ndarray,
    template: dict[str, np.ndarray],
    n_bins: int,
    *,
    rate_hz: float,
    video_fps: float = VIDEO_FPS,
) -> dict[str, np.ndarray]:
    """Locomotion speed + acceleration on the model grid.

    ``speed_ws`` is the upstream-preprocessed ``Treadmill_smoothed`` trace (mm/s,
    wavesurfer clock). Each wavesurfer sample is mapped to a camera frame
    (``sig_camIdx__idx_ws``) then to a grid bin, and speed is averaged per bin.
    Acceleration is the per-bin change in speed (mm/s²). Bins with no wavesurfer
    coverage default to 0 (stationary).
    """
    cam_for_ws = np.asarray(template["sig_camIdx__idx_ws"], dtype=float)
    m = min(len(speed_ws), len(cam_for_ws))
    cam = cam_for_ws[:m]
    spd = np.asarray(speed_ws[:m], dtype=float)
    ok = np.isfinite(cam) & np.isfinite(spd)
    binned = frames_to_grid_bins(cam[ok], video_fps, rate_hz, n_bins)
    num = np.bincount(binned, weights=spd[ok], minlength=n_bins)
    den = np.bincount(binned, minlength=n_bins)
    speed = np.zeros(n_bins)
    nz = den > 0
    speed[nz] = num[nz] / den[nz]
    accel = np.gradient(speed) * rate_hz
    return {"treadmill_speed_mm_s": speed, "treadmill_accel": accel}


def task_block(
    events: dict[str, np.ndarray],
    template: dict[str, np.ndarray],
    n_bins: int,
    *,
    rate_hz: float,
    video_fps: float = VIDEO_FPS,
    keys: tuple[str, ...] = ("Tone_L", "Tone_R", "Lick_L", "Lick_R"),
) -> dict[str, np.ndarray]:
    """Per-bin event counts on the model grid (mouse-frame event keys).

    Each event's wavesurfer sample index is mapped to a camera frame, then to a
    grid bin, and counted.
    """
    cam_for_ws = np.asarray(template["sig_camIdx__idx_ws"], dtype=float)
    out = {}
    for key in keys:
        ev = np.asarray(events[key])
        ev = ev[(ev >= 0) & (ev < len(cam_for_ws))]
        cam = cam_for_ws[ev]
        cam = cam[np.isfinite(cam)]
        bins = frames_to_grid_bins(cam, video_fps, rate_hz, n_bins)
        chan = np.zeros(n_bins)
        np.add.at(chan, bins, 1.0)
        out[key.lower()] = chan
    return out


def assemble(
    tongue: pd.DataFrame,
    jaw: pd.DataFrame,
    events: dict[str, np.ndarray],
    template: dict[str, np.ndarray],
    *,
    fac: TCAFactors | None = None,
    basis: ConsensusBasis | None = None,
    x_axis: np.ndarray | None = None,
    rate_hz: float = DEFAULT_RATE_HZ,
    video_fps: float = VIDEO_FPS,
    treadmill_speed: np.ndarray | None = None,
    tongue_angle: np.ndarray | None = None,
    meta: dict | None = None,
) -> FeatureMatrix:
    """Assemble one session's feature matrix on the model grid.

    DLC + treadmill + task feed BOTH model families. FaceRhythm is **optional**:
    pass ``fac`` + ``basis`` + ``x_axis`` for **Model B** (FR+DLC, PS46–50); omit
    them for **Model A** (DLC-only, cohort-wide incl. the severe animals that
    have no FaceRhythm).

    Parameters
    ----------
    tongue, jaw :
        Per-video-frame cleaned DLC traces (``tongue`` also sets the session's
        video length / grid size).
    events, template :
        Mouse-frame wavesurfer event-sample arrays and the ws↔camera alignment
        template.
    fac, basis, x_axis :
        FaceRhythm factors, the consensus basis, and the VQT ``x_axis`` (video
        frame per FaceRhythm latent). Provide all three to include the (grid-
        interpolated) FaceRhythm block; leave them ``None`` for Model A.
    rate_hz :
        Model-grid sampling rate (Hz); from ``configs/defaults.yaml``
        ``model.sampling_rate_hz``.
    treadmill_speed :
        Optional preprocessed ``Treadmill_smoothed`` (mm/s), cohort-wide.
    tongue_angle :
        Optional per-frame eye→spout tongue angle (deg).
    """
    n_video_frames = int(len(tongue))
    n_bins = n_grid_bins(n_video_frames, video_fps, rate_hz)
    center_frames = grid_center_frames(n_bins, video_fps, rate_hz)

    fr = (facerhythm_block(fac, basis, x_axis, center_frames)
          if (fac is not None and basis is not None and x_axis is not None) else {})
    dlc = dlc_block(tongue, jaw, n_bins, rate_hz=rate_hz, video_fps=video_fps,
                    tongue_angle=tongue_angle)
    task = task_block(events, template, n_bins, rate_hz=rate_hz, video_fps=video_fps)
    tread = (treadmill_block(treadmill_speed, template, n_bins, rate_hz=rate_hz,
                             video_fps=video_fps)
             if treadmill_speed is not None else {})

    roles = {**{k: "facerhythm" for k in fr},
             **{k: "dlc" for k in dlc},
             **{k: "treadmill" for k in tread},
             **{k: "task_input" for k in task}}
    cols = {**fr, **dlc, **tread, **task}
    table = pd.DataFrame(cols).astype(np.float32)
    table.insert(0, "session_time_s", np.arange(n_bins) / rate_hz)
    table.insert(0, "grid_idx", np.arange(n_bins))
    return FeatureMatrix(table=table, roles=roles,
                         meta={"rate_hz": rate_hz, "video_fps": video_fps,
                               "n_grid": n_bins, **(meta or {})})
