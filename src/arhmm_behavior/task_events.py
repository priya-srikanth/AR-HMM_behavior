"""Place wavesurfer task events (tone/reward/lick) onto the FaceRhythm clock.

Clock chain:  wavesurfer sample  --(alignment template)-->  camera frame
              camera frame       --(VQT x_axis)-->          FaceRhythm latent

The upstream pipeline computes the wavesurfer↔camera alignment template
(``alignment_templates/<cam>/<animal>/<date>.npz``); we consume it rather than
re-deriving. The ``licks_and_rewards`` npz is emitted in the **mouse frame**
(the upstream writer applies ``lr_convention.to_mouse_frame`` before saving), so
its ``Tone_L``/``Tone_R``/``Reward_*``/``Lick_*`` keys are mouse-anatomical.

CAVEAT (upstream phase-8a.10.1, fix b86acfa): the rig mislabels Tone↔Reward
channels cohort-wide — ``Reward_*`` carries the ~200 ms tone TTL and ``Tone_*``
the ~26 ms reward pulse; the fix swaps them at read time. In this Pavlovian
design tone and reward coincide to the sample, so event *times* are unaffected,
but any analysis that must distinguish tone- from reward-onset requires
post-fix (regenerated) outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_alignment_template(path: str | Path) -> dict[str, np.ndarray]:
    """Load the wavesurfer↔camera alignment template npz."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def ws_samples_to_latent(
    ws_idx: np.ndarray, template: dict[str, np.ndarray], x_axis: np.ndarray, n_latent: int
) -> np.ndarray:
    """Map wavesurfer sample indices to FaceRhythm latent indices.

    Uses ``sig_camIdx__idx_ws`` (camera frame per WS sample) then the VQT
    ``x_axis`` (video frame per latent sample). Out-of-range / NaN samples are
    dropped.
    """
    cam_for_ws = template["sig_camIdx__idx_ws"]
    ws_idx = np.asarray(ws_idx)
    ws_idx = ws_idx[(ws_idx >= 0) & (ws_idx < len(cam_for_ws))]
    cam = cam_for_ws[ws_idx]
    cam = cam[np.isfinite(cam)]
    return np.clip(np.searchsorted(x_axis, cam.astype(int)), 0, n_latent - 1)


def peri_event(
    signal: np.ndarray, event_latent_idx: np.ndarray, pre: int, post: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Event-triggered mean ± SEM of a latent signal.

    Returns ``(mean, sem, n)`` over windows ``[-pre, +post)`` (in latent samples)
    that fit entirely within ``signal``.
    """
    segs = [
        signal[e - pre : e + post]
        for e in event_latent_idx
        if e - pre >= 0 and e + post < len(signal)
    ]
    segs = np.asarray(segs)
    if len(segs) == 0:
        return np.zeros(pre + post), np.zeros(pre + post), 0
    return segs.mean(0), segs.std(0) / np.sqrt(len(segs)), len(segs)
