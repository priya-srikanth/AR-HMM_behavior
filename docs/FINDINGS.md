# FINDINGS — results established so far

Working example session unless noted: **PS46 / 2025-03-10** (pre-stroke). Figures
in `figures/`. All correlations use motion *energy* (per-latent-window std) so
envelope-vs-envelope comparisons are valid.

## F1. FaceRhythm output characterization
- `run_20250520` runs the full pipeline through TCA (not just spectrograms).
  Layout: `cam{2,4}/<animal>/<date>/jobNum_0/analysis_files/`.
- Per session: `VQT_Analyzer.h5` spectrogram tensor `(2 xy, 1093 points, 30 freq,
  N_time)`; `TCA.h5` rank-10 factors `(xy points 2186×10, frequency 30×10,
  time N×10)`. Latent rate **12.5 Hz**.
- **TCA is run on the whole continuous session** (`idx_windows = None`; time-factor
  length = video frames ÷ 20), not on trial subsets.
- A few sessions are missing `TCA.h5` (e.g. PS46 0311, 0314) — failed jobs.

## F2. Components reproduce → a shared basis is well-defined
- Per-session rank-10 components reproduce across PS46 pre-stroke sessions with
  cosine ~0.95; lick stays 5–6 Hz, whisk 11.5–13.6 Hz every session.
  (`figures/FaceRhythm_xsession_consistency_PS46.png`)
- **Cross-animal** (PS46 ref vs PS47–50): frequency match is high cohort-wide;
  the core components also match *spatially* (lick 0.84–0.93, jaw 0.88–0.97,
  whisk 0.73–0.88). Slow/low-power components match in frequency but not space.
  All animals share the same 1093-point registered grid.
  (`figures/FaceRhythm_crossanimal_alignment.png`)

## F3. Consensus basis (33 pre-stroke sessions, PS46–50)
- Built by matching every session to the PS46/0310 reference and averaging
  matched spatial+frequency factors. Reliability per slot 0.62–0.93 (jaw 0.93,
  lick 0.89, whisk 0.86 highest; slow comps lowest).
  (`figures/FaceRhythm_consensus_basis.png`)
- **Validation**: projecting PS46/0310 onto the consensus basis gives a lick-slot
  loading that tracks the tongue at r = 0.58 — slightly better than the
  per-session fit (0.56). Switching to shared components loses nothing.

## F4. FaceRhythm vs DLC kinematics
- Mid-frequency components track tongue + jaw motion energy; high-freq is the
  whisk candidate; slow components anti-correlate (postural/quiet).
  (`figures/FaceRhythm_vs_DLC_PS46_20250310.png`)
- **FaceRhythm and DLC are non-redundant.** FaceRhythm never sees the tongue (it's
  facial/perioral optic flow), so its lick components dissociate from tongue
  kinematics moment-to-moment even where they agree at the bout level. (Alignment
  verified clean: FR-vs-tongue/jaw lag cross-correlation peaks at ~0 s.)
  (`figures/diag_FR_jaw_lag_PS46_0310.png`)

## F5. Licking is distributed across factors — and splits by SIDE
- Two factors carry licking (c2 ~5.0 Hz, c3 ~5.9 Hz). Together they explain
  R² ≈ 0.56 of tongue motion energy vs 0.33 for the best single factor; c9 (4.3 Hz)
  is *in-band but anti-correlated* — frequency tuning alone is misleading, you must
  align to kinematics. (`figures/FaceRhythm_multilick_PS46_0310.png`)
- **c2 and c3 separate by lick side** (the key result): splitting tongue protrusions
  by x-sign, c2 ↔ x>0 licks (r = +0.71, ~0 with the other side) and c3 ↔ x<0 licks
  (r = +0.78). The two dominant lick factors are **left- and right-lick** facial
  representations. (`figures/FaceRhythm_c2c3_leftright_PS46_0310.png`)
- **Caveat**: "left/right" here is the image/tongue-x frame. Mouse-anatomical side
  (and ipsi/contralesional after stroke) requires `lr_convention.to_mouse_frame`
  (`wavesurfer_sides_match_mouse_sides = false` for this cohort). Not yet applied.

## F6. Task events on the FR clock → mouse-frame + ipsi/contra resolved
- Wavesurfer events placed on the FR clock via: WS sample → camera frame (upstream
  `alignment_templates/<cam>/<animal>/<date>.npz`, `sig_camIdx__idx_ws`) → FR latent
  (VQT `x_axis`). Validated by clean side-specific peri-tone responses.
  (`figures/FaceRhythm_peritone_c2c3_PS46_0310.png`)
- The `licks_and_rewards` npz is mouse-frame (upstream `_translate_side_keys` applies
  `to_mouse_frame` before save). Peri-event: c2 driven by side-L cue (loading 24.6 vs
  4.0), c3 by side-R (21.5 vs 0.7). With F5 (c2↔x>0, c3↔x<0): **c2 = mouse-LEFT,
  c3 = mouse-RIGHT lick.**
- Lesion side = `stroke_laterality` field in upstream `animals.yaml` (phase-8a.10),
  `"L"` for the whole cohort. Left lesion → contralesional = mouse R. Therefore
  **c2 = ipsilesional, c3 = CONTRALESIONAL** — c3 is the lick representation predicted
  to drop then recover post-stroke.
- **Code-currency caveat (verified)**: the upstream rig mislabels Tone↔Reward channels
  cohort-wide (fix `b86acfa`); `Reward_*` is the tone TTL, `Tone_*` the reward pulse.
  This does NOT affect the side result (swap is orthogonal to L/R, and Tone/Reward are
  bit-identical in the npz so onset times are unambiguous; result also independently
  confirmed by DLC F5). It WILL matter for any tone-vs-reward timing analysis — use
  post-fix code + regenerated outputs there.

## Implications for the model
Licking is represented in a distributed, lateralized, low-dimensional way in the
FaceRhythm facial-motion space, complementary to DLC tongue kinematics. This argues
for feeding the AR-HMM the *full* consensus subspace + DLC, rather than hand-picked
single "behavior channels", and makes left/right (→ ipsi/contra) lick state a
natural thing for the model to recover.
