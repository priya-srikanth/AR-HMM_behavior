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
- Built by matching every session's components to a reference and averaging the
  matched spatial+frequency factors.
- **Iterated reference (canonical, 2026-06-08).** A single pass against the
  arbitrary PS46/0310 reference biases the basis toward that session. We now
  iterate (`build_consensus_iterated`): build a consensus, promote it to the new
  reference, re-match all sessions, repeat until component→slot assignments stop
  changing (converges in ~5 iterations). This tightened every reliability
  (mean cross-session match cosine) — all components now **≥ 0.74** vs the
  single-pass 0.62–0.93. Per-component reliabilities:
  c7 2.6 Hz 0.95, c2 (ipsi lick) 5.0 Hz 0.92, c9 3.6 Hz 0.92, c0 0.5 Hz 0.90,
  c4 0.7 Hz 0.89, c1 (whisk) 11.5 Hz 0.89, c8 1.9 Hz 0.86, c5 1.0 Hz 0.79,
  **c3 (contra lick) 5.9 Hz 0.77**, c6 1.1 Hz 0.74.
  (`figures/consensus_basis_factors_detail.png` — per-component spatial map +
  frequency tuning; `figures/consensus_basis_iterated_qc.png` — overview.)
- **All 10 components kept for now.** c6 (noisy/multimodal slow factor) and c5
  are the first prune candidates if they add noise to the syllables; c3 is the
  lowest-reliability *behaviorally meaningful* component (the contralesional lick,
  spatially more diffuse than ipsi c2) but is always kept — it is the channel
  predicted to drop then recover after stroke.
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

## F7. Unified feature matrix + baseline AR-HMM (roadmap 3–4)
- Per-session 12.5 Hz observation matrix assembled for all 33 pre-stroke sessions
  (PS46–50): 10 FR consensus loadings + 6 DLC tongue/jaw summaries + 4 task event
  channels + 2 treadmill (speed/accel). Code: `src/arhmm_behavior/features/assemble.py`,
  `model/design.py`. (`figures/feature_matrix_PS46_0310.png`)
- **Side mapping quadruple-confirmed**: (1) `origin/main` code chain (emitted `Tone_L`
  = mouse-L), (2) peri-tone response, (3) DLC tongue-x, (4) biology — PS46 contralesional
  (mouse-R) licks collapse 973→84→35 the two days after the L-VLS stroke then recover,
  ipsi preserved. So **c2 = ipsilesional, c3 = contralesional** is locked.
  (`figures/PS46_contra_lick_validation.png`)
- **Baseline AR-HMM** (dynamax `LinearAutoregressiveHMM`, K=16, AR lag 1; PCA to 11 dims
  / 90% var; fit on a 12-session spread, decoded all 33). States are interpretable from
  their z-scored feature signatures: **lick-ipsilesional** (high fr_c2 + tongue),
  **lick-contralesional** (high fr_c3 + tongue), **running** (high treadmill/jaw), and
  quiescent — i.e. the lateralized lick syllables emerge unsupervised.
  (`figures/arhmm_baseline_diagnostics.png`)
- **Caveats / next refinements**: median syllable ≈0.16 s is too short — the default
  transition prior isn't sticky; add a **sticky/HDP-HMM or duration prior** (MoSeq-style)
  for ~0.3–0.5 s syllables. Batched EM is memory-heavy ((sessions,T,K,K)); production
  fitting should use a GPU / keypoint-moseq, or stochastic EM. Standardization+PCA are
  fit on pre-stroke only so post-stroke is projected through the same frame.

## F8. Canonical engine (jax_moseq) + combined beats FaceRhythm-only
- dynamax MAP-EM over-segments (0.16 s) and its stickiness bifurcates (no smooth
  control). Switched to **jax_moseq** sticky HDP-AR-HMM (KPMS engine) on the PCA
  design: smooth kappa→duration (kappa 1e5→1e10 = median 0.32→0.88 s, no collapse).
  Locked **kappa=1e8, K=16, nlags=3 → ~0.72 s** syllables; lick sub-phases consolidate.
  (`figures/arhmm_moseq_clean.png`; per-syllable trajectories + crowd-movie-analog
  clips: `figures/arhmm_syllable_trajectories.png`, `figures/arhmm_syllable_clips.png`)
- **Combined (FR+DLC+treadmill) >> FaceRhythm-only**, scored by normalized MI of
  syllables vs INDEPENDENT (wavesurfer/treadmill) labels, matched ~0.7 s, K=16:
  running 0.66 vs 0.22; licking 0.52 vs 0.32; **lick-side ipsi/contra 0.36 vs 0.03**.
  FR alone *contains* side info (c2/c3) but an unsupervised AR-HMM on FR doesn't carve
  licking by side — adding DLC tongue makes lateralized lick syllables emerge. Since the
  stroke disrupts contralesional licking, the combined model is **necessary**, not just
  nicer. (`figures/arhmm_combined_vs_fronly.png`)
- Contra-lick syllable = **S0** (fr_c3 +2.3, tongue-out +2.2); ipsi = S3/S14; run = S10/9/13.

## F9. Post-stroke projection through the pre-stroke model (the payoff)
- All post-stroke sessions project cleanly through the pre-stroke standardize+PCA+model
  and decode (states-only, fixed pre-stroke params), so post-stroke behavior is read out
  in the pre-stroke state space.
- **PS46 (moderate):** acute phase (0315–0318) both lick syllables drop + running surges
  (model-based readout of disruption); recovery — contralesional-lick syllable (S0) climbs
  back and overshoots pre-stroke usage by 0324–0327, matching the raw lick-count overshoot
  (F6). (`figures/ps46_syllable_usage_across_stroke.png`)
- **Replication PS46–50 (72 sessions):** the 0313-stroke animals (PS46/47/48) show the
  strongest effect — ipsilesional-syllable usage drops, contra rebounds/overshoots, and a
  **manifold-distortion biomarker** (per-frame AR-dynamics log-lik vs pre-stroke baseline)
  drops post-stroke (PS48 −1.2, PS47 −0.75, PS46 −0.4 nats/frame) then recovers; PS49/PS50
  show smaller changes in the sampled windows. (`figures/stroke_replication_allanimals.png`)
- **New biomarker:** per-session AR-dynamics log-likelihood under the frozen pre-stroke
  model = a continuous, model-based index of how far behavior has departed the pre-stroke
  manifold — a candidate recovery readout complementary to the raw lick counts.
- Caveats: per-session usage of a *specific* syllable (S0) varies across individuals; the
  AR-LL manifold-distortion score is the more robust cross-animal metric. One model fit,
  sampled post-stroke timepoints, no severe (PS55-class) animal has FaceRhythm. Replicate
  with more fits/seeds and full timelines.

## F10. Lateralized-lick split recovered on the 33 Hz pipeline — at a re-scaled operating point
- **Clean unsupervised ipsi/contra split.** On the rebuilt 33 Hz Model B design a fit carves
  licking by side: large **ipsi-specific** syllables (state 8 = 5327 ipsi / 0 contra) and
  large **contra-specific** ones (states 0/13/10 ≈ 5800 contra / ~10 ipsi); **~98% of ipsi
  and ~99% of contra** lick bins fall in side-specific states. **lick-side MI = 0.578** over
  all 33 pre-stroke sessions — above the F8 benchmark (0.36), vs the prior stuck 0.007–0.064.
  Task channels are excluded, so the split is unsupervised.
- **Three coupled fixes were required:** (1) **retain the lateralized axis** —
  `build_design --side-fit=present` fits standardize+PCA on tongue-PRESENT bins so the sparse
  `tongue_x_mean` (NaN ~86% of bins) is not imputed-diluted away (retention 0.09→0.49), with
  `--side-weight` on tongue_x/tongue_angle/fr_c2/fr_c3; (2) **cut latent dim** to ~5 PCs
  (`--pca-var 0.70`) so the AR-emission likelihood stops swamping the transition prior;
  (3) **soften the emission** (`moseq S_0_scale=10`). Config: 5-dim present-fit, S0=10,
  kappa=1e6, nlags=3, K=16 → 14 states used.
- **Discrepancy with F8 — kappa no longer controls duration here.** F8 reported smooth
  kappa→duration (1e5→1e10 = 0.32→0.88 s, locked kappa=1e8/0.72 s). On the rebuilt 33 Hz
  pipeline that does NOT hold: kappa is **inert across 1e1–1e8** and 1e8 **collapses** to one
  state. The fit is emission-dominated, so duration is set by S0/dim (not kappa) and lands
  **short (~0.12 s)** with no 0.5–0.7 s plateau. F8's kappa=1e8/0.72 s point is invalid for
  the current pipeline; what changed the regime (rate / feature scale / dim) is **open**.
- **Seed-robust:** 6/6 seeds give **lick-side MI 0.526–0.580 (mean 0.559 ± 0.020), 14–16
  states, no collapse** — the split is not a lucky seed. Running MI stays low (~0.07).
- **Caveats:** short syllables (~0.12 s) — the *split* is robust to this but syllable-as-
  behavioral-unit is not yet at the ~0.5–0.7 s timescale; per-state cross-tab shown on 6
  sessions (the MI is over all 33).

## Implications for the model
Licking is represented in a distributed, lateralized, low-dimensional way in the
FaceRhythm facial-motion space, complementary to DLC tongue kinematics. This argues
for feeding the AR-HMM the *full* consensus subspace + DLC, rather than hand-picked
single "behavior channels", and makes left/right (→ ipsi/contra) lick state a
natural thing for the model to recover.
