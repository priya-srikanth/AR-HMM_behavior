# DECISIONS — modeling iterations and choices

Chronological log of the analysis/modeling decisions for the AR-HMM, with
rationale, so the path is reproducible and reversible. Newest entries at the top.
Companion to `docs/FINDINGS.md` (results) and `docs/DESIGN.md` (plan).

## 2026-06-08 — AR-HMM engine: jax_moseq Gibbs over dynamax MAP-EM

**Decision:** Use **jax_moseq**'s sticky HDP-AR-HMM (the engine inside
Keypoint-MoSeq) as the canonical model, run directly on our multimodal PCA design.
Lock **kappa = 1e8, num_states = 16, nlags = 3** (alpha 5.7, gamma 1e3,
S_0_scale 0.01, K_0_scale 10.0) as the working configuration.

**Why:** Syllable duration must be controllable to a behaviorally-plausible
timescale (target 0.5–5 s; classic MoSeq mouse syllables ~0.3–0.5 s).
- `dynamax` `LinearAutoregressiveHMM` (MAP-EM) over-segments at default
  (median 0.16 s) and its stickiness prior **bifurcates**: kappa ≤300 → ~0.24 s;
  kappa ≥~1500 → collapse to a single state. k-means init only lifts the ceiling
  to ~0.32 s. No smooth 0.5–5 s regime. → rejected as the production engine.
- `jax_moseq` Gibbs sampler gives **smooth, monotonic** kappa→duration with no
  collapse: kappa 1e5→1e10 gives median 0.32→0.88 s, all 16 states retained.
  This is exactly the KPMS workflow (scan kappa to a target duration).
- kappa = 1e8 → median 0.72 s / mean 1.04 s, bout-level, and it **consolidates**
  the lick sub-phases (the dynamax model split ipsi-licking into S2↔S8 limit-cycle
  half-phases; jax_moseq merges them).

**Status:** kappa locked at 1e8 for now; revisit upward (≤~1e11) if coarser
multi-second bouts are wanted. Fit currently on a 10-session subset; decoding all
33 pre-stroke sessions is the next step. dynamax kept only as a fast sanity baseline.

## 2026-06-08 — Post-stroke projection + manifold-distortion biomarker

**Method:** Freeze the pre-stroke pipeline (standardize + PCA + jax_moseq params),
project post-stroke sessions through the *same* transform, decode states-only with
fixed params. Reads post-stroke behavior in the pre-stroke state space. Replicated
across PS46–50 (72 sessions, sampled acute→recovery), aligned to each animal's
`stroke_date`.

**New metric (decision):** adopt **per-session per-frame AR-dynamics log-likelihood
under the frozen pre-stroke model** as a continuous *manifold-distortion / recovery*
biomarker (lower = behavior further from the pre-stroke manifold). Complementary to,
and more robust cross-animal than, the usage of any single syllable.

**Results:** PS46 — acute drop of both lick syllables + running surge, then S0
(contra-lick) rebound/overshoot in recovery (matches raw lick-count overshoot, F6).
0313-stroke animals (PS46/47/48) show the clearest manifold-distortion dip
(PS48 −1.2, PS47 −0.75, PS46 −0.4 nats/frame) with recovery; PS49/50 smaller in the
sampled windows. Figures: `ps46_syllable_usage_across_stroke.png`,
`stroke_replication_allanimals.png`. Per-session table:
`data_local/stroke_timeline_allanimals.csv`.

**Caveats:** single model fit/seed; specific-syllable usage varies across individuals
(use the AR-LL metric for cross-animal claims); sampled (not full) post-stroke
timelines; no severe (PS55-class) animal has FaceRhythm in run_20250520.

## 2026-06-08 — Combined features beat FaceRhythm-only (justifies fusion)

**Result:** AR-HMM on combined (FR+DLC+treadmill) vs FaceRhythm-only, both
jax_moseq, K=16, matched ~0.7 s syllables, scored by normalized MI between
syllables and **independent** behavior labels (wavesurfer licks, treadmill —
inputs to neither model):

| behavior | combined | FR-only |
|---|---|---|
| running (treadmill) | 0.66 | 0.22 |
| licking (wavesurfer) | 0.52 | 0.32 |
| lick-side ipsi/contra (wavesurfer) | **0.36** | **0.03** |

**Takeaway:** Combined wins everywhere; the dramatic gap is **lick laterality** —
FaceRhythm-only syllables barely distinguish which side the mouse licks (0.03),
even though FR *contains* side info (c2/c3). An unsupervised AR-HMM on FR alone
doesn't carve licking by side (the L/R contrast is a low-variance direction);
adding DLC tongue-x makes lateralized lick syllables emerge. Since the stroke
specifically disrupts contralesional licking, the combined model is **necessary**
(not just nicer) to represent the behavioral axis the stroke acts on. FR also
can't see locomotion (no body/treadmill), so the combined model is needed there too.
Figure: `figures/arhmm_combined_vs_fronly.png`.

## 2026-06-08 — Feature design for the observation matrix

**Decision:** Per-session 12.5 Hz matrix = **10 FR consensus loadings + 6 DLC
tongue/jaw summaries + 2 treadmill (speed, accel)** as continuous AR observations;
**4 task event channels (tone/lick × L/R) kept as inputs/covariates**, not AR
observations. Standardize + PCA (→11 dims, 90% var) fit **pooled on pre-stroke
only**, so post-stroke is projected through the same frame.

**Why:** FR is facial optic-flow (no tongue); DLC adds tongue/jaw kinematics; the
treadmill adds locomotion FR captures weakly — complementary. Task events are
exogenous, better as inputs for a future input-driven model. PCA (MoSeq-style)
reduces parameters and decorrelates. Pre-stroke-only fit keeps the latent frame as
the reference against which stroke changes are measured.

## 2026-06-08 — Sides locked: c2 = ipsilesional, c3 = contralesional

**Decision:** FR consensus components **c2 = mouse-LEFT = ipsilesional**,
**c3 = mouse-RIGHT = contralesional** (cohort lesions all left-VLS,
`stroke_laterality: "L"` in upstream animals.yaml).

**Why (quadruple-confirmed):** (1) `origin/main` code chain — `_translate_side_keys`
emits the npz in mouse frame, so `Tone_L` = mouse-left; (2) peri-tone: c2 responds to
Tone_L, c3 to Tone_R; (3) DLC tongue-x: c2↔x>0, c3↔x<0; (4) biology — PS46
contralesional (mouse-R) licks collapse 973→84→35 the two days post-stroke then
recover, ipsi preserved. See FINDINGS F5/F6.

## 2026-06-08 — Shared FaceRhythm consensus basis

**Decision:** Build one consensus FR basis across all pre-stroke sessions/animals
(match per-session rank-10 TCA components by spatial⊗frequency cosine + Hungarian,
average), and express sessions via low-rank projection onto it.

**Why:** Per-session TCA is fit independently → components not comparable. The
behaviorally-meaningful components (lick, jaw, whisk) reproduce across sessions
(cosine ~0.95) and animals (spatial 0.74–0.97), so a shared basis is well-defined;
slow/low-power components are unreliable and down-weighted. See FINDINGS F2/F3.

## Open questions / to revisit
- Decode all 33 pre-stroke sessions at locked kappa (fit currently on subset).
- Combined-feature vs FaceRhythm-only model comparison (what does fusion add?).
- Sticky kappa gives ~geometric durations; an HSMM (explicit durations) may give
  crisper multi-second bouts if wanted.
- Post-stroke + recovery projection through the pre-stroke basis/model.
- Memory/RAM: dynamax batched EM materializes (sessions,T,K,K); jax_moseq is lighter
  but full-cohort fits may still want GPU.
