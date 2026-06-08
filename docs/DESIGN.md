# DESIGN — AR-HMM of orofacial behavior across striatal stroke

## 1. Scientific question

After a unilateral ventrolateral striatal stroke, mice performing a lateralized
Pavlovian licking task show graded behavioral deficits and recovery: mild strokes
give transient erroneous ipsilateral licking; moderate give loss then recovery of
contralateral licking; severe give transient bilateral loss then staged recovery
(ipsi- then contra-lateral). DLC and FaceRhythm each give a partial view. We want
a **model-based, low-dimensional description of behavioral structure** that:

- segments continuous behavior into recurring, interpretable states ("syllables"),
- lets us quantify how state usage, dynamics, and transitions change pre- vs
  post-stroke and over recovery, and
- fuses the complementary data streams rather than analyzing each in isolation.

The chosen model class is an **autoregressive hidden Markov model (AR-HMM)**, the
MoSeq lineage. This document records why, the inputs, and the plan.

## 2. Task and cohort

Pavlovian lateralized licking: tone A → left-spout reward, tone B → right-spout
reward. Concurrent orofacial video (250 fps) drove DLC and FaceRhythm; a treadmill
encoder and a 1000 Hz wavesurfer DAQ recorded ambulation and lick/tone/reward
events.

Cohort PS46–PS55 (subset PS46–50 has the FaceRhythm `run_20250520` outputs).
Per-animal stroke dates, session ranges, exclusions, and the
`wavesurfer_sides_match_mouse_sides` flag are the single source of truth in the
upstream repo's `configs/animals.yaml` — **do not duplicate them here**. The whole
PS46–55 cohort was acquired on a mirrored rig (`wavesurfer L = mouse R`).

## 3. Data sources and clocks

Three streams, two clocks (see `configs/data_sources.yaml`, `configs/defaults.yaml`):

| Stream | Source | Rate / clock |
| --- | --- | --- |
| FaceRhythm TCA latents | `run_20250520/.../TCA.h5` | 12.5 Hz (video ÷ 20), **video clock** |
| DLC tongue/jaw kinematics | `…/dlc_kinematics/cleaned_traces/` (parquet) | 250 fps, **video clock** |
| DLC lick events (side-labeled) | `…/dlc_kinematics/lick_events/tongue/` | event times, **video clock** |
| Task: tone/reward/lick | `…/spout_behavior/licks_and_rewards/` (npz) | 1000 Hz, **wavesurfer clock** |

Key alignment facts (verified, see FINDINGS):

- FaceRhythm and DLC share the **identical video clock** (frame counts match
  exactly). The FaceRhythm latent index → video-frame mapping is the VQT
  `x_axis` dataset. So FR ↔ DLC alignment is just frame indexing.
- The wavesurfer (task events) is a **separate clock**; the upstream pipeline
  already computes the wavesurfer↔video synchronization. We consume that rather
  than re-deriving it.
- **Side convention**: tongue-x/image side ≠ mouse-anatomical side. Translate via
  upstream `lr_convention.to_mouse_frame` before any ipsi/contralesional claim.

## 4. FaceRhythm: what the outputs are

Pipeline (whole continuous session, not trial-windowed): ROIs → optic-flow point
tracking (1093 registered points) → VQT spectrogram (variable-Q transform, 0.5–60
Hz, 30 log bins, non-negative power) → TCA (non-negative CP/PARAFAC, rank 10).

The model-ready object is the TCA **time factor**: a 10-D, non-negative,
interpretable latent at 12.5 Hz, one set per session. Each component carries a
spatial loading (which face points) and a frequency signature (which band), so
components can be characterized (lick, whisk, breathing, …) — but **only by
aligning to kinematics, not by frequency alone**.

### Consensus basis
Per-session TCA was fit independently, so component *k* is not comparable across
sessions/animals out of the box. We make it comparable by matching per-session
components (spatial⊗frequency cosine + Hungarian) to a reference and averaging into
a **consensus basis** shared across all pre-stroke sessions/animals. Sessions are
expressed in the shared basis either by projecting their VQT onto the consensus
atoms (NNLS) or, cheaply, via the low-rank route using each session's own factors.
This is implemented in `src/arhmm_behavior/facerhythm/consensus.py`.

## 5. Feature design for the AR-HMM

Candidate observation vector, all resampled to the 12.5 Hz FaceRhythm clock:

- **FaceRhythm**: consensus component loadings (the behaviorally meaningful,
  reliable subset — lick-left, lick-right, whisk, jaw, breathing; drop/῾downweight
  unstable slow components).
- **DLC**: tongue tip x/y (and protrusion magnitude/velocity), jaw position —
  the kinematic detail FaceRhythm lacks (it never sees the tongue).
- **Task** (optional, as covariates/inputs): tone/reward/spout-contact, treadmill.

Open design choice (Section 6): a single fused observation vector vs. an
input-driven model with task variables as exogenous inputs.

## 6. Modeling plan

**Why AR-HMM.** The MoSeq lineage (AR-HMM on PC-reduced video; Keypoint-MoSeq
with a keypoint noise model) is the right reference class for segmenting
continuous behavior into syllables with within-state linear dynamics, and the
transition matrix is directly comparable across conditions.

**The hard part is the observation model and fusion, not the HMM.**
- DLC tongue/jaw kinematics are time-domain position/velocity — what AR dynamics
  were built for. **Keypoint-MoSeq** is near-purpose-built for them (models DLC
  jitter/dropout as observation noise).
- FaceRhythm latents are spectral *power* (non-negative), so AR dynamics on them
  mean something different than on a position trace — likely better treated as
  additional observations or covariates than as primary AR variables.
- Task events are discrete inputs. Since the core finding is a broken/recovering
  cue→action mapping, an **input-driven HMM** (GLM-HMM / IO-HMM, or rSLDS with
  inputs) is attractive: stroke can change the cue→state mapping itself.

**Approach.** Use a Keypoint-MoSeq-style AR backbone on the DLC kinematics, bring
FaceRhythm consensus loadings + task variables in as additional observations or
inputs, and explicitly compare a "single fused observation vector" design against
an "input-driven" design. If AR-HMM proves too rigid, step up to **rSLDS**.

**Tooling.** `keypoint-moseq`; `dynamax` (JAX, actively maintained) or Linderman's
`ssm` for custom AR-HMM / (r)SLDS. (Installed via the `model` extra.)

## 7. Open questions / risks

- **Cross-animal spatial registration** holds for core components (lick/jaw/whisk)
  but not the slow ones — decide handling (drop vs frequency-only matching vs
  joint refit).
- **Number of states / AR lags / syllable timescale** — set by the 12.5 Hz clock
  and cross-validated.
- **Mouse-frame side mapping** must be applied before stroke-laterality analysis.
- **Stroke changes the manifold** (esp. severe animals) — a basis fit on
  pre-stroke data may not span post-stroke behavior; may need shared or
  separately-validated bases.
- **Modality scaling** — heterogeneous units; whiten/standardize so one stream
  doesn't dominate.

## 8. Roadmap

1. **Understand inputs** (current). FaceRhythm characterization, consensus basis,
   kinematic validation. ✔ mostly done (see FINDINGS).
2. **Mouse-frame + task events.** Apply `lr_convention`; bring wavesurfer-aligned
   tone/reward/lick onto the 12.5 Hz clock; confirm lick-left/right → ipsi/contra.
3. **Unified feature matrix.** Assemble FR + DLC (+ task) per session, standardized,
   time-aligned; persist a tidy per-session feature store.
4. **Baseline model.** Keypoint-MoSeq / AR-HMM on DLC alone; sanity-check syllables
   pre vs post stroke.
5. **Multimodal / input-driven model.** Add FaceRhythm + task; compare designs.
6. **Stroke analyses.** State usage, dynamics, transitions across severity and
   recovery; relate to the three existing analysis perspectives.
