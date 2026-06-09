# DECISIONS — modeling iterations and choices

Chronological log of the analysis/modeling decisions for the AR-HMM, with
rationale, so the path is reproducible and reversible. Newest entries at the top.
Companion to `docs/FINDINGS.md` (results) and `docs/DESIGN.md` (plan).

## 2026-06-09 — Lick-side MI failure diagnosed: PCA discarded the lateralization axis (partial fix; confirmation pending)

**Problem:** the rate-sweep fits scored lick-side MI ~0.004–0.064, vs the ~0.36
achieved before (FINDINGS F8). Diagnosed in the sandbox (no new fitting needed):

1. **Labels are fine** (L-only 4742 / R-only 3887 bins; licking ≈ 1.8% of frames).
2. **The features carry the side signal loudly** — on L-lick vs R-lick bins:
   tongue_x_mean d′=6.9, tongue_angle_mean d′=6.6, fr_c2 d′=2.9 (ipsi),
   fr_c3 d′=−3.2 (contra), all correctly signed. So the refactor did NOT break
   the signal, and the new angle feature works.
3. **Root cause: the standardize→PCA-90% step discards the side axis.** Because
   licking is rare (~2% of frames), the lateralized direction is low total
   variance, so PCA retains only 5% of `tongue_x_mean` (and 35–57% of
   fr_c2/fr_c3/angle). The AR-HMM literally never sees the loudest "which side"
   feature, and lumps ipsi+contra licks into one tongue-out state (S0).

**Fix (committed):** add a per-feature `weights` to `build_design` and up-weight
the lateralization features (`tongue_x_mean, tongue_angle_mean, fr_c2, fr_c3`)
after standardization (default `--side-weight 2`, `--pca-var 0.95`). This raises
retention to angle 0.91 / fr_c2 0.94 / fr_c3 0.96 at 13 components (no collapse).
`side_weight=1` reproduces the old (axis-discarding) behavior.

**Open question (NOT yet confirmed):** retention is necessary but may not be
sufficient. Short CPU toy fits with the weighting still merged the two sides into
one state. An AR-HMM keys on **dynamics**, and L/R licking are **mirror-symmetric**
(same oscillation, opposite sign), so the side lives in the static mean, not the
autoregressive dynamics — a sticky/long-syllable model may merge them regardless
of feature variance. The proper test is a GPU fit with the weighting across a
kappa sweep that **includes short syllables** (the only regime that showed any
side signal), with enough iters/seeds. If it still merges, the remedy is likely
one of: short-syllable kappa, much stronger side weighting, or treating lick-side
as a **decoded projection** (onto the c2−c3 / tongue-x axis) rather than expecting
unsupervised syllables to carve a mirror-symmetric distinction. Revisit F8's 0.36
setup in that light.

## 2026-06-08 — Two model families (FaceRhythm covers only PS46–50)

**Decision:** Fit **two** AR-HMM families rather than one. **Model A** =
DLC + treadmill + task, **whole cohort PS46–55** — carries the
severity-/epoch-stratified comparison (the headline). **Model B** =
FaceRhythm + DLC + treadmill, **PS46–50 only** — the richer model (lateralized
lick syllables, FR manifold-distortion biomarker), as a depth analysis.

**Why:** Confirmed by directory audit that the FaceRhythm `run_20250520` outputs
exist only for PS46–50 (cam2/cam4); the moderate-severe/severe animals (PS51,
PS54, PS55) have **no FaceRhythm**. So a FR-containing model structurally cannot
include the severe group — exactly the animals the severity contrast needs.
DLC + treadmill + task exist cohort-wide, so they back Model A. `assemble()`
makes the FaceRhythm block optional to support both from one code path.

## 2026-06-08 — Model-grid rate = 33 Hz (parameterized); tongue-angle feature added

**Decision:** Place all signals on an explicit **model grid on the video
timeline at a configurable rate** (`configs/defaults.yaml`
`model.sampling_rate_hz`, **default 33 Hz**, 50 Hz a one-line switch), instead of
FaceRhythm's native 12.5 Hz clock. DLC (250 fps) is binned onto the grid,
wavesurfer events/treadmill (1 kHz) are mapped on via ws→camera→grid, and the
12.5 Hz FaceRhythm latents are **interpolated up** onto it. Add the **eye→spout
tongue angle** (`per_frame/tongue` `angle_deg_smoothed`) as `tongue_angle_mean`
+ `tongue_angle_speed`, computed at native 250 fps then binned.

**Why:** A tongue protrusion is only ~50–120 ms (measured from `lick_events`
rise→fall), so 12.5 Hz (80 ms bins) barely samples a lick (~0.6 samples) and
destroys within-lick kinematics. 33 Hz (~30 ms bins) resolves lick bouts and
gives a 0.5 s syllable ~16 frames of AR structure; 50 Hz additionally gives a
clean 5:1 DLC binning + 4× FR interpolation + ~5 samples/protrusion. Left as a
parameter so 33 vs 50 is picked empirically (held-out / lick-side MI at matched
~0.5–0.7 s duration). FaceRhythm above 12.5 Hz is interpolated (smooth envelopes,
no new info, no artifacts). **Supersedes** the 12.5 Hz / "6 DLC summaries" feature
design entry below — the matrix is now grid-rate with 8 DLC columns (adds the two
angle features) + the angle is the eye→spout signed angle (matches per-lick
`peak_angle_deg`), not a self-computed one.

## 2026-06-08 — Tongue position: NaN-when-absent, not zeroed

**Decision:** Tongue position (`tongue_x_mean`/`y_mean`) is averaged over
**present frames only** per bin; a fully-absent bin is left **NaN** and imputed
to the pre-stroke column mean (neutral) *after* standardization in `design.py`.
Occupancy (`tongue_out_frac`) carries the absence; standardization moments are
NaN-aware so the fill doesn't bias them.

**Why:** The prior code multiplied position by a present-mask, zeroing absent
frames — which conflates "tongue retracted" with "tongue at midline (x=0)." When
the tongue is retracted there is no real position to report, so excluding those
frames (and letting occupancy encode absence) is the honest representation. A
fabricated continuous-at-rest trace was considered and rejected.

## 2026-06-08 — Consensus basis: iterated reference, all 10 components kept

**Decision:** Build the FaceRhythm consensus basis with an **iterated
(medoid-like) reference** (`build_consensus_iterated`): build a consensus,
promote it to the reference, re-match all sessions, repeat to assignment
convergence (~5 iters). **Keep all 10 components** for now.

**Why:** A single pass against the arbitrary PS46/0310 reference biases the basis
toward that session. Iterating tightened every component's reliability (mean
cross-session match cosine) — all now **≥ 0.74** vs the single-pass 0.62–0.93.
Visual QC (`figures/consensus_basis_factors_detail.png`) shows the lick (c2 ipsi
0.92, c3 contra 0.77), whisk (c1), and several focal mid-freq components are
clean; c6 (1.1 Hz, 0.74, jagged tuning) and c5 (0.79) are the first prune
candidates if they add noise to syllables. c3 (contra lick) is the lowest-
reliability *meaningful* component but always kept — it is the channel predicted
to drop then recover post-stroke. **Augments** the consensus-basis entry below.

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

## 2026-06-08 — Severity metric = integrated deficit BURDEN per side (best/current)

**Correction:** the equal-weighted z-mean severity score (entry below) wrongly conflated
PS46 and PS50 — PS46's deeper/longer *contra* deficit got offset by PS50's transient *ipsi*
nadir dip. Replaced with **integrated deficit burden** = area between baseline and the
post-stroke response curve over days (depth × duration), computed SEPARATELY for contra and
ipsi (`np.trapz` of clipped fractional deficit, in "deficit-days"). This weights *lasting*
deficits and keeps lateralization explicit. Fig: `figures/severity_deficit_burden.png`;
data `data_local/deficit_burden.pkl`.

**Result (contra burden / ipsi burden, deficit-days):**
- SEVERE: PS55 58/49 (most bilateral, ratio 1.2), PS54 64/27, then
- MOD-SEVERE: PS51 36/15.
- MODERATE: **PS46 5.2/1.1 — most lateralized (contra:ipsi ≈ 4.5), a lasting focal contra
  deficit**; ~5× PS50's contra burden.
- MILD: PS49 1.5/2.6, PS50 1.1/0.5, PS47 0.9/0.6, PS48 0.3/0.0.

So **PS46 ≫ PS50** (Priya: PS46 more severe, lasting contra≫ipsi). Total-burden order
PS55>PS54>PS51≫PS46≫{PS49,PS50,PS47,PS48}. Lateralization (contra:ipsi ratio) is itself
informative: moderate animals are unilateral (PS46 ratio 4.5), the worst is bilateral
(PS55 ratio 1.2). This is the current canonical severity metric.

## 2026-06-08 — Severity as a SPECTRUM: depth + staged recovery of both responses

**Upgrade:** add **time-to-recovery for both** responses (ipsilesional-cued→ipsi and
contralesional-cued→contra; recovery = first post day ≥70% baseline, censored at last
session) alongside acute nadirs. Build a continuous severity score (z-mean of
[1−cc_nadir, 1−ii_nadir, log contra_rec, log ipsi_rec]). Fig:
`figures/severity_spectrum_recovery.png`; data `data_local/trialtype_recovery.pkl`.

**Result (matches Priya's domain read — a spectrum, not 3 bins):**
- MILD: PS47, PS48, PS49 (shallow, recover ≤2 d).
- MODERATE: **PS46** — selective contra deficit (cc_nadir 0, ii preserved 0.86), contra
  recovers ~7 d. (PS50 borderline mild/moderate: brief acute dip, fast recovery 2–3 d —
  by deficit *persistence* PS46 > PS50, per Priya.)
- MODERATE-SEVERE: **PS51** — acute bilateral cessation BUT recovers fastest of the
  cessation animals (ipsi 15 d, contra 45 d).
- SEVERE: PS54 (28/70 d), **PS55 worst** (70/70 d, on the ipsi=contra diagonal — no ipsi sparing).

**Staged recovery confirmed:** ipsi-response recovers before contra in PS51/PS54
(below diagonal); PS55 maximally delayed on both. The recovery axis (not acute depth) is
what distinguishes PS51 (mod-severe) from PS54/PS55 (severe). Hard k-means at N=8 is
unreliable (mislabels PS50) — trust the continuous score + recovery structure + domain labels.

## 2026-06-08 — Severity refined with cue×response trial-type structure (best)

**Upgrade:** Instead of aggregate lick counts, use the per-trial **cue side × lick side**
2×2 (acute-post vs baseline): contralesional-cued→contra-lick retention, ipsilesional-
cued→ipsi-lick retention, erroneous ipsi-licks on contra-cued trials, + contra recovery
days. k-means on these gives the cleanest, most interpretable grouping — each phenotype in
its own corner of (contra-response retention × ipsi-response retention):
- MILD (both retained): PS47 88%/99%, PS48 95%/107%, PS49 76%/84%, PS50 68%/82%.
- MODERATE (contra lost, ipsi kept = *selective*): **PS46 2%/92%** (+1.1 erroneous ipsi).
- SEVERE (both ~0 = *bilateral cessation*): PS51, PS54, PS55.

This **separates PS46 (moderate) from PS50 (mild)** — matching Priya's note that PS50
recovered fast. Fig: `figures/severity_clusters_trialtype.png`; feats:
`data_local/trialtype_feats.pkl`. Supersedes the aggregate-count grouping below for the
PS46/PS50 distinction. Still N=8 exploratory; DLC tongue kinematics remain a possible add.

## 2026-06-08 — Candidate stroke-severity groups (spout, FR-independent)

**Method:** k-means on per-animal acute deficit + recovery features from the **spout**
data (lick counts/sides, mouse-frame), FR-independent, cohort PS46–55 (laser 2.2–5.0 mW).
Features = acute contralesional-count loss (depth), acute total-drop (bilateral component),
and **contralesional recovery time** (days for contra-fraction to reach 0.7×baseline,
gated on total>0.3×baseline so the metric isn't fooled by no-lick days). NOTE: an earlier
version used *total*-licking recovery, which understated moderate animals (ipsi licking
recovers fast while contra is still impaired) — contra-specific recovery is the correct
axis (per Priya: PS46 recovered over several days, PS50 quickly).
Fig: `figures/severity_clusters_spout.png`.

**Candidate groups (validate against laser power, which clustering didn't see):**
- MILD: PS47, PS49, PS48 (2.2–2.5 mW) — shallow, contra recovery 1–2 days.
- MODERATE: PS50, PS46 (2.2–2.5 mW) — deep selective contralesional loss (0.96–0.97);
  contra recovery PS50 ≈ 3 d, **PS46 ≈ 6 d** (slower-recovering of the two).
- SEVERE: PS51, PS54, PS55 (3.0–5.0 mW) — near-total bilateral cessation; contra recovery
  12 / 36 / 40 days (ordered with laser power; PS55 5.0 mW slowest).

Severity = contra-deficit DEPTH × contra RECOVERY time; recovers the mild/moderate/severe
phenotypes from behavior alone and tracks laser power. Caveats: N=8 exploratory; PS46 vs
PS50 differ in recovery speed within "moderate". Spout-only — DLC tongue-angle/side could refine.
Next: cross-reference PS46–50 groups with the FR manifold-distortion biomarker (predict
MODERATE PS46/PS50 show the strongest distortion). Features:
`data_local/severity_feats2.pkl`.

## 2026-06-08 — K (number of states) scan confirms K≈16

**Decision:** Keep **K=16**. Held-out scan (train 10 / held-out 5 sessions, kappa=1e8):
held-out log-lik/frame rises steeply to K≈16 then plateaus (K8 14.40 → K16 14.67 →
K28 14.71); lick-side ipsi/contra MI is ~0 at K=8 and jumps to ~0.18–0.27 for K≥14
(running captured at all K); states-used saturates (K28 uses only 22). So K=16 sits
just past the LL elbow, captures lateralized licking, and doesn't waste states.
Duration stays ~0.72 s across K (kappa, not K, sets duration). Figure: `figures/kscan.png`.
Caveat: single fit/seed per K — average seeds for a publication-grade curve.

## 2026-06-08 — Sticky HMM vs HSMM: HSMM NOT warranted

**Decision:** Keep the sticky (geometric-duration) jax_moseq HMM; do **not** build an
explicit-duration HSMM.

**Evidence (hazard analysis on pre-stroke bouts, `figures/sticky_vs_hsmm_hazard.png`):**
Real behavioral bouts (running, licking) have **CV ≈ 1.5** (over-dispersed vs geometric's
~1) and a **flat-to-decreasing hazard** P(end | survived). A peaked-duration HSMM is
motivated by the *opposite* signature (CV < 1, rising hazard / characteristic length),
which is absent here. A peaked HSMM would impose a timescale the data lacks and fit worse.

**Nuance:** the mild over-dispersion (heavy tail) reflects behavioral heterogeneity
(short fidgety + long sustained bouts), best handled by enough states/sub-states (or, if
explicit durations were ever wanted, a heavy-tailed negative-binomial r<1 — not a peaked
HSMM). KPMS's sticky-HMM choice is appropriate for this data.

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

## 2026-06-09 — Lateralized-lick retention fix (`--side-fit`); fit-collapse blocker

**Decision:** Add `build_design --side-fit=present`: estimate standardization + PCA
on tongue-**present** bins (drop the ~86% baseline-filled/absent bins from the
**fit**; still project all bins). The per-feature `--side-weight` on the lateralized
features (`tongue_x_mean`, `tongue_angle_mean`, `fr_c2`, `fr_c3`) stays.

**Why:** `tongue_x_mean` was already baseline-fill-masked, but is NaN ~86% of bins
(tongue out only ~13.5% of the time) and got imputed to the mean before PCA —
diluting its variance so PCA discarded it (retention **0.09** even at weight 4).
Fitting PCA on present bins recovers it: **tongue_x retention 0.09 → 0.51**. The
alternative "present-cov" (standardize on all bins, covariance on present) does NOT
help — tongue_x stays 0.09; only present-bin *standardization* lets the sparse axis
compete. No-lick / severe-stroke sessions are entirely tongue-absent → they
contribute nothing to the fit but still project (tongue imputed neutral ≈ "tongue
out 0"); guarded against the degenerate all-absent cohort.

**Status — the ipsi/contra SPLIT is still untested, blocked by fit collapse.** With
tongue_x retained, the fit (kappa 5e8/1e9, 500 iters) **collapsed to 1–2 states**
(median dwell 3621 s = whole session, lick-MI 0); the per-state cross-tab then
trivially shows all licks in one mega-state — uninformative. More iters made the
collapse worse (the sticky prior converges harder). So retention is fixed but a
separate convergence problem gates the science.

**Scale investigation (why it collapses):** PCA scores are already ~unit-scale
(per-PC std 0.6–1.8) → **whitening is not the lever**. Data is **83% quiescent**.
The kappa→duration curve is pathological: by the sticky-prior math kappa≈1e2 gives
~0.5 s dwell, yet empirically kappa 1e7 fragments (0.15 s) and 2e8 collapses — a
sharp fragment→collapse crossover with **no 0.5–0.7 s plateau**, the signature of an
over-confident AR emission (`S_0_scale=0.01` tiny → residual variance
underestimated), amplified by 14 latent dims and the rest-mass. **Proposed
calibration (before any further sweep):** cheaply map kappa→duration on the fit
subset; if no plateau exists, soften the emission (raise `S_0_scale`) and/or cut
`latent_dim` rather than only lowering kappa.

## 2026-06-09 — RESOLVED: clean ipsi/contra split via dim-cut + emission softening

**Outcome:** The calibration (`fit_arhmm --calibrate`) showed **kappa is inert across
1e1–1e8** on the rebuilt 33 Hz design — the fit is emission-dominated (collapse at stiff
emissions, 1–3-frame fragmentation at soft), with no kappa-controlled duration plateau.
The fix was the emission side, not kappa: **cut latent dim to ~5 PCs (`--pca-var 0.70`,
side features still retained — tongue_x 0.49) + soften emissions (`S_0_scale=10`)**. At
S0=10 / kappa=1e6 / nlags=3 / K=16 the fit uses **14 states and SPLITS sides cleanly**:
ipsi-specific syllables (state 8 = 5327 ipsi/0 contra) and contra-specific (states
0/13/10), **lick-side MI = 0.578** over all 33 pre-stroke sessions (vs F8's 0.36, and the
prior stuck 0.007–0.064). See FINDINGS F10.

**Note / discrepancy:** this contradicts F8's "smooth kappa→duration, locked
kappa=1e8/0.72 s." On the current pipeline kappa=1e8 collapses; duration is set by S0/dim
and lands short (~0.12 s). F8's kappa/duration regime is invalid here — what changed it
(rate? feature scale? dim?) is unresolved. Tooling added: `--calibrate`, `--s0-scale`
(fit_arhmm), `moseq S_0_scale` param.

## Open questions / to revisit
- **Duration (was: fit collapse — RESOLVED for the split):** the split is clean but
  syllables are short (~0.12 s), not the ~0.5–0.7 s behavioral timescale, and kappa can't
  lengthen them on this design (emission-dominated). For behavioral-unit duration, revisit
  via HSMM (explicit durations) or reconcile the F8-vs-now kappa discrepancy.
- Confirm the split across seeds (seed-robustness sweep) before locking the config.
- Decode all 33 pre-stroke sessions at locked kappa (fit currently on subset).
- Decode all 33 pre-stroke sessions at locked kappa (fit currently on subset).
- Combined-feature vs FaceRhythm-only model comparison (what does fusion add?).
- Sticky kappa gives ~geometric durations; an HSMM (explicit durations) may give
  crisper multi-second bouts if wanted.
- Post-stroke + recovery projection through the pre-stroke basis/model.
- Memory/RAM: dynamax batched EM materializes (sessions,T,K,K); jax_moseq is lighter
  but full-cohort fits may still want GPU.
