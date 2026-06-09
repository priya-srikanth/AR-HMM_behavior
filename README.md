# AR-HMM_behavior

Modeling orofacial **behavioral structure** before and after a ventrolateral
striatal (VLS) stroke in mice performing a lateralized (Pavlovian) licking task,
using an **autoregressive hidden Markov model (AR-HMM)**.

The scientific goal: a model-based description of behavior that fuses three data
streams into one shared latent space, segments it into recurring behavioral
states ("syllables"), and then asks **how that structure breaks and recovers
after stroke** — which syllables are lost acutely, which return, and whether the
recovery of behavioral structure tracks the severity of the lesion.

This README is meant to get a new person oriented. Start here, then read
[`docs/DESIGN.md`](docs/DESIGN.md) (the plan) and
[`docs/FINDINGS.md`](docs/FINDINGS.md) (results established so far, with figures).

---

## 1. The experiment in one paragraph

Head-fixed mice run a Pavlovian task: a tone predicts reward at one of two
spouts (left or right). Each session a high-speed (250 fps) face video, a
1 kHz "wavesurfer" recording of task/spout/treadmill signals, and a running
wheel are recorded. After several baseline (pre-stroke) days, a photothrombotic
stroke is made in the **left** VLS; recording continues for days–weeks of
recovery. Because the lesion is unilateral, the behavioral deficit is
*lateralized* — it preferentially affects the **contralesional** (here, the
mouse's **right**) side. The cohort is PS46–PS55; lesion size scales with laser
power (2.2–5.0 mW).

## 2. The three data streams

| stream | what it is | source | rate |
|---|---|---|---|
| **FaceRhythm** | unbiased rhythmic orofacial-movement latents (whisking, licking, …) from facial optic-flow spectrograms decomposed by TCA | Richie Hakim's `run_20250520` (read-only) | 12.5 Hz |
| **DeepLabCut (DLC)** | tongue and jaw kinematics — note the tongue is *not* visible to FaceRhythm, so DLC is complementary, not redundant | `stroke_orofacial_pipeline` outputs | 250 fps |
| **task / spout / treadmill** | tone, reward, lick (spout-contact) events, and locomotion speed | `stroke_orofacial_pipeline` outputs (wavesurfer-derived) | 1 kHz |

**Important coverage fact:** FaceRhythm exists only for **PS46–PS50** (the mild +
moderate animals). The moderate-severe and severe animals (PS51, PS54, PS55)
have **no FaceRhythm**. This drives the two-model-family design below.

## 3. Two model families (because of FaceRhythm coverage)

- **Model A — DLC + treadmill + spout, whole cohort (PS46–55).** The only model
  that can include the severe animals, so it carries the
  severity-stratified / epoch-stratified comparison (the headline analysis).
- **Model B — FaceRhythm + DLC + treadmill, PS46–50 only.** The richer model
  (lateralized lick syllables emerge; a manifold-distortion recovery biomarker),
  but limited to the animals that have FaceRhythm. A depth analysis, not a
  cohort-wide one.

Treadmill, spout, and DLC feed **both** families; only FaceRhythm differs.

## 4. Glossary (terms you'll hit immediately)

- **AR-HMM / syllable.** A hidden Markov model whose per-state emissions are an
  autoregressive (AR) process: each state is a short, stereotyped *movement
  dynamic*. The discrete states are the behavioral "syllables." We use a
  **sticky** AR-HMM so syllables last a realistic ~0.3–0.7 s rather than flipping
  every frame (the "stickiness" is set by a parameter, kappa).
- **TCA component vs factor.** FaceRhythm decomposes a spectrogram *tensor*
  (space × frequency × time) into K=10 rank-1 **components**. Each *component*
  (e.g. "c2") is the outer product of three **factors** — one loading vector per
  tensor mode: a **spatial** factor (where on the face), a **frequency** factor
  (what rhythm, in Hz), and a **temporal** factor (the 12.5 Hz time course that
  the model actually consumes). So a component has three factors; we often say
  "factor c2" loosely to mean "component 2."
- **Consensus basis.** Per-session TCA is fit independently, so component *k* in
  one session is not the same as component *k* in another. The consensus basis
  matches components across sessions (by spatial⊗frequency similarity) and
  averages them into one shared set of components, so a loading is comparable
  across animals and days. See §6.
- **The clocks.** FaceRhythm and DLC share the **video clock**; wavesurfer is a
  separate **1 kHz clock**. Everything is placed on the FaceRhythm **latent
  clock** (12.5 Hz). Video→latent uses the VQT `x_axis`; wavesurfer→latent uses
  an upstream alignment template (wavesurfer sample → camera frame) then the
  `x_axis`.
- **Side convention (critical).** The face/tongue-x "left/right" is the *image*
  frame, which is **not** the mouse's anatomical frame for this cohort
  (`wavesurfer_sides_match_mouse_sides = false`). Upstream code translates to the
  **mouse frame** before saving. For this left-VLS cohort: **mouse-left =
  ipsilesional, mouse-right = contralesional.** FaceRhythm component **c2 =
  ipsilesional lick, c3 = contralesional lick** (established four independent
  ways — see FINDINGS). c3 is the channel predicted to drop then recover after
  stroke.
- **Severity phenotypes & recovery epochs.** Defined in the
  `stroke_orofacial_pipeline` repo (not here) from spout behavior: a 3-tier
  (mild/moderate/severe) or 4-tier (…/mod-severe/…) grade, and per-animal
  acute / sub-acute / chronic epoch boundaries. The AR-HMM analysis is stratified
  by these.

## 5. The feature matrix

`features/assemble.py` builds one tidy table per session at the 12.5 Hz latent
clock, with four blocks of columns:

- **facerhythm** — the consensus-basis temporal loadings, one column per
  component (`fr_c0`…`fr_c9`).
- **dlc** — binned tongue/jaw summaries: position (mean over *present* frames;
  NaN when the tongue is absent, so "absent" is never confused with "midline"),
  occupancy (`tongue_out_frac`), motion energy, and the eye→spout **tongue
  angle** (per-bin mean + angular speed) which carries within-lick kinematics.
- **treadmill** — locomotion speed (mm/s) + acceleration on the latent clock,
  from the upstream-preprocessed wheel signal.
- **task_input** — per-bin tone/lick event counts per mouse-side. These are
  **covariates/labels, not AR observations** — they are *excluded* from the model
  so that lateralized licking has to *emerge* from FaceRhythm + DLC rather than
  being read off the labels.

All signals are placed on one **model grid** at a configurable rate
(`configs/defaults.yaml` → `model.sampling_rate_hz`, default **33 Hz**; 50 Hz is
a one-line switch), defined on the video timeline: DLC (250 fps) is binned onto
it, wavesurfer signals (1 kHz) are mapped onto it (ws→camera→grid), and the
12.5 Hz FaceRhythm latents are interpolated up onto it. 12.5 Hz is too coarse for
a ~50–120 ms protrusion; 33/50 Hz resolves lick bouts and gives the AR-HMM real
within-syllable dynamics. The exact rate (33 vs 50) is picked empirically at fit
time (held-out likelihood + lick-side MI).

Values are stored **raw**; standardization + PCA happen later in
`model/design.py`, fit on the **pooled pre-stroke** data so post-stroke sessions
are projected through a frozen pre-stroke reference frame (changes are measured
*against* baseline). Missing values (NaN) impute to the pre-stroke mean (neutral)
after standardization.

## 6. The consensus basis (how it's built)

`scripts/build_consensus_basis.py` → `facerhythm/consensus.py`:

1. Load each pre-stroke session's rank-10 TCA factors (PS46–50, 33 sessions).
2. **Iterated reference** (`build_consensus_iterated`): match every session to a
   reference, average matched components into a consensus, promote that consensus
   to the new reference, and repeat until the component→slot assignments stop
   changing (~5 iterations). This avoids biasing the basis toward one arbitrary
   reference session and tightens the per-component **reliability** (mean
   cross-session match cosine).
3. Each component gets a reliability score. Currently **all 10 are kept**; the
   slow factor c6 (r≈0.74, noisy/multimodal tuning) and c5 (r≈0.79) are the first
   prune candidates if they add noise to the syllables. The lick components c2
   (ipsi, r≈0.92) and c3 (contra, r≈0.77) are always kept — c3 is lower-reliability
   but scientifically essential.

QC figure (spatial map + frequency tuning per component) is generated alongside.

## 7. Repo layout

```
configs/
  defaults.yaml        analysis constants (clocks, TCA rank, frequency bands)
  data_sources.yaml    machine-specific paths (EDIT per machine; never hardcode in src)
docs/
  DESIGN.md            the plan
  FINDINGS.md          established results (F1–F9), with figures
  DECISIONS.md         decisions log
src/arhmm_behavior/
  paths.py             PathResolver — resolves every data/output location from data_sources.yaml
  facerhythm/
    io.py              load TCA factors + small VQT metadata from the HDF5 files
    alignment.py       bin per-frame (250 fps) signals onto the 12.5 Hz latent clock
    consensus.py       match components across sessions; build the (iterated) consensus basis
  dlc/
    kinematics.py      load cleaned tongue/jaw traces; protrusion etc.
  treadmill.py         load the upstream-preprocessed locomotion speed
  task_events.py       place wavesurfer events on the latent clock (ws→camera→latent)
  features/
    assemble.py        build the unified per-session feature matrix (the 4 blocks above)
  model/
    design.py          pool → standardize → PCA; project new sessions through the frozen frame
    arhmm.py           dynamax AR-HMM wrapper (baseline)
    moseq.py           jax_moseq sticky AR-HMM wrapper (canonical engine)
    arhmm.py           dynamax AR-HMM wrapper (baseline)
    moseq.py           jax_moseq sticky AR-HMM wrapper (canonical engine)
    scoring.py         lick-side / running MI, syllable usage, dwell, transitions
scripts/               the pipeline, in run order:
  build_consensus_basis.py   iterated FaceRhythm consensus basis (Model B)
  assemble_features.py       per-session feature matrices on the model grid
  build_design.py            pooled pre-stroke standardize + PCA (frozen transform)
  fit_arhmm.py               sticky AR-HMM fit + kappa/nlags sweep, scored by MI
  syllable_analysis.py       epoch/severity-stratified syllable + biomarker readout
tests/                 unit tests
data_local/            git-ignored scratch (local copies of source data, cached
                       bases, feature matrices, designs, fits, analysis outputs)
```

## 8. How to run

```bash
# one-time
conda env create -f environment.yml
conda activate arhmm_behavior
pip install -e .              # assembly/consensus/design deps
pip install -e ".[model]"     # + jax-moseq for fitting (jax-metal for Apple-GPU)

# edit configs/data_sources.yaml so the paths match THIS machine, then run the
# full pipeline (Model B = FaceRhythm+DLC, PS46–50; Model A = DLC-only, cohort):
python scripts/build_consensus_basis.py --cam cam2
python scripts/assemble_features.py --family B        # and: --family A
python scripts/build_design.py        --family B       # and: --family A
python scripts/fit_arhmm.py           --family B --kappas 1e7 1e8 1e9 --nlags 3
python scripts/syllable_analysis.py   --family B
```

To compare sampling rates, set `model.sampling_rate_hz` (33 vs 50) in
`configs/defaults.yaml` and re-run `assemble_features` → `build_design` →
`fit_arhmm`; the fit's `sweep_results.csv` (lick-side MI at a matched ~0.5–0.7 s
syllable duration) picks the rate. Fitting is the heavy step — run it on a GPU.

## 9. Status

The full pipeline is built and the code path validated end-to-end (assemble →
design → fit → decode → score) on a real session; the heavy assembly + fitting
runs execute on a workstation/GPU from the scripts above. Done: characterized
FaceRhythm outputs; built and QC'd the iterated consensus basis (PS46–50);
identified c2/c3 as the ipsi/contra lick components; ported treadmill features;
fixed tongue handling; parameterized the model-grid rate (default 33 Hz) and
added the eye→spout tongue-angle features; built the per-family feature assembly,
pooled pre-stroke design, sticky-AR-HMM fit + kappa/nlags sweep (scored by
lick-side MI), and the epoch-/severity-stratified syllable + manifold-distortion
analysis. Next: run the full sweep on a GPU, pick the rate (33 vs 50 Hz), and
interpret the epoch/severity syllable results. See FINDINGS F7–F9 for the earlier
baseline AR-HMM and post-stroke projection results.

## 10. Data safety

Source data lives on read-only network shares and in other repos; **this project
never writes to them.** Outputs go to `data_local/` (local, git-ignored) or, if
they must persist, to the `MICROSCOPE/Priya` share — never the `sabatini`/standby
share. Full ground rules in [`CLAUDE.md`](CLAUDE.md).
