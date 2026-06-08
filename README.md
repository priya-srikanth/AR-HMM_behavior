# AR-HMM_behavior

Modeling orofacial behavioral structure before and after ventrolateral striatal
stroke in mice performing a lateralized (Pavlovian) licking task, using an
**autoregressive hidden Markov model (AR-HMM)**.

The goal is a complementary, model-based view of behavior that fuses three
existing data streams into a shared latent description, then segments it into
recurring behavioral states ("syllables") whose statistics can be compared
pre- vs post-stroke and across recovery:

1. **FaceRhythm** — unbiased rhythmic orofacial movement latents (whisking,
   licking, breathing, …) from facial optic-flow spectrograms + TCA.
2. **DeepLabCut (DLC)** — tongue and jaw kinematics (the tongue is *not* in
   FaceRhythm, so DLC is complementary).
3. **Task / spout behavior** — tone, reward, and lick/spout-contact events.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full plan and
[`docs/FINDINGS.md`](docs/FINDINGS.md) for results established so far.

## Status

Early exploratory phase. We have:

- Characterized the FaceRhythm `run_20250520` outputs (per-session rank-10 TCA,
  12.5 Hz latents) and confirmed they run on the whole continuous session.
- Built a **consensus FaceRhythm basis** shared across all pre-stroke sessions
  and animals (PS46–50), validated against DLC tongue kinematics.
- Found that the two dominant lick-band factors decompose into **left- vs
  right-lick** facial-motion representations (a natural, stroke-relevant signal
  for a lateralized task).

The AR-HMM itself is not yet built; current work is understanding and
assembling the input features.

## Layout

```
configs/        YAML config: constants (clocks, ranks) and data-source paths
docs/           DESIGN.md (plan), FINDINGS.md (results), figures referenced therein
src/arhmm_behavior/
  paths.py          PathResolver — no absolute paths in code
  facerhythm/       load TCA/VQT factors, latent↔video alignment, consensus basis
  dlc/              load cleaned tongue/jaw traces and side-labeled lick events
  features/         assemble the unified, time-aligned feature matrix (in progress)
  model/            AR-HMM wrappers (planned)
scripts/        runnable entry points (e.g. build the consensus basis)
tests/          unit tests
```

## Data safety

Source data lives on read-only network shares and other repos; **this project
never writes to them**. See [`CLAUDE.md`](CLAUDE.md) for the full ground rules.

## Environment

```bash
conda env create -f environment.yml
conda activate arhmm_behavior
pip install -e .
```
