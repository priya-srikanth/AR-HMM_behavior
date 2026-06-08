# CLAUDE.md — ground rules for the AR-HMM_behavior project

Read at the start of every session working in this repo.

## Data safety (non-negotiable)

1. **Never modify, rename, or delete source data.** Source data is read-only and
   lives outside this repo:
   - FaceRhythm outputs: `…/sabatini/compute/rich/analysis/face_rhythm/run_20250520/`
     (Richie Hakim's directory — read only).
   - DLC + spout/task outputs: `…/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/`.
   - The `stroke_orofacial_pipeline` repo (Priya's, separate) — read for reference
     (`configs/animals.yaml`, `src/stroke_orofacial/lr_convention.py`,
     `spout_behavior/lick_detection.py`), do not edit from here.
2. **Never write to the read-only network shares.** All outputs go to this repo
   (locally) or, if they must persist on the network, to `MICROSCOPE/Priya`.
   Never write to the `standby`/`sabatini` share.
3. **Local copies of source data go in a `data_local/` directory and are
   git-ignored.** Do not commit `.h5`, `.parquet`, `.avi`, or large `.npz/.npy`.

## Git discipline

- Every change lands as a commit on a **feature branch**; never commit directly
  to `main`. Before every commit, confirm the current branch is not `main`.
- Never force-push. Never run destructive git operations
  (`reset --hard`, `branch -D`, `rebase`, `clean -fd`, force-push) without
  explicit approval — describe what would change and ask first.
- Keep commits small and message-clear so any change can be cleanly reverted.

## Code conventions

- Code lives in `src/arhmm_behavior/`; notebooks/scripts orchestrate and contain
  no analytical logic of their own.
- **No absolute paths in source.** Resolve data locations through
  `arhmm_behavior.paths.PathResolver`, configured from `configs/data_sources.yaml`.
- **No hardcoded animal metadata or magic constants.** Animal/stroke metadata is
  read from the `stroke_orofacial_pipeline` `configs/animals.yaml` (single source
  of truth); analysis constants (clocks, TCA rank, frequency band edges) live in
  `configs/defaults.yaml`.
- Respect the **L/R side convention**: tongue-x / image side is *not* the mouse
  frame. This cohort (PS46–55) has `wavesurfer_sides_match_mouse_sides = false`.
  Translate to mouse frame via the upstream `lr_convention` logic before making
  any ipsi-/contralesional claim.
- Public functions get docstrings. Comments explain *why*, not *what*.

## Reference

`docs/DESIGN.md` is the plan; `docs/FINDINGS.md` logs established results. Update
`FINDINGS.md` when a result is confirmed.
