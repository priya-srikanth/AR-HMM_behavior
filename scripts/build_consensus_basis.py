#!/usr/bin/env python
"""Build the FaceRhythm consensus basis from all pre-stroke sessions.

Reads pre-stroke session windows from the upstream ``animals.yaml`` (single
source of truth), enumerates FaceRhythm sessions that have a ``TCA.h5``, builds
the consensus basis, and saves it to the outputs directory.

Usage:
    python scripts/build_consensus_basis.py [--cam cam2]

All locations come from configs/data_sources.yaml via PathResolver.
"""
from __future__ import annotations

import argparse

import numpy as np
import yaml

from arhmm_behavior.paths import PathResolver
from arhmm_behavior.facerhythm.io import load_tca_factors, load_vqt_meta
from arhmm_behavior.facerhythm.consensus import build_consensus


def _yyyymmdd(iso_date: str) -> str:
    return iso_date.replace("-", "")


def prestroke_sessions(resolver: PathResolver, cam: str) -> dict[str, list[str]]:
    """Per-animal pre-stroke session dates (<= stroke_date) that have a TCA.h5."""
    with open(resolver.animals_yaml()) as f:
        animals = yaml.safe_load(f)
    out: dict[str, list[str]] = {}
    for animal, meta in animals.items():
        start = _yyyymmdd(str(meta["sessions"]["start"]))
        stroke = _yyyymmdd(str(meta["stroke_date"]))  # last baseline (inclusive)
        sess_root = resolver.facerhythm_session(animal, "", cam).parent.parent
        if not sess_root.exists():
            continue
        dates = []
        for d in sorted(p.name for p in sess_root.iterdir() if p.is_dir()):
            if start <= d <= stroke and resolver.tca_h5(animal, d, cam).exists():
                dates.append(d)
        if dates:
            out[animal] = dates
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="cam2")
    ap.add_argument("--ref-animal", default="PS46")
    ap.add_argument("--ref-date", default="20250310")
    args = ap.parse_args()

    r = PathResolver()
    sessions = prestroke_sessions(r, args.cam)
    total = sum(len(v) for v in sessions.values())
    print(f"pre-stroke sessions with TCA: {total} across {len(sessions)} animals")

    ref = load_tca_factors(r.tca_h5(args.ref_animal, args.ref_date, args.cam))
    facs = [
        load_tca_factors(r.tca_h5(a, d, args.cam))
        for a, dates in sessions.items()
        for d in dates
    ]
    basis = build_consensus(facs, ref)

    meta = load_vqt_meta(r.vqt_h5(args.ref_animal, args.ref_date, args.cam))
    freqs = meta["frequencies"]
    print("slot  peakHz  reliability")
    for k, (pf, rel) in enumerate(zip(basis.peak_freqs(freqs), basis.reliability)):
        print(f"  c{k}: {pf:5.1f}   {rel:.2f}")

    out = r.outputs().parent / "data_local" / "consensus_basis.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        spatial=basis.spatial,
        frequency=basis.frequency,
        reliability=basis.reliability,
        frequencies=freqs,
        point_positions=meta["point_positions"],
        n_sessions=basis.n_sessions,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
