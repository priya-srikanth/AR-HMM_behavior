#!/usr/bin/env python
"""Assemble per-session feature matrices on the model grid, for both families.

Two model families (see README §3):
  - "B" : FaceRhythm + DLC + treadmill + task  — PS46–50 only (FR from cam2).
  - "A" : DLC + treadmill + task (no FaceRhythm) — whole cohort PS46–55,
          including the severe animals that lack FaceRhythm.

DLC, treadmill, and task feed both; only the FaceRhythm block differs. The
model-grid sampling rate is read from ``configs/defaults.yaml``
(``model.sampling_rate_hz``). One parquet is cached per session to
``data_local/features/<family>/<animal>/<date>.parquet`` (git-ignored), plus a
``manifest.csv`` recording animal/date/rel_day/n_grid for downstream pooling.

Usage:
    python scripts/assemble_features.py --family B            # FR+DLC, PS46-50
    python scripts/assemble_features.py --family A            # DLC-only, cohort
    python scripts/assemble_features.py --family B --animals PS46 PS47
"""
from __future__ import annotations

import argparse
import csv
import datetime

import numpy as np
import yaml

from arhmm_behavior.paths import PathResolver
from arhmm_behavior.facerhythm.io import load_tca_factors, load_vqt_meta
from arhmm_behavior.facerhythm.consensus import ConsensusBasis
from arhmm_behavior.dlc.kinematics import load_cleaned_trace, load_tongue_angle
from arhmm_behavior.treadmill import load_treadmill_speed
from arhmm_behavior.task_events import load_alignment_template
from arhmm_behavior.features.assemble import assemble

_FAMILY_ANIMALS = {
    "B": ["PS46", "PS47", "PS48", "PS49", "PS50"],
    "A": ["PS46", "PS47", "PS48", "PS49", "PS50", "PS51", "PS54", "PS55"],
}
_EVENT_KEYS = ("Tone_L", "Tone_R", "Lick_L", "Lick_R")


def _load_consensus(resolver: PathResolver) -> ConsensusBasis:
    """Load the cached iterated consensus basis (built by build_consensus_basis.py)."""
    d = np.load(resolver.local_work() / "consensus_basis.npz")
    return ConsensusBasis(spatial=d["spatial"], frequency=d["frequency"],
                          reliability=d["reliability"], n_sessions=int(d["n_sessions"]))


def _rate_hz(resolver: PathResolver) -> float:
    with open(resolver.config_path.parent / "defaults.yaml") as f:
        return float(yaml.safe_load(f)["model"]["sampling_rate_hz"])


def _sessions_with_all_sources(resolver, animal, cam, need_fr):
    """Dates that have every required source for this family, in date order."""
    import os
    out_root = resolver._root("stroke_pipeline_outputs")
    have = {}
    have["tongue"] = {p.stem for p in (out_root / "dlc_kinematics" / "cleaned_traces" / "tongue" / animal).glob("*.parquet")}
    have["jaw"] = {p.stem for p in (out_root / "dlc_kinematics" / "cleaned_traces" / "jaw" / animal).glob("*.parquet")}
    have["angle"] = {p.stem for p in (out_root / "dlc_kinematics" / "per_frame" / "tongue" / animal).glob("*.parquet")}
    have["tread"] = {p.stem for p in (out_root / "spout_behavior" / "treadmill_signals_preprocessed" / animal).glob("*.npz")}
    have["licks"] = {p.stem for p in (out_root / "spout_behavior" / "licks_and_rewards" / animal).glob("*.npz")}
    have["align"] = {p.stem for p in (resolver._root("alignment_templates") / cam / animal).glob("*.npz")}
    common = set.intersection(*have.values())
    if need_fr:
        fr_root = resolver.facerhythm_session(animal, "", cam).parent.parent
        common &= {d.name for d in fr_root.iterdir()
                   if d.is_dir() and resolver.tca_h5(animal, d.name, cam).exists()}
    return sorted(d for d in common if d.isdigit())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["A", "B"], required=True)
    ap.add_argument("--cam", default="cam2")
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--rate", type=float, default=None,
                    help="model-grid rate (Hz); default from configs/defaults.yaml. "
                         "Outputs are namespaced by rate so 33 and 50 Hz coexist.")
    args = ap.parse_args()

    resolver = PathResolver()
    rate = args.rate if args.rate else _rate_hz(resolver)
    need_fr = args.family == "B"
    basis = _load_consensus(resolver) if need_fr else None
    animals = args.animals or _FAMILY_ANIMALS[args.family]

    with open(resolver.animals_yaml()) as f:
        meta_all = yaml.safe_load(f)

    out_dir = resolver.local_work() / "features" / f"{args.family}_{int(round(rate))}hz"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for animal in animals:
        stroke = datetime.datetime.strptime(
            str(meta_all[animal]["stroke_date"]).replace("-", ""), "%Y%m%d").date()
        dates = _sessions_with_all_sources(resolver, animal, args.cam, need_fr)
        (out_dir / animal).mkdir(parents=True, exist_ok=True)
        for date in dates:
            tongue = load_cleaned_trace(resolver.dlc_cleaned_trace("tongue", animal, date))
            jaw = load_cleaned_trace(resolver.dlc_cleaned_trace("jaw", animal, date))
            angle = load_tongue_angle(resolver.dlc_tongue_angle(animal, date))
            speed, _ = load_treadmill_speed(resolver.treadmill_preprocessed(animal, date))
            lr = np.load(resolver.licks_and_rewards(animal, date), allow_pickle=True)
            events = {k: lr[k] for k in _EVENT_KEYS}
            template = load_alignment_template(
                resolver._root("alignment_templates") / args.cam / animal / f"{date}.npz")
            fr_kwargs = {}
            if need_fr:
                fr_kwargs = dict(
                    fac=load_tca_factors(resolver.tca_h5(animal, date, args.cam)),
                    basis=basis,
                    x_axis=load_vqt_meta(resolver.vqt_h5(animal, date, args.cam))["x_axis"],
                )
            fm = assemble(tongue, jaw, events, template, rate_hz=rate,
                          treadmill_speed=speed, tongue_angle=angle,
                          meta={"animal": animal, "date": date}, **fr_kwargs)
            fm.table.to_parquet(out_dir / animal / f"{date}.parquet")
            rel_day = (datetime.datetime.strptime(date, "%Y%m%d").date() - stroke).days
            rows.append({"animal": animal, "date": date, "rel_day": rel_day,
                         "n_grid": fm.meta["n_grid"], "epoch": "pre" if rel_day <= 0 else "post"})
            print(f"  {args.family} {animal}/{date} rel_day={rel_day:+d} -> {fm.table.shape}")
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["animal", "date", "rel_day", "n_grid", "epoch"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nfamily {args.family}: cached {len(rows)} sessions to {out_dir} at {rate} Hz")


if __name__ == "__main__":
    main()
