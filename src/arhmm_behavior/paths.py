"""Path resolution. No absolute paths in source — everything routes through here.

`PathResolver` is configured from ``configs/data_sources.yaml`` (machine-specific
locations) so the same code runs on any machine by editing one YAML file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


class PathResolver:
    """Resolve source-data and output locations from a config file.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to a data-sources YAML. Defaults to ``configs/data_sources.yaml``
        at the repo root.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = _REPO_ROOT / "configs" / "data_sources.yaml"
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self._cfg: dict[str, Any] = yaml.safe_load(f)

    def _root(self, key: str) -> Path:
        if key not in self._cfg:
            raise KeyError(f"{key!r} not in {self.config_path}")
        p = Path(self._cfg[key]).expanduser()
        if not p.is_absolute():  # local_work / outputs are repo-relative
            p = _REPO_ROOT / p
        return p

    # --- FaceRhythm ---------------------------------------------------------
    def facerhythm_session(self, animal: str, date: str, cam: str = "cam2") -> Path:
        """Directory of a FaceRhythm session's ``analysis_files``."""
        return (
            self._root("facerhythm_run_root")
            / cam / animal / date / "jobNum_0" / "analysis_files"
        )

    def tca_h5(self, animal: str, date: str, cam: str = "cam2") -> Path:
        return self.facerhythm_session(animal, date, cam) / "TCA.h5"

    def vqt_h5(self, animal: str, date: str, cam: str = "cam2") -> Path:
        return self.facerhythm_session(animal, date, cam) / "VQT_Analyzer.h5"

    # --- DLC / spout outputs ------------------------------------------------
    def dlc_cleaned_trace(self, bodypart: str, animal: str, date: str) -> Path:
        """``cleaned_traces`` parquet for a bodypart (tongue|jaw)."""
        return (
            self._root("stroke_pipeline_outputs")
            / "dlc_kinematics" / "cleaned_traces" / bodypart / animal / f"{date}.parquet"
        )

    def dlc_lick_events(self, animal: str, date: str) -> Path:
        return (
            self._root("stroke_pipeline_outputs")
            / "dlc_kinematics" / "lick_events" / "tongue" / animal / f"{date}.parquet"
        )

    def licks_and_rewards(self, animal: str, date: str) -> Path:
        return (
            self._root("stroke_pipeline_outputs")
            / "spout_behavior" / "licks_and_rewards" / animal / f"{date}.npz"
        )

    def animals_yaml(self) -> Path:
        return self._root("stroke_pipeline_repo") / "configs" / "animals.yaml"

    # --- Outputs (writable) -------------------------------------------------
    def local_work(self) -> Path:
        p = self._root("local_work")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def outputs(self) -> Path:
        p = self._root("outputs")
        p.mkdir(parents=True, exist_ok=True)
        return p
