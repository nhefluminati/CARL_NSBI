"""YAML configuration handling.

Two files are involved per run:
  1. The *pipeline config* the user writes (see configs/example.yaml). It is the
     single source of truth for everything configurable.
  2. The *run record* (``<output_dir>/run_config_<run_name>.yaml``), created at
     the start of training and updated by each pipeline step. It stores the
     feature list/order, the fitted mean/std of the scaler, and every
     reweighting scale factor, so any trained network can be re-applied later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class RunRecord:
    """Mutable record of everything a run fixes at training time.

    Steps write into it via ``update`` and it is flushed to disk after every
    update, so a crashed run still leaves a usable record behind.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = {}
        if self.path.exists():
            with open(self.path, "r") as f:
                self.data = yaml.safe_load(f) or {}

    def update(self, **entries: Any) -> None:
        self.data.update(_to_builtin(entries))
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            yaml.safe_dump(self.data, f, sort_keys=False)


def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain python for YAML."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
