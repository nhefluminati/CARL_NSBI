"""Pipeline step: build an NSBIDataset from target/reference .h5 files.

Reference handling
------------------
The reference sample is the common denominator of every CARL network in the
analysis: S, B, SBI, qq and SBI_EW are all trained against it, so any
difference in the reference between two trainings shows up directly as a
bias in the ratio of their outputs. Three mechanisms keep it fixed:

* Its train/val/test split depends only on ``(seed, sample_name)`` and the
  sample's own event count — never on which target is present (see
  ``splitting.py``).
* ``save_reference`` writes the constructed reference sample, *including its
  split assignment*, to a single file.
* ``load_reference`` rebuilds from that file instead of from
  ``reference_paths``, which pins the reference across runs even if the seed,
  the split fractions or the reference inputs change.

``absolute_weights`` takes ``|w|`` of the reference weights only. Negative
weights (from NLO/MC@NLO-style samples) are a problem for a classifier
trained with a weighted BCE: they enter the loss with the wrong sign and can
make it unbounded below. The target keeps its signed weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py as h5
import numpy as np

from .dataset import NSBIDataset

REFERENCE_CACHE_VERSION = 1


class DatasetBuilder:
    """Reads the requested features and weights from .h5 samples.

    * Events from ``target_paths``   -> label 1
    * Events from ``reference_paths`` -> label 0
    * ``features`` fixes both the selection AND the column order of the
      network inputs; it is written to the run record by the pipeline.
    * ``weight_key`` is the name of the weight dataset inside each file.
    * ``absolute_weights`` replaces reference weights with their absolute
      value. Never applied to the target.
    * ``load_reference`` / ``save_reference`` are paths to a reference cache
      file (see module docstring).
    """

    def __init__(
        self,
        target_paths: list[str],
        reference_paths: list[str],
        features: list[str],
        weight_key: str = "weight",
        absolute_weights: bool = False,
        load_reference: str | None = None,
        save_reference: str | None = None,
    ):
        if isinstance(target_paths, str):
            target_paths = [target_paths]
        if isinstance(reference_paths, str):
            reference_paths = [reference_paths]
        self.target_paths = [str(p) for p in target_paths]
        self.reference_paths = [str(p) for p in reference_paths]
        self.features = list(features)
        self.weight_key = weight_key
        self.absolute_weights = bool(absolute_weights)
        self.load_reference = str(load_reference) if load_reference else None
        self.save_reference = str(save_reference) if save_reference else None

        # Reference samples are recorded by basename (reference_sample_scales,
        # the cache's sample list), so two reference files sharing a basename
        # would silently collapse to one entry and corrupt the run record.
        # A basename shared between the TARGET and the reference is fine and
        # deliberate -- an NSBI reference normally contains the target process
        # so that it covers its support -- so only the reference is checked.
        ref_names = [Path(p).name for p in self.reference_paths]
        dupes = sorted({n for n in ref_names if ref_names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"reference_paths contains repeated file name(s) {dupes}. Reference samples "
                "are recorded by name, so these would overwrite each other in the run record. "
                "Rename them, or list the file once."
            )

    # ------------------------------------------------------------------
    def _read_sample(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        with h5.File(path, "r") as f:
            missing = [k for k in self.features + [self.weight_key] if k not in f]
            if missing:
                raise KeyError(f"{path} is missing keys {missing}")
            x = np.stack([f[k][:] for k in self.features], axis=1).astype(np.float64)
            w = f[self.weight_key][:].astype(np.float64)
        if len(w) != len(x):
            raise ValueError(f"{path}: weight length != feature length")
        return x, w

    # ------------------------------------------------------------------
    def build(self) -> NSBIDataset:
        xs, ws, ys, sids, names, labels = [], [], [], [], [], []
        sid = 0

        for path in self.target_paths:
            x, w = self._read_sample(path)
            xs.append(x)
            ws.append(w)
            ys.append(np.ones(len(w)))
            sids.append(np.full(len(w), sid, dtype=np.int64))
            labels.append(np.full(len(w), -1, dtype=np.int8))
            names.append(Path(path).name)
            sid += 1

        if self.load_reference:
            ref = load_reference_cache(self.load_reference, self.features)
            n_ref_samples = len(ref["sample_names"])
            xs.append(ref["x"])
            ws.append(ref["w"])
            ys.append(np.zeros(len(ref["w"])))
            sids.append(ref["sample_id"] + sid)
            labels.append(ref["split_label"])
            names.extend(ref["sample_names"])
            sid += n_ref_samples
        else:
            for path in self.reference_paths:
                x, w = self._read_sample(path)
                if self.absolute_weights:
                    w = np.abs(w)
                xs.append(x)
                ws.append(w)
                ys.append(np.zeros(len(w)))
                sids.append(np.full(len(w), sid, dtype=np.int64))
                labels.append(np.full(len(w), -1, dtype=np.int8))
                names.append(Path(path).name)
                sid += 1

        return NSBIDataset(
            x=np.concatenate(xs, axis=0),
            y=np.concatenate(ys),
            w=np.concatenate(ws),
            sample_id=np.concatenate(sids),
            sample_names=names,
            feature_names=self.features,
            split_label=np.concatenate(labels),
        )


# ---------------------------------------------------------------------------
def save_reference_cache(path: str | Path, dataset: NSBIDataset, absolute_weights: bool) -> dict:
    """Write the reference events, with their split assignment, to one file.

    Called by the pipeline *after* splitting but *before* reweighting, so the
    stored weights are the raw ones read from the inputs (with ``|w|`` already
    applied if configured). Reweighting is relative to the target yield and so
    must be recomputed per run, not cached.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y = dataset.y.numpy().reshape(-1)
    ref = np.flatnonzero(y == 0.0)
    if ref.size == 0:
        raise ValueError("No reference events to save.")

    names = [dataset.sample_names[i] for i in sorted(set(dataset.sample_id[ref].tolist()))]
    remap = {old: new for new, old in enumerate(sorted(set(dataset.sample_id[ref].tolist())))}
    local_sid = np.array([remap[s] for s in dataset.sample_id[ref]], dtype=np.int64)

    with h5.File(path, "w") as f:
        for i, name in enumerate(dataset.feature_names):
            f[name] = dataset.x.numpy()[ref][:, i].astype(np.float64)
        f["weight"] = dataset.w.numpy().reshape(-1)[ref].astype(np.float64)
        f["sample_id"] = local_sid
        f["split_label"] = dataset.split_label[ref]
        f.attrs["version"] = REFERENCE_CACHE_VERSION
        f.attrs["features"] = json.dumps(list(dataset.feature_names))
        f.attrs["sample_names"] = json.dumps(names)
        f.attrs["absolute_weights"] = bool(absolute_weights)
        f.attrs["weight_key"] = "weight"

    return {
        "path": str(path),
        "n_events": int(ref.size),
        "samples": names,
        "absolute_weights": bool(absolute_weights),
    }


def load_reference_cache(path: str | Path, features: list[str]) -> dict:
    """Read a reference cache file written by :func:`save_reference_cache`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference cache {path} does not exist.")

    with h5.File(path, "r") as f:
        stored = json.loads(f.attrs["features"])
        if list(stored) != list(features):
            raise ValueError(
                f"Reference cache {path} was built with features {stored}, "
                f"but this run asks for {list(features)}. The feature list and its ORDER "
                "must match, or the network would be fed permuted inputs."
            )
        x = np.stack([f[k][:] for k in features], axis=1).astype(np.float64)
        w = f[f.attrs.get("weight_key", "weight")][:].astype(np.float64)
        sample_id = f["sample_id"][:].astype(np.int64)
        split_label = f["split_label"][:].astype(np.int8)
        names = json.loads(f.attrs["sample_names"])

    if np.any(split_label < 0):
        raise ValueError(f"Reference cache {path} contains unassigned events.")
    return {"x": x, "w": w, "sample_id": sample_id, "split_label": split_label, "sample_names": list(names)}
