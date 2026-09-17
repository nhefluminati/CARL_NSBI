"""Device-resident batching.

The NSBI datasets are *tabular and small*: a few float features per event,
everything already in RAM. The standard ``DataLoader(Subset(dataset, idx))``
path therefore spends almost all of its time in Python: one ``__getitem__``
call per event, a ``default_collate`` per batch, and (with ``num_workers>0``)
a pickle round trip per batch through a queue. For a 10M-event train split at
batch size 512 that is ~10M Python calls and ~20k IPC hops *per epoch*, to
feed a network whose forward pass is two matmuls.

:class:`DeviceBatches` removes that layer entirely. The split is uploaded to
the GPU once as three contiguous tensors; an epoch is then a single
``torch.randperm`` on the device plus tensor slicing. No workers, no pinning,
no collate, no host-device copy in the training loop.

It quacks like a ``DataLoader`` (``__iter__`` / ``__len__``), which is all
``lightning.Trainer`` requires, and it shards itself by DDP rank so multi-GPU
runs see disjoint events.
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import NSBIDataset, SplitIndices


class DeviceBatches:
    """Iterable of ``(x, y, w)`` batches served from device-resident tensors.

    Parameters
    ----------
    dataset, indices
        Source dataset and the event indices belonging to this loader
        (train split, bootstrap resample, val split, ...).
    batch_size
        Number of events per batch. Tabular MLPs want this *large* — see the
        note in ``configs/example.yaml``.
    shuffle
        Reshuffle the indices every epoch (train) or keep the fixed order
        (val/test).
    device
        Where the tensors live. ``None`` keeps them on the CPU, which is
        still much faster than the DataLoader path.
    drop_last
        Drop a trailing partial batch. Keep it ``False`` for validation so
        every event is scored.
    rank, world_size
        DDP sharding. Each rank takes ``indices[rank::world_size]``.
    """

    def __init__(
        self,
        dataset: NSBIDataset,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        device: torch.device | str | None = None,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        idx = np.asarray(indices, dtype=np.int64)
        if world_size > 1:
            idx = idx[rank::world_size]

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        # One gather + one transfer, done once for the whole run.
        sel = torch.as_tensor(idx)
        self.x = dataset.x.index_select(0, sel).contiguous().to(self.device, non_blocking=True)
        self.y = dataset.y.index_select(0, sel).reshape(-1).contiguous().to(self.device, non_blocking=True)
        self.w = (
            dataset.w.index_select(0, sel).reshape(-1).float().contiguous().to(self.device, non_blocking=True)
        )
        self.n = self.x.shape[0]

    # -- DataLoader-compatible surface -------------------------------------
    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed + self._epoch)
            order = torch.randperm(self.n, generator=g, device=self.device)
            self._epoch += 1
        else:
            order = None

        limit = len(self) * self.batch_size if self.drop_last else self.n
        for start in range(0, limit, self.batch_size):
            stop = min(start + self.batch_size, self.n)
            if order is None:
                sl = slice(start, stop)
                yield self.x[sl], self.y[sl], self.w[sl]
            else:
                sel = order[start:stop]
                yield self.x.index_select(0, sel), self.y.index_select(0, sel), self.w.index_select(0, sel)


def resolve_device(gpus: list[int] | None) -> torch.device:
    """First configured GPU if usable, else CPU."""
    if gpus and torch.cuda.is_available():
        return torch.device(f"cuda:{gpus[0]}")
    return torch.device("cpu")


def split_tensors(
    dataset: NSBIDataset, splits: SplitIndices, which: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(x, y, w)`` for one named split, on ``device``."""
    idx = torch.as_tensor(getattr(splits, which), dtype=torch.int64)
    x = dataset.x.index_select(0, idx).contiguous().to(device)
    y = dataset.y.index_select(0, idx).reshape(-1).contiguous().to(device)
    w = dataset.w.index_select(0, idx).reshape(-1).float().contiguous().to(device)
    return x, y, w
