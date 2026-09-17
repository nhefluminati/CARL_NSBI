"""Benchmark the training paths on synthetic data.

    python -m nsbi_carl.benchmark --events 2000000 --members 8 --epochs 5

Reports wall time per epoch for:
  * ``dataloader``  - the original DataLoader(Subset(...)) path, one member
  * ``device``      - device-resident batching, one member
  * ``vectorized``  - all members stacked into one batched matmul

Everything runs on whatever device is available, so the same script gives a
meaningful ratio on a laptop CPU and on a node full of A100s.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .data.dataset import NSBIDataset, SplitIndices
from .data.fastloader import DeviceBatches
from .model import CARL, weighted_bce_with_logits
from .training.trainer import ModelConfig, TrainerConfig
from .training.vectorized import StackedMLP, VectorizedConfig


def synthetic(n_events: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_t = n_events // 2
    x = np.concatenate([rng.normal(0.4, 1.0, (n_t, n_features)), rng.normal(0.0, 1.0, (n_events - n_t, n_features))])
    y = np.concatenate([np.ones(n_t), np.zeros(n_events - n_t)])
    w = rng.uniform(0.5, 1.5, n_events)
    dataset = NSBIDataset(
        x=x.astype(np.float32), y=y.astype(np.float32), w=w,
        sample_id=np.zeros(n_events, dtype=np.int64),
        sample_names=["synth"], feature_names=[f"f{i}" for i in range(n_features)],
    )
    dataset.mean = x.mean(0)
    dataset.std = x.std(0)
    idx = rng.permutation(n_events)
    cut = int(0.8 * n_events), int(0.9 * n_events)
    return dataset, SplitIndices(train=idx[: cut[0]], val=idx[cut[0] : cut[1]], test=idx[cut[1] :])


def bench_dataloader(dataset, splits, mc, batch_size, epochs, device, num_workers):
    from torch.utils.data import DataLoader, Subset

    model = CARL(dataset.n_features, mc.n_layers, mc.n_nodes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    loader = DataLoader(
        Subset(dataset, splits.train.tolist()), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(epochs):
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            loss = weighted_bce_with_logits(model.net(x).flatten(), y.flatten(), w.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    _sync(device)
    return (time.perf_counter() - t0) / epochs


def bench_device(dataset, splits, mc, batch_size, epochs, device):
    model = CARL(dataset.n_features, mc.n_layers, mc.n_nodes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    loader = DeviceBatches(dataset, splits.train, batch_size, shuffle=True, device=device, drop_last=True)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(epochs):
        for x, y, w in loader:
            loss = weighted_bce_with_logits(model.net(x).flatten(), y, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    _sync(device)
    return (time.perf_counter() - t0) / epochs


def bench_vectorized(dataset, splits, mc, batch_size, epochs, device, n_members):
    import torch.nn.functional as F

    model = StackedMLP(n_members, dataset.n_features, mc.n_layers, mc.n_nodes).to(device)
    model.init_from_seeds(list(range(n_members)))
    opt = torch.optim.SGD(model.parameters(), lr=1e-4, foreach=True)

    x_all = dataset.x.to(device)
    y_all = dataset.y.reshape(-1).to(device)
    w_all = dataset.w.reshape(-1).float().to(device)
    train = torch.as_tensor(splits.train, dtype=torch.long, device=device)
    boot = train.repeat(n_members, 1)
    n_train = boot.shape[1]
    n_batches = n_train // batch_size

    _sync(device)
    t0 = time.perf_counter()
    for _ in range(epochs):
        perm = torch.argsort(torch.rand(n_members, n_train, device=device), dim=1)
        for b in range(n_batches):
            idx = torch.gather(boot, 1, perm[:, b * batch_size : (b + 1) * batch_size])
            xb, yb, wb = x_all[idx], y_all[idx], w_all[idx]
            logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = ((bce * wb).sum(1) / wb.sum(1)).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    _sync(device)
    return (time.perf_counter() - t0) / epochs


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", type=int, default=500_000)
    p.add_argument("--features", type=int, default=2)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--n-nodes", type=int, default=256)
    p.add_argument("--old-batch-size", type=int, default=512)
    p.add_argument("--new-batch-size", type=int, default=32768)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--skip-dataloader", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    mc = ModelConfig(n_layers=args.n_layers, n_nodes=args.n_nodes)
    dataset, splits = synthetic(args.events, args.features)

    print(f"device={device}  events={args.events:,}  members={args.members}  "
          f"net={args.n_layers}x{args.n_nodes}\n")

    results = {}
    if not args.skip_dataloader:
        results["dataloader (1 member, bs=%d)" % args.old_batch_size] = bench_dataloader(
            dataset, splits, mc, args.old_batch_size, args.epochs, device, args.num_workers
        )
    results["device batches (1 member, bs=%d)" % args.new_batch_size] = bench_device(
        dataset, splits, mc, args.new_batch_size, args.epochs, device
    )
    results["vectorized (%d members, bs=%d)" % (args.members, args.new_batch_size)] = bench_vectorized(
        dataset, splits, mc, args.new_batch_size, args.epochs, device, args.members
    )

    width = max(len(k) for k in results)
    for name, secs in results.items():
        print(f"{name:<{width}}  {secs:8.3f} s/epoch")

    keys = list(results)
    if len(keys) >= 2:
        base = results[keys[0]]
        print()
        for k in keys[1:]:
            per_member = results[k] / (args.members if "vectorized" in k else 1)
            print(f"{k}: {base / results[k]:6.1f}x faster per epoch, "
                  f"{base / per_member:6.1f}x per member-epoch")


if __name__ == "__main__":
    main()
