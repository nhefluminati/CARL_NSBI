"""Regression test for the stacked-ensemble backward pass.

Background: PyTorch routes a batched matmul whose inner dimension is 1 (a
batched outer product) to a JIT-compiled Triton kernel. That kernel is built
at runtime with gcc against Python.h, so on a compute node without the Python
development headers it raises CalledProcessError in the middle of training --
long after the job has been queued and the data loaded.

The single-logit output head is exactly that shape: its weight is (M, H, 1),
so matmul's backward forms grad_input as (M, B, 1) @ (M, 1, H). StackedMLP
writes the head as a broadcast multiply and a sum instead.

This test asserts no such op is dispatched during forward OR backward, so the
failure cannot come back unnoticed. It runs on CPU and needs no GPU, Triton or
toolchain.
"""
import sys
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsbi_carl.model import build_mlp  # noqa: E402
from nsbi_carl.training.vectorized import StackedMLP  # noqa: E402


class CatchOuterProductBmm(TorchDispatchMode):
    """Flags any bmm/matmul dispatched with an inner dimension of 1."""

    def __init__(self):
        self.offenders = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if "bmm" in name or "matmul" in name:
            a, b = (args + (None, None))[:2]
            if (
                isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.ndim == 3 and b.ndim == 3
                and a.shape[2] == 1 and b.shape[1] == 1
            ):
                self.offenders.append(f"{name} {tuple(a.shape)} x {tuple(b.shape)}")
        return func(*args, **(kwargs or {}))


def main():
    M, B, Fdim, H, L = 4, 64, 3, 32, 3

    model = StackedMLP(M, Fdim, L, H)
    model.init_from_seeds([10, 11, 12, 13])

    # -- 1) no batched outer product anywhere in forward or backward -----
    x = torch.randn(M, B, Fdim)
    with CatchOuterProductBmm() as guard:
        logits = model(x)
        logits.square().mean().backward()
    assert not guard.offenders, (
        "batched outer product dispatched -- this is the shape that needs the "
        "Triton JIT and breaks on nodes without Python headers:\n  "
        + "\n  ".join(guard.offenders)
    )
    print("[ok] no inner-dim-1 bmm dispatched in forward or backward")

    # shared (B, F) input path too
    model.zero_grad()
    with CatchOuterProductBmm() as guard:
        model(torch.randn(B, Fdim)).square().mean().backward()
    assert not guard.offenders, f"shared-input path dispatches: {guard.offenders}"
    print("[ok] no inner-dim-1 bmm on the shared-input (validation) path")

    # -- 2) the rewritten head is still numerically the reference MLP ----
    for m in range(M):
        torch.manual_seed([10, 11, 12, 13][m])
        ref = build_mlp(Fdim, L, H, 0.0)
        ref.load_state_dict(
            {k.removeprefix("net."): v for k, v in model.member_state_dict(m).items()}
        )
        xa = torch.randn(B, Fdim)
        got = model(xa.unsqueeze(0).expand(M, B, Fdim))[m]
        want = ref(xa).flatten()
        dev = (got - want).abs().max().item()
        assert dev < 1e-5, f"member {m} deviates from reference MLP by {dev}"
    print("[ok] stacked head matches the reference nn.Sequential for every member")

    # -- 3) gradients still reach every member independently -------------
    model.zero_grad()
    out = model(torch.randn(M, B, Fdim))
    out[0].sum().backward()                      # only member 0 in the loss
    g = model.weights[0].grad
    assert g[0].abs().sum() > 0, "member 0 got no gradient"
    assert g[1:].abs().sum() == 0, "gradient leaked across members"
    print("[ok] gradients stay confined to the member they belong to")

    print("\nALL BACKWARD CHECKS PASSED")


if __name__ == "__main__":
    main()
