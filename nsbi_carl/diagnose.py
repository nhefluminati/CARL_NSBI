"""Per-template diagnostics: is this network's density ratio usable?

    python -m nsbi_carl.diagnose --run-dir outputs/Other --name Other \\
        --target samples/4l_SR_Other.h5 --reference data/reference_sample.h5

Motivation
----------
``E_ref[r] = 1`` is the check everyone runs, and it is nearly useless on its
own. It is an average over the REFERENCE, so it is dominated by wherever the
reference is dense, and a network can satisfy it while being wrong by orders
of magnitude everywhere the target actually lives.

The dual identity is the one that bites:

    E_ref[r]      = Int p_ref * (p_t/p_ref) = Int p_t   = 1
    E_target[1/r] = Int p_t * (p_ref/p_t)   = Int p_ref = 1

Both hold for a correct ratio. The second averages over the TARGET, so it is
sensitive exactly where the first is blind. A template whose closure plots
are a disaster while ``E_ref[r]`` sits at 1.000 will normally show it here.

Two other things are checked because they make a CARL training ill-posed
rather than merely inaccurate, and neither announces itself:

* **Negative target weights.** ``data.absolute_weights`` fixes the reference
  only (by design). A target carrying negative weights makes the weighted BCE
  unbounded below: the optimiser can lower the loss indefinitely by driving
  those events to an extreme score, and the resulting ratio is meaningless.
* **Support.** ``r = p_t/p_ref`` is only defined where ``p_ref > 0``. Target
  weight sitting where the reference has no events cannot be described by any
  ratio, however well the network is trained, and a sigmoid cannot represent
  an unbounded ``r`` in any case.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py as h5
import numpy as np

from .inference import EnsembleScorer, load_reference_sample, read_record


def weight_report(name: str, w: np.ndarray) -> dict:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    neg = w < 0
    out = {
        "name": name,
        "n": int(w.size),
        "sum": float(w.sum()),
        "sum_abs": float(np.abs(w).sum()),
        "n_negative": int(neg.sum()),
        "frac_negative": float(neg.mean()) if w.size else 0.0,
        "frac_weight_negative": (
            float(np.abs(w[neg]).sum() / np.abs(w).sum()) if w.size and np.abs(w).sum() > 0 else 0.0
        ),
        "n_eff": float(w.sum() ** 2 / (w**2).sum()) if (w**2).sum() > 0 else 0.0,
    }
    return out


def ratio_report(
    scorer: EnsembleScorer,
    x_target: np.ndarray,
    w_target: np.ndarray,
    x_ref: np.ndarray,
    w_ref: np.ndarray,
    eps: float = 1e-9,
) -> dict:
    """Both closure identities, plus where the ratio saturates."""
    s_t = np.clip(scorer.score(x_target), eps, 1.0 - eps)
    s_r = np.clip(scorer.score(x_ref), eps, 1.0 - eps)
    r_t = s_t / (1.0 - s_t)
    r_r = s_r / (1.0 - s_r)

    w_t = np.asarray(w_target, dtype=np.float64).reshape(-1)
    w_r = np.asarray(w_ref, dtype=np.float64).reshape(-1)

    e_ref_r = float(np.average(r_r, weights=w_r))
    # weight by |w| so a signed target does not silently cancel
    e_tgt_inv = float(np.average(1.0 / r_t, weights=np.abs(w_t)))

    return {
        "E_ref[r]": e_ref_r,
        "E_target[1/r]": e_tgt_inv,
        "score_target": (float(s_t.min()), float(np.median(s_t)), float(s_t.max())),
        "score_ref": (float(s_r.min()), float(np.median(s_r)), float(s_r.max())),
        "r_target_q": [float(q) for q in np.quantile(r_t, [0.01, 0.5, 0.99])],
        "r_target_max": float(r_t.max()),
        "r_ref_max": float(r_r.max()),
        # Support: how much target weight sits beyond the ratio range the
        # REFERENCE actually probes. Comparing against r_t.max() would be
        # meaningless -- that is one outlier. The reference's own high
        # quantile is the range in which r was constrained by data at all;
        # target weight past it is in a region the reference barely covers,
        # where r is not identifiable.
        "r_ref_q999": float(np.quantile(r_r, 0.999)),
        "frac_target_beyond_ref_range": float(
            np.abs(w_t[r_t > np.quantile(r_r, 0.999)]).sum() / np.abs(w_t).sum()
        ),
    }


def _read_features(path: str | Path, features: list[str], weight_key: str = "weight"):
    with h5.File(path, "r") as f:
        missing = [k for k in [*features, weight_key] if k not in f]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        x = np.stack([f[k][:] for k in features], axis=1).astype(np.float64)
        w = f[weight_key][:].astype(np.float64)
    return x, w


def diagnose(run_dir, name, target_path, reference_path, device="cpu", split="all") -> dict:
    record = read_record(run_dir, name)
    scorer = EnsembleScorer(record, device=device)

    x_t, w_t = _read_features(target_path, record.features)
    x_r, w_r = load_reference_sample(reference_path, record, split=split)

    wr_t = weight_report(f"target:{Path(target_path).name}", w_t)
    rr = ratio_report(scorer, x_t, w_t, x_r, w_r)

    print(f"\n=== [{name}] TEMPLATE DIAGNOSTIC ===")
    print(f"  members={len(record.checkpoints)}  reference_unit_weights="
          f"{record.reference_unit_weights}  absolute_weights={record.absolute_weights}")

    print("\n  -- target weights --")
    print(f"    n={wr_t['n']:,}  sum={wr_t['sum']:.6g}  sum|w|={wr_t['sum_abs']:.6g}  "
          f"N_eff={wr_t['n_eff']:,.0f}")
    if wr_t["n_negative"]:
        print(f"    *** {wr_t['n_negative']:,} NEGATIVE target weights "
              f"({wr_t['frac_negative']:.2%} of events, {wr_t['frac_weight_negative']:.2%} of |weight|).")
        print("        data.absolute_weights fixes the REFERENCE only. With a signed target the")
        print("        weighted BCE is unbounded below, so this training is ill-posed, not just")
        print("        imprecise -- the ratio it produces cannot be trusted at any accuracy.")
    else:
        print("    no negative target weights")

    print("\n  -- closure identities (both must be 1) --")
    print(f"    E_ref[r]      = {rr['E_ref[r]']:.6f}   ({100*(rr['E_ref[r]']-1):+.2f}%)"
          "   <- averaged over the REFERENCE")
    print(f"    E_target[1/r] = {rr['E_target[1/r]']:.6f}   ({100*(rr['E_target[1/r]']-1):+.2f}%)"
          "   <- averaged over the TARGET")
    if abs(rr["E_target[1/r]"] - 1) > 0.05 and abs(rr["E_ref[r]"] - 1) < 0.05:
        print("    *** the reference-side identity holds while the target-side one does not:")
        print("        the ratio is normalised correctly but has the wrong SHAPE where the")
        print("        target lives. This is what a broken closure plot looks like.")

    print("\n  -- scores and ratio range --")
    print(f"    score on target    min/med/max = {rr['score_target'][0]:.4f} / "
          f"{rr['score_target'][1]:.4f} / {rr['score_target'][2]:.4f}")
    print(f"    score on reference min/med/max = {rr['score_ref'][0]:.4f} / "
          f"{rr['score_ref'][1]:.4f} / {rr['score_ref'][2]:.4f}")
    print(f"    r on target  1%/50%/99% = {rr['r_target_q'][0]:.4g} / "
          f"{rr['r_target_q'][1]:.4g} / {rr['r_target_q'][2]:.4g}   max={rr['r_target_max']:.4g}")
    beyond = rr["frac_target_beyond_ref_range"]
    print(f"    r on reference: 99.9% quantile = {rr['r_ref_q999']:.4g}  max = {rr['r_ref_max']:.4g}")
    print(f"    target |weight| with r beyond the reference's 99.9% quantile: {beyond:.2%}")
    if beyond > 0.02:
        print("    *** a real share of the target sits past the ratio range the reference")
        print("        probes, i.e. where the reference has almost no events. r is not")
        print("        identifiable there no matter how long the network trains -- the")
        print("        reference has to cover this region.")

    return {"record": record, "target_weights": wr_t, "ratio": rr}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--target", required=True, help="the template's target .h5")
    p.add_argument("--reference", required=True, help="the saved reference .h5")
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    a = p.parse_args()
    diagnose(a.run_dir, a.name, a.target, a.reference, device=a.device, split=a.split)


if __name__ == "__main__":
    main()
