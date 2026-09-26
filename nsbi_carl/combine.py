"""Combining the members of a CARL ensemble into one score.

Each member returns a classifier score ``s_m``; the likelihood needs one density
ratio ``r = s/(1-s)`` per event. *Where* the members are averaged matters,
because ``r(s)`` is convex: averaging scores and then converting always gives a
smaller ratio than averaging the members' ratios, and the gap grows with how
much the members disagree. Members disagree most where the reference is thin
and the true ratio is large, so the choice decides how those regions come out.

Available combiners (``ensemble.combiner`` in the pipeline config):

``mean_score``    mean of s_m, then r = s/(1-s). The historical behaviour.
``mean_ratio``    arithmetic mean of r_m. Unbiased for r if each member is.
``mean_logit``    mean of log r_m, i.e. the geometric mean of the ratios.
                  Symmetric in r <-> 1/r; the natural average for a quantity
                  whose error is multiplicative.
``median_ratio``  median of r_m (= median of s_m, since r is monotone). Robust
                  to a few diverging members.
``trimmed_ratio`` mean of r_m after dropping the ``combiner_trim`` fraction of
                  members at EACH end, per event. Between mean and median.

Every combiner returns a *score* in (0, 1), converted back with s = r/(1+r),
so everything downstream that computes ``s/(1-s)`` keeps working unchanged.

Whatever the combiner, the ensemble mean ratio is no longer normalised to
E_ref[r] = 1 by construction; check (and correct) it downstream as before.
"""

from __future__ import annotations

import numpy as np

COMBINERS = ("mean_score", "mean_ratio", "mean_logit", "median_ratio", "trimmed_ratio")
DEFAULT_COMBINER = "mean_score"
_EPS = 1e-12


def validate_combiner(name: str, trim: float = 0.0) -> str:
    if name not in COMBINERS:
        raise ValueError(f"unknown ensemble combiner {name!r}; choose one of {COMBINERS}")
    if name == "trimmed_ratio" and not 0.0 <= trim < 0.5:
        raise ValueError(f"combiner_trim must be in [0, 0.5), got {trim}")
    return name


def combine_scores(
    member_scores: np.ndarray, combiner: str = DEFAULT_COMBINER, trim: float = 0.1
) -> np.ndarray:
    """Collapse an ``(n_members, n_events)`` score matrix to ``(n_events,)``."""
    validate_combiner(combiner, trim)
    s = np.asarray(member_scores, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]
    if s.shape[0] == 1 or combiner == "mean_score":
        return s.mean(axis=0)

    s = np.clip(s, _EPS, 1.0 - _EPS)
    if combiner == "median_ratio":
        return np.median(s, axis=0)          # r monotone in s: same event ranking

    if combiner == "mean_logit":
        logit = np.log(s) - np.log1p(-s)     # = log r_m, stable near 0 and 1
        return 1.0 / (1.0 + np.exp(-logit.mean(axis=0)))

    r = s / (1.0 - s)
    if combiner == "mean_ratio":
        r_bar = r.mean(axis=0)
    else:  # trimmed_ratio
        k = int(np.floor(trim * r.shape[0]))
        r_sorted = np.sort(r, axis=0)
        r_bar = r_sorted[k : r.shape[0] - k].mean(axis=0)
    return r_bar / (1.0 + r_bar)
