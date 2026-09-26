"""Checks on nsbi_carl.combine (numpy only, no training needed)."""
import importlib.util
from pathlib import Path

import numpy as np

# Load the module directly so this test does not need torch/lightning.
_spec = importlib.util.spec_from_file_location(
    "combine", Path(__file__).resolve().parents[1] / "nsbi_carl" / "combine.py"
)
combine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(combine)


def _r(s):
    return s / (1.0 - s)


def test_single_member_is_identity():
    s = np.array([[0.1, 0.5, 0.9]])
    for c in combine.COMBINERS:
        np.testing.assert_allclose(combine.combine_scores(s, c), s[0])


def test_identical_members_agree_for_every_combiner():
    s = np.tile(np.array([0.2, 0.5, 0.8]), (7, 1))
    for c in combine.COMBINERS:
        np.testing.assert_allclose(combine.combine_scores(s, c), s[0], rtol=1e-12)


def test_known_values():
    s = np.array([[0.5], [0.8]])            # r = 1 and 4
    got = {c: _r(combine.combine_scores(s, c))[0] for c in combine.COMBINERS}
    assert np.isclose(got["mean_score"], _r(0.65))       # 1.857
    assert np.isclose(got["mean_ratio"], 2.5)
    assert np.isclose(got["mean_logit"], 2.0)            # sqrt(1*4)
    assert np.isclose(got["median_ratio"], _r(0.65))     # median of 2 = mean


def test_convexity_ordering():
    """mean_score <= mean_logit <= mean_ratio whenever members disagree."""
    rng = np.random.default_rng(0)
    s = rng.uniform(0.05, 0.95, size=(20, 1000))
    ms = combine.combine_scores(s, "mean_score")
    ml = combine.combine_scores(s, "mean_logit")
    mr = combine.combine_scores(s, "mean_ratio")
    assert np.all(_r(ms) <= _r(mr) + 1e-12)
    assert np.all(_r(ml) <= _r(mr) + 1e-12)


def test_trimmed_ratio_limits():
    rng = np.random.default_rng(1)
    s = rng.uniform(0.05, 0.95, size=(11, 50))
    np.testing.assert_allclose(combine.combine_scores(s, "trimmed_ratio", trim=0.0),
                               combine.combine_scores(s, "mean_ratio"))
    s[0] = 1 - 1e-9                          # one diverging member
    tr = _r(combine.combine_scores(s, "trimmed_ratio", trim=0.1))
    assert np.all(tr < 1e3)


def test_rejects_unknown():
    for bad in (("mean", 0.1), ("trimmed_ratio", 0.5)):
        try:
            combine.validate_combiner(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
