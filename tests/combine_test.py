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


def test_saturated_member():
    """One float32-saturated member out of 40: mean_ratio explodes (and warns),
    the robust combiners do not."""
    import warnings as _w
    s = np.full((40, 1), 0.3, dtype=np.float32)
    s[0] = np.float32(1.0)                   # sigmoid(20) in float32 is exactly 1.0
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        r_mean = _r(combine.combine_scores(s, "mean_ratio"))[0]
    assert r_mean > 1e9
    assert any("mean_ratio" in str(c.message) for c in caught), "no warning"
    for c in ("median_ratio", "trimmed_ratio", "mean_score"):
        assert _r(combine.combine_scores(s, c))[0] < 1.0, c


def test_signed_ratios_round_trip():
    """PARE scores s = r/(1+r) for r < 0 lie outside [0, 1]; the conversion
    must return r, not the ~1e12 a (0, 1) clip would produce for r < -1."""
    r = np.array([-50.0, -3.0, -1.5, -0.999, -0.5, 0.0, 0.3, 1.0, 7.0])
    with np.errstate(divide="ignore"):
        s = r / (1.0 + r)
    got = combine.ratio_from_scores(s)
    np.testing.assert_allclose(got[r != 0], r[r != 0], rtol=1e-9)
    assert abs(got[r == 0][0]) < 1e-11
    assert combine.ratio_from_scores(np.array([np.inf, -np.inf]))[0] == -1.0
    # BCE scores keep the old clipped behaviour
    np.testing.assert_allclose(combine.ratio_from_scores(np.array([0.0, 1.0]))[0], 1e-12, rtol=1e-6)
    assert combine.ratio_from_scores(np.array([1.0]))[0] > 1e11


def test_signed_combination_in_ratio_space():
    r = np.array([[-2.0, 0.5], [-1.5, 0.7], [-1.0 + 1e-9, 0.9]])
    s = r / (1.0 + r)
    for c, expect in (("mean_score", r.mean(0)), ("mean_ratio", r.mean(0)),
                      ("median_ratio", np.median(r, 0)),
                      ("trimmed_ratio", r.mean(0))):
        got = combine.ratio_from_scores(combine.combine_scores(s, c, trim=0.0))
        np.testing.assert_allclose(got, expect, rtol=1e-6, err_msg=c)
    try:
        combine.combine_scores(s, "mean_logit")
        raise AssertionError("mean_logit accepted signed ratios")
    except ValueError:
        pass
    # signed=True: same rule even when this chunk happens to be all in [0, 1]
    pos = np.array([[0.2, 0.6], [0.4, 0.8]])
    np.testing.assert_allclose(
        combine.ratio_from_scores(combine.combine_scores(pos, "mean_score", signed=True)),
        combine.ratio_from_scores(pos).mean(0), rtol=1e-12)


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
