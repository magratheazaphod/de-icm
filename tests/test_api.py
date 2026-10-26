"""Validation, invariances, grid override and exact-module consistency."""

from __future__ import annotations

import numpy as np
import pytest

from deicm import ICMResult, exact, icm


def test_result_fields():
    s = [5.0, 3.0, 2.0, 1.0]
    res = icm(s, [10.0, 5.0, 1.0])
    assert isinstance(res, ICMResult)
    assert res.placements.shape == (4, 3)
    assert res.equities.shape == res.itm.shape == res.chip_ev.shape == res.delta.shape == (4,)
    np.testing.assert_allclose(res.chip_ev, np.array(s) / 11 * 16, rtol=1e-15)
    np.testing.assert_allclose(res.delta, res.equities - res.chip_ev)
    np.testing.assert_allclose(res.itm, res.placements.sum(1))
    np.testing.assert_allclose(res.equities.sum(), 16.0, rtol=1e-13)
    with pytest.raises(Exception):
        res.equities = res.equities  # frozen dataclass


@pytest.mark.parametrize(
    "stacks,prizes",
    [
        ([1.0], [1.0]),  # fewer than 2 players
        ([1.0, 0.0], [1.0]),  # zero stack
        ([1.0, -2.0], [1.0]),  # negative stack
        ([1.0, float("nan")], [1.0]),
        ([1.0, float("inf")], [1.0]),
        ([1.0, 2.0], []),  # empty prizes
        ([1.0, 2.0], [1.0, 2.0]),  # increasing prizes
        ([1.0, 2.0, 3.0], [3.0, 1.0, 2.0]),  # increasing somewhere
        ([1.0, 2.0], [1.0, 0.0]),  # zero prize
        ([1.0, 2.0], [-1.0]),  # negative prize
        ([1.0, 2.0], [3.0, 2.0, 1.0]),  # m > n
        ([1.0, 1e9 * 1.001], [1.0]),  # ratio above 1e9
    ],
)
def test_validation_errors(stacks, prizes):
    with pytest.raises(ValueError):
        icm(stacks, prizes)


def test_ratio_exactly_rho_is_accepted():
    res = icm([1.0, 1e9, 5.0], [2.0, 1.0])
    assert np.isfinite(res.placements).all()


def test_bad_h():
    with pytest.raises(ValueError):
        icm([1.0, 2.0], [1.0], h=0.3)


@pytest.mark.parametrize("n,expect", [(2, (1 / 8, 250)), (50, (1 / 8, 250)), (51, (1 / 16, 500)),
                                      (200, (1 / 16, 500)), (201, (1 / 32, 1000)),
                                      (1000, (1 / 32, 1000))])  # fmt: skip
def test_default_grid(n, expect):
    s = np.linspace(1, 3, n)
    assert icm(s, [1.0]).grid == expect


def test_h_override():
    rng = np.random.default_rng(0)
    s = np.exp(rng.normal(0, 1.5, 12))
    pr = np.array([8.0, 4.0, 2.0, 1.0])
    base = icm(s, pr)
    for h in (1 / 4, 1 / 16, 1 / 64):
        res = icm(s, pr, h=h)
        assert res.grid == (h, round(31.25 / h))
    fine = icm(s, pr, h=1 / 64)
    assert np.abs(fine.placements - base.placements).max() < 1e-13
    coarse = icm(s, pr, h=31.25 / 24)  # M = 24: visibly less accurate, but still sane
    err = np.abs(coarse.placements - base.placements).max()
    assert 1e-9 < err < 0.5


@pytest.mark.parametrize("c", [1e-7, 3.0, 4096.0, 1e9])
def test_scale_invariance(c):
    rng = np.random.default_rng(1)
    s = np.exp(rng.normal(0, 2, 40))
    pr = np.linspace(10, 1, 6)
    a = icm(s, pr)
    b = icm(s * c, pr)
    assert np.abs(a.placements - b.placements).max() < 1e-14
    np.testing.assert_allclose(a.equities, b.equities, rtol=1e-13)


def test_input_order_and_determinism():
    rng = np.random.default_rng(2)
    s = np.exp(rng.normal(0, 1, 60))
    pr = np.array([5.0, 3.0, 2.0, 1.0])
    a = icm(s, pr)
    b = icm(s, pr)
    assert np.array_equal(a.placements, b.placements)
    perm = rng.permutation(60)
    c = icm(s[perm], pr)
    assert np.abs(c.placements - a.placements[perm]).max() < 1e-14
    # inputs may be lists, tuples or arrays and are not modified
    s2 = s.copy()
    icm(tuple(s), list(pr))
    icm(s2, pr)
    assert np.array_equal(s, s2)


def test_two_players_closed_form():
    res = icm([3.0, 1.0], [1.0, 0.25])
    np.testing.assert_allclose(res.placements, [[0.75, 0.25], [0.25, 0.75]], atol=1e-15)


# ---- exact module -----------------------------------------------------------------


def _f(a):
    return exact.as_float(a)


def test_subset_dp_matches_closed_forms():
    rng = np.random.default_rng(3)
    for n in (2, 5, 9):
        s = np.exp(rng.normal(0, 2, n))
        # winner-take-all and two prizes
        Q, _ = exact.icm_exact(s, [1.0, 1.0])
        np.testing.assert_allclose(_f(Q), _f(exact.two_prizes(s)), rtol=0, atol=1e-30)
        Q1, _ = exact.icm_exact(s, [1.0])
        np.testing.assert_allclose(_f(Q1), _f(exact.winner_take_all(s)), atol=1e-30)
        # linear prizes
        lin = np.arange(n, 0, -1.0)
        _, V = exact.icm_exact(s, lin)
        np.testing.assert_allclose(_f(V), _f(exact.linear_equities(s)), rtol=1e-30)
        # flat prizes, full matrix doubly stochastic
        Qf, Vf = exact.icm_exact(s, np.full(n, 2.5))
        np.testing.assert_allclose(_f(Vf), 2.5, rtol=1e-30)
        np.testing.assert_allclose(_f(Qf).sum(0), 1.0, atol=1e-30)
        np.testing.assert_allclose(_f(Qf).sum(1), 1.0, atol=1e-30)
        # equal stacks
        Qe, _ = exact.icm_exact(np.full(n, 4.0), lin)
        np.testing.assert_allclose(_f(Qe), 1 / n, atol=1e-30)


def test_subset_dp_limit():
    with pytest.raises(ValueError):
        exact.icm_exact(np.ones(15), [1.0])


def test_grouped_dps_match_subset_dp():
    pr = np.array([9.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01])
    # two sizes
    s = np.array([1.0] * 4 + [5.0] * 6)
    Q, _ = exact.icm_exact(s, pr)
    q1, q2 = exact.two_size_dp(4, 1.0, 6, 5.0, 10)
    np.testing.assert_allclose(_f(q1), _f(Q[0]), atol=1e-30)
    np.testing.assert_allclose(_f(q2), _f(Q[9]), atol=1e-30)
    # equal with exceptions
    exc = [0.3, 2.0, 11.0]
    s = np.array(exc + [1.0] * 7)
    Q, _ = exact.icm_exact(s, pr)
    R = exact.equal_exceptions_dp(exc, 1.0, 7, 10)
    np.testing.assert_allclose(_f(R[:3]), _f(Q[:3]), atol=1e-30)
    np.testing.assert_allclose(_f(R[3]), _f(Q[3]), atol=1e-30)


def test_icm_matches_subset_dp_random():
    rng = np.random.default_rng(4)
    for n in (4, 8, 12):
        s = np.exp(rng.normal(0, 2.5, n))
        pr = np.sort(rng.random(n))[::-1] + 0.01
        Q, V = exact.icm_exact(s, pr)
        res = icm(s, pr)
        assert np.abs(res.placements - _f(Q)).max() < 1e-13
        assert (np.abs(res.equities - _f(V)) / _f(V)).max() < 1e-13
