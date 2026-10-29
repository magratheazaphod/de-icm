import numpy as np
import pytest

from deicm.payouts import bands, payout_table

# RunGood Summer Series $600 Main Event at The Lodge, June 2026: 582 entries,
# 71 paid, $292,746 pool (PokerNews payouts page). Final table and min cash.
LODGE_POOL = 292_746
LODGE_FT = [55416, 38843, 24974, 17772, 13642, 11213, 9256, 7380, 5537]
LODGE_MIN = 871


@pytest.mark.parametrize("n", [2, 9, 20, 75, 582, 2090, 10_000])
@pytest.mark.parametrize("frac", [0.08, 0.12, 0.15])
def test_sums_to_one_and_never_increases(n, frac):
    t = np.array(payout_table(n, frac))
    assert t.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(t) <= 1e-15)
    assert len(t) == min(n, max(1, round(frac * n)))


def test_first_is_ten_times_ninth():
    t = payout_table(600, 0.12)
    assert t[0] / t[8] == pytest.approx(10.0, rel=1e-6)


def test_min_cash_in_average_entries():
    n = 600
    t = payout_table(n, 0.12)
    assert t[-1] * n == pytest.approx(1.75, rel=1e-6)


def test_bands_after_final_table():
    assert bands(40)[9:] == [(10, 12), (13, 15), (16, 18), (19, 21), (22, 24), (25, 27), (28, 36), (37, 40)]
    t = payout_table(1000, 0.12)
    assert t[9] == t[10] == t[11] > t[12]


def test_matches_lodge_rungood_ladder():
    t = payout_table(582, 71 / 582)
    for model, real in zip(t[:9], LODGE_FT):
        assert model == pytest.approx(real / LODGE_POOL, rel=0.2)
    assert t[0] == pytest.approx(LODGE_FT[0] / LODGE_POOL, rel=0.05)
    assert t[-1] == pytest.approx(LODGE_MIN / LODGE_POOL, rel=0.05)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        payout_table(0)
    with pytest.raises(ValueError):
        payout_table(100, 0)
