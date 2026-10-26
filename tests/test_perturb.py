"""Perturbation mode: allin and bubble_factors, checked against the exact subset DP."""

import numpy as np
import pytest

from deicm import AllInResult, BubbleFactors, allin, bubble_factors
from deicm.exact import as_float, icm_exact


def exact_eq(stacks, prizes):
    return as_float(icm_exact(stacks, prizes)[1])


STACKS = [
    [1000.0] * 5,
    [1000.0, 2500.0, 400.0, 3000.0, 700.0],
    [5.0, 1.0, 17.0, 3.0],
    [10.0, 20.0, 30.0],
]


@pytest.mark.parametrize("stacks", STACKS)
def test_winner_take_all_bubble_factors_are_one(stacks):
    n = len(stacks)
    b = bubble_factors(stacks, [100.0])
    off = ~np.eye(n, dtype=bool)
    assert np.allclose(b.bf[off], 1.0, atol=1e-12, rtol=0)
    assert np.allclose(b.required_equity[off], 0.5, atol=1e-12, rtol=0)
    assert np.allclose(b.risk_premium[off], 0.0, atol=1e-12, rtol=0)
    assert np.all(np.isnan(np.diag(b.bf)))


def test_chip_deltas_without_bust():
    stacks = [1000.0, 2500.0, 400.0, 3000.0, 700.0]
    prizes = [50.0, 30.0, 20.0]
    # seat 2 (2500) beats seat 4 (3000): loser is larger so nobody busts; a = 2500
    r = allin(stacks, prizes, 1, 3)
    assert not r.busted and r.amount == 2500.0
    total = sum(stacks)
    pool = sum(prizes)
    assert r.chip_delta[1] == pytest.approx(2500 * pool / total, abs=1e-12)
    assert r.chip_delta[3] == pytest.approx(-2500 * pool / total, abs=1e-12)
    others = [0, 2, 4]
    assert np.allclose(r.chip_delta[others], 0.0, atol=1e-12)
    assert r.stacks_after.sum() == pytest.approx(total)
    assert r.delta.sum() == pytest.approx(0.0, abs=1e-9)


def test_bust_when_everyone_paid():
    stacks = [1000.0, 2500.0, 400.0, 3000.0]
    prizes = [50.0, 30.0, 20.0, 10.0]
    r = allin(stacks, prizes, 3, 2)  # 3000 beats 400
    assert r.busted and r.amount == 400.0
    assert r.equity_after[2] == prizes[-1]
    assert r.stacks_after[2] == 0.0
    assert r.stacks_after.sum() == pytest.approx(sum(stacks))
    assert r.equity_before.sum() == pytest.approx(sum(prizes), abs=1e-9)
    assert r.equity_after.sum() == pytest.approx(sum(prizes), abs=1e-9)
    assert r.chip_ev_after[2] == prizes[-1]
    assert r.chip_ev_after.sum() == pytest.approx(sum(prizes), abs=1e-9)


def test_bust_when_not_everyone_paid():
    stacks = [1000.0, 2500.0, 400.0, 3000.0, 700.0]
    prizes = [50.0, 30.0, 20.0]
    r = allin(stacks, prizes, 0, 2)  # 1000 beats 400
    assert r.busted
    assert r.equity_after[2] == 0.0
    assert r.chip_ev_after[2] == 0.0
    assert r.equity_before.sum() == pytest.approx(100.0, abs=1e-9)
    assert r.equity_after.sum() == pytest.approx(100.0, abs=1e-9)
    assert r.chip_ev_after.sum() == pytest.approx(100.0, abs=1e-9)
    assert r.stacks_after.sum() == pytest.approx(sum(stacks))


def test_bust_with_exactly_one_prize_fewer_than_players():
    # m = n - 1: the remaining n - 1 players use every prize
    stacks = [100.0, 300.0, 50.0, 200.0]
    prizes = [60.0, 30.0, 10.0]
    r = allin(stacks, prizes, 1, 2)
    assert r.equity_after.sum() == pytest.approx(100.0, abs=1e-9)
    ref = exact_eq([100.0, 350.0, 200.0], prizes)
    assert r.equity_after[[0, 1, 3]] == pytest.approx(ref, abs=1e-12)


def test_two_players_bust_leaves_winner_first_prize():
    r = allin([100.0, 300.0], [70.0, 30.0], 1, 0)
    assert r.busted
    assert r.equity_after[0] == 30.0 and r.equity_after[1] == 70.0
    r = allin([100.0, 300.0], [70.0], 1, 0)
    assert r.equity_after[0] == 0.0 and r.equity_after[1] == 70.0
    assert r.chip_ev_after[1] == 70.0


def test_tie_busts_loser():
    r = allin([500.0, 500.0, 500.0], [50.0, 30.0], 0, 1)
    assert r.busted and r.amount == 500.0
    assert r.stacks_after.tolist() == [1000.0, 0.0, 500.0]


def test_agreement_with_exact_no_bust():
    stacks = [1000.0, 2500.0, 400.0, 3000.0, 700.0]
    prizes = [50.0, 30.0, 20.0]
    r = allin(stacks, prizes, 2, 4)  # 400 beats 700: no bust, a = 400
    assert not r.busted
    after = [1000.0, 2500.0, 800.0, 3000.0, 300.0]
    assert r.equity_before == pytest.approx(exact_eq(stacks, prizes), abs=1e-12)
    assert r.equity_after == pytest.approx(exact_eq(after, prizes), abs=1e-12)
    assert r.delta == pytest.approx(exact_eq(after, prizes) - exact_eq(stacks, prizes), abs=1e-12)


@pytest.mark.parametrize("prizes", [[50.0, 30.0, 20.0], [40.0, 30.0, 20.0, 10.0, 5.0]])
def test_agreement_with_exact_bust(prizes):
    stacks = [1000.0, 2500.0, 400.0, 3000.0, 700.0]
    r = allin(stacks, prizes, 1, 2)  # 2500 beats 400, busts it
    assert r.busted
    rest = [0, 1, 3, 4]
    reduced = [1000.0, 2900.0, 3000.0, 700.0]
    sub = prizes[: min(len(prizes), 4)]
    expected = np.empty(5)
    expected[rest] = exact_eq(reduced, sub)
    expected[2] = prizes[4] if len(prizes) >= 5 else 0.0
    assert r.equity_after == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("prizes", [[50.0, 30.0, 20.0], [100.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
def test_equal_stacks_symmetry(prizes):
    b = bubble_factors([777.0] * 5, prizes)
    off = ~np.eye(5, dtype=bool)
    for arr in (b.bf, b.required_equity, b.risk_premium):
        vals = arr[off]
        assert np.allclose(vals, vals[0], atol=1e-10, rtol=0)


def test_flat_satellite_does_not_raise():
    prizes = [10.0] * 4
    stacks = [100.0, 250.0, 40.0, 300.0]
    b = bubble_factors(stacks, prizes)
    assert isinstance(b, BubbleFactors)
    assert b.bf.shape == (4, 4)
    off = ~np.eye(4, dtype=bool)
    assert np.all(np.isnan(b.bf[off])) and np.all(np.isnan(b.required_equity[off]))
    r = allin(stacks, prizes, 0, 1)
    assert r.equity_after == pytest.approx(r.equity_before, abs=1e-9)
    assert r.equity_after.sum() == pytest.approx(40.0, abs=1e-9)
    r = allin(stacks, prizes, 3, 2)  # bust: loser keeps prizes[-1] = 10
    assert r.equity_after == pytest.approx([10.0] * 4, abs=1e-9)


def test_default_seats_and_selection():
    stacks = [float(100 + 50 * i) for i in range(14)]
    b = bubble_factors(stacks[:6], [50.0, 30.0, 20.0])
    assert b.seats == (0, 1, 2, 3, 4, 5)
    big = bubble_factors(stacks, [50.0, 30.0, 20.0], seats=[0, 13])
    assert big.seats == (0, 13) and big.bf.shape == (2, 2)
    assert np.isnan(big.bf[0, 0]) and np.isfinite(big.bf[0, 1])


def test_pairwise_matches_allin():
    stacks = [1000.0, 2500.0, 400.0, 3000.0, 700.0]
    prizes = [50.0, 30.0, 20.0]
    b = bubble_factors(stacks, prizes, seats=[3, 0])
    gain = allin(stacks, prizes, 3, 0).delta[3]
    loss = -allin(stacks, prizes, 0, 3).delta[3]
    assert b.bf[0, 1] == pytest.approx(loss / gain)
    assert b.required_equity[0, 1] == pytest.approx(loss / (gain + loss))
    assert b.risk_premium[0, 1] == pytest.approx(loss / (gain + loss) - 0.5)


def test_result_types_frozen():
    r = allin([1.0, 2.0, 3.0], [5.0, 3.0], 0, 1)
    assert isinstance(r, AllInResult)
    with pytest.raises(Exception):
        r.winner = 2  # type: ignore[misc]


@pytest.mark.parametrize("seats, msg", [
    ([0, 0], "distinct"),
    ([0, 5], "out of range"),
    ([-1, 0], "out of range"),
    ([], "empty"),
])
def test_seat_validation(seats, msg):
    with pytest.raises(ValueError, match=msg):
        bubble_factors([1.0, 2.0, 3.0], [5.0, 3.0], seats=seats)


@pytest.mark.parametrize("w, l, msg", [(0, 0, "different"), (3, 0, "out of range"),
                                       (0, -1, "out of range")])
def test_allin_validation(w, l, msg):
    with pytest.raises(ValueError, match=msg):
        allin([1.0, 2.0, 3.0], [5.0, 3.0], w, l)


# ---- dead money ----

POT_STACKS = [1000.0, 2500.0, 400.0, 3000.0, 700.0]
POT_PRIZES = [50.0, 30.0, 20.0]


def test_pot_zero_is_bit_identical():
    for w, l in [(1, 3), (0, 2), (4, 0)]:
        a, b = allin(POT_STACKS, POT_PRIZES, w, l), allin(POT_STACKS, POT_PRIZES, w, l, pot=0.0)
        for f in ("stacks_after", "equity_after", "delta", "chip_ev_after", "chip_delta"):
            assert np.array_equal(getattr(a, f), getattr(b, f))
    a, b = bubble_factors(POT_STACKS, POT_PRIZES), bubble_factors(POT_STACKS, POT_PRIZES, pot=0.0)
    assert np.array_equal(a.bf, b.bf, equal_nan=True)
    assert np.array_equal(a.required_equity, b.required_equity, equal_nan=True)


def test_larger_pot_lowers_required_equity_strictly():
    # includes the dominant chip leader (seat 4) against short stacks
    off = ~np.eye(5, dtype=bool)
    prev = bubble_factors(POT_STACKS, POT_PRIZES, pot=0.0).required_equity
    for pot in (100.0, 500.0, 2000.0):
        cur = bubble_factors(POT_STACKS, POT_PRIZES, pot=pot).required_equity
        assert np.all(cur[off] < prev[off])
        prev = cur
    b0 = bubble_factors(POT_STACKS, POT_PRIZES)
    b1 = bubble_factors(POT_STACKS, POT_PRIZES, pot=100.0)
    assert b1.required_equity[3, 2] < b0.required_equity[3, 2]


def test_equity_fold_and_pairwise_definition():
    pot = 300.0
    r = allin(POT_STACKS, POT_PRIZES, 3, 0, pot=pot)
    folded = list(POT_STACKS)
    folded[3] += pot
    assert r.equity_fold == pytest.approx(exact_eq(folded, POT_PRIZES), abs=1e-12)
    assert np.array_equal(allin(POT_STACKS, POT_PRIZES, 3, 0).equity_fold,
                          allin(POT_STACKS, POT_PRIZES, 3, 0).equity_before)
    # seat 0 (index 0) facing seat 3: V_fold from the aggressor-wins-pot branch
    b = bubble_factors(POT_STACKS, POT_PRIZES, seats=[0, 3], pot=pot)
    v_win = allin(POT_STACKS, POT_PRIZES, 0, 3, pot=pot).equity_after[0]
    v_lose = allin(POT_STACKS, POT_PRIZES, 3, 0, pot=pot).equity_after[0]
    v_fold = r.equity_fold[0]
    assert b.required_equity[0, 1] == pytest.approx(
        (v_fold - v_lose) / (v_win - v_lose), abs=1e-12)


def test_icm_call_count(monkeypatch):
    import deicm.perturb as pt
    calls = []
    real = pt.icm
    monkeypatch.setattr(pt, "icm", lambda s, p: (calls.append(1), real(s, p))[1])
    bubble_factors(POT_STACKS, POT_PRIZES, seats=[0, 1, 2])
    base = len(calls)
    calls.clear()
    bubble_factors(POT_STACKS, POT_PRIZES, seats=[0, 1, 2], pot=50.0)
    assert len(calls) == base + 6  # one extra per ordered pair


def test_pot_stacks_sum_when_no_bust():
    r = allin(POT_STACKS, POT_PRIZES, 1, 3, pot=300.0)  # 2500 vs 3000: no bust
    assert not r.busted and r.pot == 300.0
    assert r.stacks_after.sum() == pytest.approx(sum(POT_STACKS) + 300.0)
    assert r.stacks_after[1] == 2500.0 * 2 + 300.0
    assert r.stacks_after[3] == 500.0


def test_pot_matches_exact_and_bust_rule():
    r = allin(POT_STACKS, POT_PRIZES, 1, 2, pot=200.0)  # busts seat 3
    assert r.busted and r.equity_after[2] == 0.0
    reduced = [1000.0, 2500.0 + 400.0 + 200.0, 3000.0, 700.0]
    assert r.equity_after[[0, 1, 3, 4]] == pytest.approx(exact_eq(reduced, POT_PRIZES), abs=1e-12)


def test_winner_take_all_with_pot_favours_calling():
    b0 = bubble_factors([1000.0, 2000.0, 3000.0], [100.0])
    b = bubble_factors([1000.0, 2000.0, 3000.0], [100.0], pot=500.0)
    off = ~np.eye(3, dtype=bool)
    assert np.all(b.required_equity[off] < b0.required_equity[off])


@pytest.mark.parametrize("pot", [-1.0, float("nan"), float("inf"), "x"])
def test_pot_validation(pot):
    with pytest.raises(ValueError, match="pot"):
        allin(POT_STACKS, POT_PRIZES, 0, 1, pot=pot)
    with pytest.raises(ValueError, match="pot"):
        bubble_factors(POT_STACKS, POT_PRIZES, pot=pot)
