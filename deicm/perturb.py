"""Perturbation mode: all-in what-ifs and bubble factors (docs/SPEC.md section 10).

Every evaluation goes through :func:`deicm.core.icm`. Seats are 0-based here.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from deicm.core import icm

__all__ = ["allin", "bubble_factors", "AllInResult", "BubbleFactors"]


@dataclass(frozen=True)
class AllInResult:
    """Outcome of one all-in. All arrays have length n, in input seat order."""

    stacks_after: np.ndarray  # busted loser has 0 chips
    equity_before: np.ndarray
    equity_after: np.ndarray  # a busted loser keeps their payout
    delta: np.ndarray  # equity_after - equity_before
    chip_ev_before: np.ndarray
    chip_ev_after: np.ndarray
    chip_delta: np.ndarray  # chip_ev_after - chip_ev_before
    winner: int
    loser: int
    amount: float  # a = min(s_winner, s_loser)
    busted: bool  # True when the loser is eliminated
    pot: float = 0.0  # dead money added to the winner's stack
    # Everyone's equity if the named winner (the aggressor) takes the pot and the loser folds:
    # stacks unchanged except winner + pot. Equals equity_before when pot == 0.
    equity_fold: np.ndarray | None = None


@dataclass(frozen=True)
class BubbleFactors:
    """k x k tables over ``seats`` (0-based seat numbers). Diagonals are NaN.

    Entry [a, b] is for the seat ``seats[a]`` at risk against ``seats[b]``.
    When the all-in changes nobody's equity (for example flat prizes with
    m = n) gain and loss are both 0 and the entries are NaN, not an error.
    """

    seats: tuple[int, ...]
    bf: np.ndarray
    required_equity: np.ndarray
    risk_premium: np.ndarray


def _seat(value, n: int, what: str) -> int:
    try:
        seat = operator.index(value)
    except TypeError:
        raise ValueError(f"{what} must be an integer seat index, got {value!r}") from None
    if not 0 <= seat < n:
        raise ValueError(f"{what} {seat} is out of range for {n} players")
    return seat


def _pot(pot) -> float:
    try:
        value = float(pot)
    except (TypeError, ValueError):
        raise ValueError(f"pot must be a number, got {pot!r}") from None
    if not np.isfinite(value) or value < 0:
        raise ValueError("pot must be finite and non-negative")
    return value


def _allin(
    s: np.ndarray, pr: np.ndarray, w: int, l: int, before, pot: float = 0.0, fold: bool = True
) -> AllInResult:
    n, m = s.size, pr.size
    amount = float(min(s[w], s[l]))
    busted = bool(s[l] <= s[w])
    after = s.copy()
    after[w] += amount + pot
    after[l] -= amount
    chip_before = s / s.sum() * pr.sum()
    eq_before = np.asarray(before.equities, dtype=float)

    if not busted:
        eq_after = np.asarray(icm(after, pr).equities, dtype=float)
        chip_after = after / after.sum() * pr.sum()
    else:
        paid = float(pr[n - 1]) if n <= m else 0.0
        rest = np.array([i for i in range(n) if i != l])
        sub = pr[: min(m, n - 1)]
        eq_after = np.empty(n)
        chip_after = np.empty(n)
        eq_after[l] = paid
        chip_after[l] = paid
        if rest.size == 1:  # one player left holds the whole first prize
            eq_after[rest] = sub[0]
            chip_after[rest] = sub[0]
        else:
            eq_after[rest] = np.asarray(icm(after[rest], sub).equities, dtype=float)
            chip_after[rest] = after[rest] / after[rest].sum() * sub.sum()
        after[l] = 0.0  # reported as 0 chips; the reduced field excluded them above

    if pot == 0.0 or not fold:
        eq_fold = eq_before  # exact when pot == 0; skipped when the caller does not need it
    else:
        folded = s.copy()
        folded[w] += pot
        eq_fold = np.asarray(icm(folded, pr).equities, dtype=float)

    return AllInResult(
        stacks_after=after,
        equity_before=eq_before,
        equity_after=eq_after,
        delta=eq_after - eq_before,
        chip_ev_before=chip_before,
        chip_ev_after=chip_after,
        chip_delta=chip_after - chip_before,
        winner=w,
        loser=l,
        amount=amount,
        busted=busted,
        pot=pot,
        equity_fold=eq_fold,
    )


def allin(
    stacks: Sequence[float],
    prizes: Sequence[float],
    winner: int,
    loser: int,
    pot: float = 0.0,
) -> AllInResult:
    """What happens to everyone's equity if ``winner`` beats ``loser`` all-in.

    The amount at risk is min(s_winner, s_loser). If the loser's stack is no
    larger than the winner's, the loser busts (ties bust the loser): they
    receive prizes[n-1] when n <= m and 0 otherwise, and the other n-1 players
    are re-evaluated with prizes[:min(m, n-1)]. Seats are 0-based.

    ``pot`` is dead money in the middle: the winner gains amount + pot, the
    loser loses amount, and nobody else's stack changes (listed stacks are
    already net of what the listed players posted).
    """
    s = np.asarray(stacks, dtype=np.float64)
    pr = np.asarray(prizes, dtype=np.float64)
    pot = _pot(pot)
    before = icm(s, pr)  # validates stacks and prizes
    n = s.size
    w = _seat(winner, n, "winner")
    l = _seat(loser, n, "loser")
    if w == l:
        raise ValueError("winner and loser must be different seats")
    return _allin(s, pr, w, l, before, pot)


def bubble_factors(
    stacks: Sequence[float],
    prizes: Sequence[float],
    seats: Sequence[int] | None = None,
    pot: float = 0.0,
) -> BubbleFactors:
    """Bubble factors and required equity for ordered pairs of ``seats``.

    gain[i, j] is i's equity gain doubling through j and loss[i, j] is i's
    equity loss when losing to j, both for amount min(s_i, s_j), with the rest
    of the field held fixed. bf = loss / gain, required_equity = loss / (gain +
    loss), risk_premium = required_equity - 0.5. A 0/0 pair (nobody's equity
    moves beyond 1e-9 of the pool, as with flat prizes and m = n) gives NaN. Default seats: all if
    n <= 12, else the first 9. Cost is about two ICM evaluations per ordered pair.
    """
    s = np.asarray(stacks, dtype=np.float64)
    pr = np.asarray(prizes, dtype=np.float64)
    pot = _pot(pot)
    before = icm(s, pr)
    n = s.size
    if seats is None:
        chosen = list(range(n if n <= 12 else 9))
    else:
        chosen = [_seat(x, n, "seat") for x in seats]
        if len(set(chosen)) != len(chosen):
            raise ValueError("seats must be distinct")
        if not chosen:
            raise ValueError("seats must not be empty")
    k = len(chosen)
    gain = np.full((k, k), np.nan)
    loss = np.full((k, k), np.nan)
    for a, i in enumerate(chosen):
        for b, j in enumerate(chosen):
            if a == b:
                continue
            # i facing an all-in from j: win and lose branches, and j-as-aggressor fold branch
            win = _allin(s, pr, i, j, before, pot, fold=False)
            lose = _allin(s, pr, j, i, before, pot)
            v_fold = lose.equity_fold[i]
            gain[a, b] = win.equity_after[i] - v_fold
            loss[a, b] = v_fold - lose.equity_after[i]
    # Equity moves below this are rounding noise (the 0/0 case); report NaN, not garbage.
    dead = (np.abs(gain) + np.abs(loss)) <= 1e-9 * pr.sum()
    gain = np.where(dead, np.nan, gain)
    loss = np.where(dead, np.nan, loss)
    with np.errstate(divide="ignore", invalid="ignore"):
        bf = loss / gain
        req = loss / (gain + loss)
    return BubbleFactors(
        seats=tuple(chosen), bf=bf, required_equity=req, risk_premium=req - 0.5
    )
