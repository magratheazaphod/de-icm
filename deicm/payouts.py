"""Typical live tournament payout tables, for when the real one is not to hand.

The shape is fitted to published ladders from series that run in Austin
(RunGood at The Lodge, WSOP Circuit at TCH Social, PokerAtlas Tour), 2025-26:

- Places paid is a fraction of the field, 8-15% (RunGood ~12%, WSOP Circuit ~15%).
- The final table pays each place separately, and 1st is about 10x 9th. The step
  from 1st to 2nd is the steepest (~0.68) and the steps flatten toward 9th (~0.80).
- Below the final table, places are paid in flat bands: 10-12, 13-15, ... 25-27,
  then bands of nine. The minimum cash is about 1.75x the average entry
  (prize pool / entrants), i.e. roughly 1.5-2x the buy-in after rake.
- Between 9th and the minimum cash, payouts fall as a power of the place,
  with the exponent solved so the last band lands on the minimum cash.
"""

from __future__ import annotations

import numpy as np

FT_SIZE = 9
FT_RATIO = 10.0  # 1st / 9th
RAMP = 0.017  # each final-table step is this much flatter than the one above it
MIN_CASH = 1.75  # minimum cash, in average entries


def _ft_steps(ft_ratio: float) -> np.ndarray:
    """Successive final-table ratios (2nd/1st, 3rd/2nd, ...) whose product is 1/ft_ratio."""
    k = np.arange(FT_SIZE - 1)
    lo, hi = 1e-6, 1.0 - RAMP * (FT_SIZE - 2)
    for _ in range(100):
        r0 = (lo + hi) / 2
        if np.prod(r0 + RAMP * k) * ft_ratio > 1:
            hi = r0
        else:
            lo = r0
    return lo + RAMP * k


def bands(paid: int) -> list[tuple[int, int]]:
    """Paid places grouped as they are paid: 1..9 alone, then threes to 27, then nines."""
    out = [(p, p) for p in range(1, min(paid, FT_SIZE) + 1)]
    start = FT_SIZE + 1
    while start <= paid:
        width = 3 if start <= 27 else 9
        out.append((start, min(paid, start + width - 1)))
        start += width
    return out


def payout_table(
    entrants: int,
    paid_frac: float = 0.12,
    *,
    ft_ratio: float = FT_RATIO,
    min_cash: float = MIN_CASH,
) -> list[float]:
    """Fraction of the prize pool for each paid place, 1st first; sums to 1.

    ``paid_frac`` is the share of ``entrants`` that cash. ``min_cash`` is the
    minimum cash in average entries; it is not reachable in very small fields,
    where the bottom band is simply the flattest the shape allows.
    """
    if entrants < 1:
        raise ValueError("entrants must be at least 1")
    if not 0 < paid_frac <= 1:
        raise ValueError("paid_frac must be in (0, 1]")
    paid = min(entrants, max(1, round(paid_frac * entrants)))
    steps = _ft_steps(ft_ratio)
    ft = np.concatenate([[1.0], np.cumprod(steps)])
    groups = bands(paid)
    sizes = np.array([hi - lo + 1 for lo, hi in groups], dtype=float)

    def shape(a: float) -> np.ndarray:
        v = []
        for lo, hi in groups:
            if hi <= FT_SIZE:
                v.append(ft[lo - 1])
            else:
                places = np.arange(lo, hi + 1) / FT_SIZE
                v.append(ft[-1] * float(np.mean(places ** -a)))
        v = np.array(v)
        return v / (v * sizes).sum()

    if paid > FT_SIZE:
        target = min_cash / entrants

        def last(a: float) -> float:
            return shape(a)[-1]

        lo, hi = 0.0, 20.0
        if last(lo) <= target:
            a = lo
        elif last(hi) >= target:
            a = hi
        else:
            for _ in range(100):
                a = (lo + hi) / 2
                if last(a) > target:
                    lo = a
                else:
                    hi = a
            a = (lo + hi) / 2
    else:
        a = 0.0
    v = shape(a)
    return [float(x) for x, n in zip(v, sizes) for _ in range(int(n))]
