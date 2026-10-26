"""High-precision exact ICM references (mpmath, 40 digits).

Used to validate :func:`deicm.icm`; see docs/SPEC.md section 6. All functions
take stacks and prizes as Python/NumPy numbers and return NumPy object arrays
of ``mpmath.mpf``. Use :func:`as_float` to round to float64.

Placement matrices have shape (n, m) with ``Q[i, r-1] = P(player i places r-th)``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from mpmath import mp, mpf

mp.dps = 40

__all__ = [
    "as_float",
    "winner_take_all",
    "two_prizes",
    "flat_equities",
    "linear_equities",
    "equal_stacks",
    "two_size_dp",
    "equal_exceptions_dp",
    "icm_exact",
]


def _mpf_list(xs) -> list:
    return [mpf(float(x)) for x in xs]


def _obj(rows) -> np.ndarray:
    a = np.empty(np.shape(rows), dtype=object)
    a[...] = rows
    return a


def as_float(a) -> np.ndarray:
    """Round an object array of mpf to float64."""
    return np.array([float(x) for x in np.ravel(a)], dtype=np.float64).reshape(np.shape(a))


def equities(Q: np.ndarray, prizes: Sequence[float]) -> np.ndarray:
    """V_i = sum_r pi_r Q[i, r] in mpmath."""
    pr = _mpf_list(prizes)
    m = Q.shape[1]
    return _obj([mp.fsum(Q[i, r] * pr[r] for r in range(m)) for i in range(Q.shape[0])])


def winner_take_all(stacks) -> np.ndarray:
    """Q[i, 0] = s_i / S. Returns (n, 1)."""
    s = _mpf_list(stacks)
    S = mp.fsum(s)
    return _obj([[x / S] for x in s])


def two_prizes(stacks) -> np.ndarray:
    """First two places: Q[i,1] = s_i/S, Q[i,2] = s_i sum_{j!=i} (s_j/S)/(S - s_j). (n, 2)."""
    s = _mpf_list(stacks)
    S = mp.fsum(s)
    terms = [(x / S) / (S - x) for x in s]
    A = mp.fsum(terms)
    return _obj([[x / S, x * (A - terms[i])] for i, x in enumerate(s)])


def flat_equities(n: int, prize) -> np.ndarray:
    """Every player is paid in every place: V_i = pi."""
    return _obj([mpf(float(prize))] * n)


def linear_equities(stacks) -> np.ndarray:
    """pi_r = n - r + 1: V_i = n + 1/2 - sum_j s_j / (s_i + s_j), j over all players."""
    s = _mpf_list(stacks)
    n = len(s)
    half = mpf(n) + mpf(1) / 2
    return _obj([half - mp.fsum(y / (x + y) for y in s) for x in s])


def equal_stacks(n: int, m: int) -> np.ndarray:
    """All stacks equal: Q[i, r] = 1/n. Returns (n, m)."""
    return _obj([[mpf(1) / n] * m for _ in range(n)])


def two_size_dp(n1: int, s1, n2: int, s2, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Two stack sizes (n1 players at s1, n2 at s2). Returns (q1, q2), each (m,):
    the placement distribution of one player of that type over places 1..m.

    State after r placements: x = number of type-1 players already placed.
    """
    s1, s2 = mpf(float(s1)), mpf(float(s2))
    n = n1 + n2
    m = min(m, n)
    P = {0: mpf(1)}
    q1, q2 = [], []
    for r in range(m):
        a1 = mpf(0)
        a2 = mpf(0)
        nxt: dict[int, mpf] = {}
        for x, pr in P.items():
            y = r - x
            r1, r2 = n1 - x, n2 - y
            D = r1 * s1 + r2 * s2
            if r1:
                f = pr * r1 * s1 / D  # P(state and next is type 1)
                a1 += f / n1
                nxt[x + 1] = nxt.get(x + 1, 0) + f
            if r2:
                f = pr * r2 * s2 / D
                a2 += f / n2
                nxt[x] = nxt.get(x, 0) + f
        q1.append(a1)
        q2.append(a2)
        P = nxt
    return _obj(q1), _obj(q2)


def equal_exceptions_dp(exc_stacks, reg_stack, n_reg: int, m: int) -> np.ndarray:
    """e <= 10 exceptional stacks plus n_reg regulars at reg_stack.

    Returns (e + 1, m): rows 0..e-1 are the exceptional players in order, the
    last row is one representative regular player.
    State: (bitmask of placed exceptionals, number of placed regulars).
    """
    ex = _mpf_list(exc_stacks)
    e = len(ex)
    if e > 10:
        raise ValueError("at most 10 exceptional players")
    sr = mpf(float(reg_stack))
    n = e + n_reg
    m = min(m, n)
    out = [[mpf(0)] * m for _ in range(e + 1)]
    P = {(0, 0): mpf(1)}
    for r in range(m):
        nxt: dict[tuple[int, int], mpf] = {}
        for (mask, k), pr in P.items():
            rem = n_reg - k
            D = rem * sr + mp.fsum(ex[j] for j in range(e) if not mask >> j & 1)
            for j in range(e):
                if not mask >> j & 1:
                    f = pr * ex[j] / D
                    out[j][r] += f
                    key = (mask | 1 << j, k)
                    nxt[key] = nxt.get(key, 0) + f
            if rem:
                f = pr * rem * sr / D
                out[e][r] += f / n_reg
                key = (mask, k + 1)
                nxt[key] = nxt.get(key, 0) + f
        P = nxt
    return _obj(out)


def icm_exact(stacks, prizes) -> tuple[np.ndarray, np.ndarray]:
    """General subset DP, n <= 14. Returns (Q, V): Q is (n, m), V is (n,)."""
    s = _mpf_list(stacks)
    n = len(s)
    m = len(prizes)
    if n > 14:
        raise ValueError("icm_exact is limited to n <= 14")
    full = 1 << n
    total = [mpf(0)] * full  # sum of stacks in mask
    for mask in range(1, full):
        low = mask & -mask
        total[mask] = total[mask ^ low] + s[low.bit_length() - 1]
    S = total[full - 1]
    P = [mpf(0)] * full
    P[0] = mpf(1)
    Q = [[mpf(0)] * m for _ in range(n)]
    for r in range(min(m, n)):
        for comb in combinations(range(n), r):
            mask = sum(1 << j for j in comb)
            pr = P[mask]
            D = S - total[mask]
            for j in range(n):
                if not mask >> j & 1:
                    f = pr * s[j] / D
                    Q[j][r] += f
                    P[mask | 1 << j] += f
    # layers beyond m are not needed for Q
    Qa = _obj(Q)
    return Qa, equities(Qa, prizes)
