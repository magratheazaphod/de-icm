"""DE-ICM: near-exact ICM placement probabilities (Inariba, arXiv:2610.06028).

Implements Algorithm 1 of the paper as described in docs/SPEC.md: exp-exp
trapezoidal quadrature of the Poisson-binomial integral representation, with
the full product built by direct convolution and the leave-one-out polynomials
recovered by direction-switched, clamped recurrences.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["icm", "ICMResult", "RHO_MAX"]

RHO_MAX = 1e9  # largest admissible stack ratio
HEADROOM = 20.0  # normalization s~ = s / (HEADROOM * s_max)
X_LEFT = -3.75
X_RIGHT = 27.5
_WIDTH = X_RIGHT - X_LEFT  # 31.25
_BLOCK = 64  # nodes per block; results do not depend on it beyond rounding order


@dataclass(frozen=True)
class ICMResult:
    """Result of :func:`icm`. Player order is the input order."""

    placements: np.ndarray  # (n, m), placements[i, r-1] = P(player i finishes r-th)
    equities: np.ndarray  # (n,)
    itm: np.ndarray  # (n,), P(finish in the top m)
    chip_ev: np.ndarray  # (n,), (s_i / S) * sum(prizes)
    delta: np.ndarray  # (n,), equities - chip_ev
    grid: tuple[float, int]  # (h, M) actually used


def default_step(n: int) -> float:
    """Step h by field size (paper Appendix C)."""
    if n <= 50:
        return 1 / 8
    if n <= 200:
        return 1 / 16
    if n <= 1000:
        return 1 / 32
    if n > 4000:
        warnings.warn(
            "n > 4000 is outside the range tested in the paper; using h = 1/64",
            stacklevel=3,
        )
    return 1 / 64


def _nodes(h: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes t_l = phi(x_l) and weights w_l = h phi'(x_l) on the fixed window."""
    M = int(round(_WIDTH / h))
    if M < 2 or abs(M * h - _WIDTH) > 1e-9 * _WIDTH:
        raise ValueError(f"h={h} must divide the window width {_WIDTH} into whole steps")
    x = X_LEFT + np.arange(M) * h
    emx = np.exp(-x)
    t = np.exp(x - emx)
    w = h * (1.0 + emx) * t
    return t, w


def _validate(stacks, prizes) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(stacks, dtype=np.float64)
    pr = np.asarray(prizes, dtype=np.float64)
    if s.ndim != 1 or pr.ndim != 1:
        raise ValueError("stacks and prizes must be one-dimensional sequences")
    n, m = s.size, pr.size
    if n < 2:
        raise ValueError("need at least 2 players")
    if m == 0:
        raise ValueError("prizes must not be empty")
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("all stacks must be positive and finite")
    if not np.all(np.isfinite(pr)) or np.any(pr <= 0):
        raise ValueError("all prizes must be positive and finite")
    if np.any(np.diff(pr) > 0):
        raise ValueError("prizes must be non-increasing")
    if m > n:
        raise ValueError(f"more prizes ({m}) than players ({n})")
    # tiny relative slack so ratios built as exactly 1e9 in floating point pass
    if s.max() / s.min() > RHO_MAX * (1 + 1e-9):
        raise ValueError(f"stack ratio exceeds {RHO_MAX:g}")
    return s, pr


def _block(
    st: np.ndarray, t: np.ndarray, w: np.ndarray, m: int, qhat: np.ndarray
) -> None:
    """Add one block of nodes to qhat (n, m). st = sorted normalized stacks."""
    n = st.size
    b = t.size
    arg = st[:, None] * t[None, :]  # (n, b)
    q = np.exp(-arg)
    p = -np.expm1(-arg)  # 1 - q, accurate for small arg

    # 4.1 full product by direct convolution; C[r, node]
    C = np.zeros((n + 1, b))
    C[0] = 1.0
    for j in range(n):
        tmp = p[j] * C[: j + 1]
        C[: j + 1] *= q[j]
        C[1 : j + 2] += tmp

    fwd = q >= 0.5
    anyf = np.flatnonzero(fwd.any(axis=1))
    bwd = ~fwd
    anyb = np.flatnonzero(bwd.any(axis=1))

    # forward recurrence, players with q >= 1/2 (a prefix of the sorted order)
    if anyf.size:
        nf = int(anyf[-1]) + 1
        f = fwd[:nf]
        alpha = np.zeros((nf, b))
        np.divide(p[:nf], q[:nf], out=alpha, where=f)  # members only: q >= 1/2
        wf = f * w  # zero weight for non-members
        E = np.zeros((nf, b))
        for r in range(m):
            Cr = C[r]
            E *= -alpha
            E += Cr
            np.maximum(E, 0.0, out=E)
            np.minimum(E, Cr, out=E)
            qhat[:nf, r] += np.einsum("ib,ib->i", wf, E)

    # backward recurrence, players with q < 1/2 (a suffix); starts at place n
    if anyb.size:
        lo = int(anyb[0])
        g = bwd[lo:]
        beta = np.zeros((n - lo, b))
        np.divide(q[lo:], p[lo:], out=beta, where=g)  # members only: p > 1/2
        wb = g * w
        E = np.zeros((n - lo, b))
        for r in range(n, 0, -1):
            np.subtract(C[r], E, out=E)
            E *= beta
            np.maximum(E, 0.0, out=E)
            np.minimum(E, C[r - 1], out=E)
            if r <= m:
                qhat[lo:, r - 1] += np.einsum("ib,ib->i", wb, E)


def icm(
    stacks: Sequence[float], prizes: Sequence[float], *, h: float | None = None
) -> ICMResult:
    """ICM placement probabilities and equities by DE-ICM.

    stacks: positive chip counts (ratio of largest to smallest at most 1e9).
    prizes: positive, non-increasing prizes for places 1..m, m <= n.
    h: optional quadrature step override; 31.25 / h must be an integer.
    """
    s, pr = _validate(stacks, prizes)
    n, m = s.size, pr.size
    if h is None:
        h = default_step(n)
    t, w = _nodes(h)

    order = np.argsort(s, kind="stable")
    st = s[order] / (HEADROOM * s.max())

    qhat = np.zeros((n, m))
    for a in range(0, t.size, _BLOCK):
        _block(st, t[a : a + _BLOCK], w[a : a + _BLOCK], m, qhat)
    qhat *= st[:, None]

    Q = np.empty_like(qhat)
    Q[order] = qhat

    equities = Q @ pr
    chip_ev = s / s.sum() * pr.sum()
    return ICMResult(
        placements=Q,
        equities=equities,
        itm=Q.sum(axis=1),
        chip_ev=chip_ev,
        delta=equities - chip_ev,
        grid=(float(h), int(t.size)),
    )
