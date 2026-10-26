"""Stack and prize families (spec section 7) and their exact references."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from mpmath import mp, mpf

from deicm import exact

RHO = 1e9


@dataclass
class Stacks:
    name: str
    canon: np.ndarray  # stacks in canonical order (reference order)
    perm: np.ndarray  # stacks = canon[perm]; the input order seen by icm
    kind: str = "general"  # general | equal | two | exc
    info: dict = field(default_factory=dict)

    @property
    def values(self) -> np.ndarray:
        return self.canon[self.perm]

    @property
    def n(self) -> int:
        return self.canon.size


def _rescale_log(s: np.ndarray, ratio: float) -> np.ndarray:
    r = s.max() / s.min()
    if r <= ratio:
        return s
    lo = math.log(s.min())
    return np.exp(lo + (np.log(s) - lo) * (math.log(ratio) / math.log(r)))


def stack_families(n: int) -> dict[str, Stacks]:
    rng = np.random.default_rng(12345 + n)
    perm = rng.permutation(n)
    out: dict[str, Stacks] = {}

    def add(name, canon, kind="general", **info):
        out[name] = Stacks(name, np.asarray(canon, dtype=np.float64), perm, kind, info)

    add("equal", np.full(n, 7.0), "equal")
    add("linear", np.arange(1, n + 1))
    for e in (2, 4, 6, 9):
        add(f"geom1e{e}", np.geomspace(1.0, 10.0**e, n))
    add("zipf", _rescale_log(np.arange(1, n + 1, dtype=float) ** -1.5, RHO))
    add("lognormal2", _rescale_log(rng.lognormal(0.0, 2.0, n), RHO))
    add("one_small", np.r_[1.0, np.full(n - 1, RHO)])
    add("one_large", np.r_[RHO, np.full(n - 1, 1.0)])
    for label, n1 in (("one", 1), ("half", n // 2), ("allbutone", n - 1)):
        for vl, v in (("3", 3.0), ("1e3", 1e3), ("1e9", 1e9)):
            c = np.r_[np.full(n1, 1.0), np.full(n - n1, v)]
            add(f"two_{label}_{vl}", c, "two", n1=n1, s1=1.0, n2=n - n1, s2=v)
    if n >= 12:
        reg = np.ones(n - 10)
        k = np.arange(10)
        for label, expo in (
            ("spread", -0.5 + k / 9),
            ("below", -0.5 + k / 20),
            ("above", 1 / 20 + k / 20),
        ):
            exc = RHO**expo
            add(f"exc_{label}", np.r_[exc, reg], "exc", exc=exc, n_reg=n - 10)
    return out


PRIZES = (
    "wta", "two_half", "two_1e6", "flat", "linear",
    "realistic", "topheavy", "sat_half", "sat_80", "sat_allbutlast",
)  # fmt: skip


def prize_family(name: str, n: int) -> np.ndarray:
    m15 = max(1, math.ceil(0.15 * n))
    r = np.arange(1, n + 1, dtype=float)
    match name:
        case "wta":
            return np.array([1.0])
        case "two_half":
            return np.array([1.0, 0.5])
        case "two_1e6":
            return np.array([1.0, 1e-6])
        case "flat":
            return np.full(n, 1.0)
        case "linear":
            return n - r + 1
        case "realistic":
            return r[:m15] ** -0.8
        case "topheavy":
            return 0.7 ** (r[:m15] - 1)
        case "sat_half":
            return np.ones(max(1, n // 2))
        case "sat_80":
            return np.ones(max(1, int(0.8 * n)))
        case "sat_allbutlast":
            return np.ones(n - 1)
    raise KeyError(name)


CLOSED = {"wta", "two_half", "two_1e6", "flat", "linear"}


def _canon_to_input(a: np.ndarray, st: Stacks) -> np.ndarray:
    return a[st.perm]


def _vec_dot(q_rows, pr) -> list:
    return [mp.fsum(q * mpf(float(p)) for q, p in zip(row, pr)) for row in q_rows]


@lru_cache(maxsize=None)
def _dp_groups(fam_key: tuple, n: int, m: int):
    """Cached DP output for the grouped families: (rows) of mp placement vectors
    per group, computed once at m places."""
    kind, name = fam_key
    st = stack_families(n)[name]
    if kind == "two":
        i = st.info
        q1, q2 = exact.two_size_dp(i["n1"], i["s1"], i["n2"], i["s2"], m)
        return [list(q1), list(q2)]
    rows = exact.equal_exceptions_dp(st.info["exc"], 1.0, st.info["n_reg"], m)
    return [list(r) for r in rows]


def dp_max_m(st: Stacks) -> int:
    """Largest m we are willing to ask the grouped DPs for (keeps tests fast)."""
    n = st.n
    if st.kind == "equal":
        return n
    if st.kind == "two":
        return n if n <= 500 else math.ceil(0.15 * n)
    if st.kind == "exc":
        return n - 1 if n <= 100 else math.ceil(0.15 * n)
    return 0


@lru_cache(maxsize=None)
def _subset(name: str, n: int, m: int):
    st = stack_families(n)[name]
    Q, V = exact.icm_exact(st.canon, np.ones(m))  # Q only depends on m
    return Q


def reference(st: Stacks, prizes: np.ndarray, prize_name: str):
    """Return (Qref | None, Vref) as float64 arrays in input order, or None if
    no exact reference exists. Qref has shape (n, m)."""
    n, m = st.n, prizes.size
    s = st.canon

    if n <= 10:  # subset DP for everything
        Qm = _subset(st.name, n, m)
        V = exact.equities(Qm, prizes)
        return _canon_to_input(exact.as_float(Qm), st), _canon_to_input(exact.as_float(V), st)

    if st.kind == "equal":
        V = float(mp.fsum(mpf(float(x)) for x in prizes) / n)
        return np.full((n, m), float(mpf(1) / n)), np.full(n, V)

    if prize_name == "wta":
        Qm = exact.winner_take_all(s)
        V = exact.equities(Qm, prizes)
    elif prize_name.startswith("two"):
        Qm = exact.two_prizes(s)
        V = exact.equities(Qm, prizes)
    elif prize_name == "flat":
        return None, _canon_to_input(exact.as_float(exact.flat_equities(n, prizes[0])), st)
    elif prize_name == "linear":
        return None, _canon_to_input(exact.as_float(exact.linear_equities(s)), st)
    elif st.kind in ("two", "exc") and m <= dp_max_m(st):
        rows = _dp_groups((st.kind, st.name), n, dp_max_m(st))
        V_g = [mp.fsum(q * mpf(float(p)) for q, p in zip(r[:m], prizes)) for r in rows]
        if st.kind == "two":
            group = np.r_[np.zeros(st.info["n1"], int), np.ones(st.info["n2"], int)]
        else:
            group = np.r_[np.arange(10), np.full(st.info["n_reg"], 10)]
        Qg = np.array([[float(x) for x in r[:m]] for r in rows])
        Vg = np.array([float(v) for v in V_g])
        return Qg[group][st.perm], Vg[group][st.perm]
    else:
        return None
    return _canon_to_input(exact.as_float(Qm), st), _canon_to_input(exact.as_float(V), st)
