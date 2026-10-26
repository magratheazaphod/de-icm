"""Accuracy against exact references (spec section 7)."""

from __future__ import annotations

import numpy as np
import pytest

from deicm import icm

from .conftest import record
from .helpers import PRIZES, prize_family, reference, stack_families

TOL = 1e-12


def run_case(n: int, fam: str, prize_name: str, group: str | None = None) -> bool:
    st = stack_families(n)[fam]
    pr = prize_family(prize_name, n)
    ref = reference(st, pr, prize_name)
    if ref is None:
        return False
    Qref, Vref = ref
    res = icm(st.values, pr)
    Q = res.placements
    m = pr.size

    abs_err = 0.0
    if Qref is not None:
        assert Q.shape == Qref.shape
        abs_err = float(np.abs(Q - Qref).max())
    rel_err = float((np.abs(res.equities - Vref) / Vref).max())

    col = float(np.abs(Q.sum(axis=0) - 1).max())
    row = float(np.abs(Q.sum(axis=1) - 1).max()) if m == n else 0.0
    sums = max(col, row)

    label = group or ("small n<=10" if n <= 10 else f"n={n}")
    record(label, abs_err, rel_err, sums)
    where = f"n={n} {fam} {prize_name}"
    assert abs_err <= TOL, f"abs placement error {abs_err:.2e} ({where})"
    assert rel_err <= TOL, f"rel equity error {rel_err:.2e} ({where})"
    assert col <= TOL, f"column sums {col:.2e} ({where})"
    assert row <= TOL, f"row sums {row:.2e} ({where})"
    assert res.grid[1] * res.grid[0] == pytest.approx(31.25)
    return True


def _cases(ns):
    for n in ns:
        for fam in stack_families(n):
            for pz in PRIZES:
                yield pytest.param(n, fam, pz, id=f"n{n}-{fam}-{pz}")


@pytest.mark.parametrize("n,fam,pz", list(_cases((3, 10, 30, 100))))
def test_small_and_medium(n, fam, pz):
    if not run_case(n, fam, pz):
        pytest.skip("no exact reference for this pair")


# Larger fields: every stack family, but only the prize structures that keep the
# test suite quick. Closed-form pairs use m = n only for the linear prize.
def _big_cases(n, prizes, fams=None):
    avail = stack_families(n)
    assert fams is None or set(fams) <= set(avail), set(fams) - set(avail)
    for fam in avail:
        if fams is not None and fam not in fams:
            continue
        for pz in prizes:
            yield pytest.param(n, fam, pz, id=f"n{n}-{fam}-{pz}")


@pytest.mark.parametrize(
    "n,fam,pz",
    list(_big_cases(200, ("wta", "two_half", "two_1e6", "linear", "realistic", "topheavy"))),
)
def test_n200(n, fam, pz):
    if not run_case(n, fam, pz):
        pytest.skip("no exact reference for this pair")


FAMS_500 = ("equal", "geom1e9", "lognormal2", "one_small", "two_half_1e9", "exc_spread")
FAMS_1000 = ("equal", "geom1e9", "lognormal2", "one_large", "two_allbutone_1e3", "exc_spread")


@pytest.mark.parametrize(
    "n,fam,pz", list(_big_cases(500, ("two_1e6", "linear", "realistic"), FAMS_500))
)
def test_n500(n, fam, pz):
    if not run_case(n, fam, pz):
        pytest.skip("no exact reference for this pair")


@pytest.mark.parametrize(
    "n,fam,pz", list(_big_cases(1000, ("linear", "topheavy"), FAMS_1000))
)
def test_n1000(n, fam, pz):
    if not run_case(n, fam, pz):
        pytest.skip("no exact reference for this pair")


@pytest.mark.slow
@pytest.mark.parametrize(
    "n,fam,pz",
    list(
        _big_cases(
            2000, ("linear", "realistic"), ("equal", "geom1e9", "lognormal2", "two_half_1e9", "exc_spread")
        )
    ),
)
def test_n2000(n, fam, pz):
    if not run_case(n, fam, pz):
        pytest.skip("no exact reference for this pair")
