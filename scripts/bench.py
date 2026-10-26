"""Time deicm.icm on log-normal (sigma = 1, ratio capped at 1e6) stacks.

    uv run python scripts/bench.py [--large] [--repeat K]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from deicm import icm
from deicm.core import default_step


def make_stacks(n: int) -> np.ndarray:
    rng = np.random.default_rng(12345 + n)
    s = rng.lognormal(0.0, 1.0, n)
    return np.minimum(s, s.min() * 1e6)  # cap the ratio at 1e6


def time_icm(stacks: np.ndarray, m: int, repeat: int) -> float:
    prizes = np.arange(m, 0, -1.0)
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        icm(stacks, prizes)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true", help="also time n = 2000 and 4000")
    ap.add_argument("--repeat", type=int, default=1, help="repetitions; best time is reported")
    args = ap.parse_args()

    sizes = [100, 1000] + ([2000, 4000] if args.large else [])
    print(f"{'n':>6} {'h':>7} {'M':>6} {'m':>6} {'seconds':>10}   (full matrix: m = n)")
    for n in sizes:
        s = make_stacks(n)
        for m in (n, math.ceil(0.15 * n)):
            t = time_icm(s, m, args.repeat)
            h = default_step(n)
            print(f"{n:>6} {'1/%d' % round(1 / h):>7} {round(31.25 / h):>6} {m:>6} {t:>10.3f}")


if __name__ == "__main__":
    main()
