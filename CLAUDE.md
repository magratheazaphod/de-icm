# de-icm

Python reimplementation of DE-ICM (Inariba, "Near-Exact Computation of Independent Chip Model
Placement Probabilities for Thousands of Players", arXiv:2610.06028, October 2026). The paper
describes the method in full but no implementation was released; this is an independent one.

## Purpose
- Library: `deicm.icm(stacks, prizes)` returns the n x m placement matrix and equities for any
  field size up to 4,000 players, at close to double-precision accuracy.
- Study tool: the CLI prints ICM equity next to chip-EV equity (proportional share of the prize
  pool) so the deviation introduced by ICM is always visible. That comparison is a first-class
  output, not an afterthought.
- Downstream: the payoff layer for a future tournament solver.

## Conventions
- Env: uv. `uv sync`, `uv run pytest`, `uv run deicm ...`.
- numpy only at runtime. mpmath is a dev dependency used for high-precision reference values in tests.
- `docs/SPEC.md` is the authoritative description of the algorithm, grid parameters and the
  validation suite. It is distilled from the paper; when in doubt, the paper wins and the spec
  gets corrected.
- Accuracy targets come from the paper: worst absolute placement-probability error about 1e-14,
  worst relative equity error about 4e-14 on exactly solvable instances. Tests enforce 1e-12
  hard bounds and report the observed maxima.
- No em dashes in prose. MIT license. Keep `Co-Authored-By` trailers on commits.
