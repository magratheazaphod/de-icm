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

## Workflow: worktrees and PRs
- Every change lands on `main` through a PR; never commit or push to `main` directly. GitHub
  branch protection requires the CI check, and `.githooks/pre-push` refuses direct pushes
  locally (a fresh clone needs `git config core.hooksPath .githooks` once).
- Work in a git worktree, one per branch, beside the main checkout:
  `git worktree add ../de-icm.wt/<branch> -b <branch> origin/main`. Leave the main checkout on
  `main` and clean. After the PR merges: `git worktree remove ../de-icm.wt/<branch>` and
  `git branch -d <branch>`.
- CI (`.github/workflows/ci.yml`) runs `uv run --extra web pytest`. Deploys stay manual
  (`scripts/deploy.sh`), from an up-to-date `main` after the PR merges.
