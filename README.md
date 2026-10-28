# de-icm

An independent attempt to reproduce **DE-ICM**, the algorithm published by Wataru Inariba of
GTO Wizard in *Near-Exact Computation of Independent Chip Model Placement Probabilities for
Thousands of Players*, [arXiv:2610.06028](https://arxiv.org/abs/2610.06028) (October 2026),
in Python and NumPy.

**The algorithm, its analysis and its validation design are entirely the work of the paper's
author and GTO Wizard. Nothing here is original research.** The paper describes the method in
full and states that neither the research implementation nor GTO Wizard's production code is
released, and that the description is intended to support independent reimplementation. This
repository is that: a from-scratch implementation written from the paper alone, with no access
to any GTO Wizard code, and a reproduction of the paper's accuracy claims against the exact
reference families it specifies. See [Attribution](#attribution) below.

## What ICM is

The Independent Chip Model turns chip stacks into finishing probabilities: the winner is drawn
with probability proportional to stack, then the runner-up is drawn the same way among the
remaining players, and so on. Weighting each place's probability by its prize gives every
player's equity, which differs from the proportional (chip EV) share of the prize pool.

## What DE-ICM does

Computing the full ICM placement matrix by enumerating finishing orders is only feasible for
about a dozen players, and the usual recursive shortcuts lose accuracy or time beyond a few
hundred. DE-ICM writes each placement probability as a one-dimensional integral over an
exponential clock for each player, evaluates it with a double-exponential trapezoidal rule, and
uses a direction-aware, clamped deconvolution to remove one player at a time from the
Poisson-binomial product. The result is deterministic, needs no sampling, and agrees with
exact references to near double precision (about 1e-14) for fields of thousands of players.
The paper's C++ implementation takes about 0.2 s for 1,000 players. This implementation is
pure Python and NumPy, so it is slower, but the algorithm and accuracy are the same. The
full algorithm and grid parameters are in [docs/SPEC.md](docs/SPEC.md).

## Installation

```
git clone https://github.com/magratheazaphod/de-icm.git
cd de-icm
uv sync
```

This installs the `deicm` package and command. The only runtime dependency is NumPy (2.0 or
later, Python 3.12 or later). mpmath is installed as a dev dependency for the tests.

## Python API

```python
from deicm import icm

# A nine-player table paying three places. Fields of hundreds or thousands work the same way.
stacks = [4200, 1800, 950, 3100, 2750, 600, 1500, 2200, 900]
prizes = [5000, 3000, 2000]

res = icm(stacks, prizes)
res.placements   # (n, m) array, res.placements[i, r-1] = P(player i finishes in place r)
res.equities     # ICM equity of each player, in prize units
res.itm          # probability of finishing in a paid place
res.chip_ev      # proportional share of the prize pool
res.delta        # equities - chip_ev
res.grid         # (h, M): step and number of quadrature nodes actually used
```

Stacks must be positive, prizes positive and non-increasing, and there must be at least as many
players as prizes. Invalid input raises `ValueError`. Input order is preserved in all outputs.

## Command line

```
deicm 1000 2500 400 3000 --prizes 50 30 20
deicm --stacks-file stacks.txt --prizes-file payouts.txt --json
```

The table has one row per player in input order, with these columns: seat (1-based), stack,
chip-EV equity, ICM equity, delta (ICM minus chip EV) in prize units, delta as a percentage of
chip EV, P(ITM) and P(1st). The footer gives the number of players, places paid, total prize
pool and the grid used. The chip-EV column is the point of the tool: it shows what the same
stacks are worth under chip EV, so the delta is the deviation that ICM introduces.

Stack and prize files hold numbers separated by whitespace or newlines; `#` starts a comment.
Prizes can be given as amounts (`--prizes`, `--prizes-file`) or as percentages of a pool.
Percentages are scaled to the pool and may sum to at most 100 plus a half point of rounding.

Example: a 9-handed table late in a 300-runner, 100-unit buy-in tournament that pays 45 places
(15 percent of the field). Put the nine stacks of the table plus the 291 other stacks in
`stacks.txt` and the 45 payouts in `payouts.txt`, then:

```
deicm --stacks-file stacks.txt --prizes-file payouts.txt
```

Seats 1 to 9 are the first nine stacks in the file. For a small payout structure given as
percentages of the pool:

```
deicm 5200 3100 2400 1800 900 --prizes-pct 50 30 20 --pool 1000 --places
```

Add `--places` to print the placement probability of every seat in every paid place, and
`--json` to get all of the above as a JSON object (stacks, prizes, pool, grid, chip EV,
equities, delta, delta percent, P(ITM), P(1st) and the placement matrix). Errors, including
invalid input rejected by the library, are printed to standard error with exit status 2.

### Perturbations and bubble factors

```
deicm 62000 41000 28000 22000 15500 11000 8000 5200 3100 --prizes-pct 30 20 14 10 8 6 5 4 3 --pool 30000 --allin 9 1
deicm 62000 41000 28000 22000 15500 11000 8000 5200 3100 --prizes-pct 30 20 14 10 8 6 5 4 3 --pool 30000 --bubble-factors --seats 1-5
```

`--allin W L` shows what happens to every seat's chip EV and ICM equity if seat W beats seat L
all-in (seats are 1-based); a loser whose stack is no larger than the winner's busts, and the
rest of the field is re-evaluated. `--bubble-factors` prints, for each pair of seats, the
bubble factor (equity lost when losing an all-in divided by equity gained when winning it) and
the required equity (the win probability needed to break even, with no dead money). Under chip
EV the bubble factor is 1 and the required equity 50 percent, so anything above that is ICM
pressure. `--seats` takes forms like `1-9`, `1,3,5` or `1-4,7`; the default is all seats up to
12 players and the first 9 otherwise. Add `--pot CHIPS` for dead money already in the middle
(for example the blinds and antes of other players): the winner of each all-in also collects it,
so required equity drops: it is the break-even win probability against folding, with the dead pot going to the aggressor on a fold. The listed stacks are taken as already net of what those players posted. Both flags may be combined with each other, `--places`
and `--json`. The library equivalents are `deicm.allin` and `deicm.bubble_factors`.

## Web app

A small mobile-first page for use between hands, on breaks and for deal talk. It is not a
decision aid during a hand. Enter up to nine stacks, mark the hero, optionally describe the
rest of the field, set the payouts and any dead money in the pot, and it shows the hero's ICM
equity against chip EV and, for each opponent, the equity you need and they need to call an
all-in.

```bash
uv sync --extra web
uv run deicm-web          # http://localhost:8080, or set $PORT
```

The library install stays numpy-only; FastAPI and uvicorn live in the optional `web` extra.
Set `AUTH_USERS=user:pass,user:pass` to require HTTP basic auth on everything except `/healthz`.

Deploy to Fly.io (stateless, stops itself when idle):

```bash
fly launch --copy-config --no-deploy
fly secrets set AUTH_USERS=user:pass ANTHROPIC_API_KEY=...   # the key enables photo import
fly deploy
```

## Accuracy

On instances with exact answers (winner-take-all, two prizes, flat, linear, equal stacks, two
stack sizes, equal stacks with exceptions, and small fields by subset enumeration), the paper
reports a worst absolute placement-probability error of about 1.1e-14 and a worst relative
equity error of about 4.0e-14 for fields up to 4,000 players (see its validation figures and
tables). The test suite in `tests/` reproduces that validation against high-precision mpmath
references, enforces hard bounds of 1e-12, and reports the observed maxima. The families and
parameters are listed in sections 6 and 7 of [docs/SPEC.md](docs/SPEC.md). Run it with
`uv run pytest`.

## Scope and limits

- The ratio of the largest to the smallest stack must be at most 1e9; larger ratios raise
  `ValueError`.
- Fields up to 4,000 players are validated, with grid settings from the paper. Larger fields
  run with the finest grid and a warning, in a regime the paper does not test.
- Small probabilities are accurate in absolute terms, not relative terms. A placement
  probability near 1e-13 may have no correct digits.
- Runtime grows roughly with the square of the field size.
- ICM is a model. It assumes finishing order is a stack-weighted draw, ignoring skill, position,
  blind levels and payout-aware play. How well it describes real tournaments is a separate
  question from how accurately this code computes it.

## Attribution

- **Method:** DE-ICM is due to Wataru Inariba (GTO Wizard). The paper credits an earlier version
  of the method as being in production inside GTO Wizard's tournament solver since 2024. The
  name DE-ICM is the paper's. This repository does not reproduce any GTO Wizard code or data and
  is not affiliated with or endorsed by GTO Wizard.
- **This implementation:** written by Jesse Day from the paper, with heavy assistance from
  Claude (Anthropic). The commit history records that assistance.
- **Related prior work cited by the paper:** the integral representation goes back to the
  analysis of the move-to-front rule (Fill and Holst; Flajolet, Gardy and Thimonier) and was
  independently used for Plackett-Luce propensities (Knyazev and Oosterhuis, MPL) and for ICM
  (Wolters, LAQI). See the paper's Section 6 for the full discussion.

If you use this code, please cite the paper, not this repository:

```bibtex
@misc{inariba2026deicm,
  title         = {Near-Exact Computation of Independent Chip Model Placement Probabilities for Thousands of Players},
  author        = {Inariba, Wataru},
  year          = {2026},
  month         = oct,
  eprint        = {2610.06028},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DS},
  url           = {https://arxiv.org/abs/2610.06028}
}
```

A `CITATION.cff` is included so that GitHub's citation widget points at the paper.

## License

MIT - see [LICENSE](LICENSE).
