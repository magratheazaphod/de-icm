# DE-ICM specification

Distilled from Inariba, arXiv:2610.06028 (Oct 2026), Sections 2 to 4 and Appendices C to E.
Equation numbers refer to the paper. Where this file and the paper disagree, the paper wins.

## 1. Problem

A field of `n` players has stacks `s_1..s_n > 0`, total `S`. Prizes `pi_1 >= pi_2 >= ... >= pi_m > 0`
are paid to the first `m <= n` finishers; `pi_r = 0` for `r > m`.

The ICM draws the winner with probability proportional to stack, then the runner-up among the
remaining players proportional to stack, and so on (Plackett-Luce with stacks as weights).

Outputs:
- Placement matrix `Q[i, r] = P(player i finishes in place r)` for `r = 1..m`. The full `n x n`
  matrix is doubly stochastic.
- Equities `V_i = sum_r pi_r Q[i, r]`.
- Derived: `P(ITM)_i = sum_{r<=m} Q[i, r]`, `P(out of money) = 1 - P(ITM)`.

Only stack ratios matter (scale invariance).

## 2. Integral representation (Theorem 2.2)

Give each player an exponential clock `T_j ~ Exp(s_j)`; players finish in increasing clock order.
Condition on `T_i = t`. Player `j != i` is ahead of `i` with probability `1 - q_j(t)` where
`q_j(t) = exp(-s_j t)`, independently. The number ahead is Poisson-binomial with generating
function

    G^{\i}(z; t) = prod_{j != i} ( q_j + (1 - q_j) z ) = sum_r C_r^{\i}(t) z^r        (2)

and

    Q[i, r+1] = integral_0^inf  s_i exp(-s_i t) C_r^{\i}(t) dt,   r = 0..n-1.          (3)

## 3. Normalization and grid (Section 3, Appendix C)

Admissible input: `s_max / s_min <= rho = 1e9`. Reject (ValueError) otherwise.

Normalize with headroom `c = 20`:

    s~_i = s_i / (c * s_max)          so  s~_i in [1/(c rho), 1/c]

Sort players so `s~_1 <= ... <= s~_n` (undo the sort on output).

Exp-exp transformation `t = phi(x) = exp(x - exp(-x))`, `phi'(x) = (1 + exp(-x)) phi(x)`.
Trapezoidal rule on the lattice `x_l = x_L + l h`, `l = 0..M-1`, window `[x_L, x_R) = [-3.75, 27.5)`:

    t_l = phi(x_l),   w_l = h * phi'(x_l),   M = (x_R - x_L) / h = 31.25 / h

Step by field size (Appendix C):

| n            | h     | M     |
|--------------|-------|-------|
| 2 .. 50      | 1/8   | 250   |
| 51 .. 200    | 1/16  | 500   |
| 201 .. 1000  | 1/32  | 1000  |
| 1001 .. 4000 | 1/64  | 2000  |

Nodes and weights depend only on `h` and the window, never on the input. For `n > 4000` the
paper gives no setting; use `h = 1/64` and emit a warning (untested regime).

The first place has the closed form `Q[i,1] = s_i / S` but the paper integrates it like the
others (keep it that way; it costs ~3 percent of nodes and simplifies validation).

## 4. Per-node work (Section 4, Algorithm 1)

At node `t` (the subscript `l` is dropped), with sorted normalized stacks:

    q_j = exp(-s~_j t)            non-increasing in j
    p_j = -expm1(-s~_j t)         = 1 - q_j, computed this way for accuracy

### 4.1 Full product

`G(z) = prod_{j=1..n} (q_j + p_j z) = sum_{r=0..n} C_r z^r`, by `n` successive convolutions with
the two-term vector `[q_j, p_j]`:

    C <- [1]
    for j in 1..n:  C_new[r] = q_j * C[r] + p_j * C[r-1]       (C[-1] = C[len] = 0)

Direct convolution is the most accurate standard method here (no FFT). Underflow to zero is fine.
The coefficients are probabilities and sum to 1.

### 4.2 Stable leave-one-out deconvolution

Define `E_r^{\i} = q_i C_r^{\i}` (this is what (3) needs, since `s_i exp(-s_i t) = s_i q_i`
in normalized time). With `alpha_i = p_i / q_i`:

    C_r = E_r + alpha_i * E_{r-1}

Two recurrences, and the direction rule is the whole stability result (Theorem 4.1):

- Forward, used when `q_i >= 1/2` (so `alpha_i <= 1`):

      E_{-1} = 0
      for r = 0 .. m-1:   E_r = clamp( C_r - alpha_i * E_{r-1},  0, C_r )
                          Q^[i, r+1] += w * E_r

  The forward recurrence may stop at place `m`.

- Backward, used when `q_i < 1/2` (so `1/alpha_i < 1`):

      E_n = 0
      for r = n .. 1:     E_{r-1} = clamp( (C_r - E_r) / alpha_i,  0, C_{r-1} )
                          if r <= m:  Q^[i, r] += w * E_{r-1}

  The backward recurrence must start from place `n`.

The clamp to `[0, C_r]` enforces the exact invariant `0 <= E_r <= C_r` (Lemma 4.2) and removes a
rounding residue that would otherwise produce a relative error up to `n rho u h` in the first-place
probability (Section 4.2, "Clamping"). Both the direction rule and the clamp are required; the
paper shows neither single direction is usable on its own (Table 4).

Because stacks are sorted ascending and `q_j` is non-increasing in `j`, the forward group at a node
is a prefix `1..b` and the backward group the suffix `b+1..n`, with `b = #{j : q_j >= 1/2}`
non-increasing in `t`.

### 4.3 Assembly

    Q^ <- zeros(n, m)
    for each node l:
        compute q, p, C_0..n, then the two recurrences, accumulating w_l * E into Q^
    Q[i, r] = s~_i * Q^[i, r]
    undo the sort
    V_i = sum_r pi_r Q[i, r]

Cost `O(M n^2)` time and `O(n m)` memory.

### 4.4 Vectorization notes for NumPy

The reference implementation loops over nodes and vectorizes over players. In NumPy, vectorizing
over nodes as well is usually faster:

- Full product: keep `C` as an `(M, n+1)` array; the loop over `j` does one fused update on
  `C[:, :j+2]`. Total `~ M n^2 / 2` flops.
- Recurrences: keep `E` as `(M, n)` for all players at once and loop over `r`. Use masks or
  per-group `alpha` arrays with zeros outside the group so that the wrong-direction recurrence
  is never evaluated for a player (it would overflow). Accumulate `Q^[:, r]` as a weighted sum
  over nodes per step. Total `~ M n m` flops forward plus `~ M n^2` backward.
- Memory for `(M, n)` float64 at `n = M = 1000` is 8 MB; at `n = 4000, M = 2000` it is 64 MB.
- Process nodes in blocks if memory becomes a concern; results must not depend on block size.
- Use float64 throughout. Deterministic reduction order is preferred (no threading).

Target: `n = 1000`, full matrix, within a few seconds on one core. The paper's C++ does it in
0.2 s; being 10x slower in NumPy is acceptable, 100x slower is not.

## 5. Accuracy (Section 5)

Paper, on exactly solvable instances up to `n = 4000`: worst absolute placement-probability error
`1.1e-14`, worst relative equity error `4.0e-14`. Rounding floor is roughly `(n + M) u` worst case
and `sqrt(n + M) u` typically, about `1e-14` for `n ~ M ~ 1e3`.

Small probabilities are accurate in absolute terms, not relative terms (a known limitation).

## 6. Exact reference families (Appendix D, Table 9)

Used to validate. All references must be computed in high precision (mpmath, 40+ digits), since in
plain double their own rounding error (~1e-12 at large n) exceeds what is being measured.

| Family               | Stacks            | Prizes                   | Reference |
|----------------------|-------------------|--------------------------|-----------|
| winner-take-all      | arbitrary         | m = 1                    | `Q[i,1] = s_i / S` |
| two prizes           | arbitrary         | m = 2                    | `Q[i,2] = sum_{j!=i} (s_j/S) * s_i/(S - s_j)`, O(n^2) |
| flat (satellite)     | arbitrary         | `pi_r = pi`, m = n       | `V_i = pi` |
| linear               | arbitrary         | `pi_r = n - r + 1`       | `V_i = n + 1/2 - sum_j s_j/(s_i + s_j)` (sum over all j incl. j = i, which contributes 1/2) |
| equal stacks         | `s_i = s`         | arbitrary                | `V_i = (1/n) sum_r pi_r`; also every `Q[i,r] = 1/n` |
| two stack sizes      | g = 2 values      | arbitrary                | DP on count of placed type-1 players, O(m * min(n1, n2)) |
| equal + exceptions   | e <= 10 exceptional, rest equal | arbitrary  | DP on the set of placed exceptional players plus count of placed regulars, O(m e 2^e) |
| small fields         | n <= 20 (use <= 14 in Python) | arbitrary    | DP on the set of placed players, O(n 2^n) |

DP rule (next-finisher): from a state after `r` placements, the next place goes to remaining
player `j` with probability `s_j / sum_{j' unplaced} s_j'`. For grouped states, a specific player of
group `h` is still unplaced with probability `(n_h - x_h) / n_h` by exchangeability, and is then
placed next with probability `s(h) / sum_h' s(h') (n_h' - x_h')`. `Q[i, r+1]` is the sum over
layer-`r` states of `P(state) * (those two factors)`.

## 7. Validation suite (Appendix E)

Field sizes `n in {3, 10, 30, 100, 200, 500, 1000, 2000, 4000}`. For routine tests use up to 1000
and mark 2000 and 4000 as slow.

Stacks:
- equal; linear `s_i = i`; geometric from 1 to `1e2, 1e4, 1e6, 1e9`;
- Zipf `s_i ~ i^-1.5` and log-normal `sigma = 2`, both rescaled in the log to ratio 1e9 if they exceed it;
- one stack 1e9 smaller, or larger, than the other n-1;
- two stack sizes: one player, half the field, or all but one at stack 1, rest at 3, 1e3 or 1e9;
- ten exceptional stacks among regulars of 1: spread `rho^(-1/2 + k/9)`, k = 0..9; all below
  `rho^(-1/2 + k/20)`; all above `rho^(1/20 + k/20)`.

Prizes:
- winner-take-all; two prizes with `pi_2/pi_1 = 0.5` and `1e-6`; flat over all n;
- linear `pi_r = n - r + 1`; realistic `pi_r ~ r^-0.8` and top-heavy `pi_r = 0.7^(r-1)`, both
  for the top `ceil(0.15 n)` places;
- satellites paying the top half, top 80 percent, or all but last, equally.

Pair each stack generator with the prize structures that have an exact reference (Table 9), and
the small-field family with all of them.

Checks: absolute error of every placement probability; relative error of every equity; row sums
(a player's places, for m = n) and column sums (each computed place) equal to 1.

Random draws: paper uses MT19937 seeded `12345 + n`; reproduce in distribution only.

## 8. Public API (this implementation)

```python
from deicm import icm, ICMResult

res = icm(stacks, prizes)          # sequences of positive numbers; prizes non-increasing
res.placements                      # ndarray (n, m), Q[i, r-1] = P(player i finishes r-th)
res.equities                        # ndarray (n,)
res.itm                             # ndarray (n,), P(finish in top m)
res.chip_ev                         # ndarray (n,), (s_i / S) * sum(prizes)
res.delta                           # equities - chip_ev
res.grid                            # (h, M) actually used
```

Options: `h=None` to override the step (for sweeps), `places=None` to request fewer places than
`len(prizes)` is not needed; `m = len(prizes)`. Prizes may be given shorter than n; pad with zeros
conceptually, never allocate n columns unless m = n.

Validation errors (ValueError): any stack <= 0, fewer than 2 players, prizes empty or increasing
anywhere or any prize <= 0, m > n, stack ratio > 1e9.

Also expose `deicm.exact` with the reference families (used by tests, and useful on its own for
small tables):

```python
from deicm.exact import icm_exact        # subset DP, n <= 14, returns (Q, V) as mpmath or float
```

## 9. CLI

```
deicm 1000 2500 400 3000 --prizes 50 30 20
deicm --stacks-file stacks.txt --prizes-file payouts.txt --json
```

Prints one row per player in input order: stack, chip-EV equity, ICM equity, delta (ICM minus
chip EV) in prize units and as a percent of chip EV, P(ITM), P(1st). Then a footer with the total
prize pool and the grid used. `--places` prints the placement matrix for the paid places.
`--json` emits the same as JSON. Prize structures may be given as a list of amounts, or as
`--prizes-pct 50 30 20 --pool 10000`.

The chip-EV column is the point of the tool: it is what the same stacks are worth under chip EV,
so the delta is the deviation ICM introduces.
