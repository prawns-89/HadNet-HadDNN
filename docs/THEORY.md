# Theory

Formal statements about the Hadamard decomposition model, separated from the
engineering record in `DECISIONS.md` and the direction list in `RESEARCH.md`.

Every claim below is tagged:

- **[proved]** — argued, and where possible checked numerically
- **[measured]** — an experimental fact, with the conditions it was measured under
- **[theirs]** — from Panchal, Sahoo & Behera, Neurocomputing 699 (2026) 134372
- **[open]** — not established

---

## 0. Notation

Fix `X in R^{m x n}` (tensors apply slice-wise; the paper's model is slice-wise).

| symbol | meaning | storage |
|---|---|---|
| `L_k = {A : rank(A) <= k}` | additive (truncated-SVD) class | `k(m+n)` |
| `H_r = {(W1 H1) o (W2 H2)}`, factors rank `r` | Hadamard class | `2r(m+n)` |

So **`H_r` is storage-matched to `L_2r`**, not to `L_r`. The paper compares at
matched *rank*, which gives `H_r` twice the storage of its stated baseline.

Two quantities the literature does not distinguish:

- `D(V, X) = inf_{Y in V} ||X - Y||_F / ||X||_F` — the **model limit**
- `D_hat(V, X; A)` — what **solver `A` returns**. Always `D_hat >= D`.

**Every comparison table in this literature reports `D_hat` and reads it as
`D`.** Sections 2 and 3 are consequences of that conflation.

---

## 1. Dimension of the Hadamard class

### Theorem 1 [proved]

```
dim H_r = (2r - 1)(m + n) - 2r^2 + 1
```

The parameterisation has `2r(m+n)` coordinates; the deficit `2r^2 + (m+n) - 1`
is the gauge group:

- `(W_i, H_i) -> (W_i G, G^-1 H_i)` for `G in GL(r)`, on each pair: `2r^2`
- a **shared row rescaling** `M1 -> D M1`, `M2 -> D^-1 M2` for diagonal `D`,
  since `(D M1) o (D^-1 M2) = M1 o M2` entrywise: `m`
- the same on columns: `n`
- the global scalar lies in all of the above, counted once too often: `-1`

**Verification.** Rank of the Jacobian of `(W1,H1,W2,H2) -> (W1H1) o (W2H2)` at
a generic point, float64, `rtol=1e-10`. Exact agreement on 8 cases, including
`m != n`:

| m | n | r | params | dim (Jacobian rank) | `2r^2+(m+n)-1` |
|---|---|---|---|---|---|
| 12 | 12 | 2 | 96 | 65 | 31 |
| 12 | 12 | 3 | 144 | 103 | 41 |
| 10 | 14 | 2 | 96 | 65 | 31 |
| 14 | 10 | 3 | 144 | 103 | 41 |
| 16 | 12 | 2 | 112 | 77 | 35 |
| 11 | 13 | 3 | 144 | 103 | 41 |
| 18 | 10 | 2 | 112 | 77 | 35 |
| 9 | 15 | 2 | 96 | 65 | 31 |

The shared diagonal rescalings are the term that is easy to miss; an earlier
version of this note omitted them and got `2r(m+n) - 2r^2`, which is wrong by
`(m+n) - 1` and gives the wrong sign in Corollary 1.1.

### Corollary 1.1 — the threshold rank [proved]

`dim L_2r = 2r(m+n) - 4r^2`, so

```
dim H_r - dim L_2r = 2r^2 - (m + n) + 1
```

which is **negative** for

```
r < r* = sqrt( (m + n - 1) / 2 )
```

Below `r*`, the Hadamard model at matched storage spans a **strictly
lower-dimensional** set than truncated SVD. It stores the same number of values
and reaches less.

| shape | `r*` | r=10 | r=25 | r=40 |
|---|---|---|---|---|
| 256x256 | 16.0 | **-3.2%** | +3.2% | +7.8% |
| 144x144 (Carphone slice) | 12.0 | **-1.6%** | +8.1% | — |

**The paper's Table 9 runs rank 10 on 144x144 slices, which is below `r*`.** So
`RESULTS.md` finding 1 — matched-storage SVD beating HadNet on images and video
— is predicted at the paper's own operating point by dimension counting alone,
with no reference to the spectrum.

Above `r*` the advantage exists but is a few percent, not the `r^2` factor the
rank bound suggests.

---

## 2. The published bound is vacuous where the model is useful

### Proposition 2.1 [theirs]

Prop 3.3: `rank((W1H1) o (W2H2)) <= r^2`, hence `H_r subset L_{r^2}`.
Prop 3.5: `D(H_r, X) >= D(L_{r^2}, X)`.

This is a **lower** bound on error — a necessary condition. There is no
achievability result anywhere in this literature.

### Proposition 2.2 — vacuity [proved]

`L_{r^2}` is the whole space once `r^2 >= min(m,n)`, i.e. once

```
r >= sqrt(min(m, n))
```

For 256x256 that is `r >= 16`; the paper's headline ranks are 25 and 40. At
those operating points Prop 3.5 reduces to `D >= 0` and carries no information.

### Remark 2.3 [proved]

For `m = n`, the vacuity threshold `sqrt(m)` and the dimension threshold
`r* = sqrt((2m-1)/2) ~ sqrt(m)` **coincide**. The only bound this literature
has is therefore informative exactly in the regime where Corollary 1.1 says the
model is dominated, and vacuous exactly in the regime where it has an edge.

---

## 3. Symmetry collapse of the reference initialisation

### Theorem 3.1 [proved]

Let `L(W1,H1,W2,H2) = ||X - (W1H1) o (W2H2)||`. `L` is invariant under the
involution

```
sigma : (W1, H1, W2, H2) -> (W2, H2, W1, H1)
```

Let `Delta = {W1 = W2, H1 = H2}` be its fixed set. At any point of `Delta` the
gradients satisfy `dL/dW1 = dL/dW2` and `dL/dH1 = dL/dH2`, so **`Delta` is
invariant under any `sigma`-equivariant first-order method** — gradient descent,
heavy ball, and Adam (equal parameters receive equal gradients, hence equal
first and second moment estimates, hence equal updates).

The reference `svd_init` builds factor 1 from `sqrt|X|` and factor 2 from
`sign(X) * sqrt|X|`. For `X >= 0` these are **identical**, so the init lies in
`Delta`.

**Consequence.** On non-negative data the iterates never leave `Delta`, and the
attainable class is

```
M_r = { M o M : rank(M) <= r }   strictly contained in   H_r
```

### Corollary 3.2 [proved]

- `M o M >= 0` entrywise, so the collapsed model **cannot represent negative
  entries at all**. `DECISIONS.md` Sec. 16 records that the benchmark's
  `0.01*randn` injection puts **17.79% of the MRI target below zero** — those
  entries are unfittable by construction, not by optimiser failure.
- `M_r` is parameterised by one factor pair, so it has the storage of `L_r`,
  not `L_2r`.
- `M_r` is "rank-`r` fit to `sqrt(X)`, squared" — a known construction, not a
  new model class.

### Measurement 3.3 [measured]

`128cat.png` grayscale, 256x256, all entries >= 0, `r = 10`, Adam `lr=0.1`,
2000 epochs, float64.

| quantity | value |
|---|---|
| `||M1 - M2|| / ||M1||` at init | **0.000e+00** |
| same, at epochs 10 / 100 / 500 / 1999 | **0.000e+00** throughout |
| HadNet rel. error, reference init | **0.16172** |
| SVD rank 10 (*half* the storage) | 0.16140 |
| analytic collapsed model `M o M`, rank 10 | 0.16798 |
| HadNet rel. error, + symmetry-breaking noise `1e-2` | **0.10394** |
| SVD rank 20 (matched storage) | 0.11047 |

Two readings, both exact:

1. With the reference init, HadNet at r=10 (0.16172) is **no better than a
   plain rank-10 SVD (0.16140) at half the storage.**
2. Breaking the symmetry cuts error by **35.7%**, to 0.10394, which **beats
   matched-storage SVD(20) at 0.11047 by 5.9%.**

### Measurement 3.4 -- the centring control [measured]

Same image, same rank, same solver; the only change is an additive shift, which
makes `sign(X)` non-trivial and so breaks the symmetry at init for free:

| target / init | asym @init | asym @end | HadNet | SVD(2r) | |
|---|---|---|---|---|---|
| raw `X >= 0`, reference init | **0.000e+00** | **0.000e+00** | 0.16172 | 0.11047 | loses |
| raw `X >= 0`, + break `1e-2` | 6.76e-03 | 7.40e-01 | 0.10394 | 0.11047 | **wins** |
| mean-centred (mixed sign), ref init | **1.338** | 1.470 | 0.27046 | 0.28641 | **wins** |

An additive shift does not change the spectrum in any way that bears on
"approximately low-rank", yet the verdict against matched-storage SVD reverses.
This rules out the spectral explanation for the image result and isolates
Theorem 3.1 as the mechanism.

Absolute errors are larger in the centred row because removing the DC component
shrinks `||X||`; only the within-row comparison against that row's own `SVD(2r)`
is meaningful.

**This is the control to run before acquiring any new dataset** -- it is causal,
costs nothing, and no additional data substitutes for it.

So the image half of `RESULTS.md` finding 1 is an **initialisation artifact**,
not a property of the model class — at r=10, where Corollary 1.1 says the class
is also dominated by 3.2% in dimension. Both effects point the same way; this
measurement does not separate them.

### Open [open]

The benchmark scripts inject `0.01*randn` *before* fitting, so for the noisy
targets `sign(X) != 1` and the init is **not exactly** in `Delta`. Whether that
perturbation escapes `Delta` within the 2000-epoch budget decides whether the
published tables are affected or merely fragile. Measurement 3.3 used clean
data. Untested.

### Remark 3.4 [proved]

The paper's Prop 3.4 gives uniqueness under *non-negative* `H1, H2` with
orthogonal `W` and `l1`-normalised columns. Theorem 3.1 says non-negativity is
precisely the regime where the reference init degenerates. The identifiability
theorem and the algorithm are in tension on the data the paper targets.

---

## 4. The solver gap

### Measurement 4.1 [measured]

Planted instances `X in H_{r0}`, so `D(H_{r0}, X) = 0` **by construction**.
64x64, float64, `r0 = 4`, 4000-6000 iterations, cosine-annealed.

Solvers, at `r = r0` and `r = 2 r0` (gaussian planted family):

| solver | r=4 | r=8 |
|---|---|---|
| Adam, random init (best of 4) | 3.69e-1 | 1.95e-1 |
| Adam, repo sqrt-SVD init | 3.65e-1 | 1.31e-1 |
| Stiefel heavy ball, svd init | 3.69e-1 | 1.41e-1 |
| Stiefel heavy ball, random init (best of 3) | 3.38e-1 | 1.66e-1 |

Instance families, `r0 = 4`:

| family | kurtosis | max/rms | cond | start **at** truth | svd init | random best-4 |
|---|---|---|---|---|---|---|
| gaussian | 14.6 | 8.2 | 2.2 | **0.0e+00** | 3.65e-1 | 7.92e-1 |
| nonneg | 5.8 | 4.2 | 107.4 | **0.0e+00** | 1.09e-1 | 1.85e-1 |
| orth | 13.6 | 7.8 | 1.7 | 1.3e-05 | 3.68e-1 | 8.05e-1 |

So `D_hat - D >= 0.10` on instances where `D = 0`, for every solver tried,
while the planted point itself is a **stable fixed point** (`start at truth`
column). The failure is in reaching the solution, not in holding it.

### Measurement 4.2 — loss geometry is not the cause [measured]

`||R||` against `||R||^2`, best over `lr in {0.05, 0.01}`, same planted
families. Ratios were `1.00` to four significant figures in all eight
SVD-init and gaussian cells; the four deviating cells (0.79-1.03) were all
random-init, where the measured spread across draws on one instance is already
0.369 vs 0.792. **Squared loss does not help**, so Shard's MSE choice does not
transfer.

### Measurement 4.3 — what does help [measured]

Overparameterisation, consistently. `nonneg` family, svd init:
`D_hat(H_4) = 1.09e-1 -> D_hat(H_8) = 1.26e-2`, **8.7x**, with `D = 0` for
both. This is the only intervention measured to move `D_hat` by an order of
magnitude.

### Measurement 4.4 — the predictive variable [measured]

`D_hat` tracks **entrywise kurtosis** (5.8 -> 0.109; 13.6 and 14.6 -> ~0.37)
and **anti-tracks spectral conditioning** (cond 107.4 fits best, cond 1.7 and
2.2 worst). Three families only; the direction is clear, the functional form is
not established.

---

## 5. Where the room actually is

`D(L_2r)` is what matched-storage SVD achieves; `D(L_{r^2})` is the Prop 3.5
floor on `H_r`. Measured, grayscale, float64:

| dataset | r | `D(L_2r)` matched | `D(L_{r^2})` floor | room |
|---|---|---|---|---|
| 128cat.png | 10 | 0.11047 | 0.03860 | 0.07187 |
| 128cat.png | 25 | 0.06999 | 0.00000 | 0.06999 |
| 128mri.png | 10 | 0.17549 | 0.01869 | 0.15679 |
| 128mri.png | 25 | 0.06435 | 0.00000 | 0.06435 |
| lesmis | 25 | 0.08493 | 0.00000 | 0.08493 |
| spectrometer | 25 | 0.01166 | 0.00000 | 0.01166 |
| random uniform 144x144 | 25 | 0.24675 | 0.00000 | 0.24675 |

At `r=25` on 128mri, the reported `D_hat(H_25) = 0.12036` sits **above**
matched-storage SVD (0.06435) while its own floor is 0. The shortfall cannot be
attributed to a limit of `H_25`; it is consistent with Measurement 4.1. Note the
floor is the loose Prop 3.5 bound, so "could reach 0" does **not** follow —
Corollary 1.1 is the tight statement, and Theorem 3.1 the mechanism.

Also: at `r = 40` on 144x144, `2r(m+n) = 23,040` parameters against `20,736`
entries. The "exact, 0.000000" row in `RESULTS.md` finding 1 is interpolation by
an overparameterised model, not evidence of expressiveness.

---

## 6. Open

- **[open] Achievability.** An upper bound on `D(H_r, X)`. Nothing in this
  literature has one. Measurement 4.4 says it cannot be purely spectral.
- **[open] Is `Delta` a saddle or a local minimum?** Measurement 3.3 shows
  symmetry breaking escapes and improves by 35.7%, consistent with a saddle.
  The transverse Hessian spectrum at the converged symmetric point would settle
  it and is computable at 64x64.
- **[open] Does the benchmark's noise escape `Delta` in 2000 epochs?** See
  Section 3's open item. This decides whether the published tables are affected.
- **[open] Convergence.** The paper states "as all of the methods utilize the
  gradient descent algorithm for a convex optimization problem, all of these
  have a sublinear convergence rate of O(1/N)" (Sec. 3.2.1), two paragraphs
  after noting the subproblems are "not strictly convex". The joint objective is
  a degree-4 polynomial in the factors: block-convex, globally non-convex. The
  stated rate has no support, and no convergence theory exists for the problem
  actually solved.
