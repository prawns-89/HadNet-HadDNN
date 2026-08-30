# Research pathways

Directions worth exploring for the Hadamard-decomposition optimisation work,
ranked by how directly they attack what we have measured. Add your own under
each heading — this file is meant to be argued with.

Context: current state and measurements are in `DECISIONS.md`. The short
version is that we have taken the init from O(n^5) to O(n^4), batched the
product for ~9x, and cut epochs ~3x. What remains is (a) the parameter count
scaling with the slice dimension `p`, and (b) the epoch count `k`.

---

## 1. Riemannian optimisation on the quotient manifold  ** strongest lead **

**[Riemannian Optimization for Hadamard Products of Low-Rank Matrices](https://arxiv.org/html/2606.01216)** (arXiv:2606.01216)

Directly on our problem: optimise `W = (GH^T) o (UV^T)`. It identifies and
*fixes* the exact issue the HadNet paper only notes in passing.

**The insight.** The Hadamard parameterisation has a **cross-Hadamard scaling
symmetry**: `(X,Y) -> (D_r X D_c, D_r^-1 Y D_c^-1)` for diagonal `D`. The
HadNet paper acknowledges non-uniqueness in Sec. 3.2.1 and moves on. This
paper shows the symmetry makes the landscape **ill-conditioned for Euclidean
optimisers** — which is what Adam is. That is a candidate explanation for why
HadNet needs thousands of epochs and plateaus so early (our Sec. 11 curves:
~87% of epochs buy <1% accuracy).

**The fix.** Formulate on a quotient manifold that divides out both the GL
reparameterisation and the scaling symmetry, with a scale-invariant
block-diagonal metric

    S_i^G = H^T Diag(Y_i,:^2) H ,    S_j^H = G^T Diag(Y_:,j^2) G

The Riemannian gradient then decomposes into independent r x r solves, and
the step size comes from a **tuning-free Gauss-Newton Cauchy rule** — no
`lr=0.1` magic constant.

**Why it matters for us.** Per-iteration cost **O(Nr + (m+n)r^4)**, *linear*
in N (observed entries), against BCD's O(Nr^2). It also handles non-quadratic
losses, which BCD cannot. Reported MovieLens-1M RMSE 0.87 vs BCD 0.90 vs
AGD 2.98.

**Connects to what we found.** Our Lipschitz experiment (Sec. 11) failed
partly because a joint Adam step ignores the geometry. This is the principled
version of what that experiment was reaching for. **If one thing gets tried
next, this is it.**

*Open question to argue about: their cost is `O((m+n)r^4)` in the second term.
At our r=10 that is 10^4 per slice — is that actually cheaper than what we
have, at our sizes rather than asymptotically?*

---

## 2. Sampling — where the "partial QR scan" intuition properly lands

**[Practical Leverage-Based Sampling for Low-Rank Tensor Decomposition](https://arxiv.org/abs/2006.16438)** (Larsen & Kolda, SIAM J. Matrix Anal. Appl. 43(3), 2022)

Sketches each ALS subproblem using **leverage scores** to pick a subset of
rows, with probabilistic accuracy guarantees. Headline: for a (d+1)-way
tensor the sketched system needs **O(r^d / eps) rows — independent of tensor
size and of the number of nonzeros.**

**Why we parked plain subsampling** (`DECISIONS.md` Sec. 2): the Carphone
redundancy ratio is only 3.6x (3.73M entries vs 1.04M parameters), so uniform
sampling has almost no headroom. Leverage-score sampling is the answer to
exactly that — it samples *informatively* rather than uniformly, so it needs
far fewer rows for the same accuracy.

**Caveat to check:** the guarantees are derived for CP/ALS. Whether they
transfer to the Hadamard (elementwise-product) structure is genuinely open —
the subproblems are convex least squares in each block, which is promising,
but the leverage scores would need rederiving.

Related: [More Efficient Sampling for Tensor Decomposition With Worst-Case
Guarantees](https://arxiv.org/html/2110.07631),
[Fast Low-Rank Tensor Decomposition by Ridge Leverage Score Sampling](https://arxiv.org/pdf/2107.10654).

---

## 3. More than two factors — the r^2 expressiveness argument, generalised

**[Efficient algorithms for the Hadamard decomposition](https://arxiv.org/abs/2504.13633)** (Wertz, Vandaele & Gillis, IEEE MLSP 2025) — this is ref [55] of the paper, the source of TBCD. [Code](https://github.com/StefanoSicilia/Hadamard-Decomposition).

Beyond the BCD algorithm we already benchmark, it **extends the framework to
more than two low-rank matrices**, reaching higher effective rank while
keeping parameter efficiency.

**Why this is interesting for us.** Our standing argument is that
`(W1H1) o (W2H2)` reaches rank ~r^2 using 2r(m+n) parameters, where SVD needs
r^2(m+n). With `k` factors that becomes rank ~r^k for kr(m+n) parameters. Since
per-epoch cost is *linear in r*, hitting a target accuracy with smaller r is a
direct time win. Worth testing: **k=3 factors at small r vs k=2 at large r, at
matched error, measured on wall-clock.**

---

## 4. Randomized linear algebra — the theory under what we already did

**[Finding Structure with Randomness](https://arxiv.org/abs/0909.4061)** (Halko, Martinsson & Tropp, SIAM Review 53(2), 2011)

The foundation for our `svd_initialize` change (lever 2). Worth reading for
the **error bounds**, which would let us replace "delta 1.7e-3, seems fine"
with an actual guarantee on the init quality — and would justify the `niter`
choice rather than tuning it empirically (our Sec. 13).

Sec. 10.7 has the bounds; Alg. 4.3/4.4 are the power-iteration schemes
`torch.svd_lowrank` implements.

---

## 5. Structural — stop giving every slice its own factors

Not a paper, our own finding, and possibly the biggest untapped win.

Currently `W1,H1,W2,H2` all carry the full `p` dimension, so **parameters
scale with the number of slices**. Carphone: 1.04M parameters for 3.73M
entries. For *video*, where consecutive frames are near-identical, that is
close to absurd.

Options: tie factors across `p`; or factorise along `p` (Tucker/tensor-train
style); or a shared basis with per-slice coefficients — the last is
essentially what Shard does for KV cache.

Fixes three things at once: shrinks the gradient (per-epoch cost), raises the
redundancy ratio so sampling (Sec. 2) becomes viable, and improves the
compression ratio, which is the paper's actual application.

Background: [Kolda & Bader, Tensor Decompositions and Applications](https://epubs.siam.org/doi/10.1137/07070111X) is still the best survey of the structural options.

---

## 6. Systems — the launch-overhead problem, structurally

We hand-fixed `h_prod` by batching. The general version:

- **[torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)** — fuses elementwise chains and cuts Python overhead.
- **[CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)** — captures a whole training step as one replayable graph. Aimed squarely at "many small kernels in a fixed-shape loop", which is exactly HadNet's inner loop.
- **[Triton](https://triton-lang.org/)** — if a fused Hadamard-product-plus-loss kernel is worth writing.

**Cautionary data point:** Shard ([krishgarg.com/shard](https://krishgarg.com/shard)) achieved 10x KV-cache compression and then gave back 2.5x throughput to *"many small kernel launches and tensor reshapes"* — the identical failure mode. Optimising FLOPs or memory without watching launch overhead is a known way to win on paper and lose on the clock.

---

## 7. Theory gaps — if the goal is a paper rather than a speedup

1. **Sample complexity of the Hadamard model.** Does rank-r^2 expressiveness
   at 2r(m+n) parameters actually give better recovery from partial
   observations than an additive model of matched rank? Nothing in the
   literature answers this for the Hadamard structure.
2. **Identifiability.** Sec. 3.2.1 notes non-uniqueness; Sec. 1 above
   characterises the symmetry group. What conditions give identifiability up
   to that group?
3. **Convergence rate.** Block-convexity is stated but no rate is given.
   Riemannian formulation (Sec. 1) is probably the route to one.

---

## 8. Optimisation paths not yet above — low maths risk

Ranked by expected value / maths risk. All attack `k` (now the dominant cost)
or the `p`-scaling — the two things Sec. 5 and Sec. 7 leave open.

### 8.1 Anderson acceleration / DIIS on the epoch loop  ** cheapest real win **

The 2000-epoch descent is a fixed-point iteration. Anderson acceleration
(numerical analysis; identical to Pulay's DIIS for SCF convergence in
electronic-structure codes) wraps *any* fixed-point map: keep the last `m≈5`
iterates and residuals, solve a tiny `m x m` least-squares each step for the
extrapolation coefficients. Routinely 2–5x fewer iterations, ~15 lines, no
change to the model or loss, composes with early stopping and `h_prod`
batching. The Sec. 11 curves (87% of epochs buy <1% accuracy) are exactly the
slow-tail regime it collapses. **Belongs above CUDA graphs in the
"cheap, mechanical, no-maths-risk" slot.** First try can be even simpler —
Nesterov restart.

### 8.2 Explicit balancedness / matrix-balancing — the 3-line version of Sec. 1

Sec. 1's whole point is the cross-Hadamard scaling symmetry
`(X,Y) -> (D_r X D_c, D_r^-1 Y D_c^-1)` wrecking Adam's conditioning. The full
quotient manifold is not needed to neutralise the worst of it. Deep-linear-net
theory (Arora et al., *Implicit Acceleration by Overparameterization*) shows
gradient flow keeps `W^T W − H H^T` constant; enforce it explicitly — after
each step rescale so `‖W_i‖_F = ‖H_i‖_F` per factor pair, or run a few
Sinkhorn–Knopp row/column rescales to divide out `D_r, D_c`. Picks a unique,
well-conditioned representative for a couple of norms / mat-vecs per epoch.
**Test this before committing to the Riemannian rewrite** — if it recovers most
of the epoch saving, Sec. 1 drops from "big rewrite" to "done".

### 8.3 K-FAC as the tractable Riemannian metric

The scale-invariant block metric in arXiv:2606.01216,
`S_i^G = H^T Diag(Y_{i,:}^2) H`, *is* a Gauss–Newton / Fisher block. K-FAC
(Kronecker-factored curvature, from DL optimisation) is the standard scalable
approximation: each factor block gets a small `r x r` curvature factor,
inverted cheaply. Likely the practical route to Sec. 1 without implementing
manifold retractions — geometry-aware step at ~Adam cost. At `r=10` the `r x r`
solves are trivial, so the only real cost is forming the curvature factors:
open question whether that beats one Riemannian retraction.

### 8.4 Continuous-coordinate factors for `p` — makes Sec. 5 concrete

Sec. 5 says "shared basis + per-slice coefficients". The mechanism from
graphics: represent the slice index as a coordinate and let `W1(p) = MLP(p)`
(instant-NGP hash grid, or a small SIREN), so parameters **stop scaling with
the number of slices** rather than merely shrinking. For Carphone this replaces
the `x180` on the parameter count with a fixed-size network. NeRV / HNeRV
(neural video representation) are exactly "per-frame coefficients over a shared
decoder" and report codec-competitive ratios. Fixes all three things Sec. 5
lists, and attacks `k` too if the network init is meta-learned (Sec. 9.6).

### 8.5 Adaptive Cross Approximation as the init — redeems the "partial scan"

`DECISIONS.md` Sec. 2 sent the QR-code "scan without the whole symbol in frame"
intuition to completion, then parked it. It lands cleaner on **ACA** (H-matrix
/ boundary-element literature): build a rank-`r` approximation from `O(r)`
adaptively chosen rows and columns, `O(mn r)` per slice, never forming the full
matrix. Same `O(n^4)` target as the randomized SVD (Sec. 6 / lever 2) but
smaller constant and **no power iterations to tune** — the Sec. 13 `niter`
guard disappears. Worth a bench against `torch.svd_lowrank` at the Carphone
shape.

### 8.6 Poisson / Gamma likelihood -> multiplicative updates -> no step size

The data is non-negative and the model multiplicative. Swap the Gaussian
Frobenius loss for a Poisson / Gamma likelihood with a log link and Lee–Seung
**multiplicative updates** apply: monotone decrease guaranteed, *zero*
step-size hyperparameter — deletes the `lr=0.1` magic constant (`DECISIONS.md`
Sec. 9) and, with no lr to be conservative about, can cut `k`. Risk: [0,1] data
is not count data, so the likelihood is a modelling choice — but `DECISIONS.md`
Sec. 16 already shows the current Gaussian+noise setup evaluates against an
out-of-range target, so the current likelihood is not sacred either.

---

## 9. Translations from other fields

Each entry: the construct, what it maps onto here, what to try, which measured
observation it explains or attacks.

### 9.1 Cepstral analysis / homomorphic signal processing  ** highest leverage **

In speech processing, `log` of a spectrum turns convolution into addition (the
cepstrum); homomorphic filtering separates multiplied signals in log-domain.
Here: `log|X| ≈ log|W1 H1| + log|W2 H2|` — the multiplicative quartic becomes an
**additive** model of two low-rank-in-log terms. The scaling symmetry
`D_r X D_c` becomes an *additive* rank-1 shift `(log d_r) 1^T + 1 (log d_c)^T`
that a single rank-1 term absorbs — the ill-conditioning Sec. 1 fights just
goes away. **Try:** optimise `A + B` (each low-rank) against `log|X|`, existing
sign split handling `sign(X)`, map back by `exp`; compare conditioning and `k`.
Explains Sec. 1's ill-conditioning and the Sec. 11 plateau.

### 9.2 Anderson acceleration / DIIS (quantum chemistry, numerical analysis)

See Sec. 8.1. SCF convergence acceleration in electronic-structure codes is the
same slow-fixed-point problem as the epoch tail.

### 9.3 MPS / DMRG canonical form (quantum many-body physics)

Matrix-product states have exactly the GL "bond gauge" freedom; DMRG fixes it
with a **canonical form** maintained by a left-to-right QR/SVD sweep, and
optimises one tensor at a time with the gauge frozen. The generalized Hadamard
product is a diagonal ("Hadamard") tensor-network contraction; `r -> r^2` rank
growth under `∘` is entanglement growth under a diagonal two-site gate.
**Try:** a canonical-form sweep over `W1->H1->W2->H2` (gauge-fix by QR each
move) as a lightweight alternative to the Sec. 1 quotient manifold — symmetry
removal without Riemannian retractions. Gives the block-sweep structure Sec. 11
says is the regime where `1/L` steps are actually justified.

### 9.4 Discrepancy principle / L-curve (geophysical inverse problems)

Seismic tomography and NWP data assimilation stop iterating when the data
misfit reaches the observation-error level. The scripts inject *exactly*
`0.01 * randn` (`DECISIONS.md` Sec. 16), so `E‖noise‖_F = 0.01 * sqrt(#entries)`
is known in closed form. **Stop when the residual crosses that** — a principled
`k`, no tuning, transfers across shapes. Replaces the `EarlyStop(min_rel=1e-3)`
patience knob; compare epoch count and error against it.

### 9.5 K-FAC / natural gradient / information geometry

See Sec. 8.3. The Fisher metric from statistical estimation = the
scale-invariant metric in arXiv:2606.01216; K-FAC is the productionised
approximation.

### 9.6 Neural fields + meta-learned initialisation (computer graphics)

NeRV / HNeRV represent a video as a small network `frame(t)`; Tancik et al.,
*Learned Initializations for Coordinate Networks*, meta-learn (MAML / Reptile)
an init so a *new* signal fits in a fraction of the iterations (10x+ reported).
Maps to Sec. 5 (shared basis + per-slice coefficients) and `DECISIONS.md`
Sec. 9's "reduce `k`". Fitting each Carphone from a cold SVD init is the thing
meta-learning removes. **Try:** (i) coordinate-net factors along `p` (Sec. 8.4);
(ii) meta-learn the factor init, or the `svd_initialize` hyperparameters,
across a small video corpus; measure `k` on a held-out clip.

### 9.7 Matrix balancing / Sinkhorn–Knopp (optimal transport, RAS)

The diagonal-scaling equivalence class `D_r X D_c` has a canonical
representative — the doubly-balanced one — found by alternating row/column
normalisation. Exactly the cross-Hadamard symmetry of Sec. 1; Sinkhorn gives a
unique, condition-optimal gauge for a few mat-vecs every `N` epochs. See
Sec. 8.2.

### 9.8 Phase retrieval / blind deconvolution (computational imaging)

`sqrt(|X|)` init plus signed self-product is phase-retrieval-shaped
(`X = M ∘ M` vs `|Ax|^2`). Wirtinger-flow spectral-init analysis
(Candès–Chen) is a near-drop-in route to the recovery guarantee Sec. 7.1 asks
for, and would *derive* the `sqrt(|X|)` init instead of it being folklore.
Blind deconvolution's convex lift (Ahmed–Recht–Romberg) gives a
global-optimality certificate on small instances — the tool Sec. 7.2
(identifiability) needs.

### 9.9 Adaptive Cross Approximation / H-matrices (boundary element methods)

See Sec. 8.5. BEM assembles rank-`r` blocks of a dense operator from `O(r)`
rows/columns without forming the block — the principled "partial scan" init
`DECISIONS.md` Sec. 2's QR-code instinct was reaching for.

### 9.10 Poisson matrix factorisation / multiplicative updates (recommenders, topic models)

See Sec. 8.6. Count-data recommenders fit `X ≈ WH` with a log link by
parameter-free multiplicative updates — the natural home for a multiplicative
model, and it deletes `lr=0.1`.

### 9.11 Shorter mentions

- **Spectral bias** (physics-informed NNs) — the Sec. 11 plateau may be
  frequency-dependent; a DCT-basis reparameterisation of `W,H` (natural for
  image/video) could rebalance convergence.
- **Fountain / rateless codes** — once Sec. 5 shares factors, each slice's
  residual is "parity over a shared basis"; rateless design tells you the
  samples-per-slice needed for the Sec. 2 completion angle.

---

## 10. Gaps in what is currently measured

Close these before any headline number is defensible.

| Gap | Why it matters | Fix |
|---|---|---|
| **No end-to-end GPU baseline.** `DECISIONS.md` Sec. 15 has two kernel microbenchmarks; `GPU.md` Sec. 6 item 1 (full `main` vs `perf/batch-hprod` on the RTX 4050) is still outstanding. Every whole-run wall-clock claim is a CPU projection. | The composed claim (levers 1·3·4 multiplicative) is untested; kernel speedups do not linearly compose into loop speedups once optimiser + autograd are in. | Run it — ~minutes with the AGD/IGD/BCD blocks commented out (`COMMANDS.md` Sec. 3). |
| **Early stopping and randomized init are confounded.** Both shift error ~1% (Sec. 10, 12) and are always applied together. | Can't attribute the accuracy cost or detect interaction (a rougher init making early stopping quit sooner). | 2x2 grid: {exact, randomized} x {k=2000, EarlyStop}, fixed seed, error + wall-clock for all four. |
| **Stopping rule is a patience knob, not tied to the noise floor.** `min_rel=1e-3` is heuristic; the injected noise is exactly `0.01*randn`. | The discrepancy principle (Sec. 9.4) gives a principled `k` with no tuning. | Compute the target residual from tensor size; stop there; compare against `EarlyStop`. |
| **Baselines are not like-for-like** (`DECISIONS.md` Sec. 4.4): 40000 iters vs 2000 epochs, numba-CPU vs GPU. The paper's 21x over TAGD ≈ the iteration-count ratio alone. | A reviewer catches this immediately if the goal is an extension paper. | Re-run baselines at matched iteration budgets and, where possible, matched hardware; document the honest ratio. |
| **`squared` vs unsquared Frobenius inconsistent across folders** (`DECISIONS.md` Sec. 14): `real-world/` squares, the tensor folders don't. | Cross-folder comparison of error / convergence is apples-to-oranges; the Sec. 11 Lipschitz theory only applies to the squared form. | Pick one — `squared_loss` is implemented and Sec. 11 showed it marginally better (0.3303 vs 0.3310). |
| **`lipschitz_lr`'s max-over-blocks step is shipped but rejected** (`DECISIONS.md` Sec. 11) and not a guarantee even in principle. | Dead code that looks load-bearing. | Delete or clearly quarantine. |
| **No wall-clock for Sec. 3 (k>2), Sec. 5 (sharing), Sec. 2 (leverage sampling).** All paper-only. | The suggested order ranks them but nothing is measured. | Sec. 3 is cheapest (existing codebase): `k=3` small-`r` vs `k=2` large-`r` at matched error first. |

**Theory gaps (extend Sec. 7):** a convex lift for global-optimality
certificates on small instances (Sec. 9.8); sample complexity via phase
retrieval, which also justifies the `sqrt(|X|)` init (Sec. 9.8).

---

## 11. Ensemble — rank-continuation ladder across the sweep that already runs

Not a new mechanism, a **free reuse of work already being thrown away.** Every
experiment folder runs a rank sweep — `ranks = [2, 5, 10, 20, 25]`
(`COMMANDS.md` Sec. 3: L258 / L265 / L284) — and each rank is fitted from a
**cold SVD init, independently.** The `r=25` run does not know the `r=10` run
ever happened.

**The idea.** Chain them. Fit `r=2` to convergence; embed its factors into an
`r=5` parameterisation (pad `W,H` with the next 3 randomized-SVD directions of
the *current residual*, not of `X`); continue; repeat up the ladder. Classic
numerical-continuation / homotopy: track the solution as the parameter (`r`)
moves rather than re-solving from scratch.

**Why it should pay here, tied to what is measured:**

- `DECISIONS.md` Sec. 11 — the epoch loop is ~the entire runtime, and 87% of
  epochs buy <1% accuracy. That wasted tail is spent re-discovering structure
  the smaller-`r` run already found. A warm start skips it.
- `DECISIONS.md` Sec. 12 — `EarlyStop` is relative, so it transfers across
  ranks without retuning; the ladder rungs each get it for free and each stops
  as soon as the *incremental* rank stops helping.
- `DECISIONS.md` Sec. 3 — init is 1–3% of runtime, so the extra per-rung
  randomized SVD on the residual (Sec. 4 / lever 2) is noise against the
  epoch-loop saving.
- Total sweep cost, not just peak `r`: today `sum over ranks`; with continuation
  the big ranks converge in a fraction of their cold `k`.

**Composes with:** Sec. 8.1 (Anderson accelerates each rung), Sec. 9.1 (build
the ladder in log-domain), Sec. 3 (a `k`-factor model has its own ladder — grow
factors *and* rank).

**Open question to argue about.** The Hadamard model is rank-`r^2` expressive
at inner rank `r` (the paper's core claim). Does the residual after an `r=2`
fit even *have* a clean rank-3 increment to peel off, or does the
elementwise-product structure smear the spectrum so the continuation step is no
better than a cold `r=5`? Cheap to check: one folder, `ranks=[2,5,10]`, cold vs
laddered, compare summed wall-clock at matched final error.

**Failure mode to watch.** Continuation can lock in a bad basin from the `r=2`
rung — the small model commits to structure the larger one must then fight. If
laddered final error at `r=10` is worse than cold `r=10` by more than the
Sec. 12 early-stop budget (~1%), the ladder is buying speed with accuracy and
should be dropped.

---

## 12. Multigrid / coarse-to-fine on the spatial grid (numerical PDEs, vision pyramids)

A second continuation axis, **orthogonal to Sec. 11's rank ladder**: not
`r` coarse->fine, but the *spatial resolution* of the tensor itself.

**The construct.** Multigrid solvers for PDEs, and image pyramids in optical
flow / registration, both exploit the same fact: iterative solvers kill
high-frequency error fast and low-frequency error agonisingly slowly. The fix
is to solve the low frequencies on a cheap coarse grid and *prolongate*
(interpolate) that solution up as the initial guess for the fine grid.

**Mapping to this problem.** The factors are indexed by spatial coordinates —
row `i` of `W1[:, :, p]` corresponds to image row `i`. So:

1. Pool the data tensor `144x144x... -> 72x72x...` (or a 3-level pyramid).
2. Fit `W_coarse (72 x r x p)`, `H_coarse (r x 72 x p)` — a quarter of the
   entries, so each epoch is ~4x cheaper.
3. Bilinearly upsample the factor columns along their spatial axis
   (`72 -> 144`), use as the fine-grid init; short fine-grid polish.
4. Optionally V-cycle: drop back to coarse to sweep out low-frequency
   residual that the fine grid reintroduced slowly.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 11 — 87% of epochs buy <1% accuracy, the classic
  "low-frequency error decays slowly" signature. Coarse grids attack exactly
  that mode; this is arguably the *mechanistic* reason for the plateau that
  Sec. 9.1 (log-domain) and Sec. 1 (conditioning) each explain differently —
  worth disambiguating which story the curves actually support.
- `DECISIONS.md` Sec. 3 — runtime is essentially the epoch loop, and epoch
  cost is `O(k r mnp)`; quartering `mn` on the coarse level is a direct `4x`
  per-epoch cut for however many epochs run there.
- Sec. 5 — video frames are near-identical, i.e. dominated by low spatial
  frequencies, which is precisely where the coarse grid carries the most
  information. Image/video folders should gain more than `random-tensor`,
  which has no spatial smoothness — a clean falsifiable prediction.

**Composes with:** Sec. 11 (ladder in `r` x ladder in resolution = a 2D
continuation schedule), Sec. 8.1 (Anderson per level), Sec. 9.6 (the coarse
solve is itself a cheap learned-ish init).

**Open question to argue about.** Prolongation assumes the factors are spatially
smooth. The left singular vectors of a natural image mostly are, but the
`sqrt(|X|)` / signed-split transform (lever 2) is nonlinear and may roughen
them. Check: correlation between `W_coarse` upsampled and the converged
fine-grid `W1` — if it's low, prolongation is just noise and only the data
pyramid (not the factor pyramid) is useful.

**Failure mode.** Pooling is a low-pass filter; if the target detail lives
above the coarse Nyquist limit, the coarse fit commits to a wrong low-rank
structure (same basin-lock risk as Sec. 11). Gate on the same ~1% budget.

---

## 13. Streaming / subspace tracking along `p` (adaptive signal processing)

Every method here fits all `p` slices **jointly, in one batched solve**. For
video that is the wrong shape for the data: frame `t` is nearly frame `t-1`.

**The construct.** Adaptive filtering has decades on exactly this — recover a
slowly drifting low-rank subspace from a stream: PAST, GROUSE (Grassmannian
rank-one update), incremental / updating SVD, recursive least squares. Cost
per new column is `O(mr)`, not another full decomposition.

**Mapping to this problem.** Walk the slice index as time:

1. Fit `W1,H1,W2,H2` for slice `0` (cold, via lever 2).
2. For slice `t`: initialise from slice `t-1`'s factors, take a *handful* of
   epochs — the subspace barely moved — accept.
3. Optionally keep a shared "base" subspace plus a rank-one-ish per-slice
   correction: that is Sec. 5's "shared basis + per-slice coefficients"
   arrived at from the tracking side instead of the structural side.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 2 — the redundancy-ratio arithmetic (3.73M entries /
  1.04M params = 3.6x) treats the 180 slices as independent. Consecutive-frame
  correlation is unused redundancy; tracking is how you spend it without
  needing the leverage-score machinery (Sec. 2) first.
- `DECISIONS.md` Sec. 3 — cost is `O(k r mnp)` with the *same* `k` for every
  slice. Tracking makes `k` per-slice collapse for `t>0` (only slice 0 pays
  full price), so effective cost -> `O(k r mn) + O(p * k_warm * r mn)` with
  `k_warm << k`.
- Sec. 5 predicts video gains most from sharing; tracking predicts the same
  ordering (video >> image >> `random-tensor`, which has no slice ordering at
  all) — a second falsifiable version of that prediction, by a different
  mechanism.

**Composes with:** Sec. 5 (tracking *is* a route to the shared-basis
structure), Sec. 11 / 12 (all three are warm-start schedules on different
axes — rank, resolution, slice), Sec. 8.1 (Anderson for the slice-0 cold fit).

**Open question to argue about.** Video is not monotone drift — scene cuts,
occlusion, fast motion break the "barely moved" assumption. Does a cut force a
full cold re-fit (detect via residual spike, fall back), and how often do cuts
happen in the benchmark clips (Carphone: probably never; other clips: unknown)?
If the test videos are all single continuous shots, tracking looks better than
it would in the wild — state that caveat with any number.

**Failure mode.** Error accumulates along the chain: slice `t` inherits slice
`t-1`'s fit error plus its own. Over 180 slices a slow drift in accuracy could
exceed the ~1% budget by the end. Mitigation: periodic cold "keyframe" re-fits
(again, straight out of video codec design).

---

## 14. Quantised factors — optimise the metric the paper actually reports (transform coding, PTQ)

Everything above optimises *time*. But Sec. 5 notes the paper's real deliverable
is the **compression ratio**, and there the factors are still stored as fp32.

**The construct.** Image/video codecs (JPEG, the DCT-coefficient quantisation
tables) and post-training quantisation for neural nets both do the same thing:
after fitting a continuous representation, spend bits where they buy the most
error reduction and round the rest. A rank-`r` factorisation stored at `b` bits
per entry compresses at `32/b` times the ratio the paper reports, for free, if
the error holds.

**Mapping to this problem, tied to what is measured:**

- `DECISIONS.md` Sec. 16 — data is normalised to `[0,1]` and the multiplicative
  structure is "naturally self-limiting … stays near [0,1]". Bounded factors are
  the easy case for uniform quantisation: the range is known a priori, no
  calibration pass.
- `DECISIONS.md` Sec. 15 — on GPU, fp32 vs fp64 already moves error only in the
  7th decimal, and TF32 was ruled out as *not* the cause of a visible shift. So
  there is measured headroom below fp32 that a 2000-epoch descent washes out —
  the same argument lever 2 (Sec. 4) uses to justify the randomized init.
- Sec. 11 (rank ladder) — "overshoot rank then prune" is bit allocation by
  another name: fit at `r+δ`, then quantise-and-prune the directions whose
  singular values fall below the quantisation noise floor.

**What to try.** (i) Post-hoc: fit in fp32, uniformly quantise `W,H` to int8 /
fp16, measure Frobenius error and the new compression ratio. (ii)
Quantisation-aware: straight-through estimator on the factors during the last
few hundred epochs (which Sec. 11 of `DECISIONS.md` shows are otherwise wasted),
so the fit adapts to the rounding.

**Open question to argue about.** SVD quantisation error is *additive* — round
`U, Σ, V` and the reconstruction error adds. Here the reconstruction is
`(W1 H1) ∘ (W2 H2)`, so factor rounding error **compounds multiplicatively**:
`(WH + E1) ∘ (UV + E2) = WH∘UV + E1∘UV + WH∘E2 + E1∘E2`. The cross terms scale
with the *other* factor's magnitude, so the bit budget should not be split
evenly between the two factor pairs — it should follow their dynamic ranges.
Worth deriving the optimal allocation rather than guessing int8-everywhere.

**Failure mode.** If a factor pair has a few large-magnitude columns carrying
most of the energy (likely, given the `sqrt(|X|)` init seeds from singular
values), uniform quantisation wastes codes on the small columns and clips the
large ones. Needs per-column scales — cheap, but it is another `2r` numbers to
store, which eats back some of the ratio. Account for it honestly.

---

## 15. Bundle-adjustment-style Schur-complement LM (photogrammetry, SLAM)

`DECISIONS.md` Sec. 11 establishes block-convexity: with three of the four
factors fixed, the loss is a convex quadratic in the fourth. Sec. 8.3 proposes
K-FAC to approximate the curvature; there is a more literal translation.

**The construct.** Bundle adjustment — the back end of every
structure-from-motion and SLAM pipeline — minimises a large bilinear
least-squares (camera parameters x 3-D points) by Levenberg–Marquardt, and
makes it tractable with the **Schur complement**: analytically eliminate one
block (the points) to get a much smaller dense system in the other (the
cameras), solve that, back-substitute. Ceres and g2o are built around this one
trick.

**Mapping to this problem.** The four factors split naturally into two groups.
Freeze `W2,H2`; the residual `Ω ∘ (X − (W1 H1) ∘ M2)` (with `M2 = W2 H2` a
fixed elementwise mask) is linear in `W1` and in `H1` separately — the exact
bilinear-LS shape BA solves. Form the Gauss–Newton normal equations for
`(W1,H1)`, Schur-complement out `H1` (it is block-diagonal across columns of
`X`, so the elimination is cheap), solve the reduced system for `W1`,
back-substitute. Then swap groups. One such sweep is a damped second-order step,
not a gradient step.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 11 — the per-row design matrix `G_i = diag(M2[i,:]) H1^T`
  and curvature `L_i` are *already derived there* for the Lipschitz experiment.
  BA uses that same `G_i` to build the normal equations rather than just its
  spectral norm for a step size. The groundwork is done.
- `DECISIONS.md` Sec. 11 also concludes the plain `1/L` step is "worse on both
  axes" because it is conservative and Adam already adapts. A Gauss–Newton step
  is the non-conservative use of the same second-order information — the
  direction the rejected experiment was one bad approximation away from.
- `DECISIONS.md` Sec. 3 / Sec. 6 — LM converges superlinearly near the optimum,
  so it attacks `k` (the dominant `n^4` multiplier) where the 87%-wasted tail
  lives, not the already-cheap early epochs.

**Composes with:** Sec. 9.3 (the group-swap sweep is the DMRG two-site sweep in
another accent), Sec. 1 (Schur complement is one concrete realisation of "the
Riemannian gradient decomposes into independent r x r solves").

**Open question to argue about.** BA is *sparse* — each 3-D point is seen by a
handful of cameras, which is what makes the Schur complement cheap. Here the
data tensor is **dense**, so the reduced system after eliminating `H1` is dense
`mr x mr`. At Carphone `m=144, r=10` that is `1440 x 1440` per slice — a
`~3 GFLOP` solve times 180 slices per sweep. Is that actually below the cost of
the ~1700 gradient epochs it would replace? Needs the arithmetic before coding.

**Failure mode.** LM damping (`λ`) is another schedule to get wrong, and a
Gauss–Newton step on the *joint* non-convex quartic (as opposed to one frozen
block) has no descent guarantee — the same objection Sec. 11 raises against the
joint `1/L` step. Keep it strictly to the frozen-group subproblem, or wrap it
in a line search / trust region.

---

## 16. Block-Krylov init — change the shape of the Sec. 13 tradeoff, not just its knob

`DECISIONS.md` Sec. 13 measured the power-iteration tradeoff for the randomized
init and picked `niter=4` off it:

    niter   1     2     3     4     6     8    12    24
    speed  3.6x  2.8x  2.3x  1.9x  1.4x  1.3x  0.9x  0.4x
    delta  1.5e-2 6.8e-3 3.2e-3 1.7e-3 4.7e-4 1.8e-4 2.5e-5 4.8e-7

That table is the accuracy/pass curve of **subspace iteration** — what
`torch.svd_lowrank` runs. It is not the only randomized algorithm.

**The construct.** Randomized **block Krylov** (Musco & Musco, NeurIPS 2015,
*Randomized Block Krylov Methods for Stronger and Faster Approximate SVD*)
builds the subspace from `span{Ω, AΩ, A²Ω, …, A^q Ω}` and solves inside it,
instead of just iterating `A^q Ω`. Same matvec count `q`, but the error decays
in `1/q²` rather than `1/q` — the `sqrt(eps)`-vs-`eps` gap that separates
conjugate gradients from steepest descent. In practice it hits `niter=24`
accuracy in about `niter=4`–`6` passes.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 13 — the crossover where randomisation stops paying is
  `niter·(r+oversample) < min(m,n)`, and the `niter` guard in
  `svd_initialize` clamps to stay left of it. Block Krylov reaches the target
  accuracy in fewer passes, so it moves that crossover *right* — randomisation
  stays profitable at accuracies the current scheme has to fall back to exact
  for.
- `DECISIONS.md` Sec. 14 — `real-world/` keeps `niter=24` because at
  `min(m,n)=256` it is still profitable there (delta 3.3e-7). Block Krylov
  would get that same delta in ~6 passes, i.e. a further ~3–4x on that folder's
  init with no accuracy loss — the one place the current knob is pinned high.
- `DECISIONS.md` Sec. 6 — the init is the only super-quartic term
  (`O(n^5) -> O(n^4)` was the whole point of lever 2). A smaller constant on
  the `n^4` term is a direct scaling win, not just a wall-clock one.

**Composes with:** Sec. 4 (this *is* the Halko/Martinsson/Tropp line, one
paper further on — same error-bound machinery, so Sec. 4's "replace
'seems fine' with a guarantee" goal comes with it), Sec. 11 / 13 (every rung
and every slice re-inits, so a cheaper init multiplies through the ladder).

**Open question to argue about.** Block Krylov's `1/q²` rate assumes a spectral
gap; the `sqrt(|X|)` / signed-split matrices (lever 2) may have a flat tail
(many near-equal small singular values), where block Krylov degrades toward the
power-iteration rate and the extra orthogonalisation per pass is pure overhead.
Cheap to check: plot the singular spectrum of `sqrt(|X|)` for one Carphone
slice before committing.

**Failure mode.** The Krylov basis needs reorthogonalisation each pass
(`O(m q²)` per slice) or it loses rank numerically — for large `q` that term
catches up with the matvecs it was meant to save. Keep `q` in the same
single-digit range Sec. 13 already found optimal; block Krylov changes the
accuracy you get *at* that `q`, not the `q`.

---

## 17. Additive-to-multiplicative homotopy (graduated non-convexity, robust fitting)

Sec. 11 / 12 / 13 are continuation on *rank*, *resolution*, *slice index*. This
is continuation on the **non-convexity of the objective itself** — and it turns
Sec. 9.1's additive surrogate from a rival formulation into the *start* of a
path that ends at the true model.

**The construct.** Graduated non-convexity (Blake–Zisserman; Black–Rangarajan
line processes; modern certifiable robust fitting — FGR, TEASER) replaces a
hard non-convex loss with a family `L_γ`, `γ: 0 -> 1`, where `L_0` is convex or
benign and `L_1` is the real thing, then tracks the minimiser along `γ`.

**Mapping to this problem.** Interpolate the *model*, not the loss:

    X_hat_γ = (1 − γ) (A + B)  +  γ (M1 ∘ M2)

`A, B` low-rank additive (bi-convex, textbook, no scaling symmetry — Sec. 1's
ill-conditioning is absent at `γ=0`); `M1 = W1 H1`, `M2 = W2 H2` the Hadamard
model. Solve `γ=0` cheaply, step `γ` up, warm-starting each solve from the
last, initialising `M1,M2` from `A,B` (e.g. `M1 <- sqrt|A+B|`-style, reusing
the lever-2 transform). At `γ=1` it is exactly the paper's model.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 11 — the plateau (87% of epochs, <1% gain) is the
  signature of a descent stuck near a poor stationary point of the quartic.
  GNC exists precisely to route around poor local minima by deforming in from a
  problem that has none.
- `DECISIONS.md` Sec. 11 — the Lipschitz experiment failed partly because "a
  joint Adam step ignores the geometry". The `γ=0` additive problem *has* the
  benign geometry; the homotopy carries a good iterate into the region where
  the geometry goes bad, instead of starting there.
- `DECISIONS.md` Sec. 3 — init is 1–3% of runtime, so the extra `γ=0` solve
  (one bi-convex low-rank fit, cheaper than the Hadamard fit) is affordable
  head-room, same accounting as lever 2.
- Sec. 9.1 and Sec. 1 are the two endpoints of this path — the homotopy is
  literally an *ensemble* of those two sections rather than a choice between
  them.

**Composes with:** Sec. 11 / 12 / 13 (a fourth continuation axis; schedule
`γ` jointly with rank), Sec. 8.1 (Anderson per `γ`-step), Sec. 3 (`k>2` factors
— the additive surrogate for a `k`-fold product is a sum of `k` low-rank terms).

**Open question to argue about.** Homotopy only works if the solution path is
**connected** — if the additive and multiplicative models have genuinely
different global structure, the minimiser can jump basins at some `γ*` and the
warm start is worthless past it. Cheap probe: solve at
`γ ∈ {0, 0.25, 0.5, 0.75, 1}` cold, and check whether the cold `γ=1` solution
is close to the one reached by stepping the path. If the path is smooth, the
stepped version should win big on `k`.

**Failure mode.** A fold or bifurcation in the path (common in continuation
problems) makes naive `γ`-stepping diverge or backtrack; the fix is
pseudo-arclength continuation (numerical bifurcation theory), which is real
extra machinery. Only reach for it if the uniform-`γ` schedule visibly stalls.

---

## 18. Gauge-fix by construction — Stiefel `W`, no manifold optimiser (gauge theory, orthogonal parametrisations)

Sec. 1 divides out the symmetry with a quotient-manifold *optimiser*. Sec. 8.2
divides it out *after each step* by rescaling. Sec. 9.3 divides it out with a
*sweep*. There is a fourth option that touches neither the optimiser nor the
loop: **remove the redundant coordinates from the parametrisation itself.**

**The construct.** Gauge theories fix the gauge by imposing a condition (Lorenz,
Coulomb) rather than carrying the redundancy around. The ML version is
orthogonal / Stiefel parametrisation: store `W` as `Q` from a QR / Cayley /
Householder map so `W^T W = I` holds *by construction*, and let Adam optimise
the unconstrained pre-image. `torch.nn.utils.parametrizations.orthogonal` is
exactly this and leaves the training loop untouched.

**Mapping to this problem.** Constrain `W1, W2` to orthonormal columns; push all
scale into `H1, H2`. This kills the `GL(r)` reparameterisation
`(W,H) -> (W R, R^-1 H)` outright — only an orthogonal `R` survives, a set of
measure zero next to the full `GL(r)`. What remains is the diagonal
cross-Hadamard scaling, which Sec. 8.2 / 9.7 mop up cheaply. Stiefel-`W` +
balanced-scaling together retire the whole symmetry group with no new solver.

**No expressiveness cost (for HadNet).** Any `W1 H1` has a QR form `Q R H1`, so
`W1 H1` — and therefore the reconstruction `(W1 H1) ∘ (W2 H2)` — spans the
identical set. The constraint is free for the linear factors. (Not free for
HadDNN, where the factors are nets and "orthonormal columns" is a real
restriction — flag that separately.)

**Tied to what is measured:**

- `DECISIONS.md` Sec. 11 — Sec. 1 blames the plateau on symmetry-induced
  ill-conditioning; this removes the `GL(r)` part of that symmetry with a
  two-line wrap and zero algorithm change, so it is the cheapest test of
  whether that diagnosis is right.
- `DECISIONS.md` Sec. 11 — the Lipschitz experiment's bound uses
  `sigma_max(H)` and `max|m|`; with `W` orthonormal the `W`-block's
  `sigma_max` is exactly `1`, so `block_lipschitz` stops being "conservative …
  1/L ~ 2e-5 against Adam's 0.1" for that block and the `1/L` step might
  actually be usable there.
- `DECISIONS.md` Sec. 3 — the SVD init already produces orthonormal `U`; the
  current run starts on the Stiefel manifold and immediately drifts off it.
  This just keeps it there.

**Composes with:** Sec. 8.2 / 9.7 (handle the residual diagonal scaling),
Sec. 15 (orthonormal `W` makes the Schur-complement normal equations better
conditioned — no damping guesswork), Sec. 1 (this is the "poor man's" version;
if it recovers most of the gain, Sec. 1 is not needed).

**Open question to argue about.** Is the `GL(r)` reparameterisation actually
the part that hurts, or is it the *diagonal* cross-Hadamard scaling (the one
Sec. 1 emphasises)? If the latter, Stiefel-`W` alone does little and the win is
entirely in the Sec. 8.2 rescale. The 2x2 ablation — {plain, Stiefel-W} x
{no rescale, balanced rescale} — settles it in one run.

**Failure mode.** The parametrisation runs a QR (or Cayley solve) every
forward, and the factors carry the `p` slice dim — so this is another
"180 tiny ops in a loop" hazard, exactly the lever-1 / Sec. 4.1 failure mode.
`torch.linalg.qr` batches over leading dims, so permute the slice dims forward
first (same trick as `h_prod`); if that is skipped the constraint overhead will
swamp the conditioning benefit.

---

## 19. Predictive + entropy coding of the factors along `p` (video codecs, information theory)

An ensemble: Sec. 13 (frames barely move) + Sec. 14 (quantise the factors) +
Sec. 5 (compression is the deliverable) → a storage format, not just a
fitting trick. Sec. 14 stops at fixed `b` bits per entry; **no real codec
stops there.**

**The construct.** Video codecs store frame `t` as a *residual* against a
prediction from frame `t-1` (P-frames / motion compensation), then
**entropy-code** that residual — arithmetic / range / ANS coding — because its
distribution is sharply peaked, so its entropy sits far below the bit-depth.
JPEG does the same on DCT coefficients (Laplacian-distributed → Huffman/arith).

**Mapping to this problem.** After fitting:

1. **Delta along `p`:** store `W1[:,:,0]` in full, then
   `ΔW1[:,:,t] = W1[:,:,t] − W1[:,:,t−1]` for `t>0`. For video these deltas are
   near-zero (Sec. 13's premise, reused).
2. **Quantise the deltas** (Sec. 14), not the raw factors — smaller range,
   fewer bits for the same error.
3. **Entropy-code** the quantised deltas. Low-rank factors of natural images
   are roughly Laplacian per column, like DCT coefficients, so expect a
   further ~1.5–3x beyond Sec. 14's fixed-bit ratio.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 2 — the redundancy-ratio arithmetic (3.73M / 1.04M =
  3.6x) explicitly notes "for video, where consecutive frames are highly
  correlated, that is wasteful". That waste is exactly what Δ-along-`p` spends,
  and it needs no change to the fit — just to how the result is written out.
- `DECISIONS.md` Sec. 12 — the last ~87% of epochs buy <1% accuracy. Spend
  that tail instead on a **compressibility penalty** (total variation of the
  factors along `p`, or `L1` in a transform basis) so the deltas the codec
  then sees are genuinely sparse — same "the tail is free real estate"
  argument Sec. 14 uses for the STE pass.
- `DECISIONS.md` Sec. 16 — factors stay bounded on `[0,1]`-scaled data, so the
  delta range is bounded too; no calibration.

**Composes with:** Sec. 14 (quantise deltas not factors), Sec. 13 (the tracking
warm-start *produces* consecutive factors already in register), Sec. 18 —
**and this is why Sec. 18 matters beyond conditioning:** the gauge freedom
(Sec. 1) means `W1[:,:,t]` and `W1[:,:,t−1]` are only close if they are in a
*consistent gauge*. A free per-slice rotation makes the deltas full-range and
the entropy coder useless. Delta coding **requires** either the Sec. 18
by-construction gauge or a per-slice Procrustes alignment to the predecessor.

**Open question to argue about.** Does the compressibility penalty (TV along
`p`) cost more than the ~1% reconstruction budget, and is the crossover
data-dependent (fine on Carphone, bad on a high-motion clip)? Measure the
rate–distortion curve — bits-per-entry vs Frobenius error — not a single point,
and compare against just storing the fp16 factors.

**Failure mode.** Two. (i) Per-slice Procrustes alignment is another
180-op loop — the Sec. 4.1 hazard; batch it. (ii) Entropy coding is serial and
CPU-bound; if it lands inside the timed region it wrecks the wall-clock the
rest of this document is trying to cut. It must be an offline post-process, and
the paper must then separate "decomposition time" from "encode time" and report
both honestly.

---

## 20. Sum of Hadamard products — boosting instead of one big product (gradient boosting, CP decomposition)

Sec. 3 generalises `(W1 H1) ∘ (W2 H2)` by adding *more factors to the product*:
`∘_i (W_i H_i)`, rank `~r^k`. The orthogonal generalisation — not in the paper,
not in Sec. 3 — is a **sum of products**:

    X ≈ Σ_{j=1..s} (A_j B_j) ∘ (C_j D_j)

Each term has rank `≤ r²`, so `s` terms reach rank `≤ s r²`, with the terms
fitted **one at a time on the running residual** — gradient boosting, or a CP
decomposition whose every rank-1 term has been replaced by a Hadamard product.

**Why it should help, tied to what is measured:**

- `DECISIONS.md` Sec. 3 / Sec. 6 — per-epoch cost is `O(k r mnp)`, *linear in
  r*. A sum of `s` terms at inner rank `r` costs the same per epoch as one term
  at rank `sr`, but the fit **decouples into `s` small subproblems**. Each
  small-`r` subproblem has less of the Sec. 1 scaling symmetry to fight, so it
  should clear the Sec. 11 plateau faster — you trade one hard 2000-epoch
  descent for `s` easy ones.
- `DECISIONS.md` Sec. 11 — boosting never runs a joint descent over the whole
  parameter block; it adds capacity only where the residual still has
  structure, which is precisely the "stop buying epochs that don't pay"
  finding.
- `DECISIONS.md` Sec. 16 — the injected `0.01·randn` noise means the target is
  itself ~18% out of range; a boosting stop rule (add terms until the residual
  norm hits the Sec. 9.4 noise floor) refuses to fit that noise, where a fixed
  large `r` single model will.

**Composes with:** Sec. 11 (a ladder in `s`, not just `r`), Sec. 3 (each
boosting term can itself be a `k`-fold product), Sec. 9.4 (the discrepancy
principle is the natural "how many terms" rule), Sec. 17 (the `γ=0` additive
surrogate is the `r=1`, `s→large` corner of this same family).

**Open question to argue about.** Does `rank ≤ r²` per term hold with near-
equality *generically*, or do successive boosting terms span overlapping
subspaces so that `s` terms buy far less than `s r²` effective rank? Measure the
effective rank (singular-value energy at 99%) of the fitted sum against `s r²`
on `random-tensor`, where ground-truth rank is controlled. If it collapses,
boosting is redundant with just raising `r`.

**Failure mode.** Greedy stagewise fitting is not jointly optimal — later terms
cannot fix a bad early term. Standard boosting fixes this with a shrinkage
factor (fit `ν · residual`, `ν < 1`) and optional backfitting sweeps that
revisit earlier terms; both add passes. Also: `s` separate SVD inits (Sec. 4)
per fit — cheap individually (Sec. 3: init is ~2%), but `s×` that is worth
watching.

---

## 21. Approximate message passing + state evolution (statistical physics of inference)

Every algorithm above is an *optimiser* — descend a loss, worry about its
geometry. AMP is a different lineage: a fixed-point iteration derived from
belief propagation on the dense factor graph, with an **exact asymptotic
analysis** (state evolution) that predicts its error at every iteration before
you run it.

**The construct.** For low-rank matrix estimation from noisy observations,
AMP (Rangan–Fletcher; Montanari–Venkataramanan; the "Low-RAMP" family)
alternates a linear step with a per-entry **denoiser**, plus an Onsager
correction term borrowed from spin-glass physics that decorrelates the
iterates. It converges in `O(10)` iterations, and state evolution gives the
exact MSE-vs-noise curve and the **weak-recovery phase transition** — the noise
level past which *no* estimator recovers the signal.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 11 — the 87%-wasted plateau is an optimiser pathology.
  AMP does `O(10)` iterations, not thousands; if the plateau is really the
  Sec. 1 geometry, AMP's updates sidestep it because they are not gradient
  steps on that landscape.
- `DECISIONS.md` Sec. 16 — the data has `0.01·randn` added, *exactly* the
  known-variance Gaussian-noise setting AMP assumes. State evolution would say
  in advance whether rank-10 Carphone is even recoverable at that noise, and
  what Frobenius error is information-theoretically optimal — turning the
  paper's "47.34" into "X vs the optimal Y".
- Sec. 7.1 (open theory question — sample complexity of the Hadamard model):
  state evolution *is* the instrument that answers it. It gives the exact
  threshold for `(W1 H1) ∘ (W2 H2)` recovery vs an additive model of matched
  rank, which "nothing in the literature answers" today.

**Composes with:** Sec. 9.8 (AMP is also a leading phase-retrieval solver — same
`X = M ∘ M`-shaped observation), Sec. 1 (an alternative to the whole
optimisation framing, not a patch on it), Sec. 7 (this is the "goal is a paper"
route).

**Open question to argue about.** AMP's guarantees hold for signals drawn from
an assumed prior observed through a known channel. The Hadamard *product*
observation `M1 ∘ M2` is **not** the standard low-rank-plus-Gaussian model —
deriving the correct denoiser and the state-evolution recursion for the
elementwise-product likelihood is genuine work, not a library call. Same shape
of caveat as Sec. 2's "leverage scores would need rederiving".

**Failure mode.** AMP diverges under model misspecification and when the
matrix is not from the assumed (rotationally invariant) ensemble — i.e. on
*real images*, whose factors are structured, not i.i.d. The robust variants
(VAMP / OAMP, damping) recover stability but cost iterations and complexity.
Testable prediction, same direction as Sec. 12 / 13 by yet another mechanism:
works on `random-tensor`, likely wobbles on Carphone.

---

## 22. Parallel-in-time / Parareal on the epoch loop (numerical ODE integration)

Sec. 8.1 (Anderson) and Sec. 12 (multigrid) accelerate the epoch loop by
*extrapolation* and by *coarsening space*. Parareal coarsens **time** — and
turns a sequential loop into a wall-clock-parallel one.

**The construct.** The 2000-epoch descent is a discretised gradient flow, an
ODE in "epoch time". Parareal (Lions–Maday–Turinici 2001; standard in climate
and plasma simulation) splits the trajectory into `P` intervals and iterates:

    a cheap COARSE propagator G predicts the endpoints of all P intervals in sequence;
    an expensive FINE propagator F refines all P intervals IN PARALLEL;
    a correction sweep reconciles them: U_{n+1} = G(U_n) + F(U_n^old) − G(U_n^old)

It converges to the exact sequential trajectory in `J ≪ P` sweeps, so the
sequential cost drops from `P` fine steps to `J` fine intervals.

**Mapping.** `G` = a few epochs of a cheap surrogate — coarse grid (Sec. 12),
additive model (Sec. 17), or tiny-`r`; `F` = real Adam epochs on a segment;
run the `P` segments as a batched `vmap` over trajectory replicas.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 3 — runtime *is* the epoch loop, and it is strictly
  sequential today. This is the only idea in the document that makes epochs
  run concurrently rather than just reducing their count.
- `DECISIONS.md` Sec. 11 — Parareal converges fastest when `F` and `G`
  trajectories stay close, i.e. when the flow is slow and smooth. That is
  exactly the 87%-wasted tail; the plateau is the *best case* for parallel-in-
  time, not an obstacle.
- `GPU.md` Sec. 6.5 — "memory is not a constraint … Carphone tensor ~15 MB …
  6 GB is ample". Dozens of concurrent trajectory replicas fit. This is the
  enabling fact; without spare memory the idea is dead.

**Composes with:** Sec. 12 (multigrid *is* the coarse propagator `G`),
Sec. 17 (additive surrogate as `G`), Sec. 8.1 (Anderson on the correction
sweep), Sec. 11 (coarse `G` = small-`r` rung).

**Open question to argue about.** Parareal is proven for *dissipative*
(parabolic, contractive) flows. A non-convex descent can be chaotic away from
a minimum — a small coarse-prediction error lands in a different basin and the
`F`/`G`-stay-close premise breaks. But near the Sec. 11 plateau the flow is
nearly stationary, hence contractive: the method may work *only* in the
wasteful regime — which is fine, since that is the regime worth attacking.
Probe: measure ‖F(U) − G(U)‖ along a real run; if it stays small only after
epoch ~250, apply Parareal only past there.

**Failure mode.** Speedup is capped at `P/J`, and each sweep still costs one
sequential fine interval — if `J` creeps toward `P` it degenerates to the
sequential loop with overhead. Also `F` must be batched over segments
(another lever-1 / Sec. 4.1 batching exercise) or the parallelism is notional.

---

## 23. Sign pattern as a discrete latent (MRF inference, graph cuts, EM)

`hadamard_ops.py` lever 2 already splits the init into `sqrt(|X|)` and
`sqrt(|X|)·sign(X)` — an implicit admission that **sign and magnitude want
separate treatment.** After init, the whole sign structure is handed back to
gradient descent, which can only flip a sign by pushing a factor entry through
zero. Make the split explicit and carry it through the fit.

**The construct.** Treat the reconstruction sign `S ∈ {±1}^{m×n×p}` as a
discrete latent and the magnitudes as continuous, then alternate — EM with a
discrete E-step, exactly as in MRF-based segmentation (graph cuts, α-expansion)
and Ising ground-state search:

1. **Magnitude step (continuous):** given `S`, fit `W,H` to `S ∘ |X|`, a
   smooth low-rank Hadamard problem with *no sign barriers*.
2. **Sign step (discrete):** given the magnitudes, choose `S` per entry. The
   unconstrained optimum is `sign(X)`; the rank constraint couples signs along
   rows/columns, which is where the MRF / MAX-CUT machinery (α-expansion,
   Goemans–Williamson SDP relaxation) applies.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 11 — the repo minimises the *unsquared* norm whose
  gradient `−R/‖R‖` has curvature diverging as `‖R‖→0`. Sign flips happen right
  where the residual is small; pulling the discrete part out of the smooth
  subproblem removes the worst-conditioned moves from the optimiser's job.
- `DECISIONS.md` Sec. 16 — clean MRI is **33% exactly zero**, and after the
  `0.01·randn` injection **17.79% of the target is negative** noise around
  zero. Sign is meaningless for those entries; a discrete latent with a
  "don't-care" state for `|X| < noise` stops the smooth model from spending
  capacity fitting the sign of pure noise — the exact pathology Sec. 16
  documents ("model tracks the target" including its out-of-range noise).
- `hadamard_ops.py` — the `mat_simple` / `mat_signed` pair is this idea at
  `t=0` only; this extends it to every outer iteration.

**Composes with:** Sec. 9.1 (log-domain needs `|X|`; the sign latent is its
natural partner — `log|X|` for magnitude, discrete `S` for sign, cleanly
separated), Sec. 17 (magnitude-only fit is a benign start), Sec. 20 (each
boosting term gets its own sign map).

**Open question to argue about.** Is the sign pattern of a rank-`r²` Hadamard
model *low-complexity* — few sign changes per row, i.e. low total variation?
For natural images, yes (sign of a smooth function flips rarely), and the
discrete step is cheap. For `random-tensor`, no — sign is essentially random,
the discrete problem is NP-hard with no structure to exploit. Same image-vs-
random split as Sec. 12 / 13 / 21, by yet another mechanism; measure sign-map
TV on Carphone vs random before building anything.

**Failure mode.** Exact graph cuts need submodular energies; the rank coupling
across a row is not obviously submodular, so the discrete step falls back to an
SDP relaxation at `O((mn)^1.5)` per slice — too slow at Carphone size. Likely
only practical as a *local move* (re-solve signs of near-zero entries every few
epochs), not a global discrete optimisation.

---

## 24. Gap — the compression claim is never tested against a real codec

Sec. 5 calls compression "the paper's actual application", yet every baseline
in `DECISIONS.md` Sec. 3's table (TSVD, TRRQR, TBCD, TAGD) is another *low-rank
tensor decomposition*. None is the technology this would actually compete with.
Three distinct holes, each tied to something measured:

**1. No codec baseline.** For images the reference is JPEG / JPEG2000; for video
it is x264 / x265 / AV1. The claim needs a **rate–distortion curve** —
bits-per-pixel vs PSNR — with those on it, not a single operating point against
tensor methods. Likely outcome: HadNet loses heavily to x265 on Carphone, and
that is still worth stating plainly (`DECISIONS.md` Sec. 4.4 already sets the
precedent of reporting an unflattering like-for-like number). The regime where a
Hadamard decomposition *could* win is probably not consumer video at all —
scientific / hyperspectral / simulation tensors with no mature codec — and the
paper should say so rather than imply general-purpose compression.

**2. PSNR is measured against a corrupted reference.** `DECISIONS.md` Sec. 16:
the scripts add `0.01·randn` (main_image.py:267, main_video.py:294), which puts
**17.79% of the MRI target outside `[0,1]`**, and the metric clamps. So every
reported PSNR/SSIM is distance-to-*noisy*-image. Rankings within Table 9 survive
(all methods scored the same way), but the absolute numbers are not comparable
to any externally reported codec figure — codecs are conventionally scored
against the clean original. Put a codec in the table and the noise injection
must be turned off, or applied to both.

**3. Compression-ratio accounting favours HadNet.** The reported ratio is
`entries / parameters` at fp32. Sec. 14 / 19 show the factors should be
quantised and entropy-coded to count as a real "compressed size"; a codec's
bitstream already *includes* its entropy coding. Comparing an fp32 parameter
count against a real bitstream is apples-to-oranges.

**Fix.** One rate–distortion plot per medium: HadNet (factors quantised +
coded per Sec. 14 / 19) vs JPEG / x265, PSNR against the *clean* reference,
noise off or shared. If HadNet is not meant to beat codecs, state the intended
niche and benchmark *there* instead.

---

## 25. ADMM / operator splitting — decouple the two factors with a dual variable

Sec. 1 (Riemannian), Sec. 15 (Schur-LM), Sec. 21 (AMP) each take the
multiplicative coupling `M1 ∘ M2` head-on. ADMM **splits it away**: introduce a
consensus variable and let a dual variable enforce agreement, so every
subproblem is one you already know how to solve.

**The construct.** Write

    min_{M1,M2,Z} ‖X − Z‖_F²   s.t.   Z = M1 ∘ M2,  rank(M1) ≤ r,  rank(M2) ≤ r

and run ADMM on the augmented Lagrangian (dual `U`, penalty `ρ`):

- **Z-step:** `Z = (X + ρ(M1∘M2 + U)) / (1 + ρ)` — closed form, elementwise.
- **M1-step:** `min ‖M2 ∘ (··· − M1)‖²` s.t. `rank ≤ r` — a *weighted*
  rank-`r` approximation, weights `M2²`; solved by EM-for-weighted-LRA or a
  few power iterations.
- **M2-step:** symmetric.
- **Dual:** `U += M1∘M2 − Z`.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 11 — block-convexity is exactly what makes each `M_i`
  step a tractable weighted-LRA. The Lipschitz experiment failed because "a
  joint Adam step ignores the geometry"; ADMM's dual variable is the thing that
  couples the two block solves *consistently* instead of a joint step.
- `DECISIONS.md` Sec. 9 — the `lr=0.1` "magic constant". ADMM's only knob is
  `ρ`, and residual-balancing (Boyd et al. §3.4.1) adapts it automatically —
  a genuinely tuning-free loop.
- `DECISIONS.md` Sec. 3 — each `M_i` step is `O(mnr)` per slice, one epoch's
  worth; ADMM on this problem class converges in *tens* of outer iterations,
  not thousands. Directly attacks `k`.
- `DECISIONS.md` Sec. 14 / 16 — the Z-step assumes the *squared* Frobenius
  objective, which `real-world/` already uses and Sec. 11 flagged as the
  consistent choice.

**Composes with:** Sec. 9.1 (log-domain turns `Z = M1∘M2` into `Z = A + B`,
the textbook *linear*-consensus ADMM — cleanest form), Sec. 2 / `DECISIONS.md`
Sec. 2 (the Z-step is elementwise, so a mask `Ω` for the completion setting
drops in for free), Sec. 11 / 17 (warm-start each weighted-LRA along the
ladder).

**Open question to argue about.** Weighted low-rank approximation with entrywise
weights is NP-hard in general (Srebro–Jaakkola) and has its own local minima.
Does the ADMM outer loop keep each `M_i` subproblem *easy* by warm-starting it
near the last solution, or does the weight `M2²` swing enough between outer
iterations that each subproblem is a fresh hard problem? Measure how far the
`M_i` iterate moves per outer step.

**Failure mode.** ADMM on a non-convex constraint has no convergence guarantee
— only empirical behaviour. `ρ` too small drifts, too large freezes each
subproblem. Inexact subproblem solves (loose tolerance early, tighten later)
are the standard speed trick but reintroduce a tolerance schedule to tune.

---

## 26. HadNet -> HadDNN distillation, and: does HadDNN earn its cost? (teacher-student)

`DECISIONS.md` Sec. 12 measured both models on `random-tensor`:

| | time | rel. error |
|---|---|---|
| HadNet  | **2.03 s**  | **0.364970** |
| HadDNN  | 16.73 s     | 0.376999 |

The "neural architecture" is **8x slower and worse** here. Two things follow.

**The gap.** When, if ever, does HadDNN beat HadNet? The paper pairs them as
decomposition vs neural version, but on the one head-to-head we have measured
the plain decomposition wins. Before optimising HadDNN at all, run it against
HadNet on the *image / video* data, where a nonlinearity might buy something
that it cannot on a random tensor. If HadDNN never wins outside contrived
cases, the honest recommendation is "use HadNet, fall back to HadDNN only when
it demonstrably helps" — and that reframes every HadDNN speedup below as
optional.

**The optimisation — distillation.** HadDNN trains its factor-nets from
scratch. Instead: fit HadNet first (~2 s), then initialise each factor-net by
regressing it to reproduce the converged HadNet factor (a small supervised fit
per net; closed form for a linear output layer), then fine-tune. HadDNN's job
becomes *improve on a good linear solution*, not *find structure in noise*.

**Tied to what is measured:**

- `DECISIONS.md` Sec. 12 — early stopping made HadDNN *better* (-0.08%),
  "consistent with early stopping acting as regularisation on the
  over-parameterised factor nets". An over-parameterised model that mostly
  needs regularising is exactly the case where a strong prior (the linear
  solution) beats random init.
- `DECISIONS.md` Sec. 12 — the 8x gap (16.73 vs 2.03 s) is the factor-nets'
  own training. Warm-starting from converged HadNet factors should collapse
  most of it; the HadNet fit is now cheap enough (~2 s) to be a ~12% tax on a
  16 s budget.
- `DECISIONS.md` Sec. 3 — same "init is a small fraction, spend it well"
  accounting as lever 2.

**Composes with:** Sec. 11 (HadNet's rank ladder feeds HadDNN a ladder of
inits), Sec. 8.4 (a coordinate-net factor *is* a factor-net — distillation
initialises Sec. 8.4's MLP from the linear fit), Sec. 18 (distil into
gauge-fixed factors so the student starts in a clean gauge).

**Open question to argue about.** Does HadDNN's (rare) advantage come from the
nonlinearity reaching a basin the linear model cannot, or just from
over-parameterised optimisation dynamics? If the former, a linear init could
*trap* it near the linear solution and suppress the very thing that makes it
worth running. Check: distilled vs cold HadDNN final error on any dataset where
cold HadDNN already beats HadNet.

**Failure mode.** If HadDNN never reliably beats HadNet (the gap above),
distillation is polishing a component that should not be on the critical path.
Resolve the gap question first; the distillation is only worth building if the
answer is "HadDNN wins on real media".

---

## Suggested order (revised — supersedes the list below)

1. **GPU end-to-end baseline** — nothing can be claimed without it (Sec. 10, row 1).
2. **Anderson acceleration (Sec. 8.1 / 9.2)** — cheapest attack on `k`, no maths
   risk; jumps ahead of CUDA graphs.
3. **Balancedness / Sinkhorn gauge-fix (Sec. 8.2 / 9.7)** — test before the full
   Riemannian rewrite; may make Sec. 1 unnecessary.
4. **Log-domain reparameterisation (Sec. 9.1)** — highest-leverage translation;
   if it works it subsumes Sec. 1 and the `DECISIONS.md` Sec. 11 plateau.
5. **Discrepancy-principle stopping (Sec. 9.4)** — principled `k`, ties off
   `DECISIONS.md` Sec. 16.
6. Then the original order: Riemannian (Sec. 1, as K-FAC per Sec. 8.3),
   factor sharing (Sec. 5, as coordinate-net per Sec. 8.4), k>2 factors
   (Sec. 3), leverage sampling (Sec. 2).

---

## Suggested order (original)

1. **Riemannian (Sec. 1)** — highest expected value; attacks `k`, which is now
   the dominant cost, and explains an observation we already have.
2. **Factor sharing (Sec. 5)** — biggest structural win, unblocks Sec. 2.
3. **CUDA graphs (Sec. 6)** — cheap, mechanical, no maths risk.
4. **k>2 factors (Sec. 3)** — cheap experiment against an existing codebase.
5. **Leverage sampling (Sec. 2)** — only after Sec. 5 raises the redundancy ratio.
