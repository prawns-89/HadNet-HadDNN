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

# Synthesis — what stacks, what fights, per-line coverage

Sections 27+ stop proposing new ideas and start integrating the ones above:
which compose, which contradict, and which line of attack is actually
under-served. Session-measured numbers referenced below (RTX 4050, Carphone
shape): **bf16 autocast 1.63x end-to-end, error unchanged**; **backward = 75%
of an epoch**; **torch.compile 0.97x** (laptop GPU, too few SMs); **early
stopping already cuts to ~300 epochs**, not 2000; **HadamardDNN 134.8M params,
~half provably inert**.

## 27. Composability map, contradiction list, and the `k` line in full

### (a) What stacks — and why it multiplies rather than adds

Four independent layers; wins within a layer compete, wins across layers
multiply.

| Layer | Members | Compose? |
|---|---|---|
| **Systems / precision** | `h_prod` batch (§4/lever 1, 9–41x), early stop (§12, ~3x), bf16 (session, 1.63x), TF32 (session, 1.11x, subsumed by bf16) | **Multiply** — orthogonal mechanisms (launch overhead / iteration count / arithmetic width). `torch.compile` (§6) does **not** join on this GPU. |
| **Convergence (attack `k`)** | Anderson §8.1, Parareal §22, homotopy §17, rank-ladder §11, multigrid §12, streaming §13, SWA/restarts (below) | Compose only as a **schedule with diminishing returns** — each assumes the previous has not already flattened the tail. Post-early-stop that tail is ~300 epochs, not the §11 "87% of 2000". |
| **Geometry (attack conditioning)** | Stiefel-`W` §18 **+** balancedness/Sinkhorn §8.2/§9.7 (kill GL, then diagonal — these two *do* stack); K-FAC §8.3 / Riemannian §1 / natural grad §9.5 are **alternatives**, pick one | Split: one stacking pair, the rest mutually exclusive. |
| **Model structure** | factor-sharing along `p` §5/§8.4, boosting sum §20, `k`-factor product §3 — all compose (share within each of `s` terms of a `k`-product); quantise+code §14/§19 is a post-process on whatever results | **Compose**, and the parameter counts multiply down. |

The headline stack with everything measured or low-risk:
`bf16 (1.63x) x early stop (3x) x h_prod batch (already in)` — none touches the
maths, all confirmed, ~5x over the pre-optimisation baseline before any
research-grade change.

### (b) What contradicts — pairs that cannot both be run naively

1. **Randomized/block-Krylov init (§4/§16) vs discrepancy-principle stopping
   (§9.4).** The init is deliberately approximate (delta 1.7e-3, `DECISIONS.md`
   §13); the stop rule compares the residual to the noise floor
   `0.01·sqrt(N)`. If init error ≳ noise floor the rule cannot tell "converged"
   from "bad init". Fix: tighten `niter` when discrepancy stopping is on —
   they trade against each other.
2. **bf16 (session) vs any absolute-threshold stop (§9.4).** bf16's ~1e-2
   relative rounding on the residual sits above the `0.01` noise floor in the
   last decimals; an absolute stop reads bf16 rounding as irreducible residual.
   bf16 is safe with *relative*-improvement early stopping (§12), not with a
   noise-floor stop. Contradiction is with §9.4 specifically, not §12.
3. **Stiefel-`W` §18 vs Riemannian §1 vs balancedness §8.2** all target the
   same symmetry group. Two at once double-counts and can over-constrain
   (§18's open question). One primary + optional §8.2 diagonal cleanup.
4. **Log-domain §9.1 + quantise §14.** Quantisation error in log-space is
   *multiplicative* in linear space; §14 already found factor rounding
   compounds multiplicatively through the `∘`. Stacking them is
   multiplicative-on-multiplicative — the bit-allocation derivation §14 asks
   for becomes mandatory, not optional.
5. **Streaming §13 vs factor-sharing along `p` §5.** Streaming *produces*
   warm-started per-slice factors; sharing wants *one* basis across `p`. The
   pure forms are opposites; §13's note 3 (shared base + per-slice correction)
   is the only reconciliation and it is not free.
6. **Data-dependence is opposite for two families.** AMP §21 and GNC §17
   "work on `random-tensor`, wobble on Carphone"; multigrid §12 and streaming
   §13 "gain on Carphone, nothing on `random-tensor`". A portfolio should
   **route by data type**, not stack these blindly.
7. **Coordinate-net factors §8.4 vs the HadDNN finding §26.** §26 measured that
   `MLP(learned_constant_vector)` is 130x bloat with ~half the params inert.
   §8.4 only escapes that trap if its MLP input is a genuine varying coordinate
   — a design constraint §8.4 must state explicitly or it *is* §26.

### (c) The `k` line of approach — full enumeration

Everything aimed at epoch count, in the doc and not:

- **In the doc:** Anderson §8.1, Parareal §22, homotopy §17, rank-ladder §11,
  multigrid §12, streaming §13, discrepancy stop §9.4, K-FAC §8.3, Riemannian
  §1, GN-Schur §15, AMP §21, ADMM §25, boosting §20 (`s` easy fits vs one hard).
- **Not yet in the doc:**
  - **Cyclic LR with warm restarts (SGDR).** The code's schedule only *decays*
    LR ×0.1 on plateau and never restarts (`DECISIONS.md` §4.2). A restart is
    the standard escape from the §11 plateau; one extra schedule parameter.
  - **Stochastic weight averaging over the wasted tail.** Instead of stopping
    at ~300 epochs, average epochs ~200–300. `DECISIONS.md` §11 says the tail
    buys <1%; SWA *extracts* that 1% as variance reduction instead of
    discarding it. Near-zero cost, composes with early stopping.
  - **Gradient-noise-scale measurement.** The loss is full-batch (every `mnp`
    entry each epoch). If the gradient-noise scale is low there is no reason
    not to take larger steps — measure it before assuming `lr=0.1` is near the
    ceiling.
  - **Condition-number-matched momentum (Polyak heavy ball).** One scalar,
    cheaper than Anderson, optimal for the locally-quadratic tail.

### (d) Gap + mitigation

**Gap.** The "87% of epochs wasted" motivation (`DECISIONS.md` §11) was
measured at fixed `k=2000`, *before* early stopping was wired in. Session
measurement: the fit now runs **~300 epochs**. Every `k`-attacking section is
being justified by a stale number — the live question is how much of *those
300* epochs is wasted, and it may be very little.

**Mitigation.** Re-run the §11 convergence-curve table with `EarlyStop` active,
at Carphone and MRI shapes, and recompute the compute-vs-accuracy rows. If
~300 epochs is already near the knee, the entire convergence layer (§8.1, §11,
§12, §13, §17, §22) has little left to attack and priority moves wholly to
per-epoch cost — bf16, factor-sharing (§5), lower rank — and to init cost,
which is a larger fraction of a 0.7 s run than of a 57 s one.

**Open question to argue about.** Is `k` still the dominant lever *after* early
stopping + bf16? At ~300 epochs × ~2.3 ms (bf16) ≈ 0.7 s, the randomized-SVD
init and per-epoch Python/autograd overhead may now dominate. The profiling
target is the 0.7 s run, not the paper's 57 s one — and no section yet is
written against that regime.

**Failure mode.** Stacking every `k`-attack (Anderson + restart + homotopy +
ladder + SWA) adds interacting hyperparameters — Anderson depth, restart
period, `γ` schedule, rank schedule, averaging window. Tuning them can cost
more researcher-time than the epochs saved, and each adds its own failure mode.
A portfolio of two (early stopping + one accelerator) is likely the knee of
*that* curve too.

---

## 28. The `p`-scaling line — sharing, and why it may buy compression without speed

`DECISIONS.md` §2: Carphone is **1.04M parameters for 3.73M entries, ratio
3.6x**, because all four factors carry the full `p` dimension. This section
integrates every approach aimed at that.

### (a) What stacks

The `p`-scaling attacks form a **pipeline that multiplies the parameter count
down**, each stage operating on the output of the last:

1. **Share** (§5): replace `W1 ∈ R^{m×r×p}` with `Basis[mr, s] · Coeff[s, p]`
   — parameters `mrp -> s(mr + p)`.
2. **Constrain the coefficients** (§9.11 DCT idea, applied along `p`): force
   `Coeff` to a few low-frequency modes — `s·p -> s·f`, `f << p`.
3. **Delta + entropy-code** (§19): store `Coeff` as differences along `p` and
   arithmetic-code them.
4. **Quantise** (§14): the whole thing at `b` bits.

Stages 1–2 are model structure (change what is trained); 3–4 are storage
(post-process). They compose because each shrinks a different dimension of the
same object. `k`-factor (§3) and boosting (§20) are **orthogonal** — they act
on the `(m,n)` rank, not on `p`, so they neither help nor hurt this pipeline.

### (b) What contradicts

1. **Sharing (§5) worsens the §1 conditioning it is separate from.** A shared
   basis is used by all `p` slices, so its gradient is the *sum over `p`* of
   per-slice gradients — larger dynamic range, so the cross-Hadamard scaling
   symmetry (§1) gets **more** ill-conditioned, not less. §5 (structure) and
   §8.2/§18 (geometry) stop being independent: sharing may *require* the
   gauge-fixing, where an unshared factor tolerated skipping it.
2. **Sharing (§5) vs quantising (§14).** Each shared basis entry is used `p`
   times, so a quantisation error there is amplified `p`-fold in the
   reconstruction, against once for an unshared factor. The two compression
   levers fight: sharing makes the factors far more quantisation-sensitive,
   so the §14 bit budget must rise exactly where §5 cut the parameter count.
3. **Sharing (§5) vs streaming (§13)** — noted in §27(b), sharpened here:
   sharing assumes `p`-*stationarity*, streaming assumes `p`-*drift*. Real
   video is neither; it is *slowly varying*, whose correct model is "shared
   basis + smooth per-slice coefficients" — i.e. stage 1+2 of the pipeline
   above, which is what reconciles them.
4. **Rank ladder (§11) vs sharing.** `s` (shared-basis width) is a second
   rank-like knob beside `r`. Growing both is a 2-D schedule with coupling —
   raising `r` changes the `mr` axis the basis spans, so a basis fitted at
   `r=10` does not transfer up the ladder unchanged.

### (c) Further approaches on the `p` line

- **In the doc:** tie/factorise/shared-basis §5, coordinate-MLP §8.4 (only if
  the input is a real coordinate, per §27(b)7), streaming §13, delta+code §19.
- **Not yet in the doc:**
  - **Tensor-train / tensor-ring of the factor tensor itself.** Treat
    `W1 ∈ R^{m×r×p}` as a 3-way tensor and TT-decompose along `(m, r, p)`:
    parameters `mrp -> (m + r + p)·R_tt`. This is the "factorise along `p`"
    option §5 lists but never develops; Kolda & Bader (already cited in §5) is
    the reference.
  - **Shot-clustered factor sets.** k-means the `p` slices (on frame content),
    one factor set per cluster, a cluster id per slice — between pure sharing
    (1 group) and per-slice (`p` groups). Directly addresses `DECISIONS.md`
    §2's "consecutive frames highly correlated" and §13's scene-cut failure
    mode.
  - **One unfolding SVD for the init.** A shared-basis init needs *one* SVD of
    the mode-1 unfolding (`m × np`), not `p` per-slice SVDs. Cheaper init
    (`DECISIONS.md` §3, §6's `n^5` term) that *also* directly yields the shared
    basis — init and structure solved together.
  - **Low-rank + sparse split along `p`** (robust-PCA style):
    `W1[:,:,t] = L[:,:,t] + S[:,:,t]`, `L` shared low-rank, `S` sparse
    per-slice corrections for the few slices that genuinely differ. Handles
    §13's error-accumulation-at-cuts structurally rather than with keyframes.

### (d) Gap + mitigation

**Gap.** The `p`-redundancy is *asserted* (`DECISIONS.md` §2, §5) but never
*measured*. Nobody has computed the numerical rank of the mode-3 (`p`)
unfolding of Carphone. If 180 slices are effectively rank ~15, a shared basis
with `s ≈ 15` captures everything and the ~6x parameter cut is real; if it is
effectively rank ~120, sharing barely helps and only clustering or streaming
can.

**Mitigation.** One plot: SVD the `p`-mode unfolding (`144·144 × 180`) of the
Carphone tensor, look at where the singular values fall off. Ten minutes,
decides which of the §5 sub-options is worth building and rules out the others.
Repeat for MRI/`random-tensor` to confirm the image-vs-random split predicted
in §12/§13/§21 shows up here too.

**Open question to argue about.** Does `p`-sharing help **wall-clock** at all,
or only the **compression ratio**? Per-epoch cost is `O(k·r·mnp)` no matter how
`W` is parameterised — the residual and its gradient still touch every `mnp`
entry, and §27 measured that backward is **75% of the epoch**. Sharing shrinks
the *parameter* gradient (fewer numbers for Adam) but not the *residual*
computation. So §5 may deliver its stated goal (compression — `DECISIONS.md`
§5) with near-zero speed gain, which demotes it on a speed-priority ordering
and promotes it on a ratio-priority one. The two goals want different section
orders.

**Failure mode.** A shared basis is a bottleneck: if two slices genuinely need
different `r`-dim subspaces (fast motion, a cut), a width-`s` shared basis
represents neither well and the error floor rises for **all** `p` slices, not
just the hard ones — cross-slice contamination. The low-rank+sparse split
mitigates it but adds the sparse-part solve to every epoch.

---

## 29. The conditioning-and-symmetry line — and whether Adam already caps its ceiling

The cross-Hadamard scaling symmetry (§1) and the plateau it is blamed for
(`DECISIONS.md` §11: 87% of epochs, <1% gain) motivate §1, §8.2, §8.3, §9.1,
§9.3, §18. This section asks whether that layer is real leverage or a story.

### (a) What stacks

- **§18 (Stiefel-`W`) + §8.2/§9.7 (Sinkhorn balancing) is the only truly
  composing pair.** The redundancy group is roughly `GL(r) x Diag(m) x
  Diag(n)`; Stiefel-`W` quotients `GL(r)` down to `O(r)`, balancing fixes the
  diagonal part, and what is left (`O(r)`, measure-zero) Adam handles. They
  attack **disjoint** subgroups, so the gain is the product.
- **§8.3 (K-FAC) sits on top of a gauge-fixed parameterisation**, not beside
  it: §18/§8.2 flatten the redundant directions, K-FAC scales the rest. Adding
  K-FAC *before* the gauge-fix wastes curvature estimation on directions that
  are about to be removed.
- **§9.1 (log-domain) does not stack with any of them — it replaces the
  layer.** If the symmetry becomes an additive rank-1 shift that a single term
  absorbs (§9.1), then §1/§8.2/§8.3/§18 have nothing left to do. It is a
  one-of, not a component.

### (b) What contradicts

1. **§1 vs §18 vs §8.3 are three price points for one job.** §1 costs
   `O((m+n)r^4)` per slice (§1's own open question: `10^4`/slice at `r=10`);
   §18 is ~free (one batched QR); §8.3 is `r x r` solves. Running more than one
   is not just redundant (§27(b)3) — it is paying twice at very different
   rates. Escalation order: §18, then §8.3, then §1 only if both visibly fail.
2. **§9.1 (log-domain) breaks the noise model the repo assumes.**
   `DECISIONS.md` §16's additive `0.01·randn` becomes *heteroscedastic* in log
   space — small `|X|` entries get enormous log-noise. Log-domain + the
   existing **unweighted** Frobenius loss is inconsistent; it needs a weight
   `∝ |X|²` to undo the transform's distortion. And §9.4's discrepancy stop,
   whose threshold `0.01·sqrt(N)` is a linear-space scalar, is simply wrong in
   log space.
3. **The Lipschitz rejection (`DECISIONS.md` §11) is evidence against the whole
   layer, not just the `1/L` step.** §11 concluded geometry-aware stepping was
   "worse on both axes" partly because "Adam already adapts per-parameter from
   gradient second moments, so dividing by `L` duplicates that and over-damps".
   K-FAC (§8.3) and Riemannian (§1) also layer curvature onto/against Adam — if
   that reasoning holds, they inherit the objection.

### (c) Further approaches on this line

- **In the doc:** §1, §8.2, §8.3, §9.1, §9.3, §18.
- **Not yet in the doc:**
  - **Spectral normalisation of the factors** (Miyato et al., GANs) — divide
    each factor by its spectral norm each step; one power iteration per factor,
    cheaper than a full Sinkhorn, fixes the scale ambiguity.
  - **Fixed diagonal preconditioner from the init's singular values.** The SVD
    init (§4) already returns `S`; use `S^{-1/2}` as a constant gradient
    preconditioner for the whole run. Zero extra cost, targets exactly the
    anisotropy the symmetry induces.
  - **L2 penalty as soft balancing.** `λ(‖W‖² + ‖H‖²)` has a unique minimiser
    on each scaling orbit — the balanced point. It does §8.2's job through the
    loss, no projection step, one line. The soft counterpart of §8.2 / §18.
  - **Name the symmetry coordinate.** Write `W = e^{a}·Ŵ`, `H = e^{-a}·Ĥ` with
    `Ŵ,Ĥ` unit-norm and `a` a learned per-column scalar. The symmetry
    direction becomes one explicit coordinate Adam moves freely — removed by
    naming, not quotienting.
  - **Normalise `M1`, `M2` before the product**, folding scale into a learned
    gain — batch/layer-norm's trick, makes factor scale irrelevant by
    construction.

### (d) Gap + mitigation

**Gap.** Nobody has *measured* that the landscape is ill-conditioned. §1
asserts it from an external paper; `DECISIONS.md` §11's plateau is *consistent
with* conditioning but equally with §12 (slow low-frequency mode), §17
(non-convex basin), or just a too-small `lr`. Four explanations, one
observation, zero measurements distinguishing them.

**Mitigation.** At epoch ~250 (plateau onset, `DECISIONS.md` §11): (i) a few
Hessian-vector products via Lanczos give the top and bottom eigenvalues —
compute `κ`. `κ ~ 10^2` means Adam copes and the layer is low-value;
`κ ~ 10^6` means the symmetry story holds and §18/§8.2 are justified. (ii)
Measure `‖grad‖` and the angle between consecutive gradients at the plateau: a
large oscillating gradient is conditioning (preconditioner helps); a tiny
gradient is a genuine near-stationary point (only a better basin helps — §17,
§11). One afternoon, and it tells you whether to build this whole layer.

**Open question to argue about.** If the Lipschitz rejection's "Adam already
adapts" reason is the true one, does using Adam *at all* cap the ceiling for
every geometry-aware method here? The test: plain SGD + one geometry method
(§8.2 or §8.3) vs Adam alone vs Adam + geometry. If SGD+geometry beats
Adam+geometry, the conditioning layer only pays once Adam is dropped — which
none of §1/§8.2/§8.3/§18 currently assume.

**Failure mode.** Every gauge-fix (§8.2, §18, spectral-norm, explicit-scale)
adds a per-step normalisation over factors that carry the `p` dim — the §4.1
"180 tiny ops in a loop" hazard, and these ops need backward too. §27 measured
backward is already **75% of the epoch**; a conditioning layer that adds
10–20% per-epoch cost for a benefit `DECISIONS.md` §11 hints is marginal can
come out **net negative** on wall-clock even if it reduces epoch count.

---

## 30. The compression-ratio line — the paper's stated goal, least-served axis

Sec. 5 calls compression "the paper's actual application"; §24 shows it is
never benchmarked as one. This section integrates every ratio-affecting
approach.

### (a) What stacks

The stored size is `entries / (params · bits_per_param · code_efficiency)`.
Three independent multipliers:

| Multiplier | Approaches | Typical factor |
|---|---|---|
| `entries / params` | §5 sharing, §11 prune ("overshoot rank then prune") | 2–6x (§28's open question — unmeasured) |
| `32 / b` bits per param | §14 quantise (→int8), bf16 storage (→16) | 2–4x |
| `b / H` code efficiency | §19 delta + entropy code | 1.5–3x |

They **multiply** — each acts on a different term. The paper reports **only the
first, at fp32** (`DECISIONS.md` §3 "params" column), so stages 2–3 are an
unclaimed ~4–12x sitting on the table. `k`-factor (§3) and boosting (§20) push
the *wrong way* on term 1 (more params for more rank) — they belong on a
rate–distortion curve, not a raw-ratio target.

### (b) What contradicts

1. **Run-to-convergence vs delta-coding (§19).** `DECISIONS.md` §16: the target
   is ~18% out-of-range noise. The last epochs fit some of that noise into
   high-frequency-along-`p` factor content, which **does not delta-code**.
   Early stopping (§12) and discrepancy stopping (§9.4) keep the factors smooth
   along `p` — they are *allies* of §19. Training to a tight tolerance is the
   contradiction.
2. **Boosting (§20) vs raw ratio.** Boosting adds terms to cut error — it
   improves the RD curve at the high-quality end and *worsens* the bare ratio.
   Only coherent if the target is a point on the RD curve, not "max ratio".
3. **bf16 storage vs §14.** Storing bf16 factors is a free 2x, but int8
   quantise (§14) is 4x and subsumes it. bf16 stays a *training* lever
   (session: 1.63x) with no additional storage role once §14 is in.
4. **Quantise (§14) vs share (§5)** — carried from §28(b): a shared entry used
   `p` times amplifies its quantisation error `p`-fold, so term-1 and term-2
   gains partially cancel; the §14 bit budget must rise where §5 cut params.

### (c) Further approaches on this line

- **In the doc:** quantise §14, delta+entropy-code §19, share §5, prune §11,
  and §24 (the missing benchmark).
- **Not yet in the doc:**
  - **RD-optimal rank selection.** Replace the fixed sweep `[2,5,10,20,25]`
    (`DECISIONS.md` §3) with a per-tensor `r` chosen on the RD knee — rank
    where it buys the most PSNR per bit.
  - **Learned entropy model (hyperprior, Ballé et al.).** Predict each factor
    delta's distribution from its neighbours instead of a static arithmetic
    coder — the modern neural-codec move, on top of §19.
  - **Prune at the `r²` cross-term level.** `(W1 H1)∘(W2 H2)` has rank `r²`
    (§3's argument); some of those `r²` column-pair products contribute
    negligibly. Prune which `(i,j)` pairs to keep, not which `r` columns —
    finer, and matched to the actual expressivity.
  - **Differentiable rate penalty.** Add `λ·H(quantised factors)` (relaxed
    estimate) to the loss so the fit minimises *coded size*, not just
    reconstruction error — end-to-end RD optimisation, the neural-codec
    standard.
  - **Vector / product quantisation of factor columns.** Cluster the `r`-dim
    column vectors, store indices (VQ/PQ from ANN search). `DECISIONS.md` §16's
    bounded data → bounded columns → clusterable.
  - **Coarse-init predictor + residual code.** Store a downsampled `X` (§12),
    regenerate the coarse SVD init at decode time, code only the fine
    correction — §12 × §19.

### (d) Gap + mitigation

**Gap.** There is no rate–distortion point for HadNet that a compression
audience would recognise — not even against its own baselines. `DECISIONS.md`
§3's table is `(time, Frobenius error)` per method per rank; "params" is an
fp32 count, "error" is not PSNR-vs-bitrate.

**Mitigation.** Post-process the **existing** Table-9 runs — no retraining.
For each saved factor set: quantise to int8, entropy-code, count actual bytes,
compute PSNR against the *clean* reference, plot `(bpp, PSNR)`. Adds the axis
§24 needs from data already in hand, and reveals whether the tensor baselines
(TSVD, TBCD…) or HadNet win once bits are counted honestly.

**Open question to argue about.** Is the Hadamard model the right
*representation* for compression, or only for low-Frobenius approximation?
Decode is `(W1 H1)∘(W2 H2)` — two batched GEMMs + a multiply, ~0.8 ms for the
whole Carphone tensor (session: forward = 763 µs). That is **fast decode**.
Encode is a 0.7 s+ fit per tensor, against a codec's millisecond encode. So
HadNet is a *slow-encode / fast-decode / Frobenius-optimal* method — the niche
that wants exactly that (archival, one-shot compression of scientific tensors
with no codec) should be named and benchmarked, instead of implying
general-purpose media compression.

**Failure mode.** Turning a 1-knob method (`r`) into a codec-grade pipeline
(RD rank selection + entropy model + VQ + rate penalty) means a dozen knobs,
each needing the RD curve re-measured, and the likely endpoint (§24) is still
losing to x265. A half-built codec is worse than an honest "low-rank
approximation, not a codec" framing with a fast-decode / archival pitch.

---

## 31. The init-cost line — now a bigger slice than the stale numbers say

`DECISIONS.md` §6: the SVD init is the only super-quartic term (`O(n^5)`).
Levers 2 / §13 took it to `O(n^4)` and 4.3x on GPU (`DECISIONS.md` §15). But
"init is 1–3% of runtime" (§3) was measured against the 57 s run.

### (a) What stacks

- **§4 (Gaussian-sketch randomized), §16 (block-Krylov), §8.5 (ACA) are
  alternatives, not a stack** — pick the cheapest that meets the accuracy the
  descent needs, per shape.
- **Init cost multiplies through everything that re-inits.** §11 (rank ladder)
  re-inits per rung; §20 (boosting) per term; §13 (streaming) per slice unless
  warm-started; §26 (distillation) feeds HadNet's init into HadDNN. A 5-rung
  ladder × 3-term boost is 15 inits — the init fraction of `DECISIONS.md` §3
  gets multiplied by the re-init count.
- **§28's "one unfolding SVD" + §5 (shared basis) genuinely compose**: the
  init method *produces* the structure. Same for **§17 (homotopy)** — the
  `γ=0` additive fit *is* the init for `γ=1`, folding "init" into the
  warm-start schedule.

### (b) What contradicts

1. **§16 (block-Krylov) vs the `sqrt(|X|)` transform it inits.** Block Krylov's
   `1/q²` rate needs a spectral gap (§16's open question). `sqrt` is concave —
   it compresses large singular values toward small ones, **flattening** the
   spectrum, which is exactly where block Krylov degrades to the
   power-iteration rate. §16 fights the parameterisation §4 uses.
2. **§11 (rank ladder) wants exact init at inner rungs, randomized only at the
   first.** Each rung re-inits from the *residual* (§11); randomized SVD has
   poor *relative* accuracy when the target is low-energy, which a
   post-previous-rung residual is. Cold `r=2` → randomized fine; `r=10` off an
   `r=5` residual → exact (cheap anyway, small matrix). One method does not fit
   all rungs.
3. **§8.5 (ACA) vs §4's guarantee goal.** ACA is `O(mnr)` with no power
   iterations, but greedy and heuristic — no probabilistic error bound. §4
   explicitly wants to "replace 'seems fine' with a guarantee"; ACA trades
   that away for speed.
4. **§9.1 (log-domain) invalidates the `DECISIONS.md` §13 `niter` sweep.**
   Log-domain SVDs `log|X|`, a different matrix with a different spectrum, so
   the crossover `niter·q < min(m,n)` and the `niter=4` default must be
   re-measured from scratch.
5. **`DECISIONS.md` §14 inversion.** `real-world/` keeps `niter=24` because it
   stays profitable at `min(m,n)=256`; §16 would reach that accuracy in ~6
   passes — so block Krylov helps `real-world/` *more* than the tensor folders,
   inverting the usual "tensor folders are the target" priority.

### (c) Further approaches on this line

- **In the doc:** randomized §4, block-Krylov §16, ACA §8.5, unfolding-SVD §28.
- **Not yet in the doc:**
  - **Compute one SVD at `r_max`, slice for every smaller rank.** The sweep
    (`DECISIONS.md` §3) runs `r ∈ [2,5,10,20,25]`; a truncated SVD at `r=25`
    already contains every smaller answer. ~5x off the sweep's total init cost.
    The init-side of §11.
  - **Skip init for warm-startable slices.** §13: slice `t>0` inherits slice
    `t−1`; only slice 0 pays for an SVD → init cost `→ O(1/p)` of current.
  - **Structured sketch (SRHT/SRFT).** §4/§16 use Gaussian sketches; a
    subsampled randomized Hadamard transform is `O(mn log q)` vs `O(mnq)` —
    marginal at Carphone, real at the `n^4` scale §6 targets.
  - **Shared range-finder for the two SVDs.** `svd_initialize` does two SVDs
    (`mat_simple`, `mat_signed = mat_simple · sign(X)` — same magnitudes). One
    randomized range-finder `Q` from `mat_simple` can seed both, ~halving the
    range-finding pass.
  - **Learned init map** (§9.6 meta-learning, applied to init): a fast
    downsampled-`X` → factors network, meta-trained across a corpus.

### (d) Gap + mitigation

**Gap.** Init-as-%-of-runtime has never been re-measured in the optimised
regime. "1–3%" (`DECISIONS.md` §3) / "3.5%" (§5 bench) were against 57 s.
After bf16 (session 1.63x) + early stopping (~300 epochs) the fit is ~0.7 s,
while the init still runs two batched SVDs of `180×144×144` — plausibly now
10–25% of the run. Same stale-baseline error as §27's `k`-gap.

**Mitigation.** Profile the current bf16 + early-stop fit; break out init vs
epoch loop vs Python/autograd overhead. If init is now >15% of 0.7 s, §16 and
"one SVD at `r_max`" jump the priority order; if still <5%, they stay low.

**Open question to argue about.** Does the init even need to be good anymore?
The `svd_initialize` docstring justifies approximation because it "only seeds a
2000-epoch descent" — but at ~300 early-stopped epochs the descent has far less
room to repair a bad start. Test: random vs SVD vs randomized-SVD init, all
with early stopping, compare *final* error. If random init + 300 epochs ≈ SVD
init + 300 epochs, the whole init line is low-value; a gap means init quality
now matters more than the docstring assumes.

**Failure mode.** "One SVD at `r_max`" only wins for the *full* sweep run in
one process. When the sweep is run for a single rank (common in practice),
precomputing `r=25` is pure waste — more expensive than the `r=2` init it
replaces. Guard it on "sweep length > 1".

---

## 32. The evaluation-honesty line — confounds multiply, and one of them is the headline

§10 lists measurement gaps; §24 adds three more. This section treats honesty as
its own attack axis, because an unpublishable result is a failed optimisation
regardless of the speedup.

### (a) What stacks

Evaluation fixes **compose because confounds multiply**: a result with three
uncontrolled variables is three-ways-unpublishable, and removing one still
leaves the product of the rest.

- **§10's 2×2 ablation `{exact, randomized} × {k=2000, EarlyStop}` + §24's
  "noise off, PSNR vs clean" + `DECISIONS.md` §4.4's "matched iteration
  budget"** run *together* produce one clean table; run separately each leaves
  the others' confounds standing.
- **The GPU baseline (§10 row 1 / `DECISIONS.md` §15) is the prerequisite that
  multiplies through everything.** Every ratio in this document — bf16 1.63x,
  `h_prod` 9–41x, randomized init 4.3x — is meaningless without a same-hardware
  "before". It is not one measurement among many; it is the denominator of all
  of them.
- **§9.4 (discrepancy stop) + `DECISIONS.md` §16 (noise floor):** naming the
  noise level `0.01·sqrt(N)` gives *both* a stop rule and the y-axis floor of
  every error plot. One number, two uses — you cannot beat it, so plotting
  below it is a tell.

### (b) What contradicts

1. **§10's acceptance criterion vs `DECISIONS.md` §15.** §10 says the `h_prod`
   change is "bit-identical, error/PSNR/SSIM must not move". §15 *corrected*
   this: bit-identical is false on GPU (1.4 ulp, cuBLAS reduction order).
   §10's own bar contradicts the finding; it needs rewording to §15's revised
   "exact in float64, ≤8 ulp on GPU".
2. **bf16 (session, "4th-decimal move") sits exactly on that boundary.** §10's
   old "must not move" would *reject* bf16; §15's revised "movement beyond the
   3rd decimal means something broke" would *accept* it. Which bar applies is
   unresolved, and bf16 is the test case that forces the choice.
3. **§24 ("noise off for codec comparison") vs `DECISIONS.md` §16 ("a hard
   [0,1] constraint would be actively harmful — the target is itself out of
   range").** §16's entire conclusion assumes noise *on*. Turn it off for §24
   and the §16 range-constraint experiments must be re-run; the two sections
   assume opposite noise settings.
4. **§26 (does HadDNN earn its cost) vs the paper's "neural architectures"
   framing.** Reporting HadDNN as a headline method when it is 8x slower, worse
   (`DECISIONS.md` §12), and ~half inert (session) is itself the honesty
   problem — but cutting it weakens half the stated contribution. Honesty vs
   narrative, unresolved.

### (c) Further approaches on this line

- **In the doc:** §10 (gap table), §24 (codec / corrupted reference / ratio
  accounting), §9.4 (noise floor).
- **Not yet in the doc:**
  - **Seed variance / confidence intervals.** Every number in `DECISIONS.md`
    §3/§12/§15 and this session is `n=1`. Report mean ± std over ≥5 seeds
    before any of them is load-bearing — especially "0.97x" (§6), which noise
    could push either side of 1.0.
  - **Fresh process per sweep point.** The rank sweep runs sequentially in one
    process (`DECISIONS.md` §3); RNG state, cudnn autotune cache and thermal
    throttling over a multi-hour run mean late ranks are measured under
    different conditions. Randomise rank order or isolate processes.
  - **Split decode cost from fit cost.** §30: decode ~0.8 ms, encode 0.7 s+.
    `DECISIONS.md` §3's "time" conflates fit + reconstruction + metrics.
  - **Report wall-clock, FLOPs, and epochs — all three.** `DECISIONS.md` §4.4
    shows wall-clock alone misleads (numba-CPU vs GPU, 20x iteration mismatch).
    FLOP counts are hardware-independent.
  - **Held-out evaluation for the regularisation claim.** `DECISIONS.md` §12's
    "early stopping improved HadDNN -0.08%" is a generalisation claim; it needs
    a held-out slice or a held-out noise realisation, not the fit residual.
  - **Error-at-fixed-time, not time-at-convergence.** Give every baseline the
    same wall-clock budget (`DECISIONS.md` §4.4: AGD/IGD get 40000 iters vs
    HadNet's 2000) and compare error at that budget.

### (d) Gap + mitigation

**Gap.** No section addresses statistical significance. Every measured claim —
`DECISIONS.md` §3, §12, §15; session bf16 / compile / DNN — is `n=1`. Point
estimates with unknown variance, and "torch.compile 0.97x" is close enough to
1.0 that noise could flip its sign.

**Mitigation.** `bench/variance.py`: run the core fit ≥10x per config
(fp32 / bf16 / +compile / +early-stop) across seeds, report mean ± std and a
paired test between configs. Each fit is ~0.7–1 s, so the whole thing is
minutes. Run it *before* any session number enters a paper or reorders the
priority list.

**Open question to argue about.** How much of the headline "21x faster than
TAGD" (`DECISIONS.md` §3: 57 s vs 1193 s) survives an honest re-run? §4.4
already notes it is "close to the iteration-count ratio alone" (40000/2000 =
20). Matched iteration budget and hardware — does HadNet keep *any* speed
advantage, or is the entire headline an artefact of unequal setup? Highest-
stakes question in the document; nothing currently commits to answering it.

**Failure mode.** A fully honest re-evaluation (matched budgets, matched
hardware, error bars, codec baselines, FLOP counts) may show HadNet leads on
neither speed nor ratio — reducing the contribution to "a parameterisation
with `r²` expressiveness", a weaker but still real paper. The risk of doing
the honest evaluation is having to change the framing; the risk of not doing
it is a reviewer changing it for you.

---

# Second pass — deeper on the highest-value axis

All six axes covered once (§27–§32). The session measurements point hardest at
**per-epoch cost**: backward is 75% of an epoch, bf16 buys 1.63x, and §27's
open question — is `k` even the dominant lever after early stopping — makes the
*content* of one epoch, not their count, the live target. §33+ drill into that.

## 33. Per-epoch cost, and specifically the backward pass (the measured 75%)

`k` (§27) attacks epoch *count*. This attacks epoch *content*. At ~300
early-stopped epochs × ~2.3 ms (bf16) the per-epoch anatomy is: forward 763 µs,
**backward 2860 µs**, Adam 173 µs (session, Carphone shape).

### (a) What stacks

- **bf16 (1.63x) is arithmetic width; lower `r` (§11 ladder) is arithmetic
  volume (`O(k·r·mnp)`, linear in `r`); block solves (§25 ADMM / §15 Schur /
  §1 Riemannian) replace the joint 4-factor backward with per-block work.**
  These three multiply: a low-`r` rung, in bf16, with the backward replaced by
  a block solve, cuts all three of width, volume, and structure.
- **§5 (factor sharing) does NOT join this stack** — §28 established it shrinks
  the *parameter* gradient, not the residual computation, and the residual is
  what backward is dominated by. §5 is a compression lever with ~zero per-epoch
  speed effect; do not count it here.
- **A hand-written `autograd.Function` backward for `forward()` composes with
  bf16** — it reuses `M1, M2` from the forward for the product rule instead of
  letting autograd hold/rebuild the graph, and the reused GEMMs are themselves
  bf16.

### (b) What contradicts

1. **bf16 vs the tail.** bf16 accumulates gradients and Adam moments at reduced
   precision; late in training the gradient is ~1e-3 (the §11 plateau), where
   bf16's ~1e-2 relative error corrupts the moment estimates. bf16 could
   *lengthen* the tail even while speeding each epoch — net below 1.63x. (This
   session it did not: ~300 epochs both, Carphone. At tighter tolerances or
   other shapes it might.)
2. **Custom fused backward (§6 Triton) vs the bit-identical bar.** A
   hand-written backward will not match autograd's reduction order — it lands
   under `DECISIONS.md` §15's "≤8 ulp on GPU", not §10's "bit-identical". Same
   boundary §32(b) flags for bf16.
3. **Lower `r` vs §3's whole premise.** Hadamard's selling point is rank `r²`
   at `2r(m+n)` params (§3). Cutting `r` to cheapen backward discards the
   expressiveness that motivates the method. §11's ladder is safe only if it
   *ends* at the target `r`; sustained per-epoch pressure toward a permanently
   lower `r` undercuts §3.
4. **"Replace backward" (§15/§25/§1) is not "remove backward".** A Schur
   block-solve sweep is itself `O(mnr²)` per block plus a batched
   Cholesky/lstsq over 180 slices, with its own kernel-launch profile. §15's
   own open question (the 1440×1440 solve) already doubts it beats the autograd
   backward it replaces — measure before committing.

### (c) Further approaches on this line

- **Touches the doc:** bf16, low-`r` §11, block solves §15/§25/§1, Triton §6
  (but `torch.compile` max-autotune measured **0.97x** — SM-starved laptop
  4050).
- **Not yet in the doc:**
  - **Explicit `autograd.Function` for `forward()`.** ~40 lines: cache
    `M1, M2`; `dW1 = (dX∘M2) batched-@ H1ᵀ`, `dH1 = W1ᵀ batched-@ (dX∘M2)`,
    symmetric with `M1`. Four batched GEMMs + two elementwise, no graph
    bookkeeping. Directly targets the 2860 µs.
  - **Fuse residual + loss into one kernel.** `norm(D − M1∘M2)` currently
    materialises the full `144·144·180` (15 MB) `(D − X_hat)` temp, then reads
    it back for the norm. A single pass computing residual and its
    sum-of-squares saves that write+reread — a bandwidth win (`DECISIONS.md`
    §5 notes the small-matmul regime is launch/bandwidth-bound, not FLOP-bound).
  - **`torch.compile(mode="reduce-overhead")` = CUDA graphs.** A *different*
    mechanism from the max-autotune path that scored 0.97x — it captures the
    step as a replayable graph and cuts Python/launch overhead specifically.
    Untested this session; the natural next probe after max-autotune failed.
  - **Freeze converged blocks.** If `‖grad(W2,H2)‖` collapses while `(W1,H1)`
    is still moving, stop backpropagating into `(W2,H2)` — up to ~2x off
    backward in the tail, where `DECISIONS.md` §11 says most epochs live.
  - **Squared loss.** `2(X_hat − D)` skips the `/‖R‖` divide and the norm's own
    backward node. `squared_loss` already exists (`DECISIONS.md` §11/§14);
    marginal, but it is inside the 75%.

### (d) Gap + mitigation

**Gap.** "Backward = 75%" is measured but **not decomposed**. Of the 2860 µs,
how much is the 4 batched GEMMs vs autograd graph overhead vs the norm's
backward vs memory traffic from the `(D − X_hat)` temp? bf16 helping 1.63x
hints arithmetic-bound; the ratio falling short of 2x hints overhead too.
Nobody has looked.

**Mitigation.** `torch.profiler` on one training step, kernel-level: GEMM vs
elementwise vs memcpy vs CPU-side gaps. If backward is ~all GEMM, only
bf16 / lower-`r` / block-solves help and a custom kernel is wasted effort. If
≥20% is overhead or memory, the fused backward and fused loss are worth
writing. One profile decides.

**Open question to argue about.** Is a custom `autograd.Function` worth the
maintenance cost? It is fragile — it must track `h_prod`'s `permute` layout and
the `*p` slice dims, and any later edit to `forward()` breaks it *silently*
(wrong gradients, not a crash). The repo's ethos is "the fix lands once"
(`hadamard_ops.py`); a hand-written backward is the opposite. Probably only
justified if the profiler shows >30% of backward is non-GEMM.

**Failure mode.** Kernel-level work on the laptop 4050 may be futile: the same
SM starvation that made `torch.compile` 0.97x ("Not enough SMs", session)
caps any hand-fused kernel. "Backward is 75%" might be **un-actionable on this
hardware** and only bite at the `n⁴`-tensor / workstation-GPU scale §6 and
`GPU.md` target — in which case the honest move is to measure it there before
optimising for it here.

---

## 34. The HadamardDNN finding as a cross-cutting constraint (second pass on `p`-scaling)

Session measurement: `HadamardDNN` is **134.8M params vs HadNet's 1.04M**, and
freezing every `FactorNet`'s first `Linear(259200→64)` plus its learned input
vector — **~half the parameters** — leaves training and final error unchanged
(rel err 0.367). This is not a bug to file; it re-reads several lines above.

### (a) What it reveals — a hidden composition

`FactorNet(learned_constant_vector)` with a 64-wide first layer *is* an
**accidental rank-64 factorisation** of the flattened `(m,r,p)` factor. So
HadDNN already implements §5's "shared basis + per-slice coefficients" — at
~130x the parameter cost, with the basis learned by an inert MLP instead of set.
Consequences:

- **§5 + §26 compose exactly here.** "Replace `FactorNet` with
  `Basis[mr, s] · Coeff[s, p]`, initialise `Basis` from HadNet's converged
  factor by SVD" delivers HadDNN's *intended* low-rank-along-`p` structure at
  `s(mr + p)` params, warm-started, no MLP.
- **§8.4 is the correct `FactorNet`.** With a real coordinate input (the slice
  index `p`) instead of a learned constant, `MLP(p)` genuinely varies and the
  130x bloat becomes a compact continuous representation. §27(b)7 flagged the
  trap; this is the resolution — `FactorNet` ∘ fix-the-input = §8.4.
- **It is already a compression lever (§30), unclaimed.** A rank-64 factor of
  `(m,r,p)` stored explicitly is `64(mr+p) ≈ 104k` numbers vs HadNet's
  `mrp = 259200` — ~2.5x on top of HadNet, buried inside 33M-param nets.

### (b) What contradicts

1. **§26 (distil HadNet → HadDNN) vs the bottleneck.** HadNet's converged
   factor can be full-rank; a 64-wide bottleneck cannot receive its rank->64
   content. §26's distillation is **lossy by construction** unless the
   bottleneck is widened or removed — §26 did not know this.
2. **The "neural architectures" contribution vs the measurement.** If
   `FactorNet` is provably ≈ an explicit low-rank map with half its params
   inert (session), "neural architecture" names "a low-rank reparameterisation
   with extra steps". §32(b)4's honesty tension, now concrete.
3. **§8.4-fix vs §26-fix don't compose.** Coordinate-net says "make the input a
   coordinate"; distillation says "warm-start the weights". Do the former and
   the thing being distilled *into* is a different architecture onto which
   HadNet's factors do not map cleanly.
4. **The bloat and the regularisation are the same mechanism.** `DECISIONS.md`
   §12: HadDNN *improved* with early stopping — "regularisation on the
   over-parameterised factor nets". The 64-dim bottleneck is a strong implicit
   regulariser. Remove it (the §5/§26 fix) and that regularisation goes with
   it — may need re-adding explicitly (§29's L2).

### (c) Further approaches (this is `p`-scaling revisited)

- **Sweep the bottleneck width `s`.** It is hard-coded 64 (session). Sweep
  `s ∈ {8,16,32,64,128}`, read error — this *measures the effective `p`-rank*,
  which §28(d) named as the unmeasured gap, as a free side effect.
- **One shared net + a factor-id embedding** instead of four structurally
  identical `FactorNet`s — 4x off the net params.
- **Drop the learned input, feed a fixed Fourier/coordinate basis of `p`** —
  `FactorNet` becomes an honest `p → factor-slice` map (§8.4); 33M → ~`64·mr`.
- **Just don't allocate the inert half.** Session showed freezing it costs
  nothing; not allocating it is an immediate 2x cut to HadDNN params, memory,
  and Adam state, measured, zero accuracy cost.

### (d) Gap + mitigation

**Gap.** `DECISIONS.md` §12 reports HadNet-vs-HadDNN error without either
model's parameter count. A reader sees "architecture vs architecture"; it is
actually **1.04M params vs 135M (67M inert)**. The comparison's headline
("8x slower, no better") is missing the term that explains it.

**Mitigation.** Add a parameter-count column to every HadNet/HadDNN table.
One column reframes the result as "**135x bigger**, 8x slower, no better" —
which is the finding, and changes the conclusion a reader draws.

**Open question to argue about.** If the §5/§26 fix works, is there any reason
for HadamardDNN to remain a *separate method*, or does it collapse into "HadNet
with an optional low-rank-along-`p` knob" — one method, not two? The paper's
two-named-contributions structure may not survive the fix.

**Failure mode.** The bottleneck's *nonlinearity* (LayerNorm + ReLU), not its
low rank, might be what regularises (`DECISIONS.md` §12's −0.08%). An explicit
rank-`s` *linear* replacement would then underperform, and you would be back to
tuning an explicit regulariser (§29) to match. Test the linear replacement
against the nonlinear `FactorNet` at matched width before calling the MLP pure
waste.

---

## 35. The composed-stack claim — Amdahl, ordering, and whether it has ever been run (second pass on evaluation-honesty)

§27(a) asserts "~5x over baseline before any research-grade change". No run
has ever applied all applied levers together and timed it. This section audits
that claim.

### (a) What the applied levers actually contribute at loop level

| Lever | Headline number | Level it was measured at | Loop-level effect (Amdahl) |
|---|---|---|---|
| `h_prod` batch (§4/L1) | 9x CPU / **41x GPU** (`DECISIONS.md` §15) | the *kernel* | `h_prod` ≈ 14% of the epoch (session: 526 µs of 3795) → epoch ceiling **~1.15x** |
| early stopping (§12/L3) | **3–3.3x** (`DECISIONS.md` §12) | the *loop* | ~3x, genuine |
| randomized init (§4/L2) | 2x CPU / **4.3x GPU** (`DECISIONS.md` §15) | *init* | init ≈ few % (§31 gap) → epoch-run ceiling **~1.03x** |
| bf16 (session) | **1.63x** | the *loop* | ~1.63x, genuine |

Composed loop-level: `3 × 1.63 × 1.15 × 1.03 ≈ 5.6x`, and it is **dominated by
early stopping and bf16**. The two biggest-*looking* levers — `h_prod` 41x,
init 4.3x — are the two *smallest* at the level that matters, because Amdahl
caps a component speedup by that component's runtime share. The doc never
states this; `DECISIONS.md` §15 measured the kernels, not the loop.

### (b) What contradicts inside the applied stack

1. **Early stopping cannibalises the `h_prod` win.** Cutting epochs 3x cuts
   the total time spent in `h_prod` 3x, shrinking its already-tiny absolute
   contribution further. They compose, but §12 eats §4.
2. **Randomized init + early stopping — the §10 confound, still unrun.** Both
   shift error ~1% and are always on together (`DECISIONS.md` §10). A rougher
   init *and* a shorter descent could compound to worse final error than
   either alone; the 2×2 grid §10 asks for has not been run.
3. **Stacked perturbations vs the "3rd decimal = broken" bar.** bf16 rounding
   (~1e-2 relative) + randomized-init delta (1.7e-3, `DECISIONS.md` §13),
   together, may push final error past `DECISIONS.md` §15's acceptance
   threshold. Unchecked.
4. **bf16 vs `EarlyStop`'s trigger.** `EarlyStop` fires on relative
   improvement `< 1e-3`; bf16's loss-noise floor is ~1e-2 relative, so bf16
   can make `EarlyStop` quit *before* real convergence — fewer epochs, higher
   error. An **ordering interaction** between two applied levers: measuring bf16
   speedup without re-checking the stop point overstates it.

### (c) Further approaches on this line

- **In the doc:** §10 (gap table), §24, §32, §9.4.
- **Not yet:**
  - **One frozen canonical config.** Folders disagree on `max_epochs`
    (2000 vs 10000, `DECISIONS.md` §14), loss (squared vs unsquared, §14),
    `niter` (4 vs 24, §13). No two folders measure the same thing — define one
    config, port all four, re-run.
  - **Amdahl column on every speedup.** Report component-level *and*
    loop-level (component × runtime fraction) side by side.
  - **Ablation by removal.** Start from the full stack, drop one lever at a
    time, measure the delta — the reverse of `DECISIONS.md`'s add-one-at-a-time
    story, and the only way to see which levers are load-bearing once stacked.
  - **`bench/regression.py`.** Re-runs the headline configs, asserts the
    `DECISIONS.md` numbers still hold within seed variance — catches silent
    drift as code changes.
  - **Provenance tag on every number.** `DECISIONS.md` §5's bench is
    CPU/numpy; §15 is GPU; §3 is from the paper. Tag each figure with how it
    was obtained; the doc currently reads them as interchangeable.

### (d) Gap + mitigation

**Gap.** The composed end-to-end speedup **has never been measured**. `h_prod`
(§15, kernel), early stop (§12, loop), randomized init (§13/§15, init), bf16
(session, loop) were each measured alone, on different setups and shapes. No
run applies all four and times it against the unmodified `main` HadNet path on
one GPU. §10 row 1 is still open — and now it must be baseline vs
*fully-optimised*, not vs any single lever.

**Mitigation.** One script, one GPU, one shape: `git stash` → time the `main`
HadNet path → restore → time with everything on → repeat ≥5 seeds. Report *that*
number with variance. It, not the component figures, is the paper's claim.

**Open question to argue about.** If the composed win is ~5–6x and it is
~all early stopping (3x) + bf16 (1.63x) — two changes that are (a) not novel
and (b) apply equally to the *baselines* (AGD/IGD/BCD could also early-stop and
run bf16) — does HadNet's *relative* advantage move at all? These levers may
lift all boats. Note `DECISIONS.md` §7's remark that `h_prod` batching *widens*
HadNet's lead specifically — early stopping and bf16 have no such property, so
they may leave the comparative claim exactly where it was, or erode it.

**Failure mode.** Freezing a canonical config and re-running all four folders
is a multi-hour job (`COMMANDS.md` §3: full video sweep is hours) on a 4050
that thermally throttles over long runs (§32(c)). The honest re-run may itself
be unreliable on this hardware; done properly it needs the workstation
(`GPU.md` §4). Half-doing it on the 4050 yields numbers no more trustworthy
than the ones it replaces.

---

## 36. Gauge-fixing as connective tissue, and when to do it (second pass on conditioning)

§29 audited the conditioning layer internally. The cross-cutting point: a
consistent gauge is a **precondition for four other lines**, so §18/§8.2 may
earn their place even if `DECISIONS.md` §11's plateau turns out not to be
conditioning at all.

### (a) What a fixed gauge unlocks

The scaling/`GL(r)` symmetry (§1) means "the factors" are only defined up to a
transform. Every approach that *reuses factors across something* needs them in
a consistent gauge or the reuse is noise:

- **§19 (delta-code along `p`)** — `W1[:,:,t] − W1[:,:,t−1]` is only small if
  consecutive slices share a gauge (already flagged in §19, §27).
- **§13 (streaming warm-start)** — slice `t` inheriting slice `t−1`'s factors
  assumes the same gauge.
- **§11 (rank ladder)** — embedding the `r=5` solution into `r=10` needs the
  `r=5` factors canonical, or the appended columns land arbitrarily.
- **§26 (distillation)** — distilling into a canonical-gauge student (§34).

So **one post-fit canonicalisation** — a single QR/SVD of the converged
factors, or Sinkhorn to convergence, *outside* the epoch loop — composes with
all four at near-zero cost. This is a different use of §18/§8.2 than "fix the
conditioning every step".

### (b) What contradicts

1. **Every-step gauge-fix (§18 QR, §8.2 Sinkhorn) vs the §33 per-epoch
   budget.** Both add forward+backward ops over `p`-carrying factors — the
   §4.1 loop hazard, on top of a backward that is already 75% of the epoch
   (session). The conditioning benefit and the connective benefit want
   *different schedules*: every-step for the former, once-at-end for the latter.
2. **§1 (Riemannian quotient) is architecturally incompatible with §19/§13.**
   A quotient-manifold optimiser never materialises explicit factors in a
   fixed gauge — it works with equivalence classes. The compression and
   streaming lines need explicit, gauge-consistent factor *tensors* to consume.
   §1's output cannot feed them without a projection step that undoes part of
   what §1 bought.
3. **Canonicalise-at-end vs quantise (§14).** Post-fit QR changes the factor
   magnitudes (pushes scale into `R`); the §14 per-column quantisation scales
   must be recomputed *after* canonicalisation, not before — order matters.

### (c) Further approaches on this line

- **In the doc:** §1, §8.2, §8.3, §9.1, §9.3, §18, plus §29's spectral-norm /
  `S^{-1/2}` preconditioner / L2 / named-scale-coordinate.
- **Not yet:**
  - **Canonicalise at checkpoints, not every step.** Every `N` epochs, one
    batched QR of the factors — amortises the §33 cost while stopping the gauge
    drifting far. Tunable `N` trades cost against drift.
  - **Procrustes-align each epoch's factors to the (canonical) SVD init.**
    Cheaper than full balancing; keeps the gauge stable enough for §19/§13
    without solving the conditioning.
  - **Measure the drift first.** Log the balancedness residual
    `‖W1ᵀW1 − H1H1ᵀ‖_F / ‖W1ᵀW1‖_F` per epoch. Deep-linear-net theory (§8.2,
    Arora et al.) says gradient *flow* preserves it — but Adam is not gradient
    flow. If it stays ~0 under Adam, the whole symmetry layer is unmotivated
    for HadNet-as-run.
  - **Projected Adam vs plain Adam ablation.** If Adam is what breaks the
    gauge, projecting to Stiefel each step (§18) vs not isolates the effect in
    one run.

### (d) Gap + mitigation

**Gap.** No section measures whether the gauge actually drifts under Adam.
§1/§8.2/§8.3/§18 all *assume* it does; the only cited theory (Arora et al.,
via §8.2) is for gradient flow, which the repo does not run — it runs Adam
(`lr=0.1`, `DECISIONS.md` §9).

**Mitigation.** Log `‖W1ᵀW1 − H1H1ᵀ‖_F / ‖W1ᵀW1‖_F` every epoch for one
Carphone fit and one `random-tensor` fit. `< 1e-3` throughout → the symmetry
layer (§1/§8.2/§8.3/§18) is solving a non-problem for HadNet as configured, and
only the *connective* use (36(a)) justifies §18. Growing to `O(1)` → the layer
is real and §18 (cheapest, §29) leads it.

**Open question to argue about.** If the gauge does *not* drift under Adam, why
does the `DECISIONS.md` §11 plateau happen? That single measurement would
eliminate the symmetry explanation and hand the plateau to §12 (low-frequency
mode), §17 (non-convex basin), or simply too-small `lr` — redirecting the
entire "attack the plateau" effort away from §1/§8.2/§18.

**Failure mode.** Balancedness drift may differ between `random-tensor` (no
structure) and Carphone (structured) — the §12/§13/§21 image-vs-random split
recurs. One dataset's measurement could send the priority order the wrong way;
run both before concluding.

---

## 37. Rate–distortion as the objective that reorders the document (second pass on compression-ratio)

§30 built the ratio pipeline. This asks: if the paper commits to **(bits,
PSNR)** as the headline instead of **(time, Frobenius)**, which sections become
primary, which become irrelevant, and what breaks.

### (a) What composes under an RD objective

- **Core:** §14 (quantise) → §19 (entropy-code) → §5 (share) → §11 (prune) →
  §24 (codec baseline). These are the RD story and they multiply (§30(a)).
- **Demoted to "rank per encode-second":** §1 / §8.2 / §18 (conditioning),
  §8.1 / §22 / §33 (encode speed) matter *only* insofar as they let you fit at
  higher rank in a fixed encode budget → lower distortion. They stop being
  ends and become a single knob: distortion-at-fixed-encode-time.
- **RD-curve-shape tools:** §3 (`k`-factor), §20 (boosting) — they trade
  parameters for rank, i.e. they *move you along* the RD curve rather than
  push it down.
- **A fixed gauge (§36) is still needed** — §19 delta-coding is in the core
  path.

### (b) What contradicts under this framing

1. **Frobenius loss vs perceptual metric.** The code minimises Frobenius;
   `DECISIONS.md` §10 *reports* PSNR and SSIM. A §24 codec comparison on SSIM
   is unfair to HadNet, which never optimised it. Committing to RD forces
   either an SSIM/MS-SSIM loss or an honest "PSNR-only, and here is why".
2. **Differentiable rate penalty (§30(c)) vs the stopping rules.** Adding
   `λ·H(factors)` changes the loss scale, so `EarlyStop`'s `min_rel=1e-3`
   (claimed scale-free in §12) no longer transfers, and §9.4's discrepancy
   threshold — defined on the *Frobenius* residual — is undefined for the
   rate-augmented objective.
3. **VQ/PQ (§30(c)) vs the SVD init (§4).** Vector quantisation wants
   *redundant*, clusterable column vectors; a fresh SVD init has near-
   orthogonal, maximally-spread columns — the worst case for clustering. VQ
   only pays late in training once columns have collapsed toward shared
   structure, which contradicts "quantise the init".
4. **RD framing vs `DECISIONS.md` §7's "widens the lead" note.** `h_prod`
   batching widened HadNet's *speed* lead; under an RD headline, speed is not
   the axis, so that argument stops supporting the paper's contribution.

### (c) Further approaches on this line

- **In the doc:** §14, §19, §5, §11, §24, §30(c)'s RD-rank / hyperprior /
  `r²`-prune / rate-penalty / VQ.
- **Not yet:**
  - **Report a 4-tuple, not a ratio:** `(bpp, PSNR, encode-s, decode-ms)`.
    Makes HadNet's ~0.8 ms decode (session) a visible axis where it can beat
    learned codecs (which also decode slowly).
  - **Progressive / embedded bitstream.** Order factor columns by singular
    value (already so, from the SVD init) and `r²`-term energy; any bitstream
    prefix is a valid lower-rank reconstruction. One stored file yields every
    rank in the `[2,5,10,20,25]` sweep — JPEG2000's progressive-by-quality,
    free from the existing structure.
  - **Right baseline class: per-image-overfit neural representations**
    (COIN, COIN++), not only JPEG/x265 (§24). Same "fit a model per signal,
    slow encode, fast-ish decode" profile — the fair peer.
  - **Scheduled QAT.** fp32 for the first ~200 epochs, straight-through
    quantise for the last ~100 — spend `DECISIONS.md` §12's <1%-value tail on
    quantisation robustness.
  - **Entropy-code the sign map separately** (§23). If low-TV for images it
    codes to near-zero bits; the current model carries sign in fp32 mantissas
    implicitly.

### (d) Gap + mitigation

**Gap.** The paper reports a *ratio* (`entries / params`) — a unit no
compression audience uses — and never a *bitrate* (bits per entry). Ratio
silently assumes fp32; bpp forces the assumption into the open. And every
result is a single point, not an RD curve (sharpening §24/§30: the missing
thing is the *axis*, not just a baseline on it).

**Mitigation.** Recompute every `DECISIONS.md` §3 / Table-9 entry as
`(bpp, PSNR-vs-clean)` from actual serialised byte counts — `int8 + gzip` as a
floor, arithmetic coding for the real figure — and plot. Reveals as a side
effect whether the `[2,5,10,20,25]` sweep even spans a useful bitrate range or
bunches in one corner of the RD plane.

**Open question to argue about.** Is HadNet's defensible niche "compression" at
all, or "a fast-decoding learned representation for data decoded many times"
(streaming origin, renderer texture, a scientific field queried repeatedly)?
Decode ~0.8 ms (session) is genuinely good; encode 0.7 s+ is genuinely bad.
Every framing choice follows from which number is the pitch.

**Failure mode.** An RD headline puts HadNet on a plot against 30 years of
codec engineering and a steep bpp–PSNR frontier that took learned codecs years
to reach. A low-rank-tensor method will likely sit well below it, and "worse
than x265 but decodes fast" is a niche claim that needs the decode advantage
measured against a real-time decoder, not asserted.

---

## 38. What the init *is*, not just what it costs (second pass on init cost)

§31 treated the init as a cost to cut. This asks what `svd_initialize` actually
encodes and whether it is the right prior — which changes whether the whole
`O(n^5)` story (`DECISIONS.md` §6) is even worth defending.

### (a) The init is a prior four other lines depend on

`svd_initialize` = SVD of `sqrt(|X|)` and `sqrt(|X|)·sign(X)`, scale split
50/50 into `W` and `H`. It hard-codes the assumption `M1 ≈ M2 ≈ sqrt(X)`.
Every line that *reuses or warm-starts* factors is consuming that prior:

- **§17 (homotopy)** — the `γ=0` additive fit is a *better* init (bi-convex, no
  symmetry). §17 and this line converge: "best init" = "solve the easy problem
  first".
- **§11 (rank ladder)** — rung `r+1`'s init is rung `r`'s solution +
  SVD-of-residual, so the init becomes *incremental* and cheap per rung.
- **§26 (distillation)** — HadNet's converged factors are HadDNN's init; "init"
  spans models.
- **§21 (AMP)** — carries its own analytically-prescribed spectral init, which
  the `sqrt`-split heuristic may be far from.

Composed: additive fit (§17) → SVD of it → ladder increment (§11) → seed HadDNN
(§26). One init pipeline across four sections.

### (b) What contradicts

1. **The `sqrt` split vs §3's `r²` expressiveness.** The model reaches rank
   `r²` *because* `M1 ≠ M2` — if `M1 = M2` the product is `M1²`, rank ≤ `r`.
   The init starts at `M1 ≈ M2`, i.e. the **least expressive point of the model
   class**. §4's init and §3's headline argument pull opposite ways.
2. **§16 (faster init) vs §31's open question, sharpened.** If init quality
   does *not* matter at ~300 early-stopped epochs, only speed matters and §16 /
   ACA win. If it *does* matter, a faster-but-rougher init is exactly wrong and
   you want a *better* one (§17's additive fit, AMP's spectral init) even if
   slower. §16 and "better init" are opposite bets on one unanswered question.
3. **§9.1 (log-domain) shares no init with the multiplicative model.** SVD of
   `log|X|` gives an additive low-rank init; `exp` of a low-rank matrix is not
   low-rank, so it cannot seed the multiplicative factors directly. The two
   fitting regimes need separate inits.
4. **Randomized init (§4/L2) noise vs the degenerate start.** The randomised
   SVD's perturbation might *help* by breaking `M1 = M2` — an accidental
   benefit — or be too small to matter. Unmeasured, and it interacts with (1).

### (c) Further approaches on this line

- **In the doc:** randomized §4, block-Krylov §16, ACA §8.5, one-SVD-at-`r_max`
  §31, skip-for-warm-slices §31, learned init §31.
- **Not yet:**
  - **Asymmetric init.** Seed `M1` from the positive part of `X`, `M2` from the
    negative part (or two different transforms) so the model starts with real
    `r²` expressiveness. No extra cost.
  - **Reuse the baseline SVD.** `main_video.py:291` already computes a full
    per-slice SVD for the `X_svd` baseline *before* HadNet runs, then discards
    it (`DECISIONS.md` §3: baselines run first in the sweep). HadNet's init
    could consume it — a genuinely free init.
  - **A few iterations of BCD/AGD as the init.** The baseline solvers are in
    the codebase; 5 iterations land in the right model class cheaply.
  - **Line-search the scale split.** `W ← U·S^α`, `H ← S^{1-α}·V`; a quick
    sweep over `α` at init pre-balances (§8.2) for free instead of assuming
    `α = 0.5`.
  - **No init.** Test the null directly (§31's open question): random
    orthonormal + power-scaled `W, H`. If it ties SVD init at 300 epochs,
    delete `svd_initialize` and the `O(n^5)` term with it.

### (d) Gap + mitigation

**Gap.** An init supplies three things — right subspace, right scale,
symmetry-breaking — and no experiment separates them. "SVD init beats random"
(implicit in the code) conflates all three.

**Mitigation.** Four-way ablation at a fixed epoch budget: (i) full
SVD-of-`sqrt` (current); (ii) same subspace, random scale split; (iii) random
subspace, matched scale; (iv) asymmetric `M1 ≠ M2`. Measure epochs-to-tolerance
and final error. If only "rank + scale" is load-bearing, the SVD is overkill
and a random-orthonormal + power-scaled init suffices — which removes the
init-cost line entirely.

**Open question to argue about.** Is the `O(n^5) → O(n^4)` result
(`DECISIONS.md` §6, the paper's stated complexity target) attacking a term a
proper init-necessity study would *delete*? If HadNet fits from a rank-`r`
random init, there is no SVD, no `n^5` term, and the complexity contribution
becomes "you don't even need the SVD" — a stronger result that nonetheless
reframes the paper.

**Failure mode.** The degenerate-start concern is theoretical; the randomised
init's noise or the first few Adam steps may break `M1 = M2` immediately. Log
`‖M1 − M2‖ / ‖M1‖` over the first 20 epochs first — if it is already `O(1)` by
epoch 5, the start self-corrects and the asymmetric-init idea is moot.

---

## 39. The measurement-first decision tree (synthesis of every §(d))

Every second-pass section ended with a cheap experiment that would *kill or
promote a whole layer*. §39 collects them, notes which share infrastructure,
and which outcomes contradict — because running these six is worth more than a
seventh synthesis section.

### (a) The diagnostics compose into ~one instrumented run

| # | From | Measures | One outcome that kills a layer |
|---|---|---|---|
| M1 | §27(d) | re-run the §11 curve with `EarlyStop` on | "~300 epochs is the knee" → §8.1/§11/§12/§13/§17/§22 have little left |
| M2 | §29(d) | Lanczos HVP `κ` + consecutive-grad angle at the plateau | "tiny gradient" → not conditioning; §1/§8.2/§8.3/§18 misdirected |
| M3 | §36(d) | log `‖WᵀW−HHᵀ‖/‖WᵀW‖` per epoch | "stays <1e-3 under Adam" → symmetry layer solves a non-problem |
| M4 | §28(d) | SVD the `p`-mode unfolding (`144·144 × 180`) | "effective rank ~120" → §5/§28/§34 sharing mostly dead |
| M5 | §35(d) | `git stash` composed-stack run, ≥5 seeds | the real end-to-end speedup + variance; prerequisite for every ratio |
| M6 | §38(d) | 4-way init ablation (subspace / scale / symmetry) | "only rank+scale matter" → delete `svd_initialize`, `n^5` term goes |

**M1, M2, M3, M6 fall out of one instrumented fit** (log the loss curve, grad
norms, balancedness, run with four init variants). M4 and M5 are small
standalone scripts. Add M7 = one `torch.profiler` step (§33(d), backward
GEMM-vs-overhead). Total: an afternoon on the 4050, and it resolves the
priority of ~20 sections.

### (b) Outcomes that contradict each other's premises

1. **M2 "large oscillating gradient" + M3 "no drift"** is a contradiction: a
   badly-conditioned landscape whose ill-conditioning is *not* the scaling
   symmetry. Then §1/§8.2/§18 (which target the symmetry specifically) would
   not fix it — the conditioning would trace to the `sqrt` transform or the
   unsquared loss instead. The pair can confirm the *symptom* while
   invalidating the *mechanism* the whole layer assumes.
2. **M5 depends on M1.** The composed-stack number is only meaningful if the
   early-stopping point is right, but M1 needs the composed setup to measure
   the stop point in context — circular. Resolve by running M5 twice: once at
   fixed `k`, once with `EarlyStop`, and report both.
3. **M4 + M6 interaction.** M4 "p-rank ~15" and M6 "subspace doesn't matter"
   together mean the shared basis can be a *random* rank-15 projection —
   §5 and §38 collapse into one trivial change, no SVD anywhere. M4 "p-rank
   ~120" makes §5/§28/§34 mostly dead regardless of M6.

### (c) Further diagnostics not yet proposed anywhere

- **Per-slice error histogram.** Uniform across 180 slices, or spiked at scene
  changes? One plot settles the relevance of §13, §19, §23 at once.
- **The `error(r, k)` surface.** Grid `r ∈ {5,10,20} × epochs ∈ {100,300,1000}`
  — the shape decides ladder (§11) vs boosting (§20) vs "just pick bigger `r`".
- **Noise-floor probe.** Fit `r=50`, 5000 epochs, no early stop. If error is
  still ~47 (`DECISIONS.md` §3), it is noise-limited (`DECISIONS.md` §16's
  `0.01·randn`) and every `k` / conditioning lever is chasing a floor it
  cannot beat.
- **Gradient-noise scale.** Full-batch loss today; one number says whether
  minibatching entries could help at all (§27(c)).

### (d) Gap + mitigation

**Gap.** 38 sections, ~6 applied levers, and the doc never says which *single*
measurement moves the most priorities. There is no "if you run one experiment"
answer.

**Mitigation.** Order: **M5 first** — it tells you whether the current state is
already write-up-ready and it is the §10-row-1 prerequisite everything else is
stated against. Then M7 and M1 (they fall out of the same runs). M2/M3/M4/M6
only if M5 shows a real gap worth closing. The noise-floor probe (§39(c))
before *any* `k` work — it can invalidate the entire convergence layer in one
run.

**Open question to argue about.** Has "synthesise the approaches" become a
substitute for running the six experiments that would collapse half of them?
The session's measured facts (bf16 1.63x, backward 75%, DNN half-inert) each
cost one script; the dozen synthesis sections cost comparable effort and
resolved nothing empirically. The next unit of work should be a `.py`, not a
`##`.

**Failure mode.** Treating this as a sequential plan: the composed re-run, the
`r×k` grid, and the `r=50 / 5000`-epoch probe are each multi-hour on a
thermally-throttling 4050 (§32(c)), so "an afternoon" is optimistic and some
results will need the workstation (`GPU.md` §4) to be trustworthy anyway.

---

## 40. First diagnostic run — M1/M2/M3 measured (RTX 4050, `random-tensor`, r=10)

Acting on §39: an instrumented 2000-epoch fit, `random-tensor` at Carphone
shape, one seed, fp32. `EarlyStop` fired at **epoch 300**. Logged rel-error,
`‖grad W1‖`, the balancedness residual `‖WᵀW−HHᵀ‖/‖WᵀW‖` (mean over slices,
factor pair 1), and `‖M1−M2‖/‖M1‖`:

| epoch | rel err | ‖gradW1‖ | balancedness | ‖M1−M2‖/‖M1‖ |
|---|---|---|---|---|
| 1   | 0.4499 | 1.13 | 0.049 | 0.079 |
| 25  | 0.3946 | 0.42 | 0.218 | 0.402 |
| 50  | 0.3775 | 0.15 | 0.245 | 0.709 |
| 100 | 0.3667 | 0.31 | 0.275 | 0.840 |
| 300 | 0.3627 | 0.46 | 0.170 | 0.888 |
| 1000| 0.3620 | 0.54 | 0.150 | 0.908 |
| 2000| 0.3620 | 0.46 | 0.227 | 0.916 |

### What it resolves

- **M1 — ~300 epochs is the knee, decisively.** Epoch 300 → 2000 is **6.7x
  the compute for 0.2% relative error** (0.3627 → 0.3620). `EarlyStop` already
  sits at the knee. The `DECISIONS.md` §11 "87% of epochs wasted" figure is
  real *at fixed k=2000* but early stopping already captures it — the whole
  convergence layer (§8.1 Anderson, §11 ladder, §17 homotopy, §22 Parareal)
  is fighting over ~0.2% of error below epoch 300. **§27's open question is
  answered: `k` is no longer the lever.** Per-epoch cost (§33) and everything
  else move up.
- **M2 — plateau gradient is large (~0.5) and oscillating, not vanishing.**
  Not a stationary point; a geometry the optimiser is fighting. The §29(d)
  "tiny gradient → not conditioning" branch does **not** fire.
- **M3 — the gauge drifts under Adam.** Balancedness goes 0.05 → 0.27 by epoch
  100, settles ~0.15–0.23. Not `<1e-3` (which would have killed the symmetry
  layer per §36(d)), not blowing up either — **moderate, persistent drift.**
  §1/§8.2/§18 target a mechanism that is genuinely active.
- **M2 + M3 are consistent** — oscillating gradient *and* moderate drift is the
  coherent "scaling symmetry causes ill-conditioning" picture. §36(b)'s
  contradiction case (large gradient *without* drift) did not occur.
- **§38's degenerate start self-corrects by ~epoch 25.** `‖M1−M2‖/‖M1‖`:
  0.08 → 0.40 (ep 25) → 0.89 (ep 300). §38's own failure-mode check confirms:
  the descent breaks `M1 = M2` within a few epochs, so the **asymmetric-init
  idea (§38(c)) is moot**.

### (a)/(b)/(c)/(d) in one pass

**Composability of the finding:** M1 + M3 together *reorder the document*. `k`
is settled, so the axes that still have headroom are per-epoch cost (§33 —
bf16 confirmed 1.63x, backward 75%), `p`-scaling for the *ratio* (§28/§30, no
speed claim per §28's open question), and evaluation-honesty (§35 — the
composed number still unmeasured). **Contradiction surfaced:** M3 says the
symmetry layer is legitimate, but M1 says its prize (error below epoch 300) is
~0.2% — so §1's `O((m+n)r^4)` cost (§29(b)) buys almost nothing *for
convergence*; its only remaining justification is the *connective* role (§36(a)
— gauge consistency for §11/§13/§19/§26). **Line:** this is a `k`-line result
that closes the `k` line. **Gap it exposes:** every number here is
`random-tensor`, one seed. The §12/§13/§21 image-vs-random split means Carphone
could show a lower knee (video is genuinely low-rank) or worse drift.

**Open question to argue about.** If `k` is settled at ~300 epochs and the
symmetry prize is ~0.2%, is *any* algorithmic section (§1, §8.1–8.3, §15, §17,
§21, §22, §25) worth building for HadNet as-configured — or is the entire
remaining opportunity in per-epoch cost (bf16, lower `r`) and honest
measurement (§35)? The doc may have been over-indexed on convergence.

**Failure mode.** One seed, one dataset, one shape, fp32. Carphone must be
re-run before M1's "knee at 300" is trusted for the paper's actual benchmark —
`random-tensor` has no low-rank structure to converge *to*, so its knee may be
artificially early.

---

## 41. M5 measured, and a basin probe that overturns §40's reading

Acting on §39's "M5 first". Built a **genuinely Hadamard-structured** signal
`X = (ab)∘(cd)` (iid Gaussian factors, rank 10) `/max + 0.01·randn`, so a
rank-10 Hadamard model is *exactly* rank-matched. 5 seeds, RTX 4050.

### M5 — the composed end-to-end speedup

| config | time | epochs | rel err |
|---|---|---|---|
| baseline (fp32, k=2000, exact init) | 4.97 ± 0.03 s | 2000 | 0.4859 ± 0.0016 |
| optimised (bf16 + `EarlyStop` + randomized init) | 0.46 ± 0.03 s | 300 | 0.4870 ± 0.0014 |

**Composed speedup 10.9x**, error cost +0.0011 (inside seed variance). This is
**2x higher than §35(a)'s Amdahl estimate of ~5.6x** — because early stopping
here is 6.7x (2000→300), not the 3x `DECISIONS.md` §12 measured under a
different patience. So: **`EarlyStop` 6.7x × bf16 1.6x ≈ 11x**, and `h_prod`
batching + randomized init contribute **nothing measurable at loop level** —
§35(a)'s Amdahl prediction confirmed exactly.

### The basin probe — the real finding

The rel err plateaus at **0.486** for *both* 300 and 2000 epochs. But the
noise floor for this signal is **0.180** (`0.01·√N / ‖X‖`), and:

| setup | rel err |
|---|---|
| **true factors as init** (no training) | **0.180** — model represents the signal exactly |
| true-factor init + 1000 ep Adam(0.01) | 0.156 |
| SVD-of-`sqrt` init + 6000 ep Adam(**0.1**) | 0.483 |
| SVD-of-`sqrt` init + 6000 ep Adam(0.03) | 0.484 |
| SVD-of-`sqrt` init + 6000 ep Adam(0.01) | 0.484 |

**From the `svd_initialize` basin, Adam cannot reach the solution** — it sits
at 0.48 while the optimum is 0.18, and 6000 epochs / three learning rates do
not move it. The plateau is **not a converged tail; it is a trap ~2.7x worse
than optimal.**

### What this overturns

- **§40's M1 reading was wrong.** "300 epochs ≈ 2000 epochs" is true, but it
  means *stuck*, not *converged* — `DECISIONS.md` §11's "87% of epochs wasted"
  is better read as "87% of epochs spent not escaping a bad basin". `k` being
  "settled" does **not** mean the convergence/basin layer is low-value; it
  means the opposite.
- **§38's init line is vindicated hard.** The `sqrt(|X|)` init is the
  difference between rel 0.18 and rel 0.48 here. §38's "no init / random init"
  and "additive fit (§17) as init" and "asymmetric `M1≠M2`" tests move to the
  **top** of the experiment queue.
- **§17 (homotopy), §11 (ladder), §1/§8.2/§18 (basin geometry), §21 (AMP's
  spectral init) all regain priority** — they are basin-escape methods, and a
  basin trap is now measured, not hypothesised.

### (a)/(b)/(c)/(d)

**Composability:** §38 (better init) + §17 (additive-fit init) + §11 (ladder
init) now compose into "the thing to try", because M5 shows the *applied* stack
(bf16 + early stop) is done — 11x, no maths risk — and the *next* factor of 2+
is only reachable by leaving the basin. **Contradiction:** §40 concluded "doc
over-indexed on convergence"; this section concludes the reverse. The
resolution: `k`-*count* is settled (§40 M1 stands — more epochs don't help),
but `k`-*quality* (which basin those epochs run in) is wide open. **Line:**
init/basin — the highest-value axis after all. **Gap:** every number here is
one synthetic construction; `sqrt(|X|)` init is *designed* for non-negative
smooth data (video), and signed Gaussian-factor products are close to its
worst case.

**Open question to argue about.** Does the basin trap reproduce on **Carphone**
(non-negative, smooth, what `sqrt(|X|)` init targets), or is it an artefact of
the signed-Gaussian synthetic signal? If Carphone also plateaus far from its
achievable optimum, the init/basin line is the whole paper; if Carphone
converges fine, this section is a synthetic-data curiosity and §40's reading
stands.

**Failure mode.** The synthetic signal's factors are iid Gaussian — heavy
Hadamard-product tails, sign-random, `E[X]=0`. `svd_initialize` takes
`sqrt(|X|)`, which for a zero-mean signed signal discards exactly the structure
that matters. A fairer probe uses a non-negative low-rank-times-low-rank signal
plus noise; run that before generalising.

---

## 42. The Carphone-analog run — the trap is synthetic, but `lr=0.1` walks away from a good init

§41's next step, run. Three signals, `svd_initialize` (exact), Adam, RTX 4050:

| signal | init rel | final rel | epochs | note |
|---|---|---|---|---|
| [A] non-neg **additive** rank-15, `r=10` model | 0.0417 | **0.0687** | 3000 | training made it **worse** |
| [B] non-neg **Hadamard** rank-10, `r=10` (matched), lr=0.1 | 0.0824 | **0.1341** | 3000 | training made it **worse** |
| [B] same, **lr=0.03** | 0.0824 | 0.0818 | 3000 | stable, ~no change |
| [C] **real video** (`video.mp4`, 144×144×60×3), lr=0.1 | 0.1386 | **0.0645** | 2000 | training **works**, halves the error |

### What it settles

- **§41's basin trap is a synthetic artefact.** On real video, training does its
  job — init 0.139 → 0.065, consistent with `DECISIONS.md` §3's HadNet regime.
  §40's "more epochs don't help ≈ converged" reading is closer to right *for
  the actual application* than §41's "stuck" reading.
- **But `lr=0.1` is measurably too large.** On both structured synthetics it
  *degrades* the SVD init (A: 0.042→0.069, B: 0.082→0.134); `lr=0.03` stops
  the damage. Real video survives `lr=0.1` by luck of being far enough from its
  optimum that even a too-big step is net-downhill. This is direct evidence for
  `DECISIONS.md` §9's "`lr=0.1` magic constant" being not just unprincipled but
  *wrong* in some regimes.
- **The SVD init already does most of the work when the signal is structured**
  ([A] init 0.042 is near the 0.022 noise floor before any training).

### (a) What composes

A gentle/adaptive step + a trusted init compose: the init lands close, a small
step polishes without leaving. **§16 (better init) + §8.1 (Anderson, step-size
robust) + `squared_loss` (§11/§14, well-behaved curvature) + `lr≈0.03`** is a
coherent bundle — "trust the init, step carefully". §12 `EarlyStop` composes on
top and, per [A]/[B], should sometimes fire at **epoch 0**.

### (b) What contradicts

1. **§11 (ladder) / §17 (homotopy) assume the descent is the workhorse and the
   init is a seed.** [A]/[B] show the reverse for structured data — the init is
   the workhorse and `lr=0.1` descent is a liability. Building elaborate init
   *schedules* on top of a descent that walks downhill-then-back-uphill is
   effort in the wrong place until the step size is fixed.
2. **§8.1 (Anderson) accelerates the fixed-point iteration — including a
   divergent one.** If the underlying Adam step moves *away* from the init
   ([A]/[B]), Anderson makes it move away faster. Anderson needs the base
   iteration to be descent first; applying it before fixing `lr` is actively
   harmful.
3. **§40's M1 vs [C].** M1 (`random-tensor`) said the knee is ~300 epochs; [C]
   (real video) improves out to 2000. The knee is **data-dependent** — the
   §12/§13/§21 image-vs-random split, now confirmed on the convergence curve
   itself.

### (c) Further approaches on this line (step size × init)

- **Backtracking / Armijo line search on `lr`.** One or two extra forward
  evals per step; removes the magic constant (`DECISIONS.md` §9) with a
  guarantee the step is downhill — which [A]/[B] show the fixed `0.1` is not.
- **`lr` warmup then cosine decay** — start at `~0.01`, rise, decay. Standard,
  and it would have caught [B]'s early divergence.
- **Per-block `lr`.** After the `sqrt`-split the `W` and `H` blocks have
  different scales (§38); one global `lr` over-steps one of them.
- **Init-aware early exit.** If `init_rel < τ`, train for a handful of epochs
  or not at all. [A] would stop at epoch 0 and beat its own 3000-epoch run.
- **Plain SGD+Nesterov vs Adam.** `DECISIONS.md` §11 argued Adam's adaptivity
  competes with geometry methods; [A]/[B] show Adam(0.1) also competes with a
  good init. A momentum-SGD baseline at matched tuning would isolate it.

### (d) Gap + mitigation

**Gap.** No table in `DECISIONS.md` or this doc reports **init error next to
final error**. "Training helps" is assumed everywhere and never checked; [A]/[B]
show it can measurably hurt.

**Mitigation.** Add an `init_rel` column to every result table
(`DECISIONS.md` §3, Table 9, all folders). Any row with `final ≥ init` is a
net-negative training run and must be flagged — and re-run at lower `lr`.

**Open question to argue about.** Is a chunk of HadNet's reported advantage
over TBCD/TAGD (`DECISIONS.md` §3) actually just **a good SVD-of-`sqrt`
initialisation** rather than the 2000-epoch descent? `DECISIONS.md` §4.4
already suspects the iteration-count mismatch drives the comparison; reporting
`init_rel` as its own baseline row would show how much of the gap the init
alone closes.

**Failure mode.** [C] real video genuinely benefits from training, so
"trust the init, skip the descent" is wrong for the actual application. The
synthetic degradation may not generalise — run `init_rel` vs `final_rel` on all
of MRI / cat / flower / monalisa / video before concluding `lr=0.1` is broadly
broken.

---

## 43. `init_rel` vs `final_rel` on all four image datasets — the descent is data-dependent

§42's failure-mode check, run: all four `application-images` datasets, `r∈{10,20}`,
`lr∈{0.1,0.03}`, 1500 epochs, exact init, RTX 4050.

| dataset | init rel (r=10) | final (lr=.1) | final (lr=.03) | training closes |
|---|---|---|---|---|
| **flower**   | 0.145 | 0.140 | 0.140 | **~3%** of init |
| **monalisa** | 0.110 | 0.106 | 0.106 | **~4%** |
| **cat**      | 0.169 | 0.106 | 0.102 | ~40% |
| **mri**      | 0.292 | 0.133 | 0.126 | ~57% |

(r=20 same pattern: flower/monalisa barely move, cat/mri improve a lot.)

### What it settles

1. **`lr=0.1` is not broadly broken.** Training helps on all 4 real images +
   video (§42[C]); §42's synthetic degradation ([A]/[B]) does **not**
   generalise. §41/§42's "trap" is a signed-Gaussian synthetic artefact.
2. **`lr=0.03` beats `lr=0.1` in 16/16 rows**, by ~5–10% on the harder sets
   (mri, cat). `DECISIONS.md` §9's magic constant is mis-set high — a free
   small win, no risk.
3. **The descent's value is data-dependent, and predictably so.**
   Flower/monalisa (smooth, low-frequency) — the SVD-of-`sqrt` init already
   captures ~96% and 2000 epochs add ~3%. Cat/mri (high-frequency detail) —
   the init misses it, training recovers 40–57%. Maps onto §9.11 (spectral
   bias) / §12 (multigrid): the descent's job is the high frequencies the
   rank-`r` init cannot place.

### (a) Composability

An **init-quality gate** (§42(c)) + **per-dataset `lr`** compose into: measure
`init_rel`; if `< τ` skip training (flower/monalisa → stop at epoch 0 for a ~3%
cost); else train at `lr≈0.03`. This is `EarlyStop` (§12) taken to its limit —
fire before epoch 1 — and it composes with everything downstream because it
just changes when the loop exits.

### (b) Contradictions

1. **The rank sweep's time column (`DECISIONS.md` §3) vs the descent being
   near-useless on smooth images.** For flower/monalisa, HadNet's *reported
   time* is ~2000 epochs of compute producing a result the init already had.
   The paper's time for those datasets overstates the *necessary* cost by
   20–30x — and any speedup claim (bf16, `h_prod`) on that compute is speeding
   up work that need not happen.
2. **§11 (ladder) / §17 (homotopy) / §8.1 (Anderson) all assume the descent
   matters.** For half the real datasets it barely does; elaborate
   convergence machinery there is optimising a near-no-op.
3. **§40's M1 ("~300 epoch knee") vs mri/cat** — those keep improving well past
   300 (mri r=20 best at ~0.053 needs the full run). Knee is data-dependent,
   again — flower/monalisa knee ≈ epoch 50, mri knee ≈ epoch 1000+.

### (c) Further approaches on this line

- **Report `frac_closed = (init − final) / (init − noise_floor)` per dataset.**
  One number that says whether the descent earned its compute.
- **Route by spectral content.** Cheap FFT/DCT energy ratio of the input →
  predicts whether this dataset is init-dominated or descent-dominated →
  picks epoch budget and `lr` before training.
- **Return the best-seen factors, not the last.** `HadamardNet_train`
  (`main_video.py:104`) returns `model.W1` at loop exit; `EarlyStop` tracks
  `best` as a float but snapshots no parameters. cat r=10 final 0.106 vs
  best 0.102, mri r=20 final 0.060 vs best 0.059 — HadNet ships a ~2–3% worse
  result than it transiently reached. Free fix: snapshot best params.
- **Two-phase `lr`:** `0.1` for the first ~100 epochs (fast on the
  init-dominated part), drop to `0.01` (the detail-recovery phase mri/cat
  need). A schedule matched to the two-regime structure the data shows.

### (d) Gap + mitigation

**Gap.** `DECISIONS.md` §3 / Table 9 have no `init_rel` row, so a reader cannot
see that for **half the benchmark datasets the iterative part contributes
almost nothing** — the method there is effectively a closed-form
initialisation formula.

**Mitigation.** Add `init_rel` and `frac_closed` to every table. Datasets with
`frac_closed < 0.15` get flagged "init-dominated"; for those the paper either
drops the descent or states why it is kept. This also directly answers §42's
open question per-dataset.

**Open question to argue about.** For the init-dominated datasets, is HadNet's
contribution an **initialisation formula** — truncated SVD of `sqrt(|X|)` with
a scale split, `O(n^4)`, closed form — rather than a training method? If that
formula alone beats TBCD/TAGD on flower/monalisa (untested), that subset wants
a separate, cleaner claim and no epoch loop at all.

**Failure mode.** `frac_closed` depends on the noise-floor estimate, and
`DECISIONS.md` §16 shows the injected `0.01·randn` puts the target itself out
of range — so the "floor" is fuzzy and `frac_closed` near the floor is
unreliable. Report `init` and `final` raw alongside it; do not let the derived
ratio hide the inputs.

---

## 44. BCD vs HadNet from the same init — the deficiency is Adam, not the model

§43's open question, run. Same 4 images + `0.01·randn`, `r∈{10,20}`:
plain TSVD of `X`, the Hadamard init alone, HadNet (1500 ep, `lr=0.03`), and
`Hadamard_BCD` — which uses **`SVD_initialization` + exact block least-squares**
(`hadamardLS_exact`), 100 iters.

| dataset | r | TSVD | Had-init | HadNet | **BCD** |
|---|---|---|---|---|---|
| flower | 10 | 0.141 | 0.145 | 0.140 | **0.086** |
| flower | 20 | 0.096 | 0.100 | 0.099 | **0.041** |
| monalisa | 10 | 0.107 | 0.111 | 0.106 | **0.071** |
| monalisa | 20 | 0.075 | 0.077 | 0.074 | **0.047** |
| cat | 20 | 0.111 | 0.116 | **0.072** | 0.076 |
| mri | 20 | 0.177 | 0.191 | **0.054** | 0.060 |

### What it settles

1. **The Hadamard init is *worse* than plain TSVD** on every row (flower r=10:
   0.145 vs 0.141; mri r=20: 0.191 vs 0.177). §43's "HadNet ≈ init formula"
   open question is dead — the formula loses to truncated SVD, so "just use the
   init" means "use something worse than the trivial baseline".
2. **On smooth images HadNet ≈ TSVD** (flower/monalisa: HadNet within 0.001 of
   TSVD). The whole apparatus — sqrt-split init + 1500 epochs of Adam —
   **matches plain truncated SVD** there.
3. **BCD reaches ~2x lower error than HadNet on smooth images** (flower r=20:
   0.041 vs 0.099), from **the same SVD init family**. It proves the rank-10
   Hadamard model *can* fit flower to 0.041; Adam-from-the-sqrt-basin does not.
   The deficiency is the **optimiser**, isolated — not the model (BCD fits it),
   not the init (BCD shares it), not the data.
4. **HadNet edges BCD on high-detail images** (cat/mri r=20). So it is not
   uniformly worse — the split is smooth→BCD, detailed→HadNet.

This revives §41's basin-trap in a **mild, real-data** form: §42 called the
trap synthetic; here BCD finds a 2x-better basin on real images that HadNet's
Adam misses.

### (a) Composability

**BCD's exact block-LS update + HadNet's systems levers.** §15 (Schur
complement) is precisely how to make batched block-LS fast on GPU; §15 +
"BCD wins on half the real data" is now a concrete plan, not a hypothesis. bf16
(1.63x) and `h_prod` batching compose with a block-LS inner loop the same way
they compose with Adam. `EarlyStop` (§12) composes on the outer BCD sweeps.

### (b) Contradictions

1. **`DECISIONS.md` §3 ranks BCD "5x slower, use HadNet"** — but at **2x lower
   error on smooth data**. "Slower" and "worse" were assumed to coincide; they
   do not. No error-vs-wall-clock frontier exists to say which dominates where.
2. **`DECISIONS.md` §11 rejected Lipschitz block-stepping as "worse on both
   axes"** — but that was a *joint* `1/L` step. True cyclic BCD
   (`hadamardLS_exact`, what `Hadamard_BCD` runs) is the regime §11 itself
   names as "where `1/L` is justified", and it *wins on error* here. §11's
   rejection does not extend to the method actually measured.
3. **§8.1 (Anderson) accelerates convergence to whatever basin Adam finds** —
   on flower that basin is 2x worse than BCD's. Speeding the wrong destination.

### (c) Further approaches on this line (optimiser, not `k`)

- **Hybrid: a few BCD sweeps, then Adam polish** (or Adam first, BCD to finish).
  Cheap to try against the existing `Hadamard_BCD`.
- **BCD warm-start for HadNet** — run 5–10 BCD sweeps as the init, then the
  normal loop. Tests whether Adam can *hold* a good basin even if it cannot
  *find* one.
- **GPU-batched exact block-LS** (§15) — the only reason BCD is "5x slower" is
  per-slice `scipy` solves (`DECISIONS.md` §4.4); batched on GPU it may not be.
- **`hadamardLS_lipschitz`** (already in the repo; `lr = 1/σ²_max`,
  `DECISIONS.md` §9) as a middle ground between exact-LS and Adam.

### (d) Gap + mitigation

**Gap.** `DECISIONS.md` §3 compares methods at their *default* iteration counts,
never on an error-vs-wall-clock frontier. "BCD: 5x slower, lower error" and
"HadNet: faster, higher error" are incomparable without it.

**Mitigation.** Run BCD and HadNet at 4–5 iteration budgets each, on the same
data and GPU, plot rel-error vs wall-clock. The frontier shows which method
dominates at each time budget — and whether BCD's smooth-image advantage
survives a matched-time comparison.

**Open question to argue about.** If BCD (classical alternating least squares)
reaches 0.041 on flower where HadNet (Adam + epochs + `lr`) reaches 0.099 from
the same init, is the "neural" framing (the `HadNet`/`HadDNN` naming, the
training loop) a **step backward from ALS** for this problem? The contribution
may be the `r²`-expressive *parameterisation*, with the *optimiser* choice
being a regression.

**Failure mode.** This is single 128×128 images; BCD's per-slice LS solves
were not timed on the *tensor* case, where `DECISIONS.md` §3 has TBCD at 283 s
vs HadNet 57 s (5x). On video the 180× per-slice solves could dominate and
erase BCD's error advantage on a time basis. Run the tensor frontier before
concluding BCD is the better optimiser generally.

---

## 45. Adam is unstable *in* a good basin — the warm-start hybrid fails

§44(c)'s "BCD warm-start for HadNet", run. Take BCD's converged factors, hand
them to Adam (`lr=0.03`), watch what happens. `r=20`, images + `0.01·randn`:

| dataset | pure Adam | BCD | BCD→Adam (200 ep) | BCD→Adam (+1300 ep) |
|---|---|---|---|---|
| flower   | 0.099 | **0.041** | 0.054 | 0.045 |
| monalisa | 0.074 | **0.047** | 0.062 | 0.058 |
| mri      | **0.054** | 0.060 | 0.116 | 0.108 |

### What it settles

**Adam does not hold a good basin — it walks out of one.** Handed BCD's 0.041
solution on flower, Adam degrades it to 0.054 in 200 epochs and never returns
to 0.041 over 1300 more. On **mri it is catastrophic**: BCD's 0.060 → Adam
**0.116** (2x worse), clawing back only to 0.108. And note pure-Adam-from-SVD
reaches 0.054 on mri — so Adam is *fine* from the sqrt-init basin and *blows
up* from the BCD basin. The two basins are different, and `lr=0.03` is only
stable in one of them.

- **§44(c)'s "BCD warm-start for HadNet" is dead.** Adam undoes the warm start.
  The only viable hybrid direction is **Adam → BCD** (BCD finishing), not the
  reverse.
- **This is the sharpest evidence yet that `lr` is the core deficiency**
  (`DECISIONS.md` §9). §43 showed `lr=0.1` slightly too big; §45 shows `lr=0.03`
  is *unstable near good solutions*. The method needs a decaying / line-searched
  step (§42(c)) or a non-gradient update (BCD, §15, §25, §1).

### (a) Composability

A **decaying-`lr` / line-search (§42(c)) + an exact-block-LS finisher (§15) +
`squared_loss`** compose into "Adam for the cheap early descent, a stable
method for the endgame". The finisher can be BCD, GPU-batched block-LS (§15),
or simply Adam with `lr → 0`. What provably does **not** compose: BCD → Adam
(measured broken here), and §8.1 Anderson on Adam near a good point
(accelerates the walk-out).

### (b) Contradictions

1. **§44(c) proposed BCD-warm-start; §45 kills it.** Same section pair, one
   run apart.
2. **`DECISIONS.md` §11's reason for rejecting Lipschitz stepping** — "Adam
   already adapts per-parameter from gradient second moments, so dividing by
   `L` duplicates that and over-damps" — is contradicted here: near the BCD
   basin on mri Adam **under**-damps and diverges. The per-parameter second
   moments are **not** supplying the stability a proper step-size bound would.
   §11's rejection assumed an adaptivity that §45 shows fails exactly where it
   is needed.
3. **§40's M1 ("~300-epoch knee, `k` settled")** — on mri, Adam from a good
   basin gets *worse* for 200 epochs then partially recovers over 1300. "More
   epochs don't help" hides "more epochs first hurt, then repair".

### (c) Further approaches on this line (stability, not speed)

- **Cosine `lr` decay to ~0.** Would let Adam settle rather than orbit; standard,
  untried here.
- **Per-step trust region on `‖ΔW‖ / ‖W‖`.** Cap the relative move; cheap, and
  directly targets the mri blow-up.
- **Gradient clipping.** One line; may tame the 0.060 → 0.116 excursion.
- **Adam → SGD(+momentum) switch near convergence.** SGD has no second-moment
  denominator to misbehave when gradients get small and anisotropic.
- **Restart Adam's `m`/`v` buffers on any loss increase.** Stale moment
  estimates from the descent phase may be driving the endgame instability.
- **Exact-block-LS finisher** (§15) invoked once `EarlyStop` (§12) fires —
  BCD's update, but only for the last few sweeps.

### (d) Gap + mitigation

**Gap.** No experiment has measured the landscape curvature at the *two*
distinct regions this section exposes — the sqrt-init basin (Adam stable) vs
the BCD optimum (Adam unstable). §29(d) proposed a Lanczos `κ` at "the plateau"
as if there were one.

**Mitigation.** Compute the top/bottom Hessian eigenvalues (a few HVPs) at
**both** points, for mri specifically. If `κ` at the BCD optimum is orders
larger, the loss-optimal solution sits in a region no fixed-`lr` first-order
method can occupy — which reframes HadNet's achievable error as "wherever Adam
is stable", generally above the true optimum.

**Open question to argue about.** Is the true minimiser **unreachable** by any
fixed-`lr` first-order method for datasets like mri, so that BCD (exact LS, no
gradient step) is not just faster-to-good-error but *the only way* to the
actual optimum? If so, every gradient-based section (§1 as gradient method,
§8.1, §17, §22, §33) is capped below BCD by construction.

**Failure mode.** BCD's factors go CPU-numpy → GPU-tensor and BCD runs its own
`scaling()` rebalancing — so the handoff to Adam carries a **gauge** different
from `svd_initialize`'s (§18). The mri blow-up could be partly a gauge shock,
not pure `lr` instability. Re-run with the BCD output balanced (§8.2 / §18)
before Adam to separate the two causes.

---

## 46. The BCD basin is 155,000x more curved — gauge imbalance *is* the conditioning

§45's failure-mode check + §45(d)'s curvature probe, both run, on mri, `r=20`,
squared loss.

| quantity | value |
|---|---|
| top Hessian eigenvalue at the **SVD-init basin** (Adam converges here) | 6.6e2 |
| top Hessian eigenvalue at the **BCD optimum** (Adam diverges from here) | 1.0e8 |
| **ratio** | **~1.5e5x** |

BCD→Adam(`lr=0.03`) on mri, with the fixes §45(c) proposed:

| handoff | rel err |
|---|---|
| plain (§45) | 0.081 |
| **gauge-balanced first** (`‖W‖=‖H‖` per pair) | **0.056** |
| gradient clipping (1.0) | 0.102 (worse) |
| cosine `lr` decay | 0.078 |
| **`lr` 0.03 → 0.003** | **0.059** |

### What it settles

- **The BCD optimum sits in a region five orders of magnitude more curved than
  the SVD-init basin.** A step size safe at `λ_max ≈ 660` is catastrophic at
  `λ_max ≈ 1e8`. This is the *measured, quantified* cause of §45's blow-up —
  not a mystery, a curvature mismatch.
- **§45's "`lr` is the core deficiency" was one level too shallow.** The root
  is **gauge**: balancing `‖W‖=‖H‖` moves to a flat equivalent point and Adam
  is immediately stable (0.056, beating BCD's own 0.060). Dropping `lr` 10x
  also works (0.059) — because a smaller step tolerates the curvature. Both
  fixes address the same 1e8-vs-660 fact from opposite ends.
- **Gradient clipping fails** (0.102) — clipping the *direction*'s norm does
  nothing about a step that is scale-wrong for the local curvature.

### (a) Composability

**Gauge-balance (§8.2/§18) now makes §44's "Adam → BCD → Adam-polish" pipeline
viable** where raw BCD→Adam was broken (§45): `Adam early → BCD sweeps →
balance → Adam polish`. §1 (Riemannian tuning-free Gauss-Newton step) and §8.3
(K-FAC) would both auto-absorb the 1.5e5x ratio — §46 is the measured reason
they matter, upgrading them from "principled" to "addresses a quantified
failure". `hadamardLS_lipschitz` (already in the repo, `lr = 1/σ²_max`) is the
cheap version and needs no new code.

### (b) Contradictions

1. **§40's M3 called the balancedness drift (~0.2) "moderate".** §46 shows a
   gauge that off can coincide with a `1.5e5x` curvature blow-up. "Moderate
   drift" understated it — the metric (`‖WᵀW−HHᵀ‖/‖WᵀW‖`) is not linear in the
   conditioning damage.
2. **§36 filed gauge-fixing as "connective tissue — do it once at the end".**
   §46 says it may need to be **frequent**, because curvature blows up locally
   as the iterate moves, not just at the final point.
3. **§45 concluded "`lr` is the deficiency"; §46 refines to "gauge is, `lr` is
   its symptom".** A single global `lr` cannot serve a `[660, 1e8]` curvature
   range no matter its value — the fix is to flatten the range (balance) or
   track it (`1/λ_max` step), not to pick a better constant.

### (c) Further approaches on this line

- **Log `λ_max` along the whole Adam trajectory**, not just endpoints — does it
  spike mid-training then relax, and does that coincide with the §11 plateau?
- **Balance every `N` epochs** (§36's checkpoint idea, now with a measured
  motivation) — keep `λ_max` from drifting up.
- **`lr_t = c / λ_max,t`** with `λ_max` from one power iteration every few
  epochs — the adaptive version; `hadamardLS_lipschitz` already computes the
  ingredient.
- **Report `κ` (or `λ_max`) as a standard diagnostic column** next to rel-error
  in every experiment.

### (d) Gap + mitigation

**Gap.** `DECISIONS.md` measures error, time, PSNR, SSIM — **never curvature**.
The entire plateau / conditioning discussion (§1, §11, §29) has been
qualitative. One HVP power iteration (~30 lines, ran this session) yields
`λ_max`.

**Mitigation.** Log `λ_max` every ~100 epochs for the standard HadNet fits on
all datasets. A spike marks where a balance step or `lr` drop is needed; a flat
low `λ_max` throughout means the conditioning story does not apply to that
dataset and §1/§8.2/§18 can be skipped for it.

**Open question to argue about.** Is the SVD-of-`sqrt` init "good" only because
it starts in a **low-curvature** (`λ_max ≈ 660`) region where Adam is stable —
i.e. its value is *conditioning*, not *approximation quality* (§44 showed it is
a worse approximation than plain TSVD)? If so, "better init" (§38) should mean
"lower-curvature init", a different objective from "lower-error init", and the
two may not point the same way.

**Failure mode.** `λ_max` from 30 power iterations from a random start may not
have converged at `1e8`; and this is the **squared**-loss Hessian, while the
repo minimises the **unsquared** norm (`DECISIONS.md` §11), whose Hessian
blows up separately as `‖R‖→0`. Re-measure with more iterations and on the
actual objective before treating `1.5e5x` as exact — the direction is robust,
the magnitude may not be.

---

## 47. `λ_max` along a healthy HadNet run is flat and low — the 1.5e5x is a BCD-optimum property, not a trajectory one

§46(c)'s "log `λ_max` along the whole trajectory", run. mri, `r=20`, SVD init +
Adam(`lr=0.03`), `λ_max` and balancedness every 150 epochs, both loss forms.

| epoch | rel | `λ_max` (unsq) | `λ_max` (sq) | balance |
|---|---|---|---|---|
| 0    | 0.191 | 1.1e1 | 4.0e2 | 0.00 |
| 150  | 0.059 | 4.6e1 | 4.9e2 | 0.19 |
| 600  | 0.056 | 5.7e1 | 5.8e2 | 0.20 |
| 1200 | 0.054 | 6.3e1 | 6.4e2 | 0.24 |

### What it settles

- **On a normal HadNet run, `λ_max` never spikes** — it drifts `11→63`
  (unsquared) over 1200 epochs and sits flat. The `1e8` of §46 is a property of
  **BCD's optimum specifically**, not a place the Adam trajectory visits. So
  "gauge imbalance is the conditioning" (§46) explains **why BCD's better basin
  is Adam-unreachable**, *not* why HadNet's own run plateaus.
- **The mri plateau (0.059 → 0.054 over 1000+ epochs) happens at flat, low
  curvature.** `λ_max ≈ 60` → a stable step is `~1/60 ≈ 0.017`; `lr=0.03` is
  mildly over that (consistent with §43's "0.03 beats 0.1", and the mild
  wobble), but curvature is not the plateau's cause.
- **92% of the error drop is in the first 150 epochs** (0.191 → 0.059); the
  rest of the run is a slow low-curvature crawl. Matches §40's M1 "~300-epoch
  knee" and §43's data-dependence.
- **Wrong dataset for the question.** mri is where HadNet *beats* BCD
  (0.054 vs 0.060, §44). The interesting case — where HadNet is stuck at 0.099
  vs BCD's 0.041 — is **flower**, and its `λ_max` trajectory was not measured.

### (a) Composability

§47 **bounds** where the conditioning fixes apply. §8.2 / §18 / §1 / §8.3 add
nothing to a standalone HadNet run on a dataset where it already converges at
low `λ_max` (mri) — they only compose with the **Adam→BCD→balance→polish**
pipeline (§46(a)) to reach the high-curvature better basins on the datasets in
§44's "smooth, BCD wins" class. Conditioning work is **dataset-conditional**.

### (b) Contradictions

1. **§40's M2** inferred "oscillating gradient at the plateau ⇒ conditioning".
   §47 measures **low, flat `λ_max`** at that same plateau — M2's inference was
   wrong; the oscillation is not curvature (shallow non-convex valley, or the
   unsquared-norm kink, though `‖R‖≈55` is far from 0 so not that either).
2. **§46 "gauge imbalance *is* the conditioning".** §47 shows balancedness
   `≈0.23` **and** low `λ_max` co-occurring on the normal run — so a given
   imbalance is benign here and catastrophic at the BCD point. The
   imbalance→curvature link is **not monotonic**; location matters as much as
   magnitude.
3. **§29's premise** ("the landscape is ill-conditioned"): on mri, for the
   trajectory Adam actually takes, the measured answer is **no**.

### (c) Further approaches on this line

- **Run §47's trajectory on flower and monalisa** — the HadNet-stuck datasets.
  This is *the* missing measurement.
- **Measure `λ_min`, not just `λ_max`.** `κ = λ_max/λ_min`; `λ_max ≈ 60` with
  `λ_min ≈ 1e-4` is still `κ ≈ 6e5`. Inverse iteration or Lanczos both ends.
- **Directional curvature along the update step** — `dᵀHd / dᵀd` for the actual
  Adam direction `d`. Cheaper than the spectrum and more relevant to whether
  the *step* is stable.
- **Loss-vs-`lr` sweep at the plateau** — freeze at epoch 600, sweep `lr`,
  plot loss after 50 steps. The empirical stability threshold, no HVPs.

### (d) Gap + mitigation

**Gap.** Every curvature number in the doc (§46, §47) is **mri** — the one
dataset where HadNet already wins. The measurement was run where there is no
problem to diagnose.

**Mitigation.** Re-run the §47 trajectory (`λ_max`, `λ_min`, balancedness,
rel-error every ~100 epochs) on **flower** and **monalisa**. If `λ_max` or `κ`
climbs on those while rel-error sticks at ~0.099, the trajectory itself hits a
conditioning wall and §8.2/§1/§8.3 are justified there. If `λ_max` stays flat
and low like mri, the flower plateau is a **low-curvature bad basin** and the
fix is basin escape (§17 homotopy, multi-start, BCD-finish), not
preconditioning.

**Open question to argue about.** Is the flower plateau (HadNet 0.099 vs BCD
0.041) a low-curvature *bad basin* or a high-curvature *trajectory wall*? §46
assumed the latter for BCD's optimum, but the two diagnoses call for opposite
families of fix — basin escape vs preconditioning — and one flower `λ_max` run
settles which.

**Failure mode.** `λ_max` alone is not `κ`. mri's flat `λ_max ≈ 60` is
consistent with both "well-conditioned" and "`λ_min` is tiny and Adam's
per-coordinate scaling is hiding it while still capping final error". `λ_min`
must be measured before "mri is not a conditioning problem" is stated as fact.

---

## 48. Flower/monalisa: a benign-curvature bad basin — the conditioning layer is retired for the trajectory

§47(d)'s run: `λ_max`, `λ_min`, `κ`, balancedness over a full HadNet fit on the
two datasets where it is stuck, plus an `lr` sweep at the plateau. Unsquared
loss, `r=20`.

| dataset | plateau rel | BCD rel | `λ_max` | `λ_min` | `κ` | `lr` sweep at plateau (0.1→0.001) |
|---|---|---|---|---|---|---|
| flower   | 0.095 | **0.041** | ~28 | ~0.04 | **~700** | 0.096 / 0.095 / 0.095 / 0.095 / 0.095 |
| monalisa | 0.074 | **0.047** | ~35 | ~0.06 | **~600** | 0.078 / 0.074 / 0.074 / 0.074 / 0.074 |

Flat `κ ≈ 700` for the entire run; `lr` across a 100x range gives the same
plateau loss.

### What it settles — definitively

**The flower/monalisa plateau is a low-curvature bad local minimum, not a
conditioning problem.** `κ ≈ 700` is well-conditioned; the `lr` sweep being
dead flat across 100x rules out any step-size or stability cause (two
independent methods agree). Adam has converged to a stationary point of the
loss that is **2x worse than BCD's**, and the landscape there is benign.

- **§8.2 / §18 / §1 / §8.3 / §29 / §36 will not help the trajectory.** They
  precondition or re-gauge an already-benign landscape (`κ 700 → 700`). §46's
  `1e8` was real but it is BCD's *destination*, never a point on HadNet's path
  (§47) — so gauge-fixing the trajectory fixes nothing.
- **§40's M2 ("conditioning") and §29's premise are now measured false** for
  the path Adam takes, on every dataset probed (mri §47, flower/monalisa §48).
- **The problem is basin escape.** §17 (homotopy / graduated non-convexity),
  §11 (rank ladder), multi-start, §20 (boosting), and BCD itself — methods
  that change *which* stationary point is reached — are what is left.

### (a) Composability

Basin-escape methods compose trivially because they all wrap the existing loop:
**multi-start + best-of-N + §17 (homotopy) + §11 (ladder) + §12 (`EarlyStop`
per restart)** — run the fit several ways, keep the best; memory is free
(`GPU.md` §6.5) so N restarts run in parallel. What does **not** add value:
stacking two conditioning fixes (§8.2 + §8.3) when `κ ≈ 700`.

### (b) Contradictions

1. **The doc's "Suggested order (revised)" lists Riemannian / balancedness /
   gauge as items 2, 3, 6.** §48 measures the trajectory as `κ ≈ 700` on every
   dataset — those items fix a non-problem *for the run itself*. The order
   needs rewriting around basin escape.
2. **§46 vs §48.** §46's `1e8` justified gauge-fixing; §48 shows the trajectory
   is `κ 700`. Consistent only if stated precisely: §46 was about *BCD's
   optimum*, a point Adam never visits, so it does not motivate anything
   applied to the HadNet run.
3. **§8.1 (Anderson) on flower** — accelerates convergence to a stationary
   point that is already reached by epoch 300 (§48 table) and is 2x too high.
   Faster arrival at a bad basin.

### (c) Further approaches on this line (basin escape, not preconditioning)

- **Random / perturbed restarts, best-of-N.** Fit flower from 10 varied inits
  (or 10 noised copies of the SVD init); keep the best. Embarrassingly
  parallel.
- **Basin hopping.** Converge, add noise, re-descend, repeat — keep the best
  minimum seen.
- **§17 homotopy with the additive model as `γ=0`** — the measured
  benign-curvature bad-basin case is exactly graduated non-convexity's target.
- **Population-based / evolutionary factor search** over a handful of candidates.
- **Just use BCD.** It is the measured winner on this data class (§44) and is
  already in the repo; its alternating exact-LS structure reaches basins Adam
  does not.

### (d) Gap + mitigation

**Gap.** The doc carries ~10 conditioning/symmetry sections (§1, §8.2, §8.3,
§9.1, §9.3, §18, §29, §36, §46, §47) and ~2 on basin escape (§11, §17). The
measurements (§46–§48) say the trajectory's problem is basin escape. The doc's
mass is inversely proportional to where the measured problem is.

**Mitigation.** Before any more analysis: run **multi-start on flower** — 10
inits, keep the best. If best-of-10 reaches ~0.041 (BCD's level), the flower
failure is "one bad starting point" and multi-start (trivial, parallel) solves
it — folding this line into §38 (init). If best-of-10 is still ~0.095, it is a
broad attractor and needs §17 / BCD.

**Open question to argue about.** Is HadNet-Adam's bad basin on flower an
artefact of the **specific** SVD-of-`sqrt` init (§38's line — one bad start) or
a **broad attractor** most inits fall into (the loss geometry — §17 / BCD)?
Multi-start with varied inits answers it in one run, and decides whether the
fix lives in the init line or the optimiser line.

**Failure mode.** `λ_min` came from a 50-step shifted power iteration, which
converges slowly for clustered spectra — true `κ` could be somewhat higher than
700. But the `lr` sweep being flat across 100x is independent of the HVP
estimate, and the two agree, so "not a conditioning problem" is robust even if
the `κ` figure is loose.

---

## 49. Re-measured in the paper's regime (r ∈ {40,100,200}, full-rank data) — most of §40–§48 is out of regime

**The whole §40–§48 chain was measured at r=10–20 on natural images.** The
paper's target is high inner rank on **full-rank** tensors (the `n×n×n×n`
complexity goal, `DECISIONS.md` §6; the aim is inner rank ~200). Re-run there,
the picture changes.

### Convergence — the basin trap is a low-rank artefact

Full-rank synthetic, `svd_initialize` + Adam(`lr=0.1`, ×0.1 decay):

| data | r | final rel err | epochs to rel<0.05 |
|---|---|---|---|
| full-rank 100×100 | 40 | **0.0000** | — |
| full-rank 256×256 | 100 | **0.0000** | — |
| full-rank 256×256 | 150 | **0.0000** | — |
| full-rank (256,256,20) | 100 | 1e-5 | 413 |
| full-rank (256,256,20) | **200** | **1e-6** | **127** |
| full-rank (256,256,20) | 40 | 0.30 (stuck) | never |

At the target rank HadNet **converges straight to machine precision, fast**
(r=200: 127 epochs). §41/§44/§45/§48's "basin trap / BCD beats HadNet / Adam
can't escape" is a **low-rank, structured-image** phenomenon — it does not
occur at the paper's operating point. The one residual stuck case here (r=40 on
`(256,256,20)`) is **under-rank**, not a saddle: perturbing the init
(§49-style) moves it 0.3047 → 0.3039, i.e. nothing. §49's perturbation fix is
specific to the near-degenerate-init image regime and does **not** generalise.

### Per-epoch cost — the anatomy shifts with rank

`(256,256,20)` full-rank, RTX 4050, per training step:

| r | step (ms) | backward % | Adam-step % | bf16x | `torch.compile`x | fused-Adam | peak mem |
|---|---|---|---|---|---|---|---|
| 40  | 0.91 | 57% | 19% | 1.07 | 1.11 | — | 66 MB |
| 100 | 1.47 | 48% | 33% | 1.19 | 1.18 | — | 130 MB |
| 200 | 2.79 | 44% | **41%** | 1.22 | 1.22 | **1.20x** | 243 MB |

- **`torch.compile` is 1.11–1.22x here, not 0.97x.** §33/§35's "compile is a
  dud on the 4050" was an r=10 artefact — at r≥100 the GEMMs are big enough
  that inductor's fusion pays. Reinstate §6 (CUDA graphs / compile) as a real
  lever *at the target rank*.
- **Backward drops from 75% (r=10, §33) to 44% (r=200).** §33's "backward is
  the ceiling, forward micro-opt caps at 25%" **inverts**: at r=200 the
  **Adam step is 41%** of the epoch. New lever: `torch.optim.Adam(fused=True)`
  — one argument, measured **1.20x** at r=200 (2.89 → 2.41 ms); `foreach` does
  nothing. TF32 on top: 2.41 → 2.07 ms (another 1.16x, negligible accuracy
  cost).
- **bf16 is 1.07–1.22x here, not the 1.63x of §35.** That 1.63x was Carphone
  (p=180); the bf16 benefit is shape-dependent, not rank-dependent. State it
  per-shape, never as a single number.
- **Best measured stack at r=200: fused Adam × TF32 × compile ≈ 1.7x**, all
  zero-maths-risk, all *more* effective than at low rank.

### Memory is a hard wall at the 4th-order target

`svd_initialize` + one training step at **r=200**, fused Adam:

| shape | result |
|---|---|
| (256,256,20) | 243 MB — fine |
| (200,200,50,50) | **OOM** (6 GB card) |
| (256,256,40,40) | **OOM** |

At the paper's `n×n×n×n` scale with large `p`, r=200 factors + 2× Adam state +
the `h_prod` intermediates exceed 6 GB. **§14 (quantise) and §5 (share along
`p`) stop being ratio-optimisations and become feasibility requirements** —
and `GPU.md` §6.5's "memory is not a constraint" was an r=10 statement that
reverses here. Gradient checkpointing (dismissed in §33 as "wrong direction")
is back on the table.

### Init cost — now a real fraction

`svd_initialize` at r=200 on `(256,256,20)`: **0.24 s**. Convergence: 127
epochs × 2.8 ms ≈ **0.36 s**. Init is **~40% of the run** — not the "1–3%"
(`DECISIONS.md` §3, measured at 57 s / r=10). §31's stale-baseline gap is worse
than stated: at high rank + fast convergence the init dominates. And
randomized/block-Krylov SVD (§16) is **unusable** — `r=200 > min(m,n)/2` for
any slice under ~800 wide, so the `O(n⁵)→O(n⁴)` story needs `n ≫ 800` to apply.

### (a) Composability

At r=200 the zero-risk systems levers **compose and grow**: `fused Adam (1.20x)
× TF32 (1.16x) × compile (1.22x) × bf16` — and unlike §35's low-rank stack
(dominated by early stopping), here they multiply cleanly because no single one
dominates (backward 44%, Adam 41%). Early stopping still composes but converges
in ~130 epochs already.

### (b) Contradictions with earlier sections

1. **§33 ("backward = 75%, forward opt caps at 25%")** — inverts at r=200
   (backward 44%, Adam 41%). The optimisation target moves from the autograd
   graph to the optimiser step.
2. **§35 ("bf16 1.63x", "compile 0.97x", "h_prod 41x")** — all three are
   shape/rank-specific. compile flips positive; bf16 shrinks to ~1.2x on
   low-`p` shapes; the h_prod Amdahl share falls further as Adam grows.
3. **§40–§48's entire basin/conditioning thread** — measured on out-of-regime
   data. At r≥100 full-rank, HadNet hits machine precision with no basin
   trouble. Keep those sections as a study of the *low-rank image* regime;
   do not carry their conclusions to the rank-200 goal.
4. **`GPU.md` §6.5 / §33 ("memory is free / checkpointing is wrong direction")**
   — reversed: OOM at r=200 for the 4th-order shapes.

### (c) Further approaches — the rank-200 line

- **`fused=True` Adam** — done, 1.20x, one argument. Ship it.
- **CUDA graphs** (`torch.compile(mode="reduce-overhead")`) — untested at
  r=200; the fixed-shape loop is ideal and compile already pays 1.22x.
- **Mixed fp16/bf16 factor *storage*** (not just autocast) — halves the 3.9 GB
  Adam+param footprint that OOMs.
- **8-bit Adam** (`bitsandbytes`) — Adam state is 2/3 of the memory at r=200;
  8-bit states cut it 4x.
- **Rank as the sweep, `r²` as the theory** — measure `error(r)` on full-rank
  data to find the smallest `r` that hits `<1e-4`; §3's `k`-factor then lets
  that `r` drop further (`r^k` effective).

### (d) Gap + mitigation

**Gap.** Every session measurement before this section, and much of
`DECISIONS.md` (§11 plateau, §12 early-stop 3x, §15 kernel speedups, §16
noise-floor), is at **r=10–25**. The paper's contribution lives at **r=40–200
on full-rank data**, where convergence, cost anatomy, memory, and init-share
all differ. The `docs` describe a regime the method is not aimed at.

**Mitigation.** Re-run the `DECISIONS.md` headline experiments — the §3 timing
table, the §11 convergence curve, the §12 early-stop measurement — at
r ∈ {40, 100, 200} on the full-rank `(100,100,10)` / `(100,100,10,10)` tensors
the paper actually uses. Tag every existing number with its rank. Anything
measured only at r≤25 is provisional for the paper's claims.

**Open question to argue about.** Does early stopping (`DECISIONS.md` §12, "3x,
87% of epochs wasted") even apply at r=200, where convergence to 1e-6 takes
~130 epochs with no visible plateau? The "wasted tail" may be an r=10 artefact
of fitting *compressible* data at *insufficient* rank — the opposite of the
paper's setting.

**Failure mode.** These re-measurements are on synthetic iid-Gaussian
full-rank tensors. The paper also reports real video/image at r≤25 (Table 17),
where the §40–§48 low-rank findings *do* apply. The method has two regimes —
high-rank full-rank (converges to ~0) and low-rank real-media (basin-limited) —
and a claim or optimisation valid in one may be void in the other. Every result
needs its regime stated.

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
