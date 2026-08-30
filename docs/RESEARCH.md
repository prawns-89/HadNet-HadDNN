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

## Suggested order

1. **Riemannian (Sec. 1)** — highest expected value; attacks `k`, which is now
   the dominant cost, and explains an observation we already have.
2. **Factor sharing (Sec. 5)** — biggest structural win, unblocks Sec. 2.
3. **CUDA graphs (Sec. 6)** — cheap, mechanical, no maths risk.
4. **k>2 factors (Sec. 3)** — cheap experiment against an existing codebase.
5. **Leverage sampling (Sec. 2)** — only after Sec. 5 raises the redundancy ratio.
