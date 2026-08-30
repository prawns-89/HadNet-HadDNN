# Results

Independent replication and optimisation of **Panchal, Sahoo & Behera,
*Tensor decomposition and neural architectures through generalized Hadamard
products*, Neurocomputing 699 (2026) 134372.**

17 changes tried against the authors' released code: **2 adopted, 8 rejected
with measurements, 3 findings that contradict the paper's claims.**

Hardware: RTX 4050 Laptop (6 GB), torch 2.11.0+cu128. The paper used an
RTX 4500 Ada (24 GB), so absolute times differ — **ratios are what transfer.**

Full detail and every rejected experiment: [`docs/DECISIONS.md`](docs/DECISIONS.md).
Research directions: [`docs/RESEARCH.md`](docs/RESEARCH.md).

---

## Headline

| | |
|---|---|
| Speedup, composed | **~176×** on video at rank 25 |
| — of which free | **27×**, error identical to 5 decimal places |
| — of which paid | 6.5×, costing +3.1% clean error |
| Init complexity | **O(n⁵) → O(n⁴)** |

---

## The paper's reference point

Table 9, Carphone video 144×144×3×60 at rank 10, on their RTX 4500 Ada:

| method | time (s) | rel. error | PSNR |
|---|---|---|---|
| TSVD | 0.2519 | 0.134775 | 25.16 |
| TRRQR | 25.8890 | 0.193469 | 21.95 |
| TAGD | 1193.6444 | 0.059753 | 32.42 |
| TIGD | 323.7671 | 0.072039 | 30.71 |
| TBCD | 283.8097 | 0.064049 | 31.77 |
| **HadNet** | **57.3602** | **0.058552** | **32.60** |
| HadDNN | 115.8578 | 0.079599 | 30.00 |

---

## Change by change

### Adopted

| change | what the code did | what changed | effect |
|---|---|---|---|
| **Batched `h_prod`** | One tiny matmul per slice in Python — 180 slices × 2 calls × 2000 epochs = **720,000 kernel launches** per fit | Permute slice dims to the front, one batched GEMM | **27×** end-to-end, 41.5× on the kernel. Exact in float64; 1.4 ulp on GPU (cuBLAS reduction order) |
| **Randomized SVD init** | Economy SVD computing all 144 singular triplets then keeping 10 — **93% discarded** | Truncated randomized SVD, `niter=4`, with a profitability guard | **O(n⁵) → O(n⁴)**, 4.3× on init. Removes the only super-quartic term in their Eq. (11) |
| **Noise-floor stopping** | `max_epochs=2000` with **no early exit** — the patience counter only decays the LR, never breaks | Stop at the analytic noise floor: scripts inject exactly `0.01·randn`, so `σ√N` is known | **6.5×** (309 epochs vs 2000), +3.1% clean error. Predicted floor 19.32, realised **19.32** |

### Correctness fixes

| issue | detail |
|---|---|
| **No GPU synchronisation** | `torch.cuda.synchronize()` appears **nowhere** in the repo, and timers close before the `.item()` that would force a sync. Their GPU timings measure kernel *enqueue*, not execution. Added at 35 sites |
| **Hardcoded `device='cuda'`** | A default argument in `HadamardDNN_train` crashed on CPU-only builds — the repo could not be run at all without a GPU |

### Rejected, with the numbers

| idea | result | why it failed |
|---|---|---|
| Relative-improvement early stop | **+171%** error at rank 25 | No usable operating point: patience ≥100 never fires, patience=50 costs 3× the error. Criterion wrong, not mistuned |
| Lipschitz step size (`lr = 1/L`) | worse on both axes | Adam already adapts; block-convexity doesn't license a *joint* step. Also the loss is the **unsquared** norm, so no finite L exists near the optimum |
| k>2 Hadamard factors | k=2 wins at every target | The `rᵏ` rank bound is never attained; k=5 diverges. Derived `k* = ln R`; theory unconfirmed. *Provisional* — lr held fixed across k |
| Gauge fixing / balancing | 0% gain, 7–21% slower | Provably correct (norm ratio 999.4 → 1.0000) and changes nothing. **Closes the cheap route to the Riemannian idea** |
| Anderson / DIIS | no gain | Every step is harmful (invalidates Adam's state); every 10 helps early but converges to the same point |
| Anderson + noise-floor stop | 405 vs 309 epochs | Destructive interference — extrapolation makes the trajectory less monotone, so the threshold takes longer to cross |
| Log-domain closed form | 10–13× faster, **2–4× worse error** | Minimises log-space error while scored in Frobenius, so it over-weights dark pixels. Also not a reformulation — collapses to exponential-family PCA |
| [0,1] range constraint | +0.01 dB | Real overshoot is only ~0.3%; the multiplicative form is self-limiting |

### Marginal

| idea | result |
|---|---|
| Log-domain as **initialiser** | Starts worse, **ends better** (−1.6 to −3.5% final error) — finds a better basin, so `sqrt(\|X\|)` is not optimal. Not adopted: costs 0.9 s and helps only past epoch 500, which noise-floor stopping no longer reaches |

---

## Three findings that contradict the paper

### 1. The comparison is unfair — but the method survives it on its own data

HadNet at rank `r` stores four factors — `2r(m+n)` values. SVD at rank `r`
stores `r(m+n)`. The paper compares at **matched rank**, so HadNet silently
gets double the budget. That asymmetry is real and undisclosed.

**But correcting it does not overturn the paper.** On full-rank random data —
the regime of its headline Tables 2, 5 and 6 — HadNet still wins:

| rank | SVD(r) | **HadNet(r)** | SVD(2r) *same storage* | winner |
|---|---|---|---|---|
| 10 | 0.829952 | **0.540522** | 0.680000 | **HadNet** |
| 25 | 0.611893 | **0.067476** | 0.322372 | **HadNet, 4.8×** |
| 40 | 0.427640 | **0.000000** | 0.080829 | **HadNet, exact** |

**The real finding is that it depends entirely on the spectrum.** On *images*,
the same test reverses — SVD at matched storage wins at every rank, by up to
170×:

| rank (128mri.png) | SVD(r) | HadNet(r) | **SVD(2r) same storage** |
|---|---|---|---|
| 25 | 0.14382 | 0.12036 | **0.06435** |
| 128 | 0.01022 | 0.00507 | **0.00003** |

Natural and medical images are approximately low-rank: their spectra decay
fast, so a truncated SVD is already near-optimal and extra expressiveness has
nothing to capture. Full-rank random data has a flat spectrum, where truncation
is costly and the Hadamard product's `r²` reach genuinely pays.

**So the paper's image and video applications (Tables 7, 9) are its *weakest*
case, not its strongest** — the opposite of the impression Table 9 gives.

*An earlier version of this section claimed the advantage was a
parameter-budget artifact in general. That was measured on two images only and
did not generalise; corrected in `docs/DECISIONS.md` §26.*

### 2. HadDNN's factor networks are actively harmful

Each network maps a factor to itself through a 64-unit bottleneck — but the
**final layer's bias is already full-dimensional**, so it alone spans the whole
factor space. The 332M weights upstream add a 64-dim correction to something
already complete.

Image 256×256, rank 25, 1500 epochs:

| regime | trainable | time | rel. error |
|---|---|---|---|
| **plain factors** (= HadNet) | **76,800** | **1.00 s** | **0.03990** |
| FactorNet, 98.5% frozen, bias only | 153,600 | 2.05 s | 0.10589 |
| FactorNet h=64 (= HadDNN) | 10,001,920 | 8.98 s | 0.21590 |
| FactorNet h=256 | 39,743,488 | 28.84 s | 0.45862 |

**Monotone in the wrong direction.** Freezing 98.5% of the parameters *halves*
the error; widening the bottleneck doubles it again — ruling out "make the
network bigger". Plain factors win at **every** learning rate from 0.3 to 0.001.

At rank 25 on video the networks need 4 GB before activations and **OOM on a
6 GB card** — which is why HadDNN has no rank-25 row in the sweep. The paper's
own Table 9 already shows it dominated (115.86 s / 0.079599 against HadNet's
57.36 s / 0.058552); this is the mechanism.

### 3. The benchmark rewards fitting the injected noise

The scripts add `0.01·randn` and then score error **against the noisy tensor.**
At rank 25 on video, the full run gives:

| measured against | rel. error |
|---|---|
| the noisy target (**what the paper reports**) | **0.01621** |
| the true video | **0.02361** |

The reconstruction is **closer to the noise realisation than to the real
signal.** At rank 10 the ordering reverses, because the model then lacks the
capacity to memorise noise. So every comparison at rank 20+ partly measures
noise-fitting, and a method that overfits harder scores better.

The same three lines push **17.79% of an MRI below zero**, after which metrics
are computed on clamped output.

---

## Not verified

| gap | why it matters |
|---|---|
| Different hardware | RTX 4050 (6 GB mobile) vs their RTX 4500 Ada (24 GB) |
| Baselines not like-for-like | Their AGD/IGD run **40,000 iterations on numba-CPU** against HadNet's 2,000 epochs on GPU. The reported 21× advantage over TAGD is close to the iteration ratio alone. This is in the released code |
| Two images for the rank finding | Video and the real-world matrices untested; the effect is already visibly data-dependent |
| 2000-epoch budget may under-train | "The optimiser cannot reach the good solutions" and "the solutions are not there" are different claims, and these experiments do not separate them |
| Four HadDNN fixes untested | Linear activation, no final bias, per-slice shared decoder — never measured |
| **Paper and code disagree** | §4.2 states "a linear activation function"; the code ships `Linear → LayerNorm → ReLU`. Unresolved which produced the published numbers |

---

## Branches

| branch | verdict |
|---|---|
| `perf/batch-hprod` | **adopted** — 27× free, bit-identical |
| `opt/discrepancy-stop` | **adopted** — 6.5× fewer epochs |
| `opt/balancedness` | rejected |
| `opt/anderson` | rejected |
| `opt/anderson-discrepancy` | rejected |
| `model/log-domain` | rejected as method |
| `exp/high-rank` | finding 1 |
| `exp/haddnn` | finding 2 |
| `research/rank200-remeasure` | separate session — re-measures at r=40/100/200 and reports the basin behaviour is a low-rank artefact. **Qualifies finding 1; worth reconciling** |
