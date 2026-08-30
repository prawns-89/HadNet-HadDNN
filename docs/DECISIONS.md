# Decisions log — HadNet/HadDNN optimisation

Working record of what was investigated, what was measured, what was
decided, and what is still open. Newest context at the top of each section.

**Goal:** reduce HadNet/HadDNN runtime. Stated complexity target: bring the
asymptotic cost down to **4th power**.

---

## 1. Framing: what "QR" meant

First reading of the request assumed QR *codes*. The paper's every use of
"QR" is **QR decomposition** (orthogonal Q, upper-triangular R), specifically
Rank-Revealing QR (Chan 1987). Unrelated abbreviations; no shared maths.

Also established: RRQR is not the paper's contribution. It is a **baseline**
they compete against. `TRRQR` (Algorithm 3) just loops over slices calling
`scipy.linalg.qr`.

**Decision:** dropped the QR-code line of enquiry.

## 2. Considered and parked: partial observation / completion

The intuition "a QR code scans without the whole symbol in frame" maps onto
**matrix/tensor completion**, not onto anything in the scanning pipeline.

- QR-code robustness = Reed–Solomon redundancy *added at encode time* + rigid
  known geometry. Neither exists for an arbitrary data tensor.
- The analogous property here is that **low-rank structure is the redundancy** —
  discovered, not added.
- Formulation change is small: mask the residual in Eq. (6),
  `‖Ω ∘ (X − (W1⊡H1)∘(W2⊡H2))‖_F`. The mask enters as another Hadamard
  product, so it composes with the existing algebra.

**Why parked:** the redundancy ratio is poor at the paper's scale. For the
Carphone video (144×144×3×60, r=10):

- entries `144·144·180 = 3,732,480`
- parameters `2r(m+n)p = 2·10·288·180 = 1,036,800`
- **ratio 3.6×**

Sampling needs `|Ω| ≳ #params`, capping useful subsampling at ~3× before
identifiability breaks — and noisier gradients claw some of that back. The
root cause is structural: every one of the 180 slices gets independent
factors, with no sharing along `p`. For video, where frames are highly
correlated, that is wasteful.

**Decision:** revisit only after factor-sharing along `p` raises the ratio.
Not a first-round lever.

## 3. Where the time actually goes

From the paper's Table 9 (Carphone, rank 10) and complexity Eq. (11)
`O(mnp·min(m,n) + k·r·mnp)`:

| Method | Time (s) | Frobenius error |
|---|---|---|
| TSVD | 0.25 | 108.96 |
| TRRQR | 25.89 | 156.42 |
| TBCD | 283.81 | 51.78 |
| TIGD | 323.77 | 58.24 |
| TAGD | 1193.64 | 48.31 |
| **HadNet** | **57.36** | **47.34** |

HadNet's init *is* essentially the TSVD computation, and TSVD's whole runtime
is 0.25s — so **~1–3% of HadNet's cost is init; the rest is the epoch loop.**

## 4. Repo findings (verified against source)

Fork confirmed byte-identical to upstream for `main_video.py`,
`main_image.py`, `random-tensor/main.py`, so all findings transfer.

### 4.1 `h_prod` is a Python loop over slices — the real bottleneck

`application-videos/main_video.py:67`, called twice per forward (lines 88–89):

```python
for i in itertools.product(*[range(s) for s in p]):
    C[(...,)+i] = torch.matmul(A[(...,)+i], B[(...,)+i])
```

180 slices for Carphone, each a 144×10 @ 10×144 matmul (~200k FLOPs — far too
small to saturate anything). 2 calls/epoch × 2000 epochs = **720,000 matmul
calls per fit, forward only.** On CUDA each is a separate kernel launch
(~5–10 µs overhead vs ~1 µs of work), and each `C[(...,)+i] = ...` is an
`index_put` adding an autograd node, so backward walks 360 nodes/epoch too.

The batch dims are *trailing* in the `(m, r, *p)` layout while `torch.matmul`
batches over *leading* dims — which is precisely why the loop exists.

### 4.2 `k` is hard-fixed at 2000 with no early exit

`HadamardNet_train(D, r, max_epochs=2000)` (L94); call site L581 does not
override. The `wait` patience counter **only decays the LR ×0.1 — it never
breaks.** The single `break` in the file (L270) is in the video reader.
`updatecount` is dead code; the `elif` incrementing it is commented out, so
`updatecount == 100` never fires.

### 4.3 `svd_initialize` uses economy, not truncated, SVD

`torch.linalg.svd(mat, full_matrices=False)` computes all `min(m,n)=144`
singular triplets then slices `[:, :r]` to keep 10 — **93% discarded.** Twice
per slice, inside a 180-iteration loop = 360 economy SVDs per init.

### 4.4 Incidental

- `qr_transform` (L30) round-trips GPU→CPU→GPU **per slice**
  (`.cpu().numpy()` → `scipy.linalg.qr` → back). This explains TRRQR at 25.89s
  vs TSVD's 0.25s: device transfer, not algorithmic cost.
- **Baseline timings are not like-for-like.** `maxIter = 40000` for AGD/IGD
  vs HadNet's 2000 epochs — 20×. AGD is numba-CPU, BCD is numpy/scipy-CPU,
  only HadNet/HadDNN are GPU-resident. HadNet's reported 21× advantage over
  TAGD is close to the iteration-count ratio alone.

## 5. Measurements taken

numpy ports at the exact Carphone shape (see `bench/`). **No torch was
available, so these are CPU/numpy figures, not validated on GPU.**

`bench/bench_hprod.py` — loop vs single batched GEMM:

```
max abs diff: 0.0
repo (Python loop):    15.45 ms
batched (1 GEMM)  :     1.71 ms
speedup           :      9.1x
est. fwd-only over 2000 epochs:  repo 61.8 s | batched 6.8 s
```

Output is **bit-identical**, and the 61.8s forward-only projection sits right
on the paper's reported 57.36s total — i.e. the loop is essentially the
entire runtime. On GPU the gap should widen, since launch overhead dominates
more at this matmul size than numpy's Python indexing does.

`bench/bench_svdinit.py` — init loop vs batched:

```
repo  (180-slice loop, 360 economy SVDs):  2023.4 ms
batched (2 batched SVD calls)          :  1731.3 ms  -> 1.2x
init as share of the 57.36 s reported run: 3.5%
```

Only 1.2×, because SVD is genuinely FLOP-bound rather than launch-bound.

## 6. The complexity target: O(n^5) → O(n^4)

For a 4th-order tensor `A ∈ R^{n×n×n×n}`, in the paper's `m×n×p` notation
`m = n`, `p = n²`. Eq. (11) becomes:

| Term | Cost | Source |
|---|---|---|
| `mnp·min(m,n)` | **n^5** | SVD initialisation |
| `k·r·mnp` | `k·r·`**n^4** | epoch loop |

**The SVD init is the only super-quartic term.** Replacing the economy SVD
with a randomized/truncated rank-r SVD (Halko–Martinsson–Tropp) costs
`O(mnp·r) = O(r·n^4)`, which for `r ≪ n` is **O(n^4)** — hitting the target.

### The key distinction

| Change | Effect on complexity | Effect on wall-clock now |
|---|---|---|
| Batch `h_prod` | none (constant factor) | **~9× measured** |
| Randomized SVD init | **n^5 → n^4** | ~3% (init is only 3.5%) |

These are **not** competing; they solve different problems. Batching wins at
today's sizes, randomized SVD is what makes the method scale. Both belong in
the first round. Ranking them against each other was the earlier mistake —
the wall-clock profile alone said "skip the SVD", which is wrong once the
goal is stated as asymptotic.

## 7. Lever list

| # | Lever | Expected | Effort | Status |
|---|---|---|---|---|
| 1 | Batch `h_prod` into one GEMM | **9.0× measured, bit-identical (verified)** | done | **APPLIED** |
| 2 | Randomized/truncated SVD init | **n^5 → n^4**; 2.0× wall-clock measured | done | **APPLIED** |
| 3 | Real early stopping (`k` fixed at 2000) | breaks when lr < 1e-5 | done | **APPLIED** |
| 4 | Mixed precision (`torch.autocast`) | ~2×, orthogonal | 2 lines | not applied |
| 5 | Subsampled loss | gated by the 3.6× ratio — see §2 | medium | parked |
| 6 | Share factors along `p` | unblocks #5, improves compression | large | not started |

Levers 1, 3 and 4 compose multiplicatively.

Note in favour: `h_prod` sits *inside* HadNet's 2000-epoch loop but is called
only a handful of times for the baselines' final reconstruction (L420, 476,
533). Batching therefore speeds HadNet far more than the baselines — it
**widens** the reported advantage over TAGD/TIGD/TBCD rather than shrinking it.

## 8. Environment — blocked

Nothing has actually been executed from this repo yet.

- System Python 3.12.3 is **PEP 668–managed**; bare `pip install` fails.
  Wrote `requirements.txt` (none shipped), derived from actual imports.
- `python3 -m venv .venv` + install **failed: `[Errno 28] No space left on
  device`.** Disk is 96G at **97% used, ~2.9G free**.
- The partial venv is **3.4G**, of which **2.7G is CUDA runtime**
  (`nvidia/` 2.0G, `triton/` 691M) — pulled because default `pip install
  torch` fetches the CUDA build.
- **`nvidia-smi` fails: no working NVIDIA driver on this machine.** So the
  CUDA runtime is unusable here regardless.
- Reclaimable: **~6.4G** (3.4G broken venv + 3.0G pip cache).

**Decision:** on this box use the **CPU wheel** (`--index-url
.../whl/cpu`, ~200M vs 2.5G). GPU timings must come from the RTX 4500
machine.

## 9. Open questions

- **"Share with Lipschitz"** — unresolved. Either (a) a person, or (b) fold in
  the Lipschitz/convergence analysis. Note `Hadamard_BCD.py:93`
  `hadamardLS_lipschitz` already uses `lr = 1/L` with `L = σ_max²`; the same
  idea could replace HadNet's heuristic Adam `lr=0.1` and principally reduce
  `k`, which multiplies the `n^4` term. Needs confirming before acting.
- No GPU baseline recorded yet — required before any speedup can be claimed.
- `h_prod` is duplicated across all four folders with no shared module.
  Refactor to one importable file before patching, so the fix lands once.


---

## 10. Round 1 applied — `perf/batch-hprod`

Canonical kernels now live in `hadamard_ops.py`; the four experiment folders
import from it via a small `sys.path` shim, so the fix exists once.

**Verified on CPU** (`tests/test_ops.py`, 144×144×60×3, r=10):

```
[1] h_prod           bit-identical=True  maxdiff=0.000e+00
    loop 103.10 ms | batched 11.45 ms  ->  9.0x
[2] svd_initialize (randomized=False) vs loop
    factors match=True  worst abs diff=0.000e+00
[3] svd_initialize (randomized=True)   O(n^5)->O(n^4)
    init rel. error  exact=0.882149  randomized=0.888817  delta=6.67e-03
    loop 2237.6 ms | batched-exact 1955.2 ms | randomized 1092.9 ms -> 2.0x
PASS
```

Lever 1 is **provably bit-identical**. Lever 2's randomized path is
approximate by construction (init only; `randomized=False` restores exactness
and still matches the loop to 0.0).

Also fixed, pre-existing and unrelated to perf: `HadamardDNN_train` had
`device='cuda'` hardcoded as a default, crashing with "Torch not compiled with
CUDA enabled" on CPU-only builds. Now falls back properly. This is why the
repo could not previously be smoke-tested without a GPU.

Smoke test (`random-tensor`, reduced shape, all 7 methods) runs clean.

**Still to do:** lever 4 (autocast); GPU baseline vs branch on the RTX 4500.

---

## 11. Lipschitz step size — tested, REJECTED in this form

**Blocker found first:** the repo minimises the **unsquared** Frobenius norm
(`loss = torch.norm(D - X_hat, 'fro')`, main_video.py:76). The gradient
`-R/||R||_F` has curvature that diverges as `||R||->0`, so **no finite
Lipschitz constant exists near the optimum.** Every `1/L` result assumes the
squared objective. Added `squared_loss()` as the prerequisite.

**Block-convexity is real** (paper Sec. 4.2). With W2,H1,H2 fixed the loss is
a convex quadratic in W1; row `i` has design matrix `G_i = diag(M2[i,:]) H1^T`
and curvature `L_i = 2*sigma_max(G_i)^2` — the same form as the existing
`Hadamard_BCD.hadamardLS_lipschitz`. Implemented as `block_lipschitz()` /
`lipschitz_lr()` using the submultiplicative bound
`sigma_max(diag(m)H^T) <= max|m| * sigma_max(H)`.

**Result** (`tests/test_lipschitz.py`, 60×60×8, r=6, identical SVD init):

```
relative error at epoch:      50     100     250     500    1000    2000
(a) unsquared, Adam 0.1   0.3479  0.3366  0.3330  0.3320  0.3312  0.3310  10.6s
(b) squared,   Adam 0.1   0.3508  0.3384  0.3335  0.3321  0.3311  0.3303  10.4s
(c) squared,   Adam 1/L   0.4226  0.4106  0.3834  0.3706  0.3621  0.3497  17.3s
(d) squared,   GD 1/L     0.4284  0.4223  0.4070  0.3885  0.3759  0.3705  14.2s
```

Lipschitz stepping is **worse on both axes** — higher error and ~60% more
time per epoch. Four reasons:

1. The bound is conservative; `1/L ~ 2e-5` against Adam's `0.1`.
2. Adam already adapts per-parameter from gradient second moments, so
   dividing by `L` duplicates that and then over-damps.
3. `1/L` is justified for a **block** step on the convex subproblem. HadNet
   takes a **joint** step over all four factors — a non-convex quartic. The
   convexity result does not license it.
4. Recomputing `L` each epoch costs a spectral norm per block.

The regime where it *is* justified is true block-coordinate descent, cycling
W1→H1→W2→H2 with an exact/Lipschitz step each. That is TBCD, which the paper
already reports as ~5x slower than HadNet.

**Decision:** keep Adam. Squared loss is retained as an option (marginally
better: 0.3303 vs 0.3310) but is not the win.

### The finding that actually matters

The convergence curves show **k=2000 is largely wasted**:

| stop at | rel. error | vs epoch 2000 | compute |
|---|---|---|---|
| 250 | 0.3335 | +1.0% | **12.5%** |
| 500 | 0.3321 | +0.5% | **25%** |
| 2000 | 0.3303 | — | 100% |

Roughly **87% of epochs buy under 1% of accuracy.** Stopping at 500 gives a
**4x** cut for a 0.5% error cost — second only to the `h_prod` batching, and
it composes with it. This makes lever 3 (early stopping) far more valuable
than the fixed patience threshold currently implemented.

---

## 12. Early stopping — APPLIED, ~3x

Replaced the LR-decay-only exit with `EarlyStop` (hadamard_ops.py): stops when
the best loss improves by less than `min_rel` (1e-3) over `patience` (50)
epochs. Relative, so it transfers across sizes and ranks without retuning.

Measured (random-tensor, reduced shape, identical seed):

| | before (k=2000 fixed) | after | change |
|---|---|---|---|
| HadNet time | 6.06 s | **2.03 s** | **3.0x faster** |
| HadNet rel. error | 0.362178 | 0.364970 | +0.77% |
| HadDNN time | 54.79 s | **16.73 s** | **3.3x faster** |
| HadDNN rel. error | 0.377306 | **0.376999** | **-0.08% (better)** |

HadDNN improving is consistent with early stopping acting as regularisation
on the over-parameterised factor nets. Wired into all 6 training loops across
the 3 patched folders; `real-world/` still outstanding.

---

## 13. Power iterations: Shard's niter=24 does NOT transfer

Shard (krishgarg.com/shard) uses `torch.svd_lowrank(niter=24)` "for
stability". Adopting it made our init **2.8x slower than the exact loop**.

Each power iteration is another O(m*n*q) pass, so randomisation only pays
while `niter*(r+oversample) < min(m,n)`. Measured (144x144x60x3, r=10, so
q=20 and min(m,n)=144 -> predicted crossover at niter=7):

| niter | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 24 |
|---|---|---|---|---|---|---|---|---|
| speedup | 3.6x | 2.8x | 2.3x | **1.9x** | 1.4x | 1.3x | 0.9x | 0.4x |
| err delta | 1.5e-2 | 6.8e-3 | 3.2e-3 | **1.7e-3** | 4.7e-4 | 1.8e-4 | 2.5e-5 | 4.8e-7 |

Empirical crossover lands exactly where the rule predicts. **Default set to
niter=4** (1.9x, delta 1.7e-3 — far below what a 2000-epoch descent washes
out), plus an automatic guard that clamps `niter` if it would enter the
unprofitable regime.

**Why it doesn't transfer:** Shard optimises for *memory* and accuracy, where
extra passes are effectively free. We optimise for *time*, where they are the
cost. Same function, opposite objective, opposite setting.

Note `real-world/` is matrix-only with min(m,n)=256 and small r, so it stays
profitable at niter=24 there (delta 3.3e-07) -- the folder keeps that value.

---

## 14. real-world/ patched

Structurally different from the other three: matrix-only (no slice dim, so
`h_prod` batching does not apply), `max_epochs=10000`, `lr=0.01`, and it
already used the **squared** loss (`torch.norm(X - X_hat) ** 2`) -- an
inconsistency with the tensor folders, which use the unsquared form.

Applied: randomized `svd_init` (via `sklearn.randomized_svd`, niter=24),
`EarlyStop` on both training loops, and the same `device='cuda'` crash fix.
All four folders are now covered.

---

## 15. GPU results, and a correction to the "bit-identical" claim

First run on the RTX 4050 (torch 2.11.0+cu128).

| | CPU | **GPU** |
|---|---|---|
| `h_prod` batching | 8.8-9.1x | **41.5x** |
| `svd_initialize` randomized | 2.0x | **4.3x** |

41.5x confirms the prediction in Sec. 4.1: on CUDA the loop pays ~5-10us of
launch overhead per slice against ~1us of work, 180 slices per call.
6.72 ms -> 0.16 ms.

### Correction

"Bit-identical" was verified on CPU and stated unqualified. **On GPU it is
false**, and the test caught it. Investigated:

```
float64:  exact=True   maxdiff=0.000e+00     <- identical in exact arithmetic
float32:  1.4 ulp      (eps 1.2e-07)         <- pure rounding
TF32 OFF, matmul precision "highest"         <- not a precision-mode artifact
vs float64 truth:  loop 2.465e-06 | batched 3.042e-06
```

Not a bug: cuBLAS uses a different reduction order for a batched GEMM than
for N separate ones, and float addition is not associative. The batched path
is marginally *less* accurate than the original -- the opposite of what
"bit-identical" implies, so worth stating plainly.

**Consequence for benchmarking:** earlier guidance that error/PSNR/SSIM "must
not move" holds on CPU only. On GPU they move in ~the 7th decimal. Movement
in the 3rd decimal or beyond still means something broke.

`tests/test_ops.py` now asserts the correct thing per device: exact in
float64 (the mathematical claim), bit-identical on CPU, <=8 ulp on GPU.

---

## 16. Range constraint for image/video -- NOT worth adding

Data is normalised to [0,1] and metrics use `clamp(0,1)`, but the loss is on
the **unclamped** reconstruction (main_video.py:77) -- so the model optimises
one quantity and is evaluated on another. Tested four ways of closing that gap
(`tests/test_range.py`, 256x256 MRI, r=10):

| mode | epochs | %>1 | rel err (clamped) | PSNR |
|---|---|---|---|---|
| plain (current) | 150 | 0.12% | 0.25112 | 20.69 dB |
| clamp in loss | 150 | 0.25% | 0.25089 | **20.70 dB** |
| penalty (lam=10) | 100 | 0.02% | 0.64156 | 12.54 dB |
| sigmoid output | 100 | 0.00% | 0.92851 | 9.33 dB |

**+0.01 dB.** Not worth it. The multiplicative structure is naturally
self-limiting -- a product of two factors initialised from sqrt(|X|) on [0,1]
data stays near [0,1] without being told to.

Penalty and sigmoid are much worse, but that test is **confounded**: both were
given an SVD init designed for the linear model. Sigmoid in particular would
need a logit-space init to be judged fairly. Do not read those rows as
refutations of the approach, only as "not a drop-in".

### The finding that actually matters

A rank sweep initially showed **17% of reconstructed values below zero**,
which looked like severe overshoot. It is not the model -- it is the data:

```
clean MRI  : 33.28% of pixels are exactly 0 (black background)
+0.01 noise: 17.79% below 0,  0.07% above 1
model      : 17.03% below 0,  0.34% above 1   <- tracks the target
```

The scripts add `0.01*randn` (main_image.py:267, main_video.py:294), which
pushes half of the black background negative. **The benchmark evaluates
PSNR/SSIM on clamped output against a target that is itself 17.79% out of
range.** Genuine model overshoot is only the ~0.27% excess above 1.

So a hard [0,1] constraint would be actively harmful: it would stop the model
fitting a target the benchmark itself has moved out of range. Worth revisiting
only if the noise injection is reconsidered.
