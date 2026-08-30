# CUDA optimisation opportunities — survey + monitoring log

Companion to `GPU.md`. `GPU.md` is the environment/driver/timing reference;
this file is **a running catalogue of every CUDA speedup available in the
codebase**, cross-referenced to what is already applied, measured, or untried,
plus a timestamped log of approaches observed being tried (checked ~every
15 min while active work is going on).

Not a work order — an inventory. When an item lands, move it to *Applied* with
the measured number and the commit.

---

## Environment snapshot (2026-08-30, first survey)

| | |
|---|---|
| torch | **`2.11.0+cu128`, `torch.cuda.is_available()` → True** (GPU.md §1 is stale: says `2.13.0+cpu`) |
| GPU | RTX 4050 Laptop, 6 GB, Ada — few SMs, so `torch.compile` has already measured *negative* (0.97x, RESEARCH.md §27) |
| disk | 7.9 GB free of 96 GB — CUDA venv is in place; do not `rm -rf .venv` casually |
| branches | `main`, `perf/batch-hprod`, `opt/discrepancy-stop` (HEAD). A second worktree on `main` exists under `/tmp/.../5db9bb95-.../baseline` — another session is running before/after comparisons. |
| working tree | `hadamard_ops.py` (+DiscrepancyStop), `docs/RESEARCH.md` modified; `tests/test_discrepancy.py` untracked |

**Measured so far** (RTX 4050, Carphone/MRI shapes, from DECISIONS.md §15/§17 +
RESEARCH.md §27): `h_prod` batch **41.5x** kernel / 9–27x end-to-end · randomized
SVD init **4.3x** · bf16 autocast **1.63x** end-to-end, error unchanged · TF32
**1.11x** (subsumed by bf16) · `torch.compile` **0.97x** (no gain on this GPU) ·
backward is **~75%** of an epoch · early stop already cuts 2000→~300 epochs.

---

## Applied (Tier 0 — in `hadamard_ops.py`)

| # | Item | Mechanism | GPU gain | Where |
|---|---|---|---|---|
| A1 | `h_prod` batched GEMM | slice dims → leading, one cuBLAS batched call vs 180 tiny matmuls/call | **41.5x** kernel | `hadamard_ops.py:38` |
| A2 | `svd_initialize` batched + randomized | `svd_lowrank` over all slices at once; O(n⁵)→O(n⁴) | **4.3x** | `hadamard_ops.py:85` |
| A3 | `sync()` honest timing | drains CUDA queue so `time.time()` measures execution | correctness | `hadamard_ops.py:152` |
| A4 | `EarlyStop` / `DiscrepancyStop` | cut kernel-loop iteration count 2000→~300 (not CUDA per se) | ~3x fewer epochs | `hadamard_ops.py:226`, `:355` |

Note (DECISIONS.md §15): the batched GEMM is **not** bit-identical on GPU —
cuBLAS batched reduction order differs, ~1.4 ulp fp32. Exact in fp64. Fine.

---

## Catalogue of remaining opportunities

Grouped by kind. "Risk" = risk to the maths / results, not to effort.

### Tier 1 — one-liners, no maths risk, not yet in committed code

| # | Item | Mechanism | Est. gain | Risk |
|---|---|---|---|---|
| 1 | **bf16 autocast** in `HadamardNet_train` / `HadamardDNN_train` / `igd` loops; keep Frobenius accumulation fp32 | Ada bf16 throughput; loop is pure matmul + elementwise | **1.63x** (measured) | low — error unchanged in §27; *contradicts* noise-floor `DiscrepancyStop` (§27b-2), safe with relative early-stop |
| 2 | `torch.set_float32_matmul_precision('high')` (TF32) | TF32 tensor cores for fp32 matmul | 1.11x | low, subsumed by #1 |
| 3 | `optim.Adam(..., fused=True)` | fuse the whole optimiser step into one kernel over all param tensors | small; grows with param count (factor-nets) | none |
| 4 | `torch.backends.cudnn.benchmark = True` | — | ~0 here (no conv) | none |
| 5 | Verify `foreach=True` is active on Adam (default on CUDA) — batches the per-tensor optimiser math | — | already on, just confirm | none |

### Tier 2 — kill host↔device serialisation in the `main_*.py` scripts

The scripts round-trip to CPU/NumPy constantly; every hop is a full pipeline
stall. Pure plumbing, no maths.

| # | Item | Where | Mechanism | Est. gain |
|---|---|---|---|---|
| 6 | `loss.item()` called **3×/epoch** → cache to one Python float | `main_video.py:80,82,87` (and image/real-world twins) | each `.item()` is a D2H sync that serialises the async queue | removes ~2 syncs/epoch × ~300 epochs × 7 methods × 5 ranks |
| 7 | `torch.tensor(model.W1.detach().cpu().numpy())` → `.detach().clone()` | `main_video.py:104-107, 233-236` | pointless D2H→H2D→copy of every factor after each fit | one-off per fit, but on the timed path |
| 8 | Keep `A`/`D` on GPU; stop `A[...].cpu().numpy()` per slice feeding AGD/IGD/BCD | `main_video.py:403,459,516,520` | 180 D2H copies per method per rank | large for those methods |
| 9 | `qr_transform` → batched `torch.linalg.qr` on GPU (non-pivoted) or accept pivoted-QR stays CPU | `main_video.py:38-46, 342-347` | currently scipy `qr(pivoting=True)` on CPU, per slice | moves QR baseline onto GPU |
| 10 | `svd_transform` baseline is still a 180-slice Python loop | `main_video.py:30-36, 291-296` | batch exactly like `svd_initialize` (A2) | ~ the A2 ratio for the SVD baseline |
| 11 | PSNR/SSIM per-frame `skimage` CPU loop → batched GPU metric (torch PSNR trivial; `pytorch-msssim` for SSIM) | `main_video.py:323-332` ×5 methods | serial CPU chunk on the wall clock between timed sections | removes a fixed CPU tax per method |
| 12 | `torch.no_grad()` around SVD/QR/AGD/IGD/BCD reconstruction + all metric code | throughout `main_*.py` | drop autograd bookkeeping on paths that never call `.backward()` | small, free |

### Tier 3 — port the CPU decomposition algorithms to batched GPU

These are the biggest blocks of raw compute still on the CPU. Each currently
runs **per slice in a Python loop** (180 slices) and internally is dense
linear algebra that batches cleanly.

| # | Item | Where | Mechanism | Est. gain | Risk |
|---|---|---|---|---|---|
| 13 | **AGD** — port `scaled_alternating_gradient_descent_hadDec` to batched torch/CUDA | `alternating_gradient_descent.py` (numba `@jit`, 40000 iters/slice) | body is matmul + elementwise + `norm`; stack the 180 slices into leading batch dim, run on GPU | very large — this is the dominant CPU cost in the video/image sweeps | med — must match the LR-halving + NaN-guard logic; float assoc. differences like §15 |
| 14 | **IGD** — batch the slice loop into `(p,m,r)` factors | `Hadamard_matrix.py` already torch+device, but `main_video.py:461-463` calls it per slice with a fresh `torch.tensor` each time | all ops already batched-matmul shaped; add a leading `p` dim, drop the loop | ~A1-scale on the IGD path | low |
| 15 | **HBCD** — batch the per-column least-squares `for j in range(n)` over columns *and* slices | `Hadamard_BCD.py` `updateH`, `hadamardLS_exact` | `H = A.ᵀ diag(s²) A`; `solve(H, d)` → one batched `linalg.solve` of shape `(slices·n, r, r)`; RESEARCH.md §15 Schur-complement is the fast batched form | large — **now a priority**: RESEARCH.md §44 measures BCD reaching **~2x lower error than HadNet** on smooth/real images from the same init; the "5x slower, use HadNet" verdict (`DECISIONS.md` §3) is purely the per-slice `scipy` solves — batched on GPU it may not be slower. bf16 (#1) + `h_prod` batch compose with a block-LS inner loop | med — momentum accept/reject + `beta_momentum` per iter; `svds` in `hadamardLS_lipschitz` needs a batched power-iteration replacement |
| 16 | AGD/BCD spectral init `np.linalg.svd` on random/ones matrices | `alternating_gradient_descent.py:60-72`, `Hadamard_BCD.py:42-53` | full SVD of a random matrix just to seed — replace with `svd_lowrank` or skip | small | low |
| 17 | Land all ports **once in `hadamard_ops.py`** | 4 duplicate copies: `application-videos/`, `application-images/`, `random-tensor/`, `real-world/` | the module exists for exactly this (see its docstring) | maintenance, not speed | none |

### Tier 4 — kernel- and graph-level (systems, no maths risk)

| # | Item | Mechanism | Est. gain | Risk / notes |
|---|---|---|---|---|
| 18 | **CUDA Graphs** — capture the fixed-shape training step, replay it | kills per-kernel Python launch overhead across the ~300-epoch loop; "many small kernels, fixed shapes" is the textbook case (RESEARCH.md §6, §33) | plausibly large on this launch-bound workload | needs: no `.item()` inside the captured region (do #6 first), LR-decay branch moved out of capture or a capturable optimiser, warmup iters. §33: `torch.compile(mode="reduce-overhead")` reaches this via graphs — a *different* path than plain `torch.compile` (#24, 0.97x), worth trying separately |
| 19 | **Custom `autograd.Function` for batched `h_prod` backward** | backward is **75% of an epoch**; the grad of a batched GEMM is just two more batched GEMMs — a hand-written backward skips autograd's generic path and the permute dance | up to ~1.7x on the epoch if backward halves | low maths risk but §33: won't match autograd's reduction order (breaks bit-identical bar like §15); ~40 lines caching intermediates of `forward()`. §33 open question: may only beat bf16 / low-`r` / block-solves on a bigger GPU |
| 20 | Store factors natively as `(*p, m, r)` instead of `(m, r, *p)` | `h_prod` does `permute → @ → permute` twice per call; the matmul forces a contiguous copy of the permuted view | removes 4 permutes + 2 copies per `h_prod` call | low — layout-only, but touches every call site |
| 21 | Fuse the elementwise chain `M1*M2`, `D - X_hat`, `‖·‖_F` | one fused kernel (Triton or a `torch.compile` sub-region) instead of 3–4 kernels + temporaries | memory-traffic + launch bound | low |
| 22 | **Triton** fused `h_prod` + residual (+ partial norm) kernel | one kernel does the batched product and the `D - X̂` in a single pass | large if #21 isn't enough | med effort, low maths risk |
| 23 | Preallocate / reuse the `h_prod` output buffer | `torch.empty(...)` every call; fixed shapes ⇒ reuse | small on its own; **prerequisite for #18** | none |
| 24 | `torch.compile` on the step | fuses elementwise + cuts Python overhead | **0.97x measured here — no gain on RTX 4050** (too few SMs) | catalogue only; revisit on a bigger GPU |

### Tier 5 — precision / memory (low value here; 6 GB is ample)

| # | Item | Note |
|---|---|---|
| 25 | bf16 storage of the video tensor + factors | memory is not the constraint (GPU.md §6.5); skip unless a bigger tensor shows up |
| 26 | Frobenius norm as fused `linalg.vector_norm` on the diff | marginal |

### Tier 6 — parallelism across the independent sweep

| # | Item | Note |
|---|---|---|
| 27 | 5 ranks × 7 methods are independent → separate CUDA streams, or stack the rank dim into the batched GEMM with padding | real but awkward; only worth it once per-run time is dominated by launch gaps, not work |
| 28 | Batch all 180 slices' Adam state via `foreach` | verify it's actually on (Tier 1 #5) |

---

## The stack (what multiplies)

From RESEARCH.md §27a, systems/precision layer members multiply (orthogonal
mechanisms — launch overhead / iteration count / arithmetic width):

```
h_prod batch (in, 41.5x kernel)  ×  bf16 (#1, 1.63x)  ×  early-stop iteration cut
        ×  CUDA graphs (#18, untried)  ×  batched-backward (#19, untried)
```

`torch.compile` (#24) does **not** join the stack on this GPU. Anything with an
absolute noise-floor stop (`DiscrepancyStop`) does **not** compose with bf16
(#1) — bf16 rounding sits above the `0.01·√N` floor (§27b-2).

**Amdahl caveat (RESEARCH.md §35).** The kernel-level headline numbers are not
the loop-level effect — a component speedup is capped by that component's share
of the epoch:

| Lever | Headline | Runtime share | Epoch-level ceiling |
|---|---|---|---|
| `h_prod` batch | 41x kernel | ~14% of epoch (526 µs / 3795) | **~1.15x** |
| randomized init | 4.3x | few % (init) | **~1.03x** |
| early stopping | 3–3.3x | the loop | **~3x** (genuine) |
| bf16 | 1.63x | the loop | **~1.63x** (genuine) |

Composed loop-level ≈ `3 × 1.63 × 1.15 × 1.03 ≈ 5.6x`, **dominated by early
stop + bf16**. The two biggest-*looking* levers (`h_prod` 41x, init 4.3x) are
the two *smallest* at loop level. §35 also flags an unmeasured ordering
interaction: bf16 loss-noise (~1e-2) can trip `EarlyStop`'s 1e-3 trigger
early, so measuring bf16 speedup without re-checking the stop point overstates
it.

**Measured (RESEARCH.md §41, "M5", RTX 4050, `random-tensor`, ~3 seeds).**
Composed `main` HadNet vs (bf16 + `EarlyStop` + randomized init): optimised
**0.46 ± 0.03 s at 300 epochs**, **10.9x end-to-end**, error cost +0.0011
(within seed variance). ~2x *above* the 5.6x Amdahl estimate — because
`EarlyStop` ran **6.7x** here (2000→300), not the 3x `DECISIONS.md` §12 saw at
a different patience. So the real decomposition is **`EarlyStop` 6.7x × bf16
1.6x ≈ 11x**, and **`h_prod` batching + randomized init contribute nothing
measurable at loop level** — confirming the Amdahl table above. RESEARCH.md
treats the applied stack (bf16 + early stop) as *done, no maths risk*; the next
2x is not in this layer (§41/§42 put it in init/basin quality).

---

## Count

**28 distinct opportunities** beyond the 4 already applied:
Tier 1 ×5 · Tier 2 ×7 · Tier 3 ×5 · Tier 4 ×7 · Tier 5 ×2 · Tier 6 ×2.
Of these, ~12 are one-session mechanical changes (Tiers 1–2), ~5 are real ports
(Tier 3), ~7 are kernel/graph work (Tier 4).

---

## Monitoring log

Format: `<time> — <what changed> — <notes / new items>`.

- **2026-08-30 ~21:55** — First survey. HEAD `b2a08e8` on `opt/discrepancy-stop`;
  working tree adds `DiscrepancyStop` to `hadamard_ops.py` + RESEARCH.md §9.4.
  Another session's worktree on `main` (`5db9bb95…/baseline`) — before/after runs.
  No CUDA change in flight yet; all Tier 1–4 items still open. Catalogue above
  is the baseline.
- **2026-08-30 22:25–22:40** — No CUDA change over three re-checks. No new
  commits (HEAD still `b2a08e8`), no `hadamard_ops.py` change beyond
  `DiscrepancyStop`, no new bench runs (`bench/*.log` frozen at 19:44). All
  movement is `docs/RESEARCH.md` roughly doubling each 15 min (+115→+317→+611
  lines): new Synthesis §27–32 covering the `k`, `p`-scaling, conditioning,
  compression-ratio, init-cost and evaluation-honesty research lines. Maths /
  methodology layer — re-cites existing bf16 1.63x, "backward = 75%",
  compile 0.97x; adds no new systems/CUDA approach. §31 (init-cost) recommends
  profiling the bf16 + early-stop fit to split init vs epoch vs backward — a
  measurement task, already implied by catalogue #19/#A2, not a new lever.
- **2026-08-30 ~22:55** — Still no code/bench change (HEAD `b2a08e8`,
  `bench/*.log` frozen at 19:44). `docs/RESEARCH.md` +719 lines now; new §33
  "Per-epoch cost / the backward pass (measured 75%)". Sharpens two existing
  catalogue items, no new one: **#19** — a custom `autograd.Function` for
  `forward()` is ~40 lines, will break the bit-identical bar (§15), and may
  only pay on a bigger GPU; **#18/#24** — `torch.compile(mode="reduce-overhead")`
  is a distinct CUDA-graph path, separate from the plain-`compile` 0.97x
  result, worth its own trial. Annotated #18/#19 above accordingly. Still
  analysis only — nothing implemented or measured this window.
- **2026-08-30 ~23:10** — No code/bench change (HEAD `b2a08e8`, `bench/*.log`
  frozen at 19:44). `docs/RESEARCH.md` +894 lines; new §34 (HadamardDNN finding
  vs `p`-scaling — maths layer) and **§35 (composed-stack audit)**. §35 is a
  methodology correction to the "what multiplies" section here: applied the
  **Amdahl caveat** above — `h_prod` 41x → ~1.15x at epoch level, init 4.3x →
  ~1.03x; composed loop-level ≈ 5.6x dominated by early-stop + bf16; the full
  stack has never been run end-to-end. No new lever, but the framing of the
  existing ones is now corrected.
- **2026-08-30 23:10–23:40** — Two more re-checks, no CUDA activity. No commits
  (HEAD `b2a08e8`), no code beyond `DiscrepancyStop`, `bench/*.log` still 19:44.
  `docs/RESEARCH.md` → +1166 lines; new §36–38 (gauge-fixing, rate–distortion,
  "what the init is") — all maths-layer *second passes* on §29/§30/§31. They
  re-cite the profiling recommendation (§33: `torch.profiler` on one step,
  GEMM vs elementwise vs memcpy) but add no new systems lever. Pattern over
  ~1.5 h / 8 checks: heavy research-doc writing, zero implementation.
- **2026-08-31 ~00:10** — First empirical activity: RESEARCH.md §39 (measurement
  decision tree) and **§40 "First diagnostic run — M1/M2/M3"** (instrumented
  2000-epoch fit, `random-tensor`, fp32, one seed; script not committed — only
  RESEARCH.md changed). §40 is a convergence/geometry diagnostic, no
  CUDA/systems content, but its result reinforces this catalogue's priority:
  **M1 — the epoch knee is ~300, decisively** (300→2000 is 6.7x compute for
  0.2% error), so the `k`/convergence layer is closed and "the entire
  remaining opportunity is per-epoch cost (bf16, lower `r`) and honest
  measurement". No new lever, no speedup measured for any catalogue item.
  Nothing implemented — no `hadamard_ops.py` / script / bench change; the M7
  `torch.profiler` step is proposed, not run.
- **2026-08-31 ~00:25** — RESEARCH.md +1490 lines; **§41 "M5 measured"** gives
  the first end-to-end number for this catalogue's stack: composed `main` vs
  (bf16 + `EarlyStop` + randomized init) = **10.9x**, 0.46 ± 0.03 s / 300 ep,
  error +0.0011 (seed noise). Decomposes as `EarlyStop` 6.7x × bf16 1.6x;
  `h_prod` batch + randomized init measure **0x at loop level** — the Amdahl
  caveat here confirmed empirically. Folded the M5 numbers into "The stack"
  section above. §42 (Carphone-analog run) is init/basin analysis — not CUDA.
  Still no repo code/bench change; these are uncommitted instrumented scripts.
  RESEARCH.md now treats bf16 + early-stop as "done, no maths risk"; the next
  2x is put in init quality, not the systems layer.
- **2026-08-31 ~00:40** — RESEARCH.md +1680 lines; §43 (init vs final rel on 4
  image datasets — data-dependence) and **§44 "BCD vs HadNet from the same init
  — the deficiency is Adam"**. §44 promotes catalogue **Tier 3 #15**: BCD
  reaches **~2x lower error than HadNet** on smooth/real images from an
  identical init, and its "5x slower" reputation (`DECISIONS.md` §3) is
  entirely the per-slice `scipy` solves — GPU-batched exact block-LS via the
  §15 Schur complement may erase the gap, and bf16 / `h_prod` batch compose
  with a block-LS inner loop. Annotated #15 accordingly. Also a free
  non-CUDA fix noted in passing: HadNet returns final not best-seen params —
  snapshot best. Still no repo code/bench change; scripts uncommitted.
- **2026-08-31 00:40–01:10** — Two re-checks, no CUDA activity. No commits
  (HEAD `b2a08e8`), no repo code/bench change. `docs/RESEARCH.md` → +1964
  lines; new §45 (Adam unstable in a good basin), §46 (BCD basin ~1.5e5x more
  curved — gauge imbalance is the conditioning), §47 (`λ_max` flat along a
  healthy HadNet run). All Hessian-curvature / conditioning analysis — points
  at K-FAC / Riemannian (§1/§8.3), no systems lever. Only CUDA touch: §45
  re-names "GPU-batched block-LS (§15)" as a possible hybrid finisher, already
  under #15.
