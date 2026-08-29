# GPU / CUDA

Single home for everything CUDA- or GPU-related: driver state, install
requirements, timing correctness, and the GPU work still outstanding.
Non-GPU commands live in `COMMANDS.md`.

---

## 1. Current state

| | |
|---|---|
| GPU | **NVIDIA GeForce RTX 4050 Laptop (AD107M, Ada)**, 6141 MiB |
| Driver | **595.84**, working |
| Kernel module | loaded (`nvidia`, with `nvidia_uvm`, `nvidia_modeset`) |
| Kernel | `7.0.0-30-generic` |
| Secure Boot | disabled, platform in Setup Mode (no MOK enrolment needed) |
| **Torch build** | **`2.13.0+cpu` — `cuda.is_available()` is False** |

The driver is healthy. The blocker is that the venv holds the **CPU wheel**,
and swapping it in needs disk we do not currently have. See Sec. 4.

---

## 2. How the driver was broken, and fixed

`nvidia-smi` failed with *"couldn't communicate with the NVIDIA driver"*.

The NVIDIA driver ships as **two halves** — userspace libraries and a kernel
module — in separate packages. Diagnosis showed:

- userspace `libnvidia-*` **535.309.01** installed (`ii`)
- every `linux-modules-nvidia-535-*` package **`rc`** (removed, config only)
- those packages only ever targeted kernels 6.11.x / 6.14.x / 6.17.x —
  **none for the running 7.0.0**
- no DKMS, so nothing rebuilt automatically

Classic "kernel upgraded past the driver". Branch 535 (2023) has no build for
kernel 7.0 at all, so upgrading the branch — not reinstalling 535 — was the fix.

```bash
sudo ubuntu-drivers install nvidia:595
sudo reboot                      # module loads at boot; without a reboot, no GPU
```

### Diagnostic commands

```bash
lspci -nn | grep -iE "vga|3d"                 # hardware present, regardless of driver
lsmod | grep -E "^nvidia "                    # want: nvidia ... nvidia_uvm,nvidia_modeset
uname -r                                      # running kernel
dpkg -l | grep -E "^rc.*linux-modules-nvidia" # rc = REMOVED, config only
apt-cache search "linux-modules-nvidia-.*-$(uname -r)"   # module for THIS kernel?
ubuntu-drivers devices                        # recommended branch
mokutil --sb-state                            # Secure Boot on => MOK enrolment needed
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi -l 1                               # live refresh while a job runs
```

`nvidia_wmi_ec_backlight` alone is **not** the driver — it is a backlight
helper and can be loaded while the real module is absent.

---

## 3. Torch CUDA wheels

`torch.cuda.is_available()` depends on the **wheel build**, not the driver.
A perfectly healthy driver still reports False under a `+cpu` build.

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# CPU-only  (~1.6 GB installed)
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA      (~5 GB installed, pulls pinned nvidia-* runtime libs)
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The CUDA wheel bundles its own CUDA runtime, so no system CUDA toolkit is
needed — only a driver new enough for it. 595.84 is comfortably new enough
for cu128.

---

## 4. The CUDA install needs ~8 GB free — three attempts failed

All three died with `[Errno 28] No space left on device`, the last starting
from **7.1 GB free**, which was still not enough.

Why the installed size understates it: `/` is a **single partition**, so pip's
temporary extraction competes with the install itself. Peak demand is roughly
**installed size + largest wheel unpacked**, i.e. ~8 GB for a ~5 GB result.
`--no-cache-dir` avoids storing wheels twice and helps, but not enough.

Each failure left a partial, unimportable venv (3.4–5.0 GB) which had to be
removed before retrying.

### To make room

```bash
sudo apt clean                       # 2.7 GB of cached .debs — zero risk, refetchable
rm -rf ~/.local/share/Trash/*        # 3.9 GB — YOUR deleted files, your call
go clean -modcache                   # 1.2 GB — safe, re-downloads on next build
.venv/bin/pip cache purge            # regenerable
```

`sudo apt clean` alone plus current free space should clear it.

### Then

```bash
cd /home/prawns/SOP/HadNet-HadDNN
rm -rf .venv && python3 -m venv .venv
.venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install --no-cache-dir numpy scipy numba opencv-python-headless \
    scikit-image scikit-learn networkx Pillow
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Standing caution:** the filesystem is 96 GB at ~94% used with no venv
present. A CUDA venv is ~5 GB on its own, so every rebuild will be tight.

---

## 5. Timing GPU code correctly

CUDA kernel launches are **asynchronous**. `time.time()` around GPU work
measures how long it took to *enqueue* kernels, not to execute them.

**Upstream has no `torch.cuda.synchronize()` anywhere**, and its timing blocks
close before the `.item()` call that would force a sync — so the reported GPU
timings are not sound. This matters directly for any before/after comparison.

The patched scripts call `sync()` from `hadamard_ops` at 35 sites:

```python
from hadamard_ops import sync
sync(); start = time.time()
...                                   # GPU work
sync(); end = time.time()
```

`sync()` is a no-op on CPU, so the same code is correct in both builds.

---

## 6. Outstanding GPU work

1. **Baseline `main` vs `perf/batch-hprod`** on the RTX 4050. Every number
   measured so far is CPU-only.
   ```bash
   git checkout main
   cd application-videos && ../.venv/bin/python main_video.py 2>&1 | tee ../bench/gpu_baseline.log
   cd .. && git checkout perf/batch-hprod
   cd application-videos && ../.venv/bin/python main_video.py 2>&1 | tee ../bench/gpu_optimised.log
   ```
   Error / PSNR / SSIM **must not move** for the `h_prod` change — it is
   bit-identical. Early stopping and the randomized init shift error ~1%,
   which is expected and documented in `DECISIONS.md`.

2. **`h_prod` speedup should be LARGER on GPU.** The 8.8–9.1x measured on CPU
   is bound by Python indexing; on CUDA the original pays ~5–10 us of kernel
   launch overhead per slice against ~1 us of actual work, 180 slices per call,
   360 calls per epoch. That gap is what batching removes.

3. **Lever 4 — mixed precision.** Deferred as unmeasurable on CPU. The loop is
   pure matmul + elementwise, and Ada has strong bf16, so ~2x is plausible.
   Keep the Frobenius accumulation in fp32.
   ```python
   with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
       ...
   ```

4. **Expect different absolute numbers from the paper.** Table 9 was produced
   on an RTX 4500 Ada (24 GB workstation); this is a 6 GB Max-Q mobile part
   with a far lower power budget. Absolute times will be slower; **ratios
   remain valid**, and ratios are what the optimisation work needs.

5. **Memory is not a constraint.** The Carphone tensor is 144x144x3x60 fp32
   ~= 15 MB, factors smaller. 6 GB is ample.
