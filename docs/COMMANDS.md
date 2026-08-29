# Command reference

Every command needed to set up, diagnose and run this repo, with what each one
actually does. GPU commands are explained in detail — that is where the
surprises live.

Repo root: `/home/prawns/SOP/HadNet-HadDNN`

---

## 1. Environment

Ubuntu 24.04 enforces **PEP 668**: system Python is "externally managed" and
`pip install` is refused outright. 91 apt-managed Python packages live in the
system interpreter — `apt` itself among them — so pip and dpkg fighting over
`/usr/lib/python3/dist-packages` can break OS tooling. Hence a venv.

```bash
python3 -m venv .venv                     # isolated interpreter + site-packages
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

No `activate` needed — call the interpreter directly:

```bash
.venv/bin/python tests/test_ops.py
```

`requirements.txt` was derived from the repo's actual imports; upstream ships
none.

---

## 2. GPU / CUDA

Moved to **[GPU.md](GPU.md)** — driver diagnosis, CUDA wheels, disk
requirements, async-timing correctness, and outstanding GPU work.

---

## 3. Running the experiments

**No argparse anywhere** — every setting is hardcoded, so edit then run. Each
script uses relative data paths, so `cd` into its folder first.

| Folder | Run | Edit |
|---|---|---|
| `random-tensor` | `main.py` | `ranks` (L258) |
| `real-world` | `main.py` | `dataset_name` L48 (`miserables`/`spectrometer`), `ranks` L59 |
| `application-images` | `main_image.py` | `name_` L255 (`mri`/`cat`/`flower`/`monalisa`), `ranks` L265 |
| `application-videos` | `main_video.py` | `video_path` L262, `ranks` L284 |

```bash
cd application-videos && ../.venv/bin/python main_video.py
```

### Smoke test before any long run
A full video sweep is **hours**: Table 9's rank-10 row alone sums to ~2000s
across seven methods, and `ranks = [2,5,10,20,25]` costs more at each step.

```bash
cd application-videos
cp main_video.py /tmp/mv.bak
sed -i 's/^ranks = .*/ranks = [2]/; s/maxIter = 40000/maxIter = 500/' main_video.py
../.venv/bin/python main_video.py
cp /tmp/mv.bak main_video.py            # restore
```

`maxIter` governs AGD/IGD, which dominate wall-clock. To time only
HadNet/HadDNN, comment out the AGD/IGD/BCD blocks and the sweep drops from
hours to minutes.

---

## 4. Verifying the optimisations

```bash
.venv/bin/python tests/test_ops.py         # correctness + speed of the kernels
.venv/bin/python tests/test_lipschitz.py   # convergence-regime comparison
```

`test_ops.py` asserts `h_prod` is **bit-identical** to the original loop. If
that ever fails, the batching is wrong — treat it as a hard stop.

---

## 5. Baseline vs branch

You cannot claim a speedup without a "before" measured on the same hardware.

```bash
git checkout main
cd application-videos && ../.venv/bin/python main_video.py 2>&1 | tee ../bench/baseline.log
cd .. && git checkout perf/batch-hprod
cd application-videos && ../.venv/bin/python main_video.py 2>&1 | tee ../bench/optimised.log
diff <(grep Time ../bench/baseline.log) <(grep Time ../bench/optimised.log)
```

**The error / PSNR / SSIM columns must not move** for the `h_prod` change —
it is bit-identical. Early stopping and the randomized init *will* shift error
slightly (~1%); that is expected and documented in `docs/DECISIONS.md`.

---

## 6. Disk

CUDA torch is ~2.5 GB and the venv can reach 3.5 GB. A previous install died
with `[Errno 28] No space left on device`.

```bash
df -h /                        # free space
du -sh .venv ~/.cache/pip      # the two big reclaimables
.venv/bin/pip cache purge      # safe: just downloaded wheels
rm -rf .venv                   # safe: fully rebuildable from requirements.txt
```

Delete the old venv **before** installing the CUDA one, or you will need room
for both at once.

---

## 7. Git

```bash
git status --short
git diff --stat main                    # size of the change
git diff main -- hadamard_ops.py        # review one file
git checkout main -- <file>             # revert one file
```

`.venv/` and `__pycache__/` are gitignored. Nothing is committed yet — all
work sits in the working tree on `perf/batch-hprod`.
