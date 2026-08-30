# Command reference

GPU/CUDA lives in **[GPU.md](GPU.md)**. Research directions in
**[RESEARCH.md](RESEARCH.md)**. Findings and measurements in
**[DECISIONS.md](DECISIONS.md)**.

Repo root: `/home/prawns/SOP/HadNet-HadDNN`. Everything below assumes it.

---

## 1. Setup (once)

Ubuntu 24.04 enforces PEP 668, so system `pip install` is refused; 91
apt-managed Python packages share that interpreter, `apt` among them. Hence a
venv. No `activate` needed anywhere — call the interpreter directly, which also
avoids stale-shell problems.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # CPU torch
```

For GPU, install the CUDA wheel instead — see [GPU.md](GPU.md) §3.

---

## 2. Running a model — the one pattern

**All four experiment folders work identically.** There is no argparse
anywhere, so the loop is always: *edit the constants at the bottom of the
script, then run it from inside its own folder* (the data paths are relative).

```bash
cd <folder> && ../.venv/bin/python <script>
```

That is the whole interface. What differs between folders is only **which
constants exist** and **what data they point at**:

| folder | script | data | knobs (line) |
|---|---|---|---|
| `random-tensor` | `main.py` | synthetic `10x100x10x10` | `ranks` 239 |
| `real-world` | `main.py` | `datasets/` | `dataset_name` 48 (`miserables`\|`spectrometer`), `ranks` 59 |
| `application-images` | `main_image.py` | 4 PNGs | `name_` 246 (`mri`\|`cat`\|`flower`\|`monalisa`), `ranks` 256 |
| `application-videos` | `main_video.py` | `video.mp4` | `video_path` 243, `ranks` 265 |

Every script also has `maxIter = 40000` (line 276 / 63 / 327 / 391) governing
**AGD and IGD only**. Those two are numba-CPU and dominate wall-clock; they are
not what we optimise.

### Making a run fast

Two levers, both large:

```bash
sed -i 's/^ranks = .*/ranks = [10]/'      <script>   # one rank not five
sed -i 's/maxIter = 40000/maxIter = 2000/' <script>   # AGD/IGD budget
```

Rule of thumb on the RTX 4050: images are seconds, video is minutes with
`maxIter=2000` and hours at 40000. **Iterate on images, confirm on video.**

### Edit-run-restore, safely

```bash
cp main_video.py /tmp/mv.bak
sed -i 's/^ranks = .*/ranks = [2]/; s/maxIter = 40000/maxIter = 500/' main_video.py
../.venv/bin/python main_video.py
cp /tmp/mv.bak main_video.py
```

---

## 3. Tests

```bash
.venv/bin/python tests/test_ops.py          # kernel correctness + speed
.venv/bin/python tests/test_range.py        # [0,1] constraint experiment
.venv/bin/python tests/test_kfactors.py     # k>2, matched predicted rank
.venv/bin/python tests/test_kfactors_pareto.py   # k>2, error-vs-time frontier
.venv/bin/python tests/test_lipschitz.py    # step-size regimes
```

`test_ops.py` is the gate. It asserts `h_prod` is **exact in float64**,
bit-identical on CPU, and within 8 ulp on GPU. A failure there means the
batching is wrong — stop and fix before trusting any timing.

---

## 4. Baseline vs a branch

Never claim a speedup without a before-number on the same hardware. The
harness uses a **git worktree**, so the working repo never changes branch:

```bash
bash bench/run_compare.sh                    # writes bench/*.log
```

It runs images (full sweep) then video (`maxIter` reduced) on both trees. To
compare a different branch, edit `BASE`/`REPO` at the top.

Reading results — the scripts print
`<method> Rank <r>: Time <t>s, Error <e>, Rel Error <re>`:

```bash
grep "Rank" bench/images_optimised.log
diff <(grep Rank bench/images_baseline.log) <(grep Rank bench/images_optimised.log)
```

**What must not move:** baselines (SVD/QR/BCD/AGD/IGD) should show speedup
1.0x and error delta exactly 0.00000. They are the control. Movement there
means contamination, not improvement.

---

## 5. Branch-per-approach

One branch per idea, each cut **from `main`** — not from another branch, or
you cannot attribute the result.

```bash
git checkout main && git checkout -b <prefix>/<name>
```

| prefix | meaning | error may move? |
|---|---|---|
| `perf/` | speed only, maths unchanged | **no** — must be bit-identical |
| `opt/` | different optimiser or schedule | yes, and must be justified |
| `model/` | different model class | yes, expected |

Commit after each test, whether the result is good or bad — a measured
negative is worth more than an untested idea. `DECISIONS.md` is the record.

---

## 6. Housekeeping

```bash
git worktree list                  # baseline checkout lives in scratchpad
git worktree remove <path>         # when done
du -sh .venv ~/.cache/pip          # CUDA venv is ~7.3G
```
