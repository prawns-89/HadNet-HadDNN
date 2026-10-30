"""
Run stiefel_hadDec over a range of ranks and log the results.

Put this file next to stiefel_hadamard.py (real-world/ or application-images/).

Examples
    python run_stiefel.py --data miserables
    python run_stiefel.py --data spectrometer --ranks 2 5 10 25
    python run_stiefel.py --data ../application-images/128mri.png --ranks 5 10 25 50
    python run_stiefel.py --data miserables --momentum nesterov --maxiter 20000
    python run_stiefel.py --data 128cat.png --ranks 5 10 20 30 --save-images

For images, reconstructions are saved to logs/<name>_<time>_recon/ with
--save-images (Hadamard and SVD at matched storage, rank 2r, for each rank).

Outputs go to logs/ (created if missing), one set per run, named by data + timestamp:
    <name>_<time>.log    everything printed to the screen
    <name>_<time>.csv    one row per rank: error, PSNR, storage, time, iterations, settings
    <name>_<time>.json   the same plus the full error-vs-iteration curve per rank
"""
import argparse, csv, json, os, sys, time
import numpy as np
from stiefel_hadamard import stiefel_hadDec


def load(data):
    if data == "miserables":
        import networkx as nx
        G = nx.read_gml("datasets/lesmis.gml")
        return nx.to_numpy_array(G)          # 0/1 adjacency, same as main.py
    if data == "miserables_weighted":
        import networkx as nx
        return nx.to_numpy_array(nx.read_gml("datasets/lesmis.gml"), weight="value")
    if data == "spectrometer":
        with open("datasets/lrs.data") as f:
            rows = f.read().split("(")[1:]
        return np.array([[float(v.rstrip(")")) for v in row.split()[2:]] for row in rows])
    if os.path.isfile(data):                  # any image file -> grayscale in [0, 1]
        from PIL import Image
        return np.asarray(Image.open(data).convert("L"), dtype=float) / 255.0
    sys.exit(f"Unknown --data '{data}': use miserables, miserables_weighted, spectrometer or an image path")


def psnr(X, Y, peak, clip=None):
    """PSNR in dB = 10 log10(peak^2 / MSE). Images: peak 1, reconstruction clipped
    to [0, 1] (as when saving the image). Other data: peak = max(X) - min(X),
    no clipping, unless --peak is given."""
    if clip is not None:
        Y = np.clip(Y, *clip)
    mse = np.mean((X - Y) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def svd_recon(U, s, Vt, k):
    return (U[:, :k] * s[:k]) @ Vt[:k]


class Tee:
    """Print to the screen and to the log file at the same time."""
    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout
    def write(self, s):
        self.out.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.out.flush(); self.f.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="miserables")
    p.add_argument("--ranks", type=int, nargs="+", default=[2, 5, 10, 25, 40, 50])
    p.add_argument("--maxiter", type=int, default=10000)
    p.add_argument("--momentum", default="heavy", choices=["heavy", "nesterov", "none"])
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--no-precond", action="store_true")
    p.add_argument("--break-symmetry", type=float, default=1e-2)
    p.add_argument("--init", default="svd", choices=["svd", "random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-limit", type=float, default=200.0)
    p.add_argument("--peak", type=float, default=None,
                   help="PSNR peak value (default: 1 for images, max(X)-min(X) otherwise)")
    p.add_argument("--save-images", action="store_true",
                   help="images only: save the Hadamard and SVD-2r reconstructions as PNGs")
    p.add_argument("--outdir", default="logs")
    a = p.parse_args()

    X = load(a.data)
    name = os.path.splitext(os.path.basename(a.data))[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(a.outdir, exist_ok=True)
    base = os.path.join(a.outdir, f"{name}_{stamp}")
    sys.stdout = Tee(base + ".log")

    settings = dict(momentum=a.momentum, beta=a.beta, precondition=not a.no_precond,
                    break_symmetry=a.break_symmetry, init=a.init, seed=a.seed,
                    maxiter=a.maxiter, time_limit=a.time_limit)
    normX = np.linalg.norm(X)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    is_image = os.path.isfile(a.data)
    peak = a.peak if a.peak is not None else (1.0 if is_image else float(X.max() - X.min()))
    clip = (0.0, 1.0) if is_image else None
    settings["psnr_peak"] = peak
    print(f"Data: {a.data}  shape {X.shape}  ||X||_F = {normX:.6f}  PSNR peak = {peak:g}")
    print(f"Settings: {settings}\n")
    m, n = X.shape
    if a.save_images and is_image:
        from PIL import Image
        rdir = base + "_recon"
        os.makedirs(rdir, exist_ok=True)
        save = lambda Z, f: Image.fromarray((np.clip(Z, 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(rdir, f))
        save(X, "original.png")
    fmt = lambda v: "   exact" if v == "exact" else f"{v:8.2f}"
    print("storage = numbers stored / entries in X (2r(m+n)/mn); above 100% it is not compression")
    print("PSNR is shown as 'exact' when the relative error is below 1e-10\n")
    print(f"{'rank':>5} {'storage':>8} {'rel.err':>10} {'PSNR':>8} {'SVD r':>9} {'PSNR':>8} {'SVD 2r':>9} {'PSNR':>8} {'time(s)':>9} {'iters':>7}")

    rows, curves = [], {}
    for r in a.ranks:
        t0 = time.time()
        W1, H1, W2, H2, err, times = stiefel_hadDec(
            X, r, maxiter=a.maxiter, momentum=a.momentum, beta=a.beta,
            precondition=not a.no_precond, break_symmetry=a.break_symmetry,
            init=a.init, seed=a.seed, time_limit=a.time_limit)
        t = time.time() - t0
        Y = (W1 @ H1) * (W2 @ H2)
        e = np.linalg.norm(X - Y) / normX
        svd_r = np.sqrt(np.sum(s[r:] ** 2)) / normX
        svd_2r = np.sqrt(np.sum(s[2 * r:] ** 2)) / normX
        exact = lambda rel, v: "exact" if rel < 1e-10 else v
        p_had = exact(e, psnr(X, Y, peak, clip))
        p_r = exact(svd_r, psnr(X, svd_recon(U, s, Vt, r), peak, clip))
        p_2r = exact(svd_2r, psnr(X, svd_recon(U, s, Vt, 2 * r), peak, clip))
        storage = 2 * r * (m + n) / (m * n)
        flag = "" if storage < 1 else "  (not compression)"
        print(f"{r:5d} {100*storage:7.1f}% {e:10.6f} {fmt(p_had)} {svd_r:9.6f} {fmt(p_r)} {svd_2r:9.6f} {fmt(p_2r)} "
              f"{t:9.2f} {len(err) - 1:7d}{flag}")
        if a.save_images and is_image:
            save(Y, f"hadamard_r{r:03d}.png")
            save(svd_recon(U, s, Vt, 2 * r), f"svd_r{2*r:03d}_matched.png")
        rows.append(dict(data=a.data, rank=r, storage_ratio=storage, rel_error=e, abs_error=e * normX, psnr_db=p_had,
                         svd_r=svd_r, svd_r_psnr_db=p_r, svd_2r=svd_2r, svd_2r_psnr_db=p_2r,
                         time_s=t, iters=len(err) - 1, **settings))
        curves[r] = dict(rel_error=[float(v) for v in err], time_s=[float(v) for v in times])

    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(base + ".json", "w") as f:
        json.dump(dict(data=a.data, shape=list(X.shape), settings=settings,
                       results=rows, curves=curves), f)
    print(f"\nSaved {base}.log, .csv, .json" + (f" and images in {base}_recon/" if a.save_images and is_image else ""))