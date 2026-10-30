"""
Benchmark: Stiefel-constrained momentum solver vs the reference methods.

Same layout as main.py. Runs on
  - "miserables": Les Misérables co-occurrence graph. Loaded from networkx's
    built-in copy so no datasets/ folder is needed. weight=None gives the 0/1
    adjacency; see the note in load_dataset about why that matches main.py.
  - "miserables_weighted": same graph with co-occurrence counts as weights.
  - "planted": synthetic X = (A1 B1) * (A2 B2) + noise with known ranks, to
    check recovery.

Usage:
    python benchmark.py                      # all datasets, default ranks
    python benchmark.py miserables 2 5 10    # one dataset, chosen ranks
"""

import sys, time, csv
import numpy as np
import networkx as nx
import torch

from stiefel_hadamard import stiefel_hadDec
from Hadamard_BCD import Hadamard_BCD
import Hadamard_matrix
from Hadamard_matrix import train_Net

import warnings
warnings.filterwarnings("ignore")


def load_dataset(name, seed=0):
    if name == "miserables":
        # main.py does nx.to_numpy_array(nx.read_gml('lesmis.gml')). In that GML
        # file the edge weights are stored under 'value', but to_numpy_array reads
        # 'weight' by default and uses 1 when it's missing, so main.py most likely
        # benchmarks the 0/1 adjacency. Node order differs from the GML file, but
        # a simultaneous row/column permutation leaves every error unchanged.
        return nx.to_numpy_array(nx.les_miserables_graph(), weight=None)
    if name == "miserables_weighted":
        return nx.to_numpy_array(nx.les_miserables_graph(), weight="weight")
    if name == "planted":
        rng = np.random.default_rng(seed)
        m, n, r0 = 200, 150, 5
        A1, B1 = rng.standard_normal((m, r0)), rng.standard_normal((r0, n))
        A2, B2 = rng.standard_normal((m, r0)), rng.standard_normal((r0, n))
        X = (A1 @ B1) * (A2 @ B2)
        return X + 1e-3 * np.linalg.norm(X) / np.sqrt(m * n) * rng.standard_normal((m, n))
    raise ValueError(name)


def rel(D, X_hat):
    return np.linalg.norm(D - X_hat) / np.linalg.norm(D)


def run_all(D, r):
    rows = []
    s = np.linalg.svd(D, compute_uv=False)
    nD = np.linalg.norm(D)
    # Truncated SVD at rank r, and at rank 2r (same parameter count as two rank-r factors)
    rows.append(("SVD rank r", np.sqrt(np.sum(s[r:] ** 2)) / nD, 0.0, None))
    rows.append(("SVD rank 2r", np.sqrt(np.sum(s[2 * r:] ** 2)) / nD, 0.0, None))

    t = time.time()
    W1, H1, W2, H2, err, _ = Hadamard_BCD(D, r=r)
    rows.append(("BCD (reference)", rel(D, (W1 @ H1) * (W2 @ H2)), time.time() - t, len(err) - 1))

    torch.manual_seed(0)
    t = time.time()
    _, W1, H1, W2, H2 = train_Net(D, r)
    rows.append(("HadNet Adam (reference)", rel(D, (W1 @ H1) * (W2 @ H2)), time.time() - t, 10000))

    # Same HadNet, but with the M1 == M2 symmetry broken at init (same noise
    # recipe as stiefel_hadDec). Patches the module-level svd_init it calls.
    orig = Hadamard_matrix.svd_init
    def svd_init_broken(X, r, **kw):
        W1, H1, W2, H2 = orig(X, r, **kw)
        rng = np.random.default_rng(12345)
        return (W1, H1, W2 + 1e-2 * np.linalg.norm(W2) / np.sqrt(W2.size) * rng.standard_normal(W2.shape),
                H2 + 1e-2 * np.linalg.norm(H2) / np.sqrt(H2.size) * rng.standard_normal(H2.shape))
    Hadamard_matrix.svd_init = svd_init_broken
    torch.manual_seed(0)
    t = time.time()
    _, W1, H1, W2, H2 = train_Net(D, r)
    rows.append(("HadNet Adam + symmetry break", rel(D, (W1 @ H1) * (W2 @ H2)), time.time() - t, 10000))
    Hadamard_matrix.svd_init = orig

    variants = [
        ("Stiefel heavy-ball",          dict(momentum="heavy")),
        ("Stiefel heavy, symmetric init", dict(momentum="heavy", break_symmetry=0)),
        ("Stiefel Nesterov",            dict(momentum="nesterov")),
        ("Stiefel no momentum",         dict(momentum="none")),
        ("Stiefel heavy, no precond",   dict(momentum="heavy", precondition=False)),
        ("Stiefel heavy, no balance",   dict(momentum="heavy", balance=False)),
    ]
    for name, kw in variants:
        t = time.time()
        W1, H1, W2, H2, err, _ = stiefel_hadDec(D, r, maxiter=10000, **kw)
        rows.append((name, rel(D, (W1 @ H1) * (W2 @ H2)), time.time() - t, len(err) - 1))

    # Random orthonormal init, best of 3 seeds (time = total over the 3 runs)
    t, best, iters = time.time(), np.inf, 0
    for seed in range(3):
        W1, H1, W2, H2, err, _ = stiefel_hadDec(D, r, maxiter=10000, init="random", seed=seed)
        best = min(best, rel(D, (W1 @ H1) * (W2 @ H2)))
        iters += len(err) - 1
    rows.append(("Stiefel heavy, random init x3", best, time.time() - t, iters))
    return rows


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    datasets = {"miserables": [2, 5, 10, 25, 40],
                "miserables_weighted": [2, 5, 10, 25],
                "planted": [5]}
    if len(sys.argv) > 1:
        datasets = {sys.argv[1]: [int(a) for a in sys.argv[2:]] or datasets.get(sys.argv[1], [5])}

    out = []
    for ds, ranks in datasets.items():
        D = load_dataset(ds)
        print(f"\nDataset: {ds}, shape {D.shape}, ||D||_F = {np.linalg.norm(D):.4f}")
        for r in ranks:
            print(f"\n  rank {r}")
            print(f"  {'method':30s} {'rel. error':>11s} {'time (s)':>9s} {'iters':>7s}")
            for name, e, t, it in run_all(D, r):
                print(f"  {name:30s} {e:11.6f} {t:9.2f} {'' if it is None else it:>7}")
                out.append(dict(dataset=ds, rank=r, method=name, rel_error=e, time_s=t, iters=it))

    with open("benchmark_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("\nSaved benchmark_results.csv")
