"""Correctness + speed check for the optimised kernels in hadamard_ops.py.

Run:  .venv/bin/python tests/test_ops.py
Works on CPU; on GPU the speedups should be larger (launch-bound -> batched).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hadamard_ops import (h_prod, h_prod_loop,
                          svd_initialize, svd_initialize_loop, sync)

torch.manual_seed(42)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = n = 144; p = (60, 3); r = 10          # Carphone shape from Table 9
print(f"device={dev}  shape=({m},{n},{p[0]},{p[1]})  r={r}  "
      f"slices={p[0]*p[1]}\n")

def bench(fn, *a, reps=5, **kw):
    fn(*a, **kw); sync()
    t = time.time()
    for _ in range(reps): fn(*a, **kw)
    sync()
    return (time.time() - t) / reps

ok = True

# ---- Lever 1: h_prod must be BIT-IDENTICAL -------------------------------
A = torch.randn(m, r, *p, device=dev)
B = torch.randn(r, n, *p, device=dev)
ref, got = h_prod_loop(A, B), h_prod(A, B)
exact = torch.equal(ref, got)
maxdiff = (ref - got).abs().max().item()
print("[1] h_prod")
print(f"    shape {tuple(got.shape)}  bit-identical={exact}  maxdiff={maxdiff:.3e}")
t_loop, t_batch = bench(h_prod_loop, A, B), bench(h_prod, A, B)
print(f"    loop {t_loop*1e3:8.2f} ms   batched {t_batch*1e3:8.2f} ms"
      f"   -> {t_loop/t_batch:5.1f}x")
if not exact:
    ok = False; print("    FAIL: not bit-identical")

# ---- Lever 2a: batched exact SVD must match the loop ---------------------
D = torch.randn(m, n, *p, device=dev)
print("\n[2] svd_initialize (randomized=False) vs loop")
ref = svd_initialize_loop(D, r)
got = svd_initialize(D, r, randomized=False)
worst = max((a - b).abs().max().item() for a, b in zip(ref, got))
match = all(torch.allclose(a, b, atol=1e-4) for a, b in zip(ref, got))
print(f"    factors match={match}  worst abs diff={worst:.3e}")
if not match:
    ok = False; print("    FAIL: exact path diverges from loop")

# ---- Lever 2b: randomized path -- approximate, judged on reconstruction --
print("\n[3] svd_initialize (randomized=True) -- O(n^5)->O(n^4)")
def recon_err(fac):
    W1, H1, W2, H2 = fac
    return (torch.norm(D - h_prod(W1, H1) * h_prod(W2, H2)) /
            torch.norm(D)).item()
e_loop = recon_err(ref)
e_rand = recon_err(svd_initialize(D, r, randomized=True))
print(f"    init rel. error   exact={e_loop:.6f}   randomized={e_rand:.6f}"
      f"   delta={abs(e_rand-e_loop):.2e}")
print("    (approximate by design; it only seeds 2000 epochs of descent)")

t_loop  = bench(svd_initialize_loop, D, r, reps=2)
t_exact = bench(svd_initialize, D, r, reps=2, randomized=False)
t_rand  = bench(svd_initialize, D, r, reps=2, randomized=True)
print(f"    loop {t_loop*1e3:8.1f} ms | batched-exact {t_exact*1e3:8.1f} ms"
      f" | randomized {t_rand*1e3:8.1f} ms  -> {t_loop/t_rand:.1f}x")

print("\n" + ("PASS - correctness checks green" if ok else "FAIL"))
sys.exit(0 if ok else 1)
