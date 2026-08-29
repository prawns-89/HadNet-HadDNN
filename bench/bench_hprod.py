import numpy as np, itertools, time

# Carphone shape from the repo: A.permute(1,2,0,3) -> (H, W, frames, 3)
m = n = 144; p = (60, 3); r = 10          # 180 slices, rank 10
A = np.random.randn(m, r, *p).astype(np.float32)
B = np.random.randn(r, n, *p).astype(np.float32)

def h_prod_repo(A, B):                     # verbatim port of repo h_prod
    m, _, p = A.shape[0], A.shape[1], A.shape[2:]
    _, n, _ = B.shape[0], B.shape[1], B.shape[2:]
    C = np.empty((m, n) + p, dtype=A.dtype)
    for i in itertools.product(*[range(s) for s in p]):
        C[(...,) + i] = A[(...,) + i] @ B[(...,) + i]
    return C

def h_prod_batched(A, B):                  # batch dims moved to front, one GEMM
    nb = A.ndim - 2
    At = np.moveaxis(A, (0, 1), (-2, -1))  # (*p, m, r)
    Bt = np.moveaxis(B, (0, 1), (-2, -1))  # (*p, r, n)
    C = At @ Bt                            # batched
    return np.moveaxis(C, (-2, -1), (0, 1))

r1, r2 = h_prod_repo(A, B), h_prod_batched(A, B)
print("max abs diff:", np.abs(r1 - r2).max(), "| shapes", r1.shape, r2.shape)

def bench(f, n_rep=20):
    f(A, B)
    t = time.perf_counter()
    for _ in range(n_rep): f(A, B)
    return (time.perf_counter() - t) / n_rep

t_repo, t_batch = bench(h_prod_repo), bench(h_prod_batched)
n_slices = int(np.prod(p))
print(f"\nslices per h_prod call: {n_slices}")
print(f"repo (Python loop): {t_repo*1e3:8.2f} ms")
print(f"batched (1 GEMM)  : {t_batch*1e3:8.2f} ms")
print(f"speedup           : {t_repo/t_batch:8.1f}x")
print(f"\nper HadNet epoch = 2 h_prod calls -> {2*n_slices} matmuls")
print(f"over 2000 epochs (fwd only): {2*n_slices*2000:,} matmul calls")
print(f"est. fwd-only time  repo: {t_repo*2*2000:7.1f} s | batched: {t_batch*2*2000:7.1f} s")
