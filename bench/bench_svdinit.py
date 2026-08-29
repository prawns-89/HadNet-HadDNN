import numpy as np, itertools, time
m = n = 144; p = (60, 3); r = 10
D = np.random.randn(m, n, *p).astype(np.float32)

def svd_init_repo(D, r):                    # port of repo svd_initialize
    p = D.shape[2:]
    W1 = np.zeros((m, r) + p, np.float32); H1 = np.zeros((r, n) + p, np.float32)
    W2 = np.zeros((m, r) + p, np.float32); H2 = np.zeros((r, n) + p, np.float32)
    for i in itertools.product(*[range(s) for s in p]):
        mat = D[(...,) + i]
        ms = np.sqrt(np.abs(mat)); mg = ms * np.sign(mat)
        U1, S1, V1 = np.linalg.svd(ms, full_matrices=False)   # ALL 144 triplets
        W1[(...,)+i] = U1[:, :r] * np.sqrt(S1[:r])
        H1[(...,)+i] = (np.sqrt(S1[:r]) * V1[:r, :].T).T
        U2, S2, V2 = np.linalg.svd(mg, full_matrices=False)   # ALL 144 triplets
        W2[(...,)+i] = U2[:, :r] * np.sqrt(S2[:r])
        H2[(...,)+i] = (np.sqrt(S2[:r]) * V2[:r, :].T).T
    return W1, H1, W2, H2

def svd_init_batched(D, r):                 # batched over slices, still full
    Dt = np.moveaxis(D, (0, 1), (-2, -1))                     # (*p, m, n)
    out = []
    for M in (np.sqrt(np.abs(Dt)), np.sqrt(np.abs(Dt))*np.sign(Dt)):
        U, S, V = np.linalg.svd(M, full_matrices=False)       # batched
        sr = np.sqrt(S[..., :r])
        W = U[..., :, :r] * sr[..., None, :]
        H = sr[..., :, None] * V[..., :r, :]
        out += [np.moveaxis(W, (-2, -1), (0, 1)), np.moveaxis(H, (-2, -1), (0, 1))]
    return out[0], out[1], out[2], out[3]

a = svd_init_repo(D, r); b = svd_init_batched(D, r)
print("factors match:", all(np.allclose(x, y, atol=1e-4) for x, y in zip(a, b)))

def bench(f, k=3):
    f(D, r); t = time.perf_counter()
    for _ in range(k): f(D, r)
    return (time.perf_counter() - t)/k

tr, tb = bench(svd_init_repo), bench(svd_init_batched)
print(f"\nrepo  (180-slice loop, 360 economy SVDs): {tr*1e3:7.1f} ms")
print(f"batched (2 batched SVD calls)          : {tb*1e3:7.1f} ms   -> {tr/tb:.1f}x")
print(f"\nrank kept r={r} of min(m,n)={min(m,n)} computed  ->  {100*(1-r/min(m,n)):.0f}% of each SVD discarded")
print(f"init as share of HadNet's 57.36 s reported run: {100*tr/57.36:.1f}%")
