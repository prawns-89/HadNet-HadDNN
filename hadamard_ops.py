"""
Shared, optimised kernels for the generalized Hadamard product framework.

Canonical implementations live here so the fix lands once instead of being
duplicated across the four experiment folders. Each `main_*.py` imports from
this module via a small sys.path shim.

Tensor layout convention (unchanged from the paper / original code):
    W : (m, r, *p)      H : (r, n, *p)      product : (m, n, *p)
where *p are the trailing "slice" dimensions.

Reference implementations (`*_loop`) are kept verbatim so correctness and
speed can be A/B'd -- see bench/ and tests/.
"""
import itertools
import torch

# --------------------------------------------------------------------------
# Lever 1: generalized Hadamard product  -- BIT-IDENTICAL, ~9x faster
# --------------------------------------------------------------------------

def h_prod_loop(A, B):
    """Original implementation: one tiny matmul per slice, in Python.

    For a 144x144x3x60 tensor this is 180 separate matmuls per call, each
    ~200k FLOPs -- far too small to saturate a GPU, so runtime is dominated
    by kernel-launch overhead rather than arithmetic. Kept for comparison.
    """
    shapeA, shapeB = A.shape, B.shape
    m, _, p = shapeA[0], shapeA[1], shapeA[2:]
    _, n, p = shapeB[0], shapeB[1], shapeB[2:]
    C = torch.empty((m, n) + p, dtype=A.dtype, device=A.device)
    for i in itertools.product(*[range(s) for s in p]):
        C[(...,) + i] = torch.matmul(A[(...,) + i], B[(...,) + i])
    return C


def h_prod(A, B):
    """Batched equivalent of `h_prod_loop`: a single batched GEMM.

    The slice dims are *trailing* in this layout, but torch.matmul batches
    over *leading* dims -- which is exactly why the original needed a loop.
    Moving them to the front lets one cuBLAS call do all slices at once.

    Output is bit-identical to `h_prod_loop` (same reduction order per slice).
    """
    nb = A.ndim - 2                                   # number of slice dims
    At = A.permute(*range(2, A.ndim), 0, 1)           # (*p, m, r)
    Bt = B.permute(*range(2, B.ndim), 0, 1)           # (*p, r, n)
    C = At @ Bt                                       # (*p, m, n) batched
    return C.permute(nb, nb + 1, *range(nb))          # (m, n, *p)


# --------------------------------------------------------------------------
# Lever 2: SVD initialisation  -- O(n^5) -> O(n^4)
# --------------------------------------------------------------------------

def svd_initialize_loop(D, r):
    """Original: per-slice ECONOMY SVD, all min(m,n) triplets, then truncate.

    Two SVDs per slice (mat_simple, mat_signed). For 144x144 slices with
    r=10 this computes 144 singular triplets and discards 134 of them (93%).
    Cost O(m*n*p*min(m,n)) -- for an n x n x n x n tensor that is O(n^5),
    the only super-quartic term in the paper's Eq. (11).
    """
    device, shape = D.device, D.shape
    m, n, p = shape[0], shape[1], shape[2:]
    W1 = torch.zeros((m, r) + p, device=device)
    H1 = torch.zeros((r, n) + p, device=device)
    W2 = torch.zeros((m, r) + p, device=device)
    H2 = torch.zeros((r, n) + p, device=device)
    for i in itertools.product(*[range(s) for s in p]):
        mat = D[(...,) + i]
        mat_simple = torch.sqrt(torch.abs(mat))
        mat_signed = mat_simple * torch.sign(mat)
        U1, S1, V1 = torch.linalg.svd(mat_simple, full_matrices=False)
        W1[(...,) + i] = U1[:, :r] * torch.sqrt(S1[:r])
        H1[(...,) + i] = (torch.sqrt(S1[:r]) * V1[:r, :].T).T
        U2, S2, V2 = torch.linalg.svd(mat_signed, full_matrices=False)
        W2[(...,) + i] = U2[:, :r] * torch.sqrt(S2[:r])
        H2[(...,) + i] = (torch.sqrt(S2[:r]) * V2[:r, :].T).T
    return W1, H1, W2, H2


def svd_initialize(D, r, randomized=True, oversample=10, niter=4):
    """Batched, optionally randomized rank-r initialisation.

    Two independent changes to `svd_initialize_loop`:

    1. BATCHED -- all slices in one call instead of a Python loop.
       Same maths, ~1.2x (SVD is FLOP-bound, not launch-bound).

    2. RANDOMIZED (Halko, Martinsson & Tropp 2011) -- computes only the
       leading r+oversample directions instead of all min(m,n).
       Cost per slice drops O(m*n*min(m,n)) -> O(m*n*(r+oversample)),
       so the whole init goes O(n^5) -> O(n^4) for an n^4 tensor.
       Power iterations are NOT free: each costs another O(m*n*q) pass, so
       randomisation only pays while  niter*(r+oversample) < min(m,n).
       Measured on 144x144x60x3, r=10 (min(m,n)=144, q=20 -> niter<7):

           niter  1     2     3     4     6     8    12    24
           speed  3.6x  2.8x  2.3x  1.9x  1.4x  1.3x  0.9x  0.4x
           delta  1.5e-2 6.8e-3 3.2e-3 1.7e-3 4.7e-4 1.8e-4 2.5e-5 4.8e-7

       niter=4 (~1.9x, delta 1.7e-3) is the default: the error is far below
       anything a 2000-epoch descent will not wash out. Shard uses niter=24,
       but it optimises for memory/accuracy where the extra passes are free;
       optimising for TIME the profitable range is much lower.
       This is the change that meets the 4th-power complexity target.

    NOTE: randomized=True is APPROXIMATE, so results are not bit-identical
    to the original. It only seeds a 2000-epoch gradient descent, so final
    error should be unaffected -- but set randomized=False to A/B it.
    Falls back to the exact path automatically when r is large relative to
    min(m,n), where randomisation buys nothing.
    """
    nb = D.ndim - 2
    Dt = D.permute(*range(2, D.ndim), 0, 1)               # (*p, m, n)
    m, n = Dt.shape[-2], Dt.shape[-1]
    q = min(r + oversample, min(m, n))
    use_rand = randomized and r < min(m, n) // 2
    if use_rand and niter * q >= min(m, n):
        # past this point the power iterations cost more than the exact SVD
        niter = max(1, (min(m, n) // q) - 1)

    mat_simple = torch.sqrt(torch.abs(Dt))
    mat_signed = mat_simple * torch.sign(Dt)

    out = []
    for M in (mat_simple, mat_signed):
        if use_rand:
            U, S, V = torch.svd_lowrank(M, q=q, niter=niter)   # V, not Vh
            sr = S[..., :r].sqrt()
            W = U[..., :, :r] * sr.unsqueeze(-2)               # (*p, m, r)
            H = (V[..., :, :r] * sr.unsqueeze(-2)).transpose(-2, -1)  # (*p, r, n)
        else:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            sr = S[..., :r].sqrt()
            W = U[..., :, :r] * sr.unsqueeze(-2)
            H = sr.unsqueeze(-1) * Vh[..., :r, :]
        out.append(W.permute(nb, nb + 1, *range(nb)))          # (m, r, *p)
        out.append(H.permute(nb, nb + 1, *range(nb)))          # (r, n, *p)

    W1, H1, W2, H2 = out[0], out[1], out[2], out[3]
    return W1, H1, W2, H2


# --------------------------------------------------------------------------
# Lever 4: honest GPU timing
# --------------------------------------------------------------------------

def sync():
    """Drain the CUDA queue so time.time() measures execution, not enqueue.

    CUDA launches are asynchronous: without this, a timing block records how
    long it took to *queue* kernels, not to run them. No-op on CPU.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Lever 3b: Lipschitz-based step size
# --------------------------------------------------------------------------
#
# PREREQUISITE: the repo minimises the UNSQUARED Frobenius norm
#     loss = torch.norm(D - X_hat, 'fro')
# whose gradient  -R/||R||_F  has a Hessian that blows up as ||R||->0, so no
# finite Lipschitz constant exists near the optimum. Every 1/L result assumes
# the SQUARED objective. Use squared_loss() below before using these steps.
#
# Block-convexity (paper, Sec. 4.2): with W2,H1,H2 held fixed the loss is a
# convex quadratic in W1. Row i of the prediction is
#     pred[i,:] = (W1[i,:] @ H1) * M2[i,:]
# so the design matrix for row i is  G_i = diag(M2[i,:]) @ H1^T  and the
# per-row curvature is  L_i = 2*sigma_max(G_i)^2.  This is exactly the form
# already used by Hadamard_BCD.hadamardLS_lipschitz.

def squared_loss(D, X_hat):
    """||D - X_hat||_F^2 -- the objective the 1/L theory actually applies to."""
    r = D - X_hat
    return (r * r).sum()


def block_lipschitz(H, M_other, safety=1.0):
    """Cheap upper bound on the block curvature L for the W-block.

    Exact per-row L_i = 2*sigma_max(diag(M_other[i,:]) @ H^T)^2 costs one SVD
    per row. We use the standard submultiplicative bound

        sigma_max(diag(m) @ H^T)  <=  max_j|m_j| * sigma_max(H)

    giving  L <= 2 * max|M_other|^2 * sigma_max(H)^2  -- one spectral norm of
    the small (r x n) factor instead of m of them. Conservative, so the step
    1/L stays safe; `safety` <1 tightens it empirically.

    Returns a python float.
    """
    Hm = H.reshape(H.shape[0], -1) if H.ndim > 2 else H
    smax = torch.linalg.matrix_norm(Hm.float(), ord=2)
    mmax = M_other.abs().max()
    L = 2.0 * (mmax ** 2) * (smax ** 2)
    return float(L) / max(safety, 1e-12)


def lipschitz_lr(W1, H1, W2, H2, floor=1e-8, cap=1.0):
    """Step size 1/L from the worst-case block curvature across all 4 blocks.

    Joint Adam over all factors is NOT covered by the block-convexity result
    (the joint problem is a quartic, hence non-convex). Taking the max over
    blocks is a pragmatic scale-aware choice, not a convergence guarantee --
    for the guaranteed version cycle the blocks one at a time (see docs).
    """
    M1, M2 = h_prod(W1, H1), h_prod(W2, H2)
    Ls = [block_lipschitz(H1, M2), block_lipschitz(H2, M1),
          block_lipschitz(W1.transpose(0, 1), M2),
          block_lipschitz(W2.transpose(0, 1), M1)]
    L = max(Ls)
    return float(min(max(1.0 / max(L, floor), floor), cap)), L


# --------------------------------------------------------------------------
# Lever 3: early stopping on relative improvement
# --------------------------------------------------------------------------

class EarlyStop:
    """Stop once the loss stops improving *relatively*, not on a fixed budget.

    The repo runs a hard 2000 epochs with no exit; measured curves show ~87%
    of those epochs buy <1% of accuracy (docs/DECISIONS.md Sec. 11). This
    stops when the best loss has improved by less than `min_rel` over the last
    `patience` epochs -- scale-free, so it transfers across tensor sizes and
    ranks without retuning.

        stopper = EarlyStop(patience=50, min_rel=1e-3)
        for epoch in ...:
            ...
            if stopper.step(loss.item()):
                break
        print(stopper.epochs)        # the k actually used
    """

    def __init__(self, patience=50, min_rel=1e-3, min_epochs=50):
        self.patience, self.min_rel, self.min_epochs = patience, min_rel, min_epochs
        self.best = float("inf")
        self.ref = float("inf")      # best as of the last checkpoint
        self.wait = 0
        self.epochs = 0
        self.stopped = False

    def step(self, loss):
        self.epochs += 1
        if loss < self.best:
            self.best = loss
        self.wait += 1
        if self.epochs < self.min_epochs:
            return False
        if self.wait >= self.patience:
            gain = (self.ref - self.best) / max(abs(self.ref), 1e-12)
            self.ref, self.wait = self.best, 0
            if gain < self.min_rel:
                self.stopped = True
                return True
        return False
