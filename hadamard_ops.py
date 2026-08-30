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


# --------------------------------------------------------------------------
# k-factor Hadamard decomposition
# --------------------------------------------------------------------------
#
# Generalises  X ~ (W1 H1) o (W2 H2)  to a product of k terms:
#
#     X  ~  (W1 H1) o (W2 H2) o ... o (Wk Hk)
#
# With each factor of rank r the effective rank is up to r^k, using kr(m+n)
# parameters and ~k*r*mnp per epoch. So to reach a TARGET effective rank R you
# need r = R^(1/k), and the per-epoch cost scales as
#
#     f(k) = k * R^(1/k)
#
# Minimising:  d/dk [ln k + (ln R)/k] = 1/k - ln R/k^2 = 0  =>  k* = ln R.
#
# Wertz, Vandaele & Gillis (arXiv:2504.13633) extend the MATRIX case to k>2;
# the tensor case and this optimal-k tradeoff appear to be open.

def h_prod_k(factors):
    """Hadamard product of k low-rank terms.

    factors: [(W1,H1), (W2,H2), ...] with W: (m,r,*p), H: (r,n,*p)
    """
    out = h_prod(*factors[0])
    for W, H in factors[1:]:
        out = out * h_prod(W, H)
    return out


def svd_initialize_k(D, r, k, randomized=True, niter=4):
    """Rank-r initialisation for each of k factor pairs.

    For k=2 the existing code uses sqrt(|X|), i.e. |X|^(1/2), so that the
    product of two factors recovers X. Generalising: each factor targets
    |X|^(1/k), with a single factor carrying sign(X) so the product's sign is
    correct and the other k-1 stay positive.
    """
    nb = D.ndim - 2
    Dt = D.permute(*range(2, D.ndim), 0, 1)                 # (*p, m, n)
    m, n = Dt.shape[-2], Dt.shape[-1]
    root = Dt.abs().clamp_min(1e-12).pow(1.0 / k)
    q = min(r + 10, min(m, n))
    use_rand = randomized and r < min(m, n) // 2
    if use_rand and niter * q >= min(m, n):
        niter = max(1, (min(m, n) // q) - 1)

    out = []
    for j in range(k):
        M = root * torch.sign(Dt) if j == 0 else root
        if use_rand:
            U, S, V = torch.svd_lowrank(M, q=q, niter=niter)
            sr = S[..., :r].sqrt()
            W = U[..., :, :r] * sr.unsqueeze(-2)
            H = (V[..., :, :r] * sr.unsqueeze(-2)).transpose(-2, -1)
        else:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            sr = S[..., :r].sqrt()
            W = U[..., :, :r] * sr.unsqueeze(-2)
            H = sr.unsqueeze(-1) * Vh[..., :r, :]
        out.append((W.permute(nb, nb + 1, *range(nb)),
                    H.permute(nb, nb + 1, *range(nb))))
    return out


def optimal_k(R):
    """Predicted cost-optimal number of factors for target effective rank R."""
    import math
    return math.log(R)


# --------------------------------------------------------------------------
# Anderson acceleration / DIIS  (RESEARCH.md Sec. 8.1 / 9.2)
# --------------------------------------------------------------------------
#
# The epoch loop is a fixed-point iteration x <- g(x). Sec. 17's curves show
# ~87% of epochs buy <1% of accuracy -- the slow tail Anderson exists to
# collapse. Identical to Pulay's DIIS for SCF convergence in electronic
# structure codes.
#
# Keep the last m iterates and their residuals f_i = g(x_i) - x_i; find the
# affine combination (sum(alpha) = 1) minimising ||sum alpha_i f_i||, and
# extrapolate. One small m x m least-squares per step.
#
# CAVEAT: wrapping Adam is not textbook Anderson -- Adam carries momentum and
# second-moment state that the extrapolated point invalidates. Whether that
# destabilises it is exactly what the test measures.

class Anderson:
    """Anderson/DIIS extrapolation over a parameter sequence.

        acc = Anderson(m=5)
        ...
        opt.step()
        acc.apply_(params)      # in place, after the base optimiser step
    """

    def __init__(self, m=5, beta=1.0, reg=1e-8, start=20, every=1):
        self.m, self.beta, self.reg = m, beta, reg
        self.start, self.every = start, every
        self.X, self.G = [], []
        self.n = 0

    @torch.no_grad()
    def apply_(self, params):
        self.n += 1
        flat = torch.cat([p.detach().reshape(-1) for p in params])
        if self.X:
            self.G.append(flat.clone())          # g(x_prev) = current point
        self.X.append(flat.clone())
        if len(self.X) > self.m + 1:
            self.X.pop(0)
        if len(self.G) > self.m:
            self.G.pop(0)
        if self.n < self.start or self.n % self.every or len(self.G) < 2:
            return False

        k = len(self.G)
        F = torch.stack([self.G[i] - self.X[i] for i in range(k)])     # (k, P)
        # min ||F^T a|| s.t. sum a = 1  ->  solve (FF^T + reg I) a = 1, normalise
        A = F @ F.transpose(0, 1)
        A = A + self.reg * torch.eye(k, device=A.device, dtype=A.dtype) * A.diagonal().mean().clamp_min(1e-30)
        ones = torch.ones(k, device=A.device, dtype=A.dtype)
        try:
            a = torch.linalg.solve(A, ones)
        except Exception:
            return False
        s = a.sum()
        if not torch.isfinite(s) or s.abs() < 1e-30:
            return False
        a = a / s
        new = (a.unsqueeze(1) * torch.stack(self.G[-k:])).sum(0)
        if self.beta != 1.0:
            # mix against the PRE-IMAGES, not X[-k:] -- G is appended before X
            # each call so the two tails coincide, making the old form a no-op
            # (DECISIONS.md Sec. 21).
            pre = self.X[-k-1:-1] if len(self.X) > k else self.X[:k]
            new = self.beta * new + (1 - self.beta) * (a.unsqueeze(1) * torch.stack(pre)).sum(0)
        if not torch.isfinite(new).all():
            return False
        off = 0
        for p in params:                              # write back in place
            nel = p.numel()
            p.copy_(new[off:off+nel].view_as(p))
            off += nel
        return True


# --------------------------------------------------------------------------
# Discrepancy-principle stopping  (RESEARCH.md Sec. 9.4)
# --------------------------------------------------------------------------
#
# EarlyStop has no usable operating point (DECISIONS.md Sec. 17): patience>=100
# never fires, patience=50 costs up to +171% error. The criterion is wrong, not
# mistuned -- it watches loss *improvement*, which says nothing about whether
# the remaining residual is signal or noise.
#
# The classical fix, from geophysical inverse problems and data assimilation:
# stop when the residual reaches the KNOWN noise level. Fitting below it is
# fitting noise. The scripts inject exactly `sigma * randn` (sigma=0.01), so
#
#     E||noise||_F = sigma * sqrt(N)      N = number of entries
#
# is analytic -- no tuning, and it transfers across shapes and ranks for free.

class DiscrepancyStop:
    """Stop when the residual norm reaches the noise floor.

        stop = DiscrepancyStop(sigma=0.01, n_entries=X.numel())
        for epoch in ...:
            if stop.step(loss.item()):     # loss must be ||residual||_F
                break

    tau > 1 stops earlier (more conservative about noise), tau < 1 later.

    Default tau=0.8, not the classical Morozov tau=1. Measured on video at
    rank 25 (epochs, since sub-second wall-clock is noisy):

        tau   1.2    1.0    0.9    0.8    0.7    0.6
        epoch 230    237    245    309   1113   2000
        clean +23.7% +15.2% +8.9%  +3.1% +0.0%  +0.0%

    tau=1 stops slightly early here: the model is a biased estimator, so its
    residual reaching the noise floor does not mean the remaining residual is
    all noise. tau=0.8 gives ~6.5x fewer epochs for +3.1%, against EarlyStop's
    +57% for comparable savings (DECISIONS.md Sec. 17).
    """

    def __init__(self, sigma, n_entries, tau=0.8, min_epochs=20):
        import math
        self.target = tau * sigma * math.sqrt(n_entries)
        self.min_epochs = min_epochs
        self.epochs = 0
        self.stopped = False

    def step(self, residual_norm):
        self.epochs += 1
        if self.epochs < self.min_epochs:
            return False
        if residual_norm <= self.target:
            self.stopped = True
            return True
        return False
