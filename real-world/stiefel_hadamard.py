"""
Stiefel-constrained momentum solver for Hadamard decomposition

        X  ~=  (W1 @ H1) * (W2 @ H2)        (* = Hadamard product)

Each Hadamard factor is kept in SVD-like form

        M_k = U_k @ S_k @ V_k.T,   U_k in St(m, r),  V_k in St(n, r),  S_k in R^{r x r}

so every iterate is exactly orthonormal on the outside and all scale lives in the
small r x r cores. The ideas are the ones from the SVD discussion, applied to the
Hadamard model:

  1. Constraint enforced at every step: Riemannian gradient on the Stiefel
     manifold (horizontal projection) followed by a QR retraction.
  2. Gauge fixing for the model's symmetries:
       - GL(r) ambiguity inside a factor (W -> W R, H -> R^-1 H) is removed by
         the orthonormal U, V. The leftover O(r) rotation (U -> U Q, S -> Q^T S)
         is removed by moving U, V only orthogonally to their own span and
         letting S handle in-span changes (quotient / horizontal-space geometry).
       - Scaling ambiguity between factors (M1 -> c M1, M2 -> M2 / c) is removed
         by rebalancing ||S1||_F = ||S2||_F after every step. This is an exact
         move (the product does not change) and keeps both factors on the same
         scale, so one step size suits both. Note: gradient flow already
         conserves ||M1||^2 - ||M2||^2 (because <M1, dL/dM1> = <M2, dL/dM2>),
         so from a balanced init this mostly guards against drift from finite
         steps; it matters when the init is unbalanced.
  3. Geometry-aware preconditioning: the U/V directions are right-multiplied by
     (S S^T + delta I)^-1. This makes the relative change of U, V and S per step
     scale-consistent, so a single step size works for all blocks (without it
     the U/V steps are off by a factor of ~sigma^2).
  4. Heavy-ball or Nesterov momentum carried correctly across the constraint:
       - Stiefel momentum is transported to the new horizontal space by
         projection after each retraction (old momentum points partly off it).
       - Core momentum is rescaled by the same factor as the cores when the
         gauge is rebalanced (the "rescale both terms" rule from the momentum
         power method).
  5. Symmetry breaking at init: the factor-swap symmetry M1 <-> M2 traps any
     simultaneous-update method when the init has M1 == M2 (which the reference
     sqrt|X| init produces for every nonnegative X). See `break_symmetry`.
  6. Monotone safeguard: a step that increases the loss is rejected, momentum is
     reset (adaptive restart) and the step size halved; accepted steps grow it.

Cost per iteration is O(m n r), the same as the Adam-based HadamardNet.

Public API mirrors Hadamard_BCD:
    W1, H1, W2, H2, error, times = stiefel_hadDec(X, r, ...)
"""

import time
import numpy as np
import torch


# ----------------------------------------------------------------- helpers ---

def _qf(Z):
    """QR retraction with sign fix (diag(R) > 0), so columns don't flip."""
    Q, R = torch.linalg.qr(Z)
    d = torch.sign(torch.diagonal(R))
    d[d == 0] = 1
    return Q * d


def _hproj(U, Z):
    """Project Z onto the horizontal space at U: the part of the Stiefel tangent
    space orthogonal to span(U), i.e. (I - U U^T) Z.

    The in-span part of a Stiefel tangent vector (U @ skew) only rotates the
    columns of U, which the core S can absorb (U -> U Q, S -> Q^T S). That is
    the remaining O(r) gauge freedom, so moving along it is redundant. Dropping
    it also guarantees the preconditioned step below is a descent direction;
    with the full tangent projection the skew part can point uphill."""
    return Z - U @ (U.T @ Z)


def _init_factors(X, r, randomized=True, init="svd", seed=0):
    """init="svd": same idea as SVD_initialization / svd_init in the reference
    code, M1 ~ sqrt|X| and M2 ~ sign(X) * sqrt|X|, each truncated to rank r.
    init="random": random orthonormal U, V with S = I (rescaled later).
    Returned directly in (U, S, V) form with S diagonal."""
    if init == "random":
        rng = np.random.default_rng(seed)
        m, n = X.shape
        return [(np.linalg.qr(rng.standard_normal((m, r)))[0], np.eye(r),
                 np.linalg.qr(rng.standard_normal((n, r)))[0]) for _ in range(2)]
    M = np.sqrt(np.abs(X))
    out = []
    for A in (M, M * np.sign(X)):
        if randomized and r < min(A.shape) // 2:
            from sklearn.utils.extmath import randomized_svd
            U, s, Vt = randomized_svd(A, n_components=r, n_iter=8, random_state=0)
        else:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            U, s, Vt = U[:, :r], s[:r], Vt[:r]
        out.append((U, np.diag(s), Vt.T))
    return out


# ------------------------------------------------------------------ solver ---

def stiefel_hadDec(X, r, maxiter=5000, eta0=None, beta=0.9, momentum="heavy",
                   precondition=True, balance=True, grow=1.05, shrink=0.5,
                   tol=1e-9, patience=200, time_limit=200.0,
                   device=None, dtype=torch.float64, randomized_init=True,
                   init="svd", seed=0, break_symmetry=1e-2, verbose=False):
    """Hadamard decomposition X ~ (W1 H1) * (W2 H2) with rank-r factors.

    Params:
    X: input matrix (numpy array, m x n)
    r: rank of each Hadamard factor
    maxiter: maximum number of iterations
    eta0: initial step size (None -> chosen so the first core step is ~1% relative)
    beta: momentum coefficient
    momentum: "none", "heavy" (Polyak heavy ball) or "nesterov"
    precondition: scale U/V directions by (S S^T + delta I)^-1
    balance: rebalance ||S1||_F = ||S2||_F after every step
    grow, shrink: step-size multipliers on accepted / rejected steps
    tol, patience: stop when relative error improves by < tol (relative)
                   for `patience` consecutive accepted steps
    time_limit: wall-clock cap in seconds (same role as in Hadamard_BCD)
    init: "svd" (sqrt|X| split, as in the reference code) or "random"
    seed: seed for init="random" and for the symmetry-breaking noise
    break_symmetry: relative size of the noise added to factor 2 at init.
        For X >= 0 the "svd" init gives sign(X)*sqrt|X| = sqrt|X|, so M1 == M2
        exactly. The gradients R*M2 and R*M1 are then equal, and any method that
        updates both factors simultaneously (this solver, HadNet/Adam, IGD)
        stays on M1 == M2 forever, i.e. it can only fit X ~ M * M. Block
        methods like BCD escape because they update the factors one after the
        other. Set 0 to reproduce the trapped behaviour.

    Returns (same layout as Hadamard_BCD):
    W1 (m x r), H1 (r x n), W2 (m x r), H2 (r x n): factors, with W_k = U_k S_k, H_k = V_k^T
    error: relative Frobenius error per accepted iteration
    times: elapsed time per accepted iteration
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

    Xt = t(X)
    normX = torch.linalg.norm(Xt)
    I_r = torch.eye(r, dtype=dtype, device=device)

    (U1, S1, V1), (U2, S2, V2) = _init_factors(X, r, randomized_init, init, seed)
    if break_symmetry > 0:
        rng = np.random.default_rng(seed + 12345)
        U2 = np.linalg.qr(U2 + break_symmetry * rng.standard_normal(U2.shape))[0]
        V2 = np.linalg.qr(V2 + break_symmetry * rng.standard_normal(V2.shape))[0]
    U = [t(U1), t(U2)]
    S = [t(S1), t(S2)]
    V = [t(V1), t(V2)]

    # Optimal global scale for the initial guess (as in Hadamard_BCD.scaling):
    # the product is degree 2 in (S1, S2), so each core takes sqrt(alpha).
    P = (U[0] @ S[0] @ V[0].T) * (U[1] @ S[1] @ V[1].T)
    alpha = (torch.sum(P * Xt) / torch.sum(P * P)).item()
    c = abs(alpha) ** 0.5
    S[0] = S[0] * (c if alpha >= 0 else -c)
    S[1] = S[1] * c

    def rebalance(S, bS):
        a, b = torch.linalg.norm(S[0]), torch.linalg.norm(S[1])
        c = torch.sqrt(b / a)
        S[0], S[1] = S[0] * c, S[1] / c
        bS[0], bS[1] = bS[0] * c, bS[1] / c       # rescale momentum by the same factor

    def forward(U, S, V):
        M = [U[k] @ S[k] @ V[k].T for k in range(2)]
        R = M[0] * M[1] - Xt
        return M, R, 0.5 * torch.sum(R * R).item()

    bU = [torch.zeros_like(U[k]) for k in range(2)]
    bV = [torch.zeros_like(V[k]) for k in range(2)]
    bS = [torch.zeros_like(S[k]) for k in range(2)]
    if balance:
        rebalance(S, bS)

    M, R, loss = forward(U, S, V)
    error = [np.sqrt(2 * loss) / normX.item()]
    times = [0.0]
    t0 = time.time()
    eta = eta0
    stall = 0
    use_mom = momentum in ("heavy", "nesterov")

    for it in range(maxiter):
        if time.time() - t0 > time_limit:
            break

        # ---- Euclidean gradients w.r.t. each Hadamard factor
        G = [R * M[1], R * M[0]]

        # ---- Riemannian / preconditioned directions per block
        xiU, xiV, xiS = [], [], []
        for k in range(2):
            dU = G[k] @ V[k] @ S[k].T
            dV = G[k].T @ U[k] @ S[k]
            dS = U[k].T @ G[k] @ V[k]
            if precondition:
                delta = 1e-8 * torch.sum(S[k] * S[k]) / r + 1e-30
                dU = torch.linalg.solve(S[k] @ S[k].T + delta * I_r, dU.T).T
                dV = torch.linalg.solve(S[k].T @ S[k] + delta * I_r, dV.T).T
            xiU.append(_hproj(U[k], dU))
            xiV.append(_hproj(V[k], dV))
            xiS.append(dS)

        if eta is None:   # first step changes the cores by ~1% relative
            eta = 1e-2 * min((torch.linalg.norm(S[k]) / (torch.linalg.norm(xiS[k]) + 1e-30)).item()
                             for k in range(2))

        # ---- momentum (with transport) and candidate step
        new_U, new_V, new_S = [], [], []
        new_bU, new_bV, new_bS = [], [], []
        for k in range(2):
            if use_mom:
                mU = beta * _hproj(U[k], bU[k]) + xiU[k]      # transport old momentum
                mV = beta * _hproj(V[k], bV[k]) + xiV[k]
                mS = beta * bS[k] + xiS[k]
                if momentum == "nesterov":
                    dirU, dirV, dirS = xiU[k] + beta * mU, xiV[k] + beta * mV, xiS[k] + beta * mS
                else:
                    dirU, dirV, dirS = mU, mV, mS
            else:
                mU, mV, mS = bU[k], bV[k], bS[k]
                dirU, dirV, dirS = xiU[k], xiV[k], xiS[k]
            new_U.append(_qf(U[k] - eta * dirU))            # retraction = constraint step
            new_V.append(_qf(V[k] - eta * dirV))
            new_S.append(S[k] - eta * dirS)
            new_bU.append(mU); new_bV.append(mV); new_bS.append(mS)

        if balance:
            rebalance(new_S, new_bS)

        new_M, new_R, new_loss = forward(new_U, new_S, new_V)

        # ---- monotone safeguard with adaptive restart
        if not np.isfinite(new_loss) or new_loss > loss:
            eta *= shrink
            bU = [torch.zeros_like(x) for x in bU]
            bV = [torch.zeros_like(x) for x in bV]
            bS = [torch.zeros_like(x) for x in bS]
            if eta < 1e-20:
                break
            continue

        rel_impr = 1 - new_loss / loss if loss > 0 else 0.0
        U, V, S, M, R, loss = new_U, new_V, new_S, new_M, new_R, new_loss
        bU, bV, bS = new_bU, new_bV, new_bS
        eta *= grow

        err = np.sqrt(2 * loss) / normX.item()
        error.append(err)
        times.append(time.time() - t0)
        if verbose and it % 500 == 0:
            print(f"it {it:6d}  rel.err {err:.6f}  eta {eta:.3e}")

        stall = stall + 1 if rel_impr < tol else 0
        if stall >= patience or err < 1e-12:
            break

    W1 = (U[0] @ S[0]).cpu().numpy()
    H1 = V[0].T.cpu().numpy()
    W2 = (U[1] @ S[1]).cpu().numpy()
    H2 = V[1].T.cpu().numpy()
    return W1, H1, W2, H2, error, times
