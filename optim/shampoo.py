"""
Shampoo: preconditioned stochastic tensor optimization.

Gupta, Koren & Singer, arXiv:1802.09568v2 -- written from the paper's
pseudocode, not ported: optax and jax are not in this venv, so there was
nothing to port from.

What is taken from the paper:
    Alg. 1   matrix case.  L += G Gt ; R += Gt G ; W -= eta L^(-1/4) G R^(-1/4)
    Alg. 2   order-k tensor case.  One H^i per mode, exponent -1/(2k).
    Alg. 3   diagonal variant.  Keeps only diag(G Gt), O(m+n) memory.
    Sec. 5   a mode wider than a threshold falls back to Alg. 3 for that mode
             alone, leaving the other modes fully preconditioned.  The paper
             used "around 1200" and reported no apparent loss.

What is NOT from the paper, and is off by default:
    precond_freq > 1 -- reusing the inverse roots for several steps.  The
    usual later modification; SOAP (arXiv:2409.11321) reports it degrades
    Shampoo, worse the less often you refresh.  Exposed because it is the
    first thing anyone reaches for, not because this paper endorses it.

NOT validated here.  The O(sqrt(T)) regret bound (Thm. 7) is not tested, and
nothing in this file says anything about behaviour on real networks.
tests/test_shampoo.py checks the algebra and one ill-conditioned convex
problem.  That is all it checks.

Measured on this machine (RTX 4050, torch 2.11.0+cu128), not from the paper:
    the preconditioned step has a completely different natural scale from an
    SGD step, so lr does NOT transfer.  On the test's Kronecker least-squares
    problem the best lr was 1e4 for Shampoo against 1 for SGD and 0.1 for
    AdaGrad.  Sweep lr over orders of magnitude before concluding anything.

    At that lr the trajectory transiently RISES about SEVEN orders of magnitude
    above its starting loss -- measured on the notebook's problem: start 5.77,
    peak 7.91e7 at step 2, back under the starting loss only at step 54 of 200,
    final 2.78e-6.  Cause: with H_0 = eps*I the first step is scaled by
    eps^(-1/4), which is 31.6 at eps=1e-6, so the step is enormous until the
    grams accumulate.  On a convex quadratic it recovers and still wins; on a
    non-convex loss nothing guarantees it comes back.  Raise eps or warm up if
    that matters.  Alg. 1 has no warmup, so neither does this.

Tensor layout: whatever shape the parameter already has.  Unlike
hadamard_ops.py there is no slice convention to respect -- Shampoo is
deliberately oblivious to architecture and sees only shapes.
"""
import torch


# --------------------------------------------------------------------------
# Mode algebra (Sec. 4.1 definitions)
# --------------------------------------------------------------------------

def _unfold(G, i):
    """mat_i(G): mode-i unfolding, (n_i, prod of the other dims)."""
    return torch.movedim(G, i, 0).reshape(G.shape[i], -1)


def _mode_gram(G, i):
    """G^(i) = mat_i(G) mat_i(G)^T, the contraction along all but mode i."""
    U = _unfold(G, i)
    return U @ U.T


def _mode_diag_gram(G, i):
    """diag(G^(i)) without forming G^(i) -- sum of squares over other modes."""
    axes = [a for a in range(G.ndim) if a != i]
    return (G * G).sum(dim=axes) if axes else G * G


def _mode_mul(G, M, i):
    """G x_i M, the mode-i product with an (n_i, n_i) matrix."""
    Gm = torch.movedim(G, i, 0)
    shp = Gm.shape
    out = (M @ Gm.reshape(shp[0], -1)).reshape(shp)
    return torch.movedim(out, 0, i)


def _inverse_pth_root(A, p, floor):
    """A^(-1/p) for symmetric PSD A, via eigendecomposition.

    The paper builds an SVD and powers the singular values; for a symmetric
    PSD matrix eigh returns the same factors for less work.  Done in float64
    regardless of the parameter dtype -- a 1/4 power amplifies relative error
    in the small eigenvalues, which is exactly where the step is largest.

    `floor` clamps the spectrum from below.  With H_0 = eps*I no eigenvalue
    can legitimately fall under eps, so anything lower is roundoff.
    """
    lam, Q = torch.linalg.eigh(A.double())
    lam = lam.clamp_min(floor)
    return (Q * lam.pow(-1.0 / p)) @ Q.T


# --------------------------------------------------------------------------
# Verbatim Algorithm 1, kept for A/B against the general path
# --------------------------------------------------------------------------

def shampoo_step_matrix(W, G, L, R, lr, eps):
    """One step of Alg. 1, transcribed literally.  Mutates L and R.

    Only correct for 2-D W.  The general path in `Shampoo` must reproduce
    this exactly at k=2; tests/test_shampoo.py asserts that it does.
    """
    L += G @ G.T                                        # L_t = L_{t-1} + G Gt
    R += G.T @ G                                        # R_t = R_{t-1} + Gt G
    Li = _inverse_pth_root(L, 4, eps)
    Ri = _inverse_pth_root(R, 4, eps)
    return W - lr * (Li.to(G.dtype) @ G @ Ri.to(G.dtype))


# --------------------------------------------------------------------------
# Algorithm 2, with the Sec. 5 per-mode diagonal fallback
# --------------------------------------------------------------------------

class Shampoo(torch.optim.Optimizer):
    """Shampoo in its general tensor form (Alg. 2).

    Args:
        lr: eta.  The paper carries no momentum and no weight decay in
            Alg. 1/2, so neither is implemented -- adding them would be a
            different algorithm wearing this name.
        eps: the epsilon in H_0 = eps*I.  Also the eigenvalue floor.
        diagonal_threshold: a mode wider than this uses Alg. 3 for that mode
            (Sec. 5; the paper used ~1200).  None disables the fallback.
        precond_freq: steps between root recomputations.  1 = every step, as
            the paper does.  See the module docstring before raising it.
    """

    def __init__(self, params, lr=1e-3, eps=1e-4,
                 diagonal_threshold=1200, precond_freq=1):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if precond_freq < 1:
            raise ValueError(f"precond_freq must be >= 1, got {precond_freq}")
        super().__init__(params, dict(lr=lr, eps=eps,
                                      diagonal_threshold=diagonal_threshold,
                                      precond_freq=precond_freq))

    def _init_state(self, state, G, group):
        """H_0 = eps*I per mode, or eps*1 for a mode held diagonally."""
        thr = group["diagonal_threshold"]
        state["step"] = 0
        state["H"], state["root"], state["is_diag"] = [], [], []
        for i, n in enumerate(G.shape):
            diag = thr is not None and n > thr
            state["is_diag"].append(diag)
            if diag:
                state["H"].append(torch.full((n,), group["eps"],
                                             dtype=G.dtype, device=G.device))
            else:
                state["H"].append(group["eps"] * torch.eye(
                    n, dtype=G.dtype, device=G.device))
            state["root"].append(None)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr, eps = group["lr"], group["eps"]
            freq = group["precond_freq"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad
                if G.is_sparse:
                    raise RuntimeError("Shampoo does not support sparse grads")

                # A scalar has no modes to precondition; Alg. 2 is vacuous.
                if G.ndim == 0:
                    p.add_(G, alpha=-lr)
                    continue

                state = self.state[p]
                if not state:
                    self._init_state(state, G, group)

                k = G.ndim                      # exponent is -1/(2k) (Alg. 2)
                stale = state["step"] % freq == 0
                Gt = G                          # the preconditioned gradient

                for i in range(k):
                    # H^i accumulates the gram of the RAW gradient, while Gt
                    # carries the modes already preconditioned (Alg. 2).
                    if state["is_diag"][i]:
                        state["H"][i] += _mode_diag_gram(G, i)
                        if stale or state["root"][i] is None:
                            state["root"][i] = state["H"][i].double().pow(
                                -1.0 / (2 * k)).to(G.dtype)
                        shape = [1] * k
                        shape[i] = -1
                        Gt = Gt * state["root"][i].reshape(shape)
                    else:
                        state["H"][i] += _mode_gram(G, i)
                        if stale or state["root"][i] is None:
                            state["root"][i] = _inverse_pth_root(
                                state["H"][i], 2 * k, eps).to(G.dtype)
                        Gt = _mode_mul(Gt, state["root"][i], i)

                state["step"] += 1
                p.add_(Gt, alpha=-lr)

        return loss
