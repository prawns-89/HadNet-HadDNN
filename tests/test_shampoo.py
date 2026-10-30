"""Correctness checks for optim/shampoo.py, plus one convex comparison.

Run:  .venv/bin/python tests/test_shampoo.py

What this establishes:
  1. the general order-k path (Alg. 2) reproduces verbatim Alg. 1 at k=2
  2. the per-mode diagonal fallback reproduces verbatim Alg. 3
  3. the inverse 4th root is actually an inverse 4th root
  4. the Sec. 5 threshold switches a wide mode to O(n) state
  5. an order-4 parameter trains at all (exponent -1/8)
  6. on a Kronecker-structured least-squares problem, full Shampoo beats
     diagonal AdaGrad and SGD at each one's own best learning rate

What this does NOT establish: the regret bound, or anything about real
networks.  Check 6 is the structure Shampoo is built for, by construction --
it is a sanity check that the preconditioner helps where the theory says it
must, not a benchmark.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from optim.shampoo import (Shampoo, shampoo_step_matrix, _inverse_pth_root)

torch.manual_seed(42)
torch.set_printoptions(precision=8)
dev = torch.device("cpu")          # algebra checks want float64 determinism
ok = True

def report(name, passed, detail=""):
    global ok
    ok = ok and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


# ---------------------------------------------------------------- 1. Alg 2 == Alg 1
print("1. general path (Alg. 2, k=2) vs verbatim Alg. 1")
m, n, T, lr, eps = 12, 9, 25, 1e-2, 1e-4

W_ref = torch.zeros(m, n, dtype=torch.float64)
L = eps * torch.eye(m, dtype=torch.float64)
R = eps * torch.eye(n, dtype=torch.float64)

W_gen = torch.zeros(m, n, dtype=torch.float64, requires_grad=True)
opt = Shampoo([W_gen], lr=lr, eps=eps, diagonal_threshold=None)

grads = [torch.randn(m, n, dtype=torch.float64) for _ in range(T)]
for Gt in grads:
    W_ref = shampoo_step_matrix(W_ref, Gt, L, R, lr, eps)
    W_gen.grad = Gt.clone()
    opt.step()

d = (W_ref - W_gen.detach()).abs().max().item()
report("max |Alg2 - Alg1|", d < 1e-12, f"{d:.3e}  over {T} steps")


# ---------------------------------------------------------------- 2. diagonal == Alg 3
print("\n2. diagonal fallback vs verbatim Alg. 3")
W_ref = torch.zeros(m, n, dtype=torch.float64)
Ld = eps * torch.ones(m, dtype=torch.float64)
Rd = eps * torch.ones(n, dtype=torch.float64)

W_gen = torch.zeros(m, n, dtype=torch.float64, requires_grad=True)
opt = Shampoo([W_gen], lr=lr, eps=eps, diagonal_threshold=0)   # 0 -> all diag

for Gt in grads:
    Ld += (Gt * Gt).sum(dim=1)                      # diag(G Gt)
    Rd += (Gt * Gt).sum(dim=0)                      # diag(Gt G)
    W_ref = W_ref - lr * (Ld.pow(-0.25)[:, None] * Gt * Rd.pow(-0.25)[None, :])
    W_gen.grad = Gt.clone()
    opt.step()

d = (W_ref - W_gen.detach()).abs().max().item()
report("max |diag path - Alg3|", d < 1e-12, f"{d:.3e}  over {T} steps")


# ---------------------------------------------------------------- 3. inverse root
print("\n3. inverse p-th root")
A = torch.randn(20, 20, dtype=torch.float64)
A = A @ A.T + 0.5 * torch.eye(20, dtype=torch.float64)
Ai4 = _inverse_pth_root(A, 4, 1e-12)
recon = Ai4 @ Ai4 @ Ai4 @ Ai4 @ A                      # should be I
d = (recon - torch.eye(20, dtype=torch.float64)).abs().max().item()
report("max |A^(-1/4)^4 A - I|", d < 1e-8, f"{d:.3e}  cond={torch.linalg.cond(A):.3e}")


# ---------------------------------------------------------------- 4. Sec. 5 threshold
print("\n4. Sec. 5 per-mode diagonal fallback")
P = torch.zeros(5, 2000, requires_grad=True)
opt = Shampoo([P], lr=1e-3, diagonal_threshold=1200)
P.grad = torch.randn(5, 2000)
opt.step()
st = opt.state[P]
shapes = [tuple(h.shape) for h in st["H"]]
expect = st["is_diag"] == [False, True] and shapes == [(5, 5), (2000,)]
report("mode 0 full, mode 1 diagonal", expect, f"is_diag={st['is_diag']}  H={shapes}")
full_cost = 2000 * 2000
report("wide mode is O(n) not O(n^2)", st["H"][1].numel() == 2000,
       f"{st['H'][1].numel()} vs {full_cost} elements")


# ---------------------------------------------------------------- 5. order-4
print("\n5. order-4 parameter (exponent -1/8)")
torch.manual_seed(0)
X = torch.randn(4, 3, 5, 6)
Wt = torch.randn(4, 3, 5, 6)
V = torch.zeros(4, 3, 5, 6, requires_grad=True)
opt = Shampoo([V], lr=5e-2, eps=1e-4)
first = last = None
for t in range(60):
    opt.zero_grad()
    loss = ((V - Wt) ** 2 * X.abs()).sum()
    loss.backward()
    opt.step()
    if t == 0: first = loss.item()
    last = loss.item()
report("order-4 loss decreases", last < first and torch.isfinite(V).all(),
       f"{first:.4e} -> {last:.4e}")


# ---------------------------------------------------------------- 6. convex comparison
print("\n6. Kronecker-structured least squares: min ||A W B - Y||_F^2 / 2")
torch.manual_seed(7)
mm = nn_ = 40
def illcond(k, cond):
    Q, _ = torch.linalg.qr(torch.randn(k, k, dtype=torch.float64))
    s = torch.logspace(0, -torch.log10(torch.tensor(float(cond))), k,
                       dtype=torch.float64)
    return (Q * s) @ Q.T

A = illcond(mm, 1e3)
B = illcond(nn_, 1e3)
Wstar = torch.randn(mm, nn_, dtype=torch.float64)
Y = A @ Wstar @ B
print(f"   cond(A)={torch.linalg.cond(A):.2e}  cond(B)={torch.linalg.cond(B):.2e}"
      f"  cond(Hessian)={torch.linalg.cond(A.T@A)*torch.linalg.cond(B@B.T):.2e}")

def run(make_opt, steps=200):
    W = torch.zeros(mm, nn_, dtype=torch.float64, requires_grad=True)
    o = make_opt([W])
    best = float("inf")
    for _ in range(steps):
        o.zero_grad()
        loss = 0.5 * ((A @ W @ B - Y) ** 2).sum()
        loss.backward()
        o.step()
        v = loss.item()
        if not torch.isfinite(torch.tensor(v)): return float("inf")
        best = min(best, v)
    return best

sweep = [10 ** e for e in range(-6, 6)]
results = {}
for name, ctor in [
    ("SGD",      lambda ps, lr: torch.optim.SGD(ps, lr=lr)),
    ("AdaGrad",  lambda ps, lr: torch.optim.Adagrad(ps, lr=lr)),
    ("Shampoo",  lambda ps, lr: Shampoo(ps, lr=lr, eps=1e-6)),
]:
    scores = [(run(lambda ps, lr=lr: ctor(ps, lr)), lr) for lr in sweep]
    loss, lr = min(scores)
    results[name] = loss
    print(f"   {name:9} best loss {loss:.6e}  at lr={lr:g}")

report("Shampoo < AdaGrad", results["Shampoo"] < results["AdaGrad"],
       f"{results['Shampoo']:.3e} vs {results['AdaGrad']:.3e}")
report("Shampoo < SGD", results["Shampoo"] < results["SGD"],
       f"{results['Shampoo']:.3e} vs {results['SGD']:.3e}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
