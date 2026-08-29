"""Does a Lipschitz step size beat the hard-coded Adam lr=0.1?

Four regimes on the same tensor + identical SVD init:
  (a) repo baseline : UNSQUARED loss, Adam lr=0.1
  (b) squared loss  , Adam lr=0.1
  (c) squared loss  , Adam lr = 1/L  (L recomputed each epoch)
  (d) squared loss  , plain GD step 1/L   <- the theoretically justified one
Reported: epochs to reach a relative-error target, and final relative error.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hadamard_ops import h_prod, svd_initialize, squared_loss, lipschitz_lr

torch.manual_seed(42)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = n = 60; p = (8,); r = 6
D = (torch.rand(m, n, *p, device=dev) + 0.1 * torch.randn(m, n, *p, device=dev))
nrm = torch.norm(D)
MAXEP = 2000
CKPT = [50, 100, 250, 500, 1000, 2000]
print(f"device={dev} shape={(m,n)+p} r={r}\n")

def init():
    f = svd_initialize(D, r, randomized=False)
    return [x.clone().requires_grad_(True) for x in f]

def rel(f):
    with torch.no_grad():
        return (torch.norm(D - h_prod(f[0], f[1]) * h_prod(f[2], f[3])) / nrm).item()

def run(tag, squared, mode):
    f = init(); opt = None
    if mode != "gd":
        opt = torch.optim.Adam(f, lr=0.1)
    curve, t0 = {}, time.time()
    for ep in range(1, MAXEP + 1):
        pred = h_prod(f[0], f[1]) * h_prod(f[2], f[3])
        loss = squared_loss(D, pred) if squared else torch.norm(D - pred, 'fro')
        if mode == "lip":
            lr, _ = lipschitz_lr(*[x.detach() for x in f])
            for g in opt.param_groups: g['lr'] = lr
        for x in f:
            if x.grad is not None: x.grad = None
        loss.backward()
        if mode == "gd":
            lr, _ = lipschitz_lr(*[x.detach() for x in f])
            with torch.no_grad():
                for x in f: x -= lr * x.grad
        else:
            opt.step()
        if ep in CKPT: curve[ep] = rel(f)
    dt = time.time() - t0
    print(f"  {tag:36s} " + " ".join(f"{curve[c]:7.4f}" for c in CKPT) + f"  {dt:5.1f}s")
    return curve

print("relative error at epoch:")
print(f"  {'':36s} " + " ".join(f"{c:>7d}" for c in CKPT))
run("(a) unsquared loss, Adam lr=0.1  [repo]", False, "adam")
run("(b) squared loss,   Adam lr=0.1",         True,  "adam")
run("(c) squared loss,   Adam lr=1/L",         True,  "lip")
run("(d) squared loss,   plain GD 1/L",        True,  "gd")
