"""Is k>2 Hadamard factors cheaper at matched expressiveness?  Is k* = ln R?

Theory: k factors of rank r reach effective rank ~r^k using k*r*(m+n)
parameters and ~k*r*mnp per epoch. For a target effective rank R you need
r = R^(1/k), so cost ~ f(k) = k*R^(1/k), minimised at k* = ln R.

This measures whether the predicted optimum shows up in wall-clock and error.
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
from hadamard_ops import h_prod_k, svd_initialize_k, EarlyStop

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = Image.open("application-images/128mri.png").convert("RGB")
D = torch.tensor(np.asarray(img) / 255.0, dtype=torch.float32, device=dev)
nrm = torch.norm(D)
print(f"device={dev}  image={tuple(D.shape)}\n")

def fit(k, r, max_ep=1500):
    torch.manual_seed(42)
    fac = svd_initialize_k(D, r, k, randomized=False)
    params = [x.clone().requires_grad_(True) for pair in fac for x in pair]
    pairs = [(params[2*i], params[2*i+1]) for i in range(k)]
    opt = torch.optim.Adam(params, lr=0.1); stop = EarlyStop(patience=50, min_rel=1e-3)
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(max_ep):
        loss = torch.norm(D - h_prod_k(pairs))
        opt.zero_grad(); loss.backward(); opt.step()
        if stop.step(loss.item()): break
    if dev.type == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    with torch.no_grad():
        e = (torch.norm(D - h_prod_k(pairs)) / nrm).item()
    nparam = sum(x.numel() for x in params)
    return e, dt, stop.epochs, nparam

for R in (25, 100):
    kstar = math.log(R)
    print(f"--- target effective rank R={R}   predicted k* = ln R = {kstar:.2f} ---")
    print(f"{'k':>2} {'r':>3} {'params':>9} {'pred cost k*R^(1/k)':>20} {'epochs':>7} {'time(s)':>8} {'rel_err':>9}")
    best = None
    for k in range(2, 7):
        r = max(2, round(R ** (1.0 / k)))
        pred = k * (R ** (1.0 / k))
        e, dt, ep, npar = fit(k, r)
        print(f"{k:2d} {r:3d} {npar:9,d} {pred:20.2f} {ep:7d} {dt:8.2f} {e:9.5f}")
        if best is None or dt < best[1]: best = (k, dt, e)
    print(f"  -> fastest k={best[0]} ({best[1]:.2f}s, err {best[2]:.5f}), predicted k*={kstar:.2f}\n")
