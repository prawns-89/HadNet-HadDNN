"""Error-vs-time Pareto frontier per k. The honest test of "is k>2 worth it?"

The matched-predicted-rank comparison is invalid because the r^k effective
rank is an UPPER BOUND that is not achieved in practice. What matters is:
for a given accuracy, which k gets there fastest?
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
from hadamard_ops import h_prod_k, svd_initialize_k, EarlyStop

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = Image.open("application-images/128mri.png").convert("RGB")
D = torch.tensor(np.asarray(img)/255.0, dtype=torch.float32, device=dev)
nrm = torch.norm(D)

def fit(k, r, max_ep=2000):
    torch.manual_seed(42)
    fac = svd_initialize_k(D, r, k, randomized=False)
    ps = [x.clone().requires_grad_(True) for pair in fac for x in pair]
    pairs = [(ps[2*i], ps[2*i+1]) for i in range(k)]
    opt = torch.optim.Adam(ps, lr=0.1); stop = EarlyStop(patience=50, min_rel=1e-3)
    if dev.type=="cuda": torch.cuda.synchronize()
    t0=time.time()
    for _ in range(max_ep):
        loss = torch.norm(D - h_prod_k(pairs))
        opt.zero_grad(); loss.backward(); opt.step()
        if stop.step(loss.item()): break
    if dev.type=="cuda": torch.cuda.synchronize()
    dt=time.time()-t0
    with torch.no_grad(): e=(torch.norm(D-h_prod_k(pairs))/nrm).item()
    return e, dt, sum(x.numel() for x in ps)

pts={}
print(f"device={dev}  sweeping r per k\n")
print(f"{'k':>2} {'r':>3} {'params':>9} {'time(s)':>8} {'rel_err':>9}")
for k in (2,3,4,5):
    pts[k]=[]
    for r in (2,3,5,8,10,15,20,30):
        e,dt,npar = fit(k,r)
        pts[k].append((npar,dt,e))
        print(f"{k:2d} {r:3d} {npar:9,d} {dt:8.2f} {e:9.5f}")
    print()

print("=== time to reach a target relative error (best r for that k) ===")
print(f"{'target':>8} " + " ".join(f"{'k='+str(k):>14}" for k in pts))
for tgt in (0.20, 0.15, 0.10, 0.07, 0.05):
    row=f"{tgt:8.2f} "
    for k in pts:
        ok=[(dt,npar) for npar,dt,e in pts[k] if e<=tgt]
        row += f"{(min(ok)[0] if ok else float('nan')):14.2f}" if ok else f"{'--':>14}"
    print(row)

print("\n=== params needed to reach target (lower is better compression) ===")
print(f"{'target':>8} " + " ".join(f"{'k='+str(k):>14}" for k in pts))
for tgt in (0.20, 0.15, 0.10, 0.07, 0.05):
    row=f"{tgt:8.2f} "
    for k in pts:
        ok=[npar for npar,dt,e in pts[k] if e<=tgt]
        row += f"{min(ok):14,d}" if ok else f"{'--':>14}"
    print(row)
