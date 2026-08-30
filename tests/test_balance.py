"""Does gauge-fixing by balancing improve conditioning enough to cut epochs?

RESEARCH.md Sec. 8.2: arXiv:2606.01216 argues the cross-Hadamard scaling
symmetry wrecks Euclidean optimisers. This is the cheap test of that claim --
if picking a canonical representative each step recovers most of the benefit,
the full Riemannian rewrite (Sec. 1) is unnecessary.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, cv2
from hadamard_ops import h_prod, svd_initialize, balance_all_

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture("application-videos/video.mp4"); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
A = torch.tensor(np.array(fr), dtype=torch.float32, device=dev).permute(1,2,0,3)/255.0
torch.manual_seed(0); A = A + 0.01*torch.randn_like(A); nrm = torch.norm(A)

def run(rank, every, max_ep=2000, track=(100,250,500,1000,2000)):
    torch.manual_seed(42)
    fs = [x.clone().requires_grad_(True) for x in svd_initialize(A, rank, randomized=False)]
    pairs = [(fs[0], fs[1]), (fs[2], fs[3])]
    opt = torch.optim.Adam(fs, lr=0.1)
    wait, best, curve = 100, float('inf'), {}
    torch.cuda.synchronize(); t0 = time.time()
    for ep in range(1, max_ep+1):
        loss = torch.norm(A - h_prod(*pairs[0]) * h_prod(*pairs[1]))
        opt.zero_grad(); loss.backward(); opt.step(); l = loss.item()
        if every and ep % every == 0:
            balance_all_(pairs)
        if l < best: best, wait = l, 100
        else: wait -= 1
        if wait == 0:
            for g in opt.param_groups: g['lr'] *= 0.1
            wait = 100
        if ep in track:
            with torch.no_grad():
                curve[ep] = (torch.norm(A - h_prod(*pairs[0])*h_prod(*pairs[1]))/nrm).item()
    torch.cuda.synchronize()
    return time.time()-t0, curve

for R in (10, 25):
    print(f"\n=== rank {R} : relative error at epoch ===")
    print(f"{'balancing':<20}" + "".join(f"{e:>10}" for e in (100,250,500,1000,2000)) + f"{'time':>8}")
    for every, tag in ((0,"none [baseline]"), (1,"every step"), (10,"every 10"), (50,"every 50")):
        dt, c = run(R, every)
        print(f"{tag:<20}" + "".join(f"{c[e]:>10.5f}" for e in (100,250,500,1000,2000)) + f"{dt:>8.2f}", flush=True)
