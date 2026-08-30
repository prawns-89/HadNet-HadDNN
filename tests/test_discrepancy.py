"""Discrepancy-principle stopping vs EarlyStop vs no stopping.

Key subtlety: the scripts add `0.01*randn` and then score error against the
NOISY tensor. So fitting the noise IMPROVES the reported metric while making
the actual reconstruction worse. This measures error against BOTH the noisy
target (what the paper reports) and the clean data (what we actually want).
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, cv2
from hadamard_ops import h_prod, svd_initialize, EarlyStop, DiscrepancyStop

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA = 0.01
cap = cv2.VideoCapture("application-videos/video.mp4"); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
CLEAN = torch.tensor(np.array(fr), dtype=torch.float32, device=dev).permute(1,2,0,3)/255.0
torch.manual_seed(0)
NOISY = CLEAN + SIGMA*torch.randn_like(CLEAN)
N = CLEAN.numel()
floor = SIGMA*math.sqrt(N)
print(f"video {tuple(CLEAN.shape)}  N={N:,}")
print(f"noise floor ||n||_F = {floor:.2f}   as rel err vs noisy = {floor/torch.norm(NOISY):.5f}")
print(f"true ||noise||_F    = {torch.norm(NOISY-CLEAN):.2f}\n")

def run(rank, stopper_factory, tag, max_ep=2000):
    torch.manual_seed(42)
    f = [x.clone().requires_grad_(True) for x in svd_initialize(NOISY, rank, randomized=False)]
    opt = torch.optim.Adam(f, lr=0.1); st = stopper_factory()
    wait, best = 100, float('inf')
    torch.cuda.synchronize() if dev.type=="cuda" else None
    t0 = time.time()
    for ep in range(max_ep):
        loss = torch.norm(NOISY - h_prod(f[0],f[1])*h_prod(f[2],f[3]))
        opt.zero_grad(); loss.backward(); opt.step(); l = loss.item()
        if l < best: best, wait = l, 100
        else: wait -= 1
        if wait == 0:
            for g in opt.param_groups: g['lr'] *= 0.1
            wait = 100
        if st is not None and st.step(l): break
    torch.cuda.synchronize() if dev.type=="cuda" else None
    dt = time.time()-t0
    with torch.no_grad():
        rec = h_prod(f[0],f[1])*h_prod(f[2],f[3])
        e_noisy = (torch.norm(NOISY-rec)/torch.norm(NOISY)).item()
        e_clean = (torch.norm(CLEAN-rec)/torch.norm(CLEAN)).item()
    ep_used = st.epochs if st is not None else max_ep
    print(f"  {tag:<28}{ep_used:>7}{dt:>8.2f}{e_noisy:>11.5f}{e_clean:>11.5f}")
    return dt, e_noisy, e_clean

for R in (10, 25):
    print(f"=== rank {R} ===")
    print(f"  {'stopping rule':<28}{'epochs':>7}{'time':>8}{'err(noisy)':>11}{'err(CLEAN)':>11}")
    full = run(R, lambda: None,                                        "none (2000 epochs)")
    es   = run(R, lambda: EarlyStop(patience=50, min_rel=1e-3),        "EarlyStop 1e-3")
    d10  = run(R, lambda: DiscrepancyStop(SIGMA, N, tau=1.0),          "Discrepancy tau=1.0")
    d12  = run(R, lambda: DiscrepancyStop(SIGMA, N, tau=1.2),          "Discrepancy tau=1.2")
    print(f"  -> vs full run: discrepancy is {full[0]/d10[0]:.1f}x faster, "
          f"clean err {100*(d10[2]/full[2]-1):+.1f}%, noisy err {100*(d10[1]/full[1]-1):+.1f}%\n")
