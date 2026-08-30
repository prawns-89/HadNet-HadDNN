"""Anderson's early-phase gain + discrepancy stopping's early exit.

Sec. 21: Anderson every-10 reaches at epoch 100 what the baseline needs ~200
for, but both converge to the same place by 2000 -- so it was worthless when
running to 2000. Sec. 19 now stops at ~300. This tests whether the acceleration
pays in the regime actually used, and whether the fixed beta damping helps.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, cv2
from hadamard_ops import h_prod, svd_initialize, Anderson, DiscrepancyStop

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA = 0.01
cap = cv2.VideoCapture("application-videos/video.mp4"); fr=[]
while True:
    ok,f=cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
CLEAN = torch.tensor(np.array(fr), dtype=torch.float32, device=dev).permute(1,2,0,3)/255.0
torch.manual_seed(0); NOISY = CLEAN + SIGMA*torch.randn_like(CLEAN); N = CLEAN.numel()

def run(rank, acc_cfg, use_stop, max_ep=2000):
    torch.manual_seed(42)
    fs=[x.clone().requires_grad_(True) for x in svd_initialize(NOISY,rank,randomized=False)]
    opt=torch.optim.Adam(fs,lr=0.1)
    acc=Anderson(**acc_cfg) if acc_cfg else None
    st=DiscrepancyStop(SIGMA,N,tau=0.8) if use_stop else None
    wait,best=100,float('inf')
    torch.cuda.synchronize(); t0=time.time()
    ep=0
    for ep in range(1,max_ep+1):
        loss=torch.norm(NOISY-h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3]))
        opt.zero_grad(); loss.backward(); opt.step(); l=loss.item()
        if acc is not None: acc.apply_(fs)
        if l<best: best,wait=l,100
        else: wait-=1
        if wait==0:
            for g in opt.param_groups: g['lr']*=0.1
            wait=100
        if st is not None and st.step(l): break
    torch.cuda.synchronize(); dt=time.time()-t0
    with torch.no_grad():
        rec=h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3])
        return ep, dt, (torch.norm(NOISY-rec)/torch.norm(NOISY)).item(), \
                       (torch.norm(CLEAN-rec)/torch.norm(CLEAN)).item()

for R in (20,25):
    print(f"\n=== rank {R} ===")
    print(f"{'config':<34}{'epochs':>7}{'time':>8}{'err(noisy)':>12}{'err(CLEAN)':>12}")
    full = run(R, None, False)
    print(f"{'full 2000, no accel':<34}{full[0]:>7}{full[1]:>8.2f}{full[2]:>12.5f}{full[3]:>12.5f}")
    for cfg, use, tag in (
        (None,                        True,  "discrepancy only [Sec.19]"),
        (dict(m=5,every=10),          True,  "discrepancy + Anderson e10"),
        (dict(m=5,every=10,beta=0.5), True,  "discrepancy + Anderson b=0.5"),
        (dict(m=5,every=25),          True,  "discrepancy + Anderson e25"),
    ):
        ep,dt,en,ec = run(R,cfg,use)
        print(f"{tag:<34}{ep:>7}{dt:>8.2f}{en:>12.5f}{ec:>12.5f}"
              f"   {full[1]/dt:>5.1f}x  clean {100*(ec/full[3]-1):+6.1f}%", flush=True)
