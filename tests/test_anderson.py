"""Anderson/DIIS on the epoch loop (RESEARCH.md Sec. 8.1 / 9.2).

The loop is a fixed-point iteration and Sec. 17 shows a long slow tail --
the regime Anderson exists to collapse. Caveat: wrapping Adam is not textbook
Anderson, since Adam's momentum/second-moment state is invalidated by an
extrapolated point. That is what this measures.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, cv2
from hadamard_ops import h_prod, svd_initialize, Anderson

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture("application-videos/video.mp4"); fr=[]
while True:
    ok,f = cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
A = torch.tensor(np.array(fr), dtype=torch.float32, device=dev).permute(1,2,0,3)/255.0
torch.manual_seed(0); A = A + 0.01*torch.randn_like(A); nrm = torch.norm(A)
TRACK=(100,250,500,1000,2000)

def run(rank, acc_cfg, max_ep=2000):
    torch.manual_seed(42)
    fs=[x.clone().requires_grad_(True) for x in svd_initialize(A,rank,randomized=False)]
    opt=torch.optim.Adam(fs,lr=0.1)
    acc=Anderson(**acc_cfg) if acc_cfg else None
    wait,best,curve,nacc=100,float('inf'),{},0
    torch.cuda.synchronize(); t0=time.time()
    for ep in range(1,max_ep+1):
        loss=torch.norm(A-h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3]))
        opt.zero_grad(); loss.backward(); opt.step(); l=loss.item()
        if acc is not None and acc.apply_(fs): nacc+=1
        if l<best: best,wait=l,100
        else: wait-=1
        if wait==0:
            for g in opt.param_groups: g['lr']*=0.1
            wait=100
        if ep in TRACK:
            with torch.no_grad():
                curve[ep]=(torch.norm(A-h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3]))/nrm).item()
    torch.cuda.synchronize()
    return time.time()-t0, curve, nacc

for R in (10,25):
    print(f"\n=== rank {R} : relative error at epoch ===")
    print(f"{'config':<26}"+"".join(f"{e:>10}" for e in TRACK)+f"{'time':>8}{'#acc':>6}")
    for cfg,tag in ((None,"none [baseline]"),
                    (dict(m=3,every=1),"Anderson m=3 every 1"),
                    (dict(m=5,every=1),"Anderson m=5 every 1"),
                    (dict(m=5,every=10),"Anderson m=5 every 10"),
                    (dict(m=5,every=10,beta=0.5),"Anderson m=5 e10 b=0.5")):
        dt,c,n = run(R,cfg)
        print(f"{tag:<26}"+"".join(f"{c[e]:>10.5f}" for e in TRACK)+f"{dt:>8.2f}{n:>6}", flush=True)
