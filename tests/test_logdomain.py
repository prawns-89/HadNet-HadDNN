"""Closed-form log-domain fit vs iterative HadNet, at matched parameter count.

Log-domain has no optimiser: one truncated SVD of log|X|. It deletes k rather
than reducing it. The question is what that costs in the Frobenius metric the
paper reports, since the fit minimises error in LOG space instead.

Parameter accounting (per slice):
    HadNet rank r  : 2r(m+n)   -- W1,H1,W2,H2
    log-domain L   :  r(m+n)   -- U,V
so log-domain at rank 2r is the matched-parameter comparison.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, cv2
from hadamard_ops import h_prod, svd_initialize, log_domain_fit

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture("application-videos/video.mp4"); fr=[]
while True:
    ok,f=cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
CLEAN = torch.tensor(np.array(fr), dtype=torch.float32, device=dev).permute(1,2,0,3)/255.0
torch.manual_seed(0); A = CLEAN + 0.01*torch.randn_like(CLEAN)
nrm, ncl = torch.norm(A), torch.norm(CLEAN)
m,n = A.shape[0], A.shape[1]; p = A.shape[2]*A.shape[3]

def hadnet(rank, max_ep=2000):
    torch.manual_seed(42)
    fs=[x.clone().requires_grad_(True) for x in svd_initialize(A,rank,randomized=False)]
    opt=torch.optim.Adam(fs,lr=0.1); wait,best=100,float('inf')
    torch.cuda.synchronize(); t0=time.time()
    for ep in range(max_ep):
        loss=torch.norm(A-h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3]))
        opt.zero_grad(); loss.backward(); opt.step(); l=loss.item()
        if l<best: best,wait=l,100
        else: wait-=1
        if wait==0:
            for g in opt.param_groups: g['lr']*=0.1
            wait=100
    torch.cuda.synchronize()
    with torch.no_grad(): rec=h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3])
    return time.time()-t0, rec, 2*rank*(m+n)*p

def logfit(rank):
    torch.cuda.synchronize(); t0=time.time()
    rec = log_domain_fit(A, rank, randomized=False)
    torch.cuda.synchronize()
    return time.time()-t0, rec, rank*(m+n)*p

print(f"video {tuple(A.shape)}\n")
print(f"{'method':<28}{'rank':>5}{'params':>12}{'time(s)':>10}{'err(noisy)':>12}{'err(CLEAN)':>12}")
for R in (10, 25):
    dt, rec, np_ = hadnet(R)
    en=(torch.norm(A-rec)/nrm).item(); ec=(torch.norm(CLEAN-rec)/ncl).item()
    print(f"{'HadNet (2000 ep)':<28}{R:>5}{np_:>12,}{dt:>10.2f}{en:>12.5f}{ec:>12.5f}")
    for LR in (R, 2*R):
        dt2, rec2, np2 = logfit(LR)
        en2=(torch.norm(A-rec2)/nrm).item(); ec2=(torch.norm(CLEAN-rec2)/ncl).item()
        tag = "log-domain (closed form)" + ("  [matched params]" if LR==2*R else "")
        print(f"{tag:<28}{LR:>5}{np2:>12,}{dt2:>10.4f}{en2:>12.5f}{ec2:>12.5f}"
              f"   {dt/dt2:>7.0f}x faster", flush=True)
    print()
