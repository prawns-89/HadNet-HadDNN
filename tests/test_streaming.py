"""Streaming (slice-by-slice) vs joint batched fit, on real video.

Joint fit needs the whole tensor resident -- O(m*n*p). Streaming needs one
slice plus factors -- O(m*n + r(m+n)), independent of p. The question is what
that costs in accuracy, and whether warm-starting from the previous frame
actually saves epochs.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, cv2
from hadamard_ops import h_prod, svd_initialize, StreamingHadamard

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture("application-videos/video.mp4"); fr=[]
while True:
    ok,f=cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
A = torch.tensor(np.array(fr),dtype=torch.float32,device=dev).permute(1,2,0,3)/255.0
torch.manual_seed(0); A = A + 0.01*torch.randn_like(A)
m,n = A.shape[0], A.shape[1]
flat = A.reshape(m,n,-1); P = flat.shape[2]
nrm = torch.norm(flat)
print(f"video {tuple(A.shape)} -> {P} slices of {m}x{n}\n")

def joint(rank, ep=2000):
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    fs=[x.clone().requires_grad_(True) for x in svd_initialize(flat,rank,randomized=False)]
    opt=torch.optim.Adam(fs,lr=0.1); wait,best=100,float('inf')
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(ep):
        loss=torch.norm(flat-h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3]))
        opt.zero_grad(); loss.backward(); opt.step(); l=loss.item()
        if l<best: best,wait=l,100
        else: wait-=1
        if wait==0:
            for g in opt.param_groups: g['lr']*=0.1
            wait=100
    torch.cuda.synchronize()
    with torch.no_grad():
        e=(torch.norm(flat-h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3]))/nrm).item()
    return time.time()-t0, e, torch.cuda.max_memory_allocated()/1e6, ep*P

def stream(rank, warm, cold=500):
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    s=StreamingHadamard(m,n,rank,device=dev,cold_epochs=cold,warm_epochs=warm)
    err=0.0
    torch.cuda.synchronize(); t0=time.time()
    for i in range(P):
        X=flat[:,:,i].contiguous()
        rec,_=s.push(X)
        err += torch.sum((X-rec)**2).item()
    torch.cuda.synchronize()
    return time.time()-t0, (err**0.5)/nrm.item(), torch.cuda.max_memory_allocated()/1e6, s.total_epochs

for R in (10,25):
    print(f"=== rank {R} ===")
    print(f"{'method':<34}{'time(s)':>9}{'rel_err':>10}{'peak MB':>10}{'slice-epochs':>14}")
    tj,ej,mj,sj = joint(R)
    print(f"{'joint batched (2000 ep)':<34}{tj:>9.2f}{ej:>10.5f}{mj:>10.1f}{sj:>14,}")
    for w in (20,50,100):
        ts,es,ms,ss = stream(R,w)
        print(f"{f'streaming, warm={w} ep':<34}{ts:>9.2f}{es:>10.5f}{ms:>10.1f}{ss:>14,}"
              f"   {mj/ms:>5.1f}x less memory", flush=True)
    print()
