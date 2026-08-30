"""Does the log-domain fit make a better HadNet init than sqrt(|X|)?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, cv2
from hadamard_ops import h_prod, svd_initialize, log_domain_initialize
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
cap=cv2.VideoCapture("application-videos/video.mp4"); fr=[]
while True:
    ok,f=cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
cap.release()
CLEAN=torch.tensor(np.array(fr),dtype=torch.float32,device=dev).permute(1,2,0,3)/255.0
torch.manual_seed(0); A=CLEAN+0.01*torch.randn_like(CLEAN); nrm=torch.norm(A)
TRACK=(0,50,100,250,500,1000,2000)

def run(rank, init_fn, tag, max_ep=2000):
    torch.manual_seed(42)
    torch.cuda.synchronize(); ti=time.time()
    f0=init_fn(A,rank)
    torch.cuda.synchronize(); t_init=time.time()-ti
    fs=[x.clone().requires_grad_(True) for x in f0]
    opt=torch.optim.Adam(fs,lr=0.1); wait,best=100,float('inf'); curve={}
    with torch.no_grad():
        curve[0]=(torch.norm(A-h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3]))/nrm).item()
    t0=time.time()
    for ep in range(1,max_ep+1):
        loss=torch.norm(A-h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3]))
        opt.zero_grad(); loss.backward(); opt.step(); l=loss.item()
        if l<best: best,wait=l,100
        else: wait-=1
        if wait==0:
            for g in opt.param_groups: g['lr']*=0.1
            wait=100
        if ep in TRACK:
            with torch.no_grad():
                curve[ep]=(torch.norm(A-h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3]))/nrm).item()
    torch.cuda.synchronize()
    print(f"{tag:<22}{t_init:>7.2f}"+"".join(f"{curve[e]:>10.5f}" for e in TRACK)+f"{time.time()-t0:>8.2f}", flush=True)

for R in (10,25):
    print(f"\n=== rank {R} : relative error at epoch (0 = the init itself) ===")
    print(f"{'init':<22}{'t_init':>7}"+"".join(f"{e:>10}" for e in TRACK)+f"{'train':>8}")
    run(R, lambda A,r: svd_initialize(A,r,randomized=False), "sqrt|X| SVD [current]")
    run(R, lambda A,r: log_domain_initialize(A,r,randomized=False), "log-domain")
