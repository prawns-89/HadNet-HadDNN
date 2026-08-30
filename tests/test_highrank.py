"""How far does the Hadamard advantage over SVD extend in rank?

The paper uses full-rank inputs but only tests approximation ranks <=25 --
10-20% of the smaller dimension. It never shows where the Hadamard model stops
beating SVD. This sweeps to half-rank and saves reconstructions for inspection.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
from hadamard_ops import h_prod, svd_initialize

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAME = sys.argv[1] if len(sys.argv) > 1 else "mri"
img = Image.open(f"application-images/128{NAME}.png").convert("RGB")
arr = np.asarray(img)/255.0
D = torch.tensor(arr, dtype=torch.float32, device=dev)
torch.manual_seed(0)
m, n, p = D.shape
nrm = torch.norm(D)
print(f"image={NAME} {tuple(D.shape)}  min(m,n)={min(m,n)}\n")
os.makedirs("bench/recon", exist_ok=True)
Image.fromarray((arr*255).astype(np.uint8)).save("bench/recon/original.png")

def save(t, name):
    a = t.clamp(0,1).cpu().numpy()
    Image.fromarray((a*255).astype(np.uint8)).save(f"bench/recon/{name}.png")

def svd_fit(rank):
    t0=time.time()
    out=torch.empty_like(D)
    for i in range(p):
        U,S,Vh = torch.linalg.svd(D[:,:,i], full_matrices=False)
        out[:,:,i] = (U[:,:rank]*S[:rank]) @ Vh[:rank,:]
    torch.cuda.synchronize() if dev.type=='cuda' else None
    return time.time()-t0, out, rank*(m+n)*p

def hadnet(rank, max_ep=2000):
    torch.manual_seed(42)
    fs=[x.clone().requires_grad_(True) for x in svd_initialize(D,rank,randomized=False)]
    opt=torch.optim.Adam(fs,lr=0.1); wait,best=100,float('inf')
    torch.cuda.synchronize() if dev.type=='cuda' else None
    t0=time.time()
    for _ in range(max_ep):
        loss=torch.norm(D-h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3]))
        opt.zero_grad(); loss.backward(); opt.step(); l=loss.item()
        if l<best: best,wait=l,100
        else: wait-=1
        if wait==0:
            for g in opt.param_groups: g['lr']*=0.1
            wait=100
    torch.cuda.synchronize() if dev.type=='cuda' else None
    dt=time.time()-t0
    with torch.no_grad(): out=h_prod(fs[0],fs[1])*h_prod(fs[2],fs[3])
    return dt, out, 2*rank*(m+n)*p

def psnr(x):
    mse=torch.mean((D-x.clamp(0,1))**2).item()
    return 10*np.log10(1.0/max(mse,1e-12))

print(f"{'rank':>5}{'%of dim':>9}  {'SVD err':>9}{'SVD psnr':>9}{'SVD par':>10}   "
      f"{'Had err':>9}{'Had psnr':>9}{'Had par':>10}{'Had t':>8}   winner")
RANKS=[2,5,10,25,50,75,100,128]
for r in RANKS:
    if r > min(m,n): continue
    ts,xs,ps = svd_fit(r)
    th,xh,ph = hadnet(r)
    es=(torch.norm(D-xs)/nrm).item(); eh=(torch.norm(D-xh)/nrm).item()
    save(xs, f"svd_r{r:03d}"); save(xh, f"hadnet_r{r:03d}")
    # SVD at MATCHED PARAMETERS uses rank 2r
    r2=min(2*r, min(m,n))
    ts2,xs2,ps2 = svd_fit(r2)
    es2=(torch.norm(D-xs2)/nrm).item()
    save(xs2, f"svdmatched_r{r:03d}")
    win = "Hadamard" if eh < es2 else "SVD(2r)"
    print(f"{r:>5}{100*r/min(m,n):>8.0f}%  {es:>9.5f}{psnr(xs):>9.2f}{ps:>10,}   "
          f"{eh:>9.5f}{psnr(xh):>9.2f}{ph:>10,}{th:>8.1f}   {win}  "
          f"[SVD r={r2}: {es2:.5f}]", flush=True)
