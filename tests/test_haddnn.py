"""Do HadDNN's factor networks do anything?

Architecture per factor:  Linear(d->64) -> LN -> ReLU -> Linear(64->64) -> LN
-> ReLU -> Linear(64->d).  The FINAL LAYER'S BIAS is already d-dimensional, so
it alone can represent any factor. The 64-unit path adds a 64-dim correction
on top, at the cost of ~129x overparameterisation.

Four regimes, same seed, same init, same budget:
  A  plain factors                    (= HadNet; the repo's use_factor_nets=False)
  B  full FactorNet                   (= HadDNN as shipped)
  C  FactorNet, weights FROZEN, bias trainable   <- if this matches B, nets are decorative
  D  full FactorNet, hidden_dim 256   <- if the bottleneck binds, this should help
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from PIL import Image
from hadamard_ops import h_prod, svd_initialize

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = Image.open("application-images/128mri.png").convert("RGB")
D = torch.tensor(np.asarray(img)/255.0, dtype=torch.float32, device=dev)
torch.manual_seed(0); D = D + 0.01*torch.randn_like(D)
nrm = torch.norm(D); m,n,p = D.shape

class FactorNet(nn.Module):
    def __init__(s, d, h=64):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d,h), nn.LayerNorm(h), nn.ReLU(),
                              nn.Linear(h,h), nn.LayerNorm(h), nn.ReLU(),
                              nn.Linear(h,d))
        for mod in s.net.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight); nn.init.zeros_(mod.bias)
    def forward(s,x): return s.net(x)

def run(mode, rank=25, h=64, max_ep=1500):
    torch.manual_seed(42)
    W1,H1,W2,H2 = svd_initialize(D, rank, randomized=False)
    shp = [W1.shape, H1.shape, W2.shape, H2.shape]
    if mode == "plain":
        ps=[x.clone().requires_grad_(True) for x in (W1,H1,W2,H2)]
        fwd=lambda: ps
        params=ps
    else:
        inputs=[nn.Parameter(x.flatten().unsqueeze(0)) for x in (W1,H1,W2,H2)]
        nets=[FactorNet(x.numel(), h).to(dev) for x in (W1,H1,W2,H2)]
        params=list(inputs)
        for net in nets:
            if mode == "frozen":
                for q in net.parameters(): q.requires_grad_(False)
                net.net[-1].bias.requires_grad_(True)          # final bias only
                params += [net.net[-1].bias]
            else:
                params += list(net.parameters())
        fwd=lambda: [nets[i](inputs[i]).view(shp[i]) for i in range(4)]
    n_tr = sum(q.numel() for q in params if q.requires_grad)
    opt=torch.optim.Adam(params, lr=0.1)
    torch.cuda.synchronize() if dev.type=='cuda' else None
    t0=time.time(); wait,best=100,float('inf')
    for ep in range(max_ep):
        f=fwd()
        loss=torch.norm(D - h_prod(f[0],f[1])*h_prod(f[2],f[3]))
        opt.zero_grad(); loss.backward(); opt.step(); l=loss.item()
        if l<best: best,wait=l,100
        else: wait-=1
        if wait==0:
            for g in opt.param_groups: g['lr']*=0.1
            wait=100
    torch.cuda.synchronize() if dev.type=='cuda' else None
    dt=time.time()-t0
    with torch.no_grad():
        f=fwd(); e=(torch.norm(D-h_prod(f[0],f[1])*h_prod(f[2],f[3]))/nrm).item()
    return n_tr, dt, e

print(f"image 256x256x3, rank 25, 1500 epochs\n")
print(f"{'regime':<40}{'trainable':>14}{'time(s)':>10}{'rel_err':>10}")
for mode,h,tag in (("plain",0,   "A  plain factors  [= HadNet]"),
                   ("full",64,   "B  FactorNet h=64  [= HadDNN]"),
                   ("frozen",64, "C  FactorNet, weights FROZEN, bias only"),
                   ("full",256,  "D  FactorNet h=256")):
    n,dt,e = run(mode, h=h)
    print(f"{tag:<40}{n:>14,}{dt:>10.2f}{e:>10.5f}", flush=True)
