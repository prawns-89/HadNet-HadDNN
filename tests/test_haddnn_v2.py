"""Can HadDNN be made to earn its cost?

Diagnosis (Sec. 25): the final layer's bias is already full-dimensional, so the
network is an exact reparameterisation of an unconstrained object -- it can
impose no structure, only make the landscape harder.

A factor network is only useful if it CONSTRAINS the factor usefully. Four
attempts, on video where p=180 slices makes structure meaningful:

  B  as shipped                 -- flat net over the whole factor, with bias
  E  linear (no LN/ReLU)        -- the paper's STATED design; code disagrees
  F  no final bias              -- output = W_out @ h, i.e. rank-h structure
  G  per-slice shared decoder   -- W[:,:,i] = decode(z_i), decoder shared over
                                   all 180 slices. The network finally does
                                   something the plain factors cannot: tie the
                                   slices together. (RESEARCH.md Sec. 5 / 8.4)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, cv2
from hadamard_ops import h_prod, svd_initialize

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture("application-videos/video.mp4"); fr=[]
while True:
    ok,f=cap.read()
    if not ok: break
    fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
D = torch.tensor(np.array(fr),dtype=torch.float32,device=dev).permute(1,2,0,3)/255.0
torch.manual_seed(0); D = D + 0.01*torch.randn_like(D)
nrm = torch.norm(D); m,n = D.shape[0], D.shape[1]; P = D.shape[2]*D.shape[3]
RANK, EPOCHS = 10, 1000

def flatnet(d, h=64, linear=False, bias=True):
    L=[nn.Linear(d,h)]
    if not linear: L += [nn.LayerNorm(h), nn.ReLU()]
    L += [nn.Linear(h,h)]
    if not linear: L += [nn.LayerNorm(h), nn.ReLU()]
    L += [nn.Linear(h,d,bias=bias)]
    net=nn.Sequential(*L)
    for mod in net.modules():
        if isinstance(mod,nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None: nn.init.zeros_(mod.bias)
    return net.to(dev)

class SliceDecoder(nn.Module):
    """W[:,:,i] = decode(z_i). One decoder shared across all p slices."""
    def __init__(s, a, b, p, code=32, h=128):
        super().__init__()
        s.a, s.b, s.p = a, b, p
        s.z = nn.Parameter(torch.randn(p, code)*0.1)
        s.dec = nn.Sequential(nn.Linear(code,h), nn.GELU(), nn.Linear(h,a*b))
        for mod in s.dec.modules():
            if isinstance(mod,nn.Linear):
                nn.init.xavier_uniform_(mod.weight); nn.init.zeros_(mod.bias)
    def forward(s):
        return s.dec(s.z).view(s.p, s.a, s.b).permute(1,2,0)   # (a,b,p)

def fit(mode, lr=0.01):
    torch.manual_seed(42)
    F0 = svd_initialize(D, RANK, randomized=False)
    F0 = [x.reshape(x.shape[0], x.shape[1], -1) for x in F0]      # (a,b,P)
    shp = [x.shape for x in F0]
    if mode == "plain":
        ps=[x.clone().requires_grad_(True) for x in F0]; fwd=lambda: ps; params=ps
    elif mode == "slice":
        mods=[SliceDecoder(s[0], s[1], P).to(dev) for s in shp]
        params=[q for mo in mods for q in mo.parameters()]
        fwd=lambda: [mo() for mo in mods]
    else:
        linear = mode in ("linear","linear_nobias")
        bias   = mode not in ("nobias","linear_nobias")
        inp=[nn.Parameter(x.flatten().unsqueeze(0)) for x in F0]
        nets=[flatnet(x.numel(), linear=linear, bias=bias) for x in F0]
        params=list(inp)
        for nt in nets: params += list(nt.parameters())
        fwd=lambda: [nets[i](inp[i]).view(shp[i]) for i in range(4)]
    npar=sum(q.numel() for q in params if q.requires_grad)
    opt=torch.optim.Adam(params, lr=lr); wait,best=100,float('inf')
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(EPOCHS):
        f=fwd()
        loss=torch.norm(D.reshape(m,n,-1) - h_prod(f[0],f[1])*h_prod(f[2],f[3]))
        opt.zero_grad(); loss.backward(); opt.step(); l=loss.item()
        if not np.isfinite(l): return npar, float('nan'), float('nan')
        if l<best: best,wait=l,100
        else: wait-=1
        if wait==0:
            for g in opt.param_groups: g['lr']*=0.1
            wait=100
    torch.cuda.synchronize(); dt=time.time()-t0
    with torch.no_grad():
        f=fwd(); e=(torch.norm(D.reshape(m,n,-1)-h_prod(f[0],f[1])*h_prod(f[2],f[3]))/nrm).item()
    return npar, dt, e

print(f"video {tuple(D.shape)}  rank {RANK}  {EPOCHS} epochs  P={P} slices\n")
print(f"{'variant':<44}{'params':>13}{'time(s)':>9}{'rel_err':>10}")
for mode,tag in (("plain",       "A  plain factors  [= HadNet]"),
                 ("full",        "B  flat net, LN+ReLU, bias  [= HadDNN]"),
                 ("linear",      "E  flat net, LINEAR (paper's stated design)"),
                 ("nobias",      "F  flat net, no final bias -> rank-64"),
                 ("linear_nobias","F2 linear, no final bias"),
                 ("slice",       "G  per-slice shared decoder")):
    npar,dt,e = fit(mode)
    print(f"{tag:<44}{npar:>13,}{dt:>9.2f}{e:>10.5f}", flush=True)
