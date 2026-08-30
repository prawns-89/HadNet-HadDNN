"""Do image/video reconstructions leave [0,1], and does constraining help?

Data is normalised to [0,1] and metrics are computed on clamp(0,1) output,
but the training loss is on the UNCLAMPED reconstruction. This measures the
size of that gap and compares four ways of closing it.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
from hadamard_ops import h_prod, svd_initialize, EarlyStop

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = Image.open("application-images/128mri.png").convert("RGB")
D = torch.tensor(np.asarray(img) / 255.0, dtype=torch.float32, device=dev)
nrm = torch.norm(D); r, EP = 10, 1500
print(f"device={dev}  image={tuple(D.shape)}  range=[{D.min():.3f},{D.max():.3f}]  r={r}\n")

def fit(mode, lam=10.0):
    torch.manual_seed(42)
    f = [x.clone().requires_grad_(True) for x in svd_initialize(D, r, randomized=False)]
    opt = torch.optim.Adam(f, lr=0.1); stop = EarlyStop(patience=50, min_rel=1e-3)
    for ep in range(EP):
        pred = h_prod(f[0], f[1]) * h_prod(f[2], f[3])
        if mode == "plain":     loss = torch.norm(D - pred)
        elif mode == "clamp":   loss = torch.norm(D - pred.clamp(0, 1))
        elif mode == "penalty":
            over = torch.relu(pred - 1).pow(2).sum() + torch.relu(-pred).pow(2).sum()
            loss = torch.norm(D - pred) + lam * over.sqrt()
        elif mode == "sigmoid": loss = torch.norm(D - torch.sigmoid(pred))
        opt.zero_grad(); loss.backward(); opt.step()
        if stop.step(loss.item()): break
    with torch.no_grad():
        raw = h_prod(f[0], f[1]) * h_prod(f[2], f[3])
        if mode == "sigmoid": raw = torch.sigmoid(raw)
        out = raw.clamp(0, 1)
        below = (raw < 0).float().mean().item() * 100
        above = (raw > 1).float().mean().item() * 100
        e_raw = (torch.norm(D - raw) / nrm).item()
        e_cl  = (torch.norm(D - out) / nrm).item()
        mse = torch.mean((D - out) ** 2).item()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    print(f"  {mode:8s} k={stop.epochs:4d}  out-of-range: {below:5.2f}% <0  {above:5.2f}% >1"
          f"   rel_err raw={e_raw:.5f} clamped={e_cl:.5f}   PSNR={psnr:5.2f} dB")
    return e_cl, psnr

print("mode      epochs   values outside [0,1]        error / PSNR on clamped output")
res = {m: fit(m) for m in ("plain", "clamp", "penalty", "sigmoid")}
best = max(res.items(), key=lambda kv: kv[1][1])
print(f"\nbest PSNR: {best[0]} ({best[1][1]:.2f} dB)  vs plain ({res['plain'][1]:.2f} dB)"
      f"  -> {best[1][1]-res['plain'][1]:+.2f} dB")
