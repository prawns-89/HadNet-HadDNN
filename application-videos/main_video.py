import os
import cv2
import time
import torch
import numpy as np
import itertools
from scipy.linalg import qr 
import torch.nn as nn
import torch.optim as optim

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from Hadamard_BCD import Hadamard_BCD
from alternating_gradient_descent import scaled_alternating_gradient_descent_hadDec as sAGD
from Hadamard_matrix import igd

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- perf: shared optimised kernels (see hadamard_ops.py, docs/DECISIONS.md) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from hadamard_ops import h_prod, svd_initialize, sync, EarlyStop
# --- end perf shim ---


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def svd_transform(D, r):
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    D_approx = (U_r * S_r) @ Vh_r
    return D_approx

def qr_transform(D, r):
    D = D.cpu().numpy()
    Q, R, P = qr(D, pivoting=True, mode='economic')
    Qk = Q[:, :r]
    Rk = R[:r, :]
    P_mat = np.eye(D.shape[1])[:, P]
    D_approx = Qk @ Rk @ P_mat.T
    D_approx = torch.tensor(D_approx, dtype=torch.float32, device=device)
    return D_approx



class HadamardNet(nn.Module):
    def __init__(self, X, r):
        super().__init__()
        W1_init, H1_init, W2_init, H2_init = svd_initialize(X, r)
        self.W1 = nn.Parameter(W1_init)
        self.H1 = nn.Parameter(H1_init)
        self.W2 = nn.Parameter(W2_init)
        self.H2 = nn.Parameter(H2_init)

    def forward(self):
        M1 = h_prod(self.W1, self.H1)
        M2 = h_prod(self.W2, self.H2)
        X_pred_hat = M1 * M2
        X_pred = X_pred_hat
        return X_pred

def HadamardNet_train(D, r, max_epochs=2000):
    wait = 100
    updatecount = 0
    best_err, hist = float('inf'), []
    D = torch.tensor(D, dtype=torch.float32, device=device)
    model = HadamardNet(D, r)
    opt = optim.Adam(model.parameters(), lr=0.1)
    stopper = EarlyStop(patience=50, min_rel=1e-3)
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        X_hat = model()
        loss = torch.norm(D - X_hat, 'fro')
        loss.backward()
        opt.step()
        if stopper.step(loss.item()):
            break
        hist.append(loss.item())

        # if epoch % 500 == 0:
        #     print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

        if loss.item() < best_err:
            best_err = loss.item()
            best_iter = epoch
            wait = 100
        # elif epoch > 50 and np.abs(loss.item() - hist[epoch - 50]) < 0.01*opt.param_groups[0]['lr']:
        #     updatecount += 1
        else:
            wait -= 1
        if wait == 0 or updatecount == 100:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.1
            # print(f"Reducing learning rate to {param_group['lr']} at iteration {epoch}")
            wait = 100
            updatecount = 0
        # print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        # print(f"Epoch {epoch}, Relative Loss: {loss.item()/torch.norm(D,'fro').item():.6f}")

    W1 = torch.tensor(model.W1.detach().cpu().numpy())
    H1 = torch.tensor(model.H1.detach().cpu().numpy())
    W2 = torch.tensor(model.W2.detach().cpu().numpy())
    H2 = torch.tensor(model.H2.detach().cpu().numpy())
    return W1, H1, W2, H2


class FactorNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class HadamardDNN(nn.Module):
    def __init__(self, D, r, device='cuda', use_factor_nets=True):
        super().__init__()
        self.device = device
        self.r = r
        self.D = D
        self.shapes = D.shape
        self.m, self.n = self.shapes[0], self.shapes[1]
        self.extraDim = self.shapes[2:]
        self.p = np.prod(self.extraDim)
        self.outDimW = (self.m, self.r) + self.extraDim
        self.outDimH = (self.r, self.n) + self.extraDim
        self.use_factor_nets = use_factor_nets

        W1_init, H1_init, W2_init, H2_init = svd_initialize(self.D, r)

        if use_factor_nets:
            self.W1_input = nn.Parameter(W1_init.flatten().unsqueeze(0))
            self.H1_input = nn.Parameter(H1_init.flatten().unsqueeze(0))
            self.W2_input = nn.Parameter(W2_init.flatten().unsqueeze(0))
            self.H2_input = nn.Parameter(H2_init.flatten().unsqueeze(0))
            
            self.W1_net = FactorNet(self.m * self.r * self.p, self.m * self.r * self.p).to(device)
            self.H1_net = FactorNet(self.r * self.n * self.p, self.r * self.n * self.p).to(device)
            self.W2_net = FactorNet(self.m * self.r * self.p, self.m * self.r * self.p).to(device)
            self.H2_net = FactorNet(self.r * self.n * self.p, self.r * self.n * self.p).to(device)
            
        else:
            self.W1 = nn.Parameter(W1_init)
            self.H1 = nn.Parameter(H1_init)
            self.W2 = nn.Parameter(W2_init)
            self.H2 = nn.Parameter(H2_init)

    def forward(self, base_only=False):
        if self.use_factor_nets:
            W1_out = self.W1_net(self.W1_input).view(self.outDimW)
            H1_out = self.H1_net(self.H1_input).view(self.outDimH)
            W2_out = self.W2_net(self.W2_input).view(self.outDimW)
            H2_out = self.H2_net(self.H2_input).view(self.outDimH)
        else:
            W1_out = self.W1.to(self.M.device)
            H1_out = self.H1.to(self.M.device)
            W2_out = self.W2.to(self.M.device)
            H2_out = self.H2.to(self.M.device)

        M1 = h_prod(W1_out, H1_out)
        M2 = h_prod(W2_out, H2_out)
        X_pred_hat = M1 * M2
        X_pred = X_pred_hat

        if base_only:
            return X_pred, W1_out, H1_out, W2_out, H2_out
        else:
            return X_pred

def HadamardDNN_train(D, r, max_epochs=2000, use_factor_nets=True, device=None):
    # perf: was hardcoded 'cuda', which crashes on CPU-only builds
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    D = D.to(device)
    model = HadamardDNN(D, r, device=device, use_factor_nets=use_factor_nets).to(device)

    if use_factor_nets:
        params = [model.W1_input, model.H1_input, model.W2_input, model.H2_input] + list(model.W1_net.parameters()) + list(model.H1_net.parameters()) + list(model.W2_net.parameters()) + list(model.H2_net.parameters())
    else:
        params = [model.W1, model.H1, model.W2, model.H2]

    opt = optim.Adam(params, lr=0.1)

    stopper = EarlyStop(patience=50, min_rel=1e-3)
    wait = 500
    best_err = float('inf')
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        X_hat = model()
        loss = torch.norm(D - X_hat, p='fro')
        loss.backward()
        opt.step()
        if stopper.step(loss.item()):
            break

        # if epoch % 500 == 0:
        #     print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

        if loss.item() < best_err:
            best_err = loss.item()
            wait = 500
        else:
            wait -= 1
        if wait == 0:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.1
            # print(f"Reducing learning rate to {param_group['lr']} at iteration {epoch}")
            wait = 500

        # print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    X_base, W1_out, H1_out, W2_out, H2_out = model.forward(base_only=True)

    W1 = torch.tensor(W1_out.detach().cpu().numpy())
    H1 = torch.tensor(H1_out.detach().cpu().numpy())
    W2 = torch.tensor(W2_out.detach().cpu().numpy())
    H2 = torch.tensor(H2_out.detach().cpu().numpy())
    return W1, H1, W2, H2





video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)
frames = []
max_frames = 0
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
    count += 1

cap.release()

A = torch.tensor(np.array(frames), dtype=torch.float32) / 255.0
A = A.permute(1, 2, 0, 3)
print('Video tensor shape:', A.shape)
norm_tensor = torch.norm(A, 'fro').item()

np.random.seed(42)
torch.manual_seed(42)
ranks = [2,5,10,20,25]


approx_results = []

shape = A.shape
n1, n2 = shape[0], shape[1]
remShape = shape[2:]
out_shape = (A.shape[0], A.shape[1]) + remShape

A = A + 0.01*torch.randn_like(A)

# # Save frames as .png in original folder
# os.makedirs('original_frames', exist_ok=True)
# for i, frame in enumerate(frames):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(f'original_frames/frame_{i:03d}.png', frame_rgb)

for rank in ranks:
    result = {'rank': rank}
    
    print(f"Initial Error: {norm_tensor:.6f}")

    # Benchmark Tensor SVD
    sync()
    start = time.time()
    X_svd = torch.empty(out_shape, dtype=A.dtype, device=A.device)
    for i in itertools.product(*[range(s) for s in remShape]):
        # print('Processing slice:', i, 'out of', remShape)
        matA = A[(...,)+i]
        X_svd[(...,)+i] = svd_transform(matA, r=rank)
    sync()
    end = time.time()
    err_svd = torch.norm(A - X_svd, 'fro').item()
    result['SVD'] = {
        'recon': X_svd,
        'error': err_svd,
        'rel_error': err_svd / norm_tensor,
        'time': end - start
    }
    print(f"SVD Rank {rank}: Time {end - start:.4f}s, Error: {err_svd:.6f}, Rel Error {err_svd / norm_tensor:.6f}")
    
    # Save video approximation
    X_svd_vid = X_svd.clamp(0,1).permute(2, 0, 1, 3).cpu().numpy() * 255
    X_svd_vid = X_svd_vid.astype(np.uint8)

    # out_svd = cv2.VideoWriter(f'approx_svd_rank{rank}.mp4',
    #                         cv2.VideoWriter_fourcc(*'mp4v'),
    #                         30,
    #                         (X_svd_vid.shape[2], X_svd_vid.shape[1]))  # width, height

    # for frame in X_svd_vid:  # iterate over time axis (frames)
    #     out_svd.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # out_svd.release()
    
    # Calculate PSNR and SSIM for SVD approximation
    psnr_values = []
    ssim_values = []
    for i in range(X_svd_vid.shape[0]):
        original_frame = frames[i]
        approx_frame = X_svd_vid[i]
        psnr_val = psnr(original_frame, approx_frame, data_range=255)
        ssim_val = ssim(original_frame, approx_frame, channel_axis=2, data_range=255)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"SVD Rank {rank}: Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

    # os.makedirs('svd_frames', exist_ok=True)
    # for i, frame in enumerate(X_svd_vid):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f'svd_frames/frame_{i:03d}.png', frame_rgb)

    # Benchmark Tensor QR
    sync()
    start = time.time()
    X_qr = torch.empty(out_shape, dtype=A.dtype, device=A.device)
    for i in itertools.product(*[range(s) for s in remShape]):
        # print('Processing slice:', i, 'out of', remShape)
        matA = A[(...,)+i]
        X_qr[(...,)+i] = qr_transform(matA, r=rank)
    sync()
    end = time.time()
    err_qr = torch.norm(A - X_qr, 'fro').item()
    result['QR'] = {
        'recon': X_qr,
        'error': err_qr,
        'rel_error': err_qr / norm_tensor,
        'time': end - start
    }
    print(f"QR Rank {rank}: Time {end - start:.4f}s, Error: {err_qr:.6f}, Rel Error {err_qr / norm_tensor:.6f}")
    
    # Save video approximation
    X_qr_vid = X_qr.clamp(0,1).permute(2, 0, 1, 3).cpu().numpy() * 255
    X_qr_vid = X_qr_vid.astype(np.uint8)
    
    # out_qr = cv2.VideoWriter(f'approx_qr_rank{rank}.mp4',
    #                         cv2.VideoWriter_fourcc(*'mp4v'),
    #                         30,
    #                         (X_qr_vid.shape[2], X_qr_vid.shape[1]))  # width, height    
    # for frame in X_qr_vid:  # iterate over time axis (frames)
    #     out_qr.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # out_qr.release()

    # Calculate PSNR and SSIM for QR approximation
    psnr_values = []
    ssim_values = []
    for i in range(X_qr_vid.shape[0]):
        original_frame = frames[i]
        approx_frame = X_qr_vid[i]
        psnr_val = psnr(original_frame, approx_frame, data_range=255)
        ssim_val = ssim(original_frame, approx_frame, channel_axis=2, data_range=255)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"QR Rank {rank}: Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

    # os.makedirs('qr_frames', exist_ok=True)
    # for i, frame in enumerate(X_qr_vid):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f'qr_frames/frame_{i:03d}.png', frame_rgb)


    eta = 0.01
    maxIter = 40000

    # Benchmark AGD
    sync()
    start = time.time()
    agd_A1 = torch.zeros((n1, rank) + remShape, dtype=torch.float32)
    agd_B1 = torch.zeros((rank, n2) + remShape, dtype=torch.float32)
    agd_A2 = torch.zeros((n1, rank) + remShape, dtype=torch.float32)
    agd_B2 = torch.zeros((rank, n2) + remShape, dtype=torch.float32)
    p = np.prod(remShape)
    a1, b1, a2, b2 = np.zeros((n1, rank)), np.zeros((rank, n2)), np.zeros((n1, rank)), np.zeros((rank, n2))
    for i in itertools.product(*[range(s) for s in remShape]):
        matA = A[(...,)+i].cpu().numpy()
        _,  [_, _], [a1, b1, a2, b2], [_, _] , _ = sAGD(matA, rank, eta, maxIter)
        agd_A1[(...,)+i], agd_B1[(...,)+i], agd_A2[(...,)+i], agd_B2[(...,)+i] = torch.tensor(a1), torch.tensor(b1), torch.tensor(a2), torch.tensor(b2)
    M1 = h_prod(agd_A1, agd_B1)
    M2 = h_prod(agd_A2, agd_B2)
    A_AGD = M1 * M2
    sync()
    end = time.time()
    err_AGD = torch.norm(A - A_AGD, 'fro').item()
    result['AGD'] = {
        'recon': A_AGD,
        'error': err_AGD,
        'rel_error': err_AGD / norm_tensor,
        'time': end - start
    }
    print(f"AGD Rank {rank}: Time {end - start:.4f}s, Error: {err_AGD:.6f}, Rel Error {err_AGD / norm_tensor:.6f}")
    
    # Save video approximation
    A_AGD_vid = A_AGD.clamp(0,1).permute(2, 0, 1, 3).cpu().numpy() * 255
    A_AGD_vid = A_AGD_vid.astype(np.uint8)

    # out_agd = cv2.VideoWriter(f'approx_agd_rank{rank}.mp4',
    #                         cv2.VideoWriter_fourcc(*'mp4v'),
    #                         30,
    #                         (A_AGD_vid.shape[2], A_AGD_vid.shape[1]))  # width, height
    # for frame in A_AGD_vid:  # iterate over time axis (frames)
    #     out_agd.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # out_agd.release()
    
    # Calculate PSNR and SSIM for AGD approximation
    psnr_values = []
    ssim_values = []
    for i in range(A_AGD_vid.shape[0]):
        original_frame = frames[i]
        approx_frame = A_AGD_vid[i]
        psnr_val = psnr(original_frame, approx_frame, data_range=255)
        ssim_val = ssim(original_frame, approx_frame, channel_axis=2, data_range=255)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"AGD Rank {rank}: Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")
    
    # os.makedirs('agd_frames', exist_ok=True)    
    # for i, frame in enumerate(A_AGD_vid):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f'agd_frames/frame_{i:03d}.png', frame_rgb)


    # Benchmark IGD
    sync()
    start = time.time()
    igd_A1 = torch.zeros((n1, rank) + remShape, dtype=torch.float32)
    igd_B1 = torch.zeros((rank, n2) + remShape, dtype=torch.float32)
    igd_A2 = torch.zeros((n1, rank) + remShape, dtype=torch.float32)
    igd_B2 = torch.zeros((rank, n2) + remShape, dtype=torch.float32)
    D = A.cpu().numpy()
    p = np.prod(remShape)
    for i in itertools.product(*[range(s) for s in remShape]):
        matA = D[(...,)+i]
        igd_A1[(...,)+i], igd_A2[(...,)+i], igd_B1[(...,)+i], igd_B2[(...,)+i] = igd(matA, rank, maxIter)
    M1 = h_prod(igd_A1, igd_B1)
    M2 = h_prod(igd_A2, igd_B2)
    A_IGD = M1 * M2
    sync()
    end = time.time()
    err_IGD = torch.norm(A - A_IGD, 'fro').item()
    result['IGD'] = {
        'recon': A_IGD,
        'error': err_IGD,
        'rel_error': err_IGD / norm_tensor,
        'time': end - start
    }
    approx_results.append(result)
    print(f"IGD Rank {rank}: Time {end - start:.4f}s, Error: {err_IGD:.6f}, Rel Error {err_IGD / norm_tensor:.6f}") 
    
    # Save video approximation
    A_IGD_vid = A_IGD.clamp(0,1).permute(2, 0, 1, 3).cpu().numpy() * 255
    A_IGD_vid = A_IGD_vid.astype(np.uint8)

    # out_igd = cv2.VideoWriter(f'approx_igd_rank{rank}.mp4',
    #                         cv2.VideoWriter_fourcc(*'mp4v'),
    #                         30,
    #                         (A_IGD_vid.shape[2], A_IGD_vid.shape[1]))  # width, height
    # for frame in A_IGD_vid:  # iterate over time axis (frames)
    #     out_igd.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # out_igd.release()

    # Calculate PSNR and SSIM for IGD approximation
    psnr_values = []
    ssim_values = []
    for i in range(A_IGD_vid.shape[0]):
        original_frame = frames[i]
        approx_frame = A_IGD_vid[i]
        psnr_val = psnr(original_frame, approx_frame, data_range=255)
        ssim_val = ssim(original_frame, approx_frame, channel_axis=2, data_range=255)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"IGD Rank {rank}: Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

    # os.makedirs('igd_frames', exist_ok=True)
    # for i, frame in enumerate(A_IGD_vid):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f'igd_frames/frame_{i:03d}.png', frame_rgb)


    # Benchmark BCD
    sync()
    start = time.time()
    bcd_A1, bcd_B1 = torch.zeros((n1, rank) + remShape, dtype=torch.float32), torch.zeros((rank, n2) + remShape, dtype=torch.float32)
    bcd_A2, bcd_B2 = torch.zeros((n1, rank) + remShape, dtype=torch.float32), torch.zeros((rank, n2) + remShape, dtype=torch.float32)
    D = A.cpu().numpy()
    p = np.prod(remShape)
    a1, b1, a2, b2 = np.zeros((n1, rank)), np.zeros((rank, n2)), np.zeros((n1, rank)), np.zeros((rank, n2))
    for i in itertools.product(*[range(s) for s in remShape]):
        matA = D[(...,)+i]
        a1, b1, a2, b2, _, __ = Hadamard_BCD(matA, r=rank)
        bcd_A1[(...,)+i], bcd_B1[(...,)+i], bcd_A2[(...,)+i], bcd_B2[(...,)+i] = torch.tensor(a1), torch.tensor(b1), torch.tensor(a2), torch.tensor(b2)
    M1 = h_prod(bcd_A1, bcd_B1)
    M2 = h_prod(bcd_A2, bcd_B2)
    A_BCD = M1 * M2
    sync()
    end = time.time()
    err_BCD = torch.norm(A - A_BCD, 'fro').item()
    result['BCD'] = {
        'recon': A_BCD,
        'error': err_BCD,
        'rel_error': err_BCD / norm_tensor,
        'time': end - start
    }
    approx_results.append(result)
    print(f"BCD Rank {rank}: Time {end - start:.4f}s, Error: {err_BCD:.6f}, Rel Error {err_BCD / norm_tensor:.6f}")
    
    # Save video approximation
    A_BCD_vid = A_BCD.clamp(0,1).permute(2, 0, 1, 3).cpu().numpy() * 255
    A_BCD_vid = A_BCD_vid.astype(np.uint8)

    # out_bcd = cv2.VideoWriter(f'approx_bcd_rank{rank}.mp4',
    #                         cv2.VideoWriter_fourcc(*'mp4v'),
    #                         30,
    #                         (A_BCD_vid.shape[2], A_BCD_vid.shape[1]))  # width, height
    # for frame in A_BCD_vid:  # iterate over time axis (frames)
    #     out_bcd.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # out_bcd.release()

    # Calculate PSNR and SSIM for BCD approximation
    psnr_values = []
    ssim_values = []
    for i in range(A_BCD_vid.shape[0]):
        original_frame = frames[i]
        approx_frame = A_BCD_vid[i]
        psnr_val = psnr(original_frame, approx_frame, data_range=255)
        ssim_val = ssim(original_frame, approx_frame, channel_axis=2, data_range=255)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"BCD Rank {rank}: Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

    # os.makedirs('bcd_frames', exist_ok=True)
    # for i, frame in enumerate(A_BCD_vid):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f'bcd_frames/frame_{i:03d}.png', frame_rgb)


    # HadamardNet
    sync()
    start = time.time()
    W1, H1, W2, H2 = HadamardNet_train(A, rank)
    M1 = h_prod(W1, H1)
    M2 = h_prod(W2, H2)
    X_hd = M1 * M2
    sync()
    end = time.time()
    err_hd = torch.norm(A - X_hd, 'fro').item()
    result['HadamardNet'] = {
        'recon': X_hd,
        'error': err_hd,
        'rel_error': err_hd / norm_tensor,
        'time': end - start
    }
    approx_results.append(result)
    print(f"HadamardNet Rank {rank}: Time {end - start:.4f}s, Error: {err_hd:.6f}, Rel Error {err_hd / norm_tensor:.6f}")
    
    # Save video approximation
    X_hd_vid = X_hd.clamp(0,1).permute(2, 0, 1, 3).cpu().numpy() * 255
    X_hd_vid = X_hd_vid.astype(np.uint8)
    
    # out_hd = cv2.VideoWriter(f'approx_hadamardnet_rank{rank}.mp4',
    #                         cv2.VideoWriter_fourcc(*'mp4v'),
    #                         30,
    #                         (X_hd_vid.shape[2], X_hd_vid.shape[1]))  # width, height
    # for frame in X_hd_vid:  # iterate over time axis (frames)
    #     out_hd.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # out_hd.release()
    
    # Calculate PSNR and SSIM for HadamardNet approximation
    psnr_values = []
    ssim_values = []
    for i in range(X_hd_vid.shape[0]):
        original_frame = frames[i]
        approx_frame = X_hd_vid[i]
        psnr_val = psnr(original_frame, approx_frame, data_range=255)
        ssim_val = ssim(original_frame, approx_frame, channel_axis=2, data_range=255)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"HadamardNet Rank {rank}: Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

    # os.makedirs('hd_frames', exist_ok=True)
    # for i, frame in enumerate(X_hd_vid):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f'hd_frames/frame_{i:03d}.png', frame_rgb)

    # HadamardDNN
    sync()
    start = time.time()
    W1, H1, W2, H2 = HadamardDNN_train(A, rank)
    M1 = h_prod(W1, H1)
    M2 = h_prod(W2, H2)
    X_dnn = M1 * M2
    sync()
    end = time.time()
    err_dnn = torch.norm(A - X_dnn, 'fro').item()
    result['HadamardDNN'] = {
        'recon': X_dnn,
        'error': err_dnn,
        'rel_error': err_dnn / norm_tensor,
        'time': end - start
    }
    approx_results.append(result)
    print(f"HadamardDNN Rank {rank}: Time {end - start:.4f}s, Error: {err_dnn:.6f}, Rel Error {err_dnn / norm_tensor:.6f}")
    
    # Save video approximation
    X_dnn_vid = X_dnn.clamp(0,1).permute(2, 0, 1, 3).cpu().numpy() * 255
    X_dnn_vid = X_dnn_vid.astype(np.uint8)
    
    # out_dnn = cv2.VideoWriter(f'approx_hadamarddnn_rank{rank}.mp4',
    #                         cv2.VideoWriter_fourcc(*'mp4v'),
    #                         30,
    #                         (X_dnn_vid.shape[2], X_dnn_vid.shape[1]))  # width, height
    # for frame in X_dnn_vid:  # iterate over time axis (frames)
    #     out_dnn.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # out_dnn.release()
    
    # Calculate PSNR and SSIM for HadamardDNN approximation
    psnr_values = []
    ssim_values = []
    for i in range(X_dnn_vid.shape[0]):
        original_frame = frames[i]
        approx_frame = X_dnn_vid[i]
        psnr_val = psnr(original_frame, approx_frame, data_range=255)
        ssim_val = ssim(original_frame, approx_frame, channel_axis=2, data_range=255)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"HadamardDNN Rank {rank}: Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

    # os.makedirs('dnn_frames', exist_ok=True)
    # for i, frame in enumerate(X_dnn_vid):
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(f'dnn_frames/frame_{i:03d}.png', frame_rgb)

