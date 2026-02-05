import time
import numpy as np
import torch
from PIL import Image

from Hadamard_BCD import Hadamard_BCD
from alternating_gradient_descent import scaled_alternating_gradient_descent_hadDec as sAGD
from Hadamard_matrix import igd

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.optim as optim
import time
from scipy.linalg import qr 
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def h_prod(A, B):
    m, r, p = A.shape
    r, n, p_ = B.shape
    out = torch.empty(m, n, p, device=device)
    for i in range(p):
        out[..., i] = torch.matmul(A[..., i], B[..., i])
    return out

def slice_wise_svd_transform(D, r):
    device = D.device
    m, n, p = D.shape
    D_approx = torch.zeros_like(D)
    for i in range(p):
        slice_i = D[:, :, i]
        U, S, Vh = torch.linalg.svd(slice_i, full_matrices=False)
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        D_approx[:, :, i] = (U_r * S_r) @ Vh_r
    return D_approx

def slice_wise_qr_transform(D, r):
    device = D.device
    m, n, p = D.shape
    D_approx = np.zeros(D.shape, dtype=np.float32)
    for i in range(p):
        slice_i = D[:, :, i]
        slice_i_np = slice_i.cpu().numpy()
        Q, R, P = qr(slice_i_np, pivoting=True, mode='economic')
        Qk = Q[:, :r]
        Rk = R[:r, :]
        P_mat = np.eye(slice_i_np.shape[1])[:, P]
        D_approx[:, :, i] = Qk @ Rk @ P_mat.T
    return torch.tensor(D_approx, dtype=torch.float32, device=device)


def svd_initialize(D, r):
    device = D.device
    m, n, p = D.shape    
    r = min(r, min(m, n))
    W1 = torch.zeros(m, r, p, device=device)
    H1 = torch.zeros(r, n, p, device=device)
    W2 = torch.zeros(m, r, p, device=device)
    H2 = torch.zeros(r, n, p, device=device)
    for i in range(p):
        M_simple = torch.sqrt(torch.abs(D[:, :, i]))
        M_signed = M_simple * torch.sign(D[:, :, i])
        U1, S1, V1 = torch.linalg.svd(M_simple, full_matrices=False)
        W1[:, :, i] = U1[:, :r] * torch.sqrt(S1[:r])
        H1[:, :, i] = (torch.sqrt(S1[:r]) * V1[:r, :].T).T
        U2, S2, V2 = torch.linalg.svd(M_signed, full_matrices=False)
        W2[:, :, i] = U2[:, :r] * torch.sqrt(S2[:r])
        H2[:, :, i] = (torch.sqrt(S2[:r]) * V2[:r, :].T).T
    return W1, H1, W2, H2





class HadamardNet(nn.Module):
    def __init__(self, X, r):
        super().__init__()
        m, n, rr = X.shape
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
    best_err, best_iter = float('inf'), 0
    D = torch.tensor(D, dtype=torch.float32, device=device)
    model = HadamardNet(D, r)
    opt = optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        X_hat = model()
        loss = torch.norm(D - X_hat, 'fro')
        loss.backward()
        opt.step()

        if loss.item() < best_err:
            best_err = loss.item()
            best_iter = epoch
            wait = 100
        else:
            wait -= 1
        if wait == 0:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.1
            # print(f"Reducing learning rate to {param_group['lr']} at iteration {epoch}")
            wait = 100
        # print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    W1 = torch.Tensor(model.W1.detach().cpu().numpy())
    H1 = torch.Tensor(model.H1.detach().cpu().numpy())
    W2 = torch.Tensor(model.W2.detach().cpu().numpy())
    H2 = torch.Tensor(model.H2.detach().cpu().numpy())
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
        self.D = D if isinstance(D, torch.Tensor) else torch.tensor(D, dtype=torch.float32, device=device)
        self.m, self.n, self.p = self.D.shape
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
            W1_out = self.W1_net(self.W1_input).view(self.m, self.r, self.p).to(self.device)
            H1_out = self.H1_net(self.H1_input).view(self.r, self.n, self.p).to(self.device)
            W2_out = self.W2_net(self.W2_input).view(self.m, self.r, self.p).to(self.device)
            H2_out = self.H2_net(self.H2_input).view(self.r, self.n, self.p).to(self.device)
        else:
            W1_out = self.W1.to(self.device)
            H1_out = self.H1.to(self.device)
            W2_out = self.W2.to(self.device)
            H2_out = self.H2.to(self.device)

        M1 = h_prod(W1_out, H1_out)
        M2 = h_prod(W2_out, H2_out)
        X_pred = M1 * M2

        if base_only:
            return X_pred, W1_out, H1_out, W2_out, H2_out
        else:
            return X_pred

def HadamardDNN_train(D, r, max_epochs=2000, use_factor_nets=True, device='cuda'):
    device = torch.device(device)
    D_tensor = D if isinstance(D, torch.Tensor) else torch.tensor(D, dtype=torch.float32, device=device)
    model = HadamardDNN(D_tensor, r, device=device, use_factor_nets=use_factor_nets).to(device)

    if use_factor_nets:
        params = [model.W1_input, model.H1_input, model.W2_input, model.H2_input] + list(model.W1_net.parameters()) + list(model.H1_net.parameters()) + list(model.W2_net.parameters()) + list(model.H2_net.parameters())
    else:
        params = [model.W1, model.H1, model.W2, model.H2]

    opt = optim.Adam(params, lr=0.1)
    wait = 100
    best_err = float('inf')
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        X_hat = model()
        loss = torch.norm(D_tensor - X_hat, p='fro')
        loss.backward()
        opt.step()

        if loss.item() < best_err:
            best_err = loss.item()
            wait = 500
        else:
            wait -= 1
        if wait == 0:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.1
            # print(f"Reducing learning rate to {param_group['lr']} at iteration {epoch}")
            wait = 100
        # print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    X_base, W1_out, H1_out, W2_out, H2_out = model.forward(base_only=True)
    W1 = torch.Tensor(W1_out.detach().cpu().numpy())
    H1 = torch.Tensor(H1_out.detach().cpu().numpy())
    W2 = torch.Tensor(W2_out.detach().cpu().numpy())
    H2 = torch.Tensor(H2_out.detach().cpu().numpy())
    return W1, H1, W2, H2





# Load image and preprocess
name_ = 'mri'
string_value = '128' + name_
img = Image.open(f'{string_value}.png').convert('RGB')
img_arr = np.asarray(img) / 255.0
m, n, p = img_arr.shape

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
torch.manual_seed(42)

ranks = [2,5,10,20,25]

D = torch.tensor(img_arr, dtype=torch.float32, device=device) + 0.01 * torch.randn(m, n, p, device=device)
norm_tensor = torch.norm(D, 'fro').item()

# # Save original image
# Image.fromarray((D.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(f'{name_}_original.png', dpi=(300,300))

print(f"Image shape: {D.shape}, Frobenius norm: {norm_tensor:.4f}")

approx_results = []

for rank in ranks:
    print(f"\n--- Rank {rank} Approximations ---")
    result = {'rank': rank}

    m,n,p = D.shape

    # Benchmark Tensor SVD
    start = time.time()
    X_svd = slice_wise_svd_transform(D, rank)
    end = time.time()
    err_svd = torch.norm(D - X_svd, 'fro').item()
    result['SVD'] = {
        'recon': X_svd,
        'error': err_svd,
        'rel_error': err_svd / norm_tensor,
        'time': end - start
    }
    print(f"SVD Rank {rank}: Time {end - start:.4f}s, Error {err_svd:.6f}, Rel Error {err_svd / norm_tensor:.6f}")
    # # Save image approximation
    # X_svd_img = (X_svd.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # Image.fromarray(X_svd_img).save(f'{name_}_SVD.png', dpi=(300,300))

    # Calculate PSNR and SSIM
    psnr_svd = psnr(img_arr, X_svd.clamp(0, 1).cpu().numpy(), data_range=1.0)
    ssim_svd = ssim(img_arr, X_svd.clamp(0, 1).cpu().numpy(), channel_axis=2, data_range=1.0, win_size=7)
    result['SVD']['psnr'] = psnr_svd
    result['SVD']['ssim'] = ssim_svd
    print(f"SVD Rank {rank}: PSNR {psnr_svd:.2f}, SSIM {ssim_svd:.4f}")


    # Benchmark Tensor QR
    start = time.time()
    X_qr = slice_wise_qr_transform(D, rank)
    end = time.time()
    err_qr = torch.norm(D - X_qr, 'fro').item()
    result['QR'] = {
        'recon': X_qr,
        'error': err_qr,
        'rel_error': err_qr / norm_tensor,
        'time': end - start
    }
    print(f"QR Rank {rank}: Time {end - start:.4f}s, Error {err_qr:.6f}, Rel Error {err_qr / norm_tensor:.6f}")
    # # Save image approximation
    # X_qr_img = (X_qr.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # Image.fromarray(X_qr_img).save(f'{name_}_QR.png', dpi=(300,300))

    # Calculate PSNR and SSIM
    psnr_qr = psnr(img_arr, X_qr.clamp(0, 1).cpu().numpy(), data_range=1.0)
    ssim_qr = ssim(img_arr, X_qr.clamp(0, 1).cpu().numpy(), channel_axis=2, data_range=1.0, win_size=7)
    result['QR']['psnr'] = psnr_qr
    result['QR']['ssim'] = ssim_qr
    print(f"QR Rank {rank}: PSNR {psnr_qr:.2f}, SSIM {ssim_qr:.4f}")


    eta = 0.01
    maxIter = 40000

    # Benchmark AGD
    A1, B1 = np.zeros((m, rank, p), dtype=np.float32), np.zeros((rank, n, p), dtype=np.float32)
    A2, B2 = np.zeros((m, rank, p), dtype=np.float32), np.zeros((rank, n, p), dtype=np.float32)
    for i in range(p):
        Mat = D[:, :, i].cpu().numpy()
        _,  [_, _], [A1[:, :, i], B1[:, :, i], A2[:, :, i], B2[:, :, i]], [_, _] , _ = sAGD(Mat, rank, eta, maxIter)
    A1, B1 = torch.tensor(A1, dtype=torch.float32, device=device), torch.tensor(B1, dtype=torch.float32, device=device)
    A2, B2 = torch.tensor(A2, dtype=torch.float32, device=device), torch.tensor(B2, dtype=torch.float32, device=device)
    M1 = h_prod(A1, B1)
    M2 = h_prod(A2, B2)
    D_AGD = M1 * M2
    end = time.time()
    err_AGD = torch.norm(D - D_AGD, 'fro').item()
    result['AGD'] = {
        'recon': D_AGD,
        'error': err_AGD,
        'rel_error': err_AGD / norm_tensor,
        'time': end - start
    }
    print(f"AGD Rank {rank}: Time {end - start:.4f}s, Error {err_AGD:.6f}, Rel Error {err_AGD / norm_tensor:.6f}")
    # # Save image approximation
    # D_AGD_img = (D_AGD.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # Image.fromarray(D_AGD_img).save(f'{name_}_AGD.png', dpi=(300,300))
    
    # Calculate PSNR and SSIM
    psnr_agd = psnr(img_arr, D_AGD.clamp(0, 1).cpu().numpy(), data_range=1.0)
    ssim_agd = ssim(img_arr, D_AGD.clamp(0, 1).cpu().numpy(), channel_axis=2, data_range=1.0, win_size=7)
    result['AGD']['psnr'] = psnr_agd
    result['AGD']['ssim'] = ssim_agd
    print(f"AGD Rank {rank}: PSNR {psnr_agd:.2f}, SSIM {ssim_agd:.4f}")


    # Benchmark IGD
    start = time.time()
    A1, B1 = torch.zeros((m, rank, p), dtype=torch.float32, device=device), torch.zeros((rank, n, p), dtype=torch.float32, device=device)
    A2, B2 = torch.zeros((m, rank, p), dtype=torch.float32, device=device), torch.zeros((rank, n, p), dtype=torch.float32, device=device)
    for i in range(p):
        Mat = D[:, :, i].cpu().numpy()
        A1[:, :, i], A2[:, :, i], B1[:, :, i], B2[:, :, i] = igd(Mat, rank, maxIter)
    M1 = h_prod(A1, B1)
    M2 = h_prod(A2, B2)
    D_IGD = M1 * M2
    end = time.time()
    err_IGD = torch.norm(D - D_IGD, 'fro').item()
    result['IGD'] = {
        'recon': D_IGD,
        'error': err_IGD,
        'rel_error': err_IGD / norm_tensor,
        'time': end - start
    }
    print(f"IGD Rank {rank}: Time {end - start:.4f}s, Error {err_IGD:.6f}, Rel Error {err_IGD / norm_tensor:.6f}")
    # # Save image approximation
    # D_IGD_img = (D_IGD.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # Image.fromarray(D_IGD_img).save(f'{name_}_IGD.png', dpi=(300,300))    
    
    # Calculate PSNR and SSIM
    psnr_igd = psnr(img_arr, D_IGD.clamp(0, 1).cpu().numpy(), data_range=1.0)
    ssim_igd = ssim(img_arr, D_IGD.clamp(0, 1).cpu().numpy(), channel_axis=2, data_range=1.0, win_size=7)
    result['IGD']['psnr'] = psnr_igd
    result['IGD']['ssim'] = ssim_igd
    print(f"IGD Rank {rank}: PSNR {psnr_igd:.2f}, SSIM {ssim_igd:.4f}")
    

    # Benchmark BCD
    start = time.time()
    A1, B1 = np.zeros((m, rank, p), dtype=np.float32), np.zeros((rank, n, p), dtype=np.float32)
    A2, B2 = np.zeros((m, rank, p), dtype=np.float32), np.zeros((rank, n, p), dtype=np.float32)
    for i in range(p):
        Mat = D[:, :, i].cpu().numpy()
        A1[:, :, i], B1[:, :, i], A2[:, :, i], B2[:, :, i], _, __ = Hadamard_BCD(Mat, r=rank)
    A1, B1 = torch.tensor(A1, dtype=torch.float32, device=device), torch.tensor(B1, dtype=torch.float32, device=device)
    A2, B2 = torch.tensor(A2, dtype=torch.float32, device=device), torch.tensor(B2, dtype=torch.float32, device=device)
    M1 = h_prod(A1, B1)
    M2 = h_prod(A2, B2)
    D_HBCD = M1 * M2
    end = time.time()
    err_HBCD = torch.norm(D - D_HBCD, 'fro').item()
    result['HBCD'] = {
        'recon': D_HBCD,
        'error': err_HBCD,
        'rel_error': err_HBCD / norm_tensor,
        'time': end - start
    }
    print(f"HBCD Rank {rank}: Time {end - start:.4f}s, Error {err_HBCD:.6f}, Rel Error {err_HBCD / norm_tensor:.6f}")
    # # Save image approximation
    # D_HBCD_img = (D_HBCD.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # Image.fromarray(D_HBCD_img).save(f'{name_}_HBCD.png', dpi=(300,300))  
    
    # Calculate PSNR and SSIM
    psnr_hbcd = psnr(img_arr, D_HBCD.clamp(0, 1).cpu().numpy(), data_range=1.0)
    ssim_hbcd = ssim(img_arr, D_HBCD.clamp(0, 1).cpu().numpy(), channel_axis=2, data_range=1.0, win_size=7)
    result['HBCD']['psnr'] = psnr_hbcd
    result['HBCD']['ssim'] = ssim_hbcd
    print(f"HBCD Rank {rank}: PSNR {psnr_hbcd:.2f}, SSIM {ssim_hbcd:.4f}")    


    # HadamardNet
    start = time.time()
    W1_hm, H1_hm, W2_hm, H2_hm = HadamardNet_train(D.cpu().numpy(), rank)
    M1_hm = h_prod(W1_hm.to(device), H1_hm.to(device))
    M2_hm = h_prod(W2_hm.to(device), H2_hm.to(device))
    X_hm = M1_hm * M2_hm
    end = time.time()
    err_hm = torch.norm(D - X_hm, 'fro').item()
    result['HadamardNet'] = {
        'recon': X_hm,
        'error': err_hm,
        'rel_error': err_hm / norm_tensor,
        'time': end - start
    }
    print(f"HadamardNet Rank {rank}: Time {end - start:.4f}s, Error {err_hm:.6f}, Rel Error {err_hm / norm_tensor:.6f}")
    # # Save image approximation
    # X_hm_img = (X_hm.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # Image.fromarray(X_hm_img).save(f'{name_}_HadNet.png', dpi=(300,300))

    # Calculate PSNR and SSIM
    psnr_hm = psnr(img_arr, X_hm.clamp(0, 1).cpu().numpy(), data_range=1.0)
    ssim_hm = ssim(img_arr, X_hm.clamp(0, 1).cpu().numpy(), channel_axis=2, data_range=1.0, win_size=7)
    result['HadamardNet']['psnr'] = psnr_hm
    result['HadamardNet']['ssim'] = ssim_hm
    print(f"HadamardNet Rank {rank}: PSNR {psnr_hm:.2f}, SSIM {ssim_hm:.4f}")


    # HadamardDNN
    start = time.time()
    W1_hd, H1_hd, W2_hd, H2_hd = HadamardDNN_train(D.cpu().numpy(), rank, use_factor_nets=True, device=device)
    M1_hd = h_prod(W1_hd.to(device), H1_hd.to(device))
    M2_hd = h_prod(W2_hd.to(device), H2_hd.to(device))
    X_hd = M1_hd * M2_hd
    end = time.time()
    err_hd = torch.norm(D - X_hd, 'fro').item()
    result['HadamardDNN'] = {
        'recon': X_hd,
        'error': err_hd,
        'rel_error': err_hd / norm_tensor,
        'time': end - start
    }
    print(f"HadamardDNN Rank {rank}: Time {end - start:.4f}s, Error {err_hd:.6f}, Rel Error {err_hd / norm_tensor:.6f}")
    # # Save image approximation
    # X_hd_img = (X_hd.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # Image.fromarray(X_hd_img).save(f'{name_}_HadDNN.png', dpi=(300,300))

    # Calculate PSNR and SSIM
    psnr_hd = psnr(img_arr, X_hd.clamp(0, 1).cpu().numpy(), data_range=1.0)
    ssim_hd = ssim(img_arr, X_hd.clamp(0, 1).cpu().numpy(), channel_axis=2, data_range=1.0, win_size=7)
    result['HadamardDNN']['psnr'] = psnr_hd
    result['HadamardDNN']['ssim'] = ssim_hd
    print(f"HadamardDNN Rank {rank}: PSNR {psnr_hd:.2f}, SSIM {ssim_hd:.4f}")

