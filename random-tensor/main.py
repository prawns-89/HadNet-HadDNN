import time
import torch
import numpy as np
import itertools
from scipy.linalg import qr 
import torch.nn as nn
import torch.optim as optim

from Hadamard_BCD import Hadamard_BCD
from alternating_gradient_descent import scaled_alternating_gradient_descent_hadDec as sAGD
from Hadamard_matrix import igd

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

def svd_initialize(D, r):
    device = D.device
    shape = D.shape
    m, n = shape[0], shape[1]
    p = shape[2:]
    out_W = (m,r) + p
    out_H = (r,n) + p
    W1 = torch.zeros(out_W, device=device)
    H1 = torch.zeros(out_H, device=device)
    W2 = torch.zeros(out_W, device=device)
    H2 = torch.zeros(out_H, device=device)

    for i in itertools.product(*[range(s) for s in p]):
        mat = D[(...,)+i]
        mat_simple = torch.sqrt(torch.abs(mat))
        mat_signed = mat_simple * torch.sign(mat)

        U1, S1, V1 = torch.linalg.svd(mat_simple, full_matrices=False)
        W1[(...,)+i] = U1[:, :r] * torch.sqrt(S1[:r])
        H1[(...,)+i] = (torch.sqrt(S1[:r]) * V1[:r, :].T).T

        U2, S2, V2 = torch.linalg.svd(mat_signed, full_matrices=False)
        W2[(...,)+i] = U2[:, :r] * torch.sqrt(S2[:r])
        H2[(...,)+i] = (torch.sqrt(S2[:r]) * V2[:r, :].T).T
    return W1, H1, W2, H2

def h_prod(A, B):
    shapeA = A.shape
    shapeB = B.shape
    m, _, p = shapeA[0], shapeA[1], shapeA[2:]
    _, n, p = shapeB[0], shapeB[1], shapeB[2:]
    outdim = (m,n) + p
    C = torch.empty(outdim, dtype=A.dtype, device=A.device)
    for i in itertools.product(*[range(s) for s in p]):
        C[(...,)+i] = torch.matmul(A[(...,)+i], B[(...,)+i])
    return C

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
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        X_hat = model()
        loss = torch.norm(D - X_hat, 'fro')
        loss.backward()
        opt.step()
        hist.append(loss.item())

        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

        if loss.item() < best_err:
            best_err = loss.item()
            best_iter = epoch
            wait = 100
        else:
            wait -= 1
        if wait == 0 or updatecount == 100:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.1
            # print(f"Reducing learning rate to {param_group['lr']} at iteration {epoch}")
            wait = 100
            updatecount = 0
        # print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

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

def HadamardDNN_train(D, r, max_epochs=2000, use_factor_nets=True, device='cuda'):
    device = torch.device(device)
    D = D.to(device)
    model = HadamardDNN(D, r, device=device, use_factor_nets=use_factor_nets).to(device)

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
        loss = torch.norm(D - X_hat, p='fro')
        loss.backward()
        opt.step()

        # if epoch % 100 == 0:
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
            wait = 100
        # print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    X_base, W1_out, H1_out, W2_out, H2_out = model.forward(base_only=True)

    W1 = torch.tensor(W1_out.detach().cpu().numpy())
    H1 = torch.tensor(H1_out.detach().cpu().numpy())
    W2 = torch.tensor(W2_out.detach().cpu().numpy())
    H2 = torch.tensor(H2_out.detach().cpu().numpy())
    return W1, H1, W2, H2





if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(42)
    torch.manual_seed(42)

    ranks = [2,4,6,8,10]
    for r in ranks:
        A = torch.rand(10, 100, 10, 10) + 0.1*torch.randn(10, 100, 10, 10)
        rank = r
        norm_tensor = torch.norm(A, 'fro').item()
        print(f"nD-Tensor size: {A.shape}")
        print(f"Rank: {rank}")
        shape = A.shape
        n1, n2 = shape[0], shape[1]
        remShape = shape[2:]
        out_shape = (A.shape[0], A.shape[1]) + remShape


        # Benchmark nD-Tensor SVD
        svdstart = time.time()
        Asvd = torch.empty(out_shape, dtype=A.dtype, device=A.device)
        for i in itertools.product(*[range(s) for s in remShape]):
            matA = A[(...,)+i]
            Asvd[(...,)+i] = svd_transform(matA, r=rank)
        err_svd = torch.norm(A - Asvd, 'fro').item()
        print(f"\nReconstruction error nD-Tensor SVD: {err_svd:.6f}")
        print(f"Relative error nD-Tensor SVD: {err_svd / norm_tensor:.6f}")
        print(f"nD-Tensor SVD time: {time.time() - svdstart:.5f} seconds")

        # Benchmark nD-Tensor QR
        qrstart = time.time()
        Aqr = torch.empty(out_shape, dtype=A.dtype, device=A.device)
        for i in itertools.product(*[range(s) for s in remShape]):
            matA = A[(...,)+i]
            Aqr[(...,)+i] = qr_transform(matA, r=rank)
        err_qr = torch.norm(A - Aqr, 'fro').item()
        print(f"\nReconstruction error nD-Tensor RRQR: {err_qr:.6f}")
        print(f"Relative error nD-Tensor RRQR: {err_qr / norm_tensor:.6f}")
        print(f"nD-Tensor RRQR time: {time.time() - qrstart:.5f} seconds")


        eta = 0.01
        maxIter = 40000

        # Benchmark TAGD
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
        end = time.time()
        err_AGD = torch.norm(A - A_AGD, 'fro').item()
        print(f"\nReconstruction error Tensor AGD: {err_AGD:.6f}")
        print(f"Relative error Tensor AGD: {err_AGD / norm_tensor:.6f}")
        print(f"Tensor AGD time: {end - start:.5f} seconds")

        # Benchmark TIGD
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
        end = time.time()
        err_IGD = torch.norm(A - A_IGD, 'fro').item()
        print(f"\nReconstruction error Tensor IGD: {err_IGD:.6f}")
        print(f"Relative error Tensor IGD: {err_IGD / norm_tensor:.6f}")
        print(f"Tensor IGD time: {end - start:.5f} seconds")

        # Benchmark TBCD
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
        end = time.time()
        err_BCD = torch.norm(A - A_BCD, 'fro').item()
        print(f"\nReconstruction error Tensor BCD: {err_BCD:.6f}")
        print(f"Relative error Tensor BCD: {err_BCD / norm_tensor:.6f}")
        print(f"Tensor BCD time: {end - start:.5f} seconds")


        # HadamardNet
        start = time.time()
        W1, H1, W2, H2 = HadamardNet_train(A, rank)
        # print("Sizes: ", W1.shape, H1.shape, W2.shape, H2.shape)
        M1 = h_prod(torch.tensor(W1), torch.tensor(H1))
        M2 = h_prod(torch.tensor(W2), torch.tensor(H2))
        X_pred = M1 * M2
        err = torch.norm(A - X_pred, 'fro').item()
        print(f"\nReconstruction error nD-Tensor HadamardNet: {err:.6f}")
        print(f"Relative error nD-Tensor HadamardNet: {err / norm_tensor:.6f}")
        print(f"nD-Tensor HadamardNet time: {time.time() - start:.5f} seconds")
        
        # HadamardDNN
        start = time.time()
        W1, H1, W2, H2 = HadamardDNN_train(A, rank)
        M1 = h_prod(torch.tensor(W1), torch.tensor(H1))
        M2 = h_prod(torch.tensor(W2), torch.tensor(H2))
        X_pred = M1 * M2
        err = torch.norm(A - X_pred, 'fro').item()
        print(f"\nReconstruction error nD-Tensor HadamardDNN: {err:.6f}")
        print(f"Relative error nD-Tensor HadamardDNN: {err / norm_tensor:.6f}")
        print(f"nD-Tensor HadamardDNN time: {time.time() - start:.5f} seconds")
        
        print("\n")

    print("True Frobenius Norm: ", norm_tensor)
    print("\n")
