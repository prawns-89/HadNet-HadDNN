import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import qr 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- perf: shared early-stopping (see docs/DECISIONS.md) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from hadamard_ops import EarlyStop


def qr_decomp(X, r):
    if isinstance(X, np.ndarray):
        X_np = X
    else:   
        X_np = X.cpu().numpy()
    Q, R, P = qr(X_np, pivoting=True, mode='economic')
    Qk = Q[:, :r]
    Rk = R[:r, :]
    P_mat = np.eye(X_np.shape[1])[:, P]
    X_approx = Qk @ Rk @ P_mat.T
    return X_approx

def igd(D, r, max_iter):
    D_torch = torch.tensor(D, dtype=torch.float32, device=device)
    m, n = D_torch.shape
    U1 = 0.1 * torch.rand(m, r, device=device)
    V1 = 0.1 * torch.rand(r, n, device=device)
    U2 = 0.1 * torch.rand(m, r, device=device)
    V2 = 0.1 * torch.rand(r, n, device=device)

    eta = 0.01
    lambd = 1e-6
    patience = 50
    best_err = float('inf')
    best_iter = 0
    t = 0

    D_fro = torch.norm(D_torch, 'fro')
    while t < max_iter:
        t += 1
        D_est = (U1 @ V1) * (U2 @ V2)
        diff = D_est - D_torch
        curr_err = torch.norm(diff, 'fro') / (D_fro + 1e-12)

        if curr_err < best_err:
            best_err = curr_err
            best_iter = t
        elif t - best_iter > patience:
            break

        diff_U2V2 = diff * (U2 @ V2)
        diff_U1V1 = diff * (U1 @ V1)

        grad_U1 = diff_U2V2 @ V1.T + lambd * U1
        grad_V1 = U1.T @ diff_U2V2 + lambd * V1
        grad_U2 = diff_U1V1 @ V2.T + lambd * U2
        grad_V2 = U2.T @ diff_U1V1 + lambd * V2

        U1 -= eta * grad_U1
        V1 -= eta * grad_V1
        U2 -= eta * grad_U2
        V2 -= eta * grad_V2

        if t % 100 == 0:
            eta *= 0.95

    return U1, U2, V1, V2


def svd_init(X, r, randomized=True, niter=24):
    """SVD-based initialization.

    randomized=True computes only the leading r directions instead of all
    min(m,n), dropping the cost from O(mn*min(m,n)) to O(mn*r). niter=24
    follows Shard's stability setting for torch.svd_lowrank.
    """
    M = np.sqrt(np.abs(X))
    M_signed = M * np.sign(X)
    
    # First factor from M
    if randomized and r < min(M.shape) // 2:
        from sklearn.utils.extmath import randomized_svd as _rsvd
        U1, S1, V1 = _rsvd(M, n_components=r, n_iter=niter, random_state=42)
    else:
        U1, S1, V1 = np.linalg.svd(M, full_matrices=False)
    W1 = U1[:, :r] * np.sqrt(S1[:r])
    H1 = (np.sqrt(S1[:r]) * V1[:r, :].T).T
    
    # Second factor from M_signed  
    if randomized and r < min(M_signed.shape) // 2:
        from sklearn.utils.extmath import randomized_svd as _rsvd
        U2, S2, V2 = _rsvd(M_signed, n_components=r, n_iter=niter, random_state=42)
    else:
        U2, S2, V2 = np.linalg.svd(M_signed, full_matrices=False)
    W2 = U2[:, :r] * np.sqrt(S2[:r])
    H2 = (np.sqrt(S2[:r]) * V2[:r, :].T).T
    
    return W1, H1, W2, H2

class SimpleHadamardNet(nn.Module):
    def __init__(self, X, r):
        super().__init__()
        m, n = X.shape
        W1, H1, W2, H2 = svd_init(X, r)
        self.W1 = nn.Parameter(torch.tensor(W1, dtype=torch.float32))
        self.H1 = nn.Parameter(torch.tensor(H1, dtype=torch.float32))
        self.W2 = nn.Parameter(torch.tensor(W2, dtype=torch.float32))
        self.H2 = nn.Parameter(torch.tensor(H2, dtype=torch.float32))
        
    def forward(self):
        M1 = self.W1 @ self.H1
        M2 = self.W2 @ self.H2
        X_pred = M1 * M2
        return X_pred

def train_Net(D, r, max_epochs=10000):
    """Simple training function"""
    wait = 100
    best_err, best_iter = float('inf'), 0
    X = torch.tensor(D, dtype=torch.float32)
    model = SimpleHadamardNet(D, r)
    opt = optim.Adam([model.W1, model.H1, model.W2, model.H2], lr=0.01)
    stopper = EarlyStop(patience=50, min_rel=1e-3)
    start = time.time()
    
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        X_hat = model()
        loss = torch.norm(X - X_hat) ** 2
        loss.backward()
        opt.step()
        if stopper.step(loss.item()):
            break

        if loss.item() < best_err:
            best_err = loss.item()
            best_iter = epoch
            wait = 100
        else:
            wait -= 1
        if wait == 0:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.5
            # print(f"Reducing learning rate to {param_group['lr']} at iteration {epoch}")
            wait = 100
        # print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # # Check progress
        # if epoch % 200 == 0:
        #     error = torch.norm(X - X_hat).item()
        #     print(f"Epoch {epoch:4d} | Error: {error:.6f}")
    
    training_time = time.time() - start    
    W1 = model.W1.detach().cpu().numpy()
    H1 = model.H1.detach().cpu().numpy()
    W2 = model.W2.detach().cpu().numpy()
    H2 = model.H2.detach().cpu().numpy()
    return training_time, W1, H1, W2, H2


class FactorNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
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
    def __init__(self, X, r, device=None):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        m, n = X.shape
        self.m, self.n, self.r = m, n, r
        self.device = device

        W1_init, H1_init, W2_init, H2_init = svd_init(X, r)
        
        self.W1_input = nn.Parameter(W1_init.flatten().unsqueeze(0))
        self.H1_input = nn.Parameter(H1_init.flatten().unsqueeze(0))
        self.W2_input = nn.Parameter(W2_init.flatten().unsqueeze(0))
        self.H2_input = nn.Parameter(H2_init.flatten().unsqueeze(0))

        self.W1_net = FactorNet(self.m * self.r * self.p, self.m * self.r * self.p).to(device)
        self.H1_net = FactorNet(self.r * self.n * self.p, self.r * self.n * self.p).to(device)
        self.W2_net = FactorNet(self.m * self.r * self.p, self.m * self.r * self.p).to(device)
        self.H2_net = FactorNet(self.r * self.n * self.p, self.r * self.n * self.p).to(device)


    def forward(self, X_base_only=False):
        W1 = self.W1_net(self.W1_input).view(self.m, self.r)
        H1 = self.H1_net(self.H1_input).view(self.r, self.n)
        W2 = self.W2_net(self.W2_input).view(self.m, self.r)
        H2 = self.H2_net(self.H2_input).view(self.r, self.n)

        M1 = W1 @ H1
        M2 = W2 @ H2
        X_base = M1 * M2

        if X_base_only:
            return X_base, W1, H1, W2, H2

        return X_base

def train_DNN(D, r, max_epochs=10000, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(D, dtype=torch.float32, device=device)
    model = HadamardDNN(D, r, device=device).to(device)

    params_phase1 = [model.W1_input, model.H1_input, model.W2_input, model.H2_input] + \
                        list(model.W1_net.parameters()) + \
                        list(model.H1_net.parameters()) + \
                        list(model.W2_net.parameters()) + \
                        list(model.H2_net.parameters())

    opt = optim.Adam(params_phase1, lr=0.01)

    stopper = EarlyStop(patience=50, min_rel=1e-3)
    wait = 100
    best_err = float('inf')
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        X_hat = model()
        loss = torch.norm(X - X_hat) ** 2
        loss.backward()
        opt.step()
        if stopper.step(loss.item()):
            break
        
        if loss.item() < best_err:
            best_err = loss.item()
            wait = 100
        else:
            wait -= 1
        if wait == 0:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.5
            # print(f"Reducing learning rate to {param_group['lr']} at iteration {epoch}")
            wait = 100
        # print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        # if epoch % 500 == 0:
        #     error = torch.norm(X - model()).item()
        #     print(f"Epoch {epoch:4d} | Error: {error:.6f}")

    training_time = time.time() - start
    final_error = torch.norm(X - model()).item()

    X_base, W1, H1, W2, H2 = model.forward(X_base_only=True)
    W1_np = W1.detach().cpu().numpy()
    H1_np = H1.detach().cpu().numpy()
    W2_np = W2.detach().cpu().numpy()
    H2_np = H2.detach().cpu().numpy()
    return training_time, (W1_np, H1_np, W2_np, H2_np)

