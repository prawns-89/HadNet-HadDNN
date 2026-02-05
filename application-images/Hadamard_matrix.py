import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        curr_err = torch.norm(diff, 'fro') / D_fro

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
