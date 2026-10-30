import time
import numpy as np
import torch
from Hadamard_matrix import train_Net, train_DNN, qr_decomp, igd
from PIL import Image
from Hadamard_BCD import Hadamard_BCD
from alternating_gradient_descent import scaled_alternating_gradient_descent_hadDec as sAGD
from stiefel_hadamard import stiefel_hadDec

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import networkx as nx

def rnd_initialization(X, m, n, r):
    A = np.random.randn(m,r)
    B = np.random.randn(r,n)
    C = np.random.randn(m,r)
    D = np.random.randn(r,n)

    return A, B, C, D

def load_dataset(dataset_name, m=100, n=100, h=35):
    if dataset_name == "spectrometer":
        with open("datasets/lrs.data") as inp:
            spectro = inp.read().split('(')
        D = []
        spectro.pop(0)
        for line in spectro:
            line = line.split()[2:]
            line = [float(i.rstrip(')')) for i in line]
            D.append(line)
        D = np.array(D)
    
    elif dataset_name == "miserables":
        G = nx.read_gml('datasets/lesmis.gml')
        D = nx.to_numpy_array(G)

    else:
        raise ValueError(f"Dataset '{dataset_name}' not available ")
    
    return D

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(42)
    torch.manual_seed(42)

    dataset_name = "miserables"
    D = load_dataset(dataset_name, m=256, n=256, h=256)
    m, n = D.shape
    
    norm_tensor = np.linalg.norm(D, 'fro')

    print(f"Dataset: {dataset_name}, Shape: {D.shape}")
    print(f"Frobenius norm of D: {norm_tensor}")

    p = 1

    ranks = [2,5,10,25,40,50]
    for r in ranks:
        for i in range(p):
            eta = 0.01
            maxIter = 40000
            m,n = D.shape[0], D.shape[1]
            norm_D = np.linalg.norm(D, 'fro')

            print(f"Matrix size: {D.shape}")
            print(f"Rank: {r}")

            # Benchmark SVD
            start = time.time()
            U, S, Vh = np.linalg.svd(D, full_matrices=False)
            U_r = U[:, :r]
            S_r = np.diag(S[:r])
            V_r = Vh[:r, :]
            svd_recon = U_r @ S_r @ V_r
            svderr = np.linalg.norm(D - svd_recon, 'fro')
            print(f"\nReconstruction error SVD: {svderr:.6f}")
            print(f"Relative error SVD: {svderr / norm_D:.6f}")
            print(f"SVD time: {time.time() - start:.5f} seconds")
            # recon_img = np.clip(svd_recon * 255, 0, 255).astype(np.uint8)
            # recon_pil = Image.fromarray(recon_img)
            # recon_pil.save(f'SVD.png')

            # Benchmark QR
            start = time.time()
            D_qr = qr_decomp(D, r)
            qr_error = np.linalg.norm(D - D_qr, 'fro')
            print(f"\nReconstruction error RRQR: {qr_error:.6f}")
            print(f"Relative error RRQR: {qr_error / norm_D:.6f}")
            print(f"RRQR time: {time.time() - start:.5f} seconds")
            # recon_img = np.clip(D_qr * 255, 0, 255).astype(np.uint8)
            # recon_pil = Image.fromarray(recon_img)
            # recon_pil.save(f'RRQR.png')

            # Benchmark AGD
            start = time.time()
            _,  [_, _], [A_1, B_1, A_2, B_2], [_, _] , _ =  sAGD(D, r, eta, maxIter)
            D_sAGD = (A_1 @ B_1) * (A_2 @ B_2)
            sAGD_error = np.linalg.norm(D - D_sAGD, 'fro')
            print(f"\nReconstruction error AGD: {sAGD_error:.6f}")
            print(f"Relative error AGD: {sAGD_error / norm_D:.6f}")
            print(f"AGD time: {time.time() - start:.5f} seconds")
            # recon_img = np.clip(D_sAGD * 255, 0, 255).astype(np.uint8)
            # recon_pil = Image.fromarray(recon_img)
            # recon_pil.save(f'AGD.png')

            # Benchmark BCD
            start = time.time()
            A1, B1, A2, B2, _, __ = Hadamard_BCD(D, r=r)
            D_HBCD = (A1 @ B1) * (A2 @ B2)
            HBCD_error = np.linalg.norm(D - D_HBCD, 'fro')
            print(f"\nReconstruction error BCD: {HBCD_error:.6f}")
            print(f"Relative error BCD: {HBCD_error / norm_D:.6f}")
            print(f"BCD time: {time.time() - start:.5f} seconds")
            # recon_img = np.clip(D_HBCD * 255, 0, 255).astype(np.uint8)
            # recon_pil = Image.fromarray(recon_img)
            # recon_pil.save(f'HBCD.png')

            # Benchmark Stiefel (orthonormality kept on every step, heavy-ball momentum)
            start = time.time()
            W1_st, H1_st, W2_st, H2_st, _, _ = stiefel_hadDec(D, r)
            D_st = (W1_st @ H1_st) * (W2_st @ H2_st)
            st_error = np.linalg.norm(D - D_st, 'fro')
            print(f"\nReconstruction error Stiefel: {st_error:.6f}")
            print(f"Relative error Stiefel: {st_error / norm_D:.6f}")
            print(f"Stiefel time: {time.time() - start:.5f} seconds")

            # Benchmark IGD
            start = time.time()
            U1, U2, V1, V2 = igd(D, r, maxIter)
            D_IGD = (U1 @ V1) * (U2 @ V2)
            D_IGD = D_IGD.detach().cpu().numpy()
            error = np.linalg.norm(D - D_IGD, 'fro')
            print(f"\nReconstruction error IGD: {error:.6f}")
            print(f"Relative error IGD: {error / norm_D:.6f}")
            print(f"IGD time: {time.time() - start:.5f} seconds")
            # recon_img = np.clip(D_IGD * 255, 0, 255).astype(np.uint8)
            # recon_pil = Image.fromarray(recon_img)
            # recon_pil.save(f'IGD.png')

            # HadamardNet
            training_time, W1_dnn, H1_dnn, W2_dnn, H2_dnn = train_Net(D, r)
            D_dnn = (W1_dnn @ H1_dnn) * (W2_dnn @ H2_dnn)
            dnn_error = np.linalg.norm(D - D_dnn, 'fro')
            print(f"\nReconstruction error HadamardNet: {dnn_error:.6f}")
            print(f"Relative error HadamardNet: {dnn_error / norm_D:.6f}")
            print(f"HadamardNet time:  {training_time:.5f} seconds")
            # recon_img = np.clip(D_dnn * 255, 0, 255).astype(np.uint8)
            # recon_pil = Image.fromarray(recon_img)
            # recon_pil.save(f'HadNet.png')

            # HadamardDNN
            train_time, (W1_dnn, H1_dnn, W2_dnn, H2_dnn) = train_DNN(D, r)
            D_dnn = (W1_dnn @ H1_dnn) * (W2_dnn @ H2_dnn)
            dnn_error = np.linalg.norm(D - D_dnn, 'fro')
            print(f"\nReconstruction error HadamardDNN: {dnn_error:.6f}")
            print(f"Relative error HadamardDNN: {dnn_error / norm_D:.6f}")
            print(f"HadamardDNN time:  {train_time:.5f} seconds")
            print("\n")
            # recon_img = np.clip(D_dnn * 255, 0, 255).astype(np.uint8)
            # recon_pil = Image.fromarray(recon_img)
            # recon_pil.save(f'HadDNN.png')

    print("True Frobenius Norm: ", norm_tensor)
    print("\n")

