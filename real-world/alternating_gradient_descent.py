import numpy as np 
from numba import jit
import numba as nb 
from math import isnan

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@jit(nopython=True)
def makerealdata(n,c,m,noise=0.0, zeroone=False):
    W = np.random.random_sample(size=(n,c))
    H = np.random.random_sample(size=(c,m))
    D = W@H
    return W,H,D

@jit(nopython=True)
def compute_svd(D):
    u,s,v = np.linalg.svd(D) 
    return u,s,v


@jit(nopython=True)
def scaled_alternating_gradient_descent_hadDec(D, r, eta = 0.01, T = 225000, normalize=False, normalize_gradients=True, monitoring_interval = 1000): 
    
    ''' scaled gradient descent with spectral initialization for Hadamard decomposition with two rank-r factors 
    
    Params:
    D: input matrix (numpy array) 
    r: rank of both Hadamard factors (integer) 
    eta: learning rate (float)
    T: maximum number of iterations/epochs (integer)
    
    
    Returns: 
    D_estimate: nxm reconstruction of D 
    D_1_estimate, D_2_estimate: reconstruction of Hadamard factors of D 
    A_1, B_1, A_2, B_2: reconstruction of factors of each Hadamard factor of D 
    all_diffs, all_diffs2: L1 and L2 approximation error 
    terminated: flag indicating whether the algorithm has terminated without convergence issues 
    ''' 


    # intialization     
    all_diffs = [] # L1 norm loss
    all_diffs2 = [] # L2 norm loss    
    n,m = D.shape     
    W1, H1, D1_est =  makerealdata(n,r,m, noise=0.0) 
    W2, H2, D2_est =  makerealdata(n,r,m, noise=0.0)
    
    
    #normalize in 0/1 
    if normalize:
        D1_est -= D1_est.min()
        D1_est /= D1_est.max()
        D2_est -= D2_est.min()
        D2_est /= D2_est.max()
        
        
    # spectral initialization of A_1, B_1, A_2 and B_2 
    u, d, v = np.linalg.svd(D1_est)
  
    #A_1 = u[:, :r] @ np.sqrt(np.diag(d[:r]))
    A_1 = np.ascontiguousarray(u[:, :r]) @ np.sqrt(np.diag(d[:r]))
    #B_1 =  np.sqrt(np.diag(d[:r])) @ v[:r, :]  
    B_1 =  np.sqrt(np.diag(d[:r])) @ np.ascontiguousarray(v[:r, :])
        
    u, d, v = np.linalg.svd(D2_est)
    
    # A_2 = u[:, :r] @ np.sqrt(np.diag(d[:r]))
    A_2 = np.ascontiguousarray(u[:, :r]) @ np.sqrt(np.diag(d[:r]))
    # B_2 =  np.sqrt(np.diag(d[:r])) @ v[:r, :]  
    B_2 =  np.sqrt(np.diag(d[:r])) @ np.ascontiguousarray(v[:r, :])
    D_2_estimate = (A_2 @ B_2)
               
    # convergence monitoring            
    flag2 = 10e20
    terminated = True     
    
        
     # starting iterations      
    for t in range(T): 
        # difference term in the gradient 
        diff = ( (A_1 @ B_1) * (D_2_estimate) - D )   
        
        # first Hadamard factor update 
        gradient = (diff * D_2_estimate) @ B_1.T 
        if normalize_gradients:
            gradient = gradient / np.linalg.norm(gradient)
        A_1 = A_1 - eta * gradient                    
        
        
        gradient = (( (diff.T  * D_2_estimate.T ) @ A_1)).T
        if normalize_gradients:
            gradient = gradient / np.linalg.norm(gradient)
        B_1 = B_1 - eta * gradient
        
        D_1_estimate = (A_1 @ B_1) 
        
        #  second Hadamard factor update 
        gradient = ((diff * (D_1_estimate)) @ B_2.T)
        if normalize_gradients:
            gradient = gradient / np.linalg.norm(gradient)
        A_2 = A_2 - eta * gradient
        
        
        gradient = (((diff.T * (D_1_estimate).T) @ A_2)).T
        if normalize_gradients:
            gradient = gradient / np.linalg.norm(gradient)
        B_2 = B_2 - eta * gradient
        
        D_2_estimate = (A_2 @ B_2)
    
        
        # monitor loss 
        if t % monitoring_interval==0: 
            previous_flag2 = flag2 
            flag = np.sum(np.abs(diff))
            flag2 = np.sum(diff**2)
            all_diffs.append(flag)
            all_diffs2.append(flag2)
            if isnan(flag2): 
                # print("nan detected - terminating.") 
                terminated = False 
                break
            if previous_flag2 < flag2: 
                # print("loss is increasing - halving learning rate.") 
                eta = eta * 0.5 
                
                
    
    # final estimate 
    D_estimate = D_1_estimate * D_2_estimate
    
    #final loss 
    flag2 = np.sum(diff**2)
    all_diffs2.append(flag2)
    return D_estimate,  [D_1_estimate, D_2_estimate], [A_1, B_1, A_2, B_2], [all_diffs, all_diffs2], terminated






@jit(nopython=True)
def scaled_alternating_gradient_descent_MixedhadDec(D, r, u1, v1, eta = 0.01, T = 225000, normalize_gradients=True, monitoring_interval = 1000): 
    
    ''' scaled gradient descent with spectral initialization for mixed Hadamard decomposition 
    
    Params:
    D: input matrix (numpy array)
    r: rank of both Hadamard factors (integer)
    u1: left singular vectors  (numpy array)
    v2: right singular vectors  (numpy array)
    eta: learning rate   (float)
    T: maximum number of iterations/epochs (integer)
    
    
    Returns: 
    D_estimate: nxm reconstruction of D 
    D_1_estimate, D_2_estimate: reconstruction of Hadamard factors of D 
    A_1, B_1, A_2, B_2: reconstruction of factors of each Hadamard factor of D 
    all_diffs, all_diffs2: L1 and L2 approximation error 
    terminated: flag indicating whether the algorithm has terminated without convergence issues 
    ''' 
    


    # initialize additive component 
    v1ones = np.ones_like(u1) 
    v2ones = np.ones_like(v1)
    r_add = u1.shape[1]
    all_diffs = [] # L1 norm loss
    all_diffs2 = [] # L2 norm loss
        
    # initialize Hadamard factors 
    n,m = D.shape     
    u, d, v = compute_svd(D) 
    A_1 = u[:, :r] @ np.sqrt(np.diag(d[:r]))
    B_1 =  np.sqrt(np.diag(d[:r])) @ v[:r, :]  
    
    D2_est = np.ones_like(D)
    u, d, v =  compute_svd(D2_est)
    A_2 = u[:, :r] @ np.sqrt(np.diag(d[:r]))
    B_2 =  np.sqrt(np.diag(d[:r])) @ v[:r, :]  
    D_2_estimate = (A_2 @ B_2)
    
    A_2[:,:r_add] = v1ones 
    B_2[:r_add,:] = v2ones
    
    # convergence monitoring 
    flag2 = 10e20
    terminated = True 
    
    
    for t in range(T): 
            
        # difference term in the gradient 
        diff = ( (A_1 @ B_1) * D_2_estimate - D )   
        
        # first Hadamard factor update 
        gradient = (((diff * D_2_estimate) @ B_1.T @ np.linalg.inv( B_1 @ B_1.T ) )  )[:,r_add:]  
        if normalize_gradients: 
            gradient = gradient / np.linalg.norm(gradient)
        A_1[:,r_add:] = A_1[:,r_add:] - eta  * gradient                    
        
        
        gradient = ((( (diff.T  * D_2_estimate.T ) @ A_1  @ np.linalg.inv( A_1.T @ A_1 ) )).T  )[r_add:,:]   
        if normalize_gradients: 
            gradient = gradient / np.linalg.norm(gradient)
        B_1[r_add:,:] = B_1[r_add:,:] - eta  * gradient
        
        
        D_1_estimate = (A_1 @ B_1) 
        
        #  second Hadamard factor update 
        gradient =  (((diff * D_1_estimate) @ B_2.T @ np.linalg.inv( B_2 @ B_2.T )  )   )[:,r_add:]   
        if normalize_gradients: 
            gradient = gradient / np.linalg.norm(gradient)
        A_2[:,r_add:] = A_2[:,r_add:] - eta * gradient
        
        
        gradient =  (((diff.T * D_1_estimate.T) @ A_2 @ np.linalg.inv( A_2.T @ A_2 ) ).T )[r_add:,:]
        if normalize_gradients: 
            gradient = gradient / np.linalg.norm(gradient)
        B_2[r_add:,:]  = B_2[r_add:,:] - eta * gradient
    
        D_2_estimate = (A_2 @ B_2)
        
        # monitor loss 
        if t % monitoring_interval==0: 
            previous_flag2 = flag2 
            flag = np.sum(np.abs(diff))
            flag2 = np.sum(diff**2)
            all_diffs.append(flag)
            all_diffs2.append(flag2)
            if isnan(flag2): 
                # print("nan detected - terminating.") 
                terminated = False 
                break
            if previous_flag2 < flag2: 
                # print("loss is increasing - halving learning rate") 
                eta = eta * 0.5 
                
    # final estimate 
    D_estimate = D_1_estimate * D_2_estimate
    
    # final loss 
    flag2 = np.sum(diff**2)
    all_diffs2.append(flag2)
    return D_estimate,  [D_1_estimate, D_2_estimate], [A_1, B_1, A_2, B_2], [all_diffs, all_diffs2], terminated

