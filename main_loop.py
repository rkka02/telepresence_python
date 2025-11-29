import time
import numpy as np
from numpy.fft import ifftshift
import torch

def main_loop(psi, diffuser_E, sample_mask, spk_window, shiftvec):
    # Configurations
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    gpu_ind = None
    optimization_method = 'FISTA'
    
    Y_domain = spk_window
    iterMax=100
    eta=0.5e-2
    step_size=1
    X_iter_0='random'
    use_SSMinit=True

    diffuser_E = ifftshift(diffuser_E)
    sample_mask = np.roll(sample_mask, shift=(shiftvec[0], shiftvec[1]), axis=(0,1))
    
    X_iter = sample_mask
    Y_iter = T(X_iter, diffuser_E, Y_domain)
    X_iter = T_dagger(Y_iter, diffuser_E, sample_mask)

    # Compute maximum absolute value of Xiter in spatial dimensions
    # (summing over the first two dims; assumes psi is at least 2D)
    # Constant for scaling the step-size based on psi energy
    phase_conj_delta = np.max(np.abs(X_iter), axis=(0, 1))
    psi_energy = np.sum(psi**2, axis=(0,1))
    eta_constant = np.sqrt(psi_energy / phase_conj_delta)
    
    if optimization_method in ['GD', 'FISTA']:
        # Scale the step size elementwise
        step_size = step_size / phase_conj_delta
    elif optimization_method in ['Adam', 'Nadam', 'AMSGrad']:
        eta = eta * eta_constant  # adjust eta with etaConstant
    else:
        raise ValueError("Invalid opt_method")  # should not happen
    
    # Make stack
    if psi.shape[2] is not None:
        stack_num = psi.shape[2]
        diffuser_E = np.stack([diffuser_E] * stack_num, axis=2)
        sample_mask = np.stack([sample_mask] * stack_num, axis=2)
        Y_domain = np.stack([Y_domain] * stack_num, axis=2)
        
    ######
    # Determine the padded size from psi
    pad_size = psi.shape

    # If X_iter_0 is specified as a string option, build it accordingly.
    if isinstance(X_iter_0, str):
        if X_iter_0 == 'zero':
            X_iter_0 = np.zeros(pad_size, dtype=diffuser_E.dtype)
        elif X_iter_0 in ['random', 'SSM']:
            # Create a random complex guess within the sampling mask
            X_iter_0 = (np.random.randn(*diffuser_E.shape) +
                            1j * np.random.randn(*diffuser_E.shape)) * sample_mask

            # Replicate X_iter along the third dimension if necessary.
            if X_iter.ndim < len(pad_size):
                X_iter_0 = np.tile(X_iter[..., np.newaxis], (1, 1, pad_size[2]))
            else:
                X_iter_0 = X_iter

            # Ensure the type matches diffuser_E.
            X_iter_0 = X_iter_0.astype(diffuser_E.dtype)

            # Compute normalization: ensure that the masked energy of the random guess is unit scaled.
            norm_denom = np.sqrt(np.sum(np.abs(sample_mask * X_iter_0)**2, axis=(0, 1)) /
                                        np.sum(sample_mask))
            if norm_denom.ndim == 1:
                norm_denom = norm_denom[None, None, :]  # reshape for broadcasting

            # Normalize and scale by eta_constant.
            stack_X_iter = np.stack([X_iter]*sample_mask.shape[2], axis=2)
            X_iter_0 = (sample_mask * stack_X_iter) / norm_denom * eta_constant / 2
        else:
            raise ValueError("Invalid X_iter_0")
    # If X_iter_0 is already provided as a numpy array (a matrix), leave it unchanged.
    elif isinstance(X_iter_0, np.ndarray):
        X_iter_0 = X_iter_0
    else:
        raise ValueError("Invalid X_iter_0")
    
    #######
    X_iter = X_iter_0 / np.sqrt(np.sum(np.abs(X_iter_0)**2, axis=(0,1)))
    dIyy = np.abs(psi)**2 - np.mean(np.abs(psi)**2)

    Y_iter = np.stack([Y_iter] * psi.shape[2], axis=2)

    Y_iter_torch = torch.tensor(Y_iter).cuda()
    X_iter_torch = torch.tensor(X_iter).cuda()
    diffuser_E_torch = torch.tensor(diffuser_E).cuda()
    Y_domain_torch = torch.tensor(Y_domain).cuda()
    dIyy_torch = torch.tensor(dIyy).cuda()
    sample_mask_torch = torch.tensor(sample_mask).cuda()

    for i in range(100):
        Y_iter_torch = T_cuda(X_iter_torch, diffuser_E_torch, Y_domain_torch)
        X_iter_torch = T_dagger_cuda(dIyy_torch * Y_iter_torch, diffuser_E_torch, sample_mask_torch)
        X_iter_torch = X_iter_torch / torch.sqrt(torch.sum(torch.abs(X_iter_torch)**2, axis=(0,1)))

    Y_iter = Y_iter_torch.cpu().numpy()
    X_iter = X_iter_torch.cpu().numpy()
    
    X_iter_0 = X_iter / (np.sqrt(np.sum(np.abs(sample_mask * X_iter) ** 2, axis=(0,1))/np.sum(sample_mask))) * eta_constant  / 2
    
    #########
    
    pad_size = psi.shape

    if optimization_method=='GD':
        pass
    elif optimization_method=='FISTA':
        t0 = 1
        grad_res = np.zeros(pad_size, dtype=X_iter_0.dtype)
    elif optimization_method in {'Adam', 'Nadam', 'AMSGrad'}:
        m_map = np.zeros(pad_size, dtype=X_iter_0.dtype)
        v_map = np.zeros(pad_size, dtype=X_iter_0.dtype)
                
    # Initial Forward Model
    # Use machine epsilon based on psi’s floating-point type.
    ep0 = np.finfo(psi.dtype).eps
    X_iter = X_iter_0
    
    #########
    
    Y_iter_torch = torch.tensor(Y_iter).cuda()
    X_iter_torch = torch.tensor(X_iter).cuda()
    diffuser_E_torch = torch.tensor(diffuser_E).cuda()
    Y_domain_torch = torch.tensor(Y_domain).cuda()
    psi_torch = torch.tensor(psi).cuda()
    sample_mask_torch = torch.tensor(sample_mask).cuda()
    grad_res_torch = torch.tensor(grad_res).cuda()

    ep0_torch = torch.tensor(ep0).cuda()
    step_size_torch = torch.tensor(step_size).cuda()
    t0_torch = torch.tensor(t0).cuda()
    sqrt2_torch = torch.tensor(np.sqrt(2)).cuda()


    for i in range(100):
        Y_iter_torch = T_cuda(X_iter_torch, diffuser_E_torch, Y_domain_torch)
        W_iter = torch.abs(Y_iter_torch) / (psi_torch + ep0_torch)
        mean_W_iter = torch.mean(W_iter, axis=(0,1))
        W_iter = W_iter / mean_W_iter
        
        normalization = torch.sqrt(W_iter**2 + mean_W_iter**(-2))
        y_vec = (mean_W_iter * normalization - sqrt2_torch) * (W_iter / (normalization + ep0_torch)) * psi_torch * torch.exp(1j * torch.angle(Y_iter_torch))
        
        # gradient
        g = T_dagger_cuda(y_vec, diffuser_E_torch, sample_mask_torch)
        old_grad_res = grad_res_torch
        grad_res_torch = X_iter_torch - step_size_torch * g
        t1 = (1 + torch.sqrt(1 + 4*(t0_torch**2)))/2
        beta = (t0_torch - 1) / t1
        t0_torch = t1
        X_iter_torch = grad_res_torch + beta * (grad_res_torch - old_grad_res)
        
    X_final = X_iter_torch.cpu().numpy()
    
    return X_final
        



def T(Xiter, diffuser_E, Y_domain=None):
    numel = np.prod(Xiter.shape[:2])
    # Explicitly use axes (0, 1) for the spatial dimensions
    temp = np.fft.fft2(Xiter, axes=(0, 1))
    temp = temp * diffuser_E
    Yiter = np.fft.fft2(temp, axes=(0, 1)) / numel
    if Y_domain is not None:
        Yiter = Yiter * Y_domain
    return Yiter

def T_dagger(Yiter, diffuser_E, X_domain=None):
    numel = np.prod(Yiter.shape[:2])
    temp = np.fft.fft2(np.conj(Yiter), axes=(0, 1))
    temp = temp * diffuser_E
    Xiter = np.conj(np.fft.fft2(temp, axes=(0, 1))) * X_domain / numel
    return Xiter

def T_cuda(Xiter, diffuser_E, Y_domain=None):
    X = Xiter
    E = diffuser_E
    Y_domain = Y_domain if Y_domain is not None else None

    numel = X.shape[0] * X.shape[1]
    temp = torch.fft.fft2(X, dim=(0,1))
    temp = temp * E
    Y = torch.fft.fft2(temp, dim=(0,1)) / numel
    if Y_domain is not None:
        Y = Y * Y_domain

    return Y

def T_dagger_cuda(Yiter, diffuser_E, X_domain=None):
    Y = Yiter
    E = diffuser_E
    X_domain = X_domain if X_domain is not None else 1.0

    numel = Y.shape[0] * Y.shape[1]
    temp = torch.fft.fft2(torch.conj(Y), dim=(0,1))
    temp = temp * E
    X = torch.conj(torch.fft.fft2(temp, dim=(0,1))) * X_domain / numel

    return X