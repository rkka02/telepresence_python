import time
import numpy as np
from numpy.fft import ifftshift
import torch

def main_loop(psi, diffuser_E, sample_mask, spk_window, shiftvec):
    total_start = time.time()
    
    pre_loop_start = time.time()
    optimization_method = 'FISTA'

    Y_domain = spk_window
    eta = 0.5e-2
    step_size = 1
    X_iter_0 = 'random'
    use_SSMinit = True

    diffuser_E = ifftshift(diffuser_E)
    sample_mask = np.roll(sample_mask, shift=(shiftvec[0], shiftvec[1]), axis=(0, 1))

    X_iter = sample_mask
    Y_iter = T(X_iter, diffuser_E, Y_domain)
    X_iter = T_dagger(Y_iter, diffuser_E, sample_mask)

    phase_conj_delta = np.max(np.abs(X_iter), axis=(0, 1))
    psi_energy = np.sum(psi**2, axis=(0, 1))
    eta_constant = np.sqrt(psi_energy / phase_conj_delta)

    if optimization_method in ['GD', 'FISTA']:
        step_size = step_size / phase_conj_delta
    elif optimization_method in ['Adam', 'Nadam', 'AMSGrad']:
        eta = eta * eta_constant
    else:
        raise ValueError("Invalid opt_method")

    if psi.shape[2] is not None:
        stack_num = psi.shape[2]
        diffuser_E = np.stack([diffuser_E] * stack_num, axis=2)
        sample_mask = np.stack([sample_mask] * stack_num, axis=2)
        Y_domain = np.stack([Y_domain] * stack_num, axis=2)

    pad_size = psi.shape

    if isinstance(X_iter_0, str):
        if X_iter_0 == 'zero':
            X_iter_0 = np.zeros(pad_size, dtype=diffuser_E.dtype)
        elif X_iter_0 in ['random', 'SSM']:
            X_iter_0 = (np.random.randn(*diffuser_E.shape) +
                        1j * np.random.randn(*diffuser_E.shape)) * sample_mask
            if X_iter.ndim < len(pad_size):
                X_iter_0 = np.tile(X_iter[..., np.newaxis], (1, 1, pad_size[2]))
            else:
                X_iter_0 = X_iter
            X_iter_0 = X_iter_0.astype(diffuser_E.dtype)
            norm_denom = np.sqrt(np.sum(np.abs(sample_mask * X_iter_0)**2, axis=(0, 1)) /
                                 np.sum(sample_mask))
            if norm_denom.ndim == 1:
                norm_denom = norm_denom[None, None, :]
            stack_X_iter = np.stack([X_iter]*sample_mask.shape[2], axis=2)
            X_iter_0 = (sample_mask * stack_X_iter) / norm_denom * eta_constant / 2
        else:
            raise ValueError("Invalid X_iter_0")
    elif isinstance(X_iter_0, np.ndarray):
        X_iter_0 = X_iter_0
    else:
        raise ValueError("Invalid X_iter_0")

    X_iter = X_iter_0 / np.sqrt(np.sum(np.abs(X_iter_0)**2, axis=(0, 1)))
    dIyy = np.abs(psi)**2 - np.mean(np.abs(psi)**2)
    Y_iter = np.stack([Y_iter] * psi.shape[2], axis=2)

    # Transfer to torch
    Y_iter_torch = torch.tensor(Y_iter).cuda()
    X_iter_torch = torch.tensor(X_iter).cuda()
    diffuser_E_torch = torch.tensor(diffuser_E).cuda()
    Y_domain_torch = torch.tensor(Y_domain).cuda()
    dIyy_torch = torch.tensor(dIyy).cuda()
    sample_mask_torch = torch.tensor(sample_mask).cuda()
    pre_loop_end = time.time()

    # ---------- First Loop ----------
    loop1_start = time.time()
    for i in range(100):
        Y_iter_torch = T_cuda_jit(X_iter_torch, diffuser_E_torch, Y_domain_torch)
        X_iter_torch = T_dagger_cuda_jit(dIyy_torch * Y_iter_torch, diffuser_E_torch, sample_mask_torch)
        X_iter_torch = X_iter_torch / torch.sqrt(torch.sum(torch.abs(X_iter_torch)**2, axis=(0, 1)))
    loop1_end = time.time()

    post_loop1_start = time.time()
    
    eta_constant_torch = torch.tensor(eta_constant).cuda()
    X_iter_0 = X_iter_torch / (torch.sqrt(torch.sum(torch.abs(sample_mask_torch * X_iter_torch) ** 2, axis=(0, 1)) /
                                 torch.sum(sample_mask_torch))) * eta_constant_torch / 2

    if optimization_method == 'GD':
        pass
    elif optimization_method == 'FISTA':
        t0 = 1
        grad_res_torch = torch.zeros(pad_size, dtype=torch.complex128).cuda()
    elif optimization_method in {'Adam', 'Nadam', 'AMSGrad'}:
        m_map = np.zeros(pad_size, dtype=X_iter_0.dtype)
        v_map = np.zeros(pad_size, dtype=X_iter_0.dtype)
    post_loop1_end = time.time()

    # ---------- Second Loop (FISTA) ----------
    loop2_start = time.time()
    
    psi_torch = torch.tensor(psi).cuda()

    ep0_torch = torch.tensor(np.finfo(psi.dtype).eps).cuda()
    step_size_torch = torch.tensor(step_size).cuda()
    t0_torch = torch.tensor(t0).cuda()
    sqrt2_torch = torch.tensor(np.sqrt(2)).cuda()

    for i in range(100):
        Y_iter_torch = T_cuda_jit(X_iter_torch, diffuser_E_torch, Y_domain_torch)
        W_iter = torch.abs(Y_iter_torch) / (psi_torch + ep0_torch)
        mean_W_iter = torch.mean(W_iter, axis=(0, 1))
        W_iter = W_iter / mean_W_iter

        normalization = torch.sqrt(W_iter**2 + mean_W_iter**(-2))
        y_vec = (mean_W_iter * normalization - sqrt2_torch) * \
            (W_iter / (normalization + ep0_torch)) * psi_torch * torch.exp(1j * torch.angle(Y_iter_torch))

        g = T_dagger_cuda_jit(y_vec, diffuser_E_torch, sample_mask_torch)
        old_grad_res = grad_res_torch
        grad_res_torch = X_iter_torch - step_size_torch * g
        t1 = (1 + torch.sqrt(1 + 4*(t0_torch**2)))/2
        beta = (t0_torch - 1) / t1
        t0_torch = t1
        X_iter_torch = grad_res_torch + beta * (grad_res_torch - old_grad_res)
    loop2_end = time.time()

    # ---------- Return ----------
    X_final = X_iter_torch.cpu().numpy()
    total_end = time.time()

    print(f"[main_loop] Time breakdown (sec):")
    print(f"  Pre-loop setup        : {pre_loop_end - pre_loop_start:.3f}")
    print(f"  First loop (100 iters): {loop1_end - loop1_start:.3f}")
    print(f"  Between loops         : {post_loop1_end - post_loop1_start:.3f}")
    print(f"  FISTA loop (100 iters): {loop2_end - loop2_start:.3f}")
    print(f"  Total time            : {total_end - total_start:.3f}")

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

@torch.jit.script
def T_cuda_jit(Xiter, diffuser_E, Y_domain):
    numel = Xiter.shape[0] * Xiter.shape[1]
    temp = torch.fft.fft2(Xiter, dim=(0,1))
    temp = temp * diffuser_E
    Y = torch.fft.fft2(temp, dim=(0,1)) / numel
    if Y_domain is not None:
        Y = Y * Y_domain
    return Y

@torch.jit.script
def T_dagger_cuda_jit(Yiter, diffuser_E, X_domain=None):
    Y = Yiter
    E = diffuser_E
    X_domain = X_domain if X_domain is not None else 1.0

    numel = Y.shape[0] * Y.shape[1]
    temp = torch.fft.fft2(torch.conj(Y), dim=(0,1))
    temp = temp * E
    X = torch.conj(torch.fft.fft2(temp, dim=(0,1))) * X_domain / numel

    return X