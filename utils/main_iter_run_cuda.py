import torch
import numpy as np

def main_iter_run_cuda(psi, diffuser_E, sample_mask, opt_method, **kwargs):
    """
    main_iter_run_cuda: Run the iterative optimization on the field estimate
    using GPU-accelerated convolution routines (fast_convolution) – here we simply
    translate the MATLAB code. (GPU routines T_cuda/Tdagger_cuda will be written later.)
    
    Parameters:
      psi        : Measured field intensity (or its square-root), as input data (numpy array).
      diffuser_E  : Diffuser field (complex) applied in the forward model.
      samMask    : Sampling mask (defines region of interest) in the spatial domain.
      opt_method : Optimization method ('GD', 'FISTA', 'Adam', 'Nadam', 'AMSGrad').
      kwargs     : Additional parameters, with defaults:
                   - iterMax   : 500
                   - corrCrit  : 1e-5
                   - stepSize  : 1
                   - eta       : 1e-2
                   - beta1     : 0.9
                   - beta2     : 0.999
                   - epsilon   : 1e-8
                   - gpuInd    : None
                   - YDomain   : None
                   - Xiter0    : 'zero'    (or 'random' or 'SSM')
                   - useSSMinit: 0
                   
    Returns:
      Xfinal : Final estimate after iterative optimization.
    """
    # Parse optional parameters
    iter_max    = kwargs.get('iterMax', 500)
    corr_crit   = kwargs.get('corrCrit', 1e-5)
    step_size   = kwargs.get('stepSize', 1)
    eta         = kwargs.get('eta', 1e-2)
    beta1       = kwargs.get('beta1', 0.9)
    beta2       = kwargs.get('beta2', 0.999)
    epsilon     = kwargs.get('epsilon', 1e-8)
    gpu_ind     = kwargs.get('gpuInd', None)
    Y_domain     = kwargs.get('YDomain', None)
    X_iter_0      = kwargs.get('Xiter0', 'zero')
    use_SSMinit  = kwargs.get('useSSMinit', 0)
    
    if opt_method not in ['GD', 'FISTA', 'Adam', 'Nadam', 'AMSGrad']:
        raise ValueError("opt_method must be one of: 'GD', 'FISTA', 'Adam', 'Nadam', 'AMSGrad'")
    
    # Calculae normalization constants
    X_iter = sample_mask
    Y_iter = T(X_iter, diffuser_E, Y_domain)
    X_iter = T_dagger(Y_iter, diffuser_E, sample_mask)
    # Compute maximum absolute value of Xiter in spatial dimensions
    # (summing over the first two dims; assumes psi is at least 2D)
    phase_conj_delta = np.max(np.abs(X_iter), axis=(0, 1))
    # Constant for scaling the step-size based on psi energy
    psi_energy = np.sum(psi**2, axis=(0,1))
    eta_constant = np.sqrt(psi_energy / phase_conj_delta)
    
    # Set Hyper-Parameters for the Chosen Optimization Method
    if opt_method in ['GD', 'FISTA']:
        # Scale the step size elementwise
        step_size = step_size / phase_conj_delta
    elif opt_method in ['Adam', 'Nadam', 'AMSGrad']:
        eta = eta * eta_constant  # adjust eta with etaConstant
    else:
        raise ValueError("Invalid opt_method")  # should not happen
    
    
    return X_Final

##########################################################################

def T(Xiter, diffuser_E, Y_domain=None):
    """
    Forward model: apply diffuser operator.
    Computes:
         Yiter = fft2( fft2(Xiter) * diffuser_E ) / (num. elements in one spatial slice)
         If YDomain is provided, multiplies the result elementwise.
    """
    # Compute number of spatial elements (assumes Xiter is at least 2D)
    numel = np.prod(Xiter.shape[:2])
    temp = np.fft.fft2(Xiter)
    temp = temp * diffuser_E  # elementwise multiplication
    Yiter = np.fft.fft2(temp) / numel
    if Y_domain is not None:
        Yiter = Yiter * Y_domain
    return Yiter


def T_dagger(Yiter, diffuser_E, X_domain=None):
    """
    Adjoint operator for back-projection.
    Computes:
         Xiter = conj( fft2( fft2(conj(Yiter)) * diffuser_E ) ) .* XDomain / (num. elements in one spatial slice)
    """
    numel = np.prod(Yiter.shape[:2])
    temp = np.fft.fft2(np.conj(Yiter))
    temp = temp * diffuser_E
    Xiter = np.conj(np.fft.fft2(temp)) * X_domain / numel
    return Xiter


def T_cuda(Xiter, diffuser_E, Y_domain=None):
    """
    GPU-accelerated forward operator using PyTorch.
    Computes:
         Yiter = fft2( fft2(Xiter) * diffuser_E ) / (num. elements in one spatial slice)
         Optionally multiplies by Y_domain elementwise.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to complex tensors on GPU
    X = torch.tensor(Xiter, dtype=torch.cfloat, device=device)
    E = torch.tensor(diffuser_E, dtype=torch.cfloat, device=device)
    Y_domain = torch.tensor(Y_domain, dtype=torch.cfloat, device=device) if Y_domain is not None else None

    numel = X.shape[0] * X.shape[1]
    temp = torch.fft.fft2(X)
    temp = temp * E
    Y = torch.fft.fft2(temp) / numel
    if Y_domain is not None:
        Y = Y * Y_domain

    return Y.cpu().numpy()

def T_dagger_cuda(Yiter, diffuser_E, X_domain=None):
    """
    GPU-accelerated adjoint operator using PyTorch.
    Computes:
         Xiter = conj( fft2( fft2(conj(Yiter)) * diffuser_E ) ) * XDomain / numel
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to complex tensors on GPU
    Y = torch.tensor(Yiter, dtype=torch.cfloat, device=device)
    E = torch.tensor(diffuser_E, dtype=torch.cfloat, device=device)
    X_domain = torch.tensor(X_domain, dtype=torch.cfloat, device=device) if X_domain is not None else 1.0

    numel = Y.shape[0] * Y.shape[1]
    temp = torch.fft.fft2(torch.conj(Y))
    temp = temp * E
    X = torch.conj(torch.fft.fft2(temp)) * X_domain / numel

    return X.cpu().numpy()