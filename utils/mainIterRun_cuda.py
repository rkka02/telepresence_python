import numpy as np
from .imgCorrCalc import imgCorrCalc

def main_iter_run_cuda(psi, diffuserE, samMask, opt_method, **kwargs):
    """
    main_iter_run_cuda: Run the iterative optimization on the field estimate
    using GPU-accelerated convolution routines (fast_convolution) – here we simply
    translate the MATLAB code. (GPU routines T_cuda/Tdagger_cuda will be written later.)
    
    Parameters:
      psi        : Measured field intensity (or its square-root), as input data (numpy array).
      diffuserE  : Diffuser field (complex) applied in the forward model.
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
    YDomain     = kwargs.get('YDomain', None)
    Xiter0      = kwargs.get('Xiter0', 'zero')
    useSSMinit  = kwargs.get('useSSMinit', 0)

    if opt_method not in ['GD', 'FISTA', 'Adam', 'Nadam', 'AMSGrad']:
        raise ValueError("opt_method must be one of: 'GD', 'FISTA', 'Adam', 'Nadam', 'AMSGrad'")

    # Calculate Normalization Constants
    # Initialize Xiter with the sampling mask (masking zeros outside ROI)
    Xiter = np.copy(samMask)
    # Forward model: compute Yiter = T(Xiter)
    Yiter = T(Xiter, diffuserE, YDomain)
    # Back-projection: update Xiter from Yiter via adjoint operator Tdagger
    Xiter = Tdagger(Yiter, diffuserE, samMask)
    # Compute maximum absolute value of Xiter in spatial dimensions
    # (summing over the first two dims; assumes psi is at least 2D)
    phase_conj_delta = np.max(np.abs(Xiter), axis=(0, 1))
    # Constant for scaling the step-size based on psi energy
    psi_energy = np.sum(psi**2, axis=(0,1))
    eta_constant = np.sqrt(psi_energy / phase_conj_delta)
    
    # Normalize Xiter0 so that its masked energy is unit scaled, then scale by eta_constant.
    stack_num = psi.shape[2]
    samMask = np.stack([samMask] * stack_num, axis=2)
    diffuserE = np.stack([diffuserE] * stack_num, axis=2)
    YDomain = np.stack([YDomain] * stack_num, axis=2)

    # Set Hyper-Parameters for the Chosen Optimization Method
    if opt_method in ['GD', 'FISTA']:
        # Scale the step size elementwise
        step_size = step_size / phase_conj_delta
    elif opt_method in ['Adam', 'Nadam', 'AMSGrad']:
        eta = eta * eta_constant  # adjust eta with etaConstant
    else:
        raise ValueError("Invalid opt_method")  # should not happen

    # Define Initial Estimate Xiter0
    pad_size = psi.shape
    if isinstance(Xiter0, str):
        if Xiter0 == 'zero':
            Xiter0 = np.zeros(pad_size, dtype=diffuserE.dtype)
        elif Xiter0 in ['random', 'SSM']:
            # Create a random complex initial guess inside the sampling mask.
            # Assume diffuserE and samMask are 2D; if psi is 3D, replicate along third dim.
            rnd_guess = (np.random.randn(*diffuserE.shape) + 1j * np.random.randn(*diffuserE.shape)) * samMask
            if len(pad_size) == 3 and rnd_guess.ndim == 2:
                # replicate along the third dimension
                Xiter0 = np.tile(rnd_guess[:, :, np.newaxis], (1, 1, pad_size[2]))
            else:
                Xiter0 = rnd_guess
            Xiter0 = Xiter0.astype(diffuserE.dtype)
            
            mask_norm = np.sqrt(np.sum(np.abs(samMask * Xiter0)**2, axis=(0, 1)) / np.sum(samMask))
            # Reshape mask_norm for broadcasting if needed.
            Xiter0 = samMask * Xiter0 / mask_norm * eta_constant
        else:
            raise ValueError("Invalid Xiter0 option")
    elif Xiter0.ndim == 2:
        # Provided as a matrix; leave it unchanged.
        pass
    else:
        raise ValueError("Invalid Xiter0")

    # Optional Initialization via SSM (if useSSMinit flag is set)
    if useSSMinit:
        # Normalize over spatial dimensions
        norm_factor = np.sqrt(np.sum(np.abs(Xiter0)**2, axis=(0,1), keepdims=True))
        print(norm_factor)
        Xiter = Xiter0 / norm_factor
        dIyy = np.abs(psi)**2 - np.mean(np.abs(psi)**2)
        
        
        
        for ii in range(100):
            # Use T_cuda/Tdagger_cuda stubs (currently using T/Tdagger)
            Yiter = T(Xiter, diffuserE, YDomain)
            Xiter = T(dIyy * Yiter, diffuserE, samMask)
            norm_factor = np.sqrt(np.sum(np.abs(Xiter)**2, axis=(0,1), keepdims=True))
            Xiter = Xiter / norm_factor
        # Set the initial iterate for the main loop using the SSM result
        mask_norm = np.sqrt(np.sum(np.abs(samMask * Xiter)**2, axis=(0,1)) / np.sum(samMask))
        Xiter0 = Xiter / mask_norm * eta_constant

    # Initialize Variables for Iteration Loop
    pad_size = psi.shape
    if opt_method == 'FISTA':
        t0 = 1.0  # FISTA momentum parameter initialization
        GRAD_RES = np.zeros(pad_size, dtype=Xiter0.dtype)  # Store previous gradient result
    elif opt_method == 'GD':
        # No extra initialization needed for plain gradient descent.
        pass
    elif opt_method in ['Adam', 'Nadam', 'AMSGrad']:
        mMap = np.zeros(pad_size, dtype=Xiter0.dtype)  # First moment vector
        vMap = np.zeros(pad_size, dtype=Xiter0.dtype)  # Second moment vector

    # Initial Forward Model
    # Use machine epsilon based on psi’s floating-point type.
    if np.iscomplexobj(psi):
        ep0 = np.finfo(np.float64).eps
    else:
        ep0 = np.finfo(psi.dtype).eps
    Xiter = Xiter0.copy()
    Yiter = T(Xiter, diffuserE, YDomain)

    # Main Iteration Loop
    for kk in range(iter_max):
        # Store previous Yiter (if needed for convergence checking)
        Ybefore = Yiter.copy()
        # Forward model update
        Yiter = T(Xiter, diffuserE, YDomain)
        # SAF (Spatial Amplitude Feedback) step:
        Witer = np.abs(Yiter) / (psi + ep0)
        meanWiter = np.mean(Witer, axis=(0, 1), keepdims=True)
        Witer = Witer / meanWiter
        PnormTerm = np.sqrt(Witer**2 + meanWiter**(-2))
        yVec = ((meanWiter * PnormTerm - np.sqrt(2)) *
                (Witer / (PnormTerm + ep0)) *
                psi * np.exp(1j * np.angle(Yiter)))
        # Compute the gradient via the adjoint operator.
        # In the MATLAB code, Tdagger_cuda was intended but T is used as a placeholder.
        g = T(yVec, diffuserE, samMask)

        # Update Xiter according to the chosen optimization method.
        if opt_method == 'FISTA':
            OLD_GRAD_RES = GRAD_RES.copy()
            GRAD_RES = Xiter - step_size * g
            t1 = (1 + np.sqrt(1 + 4 * t0**2)) / 2.0
            beta = (t0 - 1) / t1
            t0 = t1
            Xiter = GRAD_RES + beta * (GRAD_RES - OLD_GRAD_RES)
        elif opt_method == 'GD':
            Xiter = Xiter - step_size * g
        elif opt_method in ['Adam', 'Nadam', 'AMSGrad']:
            # A standard Adam update is implemented here.
            mMap = beta1 * mMap + (1 - beta1) * g
            vMap = beta2 * vMap + (1 - beta2) * (g ** 2)
            # Bias correction
            m_hat = mMap / (1 - beta1 ** (kk + 1))
            v_hat = vMap / (1 - beta2 ** (kk + 1))
            Xiter = Xiter - eta * m_hat / (np.sqrt(v_hat) + epsilon)
        else:
            raise ValueError("Invalid optimization method")
        # (Optional: add convergence checking based on correlation here)

    # Output Final Estimate
    Xfinal = Xiter
    return Xfinal


def T(Xiter, diffuserE, YDomain):
    """
    Forward model: apply diffuser operator.
    Computes:
         Yiter = fft2( fft2(Xiter) * diffuserE ) / (num. elements in one spatial slice)
         If YDomain is provided, multiplies the result elementwise.
    """
    # Compute number of spatial elements (assumes Xiter is at least 2D)
    numel = np.prod(Xiter.shape[:2])
    temp = np.fft.fft2(Xiter)
    temp = temp * diffuserE  # elementwise multiplication
    Yiter = np.fft.fft2(temp) / numel
    if YDomain is not None:
        Yiter = Yiter * YDomain
    return Yiter


def Tdagger(Yiter, diffuserE, XDomain):
    """
    Adjoint operator for back-projection.
    Computes:
         Xiter = conj( fft2( fft2(conj(Yiter)) * diffuserE ) ) .* XDomain / (num. elements in one spatial slice)
    """
    numel = np.prod(Yiter.shape[:2])
    temp = np.fft.fft2(np.conj(Yiter))
    temp = temp * diffuserE
    Xiter = np.conj(np.fft.fft2(temp)) * XDomain / numel
    return Xiter


def T_cuda(Xiter, diffuserE, YDomain):
    """
    GPU-accelerated forward operator using CuPy.
    Computes:
         Yiter = fft2( fft2(Xiter) * diffuserE ) / (num. elements in one spatial slice)
         If YDomain is provided, multiplies the result elementwise.
    """
    import cupy as cp
    # Transfer inputs to GPU
    X_gpu = cp.asarray(Xiter)
    diffuserE_gpu = cp.asarray(diffuserE)
    YDomain_gpu = cp.asarray(YDomain) if YDomain is not None else None

    # Compute number of spatial elements
    numel = cp.prod(cp.array(X_gpu.shape[:2]))
    temp = cp.fft.fft2(X_gpu)
    temp = temp * diffuserE_gpu  # elementwise multiplication
    Y_gpu = cp.fft.fft2(temp) / numel
    if YDomain_gpu is not None:
        Y_gpu = Y_gpu * YDomain_gpu

    # Convert back to NumPy (if necessary)
    return cp.asnumpy(Y_gpu)


def Tdagger_cuda(Yiter, diffuserE, XDomain):
    """
    GPU-accelerated adjoint operator using CuPy.
    Computes:
         Xiter = conj( fft2( fft2(conj(Yiter)) * diffuserE ) ) * XDomain / (num. elements in one spatial slice)
    """
    import cupy as cp
    # Transfer inputs to GPU
    Y_gpu = cp.asarray(Yiter)
    diffuserE_gpu = cp.asarray(diffuserE)
    XDomain_gpu = cp.asarray(XDomain)

    # Compute number of spatial elements
    numel = cp.prod(cp.array(Y_gpu.shape[:2]))
    temp = cp.fft.fft2(cp.conj(Y_gpu))
    temp = temp * diffuserE_gpu
    X_gpu = cp.conj(cp.fft.fft2(temp)) * XDomain_gpu / numel

    # Convert back to NumPy (if necessary)
    return cp.asnumpy(X_gpu)