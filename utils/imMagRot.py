import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import affine_transform
from scipy.interpolate import RegularGridInterpolator

from .mpad import mpad
from .mcrop import mcrop
from .downsample2d import downsample2d
from .mcoor import mcoor

def imMagRot(input_image, mag, rotAngle, outSize=None, interpMethod='linear', magMethod=''):
    """
    imMagRot: Magnify and rotate an image.
    
    Rotates the input image (counter-clockwise, as in MATLAB imrotate) and
    applies a magnification. Supports an optional output size and interpolation method.
    
    Parameters:
      input_image  : 2D numpy array (can be real or complex).
      mag          : scalar or 2-element vector for magnification factor.
      rotAngle     : rotation angle in degrees (counter-clockwise).
      outSize      : desired output size [rows, cols]. If None, uses input size.
      interpMethod : interpolation method ('linear', 'nearest', or 'cubic').
      magMethod    : Magnification method: 'legacy' or '' (default). If empty, oversampling is applied.
      
    Returns:
      output       : Transformed image.
    """
    # Ensure mag is a 2-element vector.
    if np.isscalar(mag):
        mag = np.array([mag, mag], dtype=float)
    else:
        mag = np.array(mag, dtype=float)
    
    if np.any(mag <= 0):
        raise ValueError("Magnification factor must be > 0")
    
    # Check trivial case: no magnification and no rotation.
    if np.allclose(mag, [1, 1]) and (rotAngle % 360 == 0):
        output = input_image.copy()
        if outSize is not None:
            # Use mpad's companion, mcrop. Here we define mcrop as centered cropping.
            output = mcrop(output, outSize)
        return output
    
    # Prepare for magnification and rotation.
    if magMethod == '':
        # Oversampling to force mag >= 1 if necessary.
        if np.any(mag < 1):
            integerOSratio = np.ceil(1.0 / mag).astype(int)
            mag0 = mag.copy()
            mag = mag0 * integerOSratio
        else:
            integerOSratio = np.array([1, 1], dtype=int)
        inSize = np.array(input_image.shape, dtype=int)
        padSize = np.floor(mag * inSize).astype(int)
        # Adjust effective magnification.
        magForInterp = mag / (padSize / inSize)
        # Oversample input via FFT padding.
        Finput = fftshift(fft2(ifftshift(input_image)))
        Finput = mpad(Finput, padSize)
        inputForInterp = fftshift(ifft2(ifftshift(Finput))) * np.prod(padSize) / np.prod(inSize)
    else:
        # Legacy: no oversampling.
        inputForInterp = input_image.copy()
        magForInterp = mag.copy()
        integerOSratio = np.array([1, 1], dtype=int)
    
    # Create coordinate grids for interpolation.
    ny, nx = inputForInterp.shape
    # Create vectors as in MATLAB: 1:n - mcoor(n)
    xvec = np.arange(1, nx + 1) - mcoor(nx)
    yvec = np.arange(1, ny + 1) - mcoor(ny)
    XX, YY = np.meshgrid(xvec, yvec)
    
    # Set output size if not provided.
    if outSize is None:
        outSize = (ny, nx)
    else:
        outSize = tuple(outSize)
    
    # Adjust output size for oversampling.
    outSizeForInterp = (outSize[0] * integerOSratio[0], outSize[1] * integerOSratio[1])
    yvec2 = np.arange(1, outSizeForInterp[0] + 1) - mcoor(outSizeForInterp[0])
    xvec2 = np.arange(1, outSizeForInterp[1] + 1) - mcoor(outSizeForInterp[1])
    XX2, YY2 = np.meshgrid(xvec2, yvec2)
    
    # Rotate and scale coordinates.
    # Note: cosd and sind use degrees.
    cosA = np.cos(np.deg2rad(rotAngle))
    sinA = np.sin(np.deg2rad(rotAngle))
    # Adjust coordinates with rotation and inverse magnification.
    # MATLAB: (XX2*cosd(rotAngle) - YY2*sind(rotAngle)) / magForInterp(2)
    XX2rot = (XX2 * cosA - YY2 * sinA) / magForInterp[1]
    YY2rot = (XX2 * sinA + YY2 * cosA) / magForInterp[0]
    
    # Interpolate the oversampled image.
    # Build an interpolator on the grid defined by (yvec, xvec).
    # Note: RegularGridInterpolator expects the grid in (y, x) order.
    if interpMethod == 'cubic':
        method = 'cubic'
    elif interpMethod == 'nearest':
        method = 'nearest'
    else:
        method = 'linear'
    interpolator = RegularGridInterpolator((yvec, xvec), inputForInterp, method=method,
                                           bounds_error=False, fill_value=0)
    # Prepare query points. They should be an (N,2) array with coordinates in (y,x) order.
    pts = np.vstack([YY2rot.ravel(), XX2rot.ravel()]).T
    output_interp = interpolator(pts).reshape(outSizeForInterp)
    
    # Use the consolidated downsample2d.
    output_down = downsample2d(output_interp, n=integerOSratio, phase=0, centeredTF=True)
    return output_down