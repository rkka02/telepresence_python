import numpy as np
import scipy.io as sio
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import zoom, affine_transform

from .mpad import mpad
from .imMagRot import imMagRot

def gpdField(gpdDir, gpdSize, gpdPix, gpdFlip, inputPol, camSize, camPix,
                     camFlip, wl, f1, f2, samFOV, **kwargs):
    """
    gpdFieldMaker_v2: Generate the diffuser field for the holographic system.
    
    Parameters:
      gpdDir    : str
          File path to the diffuser data (.mat file containing matrix 'M')
      gpdSize   : tuple or list (rows, cols)
          Expected size of the GPD.
      gpdPix    : float
          Diffuser pixel size (in micrometers)
      gpdFlip   : bool or int
          Flag to indicate whether to flip the GPD horizontally.
      inputPol  : str
          Input polarization ('L' or 'R'); for 'L', phase is inverted.
      camSize   : tuple or list (rows, cols)
          Camera sensor size (for output diffuser field).
      camPix    : float
          Camera pixel size (in micrometers).
      camFlip   : bool or int
          Flag for additional horizontal flip based on camera settings.
      wl        : float
          Wavelength (in micrometers).
      f1, f2    : float
          Focal lengths (in micrometers) of the two lenses.
      samFOV    : float
          Sample field-of-view (in micrometers).
      
    Optional keyword parameters:
      gpdRot    : float (default=0)
          Rotation angle (in degrees) to apply to the GPD.
      gpdMag    : float (default=1)
          Magnification factor for GPD adjustment.
      
    Returns:
      diffuserE : 2D complex numpy array
          The complex diffuser field.
      magFactor : numpy array (2 elements)
          Magnification factor (derived from camera and optical parameters).
      GPDwindow : 2D boolean numpy array
          Binary window applied during oversampling and cropping.
    """
    # Optional parameters.
    gpdRot = kwargs.get('gpdRot', 0)
    gpdMag = kwargs.get('gpdMag', 1)
    
    # --- Load and Prepare the Diffuser Matrix ---
    data = sio.loadmat(gpdDir)
    if 'M' not in data:
        raise ValueError("The file does not contain variable 'M'.")
    M = data['M']
    if not np.all(np.array(gpdSize) == M.shape):
        raise ValueError("GPD size mismatch")
    
    # Adjust phase sign based on polarization.
    if inputPol == 'L':
        M = -M
    # For 'R', do nothing.
    
    # Apply flip settings.
    if not gpdFlip:
        diffuser_deg = M.copy()
    else:
        diffuser_deg = -np.fliplr(M)
    if camFlip:
        diffuser_deg = np.fliplr(diffuser_deg)
    
    # Convert degrees to phase in radians and shift from [0, 2pi] to [-pi, pi].
    diffuserE = np.float32(2 * np.deg2rad(diffuser_deg) - np.pi)
    # (Optional visualization code omitted)
    
    # --- Define the Padding Field-of-View (FOV) ---
    mag = f2 / f1
    camSize = np.array(camSize, dtype=float)
    padFOV = camSize * camPix  # in micrometers
    
    # --- GPD Oversampling: x-space oversampling via k-space padding ---
    camFOV_for_oversampling = 2 * padFOV          # Double the camera FOV (um)
    gpdFOV_for_oversampling = wl * f2 / camPix      # Required GPD FOV (um); scalar
    # Desired oversampled GPD pixel size (um): note division is element-wise.
    gpdPix_for_oversampling = (wl * f2) / camFOV_for_oversampling
    # Scale factor (per dimension) to achieve oversampling.
    kPadScale = np.ceil(gpdPix / gpdPix_for_oversampling).astype(int)  # vector (2 elements)
    # New effective GPD pixel size after oversampling.
    gpdPix_afterOVS = gpdPix / kPadScale.astype(float)
    
    # Upsample (resize) the diffuser phase map using nearest-neighbor interpolation.
    # New shape: kPadScale * original shape.
    new_shape = (kPadScale[0] * diffuserE.shape[0], kPadScale[1] * diffuserE.shape[1])
    diffuserE = zoom(diffuserE, zoom=kPadScale, order=0, mode='nearest')
    
    # --- GPD Oversampling: k-space oversampling via x-space padding ---
    # Determine new GPD size required to cover the oversampled FOV.
    gpdSize_after_oversampling = np.ceil(gpdFOV_for_oversampling / gpdPix_afterOVS).astype(int)
    gpdFOV_after_oversampling  = gpdSize_after_oversampling * gpdPix_afterOVS  # in um
    
    # Pad the upsampled diffuser field to the oversampled size.
    diffuserE = mpad(diffuserE, gpdSize_after_oversampling)
    
    # --- Convert Phase to Complex Field ---
    # Create a binary window matching the padded GPD.
    # First, create an array of ones with shape: kPadScale * original gpdSize.
    base_shape = (kPadScale[0] * gpdSize[0], kPadScale[1] * gpdSize[1])
    GPDwindow = mpad(np.ones(base_shape, dtype=bool), gpdSize_after_oversampling)
    # Compute the complex field.
    diffuserE = np.exp(1j * diffuserE) * GPDwindow
    
    # --- GPD Cropping and Adjustment ---
    # Determine effective camera pixel size in the oversampled GPD FOV.
    camPix2 = (wl * f2) / gpdFOV_after_oversampling  # vector (um)
    magFactor = camPix / camPix2  # element-wise division
    if not np.all(magFactor >= 1):
        raise ValueError("magFactor should be >= 1")
    
    # Define pad size based on the camera FOV.
    padSize = np.ceil(padFOV / camPix).astype(int)  # typically equals camSize
    # Transform diffuserE to Fourier domain.
    fdiffuserE = fftshift(fft2(ifftshift(diffuserE)))
    # Rotate and magnify in Fourier domain.
    # Note: Here the scale factor passed is gpdMag/magFactor (element‐wise).
    # For the affine transform, we use a 2-element scale. Ensure it’s in the proper order.
    fdiffuserE = imMagRot(fdiffuserE, gpdMag / magFactor, gpdRot, padSize, interpMethod='cubic')
    # Transform back to spatial domain and normalize.
    normFactor = np.prod(padSize) / np.prod(gpdSize_after_oversampling)
    diffuserE = fftshift(ifft2(ifftshift(fdiffuserE))) * normFactor
    
    # --- Final GPD Trimming ---
    # Define a cropping window based on the optical parameters.
    # Calculation: windowSize = ceil(gpdSize * gpdPix / (wl*f2) .* padSize * camPix)
    windowSize = np.ceil(np.array(gpdSize, dtype=float) * gpdPix / (wl * f2) *
                         np.array(padSize, dtype=float) * camPix).astype(int)
    GPDwindow = mpad(np.ones(windowSize, dtype=bool), padSize)
    # Apply the window to the diffuser field.
    diffuserE = diffuserE * GPDwindow
    
    return diffuserE, magFactor, GPDwindow

# Example usage:
# diffuserE, magFactor, GPDwindow = gpdFieldMaker_v2(
#     'path/to/microretarders_mfile.mat',
#     (512, 512),
#     30,
#     1,
#     'R',
#     (1024, 1024),
#     3.1,
#     0,
#     0.532,
#     300000,
#     200000,
#     1200,
#     gpdRot=-0.19867,
#     gpdMag=1/0.9985
# )