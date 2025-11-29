import numpy as np
import scipy.io as sio

def SLM_LUT():
    """
    SLM_LUT: Load the SLM lookup table (LUT) mapping continuous phase values
    to discrete SLM digital levels.

    The function loads a pre-calibrated phase-corrected file, extracts the
    phase information, and then constructs a unique mapping between phase values
    and corresponding SLM digital values.

    Returns:
        phaseVal : numpy array
            Unique phase values, centered to be within [-pi, pi].
        SLMDigit : numpy array
            Corresponding digital SLM values (0-255) for each phase value.
    """
    # Load the phase calibration data.
    # The file 'HW_58_pol_60_phase_correct.mat' should contain a variable 'totalSum'.
    data = sio.loadmat('C:/rkka_Projects/telepresense/HW_58_pol_60_phase_correct.mat')
    if 'totalSum' not in data:
        raise KeyError("Variable 'totalSum' not found in calibration file.")
    
    # Extract the phase data.
    phase = data['totalSum']
    
    # Flip the phase data vertically to match the SLM's orientation.
    flipphase = np.flipud(phase)
    
    # Define the SLM digital output range.
    # 256 discrete levels from 255 down to 0.
    ydata = np.linspace(255, 0, 256)
    
    # xdata is taken from the flipped phase calibration data.
    xdata = flipphase
    
    # Create a unique mapping between phase values and SLM digits.
    # np.unique returns sorted unique values and their original indices.
    phaseVal, ia = np.unique(xdata, return_index=True)
    SLMDigit = ydata[ia]
    
    # Center the phase values.
    phaseCenter = (np.max(phaseVal) + np.min(phaseVal)) / 2
    phaseVal = phaseVal - phaseCenter
    
    return phaseVal, SLMDigit