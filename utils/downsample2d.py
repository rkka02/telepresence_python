import numpy as np

def downsample1d(arr, factor, phase):
    """
    Downsample a 1D array by taking every 'factor'-th element starting at index 'phase'.
    """
    return arr[phase::factor]

def downsample2d(input_array, n, phase=0, centeredTF=False):
    """
    downsample2d: Downsample a 2D input array by a specified factor, with an optional phase shift.
    
    Parameters:
      input_array : 2D numpy array
          The array to be downsampled.
      n           : int or sequence of 1 or 2 ints
          Downsampling factor. If a scalar is provided, the same factor is used for both dimensions.
      phase       : int or sequence of 1 or 2 ints, optional (default=0)
          Phase shift applied to the sampling grid.
      centeredTF  : bool, optional (default=False)
          If True, centers the sampling grid by adjusting the phase based on the center of the input.
    
    Returns:
      output : 2D numpy array
          The downsampled array.
    
    Raises:
      ValueError: if n has length greater than 2.
    """
    # Ensure n is a 2-element vector.
    if np.isscalar(n):
        n = [int(n), int(n)]
    else:
        n = list(n)
        if len(n) == 1:
            n = [n[0], n[0]]
        elif len(n) > 2:
            raise ValueError("length(n) must be 1 or 2")
        else:
            n = [int(n[0]), int(n[1])]
    
    # Ensure phase is a 2-element vector.
    if np.isscalar(phase):
        phase = [int(phase), int(phase)]
    else:
        phase = list(phase)
        if len(phase) == 1:
            phase = [phase[0], phase[0]]
        elif len(phase) > 2:
            raise ValueError("phase must be a scalar or a vector of length 1 or 2")
        else:
            phase = [int(phase[0]), int(phase[1])]
    
    # If centered sampling is requested, adjust the phase offsets.
    if centeredTF:
        # Compute center of input in each dimension (using floor division).
        input_shape = input_array.shape
        center_row = input_shape[0] // 2
        center_col = input_shape[1] // 2
        # Adjust phase: (floor(size/2) mod factor) + original phase.
        phase[0] = (center_row % n[0]) + phase[0]
        phase[1] = (center_col % n[1]) + phase[1]
    
    # Downsample rows with factor n[0] and phase offset phase[0].
    down_rows = downsample1d(input_array, n[0], phase[0])
    # Downsample columns by transposing, applying the same logic, and transposing back.
    down_cols = downsample1d(down_rows.T, n[1], phase[1]).T

    return down_cols
