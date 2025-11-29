import numpy as np

def mpad(input_array, final_size):
    """
    mpad: Pad an array to a specified final size by adding zeros.

    Usage:
      output = mpad(input_array, final_size)

    Parameters:
      input_array : numpy array
          The input array to be padded.
      final_size : sequence of int
          A vector specifying the desired size in each dimension.
          The length of final_size should match the number of dimensions
          to be padded.

    Returns:
      output : numpy array
          The padded array, centered relative to the input.
    """
    # Validate final_size is a vector-like sequence.
    if not isinstance(final_size, (list, tuple, np.ndarray)):
        raise ValueError("final_size must be a vector (list, tuple, or numpy array).")
    
    final_size = np.array(final_size, dtype=int)
    current_dim = final_size.size
    
    # Get the size of the input along the dimensions to pad.
    original_shape = np.array(input_array.shape[:current_dim], dtype=int)
    
    # Ensure the final size is not smaller than the original size.
    if np.any(final_size < original_shape):
        raise ValueError("The final dimension must be larger than the initial dimension.")
    
    # Compute centering offsets.
    # MATLAB uses 1-indexing, so mvec and mMsz are computed as floor(size/2)+1.
    # For numpy's 0-indexing the relative shift remains the same.
    mvec = np.floor_divide(final_size, 2) + 1
    mMsz = np.floor_divide(original_shape, 2) + 1
    shift = (mvec - mMsz).astype(int)
    
    # Compute pad widths for each dimension: pad only at the end ('post') side.
    pad_width = [(0, int(fs - os)) for os, fs in zip(original_shape, final_size)]
    
    # If the input has more dimensions than those we pad, pad the remaining dimensions with (0,0).
    extra_dims = len(input_array.shape) - current_dim
    if extra_dims > 0:
        pad_width.extend([(0, 0)] * extra_dims)
    
    # Pad the input array with zeros.
    padded = np.pad(input_array, pad_width, mode='constant')
    
    # Use np.roll to circularly shift the padded array to center the original array.
    output = padded
    for axis, shift_val in enumerate(shift):
        output = np.roll(output, shift=shift_val, axis=axis)
    
    return output
