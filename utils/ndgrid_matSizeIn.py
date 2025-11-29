import numpy as np

def ndgrid_matSizeIn(sizeMat, normalizeTF=0, meshType='default'):
    """
    ndgrid_matSizeIn: Generates N-dimensional coordinate grids based on a given size vector.
    
    Parameters:
      sizeMat     : sequence of ints
          A vector specifying the size in each dimension, e.g. [5, 6, 7, 8].
      normalizeTF : bool or str, optional (default=0)
          If a boolean, False (default) or True, determines whether to normalize the coordinates
          by the size in each dimension.
          If a string, must be either 'default' (no normalization) or 'normalized'.
      meshType    : str, optional (default='default')
          Specifies how to generate the coordinate vectors. Must be one of:
             'default'             : Regular grid with coordinates 1:sizeMat[d]
             'centerZero'          : Coordinates shifted so that the center (floor(n/2)+1)
                                     becomes zero.
             'centerZero_ifftshift': Similar to 'centerZero', but using symmetric ordering
                                     (compatible with ifftshift).
    
    Returns:
      outGridCell : tuple of numpy arrays
          Coordinate grids for each dimension.
          
    Example:
      sizeMat = [5, 6, 7, 8]
      gridCell = ndgrid_matSizeIn(sizeMat, 'normalized', 'centerZero')
    """
    # Ensure sizeMat is a 1D array of ints.
    sizeMat = np.asarray(sizeMat, dtype=int)
    dimN = len(sizeMat)
    
    # Process normalizeTF: if string, interpret accordingly.
    if isinstance(normalizeTF, str):
        if normalizeTF == 'default':
            normalizeTF = False
        elif normalizeTF == 'normalized':
            normalizeTF = True
        else:
            raise ValueError("normalizeTF must be either a boolean or 'default'/'normalized'")
    else:
        normalizeTF = bool(normalizeTF)
    
    # Validate meshType.
    if meshType not in ['default', 'centerZero', 'centerZero_ifftshift']:
        raise ValueError("meshType must be 'default', 'centerZero', or 'centerZero_ifftshift'")
    
    # Generate linear coordinate vectors.
    linearVecCell = []
    if meshType == 'default':
        # Coordinates from 1 to n.
        for dd in range(dimN):
            linearVecCell.append(np.arange(1, sizeMat[dd] + 1))
    elif meshType == 'centerZero':
        # Shift coordinates so that center becomes zero.
        for dd in range(dimN):
            dd_size = sizeMat[dd]
            mdd_size = dd_size // 2 + 1  # Center index (MATLAB: floor(n/2)+1)
            # Create vector: (1,2,...,n) shifted by mdd_size.
            linearVecCell.append(np.arange(1, dd_size + 1) - mdd_size)
    elif meshType == 'centerZero_ifftshift':
        # Create vector that, when ifftshift is applied, centers the zero frequency.
        for dd in range(dimN):
            dd_size = sizeMat[dd]
            mdd_size = dd_size // 2 + 1
            # First part: 0 to (n - mdd_size)
            first_part = np.arange(0, dd_size - mdd_size + 1)
            # Second part: (1 - mdd_size) to -1.
            second_part = np.arange(1 - mdd_size, 0)
            vec = np.concatenate((first_part, second_part))
            linearVecCell.append(vec)
    
    # Normalize coordinates if requested.
    if normalizeTF:
        for dd in range(dimN):
            linearVecCell[dd] = linearVecCell[dd] / sizeMat[dd]
    
    # Generate the N-dimensional grid using np.meshgrid with 'ij' indexing.
    outGridCell = np.meshgrid(*linearVecCell, indexing='ij')
    return outGridCell