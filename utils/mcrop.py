import numpy as np

def mcrop(img, outputSize):
    """
    mcrop: Centrally crop an image (or volume) to a specified output size.
    
    Parameters:
      img : numpy array
          Input image or volume (1D, 2D, or 3D).
      outputSize : sequence of ints
          Desired output dimensions as [yy, xx, (zz)].
          If img has more dimensions than outputSize, the extra dimensions are preserved.
    
    Returns:
      cimg : numpy array
          The centrally cropped image.
    
    Raises:
      ValueError: if the input dimensions are incompatible with outputSize or
                  if the requested crop is larger than the input.
    """
    inputSize = np.array(img.shape)
    inputDim = len(inputSize)
    outputSize = np.array(outputSize, dtype=int)
    outputDim = len(outputSize)
    
    # Ensure the input has at least as many dimensions as outputSize.
    if inputDim < outputDim:
        raise ValueError("Dimensions of input image must be larger than given dimension")
    elif inputDim > outputDim:
        # For dimensions beyond outputDim, keep the full size.
        outputSize = np.concatenate([outputSize, inputSize[outputDim:]])
    
    # Check that the requested crop size is not larger than the input size.
    if np.any(outputSize > inputSize):
        raise ValueError("Final image size must be smaller than initial image size")
    
    # Limit the function to 1D, 2D, or 3D data.
    if inputDim > 3:
        raise ValueError("This function covers up to 3-D")
    
    # Compute cropping indices for each dimension.
    # In MATLAB, center index = floor(n/2)+1; for Python (0-indexed), use n//2.
    slices = []
    for kk in range(inputDim):
        n = inputSize[kk]
        # Center index in Python (0-indexed)
        center = n // 2  
        # Compute half the desired crop size (rounded)
        half_crop = int(round((outputSize[kk] - 1) / 2.0))
        start = center - half_crop
        end = start + outputSize[kk]
        slices.append(slice(start, end))
    
    # Perform the cropping.
    cimg = img[tuple(slices)]
    return cimg