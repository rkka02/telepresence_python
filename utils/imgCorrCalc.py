import numpy as np

def imgCorrCalc(img1, img2):
    """
    Compute the normalized correlation between two images.

    Parameters:
      img1 : numpy array
          First image (any numeric array).
      img2 : numpy array
          Second image (same shape as img1).

    Returns:
      corrOut : float
          A scalar representing the normalized correlation (between 0 and 1).
    """
    # Flatten images to vectors.
    v1 = img1.flatten()
    v2 = img2.flatten()
    
    # Compute dot product and normalize by the Euclidean norms.
    corrOut = np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return corrOut