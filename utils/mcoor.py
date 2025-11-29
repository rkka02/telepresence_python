def mcoor(matSize):
    """
    mcoor: Computes the center coordinate index for a given size.
    
    For a scalar value representing the size (e.g., number of pixels),
    returns the index of the center computed as floor(matSize/2) + 1.
    
    Parameters:
      matSize : int
          Size of the dimension (e.g., number of pixels).
    
    Returns:
      int
          The center coordinate index.
    """
    return (matSize // 2) + 1
