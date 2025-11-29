import numpy as np

def mk_ellipse(*args):
    """
    mk_ellipse: Create an elliptical (or ellipsoidal) mask.
    
    2D usage:
      H = mk_ellipse(XR, YR, xx, yy)
        - XR: Radius along the x-axis.
        - YR: Radius along the y-axis.
        - xx: Number of columns of the output mask.
        - yy: Number of rows of the output mask.
    
    3D usage:
      H = mk_ellipse(XR, YR, ZR, xx, yy, zz)
        - XR, YR, ZR: Radii along x, y, and z axes.
        - xx, yy, zz: Dimensions of the output volume.
    
    The function returns a binary mask where points outside the ellipse
    (or ellipsoid) are True and points inside are False. If any radius (XR, YR, etc.)
    is not positive, the mask is filled with True.
    
    Raises:
      ValueError: if the number of arguments is not 4 (2D) or 6 (3D).
    """
    if len(args) == 4:
        # 2D case.
        XR, YR, xx, yy = args
        # If either radius is non-positive, return a full True mask.
        if XR <= 0 or YR <= 0:
            return np.ones((yy, xx), dtype=bool)
        
        # Generate coordinate vectors centered at zero.
        # MATLAB: (1:yy) - (floor(yy/2)+1)
        yvec = np.arange(1, yy + 1) - (yy // 2 + 1)
        xvec = np.arange(1, xx + 1) - (xx // 2 + 1)
        # Create a 2D grid; note that meshgrid in Python (with default indexing='xy')
        # returns arrays of shape (yy, xx)
        XX, YY = np.meshgrid(xvec / XR, yvec / YR)
        # Compute the normalized radial distance.
        # (Equivalent to MATLAB's: [~, rho] = cart2pol(XX, YY))
        rho = np.sqrt(XX**2 + YY**2)
        # Points with rho > 1 are outside the ellipse.
        H = rho > 1.0
        return H

    elif len(args) == 6:
        # 3D case.
        XR, YR, ZR, xx, yy, zz = args
        # If any of the key radii are non-positive, return a full True mask.
        if XR <= 0 or YR <= 0 or ZR <= 0:
            return np.ones((yy, xx, zz), dtype=bool)
        
        # Create coordinate grids.
        # We generate coordinates 1..xx, 1..yy, 1..zz.
        x = np.arange(1, xx + 1)
        y = np.arange(1, yy + 1)
        z = np.arange(1, zz + 1)
        # Use indexing='ij' so that XX has shape (xx, yy, zz).
        XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')
        # Define the center for each dimension (MATLAB: floor(dim/2)+1).
        center_x = xx // 2 + 1
        center_y = yy // 2 + 1
        center_z = zz // 2 + 1
        # Compute squared normalized distances from the center.
        mask = (((XX - center_x) / XR) ** 2 +
                ((YY - center_y) / YR) ** 2 +
                ((ZZ - center_z) / ZR) ** 2) > 1.0
        # Note: Depending on your desired output shape ordering, you might want to
        # transpose dimensions. Here, the output shape is (xx, yy, zz) which corresponds
        # to the order used in the meshgrid with indexing='ij'. If you need (yy, xx, zz),
        # adjust accordingly.
        return mask

    else:
        raise ValueError("mk_ellipse accepts either 4 (2D) or 6 (3D) arguments.")
