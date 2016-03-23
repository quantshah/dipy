"""smoothing function for images to convolve a uint8 volume with 
a smoothing kernel (fwhm in voxels).
"""
from __future__ import division, absolute_import, print_function


import numpy as np
import nibabel as nib


def smooth_uint8(img, fwhm):
    r"""smoothing function for images to convolve a uint8 volume with 
    a smoothing kernel (fwhm in voxels).
    
    ----------
    img : numpy.ndarray
        numpy array containing the image data obtained from 
        img = nib.load(fimg)
        where fimg is the filename of the image

    fwhm

    Returns
    -------

    """
