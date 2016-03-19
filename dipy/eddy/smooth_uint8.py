"""smoothing function for images to convolve a uint8 volume with a smoothing kernel (fwhm in voxels).
"""
from __future__ import division, absolute_import, print_function


import numpy as np
import nibabel as nib


def smooth_uint8(img, lowerclip=0.4, upper_clip=0.99, rescale_flag=False):
    r"""
    Clips the input array values (img) so that the range of values will be
    between the pixel-count proportions specified by lowerClip and upper_clip.

    Eg, if lower_clip = 0.20 and upper_clip = 0.98, then the values in img will
    be clipped so that the lowest value is that of the 20th percentile of the
    original values and the upper value is that of the 98th
    percentile.

    If rescale_flag is true, the intensities will be scaled to
    0-1. Otherwise, the original (but clipped) intensity range will
    be preserved.

    HISTORY: 2004.11.05 RFD: wrote it.
    Parameters
    ----------
    img : numpy.ndarray
        numpy array containing the image data obtained from 
        img = nib.load(fimg)
        where fimg is the filename of the image

    lower_clip : float
        lower value for clipping
    upper_clip : float
        upper value for clipping

    Returns
    -------
    clipvals : array of two elements 
        lower clip value and upper clip value used

    img : numpy.ndarray
        numpy array containing the clipped image data

    References :
    https://github.com/vistalab/vistasoft/blob/master/mrAnatomy/VolumeUtilities/mrAnatHistogramClip.m
    ----------

    Examples
    --------

    >>>fimg, fbval, fbvec = get_data('small_101D')
    >>>bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    >>>img = nib.load(fimg) 
    >>>img_clipped, clipvals= hist_clip(img.get_data(), 0.4, 0.99, rescale_flag=True)
    """
