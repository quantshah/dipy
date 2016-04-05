"""cliping function for images
"""
from __future__ import division, absolute_import, print_function

import os
import warnings

import numpy as np
import nibabel as nib
from dipy.data import get_data
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table


def hist_clip(img, lower_clip=0.4, upper_clip=0.99, rescale_flag=False):
    r"""
    Clips the input array values (img) so that the range of values will be
    between the pixel-count proportions specified by lower_Clip and upper_clip.

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
    
    if upper_clip > 1:
        upper_clip = 0.99
        lowerClip = 0.4
    else:
        count, value = np.histogram(img, 256)
        upper_clip_val = value[np.min(
            np.where((np.cumsum(count) / np.sum(count) >= upper_clip)))]
        lower_clip_val = value[np.max(
            np.where((np.cumsum(count) / np.sum(count) <= upper_clip)))]  # have to test division

        if lower_clip_val is None:
            lower_clip_val = value[0]

    img[img > upper_clip_val] = upper_clip_val
    img[img < lower_clip_val] = lower_clip_val

    if rescale_flag:
        img = img - lower_clip_val
        img = img / (upper_clip_val - lower_clip_val)
    clipvals = [lower_clip_val, upper_clip_val]
    return img, clipvals


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
    return 0
