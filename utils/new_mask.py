import os
import numpy as np
import nibabel as nib
import warnings

def update_header_and_save(raw_filepath, mask_filepath, new_mask_filepath):
    """
    This function updates the header of a mask image to match a raw image, 
    and then saves the mask image to a new location.

    Parameters:
    raw_filepath (str): The filepath to the raw image.
    mask_filepath (str): The filepath to the mask image.
    new_mask_filepath (str): The filepath where the updated mask image will be saved.

    Returns:
    None
    """

    raw_img = nib.load(raw_filepath)
    mask_img = nib.load(mask_filepath)

    # values to update
    attributes = ['extents', 'dim_info', 'slice_end', 'cal_max', 'cal_min', 
                  'glmax', 'glmin', 'srow_x', 'srow_y', 'srow_z']

    # applying original params to generated maks
    header = mask_img.header.copy()
    for attr in attributes:
        header[attr] = raw_img.header[attr]

    # creating transformation
    new_affine = np.eye(4)
    new_affine[:-1, :] = np.array([header['srow_x'], header['srow_y'], header['srow_z']])

    # saving an img to new folder
    updated_mask_img = nib.Nifti1Image(mask_img.get_fdata(), new_affine, header)
    nib.save(updated_mask_img, new_mask_filepath)