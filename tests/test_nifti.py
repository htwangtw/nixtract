"""Integration tests (and some unit tests) for NIFTI extractions

Check to ensure that the data being extracted lines up with the labels defined
in the provided roi files. The approach is fairly straightforward:
    1. Take an 3D NIFTI image and duplicate the data (10 times) so that it 
       creates a mock 4D functional image. This is performed once per user by 
       `setup_mock_data.py`, which is ran by the `mock_data` pytest fixture. 
    2. Run nixtract-nifti on the mock 4D image with the 3D image as the 
       roi_file
    3. Validate extracted timeseries against the expected array, which has 10
       rows (number of timepoints) and N columns, where N = number of labels 
       in dlabel (excluding 0). Each column should only have elements that 
       equal the label number (i.e. column 1 should be full of 1's).

A very similar approach is used to check the `--as_voxels` functionality, 
where 1) all values in the timeseries should equal the label, and 2) the number
of columns should equal the number of voxels with that label in the 3D image.
Binary masks for these tests are created in `setup_mock_data.py`.

Coordinate extraction is tested by ensuring that each coordinate (radius=None) 
equals the correct label. Note that different radii aren't tested directly, 
as this is passed into nilearn and is already tested there.

The underlying nilearn masker object is also checked via 
`test_set_volume_masker`. This will ensure that masker-specific agruments 
(e.g., `mask_img`, `radius`) are passed to the correct masker. 

The Schaefer atlas (100 region, 7 networks) is used. 
 
Additional checks where scans are discarded and regressors are used are also
performed, which are some basic functionalities of `NiftiExtractor`.
"""
import os
import pytest
import subprocess
import numpy as np
import nibabel as nib

from nixtract.extractors.nifti_extractor import _set_volume_masker
from nilearn.input_data import (NiftiLabelsMasker, NiftiMasker, 
                                NiftiSpheresMasker)


def test_set_volume_masker(data_dir, mock_data):


    mask = os.path.join(mock_data, 'schaefer_LH_Vis_4.nii.gz')
    atlas = os.path.join(
        data_dir, 
        'Schaefer2018_100Parcels_7Networks''_order_FSLMNI152_2mm.nii.gz'
    )
    coordinates = os.path.join(
        data_dir,
        'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
    )

    masker, n_rois = _set_volume_masker(mask, as_voxels=True)
    assert isinstance(masker, NiftiMasker)
    assert n_rois == 1

    masker, n_rois = _set_volume_masker(mask, )
    assert isinstance(masker, NiftiLabelsMasker)
    assert n_rois == 1

    masker, n_rois = _set_volume_masker(atlas)
    assert isinstance(masker, NiftiLabelsMasker)
    assert n_rois == 100

    masker, n_rois = _set_volume_masker(coordinates)
    assert isinstance(masker, NiftiSpheresMasker)
    assert n_rois == 100

    


# def test_mask():
#     pass


# def test_mask_voxels():
#     pass


# def test_label_atlas():
#     pass


# def test_coord_atlas():
#     pass


# def test_gifti_label():
#     pass


# def test_annot_mask():
#     pass


# def test_discard_scans():
#     pass


# def test_regressors():
#     pass