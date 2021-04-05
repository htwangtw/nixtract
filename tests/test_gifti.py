"""Integration tests for GIFTI extractions

Check to ensure that the data being extracted lines up with the labels defined
in the provided roi files. The approach is fairly straightforward:
    1. Take an annot file a) make label.gii version and b) and duplicate the 
       data (10 times) so that it creates a mock func.gii file. This is 
       performed once per user by `setup_mock_data.py`, which is ran by the 
       `mock_data` pytest fixture. 
    2. Run nixtract-gifti on the mock func.gii with the annot or label.gii as 
       the roi_file
    3. Validate extracted timeseries against the expected array, which has 10
       rows (number of timepoints) and N columns, where N = number of labels 
       in dlabel (excluding 0). Each column should only have elements that 
       equal the label number (i.e. column 1 should be full of 1's).

These approach is applied to both hemispheres, as well as each separately.

A very similar approach is used to check the `--as_vertcies` functionality, 
where 1) all values in the timeseries should equal the label, and 2) the number
of columns should equal the number of vertices with that label in the annot or
func.gii file. Binary masks for these tests are created in `setup_mock_data.py`.

The Schaefer atlas (100 region, 7 networks) in fsaverage5-space is used. 

Additional checks where scans are discarded and regressors are used are also
performed, which are some basic functionalities of `GiftiExtractor`.
"""
import os
import pytest
import subprocess
import numpy as np
import nibabel as nib
import pandas as pd


def test_annot(data_dir, mock_data, tmpdir):
    
    schaef = '.Schaefer2018_100Parcels_7Networks_order.annot'
    lh_annot = os.path.join(data_dir, 'lh' + schaef)
    rh_annot = os.path.join(data_dir, 'rh' + schaef)

    lh_func = os.path.join(mock_data, 'schaefer_hemi-L.func.gii')
    rh_func = os.path.join(mock_data, 'schaefer_hemi-R.func.gii')

    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} --rh_files {rh_func} "
           f"--lh_roi_file {lh_annot} --rh_roi_file {rh_annot}")
    subprocess.run(cmd.split())

    actual = pd.read_table(os.path.join(tmpdir, 'schaefer_hemi-LR_timeseries.tsv'))
    expected_hemi = np.tile(np.arange(1, 51), (10, 1))
    expected = np.concatenate([expected_hemi, expected_hemi], axis=1)

    assert np.array_equal(actual.values, expected)


def test_gifti_label(mock_data, tmpdir):
    schaef = '.Schaefer2018_100Parcels_7Networks_order.annot'
    lh_label = os.path.join(mock_data, 'schaefer_hemi-L.label.gii')
    rh_label = os.path.join(mock_data, 'schaefer_hemi-R.label.gii')

    lh_func = os.path.join(mock_data, 'schaefer_hemi-L.func.gii')
    rh_func = os.path.join(mock_data, 'schaefer_hemi-R.func.gii')

    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} --rh_files {rh_func} "
           f"--lh_roi_file {lh_label} --rh_roi_file {rh_label}")
    subprocess.run(cmd.split())

    actual = pd.read_table(os.path.join(tmpdir, 'schaefer_hemi-LR_timeseries.tsv'))
    expected_hemi = np.tile(np.arange(1, 51), (10, 1))
    expected = np.concatenate([expected_hemi, expected_hemi], axis=1)

    assert np.array_equal(actual.values, expected)


# def test_annot_mask():
#     pass


# def test_annot_mask_vertices():
#     pass


# def test_gifti_label_mask():
#     pass


# def test_gifti_label_mask_vertices():
#     pass


# def test_discard_scans():
#     pass


# def test_regressors():
#     pass