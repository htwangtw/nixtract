"""Integration tests for CIFTI extractions

Check to ensure that the data being extracted lines up with the labels defined
in the provided roi files. The approach is fairly straightforward:
    1. Take a dlabel file and duplicate the data (10 times) so that it creates
       a mock dtseries file. This is performed once per user by 
       `setup_mock_data.py`, which is ran by the `mock_data` pytest fixture. 
    2. Run nixtract-cifti on the mock dtseries with the dlabel as the roi_file
    3. Validate extracted timeseries against the expected array, which has 10
       rows (number of timepoints) and N columns, where N = number of labels 
       in dlabel (excluding 0). Each column should only have elements that 
       equal the label number (i.e. column 1 should be full of 1's).

Note that the validation data is exactly what would be extracted by 
`wb_command -cifti-parcellate`, so this validates `nixtract-cifti` against
connectome workbench as well.  

A very similar approach is used to check the `--as_vertcies` functionality, 
where 1) all values in the timeseries should equal the label, and 2) the number
of columns should equal the number of vertices with that label in the dlabel 
file. Binary masks for these tests are created in `setup_mock_data.py`.

The Gordon atlas and Schaefer atlas (100 region, 7 networks) are used. 
Importantly, the Schaefer atlas is also used to check the accuracy of internal 
alignment that nixtract-cifti performs; Schaefer dlabel files have ~64k 
vertices because they include the medial wall, and this is tested against a
mock dtseries of Schaefer labels with conventional ~91k vertices (see the 
setup script).  

Additional checks where scans are discarded and regressors are used are also
performed, which are some basic functionalities of `CiftiExtractor`.
"""
import os
import subprocess
import json
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import signal

def test_aligned_extraction(data_dir, mock_data, tmpdir):

    dtseries = os.path.join(mock_data, 'gordon.dtseries.nii')
    roi_file = os.path.join(data_dir, 
                            'Gordon333_FreesurferSubcortical.32k_fs_LR.dlabel.nii')

    cmd = (f"nixtract-cifti {tmpdir} --input_files {dtseries} "
           f"--roi_file {roi_file}")
    subprocess.run(cmd.split())

    actual = pd.read_table(os.path.join(tmpdir, 'gordon_timeseries.tsv'))
    expected = np.tile(np.arange(1, 353), (10, 1))
    assert np.array_equal(actual.values, expected)
    

def test_unaligned_extraction(data_dir, mock_data, tmpdir):

    dtseries = os.path.join(mock_data, 'schaefer_91k.dtseries.nii')
    roi_file = os.path.join(data_dir, 
                            'Schaefer2018_100Parcels_7Networks_order.dlabel.nii')

    cmd = (f"nixtract-cifti {tmpdir} --input_files {dtseries} "
           f"--roi_file {roi_file}")
    subprocess.run(cmd.split())

    actual = pd.read_table(os.path.join(tmpdir, 'schaefer_91k_timeseries.tsv'))
    expected = np.tile(np.arange(1, 101), (10, 1))
    assert np.array_equal(actual.values, expected)


def test_aligned_mask(mock_data, tmpdir):

    dtseries = os.path.join(mock_data, 'gordon.dtseries.nii')
    roi_file = os.path.join(mock_data, 'gordon_L_SMhand_10.dlabel.nii')

    cmd = (f"nixtract-cifti {tmpdir} --input_files {dtseries} "
           f"--roi_file {roi_file}")
    subprocess.run(cmd.split())

    actual = pd.read_table(os.path.join(tmpdir, 'gordon_timeseries.tsv'))
    expected = np.array([[273] * 10]).T
    assert np.array_equal(actual.values, expected)


def test_unaligned_mask(mock_data, tmpdir):

    dtseries = os.path.join(mock_data, 'schaefer_91k.dtseries.nii')
    roi_file = os.path.join(mock_data, 'schaefer_LH_Vis_4.dlabel.nii')

    cmd = (f"nixtract-cifti {tmpdir} --input_files {dtseries} "
           f"--roi_file {roi_file}")
    subprocess.run(cmd.split())

    actual = pd.read_table(os.path.join(tmpdir, 'schaefer_91k_timeseries.tsv'))
    expected = np.array([[4] * 10]).T
    assert np.array_equal(actual.values, expected)


def test_aligned_mask_vertices(mock_data, tmpdir):

    dtseries = os.path.join(mock_data, 'gordon.dtseries.nii')
    roi_file = os.path.join(mock_data, 'gordon_L_SMhand_10.dlabel.nii')

    cmd = (f"nixtract-cifti {tmpdir} --input_files {dtseries} "
           f"--roi_file {roi_file} --as_vertices")
    subprocess.run(cmd.split())
    actual = pd.read_table(os.path.join(tmpdir, 'gordon_timeseries.tsv'))

    # get number of vertices that appear in atlas
    roi_array = nib.load(roi_file).get_fdata()
    n_vertices = len(roi_array[roi_array == 273])
    expected = np.full((10, n_vertices), fill_value=273)
    
    assert np.array_equal(actual.values, expected)


def test_unaligned_mask_vertices(mock_data, tmpdir):

    dtseries = os.path.join(mock_data, 'schaefer_91k.dtseries.nii')
    roi_file = os.path.join(mock_data, 'schaefer_LH_Vis_4.dlabel.nii')

    cmd = (f"nixtract-cifti {tmpdir} --input_files {dtseries} "
           f"--roi_file {roi_file} --as_vertices")
    subprocess.run(cmd.split())
    actual = pd.read_table(os.path.join(tmpdir, 'schaefer_91k_timeseries.tsv'))

    # get number of vertices that appear in atlas
    roi_array = nib.load(roi_file).get_fdata()
    n_vertices = len(roi_array[roi_array == 4])
    expected = np.full((10, n_vertices), fill_value=4)

    assert np.array_equal(actual.values, expected)


def test_discard_scans(data_dir, mock_data, tmpdir):

    dtseries = os.path.join(mock_data, 'gordon.dtseries.nii')
    roi_file = os.path.join(data_dir, 
                            'Gordon333_FreesurferSubcortical.32k_fs_LR.dlabel.nii')

    cmd = (f"nixtract-cifti {tmpdir} --input_files {dtseries} "
           f"--roi_file {roi_file} --discard_scans 3")
    subprocess.run(cmd.split())

    actual = pd.read_table(os.path.join(tmpdir, 'gordon_timeseries.tsv'))
    expected = np.tile(np.arange(1, 353), (7, 1))
    assert np.array_equal(actual.values, expected)


def test_regressors(data_dir, mock_data, basic_regressor_config, tmpdir):

    dtseries = os.path.join(mock_data, 'gordon.dtseries.nii')
    roi_file = os.path.join(data_dir, 
                            'Gordon333_FreesurferSubcortical.32k_fs_LR.dlabel.nii')

    # actual data (all scans = 10)
    config_file = os.path.join(tmpdir, 'config.json')
    with open(config_file, 'w') as fp:
        json.dump(basic_regressor_config, fp)
    cmd = (f"nixtract-cifti {tmpdir} --input_files {dtseries} "
           f"--roi_file {roi_file}  -c {config_file}")
    subprocess.run(cmd.split())
    actual = pd.read_table(os.path.join(tmpdir, 'gordon_timeseries.tsv'))

    # expected data (all scans = 10)
    regressors = pd.read_table(basic_regressor_config['regressor_files'][0], 
                               usecols=basic_regressor_config['regressors'])
    expected = np.tile(np.arange(1, 353), (10, 1)).astype(np.float)
    expected = signal.clean(expected, confounds=regressors, standardize=False, 
                            detrend=False, t_r=None)

    # actual data (discard 3 scans)
    cmd = (f"nixtract-cifti {tmpdir} --input_files {dtseries} "
           f"--roi_file {roi_file} --discard_scans 3 -c {config_file}")
    subprocess.run(cmd.split())
    actual = pd.read_table(os.path.join(tmpdir, 'gordon_timeseries.tsv'))

    # expected data (discard 3 scans)
    regressors = pd.read_table(basic_regressor_config['regressor_files'][0], 
                               usecols=basic_regressor_config['regressors'])
    # discard first three rows to match up with discard scans
    regressors = regressors.values[3:, :]
    expected = np.tile(np.arange(1, 353), (7, 1)).astype(np.float)
    expected = signal.clean(expected, confounds=regressors, standardize=False, 
                            detrend=False, t_r=None)

    assert np.allclose(actual.values, expected)

