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
import json
import subprocess
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import signal

def test_annot(data_dir, mock_data, tmpdir):
    
    schaef = '.Schaefer2018_100Parcels_7Networks_order.annot'
    lh_annot = os.path.join(data_dir, 'lh' + schaef)
    rh_annot = os.path.join(data_dir, 'rh' + schaef)

    lh_func = os.path.join(mock_data, 'schaefer_hemi-L.func.gii')
    rh_func = os.path.join(mock_data, 'schaefer_hemi-R.func.gii')

    # BOTH HEMISPHERES
    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} --rh_files {rh_func} "
           f"--lh_roi_file {lh_annot} --rh_roi_file {rh_annot}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-LR_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.tile(np.arange(1, 51), (10, 1))
    expected = np.concatenate([expected_hemi, expected_hemi], axis=1)
    assert np.array_equal(actual, expected)

    # LEFT HEMISPHERE
    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} "
            f"--lh_roi_file {lh_annot}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-L_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.tile(np.arange(1, 51), (10, 1))
    assert np.array_equal(actual, expected_hemi)

    # RIGHT HEMISPHERE
    cmd = (f"nixtract-gifti {tmpdir} --rh_files {rh_func} "
            f"--rh_roi_file {rh_annot}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-R_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.tile(np.arange(1, 51), (10, 1))
    assert np.array_equal(actual, expected_hemi)


def test_gifti_label(mock_data, tmpdir):

    lh_label = os.path.join(mock_data, 'schaefer_hemi-L.label.gii')
    rh_label = os.path.join(mock_data, 'schaefer_hemi-R.label.gii')

    lh_func = os.path.join(mock_data, 'schaefer_hemi-L.func.gii')
    rh_func = os.path.join(mock_data, 'schaefer_hemi-R.func.gii')

    # BOTH HEMISPHERES
    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} --rh_files {rh_func} "
           f"--lh_roi_file {lh_label} --rh_roi_file {rh_label}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-LR_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.tile(np.arange(1, 51), (10, 1))
    expected = np.concatenate([expected_hemi, expected_hemi], axis=1)
    assert np.array_equal(actual, expected)

    # LEFT HEMISPHERE
    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} "
            f"--lh_roi_file {lh_label}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-L_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.tile(np.arange(1, 51), (10, 1))
    assert np.array_equal(actual, expected_hemi)

    # RIGHT HEMISPHERE
    cmd = (f"nixtract-gifti {tmpdir} --rh_files {rh_func} "
            f"--rh_roi_file {rh_label}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-R_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.tile(np.arange(1, 51), (10, 1))
    assert np.array_equal(actual, expected_hemi)


def test_annot_mask(mock_data, tmpdir):
    
    # LEFT HEMISPHERE
    lh_annot = os.path.join(mock_data, 'schaefer_LH_Vis_4.annot')
    lh_func = os.path.join(mock_data, 'schaefer_hemi-L.func.gii')

    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} "
            f"--lh_roi_file {lh_annot}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-L_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.full((10, 1), fill_value=4)
    assert np.array_equal(actual, expected_hemi)

    # RIGHT HEMISPHERE
    rh_annot = os.path.join(mock_data, 'schaefer_RH_Vis_4.annot')
    rh_func = os.path.join(mock_data, 'schaefer_hemi-R.func.gii')

    cmd = (f"nixtract-gifti {tmpdir} --lh_files {rh_func} "
            f"--lh_roi_file {rh_annot}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-R_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.full((10, 1), fill_value=4)
    assert np.array_equal(actual, expected_hemi)


def test_annot_mask_vertices(mock_data, tmpdir):
    
    lh_annot = os.path.join(mock_data, 'schaefer_LH_Vis_4.annot')
    lh_func = os.path.join(mock_data, 'schaefer_hemi-L.func.gii')

    # LEFT HEMISPHERE
    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} "
            f"--lh_roi_file {lh_annot} --as_vertices")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-L_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    vertices = nib.freesurfer.read_annot(lh_annot)[0]
    n_vertices = len(vertices[vertices.astype(bool)])

    expected_hemi = np.full((10, n_vertices), fill_value=4)
    assert np.array_equal(actual, expected_hemi)

    rh_annot = os.path.join(mock_data, 'schaefer_RH_Vis_4.annot')
    rh_func = os.path.join(mock_data, 'schaefer_hemi-R.func.gii')

    cmd = (f"nixtract-gifti {tmpdir} --rh_files {rh_func} "
            f"--rh_roi_file {rh_annot} --as_vertices")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-R_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    vertices = nib.freesurfer.read_annot(rh_annot)[0]
    n_vertices = len(vertices[vertices.astype(bool)])

    expected_hemi = np.full((10, n_vertices), fill_value=4)
    assert np.array_equal(actual, expected_hemi)


def test_label_mask(mock_data, tmpdir):
    
    # LEFT HEMISPHERE
    lh_label = os.path.join(mock_data, 'schaefer_LH_Vis_4.label.gii')
    lh_func = os.path.join(mock_data, 'schaefer_hemi-L.func.gii')

    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} "
            f"--lh_roi_file {lh_label}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-L_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.full((10, 1), fill_value=4)
    assert np.array_equal(actual, expected_hemi)

    # RIGHT HEMISPHERE
    rh_label = os.path.join(mock_data, 'schaefer_RH_Vis_4.label.gii')
    rh_func = os.path.join(mock_data, 'schaefer_hemi-R.func.gii')

    cmd = (f"nixtract-gifti {tmpdir} --rh_files {rh_func} "
            f"--rh_roi_file {rh_label}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-R_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.full((10, 1), fill_value=4)
    assert np.array_equal(actual, expected_hemi)


def test_label_mask_vertices(mock_data, tmpdir):
    # LEFT HEMISPHERE
    lh_label = os.path.join(mock_data, 'schaefer_LH_Vis_4.label.gii')
    lh_func = os.path.join(mock_data, 'schaefer_hemi-L.func.gii')

    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} "
            f"--lh_roi_file {lh_label} --as_vertices")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-L_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    vertices = nib.load(lh_label).agg_data().ravel()
    n_vertices = len(vertices[vertices.astype(bool)])

    expected_hemi = np.full((10, n_vertices), fill_value=4)
    assert np.array_equal(actual, expected_hemi)

    # RIGHT HEMISPHERE
    rh_label = os.path.join(mock_data, 'schaefer_RH_Vis_4.label.gii')
    rh_func = os.path.join(mock_data, 'schaefer_hemi-R.func.gii')

    cmd = (f"nixtract-gifti {tmpdir} --rh_files {rh_func} "
            f"--rh_roi_file {rh_label} --as_vertices")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-R_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    vertices = nib.load(rh_label).agg_data().ravel()
    n_vertices = len(vertices[vertices.astype(bool)])

    expected_hemi = np.full((10, n_vertices), fill_value=4)
    assert np.array_equal(actual, expected_hemi)
    

def test_discard_scans(mock_data, tmpdir):

    lh_label = os.path.join(mock_data, 'schaefer_hemi-L.label.gii')
    rh_label = os.path.join(mock_data, 'schaefer_hemi-R.label.gii')

    lh_func = os.path.join(mock_data, 'schaefer_hemi-L.func.gii')
    rh_func = os.path.join(mock_data, 'schaefer_hemi-R.func.gii')

    # BOTH HEMISPHERES
    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} --rh_files {rh_func} "
           f"--lh_roi_file {lh_label} --rh_roi_file {rh_label} "
           "--discard_scans 3")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-LR_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.tile(np.arange(1, 51), (7, 1))
    expected = np.concatenate([expected_hemi, expected_hemi], axis=1)
    assert np.array_equal(actual, expected)


def test_regressors(mock_data, tmpdir, basic_regressor_config):

    config_file = os.path.join(tmpdir, 'config.json')
    with open(config_file, 'w') as fp:
        json.dump(basic_regressor_config, fp)
    
    lh_label = os.path.join(mock_data, 'schaefer_hemi-L.label.gii')
    rh_label = os.path.join(mock_data, 'schaefer_hemi-R.label.gii')

    lh_func = os.path.join(mock_data, 'schaefer_hemi-L.func.gii')
    rh_func = os.path.join(mock_data, 'schaefer_hemi-R.func.gii')

    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} --rh_files {rh_func} "
           f"--lh_roi_file {lh_label} --rh_roi_file {rh_label} "
           f"-c {config_file}")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-LR_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.tile(np.arange(1, 51), (10, 1))
    expected = np.concatenate([expected_hemi, expected_hemi], axis=1)
    
    regressors = pd.read_table(basic_regressor_config['regressor_files'], 
                               usecols=basic_regressor_config['regressors'])
    expected = signal.clean(expected, confounds=regressors, standardize=False, 
                            detrend=False, t_r=None)

    # check with discard scans
    cmd = (f"nixtract-gifti {tmpdir} --lh_files {lh_func} --rh_files {rh_func} "
        f"--lh_roi_file {lh_label} --rh_roi_file {rh_label} "
        f"-c {config_file} --discard_scans 3")
    subprocess.run(cmd.split())

    tseries = os.path.join(tmpdir, 'schaefer_hemi-LR_timeseries.tsv')
    assert os.path.exists(tseries)
    actual = pd.read_table(tseries).values

    expected_hemi = np.tile(np.arange(1, 51), (7, 1))
    expected = np.concatenate([expected_hemi, expected_hemi], axis=1)
    
    regressors = regressors.iloc[3:, :]
    expected = signal.clean(expected, confounds=regressors, standardize=False, 
                            detrend=False, t_r=None)

    assert np.allclose(actual, expected)