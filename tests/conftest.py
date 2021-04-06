import os
import pytest
import subprocess
import pandas as pd


@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def data_dir(rootdir):
    return os.path.join(rootdir, 'data')


@pytest.fixture
def config_dir(rootdir):
    return os.path.join(rootdir, 'configs')


@pytest.fixture
def mock_data(rootdir, data_dir):
    """Build mock functional data from available atlases"""

    mock_dir = os.path.join(data_dir, 'mock')
    if not os.path.exists(mock_dir):
        subprocess.run("python setup_mock_data.py".split(), cwd=rootdir)
    
    return mock_dir

@pytest.fixture
def basic_regressor_config(data_dir):
    config = {
        'regressor_files': os.path.join(data_dir, 'example_regressors.tsv'),
        'regressors': [
            'trans_x',
            'trans_y',
            'trans_z',
            'rot_x',
            'rot_y',
            'rot_z'
        ]
    }
    return config

@pytest.fixture
def nifti_label_config(data_dir):
    schaef = 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_XYZ.tsv'
    labels = pd.read_table(os.path.join(data_dir, schaef), 
                           usecols=['ROI Name'])
    config = {
        'labels': labels['ROI Name'].tolist()
    }
    return config