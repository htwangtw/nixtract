"""Common arguments to all CLIs"""

import os
import sys
import argparse
import glob
import json
from itertools import repeat
import multiprocessing

# import for version reporting
from platform import python_version
import nilearn
import nibabel
import scipy
import sklearn
import numpy
import pandas
import natsort
import pkg_resources  # for nixtract itself

def base_cli(parser):
    """Generate CLI with arguments shared among all interfaces"""
    
    parser.add_argument('out_dir', type=str,
                        help='The path to the output directory. Created if it'
                             'does not already exist')
    parser.add_argument('--regressor_files', nargs='+', type=str,
                        help='One or more tabular files with regressors in each '
                             'column. The number of files must match the number '
                             'of input files and must be in the '
                             'same order. The number of rows in each file must '
                             'match the number of timepoints in their '
                             'respective input files. Can also be a single '
                             'string with a wildcard (*) to specify all '
                             'files matching the file pattern. If so, these '
                             'files are naturally sorted by file name prior to '
                             'extraction. Double check to make sure these are '
                             'correctly aligned with the input files.')
    parser.add_argument('--regressors', nargs='+', type=str,
                        help='Regressor names or strategy to use for confound '
                             'regression. Must be a) list of specified column '
                             'names in all of the regressor_files, b) a '
                             'predefined strategy by load_confounds, or c) a '
                             'list compatible with load_confounds flexible '
                             'denoising strategy options. See the documentation '
                             'https://github.com/SIMEXP/load_confounds. If no '
                             'regressor information provided, then all '
                             'regressors in regressor files are used')                    
    parser.add_argument('--standardize', action='store_true', default=False,
                        help='Whether to standardize (z-score) each timeseries. '
                             'Default: False')
    parser.add_argument('--t_r', type=int, 
                        help='The TR of the functional files, specified in '
                             'seconds. Must be included if temporal filtering '
                             'is specified. Default: None')
    parser.add_argument('--high_pass', type=float, 
                        help='High pass filter cut off in Hertz. Default: None')
    parser.add_argument('--low_pass', type=float, 
                        help='Low pass filter cut off in Hertz. Default: None')
    parser.add_argument('--detrend', action='store_true', default=False,
                        help='Temporally detrend the data. Default: None')
    parser.add_argument('--discard_scans', type=int, 
                        help='Discard the first N scans of each functional '
                             'image. Default: None')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='The number of CPUs to use if parallelization is '
                             'desired. Default: 1 (serial processing)')
    parser.add_argument('--config', type=str.lower,
                        help='Configuration .json file as an alternative to '
                             'command-line arguments. See online documentation '
                             'for formatting and what keys to include')
    return parser


def empty_to_none(x):
    """Replace an empty list from params with None"""
    if isinstance(x, list):
        if not x:
            x = None
    return x


def merge_params(params, config):
    """Merge CLI params with configuration file params. Configuration params 
    will overwrite the CLI params.
    """
    return {**params, **config}


def check_glob(x):
    """Run globbing if applicable"""
    if isinstance(x, str):
        return natsort.natsorted(glob.glob(x))
    elif isinstance(x, list):
        return x
    else:
        raise ValueError('Input data files (images and confounds) must be a'
                         'string or list of string')


def handle_base_args(params):
    """Check the validity of base CLI arguments"""

    # read config file if available -- overwrites CLI
    if params['config'] is not None:
        with open(params['config'], 'rb') as f:
            conf_params = json.load(f)
        params = merge_params(params, conf_params)
    params.pop('config')

    params['regressor_files'] = empty_to_none(params['regressor_files'])
    if params['regressor_files'] is not None:
        params['regressor_files'] = check_glob(params['regressor_files'])

    # coerce to list in case a string is provided by config file
    params['regressors'] = empty_to_none(params['regressors'])
    if isinstance(params['regressors'], str):
        params['regressors'] = [params['regressors']]

    # make output dirs
    os.makedirs(params['out_dir'], exist_ok=True)
    os.makedirs(os.path.join(params['out_dir'], 'nixtract_data'),
                exist_ok=True)

    return params


def _get_package_versions():
     """Get dependency versions for metadata file created by CLI"""
     versions = {
          'python': python_version(),
          'nixtract': pkg_resources.require("nixtract")[0].version,
          'numpy': numpy.__version__,
          'scipy': scipy.__version__,
          'pandas': pandas.__version__,
          'scikit-learn': sklearn.__version__,
          'nilearn': nilearn.__version__,
          'nibabel': nibabel.__version__,
          'natsort': natsort.__version__
     }
     return versions


def make_param_file(params):
    # add in meta data
    versions = _get_package_versions()

    # export command-line call and parameters to a file
    param_info = {'command': " ".join(sys.argv), 'parameters': params,
                  'meta_data': versions}

    metadata_path = os.path.join(params['out_dir'], 'nixtract_data')
    param_file = os.path.join(metadata_path, 'parameters.json')

    with open(param_file, 'w') as fp:
        json.dump(param_info, fp, indent=2)
    return metadata_path


def replace_file_ext(fname):
    for ext in ['.nii', '.nii.gz', '.func.gii', '.dtseries.nii']:
        if fname.endswith(ext):
            return os.path.basename(fname).replace(ext, '_timeseries.tsv')
            

def run_extraction(extract_func, input_files, roi_file, regressor_files, 
                   params):


    if regressor_files is None:
        regressor_files = [regressor_files] * len(input_files)

    n_jobs = params['n_jobs']
    # no parallelization
    if n_jobs == 1:
        for i, in_file in enumerate(input_files):
            extract_func(in_file, roi_file, regressor_files[i], params)
    else:
        # repeat parameters are held constant for all parallelized iterations
        args = zip(
            input_files,
            roi_file,
            regressor_files, 
            repeat(params)
        )
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pool.starmap(extract_func, args)