"""Common arguments to all CLIs"""

import os
import sys
import ast
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
                        help='One or more tab-separated files with regressors '
                             'in each column. The number of files must match '
                             'the number of input files and must be in the '
                             'same order. The number of rows in each file must '
                             'match the number of timepoints in their '
                             'respective input files. Can also be a single '
                             'string with a wildcard (*) to specify all '
                             'files matching the file pattern. If so, these '
                             'files are naturally sorted by file name prior to '
                             'extraction. Double check to make sure these are '
                             'correctly aligned with the input files (see the '
                             'parameters.json in the output)')
    parser.add_argument('--regressors', nargs='+', type=str,
                        help='Regressor names or strategy to use for confound '
                             'regression. Must be a) list of specified column '
                             'names in all of the regressor_files, b) a '
                             'predefined strategy by load_confounds, or c) a '
                             'list compatible with load_confounds flexible '
                             'denoising strategy options. See the documentation '
                             'https://github.com/SIMEXP/load_confounds. If no '
                             'regressor information provided but regressor '
                             'files are provided, then all regressors in '
                             'regressor files are used')
    parser.add_argument('--load_confounds_kwargs', type=str, 
                        help="Keyword arguments for load confound either "
                             "predefined or flexible strategies. Input must be "
                             "a Python dictionary wrapped in double quotes. " 
                             "Refer to documentation for the available "
                             "arguments for each load confound strategy: "
                             "https://github.com/SIMEXP/load_confounds")                    
    parser.add_argument('--standardize', action='store_true', default=False,
                        help='Whether to standardize (z-score) each timeseries. '
                             'Default: False')
    parser.add_argument('--t_r', type=float, 
                        help='The TR of the functional files, specified in '
                             'seconds. Required if temporal '
                             'filtering/detrending is specified')
    parser.add_argument('--high_pass', type=float, 
                        help='High pass filter cut off in Hertz. Do not use if '
                             'high pass cosine regressors are specified in '
                             '`regressors`')
    parser.add_argument('--low_pass', type=float, 
                        help='Low pass filter cut off in Hertz. Do not use if '
                             'low pass cosine regressors are specified in '
                             '`regressors`')
    parser.add_argument('--detrend', action='store_true', default=False,
                        help='Temporally detrend the data. Default: False')
    parser.add_argument('--discard_scans', type=int, 
                        help='Discard the first N scans of each functional '
                             'image')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='The number of CPUs to use if parallelization is '
                             'desired. Default: 1 (serial processing)')
    parser.add_argument('--n_decimals', type=int, 
                        help='Specify the number of decimals for output '
                             'timeseries files. Fewer decimals are recommended '
                             'for reducing disk-space, particularly for large '
                             'extractions')
    parser.add_argument('-c', '--config', type=str,
                        help='A configuration .json file to pass parameters '
                             'This will overwrite command-line arguments if '
                             'the same parameter is specified in both. See '
                             'online documentation for formatting and what '
                             'keys to include')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='Print out extraction progress')
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
    """Get files based on glob pattern

    Parameters
    ----------
    x : str, list
        A glob pattern string or a list of files. If a list, glob pattern 
        matching is not performed.

    Returns
    -------
    list
        List of extracted file names.

    Raises
    ------
    ValueError
       x is neither a string nor list of strings
    """
    if isinstance(x, str):
        return natsort.natsorted(glob.glob(x))
    elif isinstance(x, list):
        return x
    else:
        raise ValueError('Input data files (images and confounds) must be a'
                         'string or list of string')


def _parse_input_str(x):
    """Safely parse input dictionary string for load_confound_kwargs
    
    Only permits strings, numbers, tuples, lists, dicts, booleans, and None.
    """
    try:
        return ast.literal_eval(x)
    except ValueError as e:
        raise ValueError("Invalid dictionary string in load_confound_kwargs")  


def handle_base_args(params):
    """Check the validity of base CLI arguments

    Parameters
    ----------
    params : dict
        Input parameters

    Returns
    -------
    dict
        Validated parameters
    """
    # read config file if available -- overwrites CLI
    if params['config'] is not None:
        with open(params['config'], 'rb') as f:
            try:
                conf_params = json.load(f)
            except json.decoder.JSONDecodeError as e:
                raise ValueError('Invalid .json configuration file') from e
        params = merge_params(params, conf_params)
    params.pop('config')

    params['regressor_files'] = empty_to_none(params['regressor_files'])
    if params['regressor_files'] is not None:
        params['regressor_files'] = check_glob(params['regressor_files'])

    # coerce to list in case a string is provided by config file
    params['regressors'] = empty_to_none(params['regressors'])
    if isinstance(params['regressors'], str):
        params['regressors'] = [params['regressors']]

    if isinstance(params["load_confounds_kwargs"], str):
        params["load_confounds_kwargs"] = _parse_input_str(params["load_confounds_kwargs"])

    # make output dirs
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
    """Generate a parameters.json file to be saved in the output

    Parameters
    ----------
    params : dict
        Input parameters

    Returns
    -------
    str
        Path to which metadata is stored, including parameters.json
    """
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
    """Make a output _timeseries.tsv file based on the input file name

    Parameters
    ----------
    fname : str
        Input functional file

    Returns
    -------
    str
        _timeseries.tsv file to be used for output
    """
    for ext in ['.nii.gz', '.func.gii', '.dtseries.nii']:
        if fname.endswith(ext):
            return os.path.basename(fname).replace(ext, '_timeseries.tsv')


def _make_regressor_file(outputs, out_dir):

    reg_dict = {}
    for fname, extractor in outputs:
        if extractor.regressor_names is not None:
            reg_dict[fname] = list(extractor.regressor_names)

    if len(reg_dict) != 0:
        # check if all extractors used load_confounds
        if all([x[1]._load_confounds for x in outputs]):
            reg_file = os.path.join(out_dir, 'nixtract_data',
                                    'load_confounds_regressors.json')
            with open(reg_file, 'w') as f:
                json.dump(reg_dict, f, indent=2)


def run_extraction(extract_func, input_files, roi_file, params):
    """Extract timeseries from input files

    Extraction is determined by the `extract_func`, in which there is one
    extraction function defined per file type, each defined in their respective
    module (see `extract_func` parameter for each). These functions are roughly
    the same, with extract_gifti being the biggest exception because it has to
    handle left and/or right hemipsheres. 

    Most of the extraction parameters are specified in `params`, including 
    n_jobs, which `run_extraction` uses to parallelize extraction if specified.
    Note that `input_files`, `roi_file`, and `regressor_files` should also all 
    be in params; they are explictly in this function so that it can be 
    generalizeable across file types (thanks to GIFTIs being the way they 
    are...). 
    
    Parameters
    ----------
    extract_func : nixtract.cli.nifti.extract_nifti, 
                   nixtract.cli.gifti.extract_gifti, or
                   nixtract.cli.cifti.extract_cifti
        Extraction function for the file type of input_files
    input_files : list
        List of input files for extraction. If extract_func is 
        nixtract.cli.gifti.extract_gifti, then list must have tuples in which
        each tuple is (left, right) hemisphere input files
    roi_file : str or tuple
        File that defines regions of interest(s). If extract_func is 
        nixtract.cli.gifti.extract_gifti, then must be a tuple containing each
        hemisphere, i.e. (left, right) 
    regressor_files : list
        List of regressor files to pair with input_files. Should be in the 
        same order.
    params : dict
        Input parameter dictionary
    """
    regressor_files = params['regressor_files']
    if regressor_files is None:
        regressor_files = [regressor_files] * len(input_files)
    
    if len(regressor_files) != len(input_files):
        raise ValueError('Number of regressor files do not equal number of input files')

    n_jobs = params['n_jobs']
    # no parallelization
    if n_jobs == 1:
        res = []
        for i, in_file in enumerate(input_files):
            out, extractor = extract_func(in_file, roi_file, 
                                          regressor_files[i], params)
            res.append((out, extractor))
    else:
        args = zip(
            input_files,
            repeat(roi_file),
            regressor_files, 
            repeat(params)
        )
        with multiprocessing.Pool(processes=n_jobs) as pool:
            res = pool.starmap(extract_func, args)

    _make_regressor_file(res, params['out_dir'])
