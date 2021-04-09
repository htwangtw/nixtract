"""Functions for command line interface
"""
import argparse
import os
import shutil
from nilearn.signal import clean

from nixtract.cli.base import (base_cli, handle_base_args, replace_file_ext,
                               make_param_file, check_glob, run_extraction)
from nixtract.extractors import CiftiExtractor

def _cli_parser():
    """Reads CIFTI CLI arguments and returns input specifications combined with
    those from the general CLI
    """
    parser = argparse.ArgumentParser()
    # input files
    parser.add_argument('--input_files', nargs='+', type=str,
                        help='One or more input CIFTI dtseries files '
                             '(.dtseries.nii). Can also be a single string '
                             'with wildcards (*) to specify all files matching '
                             'the file pattern. If so, these files are '
                             'naturally sorted by file name prior to '
                             'extraction')
    parser.add_argument('--roi_file', type=str, 
                        help='CIFTI dlabel file (.dlabel.nii) with one or more '
                             'labels')
    # other
    parser.add_argument('--as_vertices', default=False,
                        action='store_true',
                        help='Extract the timeseries of each vertex in a '
                             'a region rather than the mean timeseries.This is '
                             'only available for when `roi_file` is single '
                             'region, i.e. a binary mask. Default: False')
    parser.add_argument('--denoise-pre-extract', default=False,
                        action='store_true',
                        help='Denoise data (e.g., filtering, confound '
                             'regression) before timeseries extraction. '
                             'Otherwise, denoising is done on the extracted '
                             'timeseries, which is consistent with nilearn and '
                             'is more computationally efficient. Default: False')                  
    parser = base_cli(parser)                         
    return parser.parse_args()


def _check_cifti_params(params):
    """Ensure that required fields are included and correctly formatted"""
    params = handle_base_args(params)

    if params['input_files'] is None:
        raise ValueError('Missing input_files. Check files')
    else:
        params['input_files'] = check_glob(params['input_files'])
        # glob returned nothing
        if not params['input_files']:
            raise ValueError('Missing input files. Check files')

    if not params['roi_file']:
        raise ValueError('Missing roi_file input.')
    
    return params


def extract_cifti(input_file, roi_file, regressor_file, params):
    """Extract timeseries from a CIFTI image

    Parameters
    ----------
    input_files : str
        File path of the input .dtseries.nii file
    roi_file : str
        File path of the input .dlabel.nii file. 
    regressor_file : str
        File path of regressor file
    params : dict
        Parameter dictionary for extraction
    """
    extractor = CiftiExtractor(
        fname=input_file, 
        roi_file=roi_file,
        as_vertices=params['as_vertices'],
        verbose=params['verbose'],
        pre_clean=params['denoise_pre_extract'],
        standardize=params['standardize'], 
        t_r=params['t_r'], 
        high_pass=params['high_pass'], 
        low_pass=params['low_pass'], 
        detrend=params['detrend']
    )
    if regressor_file is not None:
        extractor.set_regressors(regressor_file, params['regressors'], 
                                 params["load_confounds_kwargs"])

    if (params['discard_scans'] is not None) and (params['discard_scans'] > 0):
        extractor.discard_scans(params['discard_scans'])

    extractor.extract()
    out = os.path.join(params['out_dir'], replace_file_ext(input_file))
    extractor.save(out, params['n_decimals'])

    return out, extractor
    

def main():
    params = vars(_cli_parser())
    params = _check_cifti_params(params)
    metadata_path = make_param_file(params)
    shutil.copy2(params['roi_file'], metadata_path)

    run_extraction(extract_cifti, params['input_files'], params['roi_file'], 
                   params)


if __name__ == '__main__':
    raise RuntimeError("`nixtract/cli/cifti.py` should not be run directly. "
                       "Please `pip install` nixtract and use the "
                       "`nixtract-cifti` command.")

