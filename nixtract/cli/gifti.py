"""Functions for command line interface
"""
import sys
import os
import json
import shutil

from nixtract.extract import extract_gifti, run_extraction
from nixtract.cli.base import (base_cli, handle_base_args, replace_file_ext,
                               make_param_file, check_glob, empty_to_none)
from nixtract.extractors import GiftiExtractor

def _cli_parser():
    """Reads command line arguments and returns input specifications"""
    parser = base_cli()
    # input files
    parser.add_argument('--lh_files', nargs='+', type=str, metavar='lh_files',
                        help='One or more input functional GIfTI images '
                             '(.func.gii) for the left hemisphere. Can also '
                             'be a single string with a wildcard (*) to '
                             'specify all files matching the file pattern. If '
                             'so, these files are naturally sorted by file '
                             'name prior to extraction')
    parser.add_argument('--rh_files', nargs='+', type=str, metavar='rh_files',
                        help='One or more input functional GIfTI images '
                             '(.func.gii) for the right hemisphere. Can also '
                             'be a single string with a wildcard (*) to '
                             'specify all files matching the file pattern. If '
                             'so, these files are naturally sorted by file '
                             'name prior to extraction')
    # roi files
    parser.add_argument('--lh_roi_file', type=str, metavar='roi_file', 
                        help='A label GIfTI image (label.gii) or a Freesurfer '
                             'annotation file (.annot) for the left hemipshere. '
                             'Must include one or more label/annotations')
    parser.add_argument('--rh_roi_file', type=str, metavar='roi_file', 
                        help='A label GIfTI image (label.gii) or a Freesurfer '
                             'annotation file (.annot) for the right hemipshere. '
                             'Must include one or more label/annotations with '
                             'label names.')
    parser.add_argument('--as_vertices', default=False,
                        action='store_true',
                        help='Whether to extract out the timeseries of each '
                             'vertex instead of the mean timeseries. This is '
                             'only available for single ROI binary masks. '
                             'Default False.')                         
    return parser.parse_args()


def _check_hem_input(x, hem):
    if x is None:
        raise ValueError(f'No {hem} input files provided')
    else:
        x = check_glob(x)
        if not x:
            raise ValueError(f'Glob pattern did not find {hem} input files')
    return x


def _check_gifti_params(params):
    """Ensure that required fields are included and correctly formatted"""
    params = handle_base_args(params)

    for hem in ['lh', 'rh']:
        params[f'{hem}_files'] = _check_hem_input(params[f'{hem}_files'], hem)
        if not params[f'{hem}_roi_file']:
            raise ValueError(f'No {hem} roi file provided')
    
    return params


def extract_gifti(input_file, roi_file, regressor_file, params):
    """Gifti-specific mask_and_save"""
    
    # set up extraction
    extractor = GiftiExtractor(
        lh_fname=input_file[0],
        rh_fname=input_file[1],
        lh_roi_file=roi_file[0],
        rh_roi_file=roi_file[1],
        as_vertices=params['as_vertices'],
        radius=params['radius'], 
        allow_overlap=params['allow_overlap'], 
        standardize=params['standardize'], 
        t_r=params['t_r'], 
        high_pass=params['high_pass'], 
        low_pass=params['low_pass'], 
        detrend=params['detrend']
    )
    extractor.set_regressors(regressor_file, params['regressors'])
    if (params['discard_scans'] is not None) and (params['discard_scans'] > 0):
        extractor.discard_scans(params['discard_scans'])
    
    # extract timeseries and save
    extractor.extract(params['labels'])

    out = os.path.join(params['out'])
    extractor.timeseries.to_csv(out, sep='\t')
    

def main():
    """Primary entrypoint in program"""
    params = vars(_cli_parser())

    params = handle_base_args(params)
    params = _check_gifti_params(params)
    metadata_path = make_param_file(params)
    shutil.copy2(params['lh_roi_file'], metadata_path)
    shutil.copy2(params['rh_roi_file'], metadata_path)

    # setup and run extraction
    input_files = zip(params['lh_input_files'], params['rh_input_files'])
    roi_file = zip(params['lh_roi_file'], params['rh_roi_file'])
    run_extraction(extract_gifti, params['input_files'], params['roi_file'], 
                   params)


if __name__ == '__main__':
    raise RuntimeError("`nixtract/cli/nifti.py` should not be run directly. "
                       "Please `pip install` nixtract and use the "
                       "`xtract-gifti` command.")

