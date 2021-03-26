"""Functions for command line interface
"""
import argparse
import sys
import os
import json
import shutil

from nixtract.cli.base import (base_cli, handle_base_args, replace_file_ext,
                               make_param_file, check_glob, run_extraction)
from nixtract.extractors import GiftiExtractor

def _cli_parser():
    """Reads command line arguments and returns input specifications"""
    parser = argparse.ArgumentParser()
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
    # other
    parser.add_argument('--as_vertices', default=False,
                        action='store_true',
                        help='Whether to extract out the timeseries of each '
                             'vertex instead of the mean timeseries. This is '
                             'only available for single ROI binary masks. '
                             'Default False.')
    parser.add_argument('--denoise-pre-extract', default=False,
                        action='store_true',
                        help='Denoise data (e.g., filtering, confound '
                             'regression) before timeseries extraction. '
                             'Otherwise, denoising is done on the extracted '
                             'timeseries, which is consistent with nilearn and '
                             'more computationally efficient. Default False.')                  
    parser = base_cli(parser)                         
    return parser.parse_args()



def _equalize_lengths(a, b):
    if len(a) > len(b):
        b *= len(a)
    elif len(a) < len(b):
        a *= len(b)
    return a, b


def _check_gifti_params(params):
    """Ensure that required fields are included and correctly formatted"""
    params = handle_base_args(params)

    for hem in ['lh', 'rh']:
        if params[f'{hem}_files']:
            params[f'{hem}_files'] = check_glob(params[f'{hem}_files'])
            if len(params[f'{hem}_files']) == 0:
                raise ValueError(f'Glob pattern did not find {hem} input files')
        else:
            params[f'{hem}_files'] = [None]

    params[f'lh_files'], params[f'rh_files'] = _equalize_lengths(params[f'lh_files'],
                                                                 params[f'rh_files'])

    return params


def _set_out_fname(input_file, out_dir):
    if all(input_file):
        # check file naming for hemispheres and create a unified out file if valid
        if 'hemi-L' in input_file[0]:
            out_fname = input_file[0].replace('hemi-L', 'hemi-LR')
        else:
            raise ValueError("Gifti hemisphere should be identified in "
                             "filenames with 'hemi-L' or 'hemi-R'")
    elif input_file[0]:
        out_fname = input_file[0]
    elif input_file[1]:
        out_fname = input_file[1]
    else:
        raise ValueError('Must include input file from at least one '
                         'hemisphere')

    return os.path.join(out_dir, replace_file_ext(out_fname))


def extract_gifti(input_file, roi_file, regressor_file, params):
    """Gifti-specific mask_and_save"""

    # validate input file(s) and make output file before extraction
    out = _set_out_fname(input_file, params['out_dir'])

    # set up extraction
    extractor = GiftiExtractor(
        lh_file=input_file[0],
        rh_file=input_file[1],
        lh_roi_file=roi_file[0],
        rh_roi_file=roi_file[1],
        as_vertices=params['as_vertices'],
        pre_clean=params['denoise_pre_extract'],
        standardize=params['standardize'], 
        t_r=params['t_r'], 
        high_pass=params['high_pass'], 
        low_pass=params['low_pass'], 
        detrend=params['detrend']
    )
    if regressor_file is not None:
        extractor.set_regressors(regressor_file, params['regressors'])

    if (params['discard_scans'] is not None) and (params['discard_scans'] > 0):
        extractor.discard_scans(params['discard_scans'])
    
    # extract timeseries and save
    extractor.extract()
    extractor.timeseries.to_csv(out, sep='\t')
    

def main():
    """Primary entrypoint in program"""
    params = vars(_cli_parser())

    params = _check_gifti_params(params)
    metadata_path = make_param_file(params)

    for roi_file in [params['lh_roi_file'], params['rh_roi_file']]:
        if roi_file:
            shutil.copy2(roi_file, metadata_path)

    # setup and run extraction
    input_files = list(zip(params['lh_files'], params['rh_files']))
    roi_files = (params['lh_roi_file'], params['rh_roi_file'])
    run_extraction(extract_gifti, input_files, roi_files, 
                   params['regressor_files'], params)


if __name__ == '__main__':
    raise RuntimeError("`nixtract/cli/gifti.py` should not be run directly. "
                       "Please `pip install` nixtract and use the "
                       "`xtract-gifti` command.")

