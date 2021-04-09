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
    """Reads GIFTI CLI arguments and returns input specifications combined with
    those from the general CLI
    """
    parser = argparse.ArgumentParser()
    # input files
    parser.add_argument('--lh_files', nargs='+', type=str,
                        help='One or more input functional GIFTI files '
                             '(.func.gii) for the left hemisphere. Can also '
                             'be a single string with wildcards (*) to '
                             'specify all files matching the file pattern. If '
                             'so, these files are naturally sorted by file '
                             'name prior to extraction')
    parser.add_argument('--rh_files', nargs='+', type=str,
                        help='One or more input functional GIFTI files '
                             '(.func.gii) for the right hemisphere. Can also '
                             'be a single string with wildcards (*) to '
                             'specify all files matching the file pattern. If '
                             'so, these files are naturally sorted by file '
                             'name prior to extraction')
    # roi files
    parser.add_argument('--lh_roi_file', type=str, 
                        help='A label GIFTI file (.label.gii) or a Freesurfer '
                             'annotation file (.annot) for the left hemipshere. '
                             'Must include one or more labels')
    parser.add_argument('--rh_roi_file', type=str, metavar='roi_file', 
                        help='A label GIFTI file (.label.gii) or a Freesurfer '
                             'annotation file (.annot) for the right hemipshere. '
                             'Must include one or more labels')
    # other
    parser.add_argument('--as_vertices', default=False,
                        action='store_true',
                        help='Extract the timeseries of each vertex in '
                             'a region rather than the mean timeseries. This is '
                             'only available for when `lh_roi_file` and/or '
                             '`rh_roi_file` are a single region, i.e. a binary '
                             'mask. Default: False')
    parser.add_argument('--denoise-pre-extract', default=False,
                        action='store_true',
                        help='Denoise data (e.g., filtering, confound '
                             'regression) before timeseries extraction. '
                             'Otherwise, denoising is done on the extracted '
                             'timeseries, which is consistent with nilearn and '
                             'is more computationally efficient. Default: False')                
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

    # make None lists for a hemisphere the same length as the other
    params[f'lh_files'], params[f'rh_files'] = _equalize_lengths(params[f'lh_files'],
                                                                 params[f'rh_files'])
    return params


def _set_out_fname(input_files, out_dir):
    """Make output _timeseries.tsv filename based on what hemisphere are
    provided

    If left and right hemispheres are given, then file will be labeled 
    'hemi-LR'. Otherwise, the hemisphere label will be the same as the 
    provided hemisphere. 

    This is a a validation function too, as it checks for the hemisphere label 
    and also checks to make sure that at least one hemisphere is given.

    Parameters
    ----------
    input_files : tuple
        Left and right hemisphere input files: (left, right). At least one 
        must not be None. BIDS hemisphere labels ('hemi-L', 'hemi-R') must
        be in file names. 
    out_dir : str
        Output directory

    Returns
    -------
    str
        Determined output file name

    Raises
    ------
    ValueError
        No hemisphere label is in input files 
    ValueError
        No hemipshere input files provided
    """
    if all(input_files):
        if 'hemi-L' in input_files[0]:
            out_fname = input_files[0].replace('hemi-L', 'hemi-LR')
        else:
            raise ValueError("Gifti hemisphere should be identified in "
                             "filenames with 'hemi-L' or 'hemi-R'")
    elif input_files[0]:
        out_fname = input_files[0]
    elif input_files[1]:
        out_fname = input_files[1]
    else:
        raise ValueError('Must include input file from at least one '
                         'hemisphere')

    return os.path.join(out_dir, replace_file_ext(out_fname))


def extract_gifti(input_files, roi_file, regressor_file, params):
    """Extract timeseries from a GIFTI image

    Parameters
    ----------
    input_files : tuple
        Tuple of left and right hemisphere func.gii files, (left, right). If
        only one hemisphere is desired, then the other hemisphere can be 
        specified as None, e.g., left only: (left, None).  
    roi_file : tuple
        Tuple of left and right hemisphere label.gii files, (left, right). If
        only one hemisphere is desired, then the other hemisphere can be 
        specified as None, e.g., left only: (left, None).  
    regressor_file : str
        File path of regressor file
    params : dict
        Parameter dictionary for extraction
    """
    # validate input file(s) and make output file before extraction
    out = _set_out_fname(input_files, params['out_dir'])

    extractor = GiftiExtractor(
        lh_file=input_files[0],
        rh_file=input_files[1],
        lh_roi_file=roi_file[0],
        rh_roi_file=roi_file[1],
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
    extractor.save(out, params['n_decimals'])

    return out, extractor
    

def main():
    params = vars(_cli_parser())
    params = _check_gifti_params(params)
    metadata_path = make_param_file(params)

    for roi_file in [params['lh_roi_file'], params['rh_roi_file']]:
        if roi_file:
            shutil.copy2(roi_file, metadata_path)

    # setup and run extraction
    input_files = list(zip(params['lh_files'], params['rh_files']))
    roi_files = (params['lh_roi_file'], params['rh_roi_file'])
    run_extraction(extract_gifti, input_files, roi_files, params)


if __name__ == '__main__':
    raise RuntimeError("`nixtract/cli/gifti.py` should not be run directly. "
                       "Please `pip install` nixtract and use the "
                       "`nixtract-gifti` command.")

