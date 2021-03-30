"""Functions for command line interface
"""
import argparse
import sys
import os
import json
import shutil
import pandas as pd

from nixtract.atlases import get_labelled_atlas
from nixtract.cli.base import (base_cli, handle_base_args, replace_file_ext,
                               make_param_file, check_glob, empty_to_none, 
                               run_extraction)
from nixtract.extractors import NiftiExtractor

def _cli_parser():
    """Reads command line arguments and returns input specifications"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', nargs='+', type=str,
                        help='One or more input NIfTI images. Can also be a '
                             'single string with a wildcard (*) to specify all '
                             'files matching the file pattern. If so, these '
                             'files are naturally sorted by file name prior to '
                             'extraction.')
    parser.add_argument('--roi_file', type=str, metavar='roi_file', 
                        help='Parameter that defines the region(s) of interest. '
                             'This can be 1) a file path to NIfTI image that is '
                             'an atlas of multiple regions or a binary mask of '
                             'one region, 2) a nilearn query string formatted as '
                             '`nilearn:<atlas-name>:<atlas-parameters> (see '
                             'online documentation), or 3) a file path to a '
                             '.tsv file that has x, y, z columns that contain '
                             'roi_file coordinates in MNI space. Refer to online '
                             'documentation for more on how these options map '
                             'onto the underlying nilearn masker classes.')
    parser.add_argument('--mask_img', type=str, metavar='mask_img',
                        help='File path of a NIfTI mask a to be used when '
                             '`roi_file` is a) an multi-region atlas or a b) list '
                             'of coordinates. This will restrict extraction to '
                             'only voxels within the mask. If `roi_file` is a '
                             'single-region binary mask, this will be ignored.')
    parser.add_argument('--labels', nargs='+', type=str, metavar='labels',
                        help='Labels corresponding to the mask numbers in '
                             '`mask`. Can either be a list of strings, or a '
                             '.tsv file that contains a `Labels` column. Labels '
                             'must be sorted in ascending order to correctly '
                             'correspond to the atlas indices. The number of '
                             'labels provided must match the number of non-zero '
                             'indices in `mask`. Numeric indices are used if '
                             'not provided (default)')
    parser.add_argument('--as_voxels', default=False, action='store_true',
                        help='Whether to extract out the timeseries of each '
                             'voxel instead of the mean timeseries. This is '
                             'only available for single ROI binary masks. '
                             'Default False.')
    parser.add_argument('--radius', type=float, metavar='radius', 
                        help='Set the radius of the spheres (in mm) centered on '
                             'the coordinates provided in `roi_file`. Only applicable '
                             'when a coordinate .tsv file is passed to `roi_file`; '
                             'otherwise, this will be ignored. If not set, '
                             'the nilearn default of extracting from a single '
                             'voxel (the coordinates) will be used.')
    parser.add_argument('--allow_overlap', action='store_true', default=False,
                        help='Permit overlapping spheres when coordinates are '
                             'provided to `roi_file` and sphere-radius is not None.')                             
    parser = base_cli(parser)
    return parser.parse_args()


def _check_nifti_params(params):
    """Ensure that required fields are included and correctly formatted"""
    params = handle_base_args(params)

    if params['input_files'] is None:
        raise ValueError('Missing input files. Check files')
    else:
        params['input_files'] = check_glob(params['input_files'])
        # glob returned nothing
        if not params['input_files']:
            raise ValueError('Missing input files. Check files')

    if not params['roi_file']:
        raise ValueError('Missing roi_file input.')
    
    if params['roi_file'].startswith('nilearn:'):
        cache = os.path.join(params['output_dir'], 'nixtract_data')
        os.makedirs(cache, exist_ok=True)
        atlas, labels = get_labelled_atlas(params['roi_file'], data_dir=cache,
                                           return_labels=True)
        params['roi_file'] = atlas
        params['labels'] = labels

    params['labels'] = empty_to_none(params['labels'])
    if isinstance(params['labels'], str):
        if params['labels'].endswith('.tsv'):
            df = pd.read_table(params['labels'])
            params['labels'] = df['Label'].tolist()
        else:
            raise ValueError('Labels must be a filename or a list of strings.')

    return params


def extract_nifti(input_file, roi_file, regressor_file, params):
    """Nifti-specific mask_and_save"""

    # set up extraction
    extractor = NiftiExtractor(
        fname=input_file,
        roi_file=roi_file, 
        labels=params['labels'],
        as_voxels=params['as_voxels'],
        mask_img=params['mask_img'], 
        radius=params['radius'], 
        allow_overlap=params['allow_overlap'],
        verbose=params['verbose'], 
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
    out = os.path.join(params['out_dir'], replace_file_ext(input_file))
    extractor.save(out)


def main():
    """Primary entrypoint in program"""
    params = vars(_cli_parser())

    params = _check_nifti_params(params)
    metadata_path = make_param_file(params)
    shutil.copy2(params['roi_file'], metadata_path)

    # setup and run extraction
    run_extraction(extract_nifti, params['input_files'], params['roi_file'], 
                   params['regressor_files'], params)

if __name__ == '__main__':
    raise RuntimeError("`nixtract/cli/nifti.py` should not be run directly. "
                       "Please `pip install` nixtract and use the "
                       "`xtract-nifti` command.")
