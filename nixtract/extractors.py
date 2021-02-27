"""Extractor classes to extract timeseries data from functional NIfTIs, GIfTIs,
and CIfTIs.
""" 
import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import (NiftiMasker, NiftiSpheresMasker, 
                                NiftiLabelsMasker)
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
import load_confounds


def _load_from_strategy(denoiser, fname):
    """Verifies if load_confounds strategy is useable given the regressor files.
    load_confounds will raise it's own exception, but add an additional 
    nixtract-specific exception that clarifies the incompatibility.
    """ 
    error_msg = ('load_confound strategy incompatible with provided regressor '
                 'files. Check regressor files if they contain the appropriate '
                 'columns determined by load_confounds.')
    try:
        confounds = denoiser.load(fname)
        return pd.DataFrame(confounds, columns=denoiser.columns_)
    except ValueError as e:
        raise ValueError(error_msg) from e


class ImageExtractor(object):
                    
    def set_regressors(self, regressor_file, regressors=None):
        """Set appropriate regressors."""

        # specific strategies for load_confounds
        strategies = ['Params2', 'Params6', 'Params9', 'Params24', 'Params36', 
                      'AnatCompCor', 'TempCompCor'] 
        flexible_strategies = ['motion', 'high_pass', 'wm_csf', 'compcor', 
                               'global']

        if regressors is None:
            # use all regressors from file
            regs = pd.read_csv(regressor_file, sep=r'\t')
        elif len(regressors) == 1 and (regressors[0] in strategies):
            # predefined strategy
            denoiser = eval('load_confounds.{}()'.format(regressors[0]))
            regs = _load_from_strategy(denoiser, regressor_file)
        elif set(regressors) <= set(flexible_strategies):
            # flexible strategy
            denoiser = load_confounds.Confounds(strategy=regressors)
            regs = _load_from_strategy(denoiser, regressor_file)
        elif all([x not in strategies + flexible_strategies 
                  for x in regressors]):
            # list of regressor names
            try:
                regs = pd.read_csv(regressor_file, sep='\t', 
                                   usecols=regressors)
            except ValueError as e:
                msg = 'Not all regressors are found in regressor file'
                raise ValueError(msg) from e
        else:
            raise ValueError('Invalid regressors. Regressors must be a list '
                             'of column names that appear in regressor_files, '
                             'OR a defined load_confounds regressor strategy '
                             '(flexible or non-flexible).')

        self.regressor_file = regressor_file
        self.regressors = regs
        return self

    def check_extracted(self):
        if not hasattr(self, 'data'):
            raise ValueError('timeseries data does not yet exist. Must call '
                             'extract().')


### NIFTI EXTRACTION
def _read_coords(roi_file):
    """Parse and validate coordinates from file"""

    if not roi_file.endswith('.tsv'):
        raise ValueError('Coordinate file must be a tab-separated .tsv file')

    coords = pd.read_table(roi_file)
    
    # validate columns
    columns = [x for x in coords.columns if x in ['x', 'y', 'z']]
    if (len(columns) != 3) or (len(np.unique(columns)) != 3):
        raise ValueError('Provided coordinates do not have 3 columns with '
                         'names `x`, `y`, and `z`')

    # convert to list of lists for nilearn input
    return coords.values.tolist()


def _get_spheres_from_masker(masker, ref_img):
    """Re-extract spheres from coordinates to make niimg. 
    
    Note that this will take a while, as it uses the exact same function that
    nilearn calls to extract data for NiftiSpheresMasker
    """
    ref_img = nib.Nifti1Image(ref_img.get_fdata()[:, :, :, [0]], 
                              ref_img.affine)

    X, A = _apply_mask_and_get_affinity(masker.seeds, ref_img, masker.radius, 
                                        masker.allow_overlap)
    # label sphere masks
    spheres = A.toarray()
    spheres *= np.arange(1, len(masker.seeds) + 1)[:, np.newaxis]

    # combine masks, taking the maximum if overlap occurs
    arr = np.zeros(spheres.shape[1])
    for i in np.arange(spheres.shape[0]):
        arr = np.maximum(arr, spheres[i, :])
    arr = arr.reshape(ref_img.shape[:-1])
    spheres_img = nib.Nifti1Image(arr, ref_img.affine)
    
    if masker.mask_img is not None:
        mask_img_ = resample_to_img(masker.mask_img, spheres_img)
        spheres_img = math_img('img1 * img2', img1=spheres_img, 
                               img2=mask_img_)
    return spheres_img


def _set_volume_masker(roi_file, as_voxels=False, **kwargs):
    """Check and see if multiple ROIs exist in atlas file"""

    if isinstance(roi_file, str) and roi_file.endswith('.tsv'):
        roi = _read_coords(roi_file)
        n_rois = len(roi)
        print('  {} region(s) detected from coordinates'.format(n_rois))

        if kwargs.get('radius') is None:
            warnings.warn('No radius specified for coordinates; setting '
                          'to nilearn.input_data.NiftiSphereMasker default '
                          'of extracting from a single voxel')
        masker = NiftiSpheresMasker(roi, **kwargs)
    else:
        # remove args for NiftiSpheresMasker 
        if 'radius' in kwargs:
            kwargs.pop('radius')
        if 'allow_overlap' in kwargs:
            kwargs.pop('allow_overlap')
    
        roi_img = load_img(roi_file)
        n_rois = len(np.unique(roi_img.get_data())) - 1
        print('  {} region(s) detected from {}'.format(n_rois,
                                                       roi_img.get_filename()))
        if n_rois > 1:
            masker = NiftiLabelsMasker(roi_img, **kwargs)
        elif n_rois == 1:
            # binary mask for single ROI 
            if as_voxels:
                if 'mask_img' in kwargs:
                    kwargs.pop('mask_img')
                masker = NiftiMasker(roi_img, **kwargs)
            else:
                # more computationally efficient if only wanting the mean
                masker = NiftiLabelsMasker(roi_img, **kwargs)
        else:
            raise ValueError('No ROI detected; check ROI file')
    
    return masker, n_rois


class NiftiExtractor(ImageExtractor):
    def __init__(self, fname, roi_file, labels=None, as_voxels=False, 
                 **kwargs):

        self.fname = fname
        self.img = nib.load(fname)
        self.roi_file = roi_file
        self.labels = labels
        self.as_voxels = as_voxels
        
        # determine masker
        self.masker = _set_volume_masker(roi_file, as_voxels, **kwargs)
        self.masker_type = self.masker.__class__.__name__
        
    def _get_default_labels(self):
        """Generate default numerical (1-indexed) labels depending on the 
        masker
        """
        self.check_extracted()
        
        if isinstance(self.masker, NiftiMasker):
            return ['voxel {}'.format(int(i))
                    for i in np.arange(self.data.shape[1]) + 1]
        elif isinstance(self.masker, NiftiLabelsMasker): 
            # get actual numerical labels used in image          
            return ['roi {}'.format(int(i)) for i in self.masker.labels_]
        elif isinstance(self.masker, NiftiSpheresMasker):
            return ['roi {}'.format(int(i)) 
                    for i in np.arange(len(self.masker.seeds)) + 1]

    def discard_scans(self, n_scans):
        """Discard first N scans from data and regressors, if available 

        Parameters
        ----------
        n_scans : int
            Number of initial scans to remove
        """
        arr = self.img.get_data()
        arr = arr[:, :, :, n_scans:]
        self.img = nib.Nifti1Image(arr, self.img.affine)

        if self.regressors is not None:
            self.regressors = self.regressors.iloc[n_scans:, :]
        
        return self

    def extract(self):
        """Extract timeseries data using the determined nilearn masker"""
        print('  Extracting from {}'.format(os.path.basename(self.fname)))
        timeseries = self.masker.fit_transform(self.img, 
                                               confounds=self.regressors.values)
        self.timeseries = pd.DataFrame(timeseries)
        
        if self.labels is None:
            self.timeseries.columns = self._get_default_labels()
        else:
            self.timeseries.columns = self.labels
        
        return self

    def get_fitted_roi_img(self):
        """Return fitted roi img from nilearn maskers

        Returns
        -------
        nibabel.Nifti1Image
            Image generated and used by underlying nilearn masker class.  
        """
        if isinstance(self.masker, NiftiMasker):
            return self.masker.mask_img_
        elif isinstance(self.masker, NiftiLabelsMasker):
            return self.masker.labels_img
        elif isinstance(self.masker, NiftiSpheresMasker):
            return _get_spheres_from_masker(self.masker, self.img)


### SURFACE EXTRACTION
def _read_gifti_label(fname):
    """Read valid roi label files for giftis"""
    if fname.endswith('.annot'):
        annot = nib.freesurfer.read_annot(fname)
        return annot[0], annot[2]
    
    else:
        img = nib.load(fname)
        darray = img.agg_data()
        # ensure one scan
        if len(darray.shape) != 1:
            raise ValueError('.label.gii mask img must be 1D (a single scan)')

        labels = img.labeltable.get_labels_as_dict()
        if labels:
            return darray, labels
        else:
            raise ValueError('Empty label table in .label.gii mask img')


def _read_cifti_dlabel(fname):
    """Read and validate .dlabel.nii"""
    pass


class GiftiExtractor(ImageExtractor):
    def __init__(self, fname, roi_file, as_vertices=False, **kwargs):
        
        
        self.fname = fname
        self.img = nib.load(fname)
        self.as_vertices = as_vertices
        self.roi_file = _read_gifti_label(roi_file)


    def _get_default_labels(self):
        pass

    def discard_scans(self, n_scans):
        pass
    

class CiftiExtractor(ImageExtractor):
    def __inti__(self,):
        pass

    def discard_scans(self, n_scans):
        pass
    
    def extract(self):
        pass