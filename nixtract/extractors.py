"""Extractor classes to extract timeseries data from functional NIfTIs, GIfTIs,
and CIfTIs.
""" 
import os
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import (NiftiMasker, NiftiSpheresMasker, 
                                NiftiLabelsMasker)
from nilearn import signal, image
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
        return confounds, denoiser.columns_
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
            regs, names = pd.read_csv(regressor_file, sep=r'\t')
        elif len(regressors) == 1 and (regressors[0] in strategies):
            # predefined strategy
            denoiser = eval('load_confounds.{}()'.format(regressors[0]))
            regs, names = _load_from_strategy(denoiser, regressor_file)
        elif set(regressors) <= set(flexible_strategies):
            # flexible strategy
            denoiser = load_confounds.Confounds(strategy=regressors)
            regs, names = _load_from_strategy(denoiser, regressor_file)
        elif all([x not in strategies + flexible_strategies 
                  for x in regressors]):
            # list of regressor names
            try:
                regs = pd.read_csv(regressor_file, sep='\t', 
                                   usecols=regressors)
                names = regressors
            except ValueError as e:
                msg = 'Not all regressors are found in regressor file'
                raise ValueError(msg) from e
        else:
            raise ValueError('Invalid regressors. Regressors must be a list '
                             'of column names that appear in regressor_files, '
                             'OR a defined load_confounds regressor strategy '
                             '(flexible or non-flexible).')

        self.regressor_file = regressor_file
        self.regressor_names = names
        self.regressor_array = regs
        return self

    def check_extracted(self):
        if not hasattr(self, 'timeseries'):
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
        mask_img_ = image.resample_to_img(masker.mask_img, spheres_img)
        spheres_img = image.math_img('img1 * img2', img1=spheres_img, 
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
    
        roi_img = image.load_img(roi_file)
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
        self.masker, self.n_rois = _set_volume_masker(roi_file, as_voxels, 
                                                      **kwargs)
        self.masker_type = self.masker.__class__.__name__
        self.regressor_names = None
        self.regressor_array = None
        
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

        if self.regressor_array is not None:
            self.regressor_array = self.regressor_array.iloc[n_scans:, :]
        
        return self

    def extract(self):
        """Extract timeseries data using the determined nilearn masker"""
        print('  Extracting from {}'.format(os.path.basename(self.fname)))
        timeseries = self.masker.fit_transform(self.img, 
                                               confounds=self.regressor_array)
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

def _load_gifti_array():
    # agg_data will sometimes return tuple instead of numpy array, so make
    # sure to always return numpy array
    pass

def _read_annot(fname):
    try:
        annot = nib.freesurfer.read_annot(fname)
        darray = annot[0]
        labels = np.array(annot[2], dtype=np.str)
        return darray, labels
    except ValueError:
        raise ValueError('Invalid .annot file')


def _read_gifti_label(fname):
    img = nib.load(fname)
    if not isinstance(img, nib.GiftiImage):
        raise ValueError(f'{fname} not an read as a GiftiImage')
    # check if one scan and validate labels                  
    darray = img.agg_data()
    if len(darray.shape) != 1:
        raise ValueError(f'{fname} is not 1D')
    labels = img.labeltable.get_labels_as_dict()
    if not labels:
        raise ValueError(f'Empty label table in {fname}')
    return darray, labels


def _load_gifti_roi(fname):

    if fname.endswith('.annot'):
        darray, labels = _read_annot(fname)
    elif fname.endswith('.gii'):
        darray, labels = _read_gifti_label(fname)
    else:
        raise ValueError(f'{fname} must be a valid .annot or .gii file')
    return darray, labels


def _load_hem(in_file, roi_file):
    if in_file:
        in_array = nib.load(in_file).agg_data()
        
        if roi_file:
            roi_darray, labels = _load_gifti_roi(roi_file)
            loaded = True
        else:
            raise ValueError('Missing ROI file')
        
        return in_array, roi_darray, labels, loaded
    else:
        return None, None, None, False


def _detect_hem(lh_array, rh_array):
    if all([isinstance(i, np.ndarray) for i in [lh_array, rh_array]]):
        return 'both'
    


    

def _mask_vertices(darray, roi, as_vertices=False):
    labels = np.unique(roi)
    if len(labels) > 2 and as_vertices:
        raise ValueError('Using as_vertices=True with more than one region '
                         'in roi file. Vertex-level extraction can only be '
                         'performed with a single-region (binary) roi file.')
    if as_vertices:
        timeseries = darray[:, roi.ravel().astype(bool)]
    else:
        timeseries = np.zeros((darray.shape[0], len(labels)))
        for i, l in enumerate(labels):
            mask = np.where(roi == l, 1, 0).astype(bool)
            timeseries[:, i] = darray[:, mask].mean(axis=1)
    return timeseries


def mask_gifti(darray, roi, regressors=None, as_vertices=False, 
               pre_clean=False, **kwargs):
    x = darray.copy().T
    if pre_clean:
        x = signal.clean(x, confounds=regressors, **kwargs)
        return _mask_vertices(x, roi, as_vertices)
    else:
        timeseries = _mask_vertices(x, roi, as_vertices)
        return signal.clean(timeseries, confounds=regressors, **kwargs)


def _combine_timeseries(lh, rh):
    cols = np.hstack([lh.columns, rh.columns])
    if len(cols) > len(set(cols)):
        # both hemispheres share at least one column name so add suffix
        # to prevent overlap
        lh.columns = ['L_' + i for i in lh.columns]
        rh.columns = ['R_' + i for i in rh.columns]
    return pd.concat([lh, rh], axis=1)


class GiftiExtractor(ImageExtractor):
    def __init__(self, lh_file=None, rh_file=None, lh_roi_file=None, 
                 rh_roi_file=None,  as_vertices=False, pre_clean=False, 
                 **kwargs):
            
        self.lh_file = lh_file
        self.lh_roi_file = lh_roi_file
        (self.lh_darray, self.lh_roi, 
         self.lh_labels, self._lh) = _load_hem(lh_file, lh_roi_file)

        self.rh_file = rh_file
        self.rh_roi_file = rh_roi_file
        (self.rh_darray, self.rh_roi, 
         self.rh_labels, self._rh) = _load_hem(rh_file, rh_roi_file)

        if not any([self._lh, self._rh]):
            raise ValueError('At least one hemisphere must be provided to '
                             'GiftiExtractor')

        self.as_vertices = as_vertices
        self.pre_clean = pre_clean
        self._clean_kwargs = kwargs

        self.regressor_names = None
        self.regressor_array = None

    def discard_scans(self, n_scans):
        """Discard first N scans from data and regressors, if available 

        Parameters
        ----------
        n_scans : int
            Number of initial scans to remove
        """
        if self._lh:
            self.lh_darray = self.lh_darray[:, n_scans:]
        if self._rh:
            self.rh_darray = self.rh_darray[:, n_scans:]

        if self.regressor_array is not None:
            self.regressor_array = self.regressor_array.iloc[n_scans:, :]
    
    def extract(self):

        if self._lh:
            lh_tseries = mask_gifti(self.lh_darray, self.lh_roi, 
                                    self.regressor_array, self.as_vertices, 
                                    self.pre_clean, **self._clean_kwargs)
            lh_tseries = pd.DataFrame(lh_tseries, columns=self.lh_labels)
        
        if self._rh:
            rh_tseries = mask_gifti(self.rh_darray, self.rh_roi, 
                                    self.regressor_array, self.as_vertices, 
                                    self.pre_clean, **self._clean_kwargs)
            rh_tseries = pd.DataFrame(rh_tseries, columns=self.rh_labels)

        if self._lh and self._rh:
            self.timeseries = _combine_timeseries(lh_tseries, rh_tseries)
        elif self._lh:
            self.timeseries = lh_tseries
        elif self._rh:
            self.timeseries = rh_tseries

# def _read_cifti_dlabel(fname):

#     img = nib.load(fname)
#     if not isinstance(img, nib.Cifti2Image):
#         raise ValueError('.dlabel.nii (roi file) not an instance of '
#                          'Cifti2Image')
#     darray = np.array(img.get_fdata()).ravel()

# def _mask_cifti():
#     timeseries = _mask_vertices(x, roi, as_vertices)
#     return signal.clean(timeseries, confounds=regressors, **kwargs)


# class CiftiExtractor(ImageExtractor):
#     def __inti__(self, fname, roi_file, as_vertices, **kwargs):
        
#         self.fname = fname
#         self.darray = nib.load(fname)
#         self.roi_file = _read_cifti_dlabel(roi_file)
#         self.as_vertices = as_vertices
#         self._clean_kwargs = kwargs
        

#     def discard_scans(self, n_scans):
#         pass
    
#     def extract(self):
#         pass