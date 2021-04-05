

import numpy as np
import pandas as pd
import nibabel as nib

from .base_extractor import BaseExtractor
from .utils import mask_data, label_timeseries


def _check_labels(darray, labels, fname):
    """Verify that labels are the same as what appears in the vertices"""
    detected_labels = np.unique(darray)
    n_detected = len(detected_labels)
    n_labels = len(labels)

    if n_detected > n_labels:
        raise ValueError(f'{fname} label table contains fewer labels '
                         f'({n_labels}) than detected labels ({n_detected}) '
                         ' in vertices')
    elif n_detected < n_labels:
        raise ValueError(f'{fname} label table contains more labels '
                         f'({n_labels}) than detected labels ({n_detected}) '
                         ' in vertices')
    else:
        return labels


def _read_annot(fname):
    """Safely read a .annot file and return the vertices + labels"""
    try:
        annot = nib.freesurfer.read_annot(fname)
    except ValueError:
        raise ValueError(f'Cannot read {fname}')

    darray = annot[0]
    labels = np.array(annot[2], dtype=np.str)
    labels = _check_labels(darray, labels, fname)
    labels = dict(zip(np.unique(darray), labels))
    return darray, labels
    

def _read_gifti_label(fname):
    """Safely read a func.gii file"""
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

    labels = _check_labels(darray, labels, fname)
    return darray, labels


def _load_gifti_roi(fname):
    """Only read acceptable roi files"""
    if fname.endswith('.annot'):
        darray, labels = _read_annot(fname)
    elif fname.endswith('.label.gii'):
        darray, labels = _read_gifti_label(fname)
    else:
        raise ValueError(f'{fname} must be a valid .annot or .label.gii file')
    return darray, labels


def _load_hem(in_file, roi_file):
    """Load hemisphere only if func.gii and label.gii are available"""
    if in_file:
        in_array = nib.load(in_file).agg_data()
        
        if roi_file:
            roi_darray, labels = _load_gifti_roi(roi_file)
            loaded = True
        else:
            raise ValueError('Missing ROI file')
        
        return in_array, roi_darray, labels, loaded
    else:
        loaded = False
        return None, None, None, loaded


def drop_zeros(tseries, labels, as_vertices):
    """Drop extracted data from label 0 if present"""
    if (0 in labels.keys()) and not as_vertices:
        zero_column = labels[0]
        # drop first column 
        return tseries.drop(zero_column, axis=1)
    else:
        return tseries
    

def _combine_timeseries(lh, rh):
    """Combine hemispheres into a single timeseries table"""
    cols = np.hstack([lh.columns, rh.columns])
    if len(cols) > len(set(cols)):
        # both hemispheres share at least one column name so add suffix
        # to prevent overlap
        lh.columns = ['L_' + i for i in lh.columns]
        rh.columns = ['R_' + i for i in rh.columns]
    return pd.concat([lh, rh], axis=1)


class GiftiExtractor(BaseExtractor):
    def __init__(self, lh_file=None, rh_file=None, lh_roi_file=None, 
                 rh_roi_file=None,  as_vertices=False, pre_clean=False, 
                 verbose=False, drop_zero_label=True, **kwargs):
        """Gifti extraction class. 

        Either left, right or both hemispheres can be provided. To use a 
        hemisphere, both the respective *h_file and the *h_roi_file must be 
        provided. At least one hemisphere must be provided.

        Parameters
        ----------
        lh_file : str, optional
            Left hemisphere func.gii file, by default None
        rh_file : str, optional
            Right hemisphere func.gii file, by default None
        lh_roi_file : str, optional
            Left hemisphere label.gii or .annot file containing region 
            labels, by default None
        rh_roi_file : str, optional
            Right hemisphere label.gii or .annot file containing region 
            labels, by default None
        as_vertices : bool, optional
            Extract the individual vertex timeseries from a region. Only 
            possible when roi_file is a binary mask (single region), by 
            default False
        pre_clean : bool, optional
            Denoise data (e.g., filtering, confound regression) before 
            timeseries extraction. Otherwise, denoising is done on the 
            extracted timeseries, which is consistent with nilearn and is more
            computationally efficient. By default False
        verbose : bool, optional
            Print out extraction timestamp, by default False
        **kwargs
            Arguments to pass to nilearn.signal.clean other than 
            confounds_regressors

        Raises
        ------
        ValueError
            No hemispheres are provided
        """
            
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
        self.verbose = verbose
        self.drop_zero_label = drop_zero_label
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
            self.regressor_array = self.regressor_array[n_scans:, :]
    
    def extract(self):
        """Extract timeseries"""
        if self._lh:
            self.show_extract_msg(self.lh_file)
            lh_tseries = mask_data(self.lh_darray.T, self.lh_roi, 
                                   self.regressor_array, self.as_vertices, 
                                   self.pre_clean, **self._clean_kwargs)
            lh_tseries = label_timeseries(lh_tseries, self.lh_labels.values(), 
                                          self.as_vertices)
            if self.drop_zero_label:
                lh_tseries = drop_zeros(lh_tseries, self.lh_labels, 
                                        self.as_vertices)
            
        if self._rh:
            self.show_extract_msg(self.rh_file)
            rh_tseries = mask_data(self.rh_darray.T, self.rh_roi, 
                                   self.regressor_array, self.as_vertices, 
                                   self.pre_clean, **self._clean_kwargs)
            rh_tseries = label_timeseries(rh_tseries, self.rh_labels.values(), 
                                             self.as_vertices)
            if self.drop_zero_label:
                rh_tseries = drop_zeros(rh_tseries, self.rh_labels, 
                                        self.as_vertices)

        if self._lh and self._rh:
            self.timeseries = _combine_timeseries(lh_tseries, rh_tseries)
        elif self._lh:
            self.timeseries = lh_tseries
        elif self._rh:
            self.timeseries = rh_tseries