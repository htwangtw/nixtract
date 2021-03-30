

import numpy as np
import pandas as pd
import nibabel as nib

from .base_extractor import BaseExtractor
from .utils import mask_vertices

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


def _make_timeseries_df(tseries, labels, as_vertices):
    if as_vertices:
        cols = [f'vert{i}' for i in np.arange(tseries.shape[1])]
        return pd.DataFrame(tseries, columns=cols)
    else:
        return pd.DataFrame(tseries, columns=labels)


def _combine_timeseries(lh, rh):
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
                 verbose=False, **kwargs):
            
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
            self.show_extract_msg(self.lh_file)
            lh_tseries = mask_vertices(self.lh_darray, self.lh_roi, 
                                       self.regressor_array, self.as_vertices, 
                                       self.pre_clean, **self._clean_kwargs)
            lh_tseries = _make_timeseries_df(lh_tseries, self.lh_labels, 
                                             self.as_vertices)
        
        if self._rh:
            self.show_extract_msg(self.rh_file)
            rh_tseries = mask_vertices(self.rh_darray, self.rh_roi, 
                                       self.regressor_array, self.as_vertices, 
                                       self.pre_clean, **self._clean_kwargs)
            rh_tseries = _make_timeseries_df(rh_tseries, self.rh_labels, 
                                             self.as_vertices)

        if self._lh and self._rh:
            self.timeseries = _combine_timeseries(lh_tseries, rh_tseries)
        elif self._lh:
            self.timeseries = lh_tseries
        elif self._rh:
            self.timeseries = rh_tseries