import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import signal

from .base_extractor import BaseExtractor
from .utils import mask_data, label_timeseries



def _check_cifti(fname):
    img = nib.load(fname)
    if not isinstance(img, nib.Cifti2Image):
        raise ValueError(f'{fname} not an instance of Cifti2Image')
    return img


def _read_dtseries(fname):
    if not fname.endswith('.dtseries.nii'):
        raise ValueError(f'{fname} must be a .dtseries.nii file')
    return _check_cifti(fname)


def _read_dlabel(fname):

    if not fname.endswith('.dlabel.nii'):
        raise ValueError(f'{fname} must be a .dlabel.nii file')
    img = _check_cifti(fname)

    label_dict = img.header.get_axis(index=0).label[0]
    labels = []
    for k, v in label_dict.items():
        labels.append(v[0])
    return img, labels


def _get_models(img):

    brain_models = list(img.header.get_index_map(1).brain_models)

    models = {}
    for i, m in enumerate(brain_models):
        struct = m.brain_structure
        n_vertices = m.surface_number_of_vertices 
        models[struct] = {'count': m.index_count, 'offset': m.index_offset, 
                          'model_index': i, 'type': m.model_type}
        if m.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
            models[struct]['n_indices'] = m.surface_number_of_vertices
            models[struct]['indices'] = np.asarray(m.vertex_indices)
        else:
            models[struct]['n_indices'] = m.index_count
            models[struct]['indices'] = np.arange(m.index_count)

    # ensure sorted by offset
    models = {k: v for k, v in sorted(models.items(), 
                                      key=lambda x: x[1]['offset'])}
    return models
    

def _has_medwall(model):
    """Check if structure has medial wall, which is when the model count is 
    equal to the number of vertices. Always false for non surface models
    """
    if ((model['type'] == 'CIFTI_MODEL_TYPE_SURFACE') and 
        (model['count'] == model['n_indices'])):
        return True
    else:
        return False


def _load_and_align_ciftis(dlabel, dtseries):

    dlabel_data = dlabel.get_fdata().ravel()
    dtseries_data = dtseries.get_fdata()

    if dlabel.shape[1] == dtseries.shape[1]:
        return dlabel_data, dtseries_data
    else:
        warnings.warn(f'dlabel has shape {dlabel.shape[1]} and dtseries has '
                      f'shape {dtseries.shape[1]}. Aligning files via '
                      'brain structures present in each file. Double check '
                      'results!')
        dl_models = _get_models(dlabel)
        dts_models = _get_models(dtseries)

        dlabel_list = []
        dtseries_list = []
        for k, v in dts_models.items():
  
            if k not in dl_models.keys():
                continue

            if dl_models[k]['count'] == v['count']:
                # both are the same so doesnt matter if medial wall or not
                dts_idx = np.arange(v['count']) + v['offset']
                dl_idx = dts_idx
            elif _has_medwall(dl_models[k]) and not _has_medwall(v):
                # select dtseries vertices from dlabel
                dl_idx = v['indices'] + dl_models[k]['offset']
                dts_idx = np.arange(v['count']) + v['offset']
            elif _has_medwall(v) and not _has_medwall(dl_models[k]):
                # select dlabel vertices from dtseries
                dts_idx = dl_models[k]['indices'] + v['offset']
                dl_idx = np.arange(dl_models[k]['count']) + dl_models[k]['offset']
            else:
                # no medial wall in both but also not equal
                raise ValueError('Cannot align dlabel with dtseries.')
            
            dlabel_list.append(dlabel_data[dl_idx])
            dtseries_list.append(dtseries_data[:, dts_idx])

        return np.hstack(dlabel_list), np.hstack(dtseries_list)
            

class CiftiExtractor(BaseExtractor):
    def __init__(self, fname, roi_file, as_vertices=False, pre_clean=False, 
                 verbose=False, **kwargs):
        
        self.fname = fname
        self.dtseries = _read_dtseries(fname)
        self.roi_file = roi_file
        self.dlabel, self.labels = _read_dlabel(roi_file)     
        self.as_vertices = as_vertices
        self.pre_clean = pre_clean
        self.verbose = verbose
        self._clean_kwargs = kwargs

        self.dlabel_array, self.darray = _load_and_align_ciftis(self.dlabel, 
                                                                self.dtseries)
        self.regressor_names = None
        self.regressor_array = None
        
    def discard_scans(self, n_scans):
        """Discard first N scans from data and regressors, if available 

        Parameters
        ----------
        n_scans : int
            Number of initial scans to remove
        """
        self.darray = self.darray[:, n_scans:]

        if self.regressor_array is not None:
            self.regressor_array = self.regressor_array.iloc[n_scans:, :]
    
    def extract(self):
        
        self.show_extract_msg(self.fname)
        tseries = mask_data(self.darray, self.dlabel_array, 
                            self.regressor_array, self.as_vertices, 
                            self.pre_clean, **self._clean_kwargs)
        self.timeseries = label_timeseries(tseries, self.labels, 
                                           self.as_vertices)
        # remove extracted background signal if any
        if '???' in self.timeseries.columns:
            self.timeseries = self.timeseries.drop('???', axis=1)