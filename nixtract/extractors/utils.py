import numpy as np
import pandas as pd
from nilearn import signal


def _mask(darray, roi, as_vertices=False):
    """Extract timeseries for each unique value in roi mask

    Parameters
    ----------
    darray : numpy.ndarray, (n_timepoints, n_vertices)
        Functional vertices
    roi : numpy.ndarray, (n_vertices,)
        Vertices with integer labels denoting the regions
    as_vertices : bool, optional
        Extract all vertices beloging to a label in roi. Only possible when
        roi is a binary mask, by default False

    Returns
    -------
    numpy.ndarray
        Masked array

    Raises
    ------
    ValueError
        roi contains multiple regions but as_vertices=True
    """
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
            mask = (roi.ravel() == l)
            timeseries[:, i] = darray[:, mask].mean(axis=1)
    
    return timeseries


def mask_data(darray, roi, regressors=None, as_vertices=False, 
              pre_clean=False, **kwargs):
    """[summary]

    Parameters
    ----------
    darray : numpy.ndarray, (n_timepoints, n_vertices)
        Functional vertices
    roi : numpy.ndarray, (n_vertices,)
        Vertices with integer labels denoting the regions
    regressors : numpy.ndarray, optional
        Confound regressors to regress from timeseries, by default None
    as_vertices : bool, optional
        Extract all vertices beloging to a label in roi. Only possible when
        roi is a binary mask, by default False
    pre_clean : bool, optional
        Run nilearn.signal.clean on all vertices prior to masking, rather than
        after. By default False

    Returns
    -------
    numpy.ndarray
        Extracted timeseries
    """
    x = darray.copy()
    if pre_clean:
        x = signal.clean(x, confounds=regressors, **kwargs)
        return _mask(x, roi, as_vertices)
    else:
        timeseries = _mask(x, roi, as_vertices)
        out = signal.clean(timeseries, confounds=regressors, **kwargs)
        return out


def label_timeseries(tseries, labels, as_vertices):
    """Label timeseries based on input labels of individual vertices

    Parameters
    ----------
    tseries : numpy.ndarray, (n timepoints, n timeseries)
        Extracted timeseries
    labels : list, 
        Timeseries labels, must equal number of timeseries in tseries
    as_vertices : [type]
        If timeseries is all vertices. If False, then assume each column in
        dtseries is a region mean

    Returns
    -------
    pandas.DataFrame
        Data table containing each timeseries with labels as column headers 
    """
    if as_vertices:
        cols = [f'vert{i}' for i in np.arange(tseries.shape[1])]
        return pd.DataFrame(tseries, columns=cols)
    else:
        return pd.DataFrame(tseries, columns=labels)