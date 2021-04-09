
import os
from datetime import datetime
import numpy as np
import pandas as pd
import load_confounds


def _load_from_strategy(denoiser, fname):
    """Verifies if load_confounds strategy is useable given the regressor files.
    load_confounds will raise it's own exception, but add an additional 
    nixtract-specific exception that clarifies the incompatibility.

    Parameters
    ----------
    denoiser : load_confounds.Denoiser
        load_confounds Denoiser instance
    fname : str
        Regressor file

    Returns
    -------
    np.ndarray, list
        Regressor array and the associated regressor names

    Raises
    ------
    ValueError
        Incorrect load_confound strategy
    """
    error_msg = ('load_confound strategy incompatible with provided regressor '
                 'files. Check regressor files if they contain the appropriate '
                 'columns determined by load_confounds.')
    try:
        confounds = denoiser.load(fname)
        return confounds, denoiser.columns_
    except ValueError as e:
        raise ValueError(error_msg) from e


def _load_predefined_strategy(regressors, regressor_file, kwargs=None):
    """Read a pre-defined strategy from load_confounds"""
    if kwargs is not None:
        cmd = 'load_confounds.{}(**kwargs)'.format(regressors)
    else:
        cmd = 'load_confounds.{}()'.format(regressors)
    
    denoiser = eval(cmd)
    return _load_from_strategy(denoiser, regressor_file)


def _load_flexible_strategy(regressors, regressor_file, kwargs=None):
    """Load a flexible strategy from load_confounds"""
    if kwargs is not None:
        if "strategy" in kwargs:
            kwargs.pop('strategy')
        denoiser = load_confounds.Confounds(strategy=regressors, **kwargs)
    else:
        denoiser = load_confounds.Confounds(strategy=regressors)
    return _load_from_strategy(denoiser, regressor_file)


def _load_regressor_names(regressors, regressor_file):
    """Load a list of regressor names"""
    try:
        regs = pd.read_table(regressor_file, usecols=regressors)
        return regs.values, regs.columns
    except ValueError as e:
        msg = 'Not all regressors are found in regressor file'
        raise ValueError(msg) from e


class BaseExtractor(object):

    def set_regressors(self, regressor_file, regressors=None, 
                       load_confounds_kwargs=None):
        """Set regressors to be used with extraction

        NaN imputation is done by default because nilearn.signal.clean will 
        crash if NaNs are present. Any initial NaNs (derivatives, 
        framewise_displacement, etc) are replaced with the value in subsequent 
        row, which is the load_confound approach. If initial scans are to be 
        discarded then this is a non-issue. 

        Parameters
        ----------
        regressor_file : str
            Regressor file where each column is a separate regressor. The
            first row must be column headers, i.e. regressor names
        regressors : list or str, optional
            List of regressor names, or a load_confounds regressor strategy. 
            Must be compatible with regressor_file. By default None

        Raises
        ------
        ValueError
            Regressor files does not include all specified regressors
        ValueError
            Regressors is not a list or a str
        """
        # strategy options in load confounds
        strategies = ['Params2', 'Params6', 'Params9', 'Params24', 'Params36', 
                      'AnatCompCor', 'TempCompCor'] 
        flexible_strategies = ['motion', 'high_pass', 'wm_csf', 'compcor', 
                               'global']

        if isinstance(regressors, str):
            regressors = [regressors]

        # load regressors based on regressors provided
        if regressors is None:
            regs = pd.read_table(regressor_file)
            names = regs.columns
            regs = regs.values
            self._load_confounds = False
        
        elif len(regressors) == 1 and (regressors[0] in strategies):
            regs, names = _load_predefined_strategy(regressors[0], 
                                                    regressor_file, 
                                                    load_confounds_kwargs)
            self._load_confounds = True

        elif set(regressors) <= set(flexible_strategies):
            regs, names = _load_flexible_strategy(regressors, regressor_file, 
                                                  load_confounds_kwargs)
            self._load_confounds = True

        elif all([x not in strategies + flexible_strategies 
                  for x in regressors]):
            regs, names = _load_regressor_names(regressors, regressor_file)
            self._load_confounds = False
        
        else:
            raise ValueError('Invalid regressors. Regressors must be a list '
                             'of column names that appear in regressor_files, '
                             'OR a defined load_confounds regressor strategy '
                             '(flexible or non-flexible).')

        self.regressor_file = regressor_file
        self.regressor_names = names
        self.regressor_array = regs

        # impute initial NaN using load_confounds approach
        mask = np.isnan(self.regressor_array[0, :])
        self.regressor_array[0, mask] = self.regressor_array[1, mask]

    def check_extracted(self):
        """Check is extraction has been performed

        Raises
        ------
        ValueError
            Object has no timeseries attribute yet  
        """
        if not hasattr(self, 'timeseries'):
            raise ValueError('timeseries data does not yet exist. Must call '
                             'extract().')

    def save(self, out, n_decimals=None):
        """Save file to a .tsv file

        Parameters
        ----------
        out : str
            Output file name
        """
        self.check_extracted()
        float_format = f'%.{n_decimals}f' if n_decimals else None
        self.timeseries.to_csv(out, sep='\t', index=False, 
                               float_format=float_format)

    def show_extract_msg(self, fname):
        """Display extraction message if verbosity is set

        Parameters
        ----------
        fname : str
            Extractor file name
        """
        if self.verbose:
            t = datetime.now().strftime("%H:%M:%S")
            print(f'[{t}] Extracting {os.path.basename(fname)}')