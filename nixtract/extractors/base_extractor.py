
import os
from datetime import datetime
import pandas as pd
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


class BaseExtractor(object):

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

    def save(self, out):
        self.check_extracted()
        self.timeseries.to_csv(out, sep='\t', index=False)

    def show_extract_msg(self, fname):
        if self.verbose:
            t = datetime.now().strftime("%H:%M:%S")
            print(f'[{t}] Extracting {os.path.basename(fname)}')