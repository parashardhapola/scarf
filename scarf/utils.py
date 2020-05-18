import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    import pandas as pd
    import numpy as np
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from typing import Callable
    from dask.diagnostics import ProgressBar
    import functools


def fit_lowess(a, b, n_bins: int,
               lowess_frac: float) -> np.array:
    stats = pd.DataFrame({'a': a, 'b': b}).apply(np.log)
    bin_edges = np.histogram(stats.a, bins=n_bins)[1]
    bin_edges[-1] += 0.1  # For including last gene
    bin_idx = []
    for i in range(n_bins):
        idx = pd.Series((stats.a >= bin_edges[i]) & (stats.a < bin_edges[i + 1]))
        if sum(idx) > 0:
            bin_idx.append(list(idx[idx].index))
    bin_vals = []
    for idx in bin_idx:
        temp_stat = stats.reindex(idx)
        temp_gene = temp_stat.idxmin().b
        bin_vals.append(
            [temp_stat.b[temp_gene], temp_stat.a[temp_gene]])
    bin_vals = np.array(bin_vals).T
    bin_cor_fac = lowess(bin_vals[0], bin_vals[1], return_sorted=False,
                         frac=lowess_frac, it=100).T
    fixed_var = {}
    for bcf, indices in zip(bin_cor_fac, bin_idx):
        for idx in indices:
            fixed_var[idx] = np.e ** (stats.b[idx] - bcf)
    return np.array([fixed_var[x] for x in range(len(a))])


def show_progress(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pbar = ProgressBar()
        pbar.register()
        ret_val = func(*args, **kwargs)
        pbar.unregister()
        return ret_val
    return wrapper


def load_zarr_table(zgrp) -> pd.DataFrame:
    keys = ['I', 'ids', 'names']
    keys = keys + [x for x in zgrp.keys() if x not in keys]
    return pd.DataFrame({x: zgrp[x][:] for x in keys})
