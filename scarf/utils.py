import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Callable
from dask.diagnostics import ProgressBar
from dask.distributed import progress
import functools
import subprocess
import shlex
from scipy.stats import norm


def fit_lowess(a, b, n_bins: int,
               lowess_frac: float) -> np.ndarray:
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


def rescale_array(a: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """
    Performs edge trimming on values of the input vector and constraints them between within frac and 1-frac density of
    normal distribution created with the sample mean and std. dev. of a

    :param a: numeric vector
    :param frac: Value between 0 and 1.
    :return:
    """
    loc = (np.median(a) + np.median(a[::-1])) / 2
    dist = norm(loc, np.std(a))
    minv, maxv = dist.ppf(1 - frac), dist.ppf(frac)
    a[a < minv] = minv
    a[a > maxv] = maxv
    return a


def clean_array(x, fill_val: int = 0):
    """
    Remove nan and infinite values from
    :param x:
    :param fill_val:
    :return:
    """
    x = np.nan_to_num(x, copy=True)
    x[(x == np.Inf) | (x == -np.Inf)] = 0
    x[x == 0] = fill_val
    return x


def show_progress(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pbar = ProgressBar()
        pbar.register()
        ret_val = func(*args, **kwargs)
        pbar.unregister()
        return ret_val
    return wrapper


def calc_computed(a, msg: str = None):
    if msg is not None:
        print(msg, flush=True)
    a = a.persist()
    progress(a, notebook=False)
    print(flush=True)
    return a.compute()


def system_call(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip())
    process.poll()
    return None


def load_zarr_table(zgrp) -> pd.DataFrame:
    keys = ['I', 'ids', 'names']
    keys = keys + [x for x in zgrp.keys() if x not in keys]
    return pd.DataFrame({x: zgrp[x][:] for x in keys})
