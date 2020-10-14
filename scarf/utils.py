import numpy as np
import pandas as pd
from random import seed
from random import random
import math

def fit_lowess(a, b, n_bins: int, lowess_frac: float) -> np.ndarray:
    from statsmodels.nonparametric.smoothers_lowess import lowess

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

def resample_array(a: np.ndarray, target=1000 ) -> np.ndarray:
    """
    Performs a resampling of the data leading to EXACTLY <target> reads.
    In short it divides the result = array by array /sum(array) * target.

    All fractions can either be set to 1 or 0 and a random number check is used to decide that.
    If the fraction is set to 1 the difference between fraction and random value is stored. 
    The same is true for the case the expression is lost (0).

    If the total sum(result) is not eual to target the respective ids from the before stored 0 and 1 cases are sorted by
    difference and the tests with the least difference are revered in outcome until the sum(result) equals the target.
    """

    results = a/np.sum(a) * target
    li = [[-1,0]] * len(results)

    for i in range(0, len(results)):
        frac, total = math.modf( results[i] )
        results[i] = int(total)
        li[i]=[i, frac]

    #print ( "sum of results:" + str(sum(results)) )

    def takeSecond(elem):
        return elem[1]
    li.sort(key=takeSecond, reverse=True)

    #print ( "less by:" + str(int(target-sum(results))) )

    for i in range(0, int(target-sum(results))):
        #print ( "add 1 to results["+str(li[i][0])+"] ("+str(li[i][1])+") value before == "+ str(results[li[i][0]]) )
        results[li[i][0]] = results[li[i][0]] +1.0

    return results





def rescale_array(a: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """
    Performs edge trimming on values of the input vector and constraints them between within frac and 1-frac density of
    normal distribution created with the sample mean and std. dev. of a

    :param a: numeric vector
    :param frac: Value between 0 and 1.
    :return:
    """
    from scipy.stats import norm

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


def controlled_compute(arr, nthreads):
    from multiprocessing.pool import ThreadPool
    import dask

    with dask.config.set(schedular='threads', pool=ThreadPool(nthreads)):
        res = arr.compute()
    return res


def show_progress(arr, msg: str = None, nthreads: int = 1):
    from dask.diagnostics import ProgressBar

    if msg is not None:
        print(msg, flush=True)
    pbar = ProgressBar()
    pbar.register()
    res = controlled_compute(arr, nthreads)
    pbar.unregister()
    return res


# def show_progress(func: Callable):
#     from dask.diagnostics import ProgressBar
#     import functools
#
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         pbar = ProgressBar()
#         pbar.register()
#         ret_val = func(*args, **kwargs)
#         pbar.unregister()
#         return ret_val
#     return wrapper


# def calc_computed(a, msg: str = None):
#     from dask.distributed import progress
#
#     if msg is not None:
#         print(msg, flush=True)
#     a = a.persist()
#     progress(a, notebook=False)
#     print(flush=True)
#     return a.compute()


def system_call(command):
    import subprocess
    import shlex

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip())
    process.poll()
    return None
