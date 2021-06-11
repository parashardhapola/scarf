"""
Utility methods.

- Methods:
    - clean_array: returns input array with nan and infinite values removed
    - controlled_compute: performs computation with Dask
    - rescale_array: performs edge trimming on values of the input vector
    - show_progress: performs computation with Dask and shows progress bar
    - system_call: executes a command in the underlying operative system
"""

import numpy as np
from .logging_utils import logger

__all__ = ['system_call', 'rescale_array', 'clean_array', 'show_progress', 'controlled_compute']


def rescale_array(a: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """
    Performs edge trimming on values of the input vector.

    Performs edge trimming on values of the input vector and constraints them between within frac and 1-frac density of
    normal distribution created with the sample mean and std. dev. of a.

    Args:
        a: numeric vector
        frac: Value between 0 and 1.

    Return:
        The input array, edge trimmed and constrained.
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
    Returns input array with nan and infinite values removed.

    Args:
        x (np.ndarray): input array
        fill_val: value to fill zero values with (default: 0)
    """
    x = np.nan_to_num(x, copy=True)
    x[(x == np.Inf) | (x == -np.Inf)] = 0
    x[x == 0] = fill_val
    return x


def controlled_compute(arr, nthreads):
    """
    Performs computation with Dask.

    Args:
        arr:
        nthreads: number of threads to use for computation

    Returns:
        Result of computation.
    """
    from multiprocessing.pool import ThreadPool
    import dask

    with dask.config.set(schedular='threads', pool=ThreadPool(nthreads)):
        res = arr.compute()
    return res


def show_progress(arr, msg: str = None, nthreads: int = 1):
    """
    Performs computation with Dask and shows progress bar.

    Args:
        arr:
        msg: message to log, default None
        nthreads: number of threads to use for computation, default 1

    Returns:
        Result of computation.
    """
    from dask.diagnostics import ProgressBar

    if msg is not None:
        logger.info(msg)
    pbar = ProgressBar()
    pbar.register()
    res = controlled_compute(arr, nthreads)
    pbar.unregister()
    return res


def system_call(command):
    """Executes a command in the underlying operative system."""
    import subprocess
    import shlex

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            logger.info(output.strip())
    process.poll()
    return None
