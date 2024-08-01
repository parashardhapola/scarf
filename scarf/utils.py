"""Utility methods.

- Methods:
    - clean_array: returns input array with nan and infinite values removed
    - controlled_compute: performs computation with Dask
    - rescale_array: performs edge trimming on values of the input vector
    - show_progress: performs computation with Dask and shows progress bar
    - system_call: executes a command in the underlying operative system
    - rolling_window: applies rolling window mean over a vector
"""

import sys
from typing import Union, Optional, TypeAlias

import numpy as np
import zarr
from dask.array.core import Array
from loguru import logger
from numba import jit
from tqdm.dask import TqdmCallback

__all__ = [
    "logger",
    "tqdmbar",
    "tqdm_params",
    "set_verbosity",
    "get_log_level",
    "system_call",
    "rescale_array",
    "clean_array",
    "load_zarr",
    "show_dask_progress",
    "controlled_compute",
    "rolling_window",
]

logger.remove()
logger.add(
    sys.stdout, colorize=True, format="<level>{level}</level>: {message}", level="INFO"
)

tqdm_params = {
    "bar_format": "{desc}: {percentage:3.0f}%| {bar} {n_fmt}/{total_fmt} [{elapsed}]",
    "ncols": 500,
    "colour": "#34abeb",
}

ZARRLOC: TypeAlias = Union[str, zarr.LRUStoreCache]


def get_log_level():
    # noinspection PyUnresolvedReferences
    return logger._core.min_level  # type: ignore


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


def tqdmbar(*args, **kwargs):
    params = dict(tqdm_params)
    for i in kwargs:
        if i in params:
            del params[i]
    if "disable" not in kwargs and "disable" not in params:
        if get_log_level() <= 20:
            params["disable"] = False
        else:
            params["disable"] = True
    if is_notebook():
        from tqdm import tqdm_notebook

        return tqdm_notebook(*args, **kwargs, **params)
    else:
        from tqdm.auto import tqdm

        return tqdm(*args, **kwargs, **params)


def set_verbosity(level: Optional[str] = None, filepath: Optional[str] = None):
    """Set verbosity level of Scarf's output. Setting value of level='CRITICAL'
    should silence all logs. Progress bars are automatically disabled for
    levels above 'INFO'.

    Args:
        level: A valid level name. Run without any parameter to see available options
        filepath: The output file path. All logs will be saved to this file. If no file path is
                  is provided then all the logs are printed on standard output.

    Returns:
    """
    # noinspection PyUnresolvedReferences
    available_levels = logger._core.levels.keys()  # type: ignore

    if level is None or level not in available_levels:
        raise ValueError(
            f"Please provide a value for level: {', '.join(available_levels)}"
        )
    logger.remove()
    if filepath is None:
        filepath = sys.stdout  # type: ignore
    logger.add(
        filepath,  # type: ignore
        colorize=True,
        format="<level>{level}</level>: {message}",
        level=level,
    )


def rescale_array(a: np.ndarray, frac: float = 0.9) -> np.ndarray:
    """Performs edge trimming on values of the input vector.

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
    """Returns input array with nan and infinite values removed.

    Args:
        x (np.ndarray): input array
        fill_val: value to fill zero values with (default: 0)
    """
    x = np.nan_to_num(x, copy=True)
    x[(x == np.inf) | (x == -np.inf)] = 0
    x[x == 0] = fill_val
    return x


def load_zarr(
    zarr_loc: ZARRLOC, mode: str, synchronizer: Optional[zarr.ThreadSynchronizer] = None
) -> zarr.Group:
    if synchronizer is None:
        synchronizer = zarr.ThreadSynchronizer()
    if isinstance(zarr_loc, str):
        return zarr.open_group(zarr_loc, mode=mode, synchronizer=synchronizer)
    else:
        return zarr.group(zarr_loc, synchronizer=synchronizer)


def controlled_compute(arr, nthreads):
    """Performs computation with Dask.

    Args:
        arr:
        nthreads: number of threads to use for computation

    Returns:
        Result of computation.
    """
    import dask

    try:
        # Multiprocessing may be faster, but it throws exception if SemLock is not implemented.
        # For example, multiprocessing won't work on AWS Lambda, in those scenarios we switch ThreadPoolExecutor
        from multiprocessing.pool import ThreadPool

        pool = ThreadPool(nthreads)
    except Exception:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(nthreads)

    with dask.config.set(schedular="threads", pool=pool):  # type: ignore
        res = arr.compute()
    return res


def show_dask_progress(arr: Array, msg: Optional[str] = None, nthreads: int = 1):
    """Performs computation with Dask and shows progress bar.

    Args:
        arr: A Dask array
        msg: message to log, default None
        nthreads: number of threads to use for computation, default 1

    Returns:
        Result of computation.
    """

    params = dict(tqdm_params)
    if "disable" not in params:
        if get_log_level() <= 20:
            params["disable"] = False
        else:
            params["disable"] = True
    with TqdmCallback(desc=msg, **params):
        res = controlled_compute(arr, nthreads)
    return res


def system_call(command):
    """Executes a command in the underlying operative system."""
    import subprocess
    import shlex

    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()  # type: ignore
        if process.poll() is not None:
            break
        if output:
            logger.info(output.strip())
    process.poll()
    return None


@jit(nopython=True)
def rolling_window(a, w):
    n, m = a.shape
    b = np.zeros(shape=(n, m))
    for i in range(n):
        if i < w:
            x = i
            y = w - i
        elif (n - i) < w:
            x = w - (n - i)
            y = n - i
        else:
            x = w // 2
            y = w // 2
        x = i - x
        y = i + y
        for j in range(m):
            b[i, j] = a[x:y, j].mean()
    return b
