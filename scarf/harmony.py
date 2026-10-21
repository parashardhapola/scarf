"""Batch correction with Harmony (Korsunsky et al. 2019).

Scarf does not include an implementation of Harmony. It calls the `harmonypy` package, which is
licensed under the GPL-3.0-or-later and is installed only with the optional `harmony` extra.
"""

import numpy as np
import pandas as pd


def import_harmonypy():
    """Imports `harmonypy`, or raises an ImportError that explains how to install it."""
    try:
        import harmonypy
    except ImportError:
        raise ImportError(
            "ERROR: 'harmonypy' package is not installed. Scarf uses it for Harmony batch "
            "correction and LISI scores, but does not install it by default because it is "
            "licensed under the GPL-3.0-or-later. Install it with: pip install scarf[harmony]"
        )
    return harmonypy


def run_harmony(
    data_mat: np.ndarray,
    meta_data: pd.DataFrame,
    theta=None,
    lamb=None,
    sigma=0.1,
    nclust=None,
    tau=0,
    block_size=0.05,
    max_iter_harmony=50,
    max_iter_kmeans=20,
    epsilon_cluster=1e-5,
    epsilon_harmony=1e-4,
    random_state=0,
    cluster_fn="kmeans",
) -> np.ndarray:
    """Run Harmony.

    Args:
        data_mat: Values to correct, with dimensions in rows and cells in columns.
        meta_data: Batch labels of the cells, one column per batch variable.

    The remaining parameters are passed to `harmonypy.run_harmony`.

    Returns:
        The corrected values, with dimensions in rows and cells in columns.
    """
    harmonypy = import_harmonypy()
    return harmonypy.run_harmony(
        data_mat,
        meta_data,
        list(meta_data.columns),
        theta=theta,
        lamb=lamb,
        sigma=sigma,
        nclust=nclust,
        tau=tau,
        block_size=block_size,
        max_iter_harmony=max_iter_harmony,
        max_iter_kmeans=max_iter_kmeans,
        epsilon_cluster=epsilon_cluster,
        epsilon_harmony=epsilon_harmony,
        random_state=random_state,
        cluster_fn=cluster_fn,
    ).Z_corr
