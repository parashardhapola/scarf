"""
Module to find biomarkers.
"""
from .assay import Assay
from .utils import controlled_compute, tqdmbar
from numba import jit
import numpy as np
import pandas as pd


__all__ = ["find_markers_by_rank"]


def find_markers_by_rank(
    assay: Assay,
    group_key: str,
    cell_key: str,
    nthreads: int,
    threshold: float = 0.25,
    gene_batch_size: int = 50,
) -> dict:
    """

    Args:
        assay:
        group_key:
        cell_key:
        nthreads:
        threshold:
        gene_batch_size:

    Returns:

    """

    @jit(nopython=True)
    def calc_mean_rank(v):
        """
        Calculates the mean rank of the data.
        """
        r = np.ones(n_groups)
        for x in range(n_groups):
            r[x] = v[int_indices == x].mean()
        return r / r.sum()

    def mean_rank_wrapper(v):
        """
        Wraps `calc_mean_rank` function.
        """
        return calc_mean_rank(v.values)

    groups = assay.cells.fetch(group_key, cell_key)
    group_set = sorted(set(groups))
    n_groups = len(group_set)
    # Since, numba needs int arrays to work properly but the dtype of 'groups' may not be integer type
    # Hence we need to create a indexed version of 'groups'
    idx_map = dict(zip(group_set, range(n_groups)))
    rev_idx_map = {v: k for k, v in idx_map.items()}
    int_indices = np.array([idx_map[x] for x in groups])

    data = assay.normed(cell_idx=assay.cells.active_index(cell_key))
    gene_ids = assay.feats.fetch("ids")
    chunks = np.array_split(
        np.arange(0, data.shape[1]), int(data.shape[1] / gene_batch_size)
    )

    results = {x: [] for x in group_set}
    for chunk in tqdmbar(chunks, desc="Finding markers", total=len(chunks)):
        val = (
            pd.DataFrame(
                controlled_compute(data[:, chunk], nthreads), columns=gene_ids[chunk]
            )
            .rank(method="dense")
            .astype(int)
        )
        res = val.apply(mean_rank_wrapper)
        # Removing genes that were below the threshold in all the groups
        res = res.T[(res < threshold).sum() != n_groups]
        for j in res:
            results[rev_idx_map[j]].append(res[j][res[j] > threshold])
    for i in results:
        results[i] = pd.concat(results[i]).sort_values(ascending=False)
    return results
