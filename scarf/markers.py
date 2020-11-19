from .assay import Assay
from .utils import controlled_compute
from .logging_utils import logger
from numba import jit
import numpy as np
import pandas as pd
from tqdm import tqdm


__all__ = ['find_markers_by_rank']


def find_markers_by_rank(assay: Assay, group_key: str, cell_key: str,
                         nthreads: int, threshold: float = 0.25) -> dict:

    @jit()
    def calc_mean_rank(v):
        r = np.ones(n_groups)
        for x in range(n_groups):
            r[x] = v[int_indices == x].mean()
        return r / r.sum()

    def mean_rank_wrapper(v):
        return calc_mean_rank(v.values)

    groups = assay.cells.fetch(group_key, cell_key)
    group_set = sorted(set(groups))
    n_groups = len(group_set)
    # Since, numba needs int arrays to work properly but the dtype of 'groups' may not be integer type
    # Hence we need to create a indexed version of 'groups'
    idx_map = dict(zip(group_set, range(n_groups)))
    rev_idx_map = {v:k for k,v in idx_map.items()}
    int_indices = np.array([idx_map[x] for x in groups])

    # We create dask array of the normalized data. The array is transposed in order to allow chunk-wise iteration
    # over genes rather than cells.
    data = assay.normed(cell_idx=assay.cells.active_index(cell_key)).T
    # All the genes to be tested are split into arrays. Each array being of same size as the transposed chunk
    # dimension
    gene_names = np.split(assay.feats.fetch('ids'), np.cumsum(data.chunks[0]))[:-1]
    results = {x: [] for x in group_set}
    
    for i, names in tqdm(zip(data.blocks, gene_names), desc='Finding markers', total=data.numblocks[0]):
        val = pd.DataFrame(controlled_compute(i, nthreads), index=names).T.rank(method='dense').astype(int)
        res = val.apply(mean_rank_wrapper)
        # Removing genes that were below the threshold in all the groups
        res = res.T[(res < threshold).sum() != n_groups]
        for j in res:
            results[rev_idx_map[j]].append(res[j][res[j] > threshold])
    for i in results:
        results[i] = pd.concat(results[i]).sort_values(ascending=False)
    return results
