from .assay import Assay
from .utils import controlled_compute

__all__ = ['find_markers_by_rank']


def find_markers_by_rank(assay: Assay, group_key: str, subset_key: str, nthreads: int, threshold: float = 0.25) -> dict:
    from numba import jit
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    @jit()
    def calc_mean_rank(v):
        #  TODO: fix for non-numeric index set
        r = np.ones(n_indices)
        for x in range(n_indices):
            r[x] = v[int_index_set == x].mean()
        return r / r.sum()

    def mean_rank_wrapper(v):
        return calc_mean_rank(v.values)

    c_idx = assay.cells.active_index(subset_key)
    indices = assay.cells.fetch(group_key, subset_key)
    index_set = np.array(sorted(set(indices)))
    n_indices = len(index_set)
    idx_map = dict(zip(index_set, range(n_indices)))
    int_index_set = np.array([idx_map[x] for x in index_set])

    data = assay.normed(cell_idx=c_idx).T
    gene_names = np.split(assay.feats.fetch('ids'), np.cumsum(data.chunks[0]))[:-1]
    results = {x: [] for x in index_set}
    for i, names in tqdm(zip(data.blocks, gene_names), desc='Finding markers', total=data.numblocks[0]):
        val = pd.DataFrame(controlled_compute(i, nthreads), index=names).T.rank(method='dense').astype(int)
        res = val.apply(mean_rank_wrapper)
        res = res.T[(res < threshold).sum() != n_indices].T
        res.index = index_set
        # The following line was commented out to return results as gene IDs rather than gene names
        # res.columns = assay.feats.table.names[assay.feats.table.ids.isin(res.columns)].values
        res = res.T
        for j in res:
            results[j].append(res[j][res[j] > threshold])
    for i in results:
        results[i] = pd.concat(results[i]).sort_values(ascending=False)
    return results
