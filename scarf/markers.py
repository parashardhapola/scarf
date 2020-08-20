__all__ = ['find_markers_by_rank']


def find_markers_by_rank(assay, group_key: str, threshold: float = 0.25) -> dict:
    from numba import jit
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    @jit()
    def calc_mean_rank(v):
        n_indices = index_set.shape[0]
        r = np.ones(n_indices)
        for x in range(n_indices):
            r[x] = v[indices == index_set[x]].mean()
        return r / r.sum()

    def mean_rank_wrapper(v):
        return calc_mean_rank(v.values)

    indices = assay.cells.fetch(group_key)
    index_set = np.array(sorted(set(indices)))
    data = assay.normed().T
    gene_names = np.split(assay.feats.fetch('ids'), np.cumsum(data.chunks[0]))[:-1]
    results = {x: [] for x in index_set}
    for i, names in tqdm(zip(data.blocks, gene_names), desc='Finding markers', total=data.numblocks[0]):
        val = pd.DataFrame(i.compute(), index=names).T.rank(method='dense').astype(int)
        res = val.apply(mean_rank_wrapper)
        res = res.T[(res < threshold).sum() != len(index_set)].T
        res.index = index_set
        # The following line was commented out to return results as gene IDs rather than gene names
        # res.columns = assay.feats.table.names[assay.feats.table.ids.isin(res.columns)].values
        res = res.T
        for j in res:
            results[j].append(res[j][res[j] > threshold])
    for i in results:
        results[i] = pd.concat(results[i]).sort_values(ascending=False)
    return results
