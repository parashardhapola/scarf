"""
Module to find biomarkers.
"""
from .assay import Assay
from .utils import controlled_compute, tqdmbar
from numba import jit
import numpy as np
import pandas as pd
from scipy.stats import linregress


__all__ = [
    "find_markers_by_rank",
    "find_markers_by_regression",
    "aggregate_over_cell_ordering",
    "knn_clustering",
]


def iter_data_feature_wise(
    assay, cell_key, feat_key, feature_ids, batch_size, nthreads, msg
):
    data = assay.normed(
        cell_idx=assay.cells.active_index(cell_key),
        feat_key=assay.feats.active_index(feat_key),
    )
    chunks = np.array_split(
        np.arange(0, data.shape[1]), int(data.shape[1] / batch_size)
    )
    if len(feature_ids) != data.shape[1]:
        raise ValueError(
            "ERROR: The number of provided feature ids is not same as the requested normed data"
        )
    for chunk in tqdmbar(chunks, desc=msg, total=len(chunks)):
        yield pd.DataFrame(
            controlled_compute(data[:, chunk], nthreads), columns=feature_ids[chunk]
        )


def find_markers_by_rank(
    assay: Assay,
    group_key: str,
    cell_key: str,
    nthreads: int,
    threshold: float = 0.25,
    batch_size: int = 50,
) -> dict:
    """

    Args:
        assay:
        group_key:
        cell_key:
        nthreads:
        threshold:
        batch_size:

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
    feature_ids = assay.feats.fetch("ids", key="I")
    results = {x: [] for x in group_set}
    for val in iter_data_feature_wise(
        assay, cell_key, "I", feature_ids, batch_size, nthreads, "Finding markers"
    ):
        res = val.rank(method="dense").astype(int).apply(mean_rank_wrapper)
        # Removing genes that were below the threshold in all the groups
        res = res.T[(res < threshold).sum() != n_groups]
        for j in res:
            results[rev_idx_map[j]].append(res[j][res[j] > threshold])
    for i in results:
        results[i] = pd.concat(results[i]).sort_values(ascending=False)
    return results


def find_markers_by_regression(
    assay: Assay,
    cell_key: str,
    regressor: np.ndarray,
    nthreads: int,
    min_cells: int,
    batch_size: int = 50,
) -> pd.DataFrame:
    """

    Args:
        assay:
        cell_key:
        regressor:
        nthreads:
        min_cells:
        batch_size:

    Returns:

    """

    feature_ids = assay.feats.fetch("ids")
    res = {}
    for df in iter_data_feature_wise(
        assay,
        cell_key,
        "I",
        feature_ids,
        batch_size,
        nthreads,
        "Finding correlated features",
    ):
        for i in df:
            v = df[i].values
            if (v > 0).sum() > min_cells:
                lin_obj = linregress(regressor, v)
                res[i] = (lin_obj.rvalue, lin_obj.pvalue)
            else:
                res[i] = (0, 1)
    res = pd.DataFrame(res, index=["r_value", "p_value"]).T
    return res


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


def aggregate_over_cell_ordering(
    assay,
    cell_key: str,
    feat_key: str,
    cell_ordering: np.ndarray,
    feature_ids: np.ndarray,
    min_exp: float = 10,
    window_size: int = 200,
    chunk_size: int = 50,
    smoothen: bool = True,
    z_scale: bool = True,
    nthreads: int = 2,
    batch_size: int = 100,
):
    """

    Args:
        assay:
        cell_key:
        feat_key:
        cell_ordering:
        feature_ids:
        min_exp:
        window_size:
        chunk_size:
        smoothen:
        z_scale:
        nthreads:
        batch_size:

    Returns:

    """

    agg = []
    if cell_ordering.shape[0] != assay.cells.fetch(cell_key).sum():
        raise ValueError(
            "ERROR: The number of cells in `cell_ordering` not same as valid cells in `cell_key`"
        )
    idx = np.argsort(cell_ordering)
    for df in iter_data_feature_wise(
        assay,
        cell_key,
        feat_key,
        feature_ids,
        batch_size,
        nthreads,
        "Smoothening features",
    ):
        valid_genes = df.columns[df.sum() > min_exp]
        df = df[valid_genes]
        if smoothen:
            df = rolling_window(df.reindex(idx).values, window_size)
        if z_scale:
            df = (df - df.mean(axis=0)) / df.std(axis=0)
        df = np.array([x.mean(axis=0) for x in np.array_split(df, chunk_size)])
        agg.append(pd.DataFrame(df, columns=valid_genes).T)
    return pd.concat(agg)


def knn_clustering(df, k, n_clusts, ann_params=None):
    """

    Args:
        df:
        k:
        n_clusts:
        ann_params:

    Returns:

    """

    from scarf.ann import instantiate_knn_index

    def make_knn_mat():
        """

        Returns:

        """
        from scarf.ann import fix_knn_query
        from scipy.sparse import csr_matrix

        ann_idx.add_items(df)
        inds, d = ann_idx.knn_query(df, k=k + 1)
        inds, _, _ = fix_knn_query(inds, d, np.array(range(df.shape[0])))
        return csr_matrix(
            (
                np.ones(inds.shape[0] * inds.shape[1]),
                (np.repeat(range(inds.shape[0]), inds.shape[1]), inds.flatten()),
            ),
            shape=(inds.shape[0], inds.shape[0]),
        )

    def make_clusters(mat):
        """

        Args:
            mat:

        Returns:

        """
        import sknetwork as skn

        paris = skn.hierarchy.Paris(reorder=False)
        dendrogram = paris.fit_transform(mat)
        return skn.hierarchy.cut_straight(dendrogram, n_clusters=n_clusts)

    def fix_cluster_order(clusters):
        """

        Args:
            clusters:

        Returns:

        """
        cmm = (
            pd.DataFrame([df.idxmax(axis=1).values, clusters])
            .T.groupby(1)
            .median()[0]
            .sort_values()
        )
        updated_ids = (
            pd.Series(clusters)
            .replace(dict(zip(cmm.index, range(1, 1 + len(cmm)))))
            .values
        )
        idx = np.argsort(updated_ids)
        return df.iloc[idx], updated_ids[idx]

    default_ann_params = {
        "space": "l2",
        "dim": df.shape[1],
        "max_elements": df.shape[0],
        "ef_construction": 80,
        "M": 50,
        "random_seed": 444,
        "ef": 80,
        "num_threads": 2,
    }
    if ann_params is None:
        ann_params = {}
    default_ann_params.update(ann_params)
    ann_idx = instantiate_knn_index(**default_ann_params)
    return fix_cluster_order(make_clusters(make_knn_mat()))
