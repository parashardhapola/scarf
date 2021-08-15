"""
Module to find biomarkers.
"""
from .assay import Assay
from .utils import logger
from numba import jit
import numpy as np
import pandas as pd
from scipy.stats import linregress


__all__ = [
    "find_markers_by_rank",
    "find_markers_by_regression",
    "knn_clustering",
]


def find_markers_by_rank(
    assay: Assay,
    group_key: str,
    cell_key: str,
    threshold: float = 0.25,
    batch_size: int = 50,
    **norm_params,
) -> dict:
    """

    Args:
        assay:
        group_key:
        cell_key:
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
    for val in assay.iter_normed_feature_wise(
        cell_key,
        "I",
        batch_size,
        "Finding markers",
        **norm_params,
    ):
        val.index = feature_ids
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
    min_cells: int,
    batch_size: int = 50,
    **norm_params,
) -> pd.DataFrame:
    """

    Args:
        assay:
        cell_key:
        regressor:
        min_cells:
        batch_size:

    Returns:

    """

    feature_ids = assay.feats.fetch("ids")
    res = {}
    for df in assay.iter_normed_feature_wise(
        cell_key,
        "I",
        batch_size,
        "Finding correlated features",
        **norm_params,
    ):
        df.index = feature_ids
        for i in df:
            v = df[i].values
            if (v > 0).sum() > min_cells:
                lin_obj = linregress(regressor, v)
                res[i] = (lin_obj.rvalue, lin_obj.pvalue)
            else:
                res[i] = (0, 1)
    res = pd.DataFrame(res, index=["r_value", "p_value"]).T
    return res


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
    logger.info("Performing clustering, this might take a while...")
    return fix_cluster_order(make_clusters(make_knn_mat()))
