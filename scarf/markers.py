"""Module to find biomarkers."""

from typing import Optional

import numpy as np
import pandas as pd
from numba import jit
from scipy.stats import linregress
from scipy.stats import rankdata

from scarf.assay import Assay
from scarf.utils import logger, tqdmbar


def read_prenormed_batches(store, cell_idx: np.ndarray, batch_size: int, desc: str):
    batch = {}
    for i in tqdmbar(store.keys(), desc=desc):
        batch[int(i)] = store[i][:][cell_idx]
        if len(batch) == batch_size:
            yield pd.DataFrame(batch)
            batch = {}
    if len(batch) > 0:
        yield pd.DataFrame(batch)


def find_markers_by_rank(
    assay: Assay,
    group_key: str,
    cell_key: str,
    feat_key: str,
    batch_size: int,
    use_prenormed: bool,
    prenormed_store: Optional[str],
    n_threads: int,
    **norm_params,
) -> dict:
    """Identify marker genes/features for given groups using a rank-based approach.

    Args:
        assay: An Assay object containing the data to analyze (accessed via iter_normed_feature_wise) 
        group_key: Column name in cell metadata containing group labels
        cell_key: Column name in cell metadata indicating which cells to use
        feat_key: Column name in feature metadata indicating which features to analyze
        batch_size: Number of features to process at once for memory efficiency
        use_prenormed: Whether to use pre-normalized data if available
        prenormed_store: Name of the store containing pre-normalized data
        n_threads: Number of threads to use for parallel processing
        **norm_params: Additional parameters to pass to normalization functions

    Returns:
        dict: Dictionary containing marker analysis results for each group, with statistics
              like fold changes, p-values, and effect sizes
    """

    from joblib import Parallel, delayed

    def calc(vdf):
        r = vdf.rank(method="dense").groupby(groups).mean().reindex(group_set)
        r = r / r.sum()

        g = np.array([pd.Series(groups).value_counts().reindex(group_set).values]).T
        g_o = len(groups) - g

        s = vdf.groupby(groups).sum().reindex(group_set)
        m = s / g
        m_o = (s.sum() - s) / g_o

        s = (vdf > 0).groupby(groups).sum().reindex(group_set)
        e = s / g
        e_o = (s.sum() - s) / g_o

        fc = (m / m_o).fillna(0)

        return np.array(
            [r.values, m.values, m_o.values, e.values, e_o.values, fc.values]
        ).T

    @jit(nopython=True)
    def calc_rank_mean(v):
        """Calculates the mean rank of the data."""
        r = np.ones(n_groups)
        for x in range(n_groups):
            r[x] = v[int_indices == x].mean()
        return r / r.sum()

    @jit(nopython=True)
    def calc_frac_fc(v):
        """Calculates the mean rank of the data."""
        m = np.zeros(n_groups)
        m_o = np.zeros(n_groups)
        e = np.zeros(n_groups)
        e_o = np.zeros(n_groups)
        fc = np.zeros(n_groups)
        for x in range(n_groups):
            i = int_indices == x
            m[x] = v[i].mean()
            m_o[x] = v[~i].mean()
            e[x] = v[i].nonzero()[0].shape[0] / i.sum()
            e_o[x] = v[~i].nonzero()[0].shape[0] / (i.shape[0] - i.sum())
            if m_o[x] == 0:
                fc[x] = 100.100
            else:
                fc[x] = m[x] / (m_o[x])
        return m, m_o, e, e_o, fc

    def prenormed_mean_rank_wrapper(gene_idx):
        d = prenormed_store[gene_idx][:][cell_idx]
        r = calc_rank_mean(rankdata(d, method="dense"))
        m, m_o, e, e_o, fc = calc_frac_fc(d)
        return gene_idx, np.vstack([r, m, m_o, e, e_o, fc])

    groups = assay.cells.fetch(group_key, cell_key)
    group_set = np.array(sorted(set(groups)))
    n_groups = len(group_set)
    idx_map = dict(zip(group_set, range(n_groups)))
    int_indices = np.array([idx_map[x] for x in groups])
    out_cols = [
        "feature_index",
        "score",
        "mean",
        "mean_rest",
        "frac_exp",
        "frac_exp_rest",
        "fold_change",
    ]
    if use_prenormed:
        if prenormed_store is None:
            if "prenormed" in assay.z:
                prenormed_store = assay.z["prenormed"]
            else:
                raise ValueError(
                    "Could not find prenormed values. Run with use_prenormed=False or create pre-normed values."
                )

        results = {x: [] for x in group_set}
        cell_idx = assay.cells.active_index(cell_key)
        batch_iterator = tqdmbar(prenormed_store.keys(), desc="Finding markers")
        temp = Parallel(n_jobs=n_threads)(
            delayed(prenormed_mean_rank_wrapper)(i) for i in batch_iterator
        )
        for i in temp:
            for j, k in zip(group_set, i[1].T):
                results[j].append([i[0]] + list(k))
        for i in results:
            results[i] = (
                pd.DataFrame(results[i], columns=out_cols)
                .sort_values(by="score", ascending=False)
                .round(5)
            )
        return results
    else:
        batch_iterator = assay.iter_normed_feature_wise(
            cell_key=cell_key,
            feat_key=feat_key,
            batch_size=batch_size,
            msg="Finding markers",
            **norm_params,
        )
        temp = np.vstack([calc(x) for x in batch_iterator])
        results = {}
        feat_index = assay.feats.active_index(feat_key)
        for n, i in enumerate(group_set):
            results[i] = (
                pd.DataFrame(temp[:, n, :], columns=out_cols[1:], index=feat_index)
                .sort_values(by="score", ascending=False)
                .round(5)
            )
            results[i]["feature_index"] = results[i].index
            results[i] = results[i][out_cols]
        return results


def find_markers_by_regression(
    assay: Assay,
    cell_key: str,
    feat_key: str,
    regressor: np.ndarray,
    min_cells: int,
    batch_size: int = 50,
    **norm_params,
) -> pd.DataFrame:
    """Find features that correlate with a continuous variable using linear regression.

    Args:
        assay: An Assay object containing the data to analyze
        cell_key: Column name in cell metadata indicating which cells to use
        feat_key: Column name in feature metadata indicating which features to analyze
        regressor: 1D numpy array containing the continuous variable to correlate against
        min_cells: Minimum number of cells where feature must be expressed to be analyzed
        batch_size: Number of features to process at once for memory efficiency
        **norm_params: Additional parameters to pass to normalization functions

    Returns:
        pd.DataFrame: DataFrame containing correlation results with columns:
            - r_value: Pearson correlation coefficient
            - p_value: Statistical significance of correlation
    """

    res = {}
    for df in assay.iter_normed_feature_wise(
        cell_key=cell_key,
        feat_key=feat_key,
        batch_size=batch_size,
        msg="Finding correlated features",
        **norm_params,
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


def knn_clustering(
    d_array, n_neighbours: int, n_clusters: int, n_threads: int, ann_params: dict = None
) -> np.ndarray:
    """

    Args:
        d_array: 2D numpy array of data to cluster (n_samples x n_features)
        n_neighbours: Number of nearest neighbors to use for building the graph
        n_clusters: Number of clusters to generate
        n_threads: Number of threads to use for parallel processing
        ann_params: Dictionary of parameters for approximate nearest neighbor search.
                   See default_ann_params in function for available options.

    Returns:
        np.ndarray: 1D array of cluster assignments (integers from 1 to n_clusters)
    """

    from .ann import instantiate_knn_index, fix_knn_query
    from .utils import controlled_compute, tqdmbar, show_dask_progress
    from scipy.sparse import csr_matrix

    def make_knn_mat(data, k, t):
        """Create a sparse KNN adjacency matrix from the input data.

        Args:
            data: Input data array to build KNN graph from
            k: Number of nearest neighbors to find for each point
            t: Number of threads to use for parallel processing

        Returns:
            scipy.sparse.csr_matrix: Sparse adjacency matrix representing the KNN graph
        """

        for i in tqdmbar(data.blocks, desc="Fitting KNNs", total=data.numblocks[0]):
            i = controlled_compute(i, t)
            ann_idx.add_items(i)
        s, e = 0, 0
        indices = []
        for i in tqdmbar(
            data.blocks, desc="Identifying feature KNNs", total=data.numblocks[0]
        ):
            e += i.shape[0]
            i = controlled_compute(i, t)
            inds, d = ann_idx.knn_query(i, k=k + 1)
            inds, _, _ = fix_knn_query(inds, d, np.arange(s, e))
            indices.append(inds)
            s = e
        indices = np.vstack(indices)
        assert indices.shape[0] == data.shape[0]

        return csr_matrix(
            (
                np.ones(indices.shape[0] * indices.shape[1]),
                (
                    np.repeat(range(indices.shape[0]), indices.shape[1]),
                    indices.flatten(),
                ),
            ),
            shape=(indices.shape[0], indices.shape[0]),
        )

    def make_clusters(mat, nc):
        """Generate clusters from a KNN adjacency matrix using hierarchical clustering.

        Args:
            mat: Sparse adjacency matrix representing the KNN graph
            nc: Number of clusters to generate

        Returns:
            np.ndarray: Cluster assignments for each point
        """
        import sknetwork as skn

        paris = skn.hierarchy.Paris(reorder=False)
        logger.info("Performing clustering, this might take a while...")
        dendrogram = paris.fit_transform(mat)
        return skn.hierarchy.cut_straight(dendrogram, n_clusters=nc)

    def fix_cluster_order(data, clusters, t):
        """Reorder cluster labels based on feature expression patterns.

        Args:
            data: Original data array used for clustering
            clusters: Initial cluster assignments
            t: Number of threads to use for parallel processing

        Returns:
            np.ndarray: Reordered cluster assignments (1-based indexing)
        """

        idxmax = show_dask_progress(data.argmax(axis=1), "Sorting clusters", t)
        cmm = pd.DataFrame([idxmax, clusters]).T.groupby(1).median()[0].sort_values()
        return (
            pd.Series(clusters)
            .replace(dict(zip(cmm.index, range(1, 1 + len(cmm)))))
            .values
        )

    default_ann_params = {
        "space": "l2",
        "dim": d_array.shape[1],
        "max_elements": d_array.shape[0],
        "ef_construction": 80,
        "M": 50,
        "random_seed": 444,
        "ef": 80,
        "num_threads": 1,
    }
    if ann_params is None:
        ann_params = {}
    default_ann_params.update(ann_params)
    ann_idx = instantiate_knn_index(**default_ann_params)
    return fix_cluster_order(
        d_array,
        make_clusters(make_knn_mat(d_array, n_neighbours, n_threads), n_clusters),
        n_threads,
    )
