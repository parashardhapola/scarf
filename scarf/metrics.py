"""
Methods and classes for evluation
"""

import math
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import zarr
from dask.array import from_array
from dask.array.core import Array as daskArrayType
from scipy.io import mmwrite
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
from zarr.core import Array as zarrArrayType

from .assay import (
    ADTassay,
    ATACassay,
    RNAassay,
)
from .datastore.datastore import DataStore
from .metadata import MetaData
from .utils import (
    ZARRLOC,
    controlled_compute,
    load_zarr,
    logger,
    permute_into_chunks,
    tqdmbar,
)
from .writers import create_zarr_count_assay, create_zarr_obj_array


# LISI - The Local Inverse Simpson Index
def compute_lisi(
    # X: np.array,
    distances: zarrArrayType,
    indices: zarrArrayType,
    metadata: pd.DataFrame,
    label_colnames: Iterable[str],
    perplexity: float = 30,
):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI is a statistic computed for each item (row) in the data matrix X.

    The following example may help to interpret the LISI values.

    Suppose one of the columns in metadata is a categorical variable with 3 categories.

        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.

        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.

    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].

    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """
    # # We need at least 3 * n_neigbhors to compute the perplexity
    # knn = NearestNeighbors(n_neighbors = math.ceil(perplexity * 3), algorithm = 'kd_tree').fit(X)
    # distances, indices = knn.kneighbors(X)

    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    # Don't count yourself
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    # Save the result
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        logger.info(f"Computing LISI for {label}")
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(
            distances.T, indices.T, labels, n_categories, perplexity
        )
        lisi_df[:, i] = 1 / simpson
    # lisi_df = lisi_df.flatten()
    return lisi_df


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float = 1e-5,
):
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson


# SILHOUETTE SCORE - The Silhouette Score
def knn_to_csr_matrix(
    neighbor_indices: np.ndarray, neighbor_distances: np.ndarray
) -> csr_matrix:
    """
    Convert k-nearest neighbors data to a Compressed Sparse Row (CSR) matrix.

    Parameters:
    neighbor_indices : 2D array
        Indices of k-nearest neighbors for each data point.
    neighbor_distances : 2D array
        Distances to k-nearest neighbors for each data point.

    Returns:
    scipy.sparse.csr_matrix
        A sparse matrix representation of the KNN graph.
    """
    num_samples, num_neighbors = neighbor_indices.shape
    row_indices = np.repeat(np.arange(num_samples), num_neighbors)
    return csr_matrix(
        (neighbor_distances[:].flatten(), (row_indices, neighbor_indices[:].flatten())),
        shape=(num_samples, num_samples),
    )


def calculate_weighted_cluster_similarity(
    knn_graph: csr_matrix, cluster_labels: np.ndarray
) -> np.ndarray:
    """
    Calculate similarity between clusters based on shared weighted edges.

    Parameters:
    - knn_graph: CSR matrix representing the KNN graph
    - cluster_labels: 1D array with cluster/community index for each node. Contiguous and start from 1.

    Returns:
    - similarity_matrix: 2D numpy array with similarity scores between clusters
    """
    unique_cluster_ids = np.unique(cluster_labels)
    expected_cluster_ids = np.arange(0, len(unique_cluster_ids))
    assert np.array_equal(
        unique_cluster_ids, expected_cluster_ids
    ), "Cluster labels must be contiguous integers starting at 1"

    num_clusters = len(unique_cluster_ids)
    inter_cluster_weights = np.zeros((num_clusters, num_clusters))

    for cluster_id in unique_cluster_ids:
        nodes_in_cluster = np.where(cluster_labels == cluster_id)[0]
        neighbor_cluster_labels = cluster_labels[knn_graph[nodes_in_cluster].indices]
        neighbor_edge_weights = knn_graph[nodes_in_cluster].data

        for neighbor_cluster, edge_weight in zip(
            neighbor_cluster_labels, neighbor_edge_weights
        ):
            inter_cluster_weights[cluster_id, neighbor_cluster] += edge_weight

    assert inter_cluster_weights.sum() == knn_graph.data.sum()

    # Ensure symmetry
    inter_cluster_weights = (inter_cluster_weights + inter_cluster_weights.T) / 2

    # Calculate total weights for each cluster
    total_cluster_weights = np.array(
        [inter_cluster_weights[i - 1].sum() for i in unique_cluster_ids]
    )

    # Calculate similarity using weighted Jaccard index
    similarity_matrix = np.zeros((num_clusters, num_clusters))

    for i in range(num_clusters):
        for j in range(i, num_clusters):
            weight_union = (
                total_cluster_weights[i]
                + total_cluster_weights[j]
                - inter_cluster_weights[i, j]
            )
            if weight_union > 0:
                similarity = inter_cluster_weights[i, j] / weight_union
                similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    # Set diagonal to 1 (self-similarity)
    # np.fill_diagonal(similarity_matrix, 1.0)

    return similarity_matrix


def calculate_top_k_neighbor_distances(
    matrix_a: np.ndarray, matrix_b: np.ndarray, k: int
) -> np.ndarray:
    """
    Calculate the distances of the top k nearest neighbors from matrix_b for each point in matrix_a.

    Parameters:
    matrix_a : numpy.ndarray
        First matrix of shape (m, d)
    matrix_b : numpy.ndarray
        Second matrix of shape (n, d)
    k : int
        Number of nearest neighbors to consider

    Returns:
    numpy.ndarray
        Array of shape (m, k) containing the distances of the k nearest neighbors
        from matrix_b for each point in matrix_a
    """
    # Check if the matrices have the same number of features (d)
    assert (
        matrix_a.shape[1] == matrix_b.shape[1]
    ), "Matrices must have the same number of features"

    # Ensure k is not larger than the number of points in matrix_b
    k = min(k, matrix_b.shape[0])

    # Calculate squared Euclidean distances
    a_squared = np.sum(np.square(matrix_a), axis=1, keepdims=True)
    b_squared = np.sum(np.square(matrix_b), axis=1)

    # Use broadcasting to compute pairwise distances
    distances = a_squared + b_squared - 2 * np.dot(matrix_a, matrix_b.T)

    # Use np.maximum to avoid small negative values due to floating point errors
    distances = np.maximum(distances, 0)

    # Find the k smallest distances for each point in matrix_a
    top_k_distances = np.partition(distances, k, axis=1)[:, :k]

    # Calculate the square root to get Euclidean distances
    return np.sqrt(top_k_distances)


def process_cluster(cluster_cells, hvg_data, ann_obj, k):
    np.random.shuffle(cluster_cells)
    data_cells = np.array(
        [ann_obj.reducer(hvg_data[i]) for i in sorted(cluster_cells[:k])]
    )
    data_cells_2 = np.array(
        [ann_obj.reducer(hvg_data[i]) for i in sorted(cluster_cells[k : 2 * k])]
    )
    return data_cells, data_cells_2


def silhouette_scoring(ds, ann_obj, graph, hvg_data, assay_type, res_label):
    try:
        clusters = ds.cells.fetch(f"{assay_type}_{res_label}") - 1 # RNA_{res_label}
    except KeyError:
        logger.error(f"Cluster labels not found for {assay_type}_{res_label}")
        return None

    cluster_similarity = calculate_weighted_cluster_similarity(graph, clusters)

    k = 11
    score = []

    for n, i in enumerate(cluster_similarity):
        this_cluster_cells = np.where(clusters == n)[0]
        if len(this_cluster_cells) < 2 * k:
            k = int(len(this_cluster_cells) / 2)
            logger.warning(
                f"Warning: Cluster {n} has fewer than 22 cells. Will adjust k to {k} instead"
            )

    for n, i in tqdmbar(enumerate(cluster_similarity), total=len(cluster_similarity)):
        this_cluster_cells = np.where(clusters == n)[0]
        np.random.shuffle(this_cluster_cells)
        data_this_cells, data_this_cells_2 = process_cluster(
            # n,
            this_cluster_cells,
            hvg_data,
            ann_obj,
            k,
        )

        if data_this_cells.size == 0 or data_this_cells_2.size == 0:
            logger.warning(f"Warning: Reduced data for cluster {n} is empty. Skipping.")
            score.append(np.nan)
            continue

        k_neighbors = min(k - 1, data_this_cells_2.shape[0] - 1)

        if k_neighbors < 1:
            logger.warning(
                f"Warning: Not enough points in cluster {n} for comparison. Skipping."
            )
            score.append(np.nan)
            continue

        self_dist = calculate_top_k_neighbor_distances(
            data_this_cells, data_this_cells_2, k - 1
        ).mean()

        nearest_cluster = np.argsort(i)[-1]
        nearest_cluster_cells = np.where(clusters == nearest_cluster)[0]
        np.random.shuffle(nearest_cluster_cells)

        if len(nearest_cluster_cells) < k:
            logger.warning(
                f"Warning: Nearest cluster {nearest_cluster} has fewer than {k} cells. Skipping."
            )
            score.append(np.nan)
            continue

        data_nearest_cells, _ = process_cluster(
            # nearest_cluster,
            nearest_cluster_cells,
            hvg_data,
            ann_obj,
            k,
        )

        if data_nearest_cells.size == 0:
            logger.warning(
                f"Warning: Reduced data for nearest cluster {nearest_cluster} is empty. Skipping."
            )
            score.append(np.nan)
            continue

        other_dist = calculate_top_k_neighbor_distances(
            data_this_cells, data_nearest_cells, k - 1
        ).mean()

        score.append((other_dist - self_dist) / max(self_dist, other_dist))

    return np.array(score)


def integration_score(
        batch_labels: np.ndarray,
        metric: str = 'ari'
):
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import normalized_mutual_info_score
    # from sklearn.metrics import calinski_harabasz_score
    # from sklearn.metrics import davies_bouldin_score

    if metric == 'ari':
        return adjusted_rand_score(batch_labels[0], batch_labels[1])
    elif metric == 'nmi':
        return normalized_mutual_info_score(batch_labels[0], batch_labels[1])
    # elif metric == 'calinski_harabasz':
    #     return calinski_harabasz_score(batch_labels[0], batch_labels[1])
    # elif metric == 'davies_bouldin':
    #     return davies_bouldin_score(batch_labels[0], batch_labels[1])
    else:
        logger.error(f"Metric {metric} not recognized. Please choose from 'ari', 'nmi', 'calinski_harabasz', or 'davies_bouldin'.")
        return None