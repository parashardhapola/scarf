import numpy as np
from umap.umap_ import smooth_knn_dist, compute_membership_strengths
from .writers import create_zarr_dataset
from .ann import AnnStream

__all__ = ['make_knn_graph']


def make_knn_graph(ann_obj: AnnStream, chunk_size: int, store,
                  lc: float = 1, bw: float = 1.5, save_raw_dists: bool = False):
    # bandwidth: Higher value will push the mean of distribution of graph edge weights towards right
    # local_connectivity: Higher values will create push distribution of edge weights towards terminal values (binary
    # like) Lower values will accumulate edge weights around the mean produced by bandwidth

    n_cells, n_neighbors = ann_obj.nCells, ann_obj.k
    if save_raw_dists:
        z_knn = create_zarr_dataset(store, 'indices', (chunk_size,), 'u8',
                                    (n_cells, n_neighbors))
        z_dist = create_zarr_dataset(store, 'distances', (chunk_size,), 'f8',
                                        (n_cells, n_neighbors))
    zge = create_zarr_dataset(store, 'edges', (chunk_size,), ('u8', 'u8'),
                              (n_cells * n_neighbors, 2))
    zgw = create_zarr_dataset(store, 'weights', (chunk_size,), 'f8',
                              (n_cells * n_neighbors,))
    last_row = 0
    val_counts = 0
    nsample_start = 0
    for i in ann_obj.iter_blocks(msg='Saving KNN graph'):
        ki, kv = ann_obj.transform_ann(ann_obj.reducer(i))
        kv = kv.astype(np.float32, order='C')
        sigmas, rhos = smooth_knn_dist(kv, k=n_neighbors,
                                       local_connectivity=lc, bandwidth=bw)
        rows, cols, vals = compute_membership_strengths(ki, kv, sigmas, rhos)
        rows = rows + last_row
        start = val_counts
        end = val_counts + len(rows)
        last_row = rows[-1] + 1
        val_counts += len(rows)
        nsample_end = nsample_start + len(ki)
        if save_raw_dists:
            z_knn[nsample_start:nsample_end, :] = ki
            z_dist[nsample_start:nsample_end, :] = kv
        zge[start:end, 0] = rows
        zge[start:end, 1] = cols
        zgw[start:end] = vals
        nsample_start = nsample_end
    return None
