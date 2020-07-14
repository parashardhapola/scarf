import os
import numpy as np
from typing import List, Iterable, Union
import pandas as pd
import re
import zarr
from dask.distributed import Client, LocalCluster
from scipy.sparse import coo_matrix, csr_matrix, triu
from scipy.stats import norm
from .writers import create_zarr_dataset, create_zarr_obj_array
from .metadata import MetaData
from .assay import Assay, RNAassay, ATACassay, ADTassay
from .utils import calc_computed, system_call, rescale_array, clean_array
from tqdm import tqdm


__all__ = ['DataStore']


def sanitize_hierarchy(z: zarr.hierarchy, assay_name: str) -> bool:
    """
    Test if an assay node in zarr object was created properly
    :param z: Zarr hierarchy object
    :param assay_name: string value with name of assay
    :return: True if assay_name is present in z and contains `counts` and `featureData` child nodes else raises error
    """
    if assay_name in z:
        if 'counts' not in z[assay_name]:
            raise KeyError(f"ERROR: 'counts' not found in {assay_name}")
        if 'featureData' not in z[assay_name]:
            raise KeyError(f"ERROR: 'featureData' not found in {assay_name}")
    else:
        raise KeyError(f"ERROR: {assay_name} not found in zarr file")
    return True


class DataStore:
    def __init__(self, zarr_loc: str, assay_types: dict = None, default_assay: str = None,
                 min_features_per_cell: int = 200, min_cells_per_feature: int = 20,
                 auto_filter: bool = False, show_qc_plots: bool = True, force_recalc: bool = False,
                 mito_pattern: str = None, ribo_pattern: str = None, nthreads: int = 2, dask_client=None):
        self._fn: str = zarr_loc
        self._z: zarr.hierarchy = zarr.open(self._fn, 'r+')
        self.nthreads = nthreads
        if dask_client is None:
            cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=nthreads)
            self.daskClient = Client(cluster)
        self.daskClient = dask_client
        # The order is critical here:
        self.cells = self._load_cells()
        self.assayNames = self._get_assay_names()
        self.defaultAssay = self._set_default_assay_name(default_assay)
        self._load_assays(assay_types, min_cells_per_feature)
        # TODO: Reset all attrs, pca, dendrogram etc
        self._ini_cell_props(min_features_per_cell)
        if mito_pattern is None:
            mito_pattern = 'MT-'
        if ribo_pattern is None:
            ribo_pattern = 'RPS|RPL|MRPS|MRPL'
        assay = self._get_assay('')
        if type(assay) == RNAassay:
            assay.add_percent_feature(mito_pattern, 'percentMito', verbose=False)
            assay.add_percent_feature(ribo_pattern, 'percentRibo', verbose=False)
        if auto_filter:
            if show_qc_plots:
                self.plot_cells_dists(cols=['percent*'], all_cells=True)
            self.auto_filter_cells(attrs=['nCounts', 'nFeatures', 'percentMito', 'percentRibo'])
            if show_qc_plots:
                self.plot_cells_dists(cols=['percent*'])

    def _load_cells(self) -> MetaData:
        if 'cellData' not in self._z:
            raise KeyError("ERROR: cellData not found in zarr file")
        return MetaData(self._z['cellData'])

    def _get_assay_names(self) -> List[str]:
        assays = []
        for i in self._z.group_keys():
            if 'is_assay' in self._z[i].attrs.keys():
                sanitize_hierarchy(self._z, i)
                assays.append(i)
        return assays

    def _set_default_assay_name(self, assay_name: str) -> str:
        if len(self.assayNames) > 1:
            if assay_name is None:
                raise ValueError("ERROR: You have more than one assay data. Please provide a name for default assay "
                                 f"using 'default_assay' parameter. Choose one from: {' '.join(self.assayNames)}\n"
                                 "Please note that names are case-sensitive.")
            elif assay_name not in self.assayNames:
                raise ValueError(f"ERROR: The provided default assay name: {assay_name} was not found. "
                                 f"Please Choose one from: {' '.join(self.assayNames)}\n"
                                 "Please note that names are case-sensitive.")
        else:
            if self.assayNames[0] != assay_name:
                print(f"INFO: Default assay name was reset to {self.assayNames[0]}", flush=True)
                assay_name = self.assayNames[0]
        return assay_name

    def _load_assays(self, predefined_assays: dict, min_cells) -> None:
        assay_types = {'RNA': RNAassay, 'ATAC': ATACassay, 'ADT': ADTassay}
        # print_options = '\n'.join(["{'%s': '" + x + "'}" for x in assay_types])
        caution_statement = "CAUTION: %s was set as a generic Assay with no normalization. If this is unintended " \
                            "then please make sure that you provide a correct assay type for this assay using " \
                            "'assay_types' parameter."
        caution_statement = caution_statement + "\nIf you have more than one assay in the dataset then you can set" \
                                                "assay_types={'assay1': 'RNA', 'assay2': 'ADT'} " \
                                                "Just replace with actual assay names instead of assay1 and assay2"
        if predefined_assays is None:
            predefined_assays = {}
        for i in self.assayNames:
            if i in predefined_assays:
                if predefined_assays[i] in assay_types:
                    assay = assay_types[predefined_assays[i]]
                else:
                    print(f"WARNING: {predefined_assays[i]} is not a recognized assay type. Has to be one of "
                          f"{', '.join(list(assay_types.keys()))}\nPLease note that the names are case-sensitive.")
                    print(caution_statement % i)
                    assay = Assay
            else:
                if i in assay_types:
                    assay = assay_types[i]
                else:
                    print(caution_statement % i)
                    assay = Assay
            setattr(self, i, assay(self._z, i, self.cells, min_cells_per_feature=min_cells))
        return None

    def _get_assay(self, from_assay: str):
        if from_assay is None or from_assay == '':
            from_assay = self.defaultAssay
        return self.__getattribute__(from_assay)

    def _get_latest_feat_key(self, from_assay: str):
        assay = self._get_assay(from_assay)
        return assay.attrs['latest_feat_key']

    def _ini_cell_props(self, min_features: int) -> None:
        assay = self._get_assay('')
        if 'nCounts' in self.cells.table.columns and 'nFeatures' in self.cells.table.columns:
            pass
        else:
            n_c = calc_computed(assay.rawData.sum(axis=1), f"INFO: Computing nCounts")
            n_f = calc_computed((assay.rawData > 0).sum(axis=1), f"INFO: Computing nFeatures")
            self.cells.add('nCounts', n_c, overwrite=True)
            self.cells.add('nFeatures', n_f, overwrite=True)
            self.cells.update(self.cells.sift(self.cells.fetch('nFeatures'),
                                              min_features, np.Inf))

    def _col_renamer(self, from_assay: str, col_key: str, suffix: str) -> str:
        if from_assay == self.defaultAssay:
            if col_key == 'I':
                ret_val = suffix
            else:
                ret_val = '_'.join(list(map(str, [col_key, suffix])))
        else:
            if col_key == 'I':
                ret_val = '_'.join(list(map(str, [from_assay, suffix])))
            else:
                ret_val = '_'.join(list(map(str, [from_assay, col_key, suffix])))
        return ret_val

    def filter_cells(self, *, attrs: Iterable[str], lows: Iterable[int],
                     highs: Iterable[int]) -> None:
        for i, j, k in zip(attrs, lows, highs):
            if i not in self.cells.table.columns:
                print(f"WARNING: {i} not found in cell metadata. Will ignore {i} for filtering")
                continue
            x = self.cells.sift(self.cells.table[i].values, j, k)
            print(f"INFO: {len(x) - x.sum()} cells failed filtering for {i}", flush=True)
            self.cells.update(x)

    def auto_filter_cells(self, *, attrs: Iterable[str], min_p: float = 0.01, max_p: float = 0.99) -> None:
        for i in attrs:
            if i not in self.cells.table.columns:
                print(f"WARNING: {i} not found in cell metadata. Will ignore {i} for filtering")
                continue
            a = self.cells.table[i]
            dist = norm(np.median(a), np.std(a))
            self.filter_cells(attrs=[i], lows=[dist.ppf(min_p)], highs=[dist.ppf(max_p)])

    @staticmethod
    def _choose_reduction_method(assay: Assay, reduction_method: str) -> str:
        reduction_method = reduction_method.lower()
        if reduction_method not in ['pca', 'lsi', 'auto']:
            raise ValueError("ERROR: Please choose either 'pca' or 'lsi' as reduction method")
        assay_type = str(assay.__class__).split('.')[-1][:-2]
        if reduction_method == 'auto':
            if assay_type == 'ATACassay':
                print("INFO: Using LSI for dimension reduction", flush=True)
                reduction_method = 'lsi'
            else:
                print("INFO: Using PCA for dimension reduction", flush=True)
                reduction_method = 'pca'
        return reduction_method

    def make_graph(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                   reduction_method: str = 'auto', dims: int = None, k: int = None,
                   ann_metric: str = None, ann_efc: int = None, ann_ef: int = None, ann_m: int = None,
                   rand_state: int = None, n_centroids: int = None, batch_size: int = None,
                   log_transform: bool = None, renormalize_subset: bool = None,
                   local_connectivity: float = None, bandwidth: float = None, return_ann_obj: bool = False):
        from .ann import AnnStream

        if from_assay is None:
            from_assay = self.defaultAssay
        assay = self._get_assay(from_assay)
        if batch_size is None:
            batch_size = assay.rawData.chunksize[0]
        if feat_key is None:
            bool_cols = [x.split('__', 1) for x in assay.feats.table.columns if assay.feats.table[x].dtype == bool
                         and x != 'I']
            bool_cols = [f"{x[1]}({x[0]})" for x in bool_cols]
            bool_cols = ' '.join(map(str, bool_cols))
            raise ValueError("ERROR: You have to choose which features that should be used for graph construction. "
                             "Ideally you should have performed a feature selection step before making this graph. "
                             "Feature selection step adds a column to your feature table. You can access your feature "
                             f"table for for assay {from_assay} like this ds.{from_assay}.feats.table replace 'ds' "
                             f"with the name of DataStore object.\nYou have following boolean columns in the feature "
                             f"metadata of assay {from_assay} which you can choose from: {bool_cols}\n The values in "
                             f"brackets indicate the cell_key for which the feat_key is available. Choosing 'I' "
                             f"as `feat_key` means that you will use all the genes for graph creation.")
        reduction_method = self._choose_reduction_method(assay, reduction_method)

        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        if log_transform is None or renormalize_subset is None:
            if normed_loc in self._z and 'subset_params' in self._z[normed_loc].attrs:
                # This works in coordination with save_normalized_data
                c_log_transform, c_renormalize_subset = self._z[normed_loc].attrs['subset_params']
            else:
                c_log_transform, c_renormalize_subset = None, None
            if log_transform is None and c_log_transform is not None:
                log_transform = c_log_transform
                print(f'INFO: No value provided for parameter `log_transform`. '
                      f'Will use previously used value: {log_transform}', flush=True)
            else:
                log_transform = False
                print(f'INFO: No value provided for parameter `log_transform`. '
                      f'Will use default value: {log_transform}', flush=True)
            if renormalize_subset is None and c_renormalize_subset is not None:
                renormalize_subset = c_renormalize_subset
                print(f'INFO: No value provided for parameter `renormalize_subset`. '
                      f'Will use previously used value: {renormalize_subset}', flush=True)
            else:
                renormalize_subset = True
                print(f'INFO: No value provided for parameter `renormalize_subset`. '
                      f'Will use default value: {renormalize_subset}', flush=True)
        else:
            log_transform = bool(log_transform)
            renormalize_subset = bool(renormalize_subset)

        data = assay.save_normalized_data(cell_key, feat_key, batch_size, normed_loc.split('/')[-1],
                                          log_transform, renormalize_subset)

        if dims is None:
            if normed_loc in self._z and 'latest_reduction' in self._z[normed_loc].attrs:
                reduction_loc = self._z[normed_loc].attrs['latest_reduction']
                dims = int(reduction_loc.rsplit('__', 1)[1])
                print(f'INFO: No value provided for parameter `dims`. '
                      f'Will use previously used value: {dims}', flush=True)
            else:
                dims = 11
                print(f'INFO: No value provided for parameter `dims`. Will use default value: {dims}', flush=True)
        reduction_loc = f"{normed_loc}/reduction__{reduction_method}__{dims}"

        if ann_metric is None or ann_efc is None or ann_ef is None or ann_m is None or rand_state is None:
            if reduction_loc in self._z and 'latest_ann' in self._z[reduction_loc].attrs:
                ann_loc = self._z[reduction_loc].attrs['latest_ann']
                c_ann_metric, c_ann_efc, c_ann_ef, c_ann_m, c_rand_state = \
                    ann_loc.rsplit('/', 1)[1].split('__')[1:]
            else:
                c_ann_metric, c_ann_efc, c_ann_ef, c_ann_m, c_rand_state = \
                    None, None, None, None, None
            if ann_metric is None and c_ann_metric is not None:
                ann_metric = c_ann_metric
                print(f'INFO: No value provided for parameter `ann_metric`. '
                      f'Will use previously used value: {ann_metric}', flush=True)
            else:
                ann_metric = 'l2'
                print(f'INFO: No value provided for parameter `ann_metric`. '
                      f'Will use default value: {ann_metric}', flush=True)
            if ann_efc is None and c_ann_efc is not None:
                ann_efc = int(c_ann_efc)
                print(f'INFO: No value provided for parameter `ann_efc`. '
                      f'Will use previously used value: {ann_efc}', flush=True)
            else:
                ann_efc = 10
                print(f'INFO: No value provided for parameter `ann_efc`. Will use default value: {ann_efc}', flush=True)
            if ann_ef is None and c_ann_ef is not None:
                ann_ef = int(c_ann_ef)
                print(f'INFO: No value provided for parameter `ann_ef`. '
                      f'Will use previously used value: {ann_ef}', flush=True)
            else:
                ann_ef = None  # Will be set after value for k is determined
            if ann_m is None and c_ann_m is not None:
                ann_m = int(c_ann_m)
                print(f'INFO: No value provided for parameter `ann_m`. '
                      f'Will use previously used value: {ann_m}', flush=True)
            else:
                ann_m = int(dims * 1.5)
                print(f'INFO: No value provided for parameter `ann_m`. Will use default value: {ann_m}', flush=True)
            if rand_state is None and c_rand_state is not None:
                rand_state = int(c_rand_state)
                print(f'INFO: No value provided for parameter `rand_state`. '
                      f'Will use previously used value: {rand_state}', flush=True)
            else:
                rand_state = 4466
                print(f'INFO: No value provided for parameter `rand_state`. '
                      f'Will use default value: {rand_state}', flush=True)
        else:
            ann_metric = str(ann_metric)
            ann_efc = int(ann_efc)
            ann_ef = int(ann_ef)
            ann_m = int(ann_m)
            rand_state = int(rand_state)

        if k is None:
            if reduction_loc in self._z and 'latest_ann' in self._z[reduction_loc].attrs:
                ann_loc = self._z[reduction_loc].attrs['latest_ann']
                knn_loc = self._z[ann_loc].attrs['latest_knn']
                k = int(knn_loc.rsplit('__', 1)[1])  # depends on param_joiner
                print(f'INFO: No value provided for parameter `k`. Will use previously used value: {k}', flush=True)
            else:
                k = 11
                print(f'INFO: No value provided for parameter `k`. Will use default value: {k}', flush=True)
        else:
            k = int(k)
        if ann_ef is None:
            ann_ef = k * 2
        ann_loc = f"{reduction_loc}/ann__{ann_metric}__{ann_efc}__{ann_ef}__{ann_m}__{rand_state}"
        ann_idx_loc = f"{self._fn}/{ann_loc}/ann_idx"
        knn_loc = f"{ann_loc}/knn__{k}"

        if n_centroids is None:
            if reduction_loc in self._z and 'latest_kmeans' in self._z[reduction_loc].attrs:
                kmeans_loc = self._z[reduction_loc].attrs['latest_kmeans']
                n_centroids = int(kmeans_loc.split('/')[-1].split('__')[1])  # depends on param_joiner
                print(f'INFO: No value provided for parameter `n_centroids`.'
                      f' Will use previously used value: {n_centroids}', flush=True)
            else:
                n_centroids = min(data.shape[0]/10, max(500, data.shape[0]/100))
                print(f'INFO: No value provided for parameter `n_centroids`. '
                      f'Will use default value: {n_centroids}', flush=True)
        kmeans_loc = f"{reduction_loc}/kmeans__{n_centroids}__{rand_state}"

        if local_connectivity is None or bandwidth is None:
            if knn_loc in self._z and 'latest_graph' in self._z[knn_loc].attrs:
                graph_loc = self._z[knn_loc].attrs['latest_graph']
                c_local_connectivity, c_bandwidth = map(float, graph_loc.rsplit('/')[-1].split('__')[1:])
            else:
                c_local_connectivity, c_bandwidth = None, None
            if local_connectivity is None and c_local_connectivity is not None:
                local_connectivity = c_local_connectivity
                print(f'INFO: No value provided for parameter `local_connectivity`. '
                      f'Will use previously used value: {local_connectivity}', flush=True)
            else:
                local_connectivity = 1.0
                print(f'INFO: No value provided for parameter `local_connectivity`. '
                      f'Will use default value: {local_connectivity}', flush=True)
            if bandwidth is None and c_bandwidth is not None:
                bandwidth = c_bandwidth
                print(f'INFO: No value provided for parameter `bandwidth`. '
                      f'Will use previously used value: {bandwidth}', flush=True)
            else:
                bandwidth = 1.5
                print(f'INFO: No value provided for parameter `bandwidth`. Will use default value: {bandwidth}',
                      flush=True)
        else:
            local_connectivity = float(local_connectivity)
            bandwidth = float(bandwidth)
        graph_loc = f"{knn_loc}/graph__{local_connectivity}__{bandwidth}"

        loadings = None
        fit_kmeans = True
        fit_ann = True
        mu, sigma = np.ndarray([]), np.ndarray([])

        if reduction_loc in self._z:
            loadings = self._z[reduction_loc]['reduction'][:]
            if reduction_method == 'pca':
                mu = self._z[reduction_loc]['mu'][:]
                sigma = self._z[reduction_loc]['sigma'][:]
            print(f"INFO: Using existing loadings for {reduction_method} with {dims} dims", flush=True)
        else:
            if reduction_method == 'pca':
                mu = clean_array(calc_computed(data.mean(axis=0),
                                               'INFO: Calculating mean of norm. data'))
                sigma = clean_array(calc_computed(data.std(axis=0),
                                                  'INFO: Calculating std. dev. of norm. data'), 1)
        if ann_loc in self._z:
            fit_ann = False
            print(f"INFO: Using existing ANN index", flush=True)
        if kmeans_loc in self._z:
            fit_kmeans = False
            print(f"INFO: using existing kmeans cluster centers", flush=True)
        ann_obj = AnnStream(data=data, k=k, n_cluster=n_centroids, reduction_method=reduction_method,
                            dims=dims, loadings=loadings, mu=mu, sigma=sigma, ann_metric=ann_metric, ann_efc=ann_efc,
                            ann_ef=ann_ef, ann_m=ann_m, ann_idx_loc=ann_idx_loc, nthreads=self.nthreads,
                            rand_state=rand_state, do_ann_fit=fit_ann, do_kmeans_fit=fit_kmeans)

        if loadings is None:
            self._z.create_group(reduction_loc, overwrite=True)
            g = create_zarr_dataset(self._z[reduction_loc], 'reduction', (1000, 1000), 'f8', ann_obj.loadings.shape)
            g[:, :] = ann_obj.loadings
            g = create_zarr_dataset(self._z[reduction_loc], 'mu', (100000,), 'f8', mu.shape)
            g[:] = mu
            g = create_zarr_dataset(self._z[reduction_loc], 'sigma', (100000,), 'f8', sigma.shape)
            g[:] = sigma
        if ann_loc not in self._z:
            self._z.create_group(ann_loc, overwrite=True)
            ann_obj.annIdx.save_index(ann_idx_loc)
        if fit_kmeans:
            self._z.create_group(kmeans_loc, overwrite=True)
            g = create_zarr_dataset(self._z[kmeans_loc], 'cluster_centers',
                                    (1000, 1000), 'f8', ann_obj.kmeans.cluster_centers_.shape)
            g[:, :] = ann_obj.kmeans.cluster_centers_
            g = create_zarr_dataset(self._z[kmeans_loc], 'cluster_labels', (100000,), 'f8', ann_obj.clusterLabels.shape)
            g[:] = ann_obj.clusterLabels

        if knn_loc in self._z and graph_loc in self._z:
            print(f"INFO: KNN graph already exists will not recompute.", flush=True)
        else:
            from .knn_utils import self_query_knn, smoothen_dists
            if knn_loc not in self._z:
                self_query_knn(ann_obj, self._z.create_group(knn_loc, overwrite=True), batch_size, self.nthreads)
            smoothen_dists(self._z.create_group(graph_loc, overwrite=True),
                           self._z[knn_loc]['indices'], self._z[knn_loc]['distances'],
                           local_connectivity, bandwidth)

        self._z[normed_loc].attrs['latest_reduction'] = reduction_loc
        self._z[reduction_loc].attrs['latest_ann'] = ann_loc
        self._z[reduction_loc].attrs['latest_kmeans'] = kmeans_loc
        self._z[ann_loc].attrs['latest_knn'] = knn_loc
        self._z[knn_loc].attrs['latest_graph'] = graph_loc
        if return_ann_obj:
            return ann_obj
        return None

    def _get_latest_graph_loc(self, from_assay: str, cell_key: str, feat_key: str):
        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = self._z[normed_loc].attrs['latest_reduction']
        ann_loc = self._z[reduction_loc].attrs['latest_ann']
        knn_loc = self._z[ann_loc].attrs['latest_knn']
        return self._z[knn_loc].attrs['latest_graph']

    def _load_graph(self, from_assay: str, cell_key: str, feat_key: str, graph_format: str,
                    min_edge_weight: float = 0, symmetric: bool = True):
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        if graph_loc not in self._z:
            print(f"ERROR: {graph_loc} not found in zarr location {self._fn}. Run `make_graph` for assay {from_assay}")
            return None
        if graph_format not in ['coo', 'csr']:
            raise KeyError("ERROR: format has to be either 'coo' or 'csr'")
        store = self._z[graph_loc]
        knn_loc = self._z[graph_loc.rsplit('/', 1)[0]]
        n_cells = knn_loc['indices'].shape[0]
        # TODO: can we have a progress bar for graph loading. Append to coo matrix?
        graph = coo_matrix((store['weights'][:], (store['edges'][:, 0], store['edges'][:, 1])),
                           shape=(n_cells, n_cells))
        if symmetric:
            graph = triu((graph + graph.T) / 2)
        idx = graph.data > min_edge_weight
        if graph_format == 'coo':
            return coo_matrix((graph.data[idx], (graph.row[idx], graph.col[idx])), shape=(n_cells, n_cells))
        else:
            return csr_matrix((graph.data[idx], (graph.row[idx], graph.col[idx])), shape=(n_cells, n_cells))

    def _ini_embed(self, from_assay: str, cell_key: str, feat_key: str, n_comps: int):
        from sklearn.decomposition import PCA
        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = self._z[normed_loc].attrs['latest_reduction']
        kmeans_loc = self._z[reduction_loc].attrs['latest_kmeans']
        pc = PCA(n_components=n_comps).fit_transform(self._z[kmeans_loc]['cluster_centers'][:])
        for i in range(n_comps):
            pc[:, i] = rescale_array(pc[:, i])
        clusters = self._z[kmeans_loc]['cluster_labels'][:].astype(np.uint32)
        return np.array([pc[x] for x in clusters]).astype(np.float32, order="C")

    def run_tsne(self, sgtsne_loc, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                 tsne_dims: int = 2, lambda_scale: float = 1.0, max_iter: int = 500, early_iter: int = 200,
                 alpha: int = 10, box_h: float = 0.7, temp_file_loc: str = '.', verbose: bool = True) -> None:
        from uuid import uuid4
        from .knn_utils import export_knn_to_mtx
        from pathlib import Path

        if from_assay is None:
            from_assay = self.defaultAssay
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)

        uid = str(uuid4())
        ini_emb_fn = Path(temp_file_loc, f'{uid}.txt').resolve()
        with open(ini_emb_fn, 'w') as h:
            ini_emb = self._ini_embed(from_assay, cell_key, feat_key, tsne_dims).flatten()
            h.write('\n'.join(map(str, ini_emb)))
        knn_mtx_fn = Path(temp_file_loc, f'{uid}.mtx').resolve()
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        knn_loc = self._z[graph_loc.rsplit('/', 1)[0]]
        n_cells, n_neighbors = knn_loc['indices'].shape
        export_knn_to_mtx(knn_mtx_fn, self._z[graph_loc], n_cells, n_neighbors)
        out_fn = Path(temp_file_loc, f'{uid}_output.txt').resolve()

        cmd = f"{sgtsne_loc} -m {max_iter} -l {lambda_scale} -d {tsne_dims} -e {early_iter} -p 1 -a {alpha}" \
              f" -h {box_h} -i {ini_emb_fn} -o {out_fn} {knn_mtx_fn}"
        if verbose:
            system_call(cmd)
        else:
            os.system(cmd)
        emb = pd.read_csv(out_fn, header=None, sep=' ')[[0, 1]].values.T
        for i in range(tsne_dims):
            self.cells.add(self._col_renamer(from_assay, cell_key, f'tSNE{i + 1}'),
                           emb[i], key=cell_key, overwrite=True)
        for fn in [out_fn, knn_mtx_fn, ini_emb_fn]:
            Path.unlink(fn)

    def run_umap(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                 min_edge_weight: float = 0, ini_embed: np.ndarray = None, umap_dims: int = 2,
                 spread: float = 2.0, min_dist: float = 1, fit_n_epochs: int = 200, tx_n_epochs: int = 100,
                 random_seed: int = 4444, parallel: bool = False, **kwargs) -> None:
        from .umap import fit_transform
        if from_assay is None:
            from_assay = self.defaultAssay
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        graph = self._load_graph(from_assay, cell_key, feat_key, 'coo', min_edge_weight, symmetric=False)
        if ini_embed is None:
            ini_embed = self._ini_embed(from_assay, cell_key, feat_key, umap_dims)
        t = fit_transform(graph, ini_embed, spread=spread, min_dist=min_dist,
                          tx_n_epochs=tx_n_epochs, fit_n_epochs=fit_n_epochs,
                          random_seed=random_seed, parallel=parallel, **kwargs)
        for i in range(umap_dims):
            self.cells.add(self._col_renamer(from_assay, cell_key, f'UMAP{i + 1}'),
                           t[:, i], key=cell_key, overwrite=True)
        return None

    def run_clustering(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                       n_clusters: int = None, min_edge_weight: float = 0, balanced_cut: bool = False,
                       max_size: int = None, min_size: int = None, max_distance_fc: float = 2,
                       return_clusters: bool = False, force_recalc: bool = False) -> Union[None, pd.Series]:
        import sknetwork as skn

        if from_assay is None:
            from_assay = self.defaultAssay
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        if balanced_cut is False:
            if n_clusters is None:
                raise ValueError("ERROR: Please provide a value for n_clusters parameter. We are working on making "
                                 "this parameter free")
        else:
            if n_clusters is not None:
                print("INFO: Using balanced cut method for cutting dendrogram. `n_clusters` will be ignored.",
                      flush=True)
            if max_size is None or min_size is None:
                raise ValueError("ERROR: Please provide value for max_size and min_size")
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        dendrogram_loc = f"{graph_loc}/dendrogram__{min_edge_weight}"
        # tuple are changed to list when saved as zarr attrs
        if dendrogram_loc in self._z and force_recalc is False:
            dendrogram = self._z[dendrogram_loc][:]
            print("INFO: Using existing dendrogram", flush=True)
        else:
            paris = skn.hierarchy.Paris()
            graph = self._load_graph(from_assay, cell_key, feat_key, 'csr', min_edge_weight=min_edge_weight,
                                     symmetric=True)
            dendrogram = paris.fit_transform(graph)
            dendrogram[dendrogram == np.Inf] = 0
            g = create_zarr_dataset(self._z[graph_loc], dendrogram_loc.rsplit('/', 1)[1],
                                    (5000,), 'f8', (graph.shape[0] - 1, 4))
            g[:] = dendrogram
        self._z[graph_loc].attrs['latest_dendrogram'] = dendrogram_loc
        if balanced_cut:
            from .dendrogram import BalancedCut
            labels = BalancedCut(dendrogram, max_size, min_size, max_distance_fc).get_clusters()
        else:
            # n_cluster - 1 because cut_straight possibly has a bug so generates one extra
            labels = skn.hierarchy.cut_straight(dendrogram, n_clusters=n_clusters-1) + 1
        if return_clusters:
            return pd.Series(labels, index=self.cells.table[cell_key].index[self.cells.table[cell_key]])
        else:
            self.cells.add(self._col_renamer(from_assay, cell_key, 'cluster'), labels,
                           fill_val=-1, key=cell_key, overwrite=True)

    def run_marker_search(self, *, from_assay: str = None, group_key: str = None, threshold: float = 0.25) -> None:
        from .markers import find_markers_by_rank

        if group_key is None:
            print("INFO: No value provided for group_key. Will autoset `group_key` to 'cluster'", flush=True)
            group_key = 'cluster'
        assay = self._get_assay(from_assay)
        markers = find_markers_by_rank(assay, group_key, threshold)
        z = self._z[assay.name]
        if 'markers' not in z:
            z.create_group('markers')
        group = z['markers'].create_group(group_key, overwrite=True)
        for i in markers:
            g = group.create_group(i)
            vals = markers[i]
            if len(vals) != 0:
                create_zarr_obj_array(g, 'names', list(vals.index))
                g_s = create_zarr_dataset(g, 'scores', (10000,), float, vals.values.shape)
                g_s[:] = vals.values
        return None

    def run_mapping(self, *, target_assay: Assay, target_name: str, from_assay: str = None,
                    cell_key: str = 'I', feat_key: str = None, save_k: int = 1, batch_size: int = 1000):
        assay = self._get_assay(from_assay)
        from_assay = assay.name
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        feat_ids = assay.feats.table.ids[assay.feats.table[cell_key+'__'+feat_key]].values
        # FIXME: find a better way to initialize this
        tfk = 'asdfverasfa'
        target_assay.feats.add(k='I__'+tfk, v=target_assay.feats.table.ids.isin(feat_ids).values,
                               fill_val=False, overwrite=True)
        colnames = target_assay.feats.table.ids[target_assay.feats.table['I__'+tfk]].values
        if len(colnames) == 0:
            raise ValueError("ERROR: No common features found between the two datasets")
        else:
            print(f"INFO: {len(colnames)} common features from {len(feat_ids)} "
                  f"reference features will be used", flush=True)

        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        if 'projections' not in self._z[graph_loc]:
            self._z[graph_loc].create_group('projections')
        store = self._z[graph_loc]['projections'].create_group(target_name, overwrite=True)
        nc, nk = target_assay.cells.table.I.sum(), save_k
        zi = create_zarr_dataset(store, 'indices', (batch_size,), 'u8', (nc, nk))
        zd = create_zarr_dataset(store, 'distances', (batch_size,), 'f8', (nc, nk))

        ann_obj = self.make_graph(from_assay=from_assay, cell_key=cell_key, feat_key=feat_key, return_ann_obj=True)
        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        norm_params = dict(zip(['log_transform', 'renormalize_subset'], self._z[normed_loc].attrs['subset_params']))
        data = target_assay.save_normalized_data(cell_key, tfk, batch_size, f"normed__I__{tfk}", **norm_params)
        entry_start = 0
        for i in tqdm(data.blocks, desc='Mapping'):
            i = pd.DataFrame(i.compute(), columns=colnames).T.reindex(feat_ids).fillna(0).T.values
            ki, kd = ann_obj.transform_ann(ann_obj.reducer(i), k=save_k)
            entry_end = entry_start + len(ki)
            zi[entry_start:entry_end, :] = ki
            zd[entry_start:entry_end, :] = kd
            entry_start = entry_end
        target_assay.feats.remove(cell_key+'__'+tfk)
        return None

    def get_mapping_score(self,  *, target_name: str, target_groups: np.ndarray = None,
                          from_assay: str = None, cell_key: str = 'I', feat_key: str = None,) -> np.ndarray:
        if from_assay is None:
            from_assay = self.defaultAssay
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)

        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        store_loc = f"{graph_loc}/projections/{target_name}"
        if store_loc not in self._z:
            raise KeyError(f"ERROR: Projections have not been computed for {target_name} in th latest graph. Please"
                           f" run `run_mapping` or update latest_graph by running `make_graph` with desired parameters")
        store = self._z[store_loc]

        indices = store['indices'][:]
        dists = store['distances'][:]
        dists = 1 / np.log1p(dists)
        n_cells = indices.shape[0]

        if target_groups is not None:
            if len(target_groups) != n_cells:
                raise ValueError(f"ERROR: Length of target_groups {len(target_groups)} not same as number of target "
                                 f"cells in the projection {n_cells}")
            groups = pd.Series(target_groups)
        else:
            groups = pd.Series(np.zeros(n_cells))

        ref_n_cells = self.cells.table[cell_key].sum()
        for group in sorted(groups.unique()):
            coi = {x: None for x in groups[groups == group].index.values}
            ms = np.zeros(ref_n_cells)
            for n, i, j in zip(range(len(indices)), indices, dists):
                if n in coi:
                    for x, y in zip(i, j):
                        ms[x] += y
            ms = np.log1p(1000 * ms / len(coi))
            yield group, ms

    def plot_cells_dists(self, cols: List[str] = None, all_cells: bool = False, **kwargs):
        from .plots import plot_qc
        plot_cols = ['nCounts', 'nFeatures']
        if cols is not None:
            if type(cols) != list:
                raise ValueError("ERROR: 'attrs' argument must be of type list")
            for i in cols:
                matches = [x for x in self.cells.table.columns if re.search(i, x)]
                if len(matches) > 0:
                    plot_cols.extend(matches)
                else:
                    print(f"WARNING: {i} not found in cell metadata")
        if all_cells:
            plot_qc(self.cells.table[plot_cols], **kwargs)
        else:
            if 'color' not in kwargs:
                kwargs['color'] = 'coral'
            plot_qc(self.cells.table[self.cells.table.I][plot_cols], **kwargs)
        return None

    def get_cell_vals(self, *, from_assay: str, cell_key: str, k: str, clip_fraction: float = 0):
        cell_idx = self.cells.active_index(cell_key)
        if k not in self.cells.table.columns:
            assay = self._get_assay(from_assay)
            feat_idx = assay.feats.get_idx_by_names([k], True)
            if len(feat_idx) == 0:
                raise ValueError(f"ERROR: {k} not found in {from_assay} assay.")
            else:
                if len(feat_idx) > 1:
                    print(f"WARNING: Plotting mean of {len(feat_idx)} features because {k} is not unique.")
            vals = assay.normed(cell_idx, feat_idx).mean(axis=1).compute()
            if clip_fraction > 0:
                min_v = np.percentile(vals, 100 * clip_fraction)
                max_v = np.percentile(vals, 100 - 100 * clip_fraction)
                vals[vals < min_v] = min_v
                vals[vals > max_v] = max_v
            return pd.Series(vals, dtype=float)
        else:
            vals = self.cells.fetch(k, cell_key)
            if vals.dtype == object:
                vals = pd.Series(vals, dtype="category")
            elif vals.dtype == int:
                vals = pd.Series(vals, dtype="category")
            else:
                vals = pd.Series(vals, dtype=float)
            return vals

    def plot_layout(self, *, from_assay: str = None, cell_key: str = 'I', feat_assay: str = None,
                    layout_key: str = 'UMAP', color_by: str = None, clip_fraction: float = 0.01, shade: bool = False,
                    labels_kwargs: dict = None, legends_kwargs: dict = None, **kwargs):
        from .plots import plot_scatter, shade_scatter
        if from_assay is None:
            from_assay = self.defaultAssay
        if feat_assay is None:
            feat_assay = from_assay
        if clip_fraction >= 0.5:
            raise ValueError("ERROR: clip_fraction cannot be larger than or equal to 0.5")
        x = self.cells.fetch(self._col_renamer(from_assay, cell_key, f'{layout_key}1'), cell_key)
        y = self.cells.fetch(self._col_renamer(from_assay, cell_key, f'{layout_key}2'), cell_key)
        if color_by is not None:
            v = self.get_cell_vals(from_assay=feat_assay, cell_key=cell_key, k=color_by,
                                   clip_fraction=clip_fraction)
        else:
            v = np.ones(len(x))
        df = pd.DataFrame({f'{layout_key} 1': x, f'{layout_key} 2': y, 'v': v})
        if shade:
            return shade_scatter(df, labels_kwargs=labels_kwargs, legends_kwargs=legends_kwargs, **kwargs)
        else:
            return plot_scatter(df, labels_kwargs=labels_kwargs, legends_kwargs=legends_kwargs, **kwargs)

    def plot_cluster_tree(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                          cluster_key: str = 'cluster', width: float = 1.5, lvr_factor: float = 0.5,
                          min_node_size: float = 20, node_size_expand_factor: float = 2, cmap='tab20'):
        from .plots import plot_cluster_hierarchy
        from .dendrogram import SummarizedTree

        if from_assay is None:
            from_assay = self.defaultAssay
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        clusts = self.cells.fetch(cluster_key)
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        dendrogram_loc = self._z[graph_loc].attrs['latest_dendrogram']
        sg = SummarizedTree(self._z[dendrogram_loc][:]).extract_ancestor_tree(clusts)
        plot_cluster_hierarchy(sg, clusts, width=width, lvr_factor=lvr_factor, min_node_size=min_node_size,
                               node_size_expand_factor=node_size_expand_factor, cmap=cmap)

    def plot_marker_heatmap(self, *, from_assay: str = None, group_key: str = None, topn: int = 5,
                            log_transform: bool = True, vmin: float = -1, vmax: float = 2,
                            batch_size: int = None, **heatmap_kwargs):
        from .plots import plot_heatmap

        assay = self._get_assay(from_assay)
        if batch_size is None:
            batch_size = min(999, int(1e7 / assay.cells.N)) + 1
        if group_key is None:
            print("INFO: No value provided for group_key. Will autoset `group_key` to 'cluster'")
            group_key = 'cluster'
        if 'markers' not in self._z[assay.name]:
            raise KeyError("ERROR: Please run `run_marker_search` first")
        if group_key not in self._z[assay.name]['markers']:
            raise KeyError(f"ERROR: Please run `run_marker_search` first with {group_key} as `group_key`")
        g = self._z[assay.name]['markers'][group_key]
        goi = []
        for i in g.keys():
            if 'names' in g[i]:
                goi.extend(g[i]['names'][:][:topn])
        goi = sorted(set(goi))
        cdf = []
        for i in np.array_split(goi, len(goi) // batch_size + 1):
            feat_idx = assay.feats.get_idx_by_names(i)
            df = pd.DataFrame(assay.normed(feat_idx=feat_idx, log_transform=log_transform).compute(), columns=i)
            df['cluster'] = assay.cells.fetch('cluster')
            df = df.groupby('cluster').mean().T
            df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
            cdf.append(df)
        cdf = pd.concat(cdf, axis=0)
        # noinspection PyTypeChecker
        cdf[cdf < vmin] = vmin
        # noinspection PyTypeChecker
        cdf[cdf > vmax] = vmax
        plot_heatmap(cdf, **heatmap_kwargs)

    def __repr__(self):
        x = ' '.join(self.assayNames)
        return f"DataStore with {self.cells.N} cells containing {len(self.assayNames)} assays: {x}"
