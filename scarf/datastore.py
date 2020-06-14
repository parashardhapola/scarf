import zarr
import os
import shutil
import numpy as np
from typing import List, Iterable, Union
import re
from dask import optimize
from scipy.stats import norm
import pandas as pd
from uuid import uuid4
from scipy.sparse import coo_matrix, csr_matrix, triu
from .writers import create_zarr_dataset
from .metadata import MetaData
from .assay import Assay, RNAassay, ATACassay, ADTassay
from .utils import show_progress

__all__ = ['DataStore']


def sanitize_hierarchy(z, assay_name) -> bool:
    if assay_name in z:
        if 'counts' not in z[assay_name]:
            raise KeyError(f"ERROR: 'counts' not found in {assay_name}")
        if 'featureData' not in z[assay_name]:
            raise KeyError(f"ERROR: 'featureData' not found in {assay_name}")
    else:
        raise KeyError(f"ERROR: {assay_name} not found in zarr file")
    return True


def rescale_array(a: np.ndarray, frac: float = 0.9) -> np.ndarray:
    loc = (np.median(a) + np.median(a[::-1])) / 2
    dist = norm(loc, np.std(a))
    minv, maxv = dist.ppf(1 - frac), dist.ppf(frac)
    a[a < minv] = minv
    a[a > maxv] = maxv
    return a


def clean_array(x, fill_val: int = 0):
    x = np.nan_to_num(x, copy=True)
    x[(x == np.Inf) | (x == -np.Inf)] = 0
    x[x == 0] = fill_val
    return x


class DataStore:

    def __init__(self, zarr_loc: str, assay_types: dict = None, default_assay: str = 'RNA',
                 min_genes_per_cell: int = 200, min_cells_per_gene: int = 20,
                 auto_filter: bool = False, show_qc_plots: bool = True, force_recalc: bool = False,
                 mito_pattern: str = None, ribo_pattern: str = None):
        self._fn = zarr_loc
        self._z = zarr.open(self._fn, 'r+')
        # The order is critical here:
        self.cells = self._load_cells()
        self.assayNames = self._get_assay_names()
        self.defaultAssay = self._set_default_assay_name(default_assay)
        self._load_assays(assay_types)
        # TODO: Reset all attrs, pca, dendrogram etc
        self._ini_cell_props(min_genes_per_cell, min_cells_per_gene, force_recalc)
        if mito_pattern is None:
            mito_pattern = 'MT-'
        if ribo_pattern is None:
            ribo_pattern = 'RPS|RPL|MRPS|MRPL'
        assay = self.__getattribute__(self.defaultAssay)
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

    def _load_assays(self, assays: dict) -> None:
        assay_types = {'RNA': RNAassay, 'ATAC': ATACassay, 'ADT': ADTassay}
        print_options = '\n'.join(["{'%s': '" + x + "'}" for x in assay_types])
        caution_statement = "CAUTION: %s was set as a generic Assay with no normalization. If this is unintended " \
                            "then please make sure that you provide a correct assay type for this assay using " \
                            "'assay_types' parameter. You can provide assay type in one these ways:\n" + print_options
        caution_statement = caution_statement + "\nIf you have more than one assay in the dataset then you can set" \
                                                "assay_types={'assay1': 'RNA', 'assay2': 'ADT'} " \
                                                "Just replace with actual assay names instead of assay1 and assay2"
        if assays is None:
            assays = {}
        for i in self.assayNames:
            if i in assays:
                if assays[i] in assay_types:
                    assay = assay_types[assays[i]](self._fn, i, self.cells)
                else:
                    print(f"WARNING: {assays[i]} is not a recognized assay type. Has to be one of "
                          f"{', '.join(list(assay_types.keys()))}\nPLease note that the names are case-sensitive.")
                    print(caution_statement % i)
                    assay = Assay(self._fn, i, self.cells)
            else:
                if i in assay_types:
                    # FIXME: Should this be silent?
                    assay = assay_types[i](self._fn, i, self.cells)
                else:
                    print(caution_statement % i)
                    assay = Assay(self._fn, i, self.cells)
            setattr(self, i, assay)
        return None

    @show_progress
    def _ini_cell_props(self, min_features: int, min_cells: int, force_recalc: bool = False) -> None:
        # TODO: add adt data as well?
        assay = self.__getattribute__(self.defaultAssay)
        n_c = assay.rawData.sum(axis=1)
        n_f = (assay.rawData > 0).sum(axis=1)
        fn = (assay.rawData > 0).sum(axis=0)
        n_c, n_f, fn = optimize(n_c, n_f, fn)
        if 'nCounts' in self.cells.table.columns and \
                'nFeatures' in self.cells.table.columns and force_recalc is False:
            pass
        else:
            print(f"INFO: Computing nCounts", flush=True)
            self.cells.add('nCounts', n_c.compute(), overwrite=True)
            print(f"INFO: Computing nFeatures", flush=True)
            self.cells.add('nFeatures', n_f.compute(), overwrite=True)
            self.cells.update(self.cells.sift(self.cells.fetch('nFeatures'),
                                              min_features, np.Inf))
            print(f"INFO: Filtering Features", flush=True)
            assay.feats.update(fn.compute() > min_cells)

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

    def make_graph(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = 'I',
                   reduction_method: str = 'auto', k: int = 11, n_cluster: int = 100, dims: int = None,
                   ann_metric: str = 'l2', ann_efc: int = 100, ann_ef: int = 5, ann_nthreads: int = 1,
                   rand_state: int = 4466, batch_size: int = None,
                   log_transform: bool = False, renormalize_subset: bool = True,
                   local_connectivity: float = 1, bandwidth: float = 1.5,
                   save_ann_obj: bool = False, save_raw_dists: bool = False, **kmeans_kwargs):
        from .ann import AnnStream
        if from_assay is None:
            from_assay = self.defaultAssay
        assay = self.__getattribute__(from_assay)
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
        if batch_size is None:
            batch_size = assay.rawData.chunksize[0]
        data = assay.select_and_normalize(cell_key, feat_key, batch_size,
                                          log_transform=log_transform,
                                          renormalize_subset=renormalize_subset)
        # data is not guaranteed to have same indices as raw data and may be shifted. we handle this in the pipeline
        mu = clean_array(data.mean(axis=0).compute())
        sigma = clean_array(data.std(axis=0).compute(), 1)

        loadings = None
        loadings_name = reduction_method if cell_key == 'I' else cell_key + '_' + reduction_method
        loadings_hash = hash((assay.create_subset_hash(cell_key, feat_key), dims,
                              log_transform, renormalize_subset))
        if loadings_name in assay.attrs and assay.attrs[loadings_name] == loadings_hash and \
                loadings_name in assay._z[from_assay]:
            print("INFO: Loading cached component coefficients/loadings", flush=True)
            loadings = self._z[from_assay][loadings_name][:]
        if dims is None and loadings is None:
            raise ValueError("ERROR: No cached data found. Please provide a value for 'dims'")

        ann_obj = AnnStream(data, k, n_cluster, reduction_method, dims, loadings,
                            ann_metric, ann_efc, ann_ef,
                            ann_nthreads, rand_state, mu, sigma, **kmeans_kwargs)
        if save_ann_obj:
            assay.annObj = ann_obj  # before calling fit, so that annObj can be diagnosed if an error arises
        ann_obj.fit()

        if loadings is None:
            g = create_zarr_dataset(self._z[from_assay], loadings_name,
                                    (1000, 1000), 'f8', ann_obj.loadings.shape)
            g[:, :] = ann_obj.loadings
            assay.attrs[loadings_name] = loadings_hash

        self.cells.add(self._col_renamer(from_assay, cell_key, 'kmeans_cluster'), ann_obj.clusterLabels,
                       fill_val=-1, key=cell_key, overwrite=True)
        kmeans_loc = 'kmeans_cluster_centers' if cell_key == 'I' else cell_key + '_kmeans_cluster_centers'
        g = create_zarr_dataset(self._z[from_assay], kmeans_loc,
                                (1000, 1000), 'f8', ann_obj.kmeans.cluster_centers_.shape)
        g[:, :] = ann_obj.kmeans.cluster_centers_

        graph_loc = 'graph' if cell_key == 'I' else cell_key + '_graph'
        if loadings is not None:
            # This means that we used cached PCA data, which means that nothing changed upstream.
            store = self._z[from_assay][graph_loc]
            if store.attrs['k'] == k:
                print("INFO: ANN index instantiated but will reuse the existing graph.", flush=True)
                return None
        graph_dir = f"{self._fn}/{from_assay}/{graph_loc}"
        if os.path.isdir(graph_dir):
            shutil.rmtree(graph_dir, ignore_errors=True)
        store = self._z[from_assay].create_group(graph_loc)
        store.attrs['n_cells'] = ann_obj.nCells
        store.attrs['k'] = ann_obj.k
        store.attrs['self_uuid'] = uuid4().hex
        from .knn_utils import make_knn_graph
        make_knn_graph(ann_obj, batch_size, store, local_connectivity, bandwidth, save_raw_dists)
        return None

    def _load_graph(self, from_assay: str, cell_key: str, graph_format: str,
                    min_edge_weight: float = 0, symmetric: bool = True):
        graph_loc = 'graph' if cell_key == 'I' else cell_key + '_graph'
        if graph_loc not in self._z[from_assay]:
            print(f"ERROR: {graph_loc} not found for assay {from_assay}. Run `make_graph` for assay {from_assay}")
            return None
        if graph_format not in ['coo', 'csr']:
            raise KeyError("ERROR: format has to be either 'coo' or 'csr'")
        store = self._z[from_assay][graph_loc]
        n_cells = store.attrs['n_cells']
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

    def _ini_embed(self, from_assay: str, cell_key: str, n_comps: int):
        from sklearn.decomposition import PCA
        pc = PCA(n_components=n_comps).fit_transform(self._z[from_assay]['kmeans_cluster_centers'][:])
        for i in range(n_comps):
            pc[:, i] = rescale_array(pc[:, i])
        clusters = self.cells.table[self._col_renamer(from_assay, 'I', 'kmeans_cluster')]
        clusters = clusters[self.cells.table[cell_key]]
        return np.array([pc[x] for x in clusters]).astype(np.float32, order="C")

    def run_tsne(self, sgtsne_loc, from_assay: str = None, cell_key: str = 'I',
                 tsne_dims: int = 2, lambda_scale: float = 1.0, max_iter: int = 500, early_iter: int = 200,
                 alpha: int = 10, box_h: float = 0.7, temp_file_loc: str = '.') -> None:
        from uuid import uuid4
        from .knn_utils import export_knn_to_mtx
        from pathlib import Path

        if from_assay is None:
            from_assay = self.defaultAssay
        uid = str(uuid4())

        ini_emb_fn = Path(temp_file_loc, f'{uid}.txt').resolve()
        with open(ini_emb_fn, 'w') as h:
            ini_emb = self._ini_embed(from_assay, cell_key, 2).flatten()
            h.write('\n'.join(map(str, ini_emb)))
        knn_mtx_fn = Path(temp_file_loc, f'{uid}.mtx').resolve()
        export_knn_to_mtx(knn_mtx_fn, self._z[from_assay].graph)
        out_fn = Path(temp_file_loc, f'{uid}_output.txt').resolve()

        cmd = f"{sgtsne_loc} -m {max_iter} -l {lambda_scale} -d {tsne_dims} -e {early_iter} -p 1 -a {alpha}" \
              f" -h {box_h} -i {ini_emb_fn} -o {out_fn} {knn_mtx_fn}"
        os.system(cmd)
        emb = pd.read_csv(out_fn, header=None, sep=' ')[[0, 1]].values.T
        for i in range(tsne_dims):
            self.cells.add(self._col_renamer(from_assay, cell_key, f'tSNE{i + 1}'),
                           emb[i], key=cell_key, overwrite=True)
        for fn in [out_fn, knn_mtx_fn, ini_emb_fn]:
            Path.unlink(fn)

    def run_umap(self, *, from_assay: str = None, cell_key: str = 'I', use_full_graph: bool = True,
                 min_edge_weight: float = 0, ini_embed: np.ndarray = None, umap_dims: int = 2,
                 spread: float = 2.0, min_dist: float = 1, fit_n_epochs: int = 200, tx_n_epochs: int = 100,
                 random_seed: int = 4444, parallel: bool = False, **kwargs) -> None:
        from .umap import fit_transform
        if from_assay is None:
            from_assay = self.defaultAssay
        if use_full_graph:
            graph = self._load_graph(from_assay, cell_key, 'coo', min_edge_weight, symmetric=False)
        else:
            graph = self._load_graph(from_assay, 'I', 'csr', min_edge_weight, symmetric=False)
            nodes = np.where(self.cells.table[self.cells.table.I][cell_key].values)[0]
            graph = graph[nodes, :][:, nodes].tocoo()
        if ini_embed is None:
            ini_embed = self._ini_embed(from_assay, cell_key, umap_dims)
        t = fit_transform(graph, ini_embed, spread=spread, min_dist=min_dist,
                          tx_n_epochs=tx_n_epochs, fit_n_epochs=fit_n_epochs,
                          random_seed=random_seed, parallel=parallel, **kwargs)
        for i in range(umap_dims):
            self.cells.add(self._col_renamer(from_assay, cell_key, f'UMAP{i + 1}'),
                           t[:, i], key=cell_key, overwrite=True)
        return None

    def run_clustering(self, *, from_assay: str = None, cell_key: str = 'I',
                       n_clusters: int = None, min_edge_weight: float = 0, balanced_cut: bool = False,
                       max_size: int = None, min_size: int = None, max_distance_fc: float = 2,
                       return_clusters: bool = False) -> Union[None, pd.Series]:
        import sknetwork as skn
        from .dendrogram import BalancedCut
        if from_assay is None:
            from_assay = self.defaultAssay
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
        graph = self._load_graph(from_assay, cell_key, 'csr', min_edge_weight=min_edge_weight, symmetric=True)
        assay = self.__getattribute__(from_assay)
        graph_loc = 'graph' if cell_key == 'I' else cell_key + '_graph'
        graph_uuid = self._z[from_assay][graph_loc].attrs['self_uuid']
        params = [graph_uuid, min_edge_weight]  # this should not be changed to a tuple.
        # tuple are changed to list when saved as zarr attrs
        if 'dendrogram' in self._z[from_assay][graph_loc] and \
                assay.attrs['dendrogram_hash'] == params:
            dendrogram = self._z[from_assay][graph_loc]['dendrogram'][:]
            print("INFO: Using existing dendrogram", flush=True)
        else:
            paris = skn.hierarchy.Paris()
            dendrogram = paris.fit_transform(graph)
            dendrogram[dendrogram == np.Inf] = 0
            g = create_zarr_dataset(self._z[from_assay][graph_loc], 'dendrogram',
                                    (5000,), 'f8', (graph.shape[0] - 1, 4))
            g[:] = dendrogram
            assay.attrs['dendrogram_hash'] = params
        if balanced_cut:
            labels = BalancedCut(dendrogram, max_size, min_size, max_distance_fc).get_clusters()
        else:
            labels = skn.hierarchy.cut_straight(dendrogram, n_clusters=n_clusters) + 1
        if return_clusters:
            return pd.Series(labels, index=self.cells.table[cell_key].index[self.cells.table[cell_key]])
        else:
            self.cells.add(self._col_renamer(from_assay, cell_key, 'cluster'), labels,
                           fill_val=-1, key=cell_key, overwrite=True)

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

    def plot_gene_mean_var(self, **kwargs):
        from .plots import plot_mean_var
        assay = self.__getattribute__(self.defaultAssay)
        if 'hvgs' not in assay.feats.table.columns:
            raise KeyError(f"ERROR: HVGs have not been marked yet. Run 'mark_hvgs' first in {self.defaultAssay} assay.")
        nzm, vf, nc = [assay.feats.fetch(x).astype('float') for x in ['nz_mean', 'c_var', 'nCells']]
        plot_mean_var(nzm, vf, nc, assay.feats.fetch('hvgs'), **kwargs)

    def get_cell_vals(self, *, from_assay: str, cell_key: str, k: str, clip_fraction: float = 0):
        cell_idx = self.cells.active_index(cell_key)
        if k not in self.cells.table.columns:
            assay = self.__getattribute__(from_assay)
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

    def __repr__(self):
        x = ' '.join(self.assayNames)
        return f"DataStore with {self.cells.N} cells containing {len(self.assayNames)} assays: {x}"
