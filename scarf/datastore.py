import os
import numpy as np
from typing import List, Iterable, Tuple, Generator
import pandas as pd
import zarr
from tqdm import tqdm
import dask.array as daskarr
from .writers import create_zarr_dataset, create_zarr_obj_array
from .metadata import MetaData
from .assay import Assay, RNAassay, ATACassay, ADTassay
from .utils import calc_computed, system_call, clean_array

__all__ = ['DataStore']


def sanitize_hierarchy(z: zarr.hierarchy, assay_name: str) -> bool:
    """
    Test if an assay node in zarr object was created properly

    Args:
        z: Zarr hierarchy object
        assay_name: string value with name of assay

    Returns: True if assay_name is present in z and contains `counts` and `featureData` child nodes else raises error

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
    """
    DataStore objects provide primary interface to interact with the data.

    Args:
        zarr_loc: Path to Zarr file created using one of writer functions of Scarf
        assay_types: A dictionary with keys as assay names present in the Zarr file and values as either one of:
                     'RNA', 'ADT', 'ATAC' or 'GeneActivity'
        default_assay: Name of assay that should be considered as default. It is mandatory to provide this value
                       when DataStore loads a Zarr file for the first time
        min_features_per_cell: Minimum number of non-zero features in a cell. If lower than this then the cell
                               will be filtered out.
        min_cells_per_feature: Minimum number of cells where a feature has a non-zero value. Genes with values
                               less than this will be filtered out
        auto_filter: If True then the auto_filter method will be triggered
        show_qc_plots: If True then violin plots with per cell distribution of features will be shown. This does
                       not have an effect if `auto_filter` is False
        mito_pattern: Regex pattern to capture mitochondrial genes (default: 'MT-')
        ribo_pattern: Regex pattern to capture ribosomal genes (default: 'RPS|RPL|MRPS|MRPL')
        nthreads: Number of maximum threads to use in all multi-threaded functions
        dask_client: Dask client object to use instead of creating a new one
    """

    def __init__(self, zarr_loc: str, assay_types: dict = None, default_assay: str = None,
                 min_features_per_cell: int = 200, min_cells_per_feature: int = 20,
                 auto_filter: bool = False, show_qc_plots: bool = True,
                 mito_pattern: str = None, ribo_pattern: str = None, nthreads: int = 2, dask_client=None):
        from dask.distributed import Client, LocalCluster

        self._fn: str = zarr_loc
        self.z: zarr.hierarchy = zarr.open(self._fn, 'r+')
        self.nthreads = nthreads
        if dask_client is None:
            cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=nthreads)
            self.daskClient = Client(cluster)
        self.daskClient = dask_client
        # The order is critical here:
        self.cells = self._load_cells()
        self.assayNames = self._get_assay_names()
        self._defaultAssay = self._load_default_assay(default_assay)
        self._load_assays(min_cells_per_feature, assay_types)
        # TODO: Reset all attrs, pca, dendrogram etc
        self._ini_cell_props(min_features_per_cell, mito_pattern, ribo_pattern)
        if auto_filter:
            filter_attrs = ['nCounts', 'nFeatures', 'percentMito', 'percentRibo']
            if show_qc_plots:
                self.plot_cells_dists(cols=[self._defaultAssay + '_percent*'], all_cells=True)
            self.auto_filter_cells(attrs=[f'{self._defaultAssay}_{x}' for x in filter_attrs])
            if show_qc_plots:
                self.plot_cells_dists(cols=[self._defaultAssay + '_percent*'])

    def _load_cells(self) -> MetaData:
        """
        This convenience function loads cellData level from Zarr hierarchy

        Returns:
            Metadata object

        """
        if 'cellData' not in self.z:
            raise KeyError("ERROR: cellData not found in zarr file")
        return MetaData(self.z['cellData'])

    def _get_assay_names(self) -> List[str]:
        """
        Load all assay names present in the zarr file. Zarr writers create an 'is_assay' attribute in the assay level
        and this function looks for presence of those attributes to load assay names.

        Returns:
            Names of assays present in a Zarr file

        """
        assays = []
        for i in self.z.group_keys():
            if 'is_assay' in self.z[i].attrs.keys():
                sanitize_hierarchy(self.z, i)
                assays.append(i)
        return assays

    def _load_default_assay(self, assay_name: str = None) -> str:
        """
        This function sets a given assay name as defaultAssay attribute. If `assay_name` value is None then the
        top-level directory attributes in the Zarr file are looked up for presence of previously used default assay.

        Args:
            assay_name: Name of the assay to be considered for setting as default

        Returns:
            Name of the assay to be set as default assay

        """
        if assay_name is None:
            if 'defaultAssay' in self.z.attrs:
                assay_name = self.z.attrs['defaultAssay']
            else:
                if len(self.assayNames) == 1:
                    assay_name = self.assayNames[0]
                    self.z.attrs['defaultAssay'] = assay_name
                else:
                    raise ValueError("ERROR: You have more than one assay data. "
                                     f"Choose one from: {' '.join(self.assayNames)}\n using 'default_assay' parameter. "
                                     "Please note that names are case-sensitive.")
        else:
            if assay_name in self.assayNames:
                if 'defaultAssay' in self.z.attrs:
                    if assay_name != self.z.attrs['defaultAssay']:
                        print(f"ATTENTION: Default assay changed from {self.z.attrs['defaultAssay']} to {assay_name}")
                self.z.attrs['defaultAssay'] = assay_name
            else:
                raise ValueError(f"ERROR: The provided default assay name: {assay_name} was not found. "
                                 f"Please Choose one from: {' '.join(self.assayNames)}\n"
                                 "Please note that the names are case-sensitive.")
        return assay_name

    def _load_assays(self, min_cells: int, predefined_assays: dict = None) -> None:
        """
        This function loads all the assay names present in attribute `assayNames` as Assay objects. An attempt is made
        to automatically determine the most appropriate Assay class for each assay based on following mapping:

        literal_blocks::
            {'RNA': RNAassay, 'ATAC': ATACassay, 'ADT': ADTassay, 'GeneActivity': RNAassay}

        If an assay name does not match any of the keys above then it is assigned as generic assay class. This can be
        overridden using `predefined_assays` parameter

        Args:
            predefined_assays: A mapping of assay names to Assay class type to associated with. If
            min_cells: Minimum number of cells that a feature in each assay must be present to not be discarded (i.e.
                       receive False value in `I` column)

        Returns:
        """

        assay_types = {'RNA': RNAassay, 'ATAC': ATACassay, 'ADT': ADTassay, 'GeneActivity': RNAassay}
        # print_options = '\n'.join(["{'%s': '" + x + "'}" for x in assay_types])
        caution_statement = "CAUTION: %s was set as a generic Assay with no normalization. If this is unintended " \
                            "then please make sure that you provide a correct assay type for this assay using " \
                            "'assay_types' parameter."
        caution_statement = caution_statement + "\nIf you have more than one assay in the dataset then you can set " \
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
            setattr(self, i, assay(self.z, i, self.cells, min_cells_per_feature=min_cells))
        return None

    def _get_assay(self, from_assay: str) -> Assay:
        """
        This is convenience function used internally to quickly obtain the assay object that is linked to a assay name

        Args:
            from_assay: Name of the assay whose object is to be returned

        Returns:

        """
        if from_assay is None or from_assay == '':
            from_assay = self._defaultAssay
        return self.__getattribute__(from_assay)

    def _ini_cell_props(self, min_features: int, mito_pattern: str, ribo_pattern: str) -> None:
        """
        This function is called on class initialization. For each assay, it calculates per-cell statistics i.e. nCounts,
        nFeatures, percentMito and percentRibo. These statistics are then populated into the cell metadata table

        Args:
            min_features: Minimum features that a cell must have non-zero value before being filtered out
            mito_pattern: Regex pattern for identification of mitochondrial genes
            ribo_pattern: Regex pattern for identification of ribosomal genes

        Returns:

        """
        for from_assay in self.assayNames:
            assay = self._get_assay(from_assay)

            var_name = from_assay + '_nCounts'
            if var_name not in self.cells.table.columns:
                n_c = calc_computed(assay.rawData.sum(axis=1), f"INFO: ({from_assay}) Computing nCounts")
                self.cells.add(var_name, n_c, overwrite=True)

            var_name = from_assay + '_nFeatures'
            if var_name not in self.cells.table.columns:
                n_f = calc_computed((assay.rawData > 0).sum(axis=1), f"INFO: ({from_assay}) Computing nFeatures")
                self.cells.add(var_name, n_f, overwrite=True)

            if type(assay) == RNAassay:
                if mito_pattern is None:
                    mito_pattern = 'MT-'
                var_name = from_assay + '_percentMito'
                assay.add_percent_feature(mito_pattern, var_name)

                if ribo_pattern is None:
                    ribo_pattern = 'RPS|RPL|MRPS|MRPL'
                var_name = from_assay + '_percentRibo'
                assay.add_percent_feature(ribo_pattern, var_name)

            if from_assay == self._defaultAssay:
                v = self.cells.fetch(from_assay + '_nFeatures')
                if min_features > np.median(v):
                    print(f"WARNING: More than of half of the less have less than {min_features} features for assay: "
                          f"{from_assay}. Will not remove low quality cells automatically.")
                else:
                    self.cells.update(self.cells.sift(v, min_features, np.Inf))

    @staticmethod
    def _col_renamer(from_assay: str, cell_key: str, suffix: str) -> str:
        """
        A convenience function for internal usage that creates naming rule for the metadata columns

        Args:
            from_assay: name of the assay
            cell_key: cell key to use
            suffix: base name for the column

        Returns:
            column name updated as per the naming rule

        """
        if cell_key == 'I':
            ret_val = '_'.join(list(map(str, [from_assay, suffix])))
        else:
            ret_val = '_'.join(list(map(str, [from_assay, cell_key, suffix])))
        return ret_val

    def get_latest_feat_key(self, from_assay: str) -> str:
        """
        Looks up the the value in assay level attributes for key 'latest_feat_key'

        Args:
            from_assay: Assay whose latest feature is to be returned

        Returns:
            Name of the latest feature that was used to run `save_normalized_data`

        """
        assay = self._get_assay(from_assay)
        return assay.attrs['latest_feat_key']

    def set_default_assay(self, assay_name: str) -> None:
        """
        Override default assay

        Args:
            assay_name: Name of the assay that should be set as default

        Returns:

        Raises:
            ValueError: if `assay_name` is not found in attribute `assayNames`

        """
        if assay_name in self.assayNames:
            self._defaultAssay = assay_name
            self.z.attrs['defaultAssay'] = assay_name
        else:
            raise ValueError(f"ERROR: {assay_name} assay was not found.")

    def filter_cells(self, *, attrs: Iterable[str], lows: Iterable[int], highs: Iterable[int]) -> None:
        """
        Filter cells based on the cell metadata column values. Filtering triggers `update` method on  'I' column of
        cell metadata which uses 'and' operation. This means that cells that are not within the filtering thresholds
        will have value set as False in 'I' column of cell metadata table

        Args:
            attrs: Names of columns to be used for filtering
            lows: Lower bounds of thresholds for filtering. Should be in same order as the names in `attrs` parameter
            highs: Upper bounds of thresholds for filtering. Should be in same order as the names in `attrs` parameter

        Returns:

        """
        for i, j, k in zip(attrs, lows, highs):
            if i not in self.cells.table.columns:
                print(f"WARNING: {i} not found in cell metadata. Will ignore {i} for filtering")
                continue
            x = self.cells.sift(self.cells.table[i].values, j, k)
            print(f"INFO: {len(x) - x.sum()} cells flagged for filtering out using attribute {i}", flush=True)
            self.cells.update(x)

    def auto_filter_cells(self, *, attrs: Iterable[str], min_p: float = 0.01, max_p: float = 0.99) -> None:
        """
        Filter cells based on columns of the cell metadata table. This is wrapper function for `filer_cells` and
        determines the threshold values to be used for each column. For each cell metadata column, the function models a
        normal distribution using the median value and std. dev. of the column and then determines the point estimates
        of values at `min_p` and `max_p` fraction of densities.

        Args:
            attrs: column names to be used for filtering
            min_p: fractional density point to be used for calculating lower bounds of threshold
            max_p: fractional density point to be used for calculating lower bounds of threshold

        Returns:

        """
        from scipy.stats import norm

        for i in attrs:
            if i not in self.cells.table.columns:
                print(f"WARNING: {i} not found in cell metadata. Will ignore {i} for filtering")
                continue
            a = self.cells.table[i]
            dist = norm(np.median(a), np.std(a))
            self.filter_cells(attrs=[i], lows=[dist.ppf(min_p)], highs=[dist.ppf(max_p)])

    @staticmethod
    def _choose_reduction_method(assay: Assay, reduction_method: str) -> str:
        """
        This is convenience function to determine the linear dimension reduction method to be used for a given assay.
        It is uses a predetermine rule to make this determination.

        Args:
            assay: Assay object
            reduction_method: name of reduction method to use. It can be one from either: 'pca', 'lsi', 'auto'

        Returns:
            The name of dimension reduction method to be used. Either 'pca' or 'lsi'

        Raises:
            ValueError: If `reduction_method` is not one of either 'pca', 'lsi', 'auto'

        """
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

    def _set_graph_params(self, from_assay, cell_key, feat_key, log_transform=None, renormalize_subset=None,
                          reduction_method='auto', dims=None, pca_cell_key=None,
                          ann_metric=None, ann_efc=None, ann_ef=None, ann_m=None,
                          rand_state=None, k=None, n_centroids=None, local_connectivity=None, bandwidth=None) -> tuple:
        """
        This function allows determination of values for the paramters of `make_graph` function. This function harbours
        the default values for each parameter.  If parameter value is None, then before choosing the default, it tries
        to use the values from latest iteration of the step within the same hierarchy tree.
        Find details for parameters in the `make_graph` method

        Args:
            from_assay:
            cell_key:
            feat_key:
            log_transform:
            renormalize_subset:
            reduction_method:
            dims:
            pca_cell_key:
            ann_metric:
            ann_efc:
            ann_ef:
            ann_m:
            rand_state:
            k:
            n_centroids:
            local_connectivity:
            bandwidth:

        Returns:
            Finalized values for the all the optional parameters in the same order

        """
        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        if log_transform is None or renormalize_subset is None:
            if normed_loc in self.z and 'subset_params' in self.z[normed_loc].attrs:
                # This works in coordination with save_normalized_data
                subset_params = self.z[normed_loc].attrs['subset_params']
                c_log_transform, c_renormalize_subset = (subset_params['log_transform'],
                                                         subset_params['renormalize_subset'])
            else:
                c_log_transform, c_renormalize_subset = None, None
            if log_transform is None:
                if c_log_transform is not None:
                    log_transform = bool(c_log_transform)
                    print(f'INFO: No value provided for parameter `log_transform`. '
                          f'Will use previously used value: {log_transform}', flush=True)
                else:
                    log_transform = True
                    print(f'INFO: No value provided for parameter `log_transform`. '
                          f'Will use default value: {log_transform}', flush=True)
            if renormalize_subset is None:
                if c_renormalize_subset is not None:
                    renormalize_subset = bool(c_renormalize_subset)
                    print(f'INFO: No value provided for parameter `renormalize_subset`. '
                          f'Will use previously used value: {renormalize_subset}', flush=True)
                else:
                    renormalize_subset = True
                    print(f'INFO: No value provided for parameter `renormalize_subset`. '
                          f'Will use default value: {renormalize_subset}', flush=True)
        log_transform = bool(log_transform)
        renormalize_subset = bool(renormalize_subset)

        if dims is None or pca_cell_key is None:
            if normed_loc in self.z and 'latest_reduction' in self.z[normed_loc].attrs:
                reduction_loc = self.z[normed_loc].attrs['latest_reduction']
                c_dims, c_pca_cell_key = reduction_loc.rsplit('__', 2)[1:]
            else:
                c_dims, c_pca_cell_key = None, None
            if dims is None:
                if c_dims is not None:
                    dims = int(c_dims)
                    print(f'INFO: No value provided for parameter `dims`. '
                          f'Will use previously used value: {dims}', flush=True)
                else:
                    dims = 11
                    print(f'INFO: No value provided for parameter `dims`. '
                          f'Will use default value: {dims}', flush=True)
            if pca_cell_key is None:
                if c_pca_cell_key is not None:
                    pca_cell_key = c_pca_cell_key
                    print(f'INFO: No value provided for parameter `pca_cell_key`. '
                          f'Will use previously used value: {pca_cell_key}', flush=True)
                else:
                    pca_cell_key = cell_key
                    print(f'INFO: No value provided for parameter `pca_cell_key`. '
                          f'Will use same value as cell_key: {pca_cell_key}', flush=True)
            else:
                if pca_cell_key not in self.cells.table.columns:
                    raise ValueError(f"ERROR: `pca_use_cell_key` {pca_cell_key} does not exist in cell metadata")
                if self.cells.table[pca_cell_key].dtype != bool:
                    raise TypeError("ERROR: Type of `pca_use_cell_key` column in cell metadata should be `bool`")
        dims = int(dims)
        reduction_method = self._choose_reduction_method(self._get_assay(from_assay), reduction_method)
        reduction_loc = f"{normed_loc}/reduction__{reduction_method}__{dims}__{pca_cell_key}"

        if ann_metric is None or ann_efc is None or ann_ef is None or ann_m is None or rand_state is None:
            if reduction_loc in self.z and 'latest_ann' in self.z[reduction_loc].attrs:
                ann_loc = self.z[reduction_loc].attrs['latest_ann']
                c_ann_metric, c_ann_efc, c_ann_ef, c_ann_m, c_rand_state = \
                    ann_loc.rsplit('/', 1)[1].split('__')[1:]
            else:
                c_ann_metric, c_ann_efc, c_ann_ef, c_ann_m, c_rand_state = \
                    None, None, None, None, None
            if ann_metric is None:
                if c_ann_metric is not None:
                    ann_metric = c_ann_metric
                    print(f'INFO: No value provided for parameter `ann_metric`. '
                          f'Will use previously used value: {ann_metric}', flush=True)
                else:
                    ann_metric = 'l2'
                    print(f'INFO: No value provided for parameter `ann_metric`. '
                          f'Will use default value: {ann_metric}', flush=True)
            if ann_efc is None:
                if c_ann_efc is not None:
                    ann_efc = int(c_ann_efc)
                    print(f'INFO: No value provided for parameter `ann_efc`. '
                          f'Will use previously used value: {ann_efc}', flush=True)
                else:
                    ann_efc = None  # Will be set after value for k is determined
                    print(f'INFO: No value provided for parameter `ann_efc`. Will use default value:'
                          f'min(100, max(k * 3, 50))', flush=True)
            if ann_ef is None:
                if c_ann_ef is not None:
                    ann_ef = int(c_ann_ef)
                    print(f'INFO: No value provided for parameter `ann_ef`. '
                          f'Will use previously used value: {ann_ef}', flush=True)
                else:
                    ann_ef = None  # Will be set after value for k is determined
                    print(f'INFO: No value provided for parameter `ann_efc`. Will use default value: '
                          f'min(100, max(k * 3, 50))', flush=True)
            if ann_m is None:
                if c_ann_m is not None:
                    ann_m = int(c_ann_m)
                    print(f'INFO: No value provided for parameter `ann_m`. '
                          f'Will use previously used value: {ann_m}', flush=True)
                else:
                    ann_m = min(max(48, int(dims * 1.5)), 64)
                    print(f'INFO: No value provided for parameter `ann_m`. Will use default value: {ann_m}', flush=True)
            if rand_state is None:
                if c_rand_state is not None:
                    rand_state = int(c_rand_state)
                    print(f'INFO: No value provided for parameter `rand_state`. '
                          f'Will use previously used value: {rand_state}', flush=True)
                else:
                    rand_state = 4466
                    print(f'INFO: No value provided for parameter `rand_state`. '
                          f'Will use default value: {rand_state}', flush=True)
        ann_metric = str(ann_metric)
        ann_m = int(ann_m)
        rand_state = int(rand_state)

        if k is None:
            if reduction_loc in self.z and 'latest_ann' in self.z[reduction_loc].attrs:
                ann_loc = self.z[reduction_loc].attrs['latest_ann']
                knn_loc = self.z[ann_loc].attrs['latest_knn']
                k = int(knn_loc.rsplit('__', 1)[1])  # depends on param_joiner
                print(f'INFO: No value provided for parameter `k`. Will use previously used value: {k}', flush=True)
            else:
                k = 11
                print(f'INFO: No value provided for parameter `k`. Will use default value: {k}', flush=True)
        k = int(k)
        if ann_ef is None:
            ann_ef = min(100, max(k * 3, 50))
        ann_ef = int(ann_ef)
        if ann_efc is None:
            ann_efc = min(100, max(k * 3, 50))
        ann_efc = int(ann_efc)
        ann_loc = f"{reduction_loc}/ann__{ann_metric}__{ann_efc}__{ann_ef}__{ann_m}__{rand_state}"
        knn_loc = f"{ann_loc}/knn__{k}"

        if n_centroids is None:
            if reduction_loc in self.z and 'latest_kmeans' in self.z[reduction_loc].attrs:
                kmeans_loc = self.z[reduction_loc].attrs['latest_kmeans']
                n_centroids = int(kmeans_loc.split('/')[-1].split('__')[1])  # depends on param_joiner
                print(f'INFO: No value provided for parameter `n_centroids`.'
                      f' Will use previously used value: {n_centroids}', flush=True)
            else:
                # n_centroids = min(data.shape[0]/10, max(500, data.shape[0]/100))
                n_centroids = 500
                print(f'INFO: No value provided for parameter `n_centroids`. '
                      f'Will use default value: {n_centroids}', flush=True)
        n_centroids = int(n_centroids)

        if local_connectivity is None or bandwidth is None:
            if knn_loc in self.z and 'latest_graph' in self.z[knn_loc].attrs:
                graph_loc = self.z[knn_loc].attrs['latest_graph']
                c_local_connectivity, c_bandwidth = map(float, graph_loc.rsplit('/')[-1].split('__')[1:])
            else:
                c_local_connectivity, c_bandwidth = None, None
            if local_connectivity is None:
                if c_local_connectivity is not None:
                    local_connectivity = c_local_connectivity
                    print(f'INFO: No value provided for parameter `local_connectivity`. '
                          f'Will use previously used value: {local_connectivity}', flush=True)
                else:
                    local_connectivity = 1.0
                    print(f'INFO: No value provided for parameter `local_connectivity`. '
                          f'Will use default value: {local_connectivity}', flush=True)
            if bandwidth is None:
                if c_bandwidth is not None:
                    bandwidth = c_bandwidth
                    print(f'INFO: No value provided for parameter `bandwidth`. '
                          f'Will use previously used value: {bandwidth}', flush=True)
                else:
                    bandwidth = 1.5
                    print(f'INFO: No value provided for parameter `bandwidth`. Will use default value: {bandwidth}',
                          flush=True)
        local_connectivity = float(local_connectivity)
        bandwidth = float(bandwidth)

        return (log_transform, renormalize_subset, reduction_method, dims, pca_cell_key,
                ann_metric, ann_efc, ann_ef, ann_m, rand_state, k, n_centroids, local_connectivity, bandwidth)

    def make_graph(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                   pca_cell_key: str = None, reduction_method: str = 'auto', dims: int = None, k: int = None,
                   ann_metric: str = None, ann_efc: int = None, ann_ef: int = None, ann_m: int = None,
                   rand_state: int = None, n_centroids: int = None, batch_size: int = None,
                   log_transform: bool = None, renormalize_subset: bool = None,
                   local_connectivity: float = None, bandwidth: float = None,
                   update_feat_key: bool = True, return_ann_object: bool = False, feat_scaling: bool = True):
        """
        Creates a cell neighbourhood graph. Performs following steps in the process:

        - Normalizes the data calling the `save_normalized_data` for the assay
        - instantiates ANNStream class which perform dimension reduction, feature scaling (optional) and fits ANN index
        - queries ANN index for nearest neighbours and saves the distances and indices of the neighbours
        - recalculates the distances to bound them into 0 and 1 (check out knn_utils module for details)
        - saves the indices and distances in sparse graph friendly form
        - fits a MiniBatch kmeans on the data

        The data for all the steps is saved in the Zarr in the following hierarchy which is organized based on data
        dependency. Parameter values for each step are incorporated into group names in the hierarchy::

            RNA
            ├── normed__I__hvgs
            │   ├── data (7648, 2000) float64                 # Normalized data
            │   └── reduction__pca__31__I                     # Dimension reduction group
            │       ├── mu (2000,) float64                    # Means of normalized feature values
            │       ├── sigma (2000,) float64                 # Std dev. of normalized feature values
            │       ├── reduction (2000, 31) float64          # PCA loadings matrix
            │       ├── ann__l2__63__63__48__4466             # ANN group named with ANN parameters
            │       │   └── knn__21                           # KNN group with value of k in name
            │       │       ├── distances (7648, 21) float64  # Raw distance matrix for k neighbours
            │       │       ├── indices (7648, 21) uint64     # Indices for k neighbours
            │       │       └── graph__1.0__1.5               # sparse graph with continuous form distance values
            │       │           ├── edges (160608, 2) uint64
            │       │           └── weights (160608,) float64
            │       └── kmeans__100__4466                     # Kmeans groups
            │           ├── cluster_centers (100, 31) float64 # Centroid matrix
            │           └── cluster_labels (7648,) float64    # Cluster labels for cells
            ...

        The most recent child of each hierarchy node is noted for quick retrieval and in cases where multiple child
        nodes exist. Parameters starting with `ann` are forwarded to HNSWlib. More details about these parameters can
        be found here: https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        
        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for graph creation. By default all cells with True value in 'I' will be used.
                      The provided value for `cell_key` should be a column in cell metadata table with boolean values.
            feat_key: Features to use for graph creation. It is a required parameter. We have chosen not to set this
                      to 'I' by default because this might lead to usage of too many features and may lead to poor
                      results. The value for `feat_key` should be a column in feature metadata from the `from_assay`
                      assay and should be boolean type.
            pca_cell_key: Name of a column from cell metadata table. This column should be boolean type. If no value is
                          provided then the value is set to same as `cell_key` which means all the cells in the
                          normalized data will be used for fitting the pca. This parameter, hence, basically provides a
                          mechanism to subset the normalized data only for PCA fitting step. This parameter can be
                          useful, for example, the data has cells from multiple replicates which wont merge together, in
                          which case the `pca_cell_key` can be used to fit PCA on cells from only one of the replicate.
            reduction_method: Method to use for linear dimension reduction. Could be either 'pca', 'lsi' or 'auto'. In
                              case of 'auto' `_choose_reduction_method` will be used to determine best reduction type
                              for the assay.
            dims: Number of top reduced dimensions to use (Default value: 11)
            k: Number of nearest neighbours to query for each cell (Default value: 11)
            ann_metric: Refer to HNSWlib link above (Default value: 'l2')
            ann_efc: Refer to HNSWlib link above (Default value: min(100, max(k * 3, 50)))
            ann_ef: Refer to HNSWlib link above (Default value: min(100, max(k * 3, 50)))
            ann_m: Refer to HNSWlib link above (Default value: min(max(48, int(dims * 1.5)), 64) )
            rand_state: Random seed number (Default value: 4466)
            n_centroids: Number of centroids for Kmeans clustering. As a general idication, have a value of 1+ for every
                         100 cells. Small small (<2000 cells) and very small (<500 cells) use a ballpark number for max
                         expected number of clusters (Default value: 500). The results of kmeans clustering are only
                         used to provide initial embedding for UMAP and tSNE. (Default value: 500)
            batch_size: Number of cells in a batch. This number is guided by number of features being used and the
                        amount of available free memory. Though the full data is already divided into chunks, however,
                        if only a fraction of features are being used in the normalized dataset, then the chunk size
                        can be increased to speed up the computation (i.e. PCA fitting and ANN index building).
                        (Default value: 1000)
            log_transform: If True, then the normalized data is log-transformed (only affects RNAassay type assays).
                           (Default value: True)
            renormalize_subset: If True, then the data is normalized using only those features that are True in
                                `feat_key` column rather using total expression of all features in a cell (only affects
                                RNAassay type assays). (Default value: True)
            local_connectivity: This parameter is forwarded to `smooth_knn_dist` function from UMAP package. Higher
                                value will push distribution of edge weights towards terminal values (binary like).
                                Lower values will accumulate edge weights around the mean produced by `bandwidth`
                                parameter. (Default value: 1.0)
            bandwidth: This parameter is forwarded to `smooth_knn_dist` function from UMAP package. Higher value will
                       push the mean of distribution of graph edge weights towards right.  (Default value: 1.5). Read
                       more about `smooth_knn_dist` function here:
                       https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.smooth_knn_dist
            update_feat_key: If True (default) then `latest_feat_key` zarr attribute of the assay will be updated.
                             Choose False if you are experimenting with a `feat_key` do not want to overide existing
                             `latest_feat_key` and by extension `latest_graph`.
            return_ann_object: If True then returns the ANNStream object. This allows one to directly interact with the
                               PCA transformer and HNSWlib index. Check out ANNStream documentation to know more.
                               (Default: False)
            feat_scaling: If True (default) then the feature will be z-scaled otherwise not. It is highly recommended to
                          keep this as True unless you know what you are doing. `feat_scaling` is internally turned off
                          when during cross sample mapping using CORAL normalized values are being used. Read more about
                          this in `run_mapping` method.

        Returns:
            Either None or `AnnStream` object

        """
        from .ann import AnnStream

        if from_assay is None:
            from_assay = self._defaultAssay
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
        (log_transform, renormalize_subset, reduction_method, dims, pca_cell_key, ann_metric, ann_efc, ann_ef, ann_m,
         rand_state, k, n_centroids, local_connectivity, bandwidth) = self._set_graph_params(
            from_assay, cell_key, feat_key, log_transform, renormalize_subset, reduction_method, dims, pca_cell_key,
            ann_metric, ann_efc, ann_ef, ann_m, rand_state, k, n_centroids, local_connectivity, bandwidth
        )
        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = f"{normed_loc}/reduction__{reduction_method}__{dims}__{pca_cell_key}"
        ann_loc = f"{reduction_loc}/ann__{ann_metric}__{ann_efc}__{ann_ef}__{ann_m}__{rand_state}"
        ann_idx_loc = f"{self._fn}/{ann_loc}/ann_idx"
        knn_loc = f"{ann_loc}/knn__{k}"
        kmeans_loc = f"{reduction_loc}/kmeans__{n_centroids}__{rand_state}"
        graph_loc = f"{knn_loc}/graph__{local_connectivity}__{bandwidth}"

        data = assay.save_normalized_data(cell_key, feat_key, batch_size, normed_loc.split('/')[-1],
                                          log_transform, renormalize_subset, update_feat_key)
        loadings = None
        fit_kmeans = True
        fit_ann = True
        mu, sigma = np.ndarray([]), np.ndarray([])
        use_for_pca = self.cells.fetch(pca_cell_key, key=cell_key)
        if reduction_loc in self.z:
            loadings = self.z[reduction_loc]['reduction'][:]
            if reduction_method == 'pca':
                mu = self.z[reduction_loc]['mu'][:]
                sigma = self.z[reduction_loc]['sigma'][:]
            print(f"INFO: Using existing loadings for {reduction_method} with {dims} dims", flush=True)
        else:
            if reduction_method == 'pca':
                mu = clean_array(calc_computed(data.mean(axis=0),
                                               'INFO: Calculating mean of norm. data'))
                sigma = clean_array(calc_computed(data.std(axis=0),
                                                  'INFO: Calculating std. dev. of norm. data'), 1)
        if ann_loc in self.z:
            fit_ann = False
            print(f"INFO: Using existing ANN index", flush=True)
        if kmeans_loc in self.z:
            fit_kmeans = False
            print(f"INFO: using existing kmeans cluster centers", flush=True)
        ann_obj = AnnStream(data=data, k=k, n_cluster=n_centroids, reduction_method=reduction_method,
                            dims=dims, loadings=loadings, use_for_pca=use_for_pca,
                            mu=mu, sigma=sigma, ann_metric=ann_metric, ann_efc=ann_efc,
                            ann_ef=ann_ef, ann_m=ann_m, ann_idx_loc=ann_idx_loc, nthreads=self.nthreads,
                            rand_state=rand_state, do_ann_fit=fit_ann, do_kmeans_fit=fit_kmeans,
                            scale_features=feat_scaling)

        if loadings is None:
            self.z.create_group(reduction_loc, overwrite=True)
            g = create_zarr_dataset(self.z[reduction_loc], 'reduction', (1000, 1000), 'f8', ann_obj.loadings.shape)
            g[:, :] = ann_obj.loadings
            if reduction_method == 'pca':
                g = create_zarr_dataset(self.z[reduction_loc], 'mu', (100000,), 'f8', mu.shape)
                g[:] = mu
                g = create_zarr_dataset(self.z[reduction_loc], 'sigma', (100000,), 'f8', sigma.shape)
                g[:] = sigma
        if ann_loc not in self.z:
            self.z.create_group(ann_loc, overwrite=True)
            ann_obj.annIdx.save_index(ann_idx_loc)
        if fit_kmeans:
            self.z.create_group(kmeans_loc, overwrite=True)
            g = create_zarr_dataset(self.z[kmeans_loc], 'cluster_centers',
                                    (1000, 1000), 'f8', ann_obj.kmeans.cluster_centers_.shape)
            g[:, :] = ann_obj.kmeans.cluster_centers_
            g = create_zarr_dataset(self.z[kmeans_loc], 'cluster_labels', (100000,), 'f8', ann_obj.clusterLabels.shape)
            g[:] = ann_obj.clusterLabels

        if knn_loc in self.z and graph_loc in self.z:
            print(f"INFO: KNN graph already exists will not recompute.", flush=True)
        else:
            from .knn_utils import self_query_knn, smoothen_dists
            if knn_loc not in self.z:
                self_query_knn(ann_obj, self.z.create_group(knn_loc, overwrite=True), batch_size, self.nthreads)
            smoothen_dists(self.z.create_group(graph_loc, overwrite=True),
                           self.z[knn_loc]['indices'], self.z[knn_loc]['distances'],
                           local_connectivity, bandwidth)

        self.z[normed_loc].attrs['latest_reduction'] = reduction_loc
        self.z[reduction_loc].attrs['latest_ann'] = ann_loc
        self.z[reduction_loc].attrs['latest_kmeans'] = kmeans_loc
        self.z[ann_loc].attrs['latest_knn'] = knn_loc
        self.z[knn_loc].attrs['latest_graph'] = graph_loc
        if return_ann_object:
            return ann_obj
        return None

    def _get_latest_graph_loc(self, from_assay: str, cell_key: str, feat_key: str) -> str:
        """
        Convenience function to identify location of latest graph in the Zarr hierarchy.

        Args:
            from_assay: Name of the assay
            cell_key: Cell key used to create the graph
            feat_key: Feature key used to create the graph

        Returns:
            Path of graph in the Zarr hierarchy

        """
        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = self.z[normed_loc].attrs['latest_reduction']
        ann_loc = self.z[reduction_loc].attrs['latest_ann']
        knn_loc = self.z[ann_loc].attrs['latest_knn']
        return self.z[knn_loc].attrs['latest_graph']

    def load_graph(self, from_assay: str, cell_key: str, feat_key: str, graph_format: str,
                   min_edge_weight: float, symmetric: bool, upper_only: bool):
        """
        Load the cell neighbourhood as a scipy sparse matrix

        Args:
            from_assay: Name of the assay/
            cell_key: Cell key used to create the graph
            feat_key: Feature key used to create the graph
            graph_format: Can be either 'csr' or 'coo'.
            min_edge_weight: Edges with weights less than this value are removed.
            symmetric: If True, makes the graph symmetric by adding it to its transpose.
            upper_only: If True, then only the values from upper triangular of the matrix are returned. This is only
                       used when symmetric is True

        Returns:
            A scipy sparse matrix representing cell neighbourhood graph.

        """
        from scipy.sparse import coo_matrix, csr_matrix, triu

        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        if graph_loc not in self.z:
            print(f"ERROR: {graph_loc} not found in zarr location {self._fn}. Run `make_graph` for assay {from_assay}")
            return None
        if graph_format not in ['coo', 'csr']:
            raise KeyError("ERROR: format has to be either 'coo' or 'csr'")
        store = self.z[graph_loc]
        knn_loc = self.z[graph_loc.rsplit('/', 1)[0]]
        n_cells = knn_loc['indices'].shape[0]
        # TODO: can we have a progress bar for graph loading. Append to coo matrix?
        graph = coo_matrix((store['weights'][:], (store['edges'][:, 0], store['edges'][:, 1])),
                           shape=(n_cells, n_cells))
        if symmetric:
            graph = (graph + graph.T) / 2
            if upper_only:
                graph = triu(graph)
            else:
                graph = graph.tocoo()
        idx = graph.data > min_edge_weight
        if graph_format == 'coo':
            return coo_matrix((graph.data[idx], (graph.row[idx], graph.col[idx])), shape=(n_cells, n_cells))
        else:
            return csr_matrix((graph.data[idx], (graph.row[idx], graph.col[idx])), shape=(n_cells, n_cells))

    def get_ini_embed(self, from_assay: str, cell_key: str, feat_key: str, n_comps: int) -> np.ndarray:
        """
        Runs PCA on kmeans cluster centers and ascribes the PC values to individual cells based on their cluster
        labels. This is used in `run_umap` and `run_tsne` for initial embedding of cells. Uses `rescale_array` to
        to reduce the magnitude of extreme values.

        Args:
            from_assay: Name fo the assay for which Kmeans was fit
            cell_key: Cell key used
            feat_key: Feature key used
            n_comps: Number of PC components to use

        Returns:
            Matrix with n_comps dimensions representing initial embedding of cells.

        """
        from sklearn.decomposition import PCA
        from .utils import rescale_array

        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = self.z[normed_loc].attrs['latest_reduction']
        kmeans_loc = self.z[reduction_loc].attrs['latest_kmeans']
        pc = PCA(n_components=n_comps).fit_transform(self.z[kmeans_loc]['cluster_centers'][:])
        for i in range(n_comps):
            pc[:, i] = rescale_array(pc[:, i])
        clusters = self.z[kmeans_loc]['cluster_labels'][:].astype(np.uint32)
        return np.array([pc[x] for x in clusters]).astype(np.float32, order="C")

    def run_tsne(self, *, sgtsne_loc, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                 min_edge_weight: float = -1, symmetric_graph: bool = False, graph_upper_only: bool = False,
                 ini_embed: np.ndarray = None, tsne_dims: int = 2, lambda_scale: float = 1.0, max_iter: int = 500,
                 early_iter: int = 200, alpha: int = 10, box_h: float = 0.7, temp_file_loc: str = '.',
                 label: str = 'tSNE', verbose: bool = True) -> None:
        """
        Run SGtSNE-pi (Read more here: https://github.com/fcdimitr/sgtsnepi/tree/v1.0.1). This is an implementaion of
        tSNE that runs directly on graph structures. We use the graphs generated by `make_graph` method to create a
        layout of cells using tSNE algorithm. This function makes a system call to sgtSNE binary.
        To get a better understanding of how the parameters affect the embedding, check this out:
        http://t-sne-pi.cs.duke.edu/

        Args:
            sgtsne_loc: Location of sgtSNE binary
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            min_edge_weight: This parameter is forwarded to `load_graph` and is same as there. (Default value: -1)
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            ini_embed: Initial embedding coordinates for the cells in cell_key. Should have same number of columns as
                       tsne_dims. If not value is provided then the initial embedding is obtained using `get_ini_embed`.
            tsne_dims: Number of tSNE dimensions to compute (Default value: 2)
            lambda_scale: λ rescaling parameter (Default value: 1.0)
            max_iter: Maximum number of iterations (Default value: 500)
            early_iter: Number of early exaggeration iterations (Default value: 200)
            alpha: Early exaggeration multiplier (Default value: 10)
            box_h: Grid side length (accuracy control). Lower values might drastically slow down
                   the algorithm (Default value: 0.7)
            temp_file_loc: Location of temporary file. By default these files will be created in the current working
                           directory. These files are deleted before the method returns.
            label: base label for tSNE dimensions in the cell metadata column (Default value: 'tSNE')
            verbose: If True (default) then the full log from SGtSNEpi algorithm is shown.

        Returns:

        """
        from uuid import uuid4
        from .knn_utils import export_knn_to_mtx
        from pathlib import Path

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)

        uid = str(uuid4())

        knn_mtx_fn = Path(temp_file_loc, f'{uid}.mtx').resolve()
        graph = self.load_graph(from_assay, cell_key, feat_key, 'csr', min_edge_weight,
                                symmetric_graph, graph_upper_only)
        export_knn_to_mtx(knn_mtx_fn, graph)

        ini_emb_fn = Path(temp_file_loc, f'{uid}.txt').resolve()
        with open(ini_emb_fn, 'w') as h:
            if ini_embed is None:
                ini_embed = self.get_ini_embed(from_assay, cell_key, feat_key, tsne_dims).flatten()
            else:
                if ini_embed.shape != (graph.shape[0], tsne_dims):
                    raise ValueError("ERROR: Provided initial embedding does not shape required shape: "
                                     f"{(graph.shape[0], tsne_dims)}")
            h.write('\n'.join(map(str, ini_embed)))
        out_fn = Path(temp_file_loc, f'{uid}_output.txt').resolve()
        cmd = f"{sgtsne_loc} -m {max_iter} -l {lambda_scale} -d {tsne_dims} -e {early_iter} -p 1 -a {alpha}" \
              f" -h {box_h} -i {ini_emb_fn} -o {out_fn} {knn_mtx_fn}"
        if verbose:
            system_call(cmd)
        else:
            os.system(cmd)
        emb = pd.read_csv(out_fn, header=None, sep=' ')[[0, 1]].values.T
        for i in range(tsne_dims):
            self.cells.add(self._col_renamer(from_assay, cell_key, f'{label}{i + 1}'),
                           emb[i], key=cell_key, overwrite=True)
        for fn in [out_fn, knn_mtx_fn, ini_emb_fn]:
            Path.unlink(fn)

    def run_umap(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                 min_edge_weight: float = -1, symmetric_graph: bool = False, graph_upper_only: bool = False,
                 ini_embed: np.ndarray = None, umap_dims: int = 2, spread: float = 2.0, min_dist: float = 1,
                 fit_n_epochs: int = 200, tx_n_epochs: int = 100, set_op_mix_ratio: float = 1.0,
                 repulsion_strength: float = 1.0, initial_alpha: float = 1.0, negative_sample_rate: float = 5,
                 random_seed: int = 4444, label='UMAP') -> None:
        """
        Runs UMAP algorithm using the precomputed cell-neighbourhood graph. The calculated UMAP coordinates are saved
        in the cell metadata table

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            min_edge_weight: This parameter is forwarded to `load_graph` and is same as there. (Default value: -1)
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: False)
            ini_embed: Initial embedding coordinates for the cells in cell_key. Should have same number of columns as
                       umap_dims. If not value is provided then the initial embedding is obtained using `get_ini_embed`.
            umap_dims: Number of dimensions of UMAP embedding (Default value: 2)
            spread: Same as spread in UMAP package.  The effective scale of embedded points. In combination with
                    ``min_dist`` this determines how clustered/clumped the embedded points are.
            min_dist: Same as min_dist in UMAP package. The effective minimum distance between embedded points.
                      Smaller values will result in a more clustered/clumped embedding where nearby points on the
                      manifold are drawn closer together, while larger values will result on a more even dispersal of
                      points. The value should be set relative to the ``spread`` value, which determines the scale at
                      which embedded points will be spread out. (Default value: 1)
            fit_n_epochs: Same as n_epochs in UMAP package. The number of training epochs to be used in optimizing the
                          low dimensional embedding. Larger values result in more accurate embeddings.
                          (Default value: 200)
            tx_n_epochs: NUmber of epochs during transform (Default value: 100)
            set_op_mix_ratio: Same as set_op_mix_ratio in UMAP package. Interpolate between (fuzzy) union and
                              intersection as the set operation used to combine local fuzzy simplicial sets to obtain
                              a global fuzzy simplicial sets. Both fuzzy set operations use the product t-norm.
                              The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a
                              pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
            repulsion_strength: Same as repulsion_strength in UMAP package. Weighting applied to negative samples in
                                low dimensional embedding optimization. Values higher than one will result in greater
                                weight being given to negative samples. (Default value: 1.0)
            initial_alpha: Same as learning_rate in UMAP package. The initial learning rate for the embedding
                           optimization. (Default value: 1.0)
            negative_sample_rate: Same as negative_sample_rate in UMAP package. The number of negative samples to
                                  select per positive sample in the optimization process. Increasing this value will
                                  result in greater repulsive force being applied, greater optimization cost, but
                                  slightly more accuracy. (Default value: 5)
            random_seed: (Default value: 4444)
            label: base label for UMAP dimensions in the cell metadata column (Default value: 'UMAP')

        Returns:

        """
        from .umap import fit_transform
        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)
        graph = self.load_graph(from_assay, cell_key, feat_key, 'coo', min_edge_weight,
                                symmetric_graph, graph_upper_only)
        if ini_embed is None:
            ini_embed = self.get_ini_embed(from_assay, cell_key, feat_key, umap_dims)
        t = fit_transform(graph=graph, ini_embed=ini_embed, spread=spread, min_dist=min_dist,
                          tx_n_epochs=tx_n_epochs, fit_n_epochs=fit_n_epochs,
                          random_seed=random_seed, set_op_mix_ratio=set_op_mix_ratio,
                          repulsion_strength=repulsion_strength, initial_alpha=initial_alpha,
                          negative_sample_rate=negative_sample_rate)
        for i in range(umap_dims):
            self.cells.add(self._col_renamer(from_assay, cell_key, f'{label}{i + 1}'),
                           t[:, i], key=cell_key, overwrite=True)
        return None

    def run_leiden_clustering(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                              resolution: int = 1, min_edge_weight: float = -1,
                              symmetric_graph: bool = True, graph_upper_only: bool = True,
                              label: str = 'leiden_cluster', random_seed: int = 4444) -> None:
        """
        Executes Leiden graph clustering algorithm on the cell-neighbourhood graph and saves cluster identities in the
        cell metadata column.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            resolution: Resolution parameter for `RBConfigurationVertexPartition` configuration
            min_edge_weight: This parameter is forwarded to `load_graph` and is same as there. (Default value: -1)
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            label: base label for cluster identity in the cell metadata column (Default value: 'leiden_cluster')
            random_seed: (Default value: 4444)

        Returns:

        """
        try:
            import leidenalg
        except ImportError:
            raise ImportError("ERROR: 'leidenalg' package is not installed. Please find the installation instructions "
                              "here: https://github.com/vtraag/leidenalg#installation. Also, consider running Paris "
                              "instead of Leiden clustering using `run_clustering` method")
        import igraph  # python-igraph

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)

        adj = self.load_graph(from_assay, cell_key, feat_key, 'csr', min_edge_weight,
                              symmetric_graph, graph_upper_only)
        sources, targets = adj.nonzero()
        g = igraph.Graph()
        g.add_vertices(adj.shape[0])
        g.add_edges(list(zip(sources, targets)))
        g.es['weight'] = adj[sources, targets].A1
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution,
                                        seed=random_seed)
        self.cells.add(self._col_renamer(from_assay, cell_key, label),
                       np.array(part.membership) + 1, fill_val=-1, key=cell_key, overwrite=True)
        return None

    def run_clustering(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                       n_clusters: int = None, min_edge_weight: float = -1, symmetric_graph: bool = True,
                       graph_upper_only: bool = True, balanced_cut: bool = False,
                       max_size: int = None, min_size: int = None, max_distance_fc: float = 2,
                       force_recalc: bool = False, label: str = 'cluster') -> None:
        """
        Executes Paris clustering algorithm (https://arxiv.org/pdf/1806.01664.pdf) on the cell-neighbourhood graph.
        The algorithm captures the multiscale structure of the graph in to an ordinary dendrogram structure. The
        distances in the dendrogram are are based on probability of sampling node (aka cell) pairs. This methods creates
        this dendrogram if it doesn't already exits for the graph and induces either a straight cut or balanced cut
        to obtain clusters of cells.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            n_clusters: Number of desired clusters (required if balanced_cut is False)
            min_edge_weight: This parameter is forwarded to `load_graph` and is same as there. (Default value: -1)
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            balanced_cut: If True, then uses the balanced cut algorithm as implemented in ``BalancedCut`` to obtain
                          clusters (Default value: False)
            max_size: Same as `max_size` in ``BalancedCut``. The limit for a maximum number of cells in a cluster.
                      This parameter value is required if `balanced_cut` is True.
            min_size: Same as `min_size` in ``BalancedCut``. The limit for a minimum number of cells in a cluster.
                      This parameter value is required if `balanced_cut` is True.
            max_distance_fc:  Same as `max_distance_fc` in ``BalancedCut``. The threshold of ratio of distance between
                              two clusters beyond which they will not be merged. (Default value: 2)
            force_recalc: Forces recalculation of dendrogram even if one already exists for the graph
            label: Base label for cluster identity in the cell metadata column (Default value: 'cluster')

        Returns:

        """
        import sknetwork as skn

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)
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
        if dendrogram_loc in self.z and force_recalc is False:
            dendrogram = self.z[dendrogram_loc][:]
            print("INFO: Using existing dendrogram", flush=True)
        else:
            paris = skn.hierarchy.Paris()
            graph = self.load_graph(from_assay, cell_key, feat_key, 'csr', min_edge_weight,
                                    symmetric_graph, graph_upper_only)
            dendrogram = paris.fit_transform(graph)
            dendrogram[dendrogram == np.Inf] = 0
            g = create_zarr_dataset(self.z[graph_loc], dendrogram_loc.rsplit('/', 1)[1],
                                    (5000,), 'f8', (graph.shape[0] - 1, 4))
            g[:] = dendrogram
        self.z[graph_loc].attrs['latest_dendrogram'] = dendrogram_loc
        if balanced_cut:
            from .dendrogram import BalancedCut
            labels = BalancedCut(dendrogram, max_size, min_size, max_distance_fc).get_clusters()
            print(f"INFO: {len(set(labels))} clusters found", flush=True)
        else:
            labels = skn.hierarchy.cut_straight(dendrogram, n_clusters=n_clusters) + 1
        self.cells.add(self._col_renamer(from_assay, cell_key, label), labels,
                       fill_val=-1, key=cell_key, overwrite=True)

    def run_marker_search(self, *, from_assay: str = None, group_key: str = None, subset_key: str = None,
                          threshold: float = 0.25) -> None:
        """
        Identifies group specific features for a given assay. Please check out the ``find_markers_by_rank`` function
        for further details of how marker features for groups are identified. The results are saved into the Zarr
        hierarchy under `markers` group.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            group_key: Required parameter. This has to be a column name from cell metadata table. This column dictates
                       how the cells will be grouped. Usually this would be a column denoting cell clusters.
            subset_key: To run run the the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table.
            threshold: This value dictates how specific the feature value has to be in a group before it is considered a
                       marker for that group. The value has to be greater than 0 but less than or equal to 1
                       (Default value: 0.25)
        Returns:

        """
        from .markers import find_markers_by_rank

        if group_key is None:
            raise ValueError("ERROR: Please provide a value for `group_key`. This should be the name of a column from "
                             "cell metadata object that has information on how cells should be grouped.")
        if subset_key is None:
            subset_key = 'I'
        assay = self._get_assay(from_assay)
        markers = find_markers_by_rank(assay, group_key, subset_key, threshold)
        z = self.z[assay.name]
        slot_name = f"{subset_key}__{group_key}"
        if 'markers' not in z:
            z.create_group('markers')
        group = z['markers'].create_group(slot_name, overwrite=True)
        for i in markers:
            g = group.create_group(i)
            vals = markers[i]
            if len(vals) != 0:
                create_zarr_obj_array(g, 'names', list(vals.index))
                g_s = create_zarr_dataset(g, 'scores', (10000,), float, vals.values.shape)
                g_s[:] = vals.values
        return None

    def run_mapping(self, *, target_assay: Assay, target_name: str, target_feat_key: str, from_assay: str = None,
                    cell_key: str = 'I', feat_key: str = None, save_k: int = 3, batch_size: int = 1000,
                    ref_mu: bool = True, ref_sigma: bool = True, run_coral: bool = False,
                    exclude_missing: bool = False,  filter_null: bool = False, feat_scaling: bool = True) -> None:
        """
        Projects cells from external assays into the cell-neighbourhood graph using existing PCA loadings and ANN index.
        For each external cell (target) nearest neighbours are identified and save within the Zarr hierarchy group
        `projections`.

        Args:
            target_assay: Assay object of the target dataset
            target_name: Name of target data. This used to keep track of projections in the Zarr hierarchy
            target_feat_key: This will used to name wherein the normalized target data will be saved in its own
                             zarr hierarchy.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            save_k: Number of nearest numbers to identify for each target cell (Default value: 3)
            batch_size: Number of cells that will be projected as a batch. This used to decide the chunk size when
                        normalized data for the target cells is saved to disk.
            ref_mu: If True (default), Then mean values of features as in the reference are used,
                    otherwise mean is calculated using target cells. Turning this to False is not recommended.
            ref_sigma: If True (default), Then standard deviation values of features as present in the reference are
                       used, otherwise std. dev. is calculated using target cells. Turning this to False is not
                       recommended.
            run_coral: If True then CORAL feature rescaling algorithm is used to correct for domain shift in target
                       cells. Read more about CORAL algorithm in function ``coral``. This algorithm creates a m by m
                       matrix where m is the number of features being used for mapping; so it is not advised to use this
                       in a case where a large number of features are being used (>10k for example).
                       (Default value: False)
            exclude_missing: If set to True then only those features that are present in both reference and
                             target are used. If not all reference features from `feat_key` are present in target data
                             then a new graph will be created for reference and mapping will be done onto that graph.
                             (Default value: False)
            filter_null: If True then those features that have a total sum of 0 in the target cells are removed.
                         This has an affect only when `exclude_missing` is True. (Default value: False)
            feat_scaling: If False then features from target cells are not scaled. This is automatically set to False
                          if `run_coral` is True (Default value: True). Setting this to False is not recommended.

        Returns:

        """
        from .mapping_utils import align_features, coral

        source_assay = self._get_assay(from_assay)
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)
        from_assay = source_assay.name
        if target_feat_key == feat_key:
            raise ValueError(f"ERROR: `target_feat_key` cannot be sample as `feat_key`: {feat_key}")
        feat_idx = align_features(source_assay, target_assay, cell_key, feat_key,
                                  target_feat_key, filter_null, exclude_missing)
        print(f"INFO: {len(feat_idx)} features being used for mapping", flush=True)
        if np.all(source_assay.feats.active_index(cell_key + '__' + feat_key) == feat_idx):
            ann_feat_key = feat_key
        else:
            ann_feat_key = f'{feat_key}_common_{target_name}'
            a = np.zeros(source_assay.feats.N).astype(bool)
            a[feat_idx] = True
            source_assay.feats.add(cell_key + '__' + ann_feat_key, a, fill_val=False, overwrite=True)
        if run_coral:
            feat_scaling = False
        ann_obj = self.make_graph(from_assay=from_assay, cell_key=cell_key, feat_key=ann_feat_key,
                                  return_ann_object=True, update_feat_key=False,
                                  feat_scaling=feat_scaling)
        if save_k > ann_obj.k:
            print(f"WARNING: `save_k` was decreased to {ann_obj.k}", flush=True)
            save_k = ann_obj.k
        target_data = daskarr.from_zarr(target_assay.z[f"normed__I__{target_feat_key}/data"])
        if run_coral is True:
            # Reversing coral here to correct target data
            coral(target_data, ann_obj.data, target_assay, target_feat_key)
            target_data = daskarr.from_zarr(target_assay.z[f"normed__I__{target_feat_key}/data_coral"])
        if ann_obj.method == 'pca' and run_coral is False:
            if ref_mu is False:
                mu = calc_computed(target_data.mean(axis=0), 'INFO: Calculating mean of target norm. data')
                ann_obj.mu = clean_array(mu)
            if ref_sigma is False:
                sigma = calc_computed(target_data.std(axis=0), 'INFO: Calculating std. dev. of target norm. data')
                ann_obj.sigma = clean_array(sigma, 1)
        if 'projections' not in source_assay.z:
            source_assay.z.create_group('projections')
        store = source_assay.z['projections'].create_group(target_name, overwrite=True)
        nc, nk = target_assay.cells.table.I.sum(), save_k
        zi = create_zarr_dataset(store, 'indices', (batch_size,), 'u8', (nc, nk))
        zd = create_zarr_dataset(store, 'distances', (batch_size,), 'f8', (nc, nk))
        entry_start = 0
        for i in tqdm(target_data.blocks, desc='Mapping'):
            a: np.ndarray = i.compute()
            ki, kd = ann_obj.transform_ann(ann_obj.reducer(a), k=save_k)
            entry_end = entry_start + len(ki)
            zi[entry_start:entry_end, :] = ki
            zd[entry_start:entry_end, :] = kd
            entry_start = entry_end
        return None

    def get_mapping_score(self, *, target_name: str, target_groups: np.ndarray = None, from_assay: str = None,
                          cell_key: str = 'I', log_transform: bool = True,
                          multiplier: float = 1000) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Mapping scores are an indication of degree of similarity of reference cells in the graph to the target cells.
        The more often a reference cell is found in the nearest neighbour list of the target cells, the higher will be
        the mapping score for that cell.

        Args:
            target_name: Name of target data. This used to keep track of projections in the Zarr hierarchy
            target_groups: Group/cluster identity of target cells. This will then be used to calculate mapping score
                           for each group separately.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            log_transform: If True (default) then the mapping scores will be log transformed
            multiplier: A scaling factor for mapping scores. All scores al multiplied this value. This mostly intended
                        for visualization of mapping scores (Default: 1000)

        Yields:
            A tuple of group name and mapping score of reference cells for that target group.

        """
        if from_assay is None:
            from_assay = self._defaultAssay
        store_loc = f"{from_assay}/projections/{target_name}"
        if store_loc not in self.z:
            raise KeyError(f"ERROR: Projections have not been computed for {target_name} in th latest graph. Please"
                           f" run `run_mapping` or update latest_graph by running `make_graph` with desired parameters")
        store = self.z[store_loc]

        indices = store['indices'][:]
        dists = store['distances'][:]
        # TODO: add more robust options for distance calculation here
        dists = 1 / (np.log1p(dists) + 1)
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
            ms = multiplier * ms / len(coi)
            if log_transform:
                ms = np.log1p(ms)
            yield group, ms

    def load_unified_graph(self, from_assay, cell_key, feat_key, target_name, use_k, target_weight,
                           sparse_format: str = 'coo'):
        """
        This is similar to ``load_graph`` but includes projected cells and their edges.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            target_name: Name of target data. This used to keep track of projections in the Zarr hierarchy
            use_k: Number of nearest neighbour edges of each projected cell to be included. If this value is larger than
                   than `save_k` parameter while running mapping for the `target_name` target then `use_k` is reset to
                   'save_k'
            target_weight: A constant uniform weight to be ascribed to each target-reference edge.
            sparse_format: Format for sparse graph. Can be either 'coo' (default) or 'csr'

        Returns:

        """
        from scipy.sparse import coo_matrix, csr_matrix

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)
        if sparse_format not in ['csr', 'coo']:
            raise KeyError("ERROR: `sparse_format` should be either 'coo' or 'csr'")
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        edges = self.z[graph_loc].edges[:]
        weights = self.z[graph_loc].weights[:]
        n_cells = self.cells.table[cell_key].sum()
        pidx = self.z[from_assay].projections[target_name].indices[:, :use_k]
        ne = []
        nw = []
        for n, i in enumerate(pidx):
            for j in i:
                ne.append([n_cells + n, j])
                # TODO: Better way to weigh the target edges
                nw.append(target_weight)
        me = np.vstack([edges, ne]).astype(int)
        mw = np.hstack([weights, nw])
        tot_cells = n_cells + pidx.shape[0]
        if sparse_format == 'coo':
            return coo_matrix((mw, (me[:, 0], me[:, 1])), shape=(tot_cells, tot_cells))
        elif sparse_format == 'csr':
            return csr_matrix((mw, (me[:, 0], me[:, 1])), shape=(tot_cells, tot_cells))

    def run_unified_umap(self, target_name: str, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                         use_k: int = 3, target_weight: float = 0.5, spread: float = 2.0, min_dist: float = 1,
                         fit_n_epochs: int = 200, tx_n_epochs: int = 100, random_seed: int = 4444,
                         ini_embed_with: str = 'kmeans', label: str = 'UMAP'):
        """

        Args:
            target_name:
            from_assay:
            cell_key:
            feat_key:
            use_k:
            target_weight:
            spread:
            min_dist:
            fit_n_epochs:
            tx_n_epochs:
            random_seed:
            ini_embed_with:
            label:

        Returns:

        """
        from .umap import fit_transform

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)

        graph = self.load_unified_graph(from_assay, cell_key, feat_key, target_name, use_k, target_weight)
        if ini_embed_with == 'kmeans':
            ini_embed = self.get_ini_embed(from_assay, cell_key, feat_key, 2)
        else:
            x = self.cells.fetch(f'{ini_embed_with}1', cell_key)
            y = self.cells.fetch(f'{ini_embed_with}2', cell_key)
            ini_embed = np.array([x, y]).T.astype(np.float32, order="C")
        pidx = self.z[from_assay].projections[target_name].indices[:, 0]
        ini_embed = np.vstack([ini_embed, ini_embed[pidx]])
        t = fit_transform(graph, ini_embed, spread=spread, min_dist=min_dist,
                          tx_n_epochs=tx_n_epochs, fit_n_epochs=fit_n_epochs,
                          random_seed=random_seed, parallel=False)
        g = create_zarr_dataset(self.z[from_assay].projections[target_name], label, (1000, 2), 'float64', t.shape)
        g[:] = t
        label = f"{label}_{target_name}"
        n_ref_cells = self.cells.fetch(cell_key).sum()
        for i in range(2):
            self.cells.add(self._col_renamer(from_assay, cell_key, f'{label}{i + 1}'),
                           t[:n_ref_cells, i], key=cell_key, overwrite=True)
        return None

    def run_unified_tsne(self, sgtsne_loc, target_name: str, from_assay: str = None, cell_key: str = 'I',
                         feat_key: str = None, use_k: int = 3, target_weight: float = 0.5,
                         lambda_scale: float = 1.0, max_iter: int = 500, early_iter: int = 200, alpha: int = 10,
                         box_h: float = 0.7, temp_file_loc: str = '.', verbose: bool = True,
                         ini_embed_with: str = 'kmeans', label: str = 'tSNE'):
        """

        Args:
            sgtsne_loc:
            target_name:
            from_assay:
            cell_key:
            feat_key:
            use_k:
            target_weight:
            lambda_scale:
            max_iter:
            early_iter:
            alpha:
            box_h:
            temp_file_loc:
            verbose:
            ini_embed_with:
            label:

        Returns:

        """
        from uuid import uuid4
        from .knn_utils import export_knn_to_mtx
        from pathlib import Path

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)

        if ini_embed_with == 'kmeans':
            ini_embed = self.get_ini_embed(from_assay, cell_key, feat_key, 2)
        else:
            x = self.cells.fetch(f'{ini_embed_with}1', cell_key)
            y = self.cells.fetch(f'{ini_embed_with}2', cell_key)
            ini_embed = np.array([x, y]).T.astype(np.float32, order="C")
        pidx = self.z[from_assay].projections[target_name].indices[:, 0]
        ini_embed = np.vstack([ini_embed, ini_embed[pidx]])
        uid = str(uuid4())
        ini_emb_fn = Path(temp_file_loc, f'{uid}.txt').resolve()
        with open(ini_emb_fn, 'w') as h:
            h.write('\n'.join(map(str, ini_embed.flatten())))
        del ini_embed

        knn_mtx_fn = Path(temp_file_loc, f'{uid}.mtx').resolve()
        export_knn_to_mtx(knn_mtx_fn, self.load_unified_graph(from_assay, cell_key, feat_key, target_name, use_k,
                                                               target_weight, sparse_format='csr'))

        out_fn = Path(temp_file_loc, f'{uid}_output.txt').resolve()
        cmd = f"{sgtsne_loc} -m {max_iter} -l {lambda_scale} -d {2} -e {early_iter} -p 1 -a {alpha}" \
              f" -h {box_h} -i {ini_emb_fn} -o {out_fn} {knn_mtx_fn}"
        if verbose:
            system_call(cmd)
        else:
            os.system(cmd)

        t = pd.read_csv(out_fn, header=None, sep=' ')[[0, 1]].values
        g = create_zarr_dataset(self.z[from_assay].projections[target_name], label, (1000, 2), 'float64', t.shape)
        g[:] = t
        label = f"{label}_{target_name}"
        n_ref_cells = self.cells.fetch(cell_key).sum()
        for i in range(2):
            self.cells.add(self._col_renamer(from_assay, cell_key, f'{label}{i + 1}'),
                           t[:n_ref_cells, i], key=cell_key, overwrite=True)
        for fn in [out_fn, knn_mtx_fn, ini_emb_fn]:
            Path.unlink(fn)
        return None

    def calc_node_density(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                          min_edge_weight: float = -1, neighbourhood_degree=2, label: str = 'node_density'):
        """

        Args:
            from_assay:
            cell_key:
            feat_key:
            min_edge_weight:
            neighbourhood_degree:
            label:

        Returns:

        """
        from .pcst import calc_neighbourhood_density

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)
        graph = self.load_graph(from_assay, cell_key, feat_key, 'csr', min_edge_weight, False, False)
        density = calc_neighbourhood_density(graph, nn=neighbourhood_degree)
        self.cells.add(self._col_renamer(from_assay, cell_key, label), density,
                       fill_val=0, key=cell_key, overwrite=True)

    def run_subsampling(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                        cluster_key: str = None, density_key: str = None,
                        min_edge_weight: float = -1, seed_frac: float = 0.05,
                        dynamic_seed_frac: bool = True, min_nodes: int = 3, rewards: tuple = (3, 0.1),
                        rand_state: int = 4466, return_vals: bool = False, label: str = 'sketched'):
        """

        Args:
            from_assay:
            cell_key:
            feat_key:
            cluster_key:
            density_key:
            min_edge_weight:
            seed_frac:
            dynamic_seed_frac:
            min_nodes:
            rewards:
            rand_state:
            return_vals:
            label:

        Returns:

        """
        from .pcst import pcst

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)
        if cluster_key is None:
            raise ValueError("ERROR: Please provide a value for cluster key")
        clusters = pd.Series(self.cells.fetch(cluster_key, cell_key))
        graph = self.load_graph(from_assay, cell_key, feat_key, 'csr', min_edge_weight, False, False)
        if len(clusters) != graph.shape[0]:
            raise ValueError(f"ERROR: cluster information exists for {len(clusters)} cells while graph has "
                             f"{graph.shape[0]} cells.")
        if dynamic_seed_frac and density_key is None:
            print("WARNING: `dynamic_seed_frac` will be ignored because node_density has not been calculated.",
                  flush=True)
            dynamic_seed_frac = False
        if dynamic_seed_frac:
            if density_key not in self.cells.table:
                raise ValueError(f"ERROR: {density_key} not found in cell metadata table")
            else:
                cff = self.cells.table[self.cells.table.I].groupby(cluster_key)[density_key].median()
                cff = (cff - cff.min()) / (cff.max() - cff.min())
                cff = 1 - cff
        else:
            n_clusts = clusters.nunique()
            cff = pd.Series(np.zeros(n_clusts), index=list(range(1, n_clusts + 1)))
        steiner_nodes, steiner_edges = pcst(
            graph=graph, clusters=clusters, seed_frac=seed_frac, cluster_factor=cff, min_nodes=min_nodes,
            rewards=rewards, pruning_method='strong', rand_state=rand_state)
        a = np.zeros(self.cells.table[cell_key].values.sum()).astype(bool)
        a[steiner_nodes] = True

        key = self._col_renamer(from_assay, cell_key, label)
        self.cells.add(key, a, fill_val=False, key=cell_key, overwrite=True)
        print(f"INFO: Sketched cells saved with keyname '{key}'")
        if return_vals:
            return steiner_nodes, steiner_edges

    def plot_cells_dists(self, from_assay: str = None, cols: List[str] = None, all_cells: bool = False, **kwargs):
        """

        Args:
            from_assay:
            cols:
            all_cells:
            **kwargs:

        Returns:

        """
        from .plots import plot_qc
        import re

        if from_assay is None:
            from_assay = self._defaultAssay
        plot_cols = [f'{from_assay}_nCounts', f'{from_assay}_nFeatures']
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
        """

        Args:
            from_assay:
            cell_key:
            k:
            clip_fraction:

        Returns:

        """
        cell_idx = self.cells.active_index(cell_key)
        if k not in self.cells.table.columns:
            assay = self._get_assay(from_assay)
            feat_idx = assay.feats.get_idx_by_names([k], True)
            if len(feat_idx) == 0:
                raise ValueError(f"ERROR: {k} not found in {from_assay} assay.")
            else:
                if len(feat_idx) > 1:
                    print(f"WARNING: Plotting mean of {len(feat_idx)} features because {k} is not unique.")
            vals = assay.normed(cell_idx, feat_idx).mean(axis=1).compute().astype(np.float_)
        else:
            vals = self.cells.fetch(k, cell_key)
        if clip_fraction > 0:
            if vals.dtype in [np.float_, np.uint64]:
                min_v = np.percentile(vals, 100 * clip_fraction)
                max_v = np.percentile(vals, 100 - 100 * clip_fraction)
                vals[vals < min_v] = min_v
                vals[vals > max_v] = max_v
        return vals

    def plot_layout(self, *, from_assay: str = None, cell_key: str = 'I',
                    layout_key: str = None, color_by: str = None, subselection_key: str = None,
                    size_vals=None, clip_fraction: float = 0.01,
                    width: float = 6, height: float = 6, default_color: str = 'steelblue',
                    missing_color: str = 'k', colormap=None, point_size: float = 10,
                    ax_label_size: float = 12, frame_offset: float = 0.05, spine_width: float = 0.5,
                    spine_color: str = 'k', displayed_sides: tuple = ('bottom', 'left'),
                    legend_ondata: bool = True, legend_onside: bool = True, legend_size: float = 12,
                    legends_per_col: int = 20, marker_scale: float = 70, lspacing: float = 0.1,
                    cspacing: float = 1, savename: str = None, scatter_kwargs: dict = None):
        """

        Args:
            from_assay:
            cell_key:
            layout_key:
            color_by:
            subselection_key:
            size_vals:
            clip_fraction:
            width:
            height:
            default_color:
            missing_color:
            colormap:
            point_size:
            ax_label_size:
            frame_offset:
            spine_width:
            spine_color:
            displayed_sides:
            legend_ondata:
            legend_onside:
            legend_size:
            legends_per_col:
            marker_scale:
            lspacing:
            cspacing:
            savename:
            scatter_kwargs:

        Returns:

        """
        from .plots import plot_scatter
        if from_assay is None:
            from_assay = self._defaultAssay
        if layout_key is None:
            raise ValueError("Please provide a value for `layout_key` parameter.")
        if clip_fraction >= 0.5:
            raise ValueError("ERROR: clip_fraction cannot be larger than or equal to 0.5")
        x = self.cells.fetch(f'{layout_key}1', cell_key)
        y = self.cells.fetch(f'{layout_key}2', cell_key)
        if color_by is not None:
            v = self.get_cell_vals(from_assay=from_assay, cell_key=cell_key, k=color_by,
                                   clip_fraction=clip_fraction)
        else:
            v = np.ones(len(x))
        df = pd.DataFrame({f'{layout_key} 1': x, f'{layout_key} 2': y, 'vc': v})
        if size_vals is not None:
            if len(size_vals) != len(x):
                raise ValueError("ERROR: `size_vals` is not of same size as layout_key")
            df['s'] = size_vals
        if subselection_key is not None:
            idx = self.cells.fetch(subselection_key, cell_key)
            if idx.dtype != bool:
                print(f"WARNING: `subselection_key` {subselection_key} is not bool type. Will not sub-select",
                      flush=True)
            else:
                df = df[idx]
        return plot_scatter(df, None, None, width, height, default_color, missing_color, colormap, point_size,
                            ax_label_size, frame_offset, spine_width, spine_color, displayed_sides, legend_ondata,
                            legend_onside, legend_size, legends_per_col, marker_scale, lspacing, cspacing, savename,
                            scatter_kwargs)

    def plot_unified_layout(self, *, target_name: str, from_assay: str = None, cell_key: str = 'I',
                            layout_key: str = 'UMAP', show_target_only: bool = False,
                            ref_color: str = 'coral', target_color='k', width: float = 6,
                            height: float = 6, colormap=None, point_size: float = 10,
                            ax_label_size: float = 12, frame_offset: float = 0.05, spine_width: float = 0.5,
                            spine_color: str = 'k', displayed_sides: tuple = ('bottom', 'left'),
                            legend_ondata: bool = True, legend_onside: bool = True, legend_size: float = 12,
                            legends_per_col: int = 20, marker_scale: float = 70, lspacing: float = 0.1,
                            cspacing: float = 1, savename: str = None, scatter_kwargs: dict = None,
                            shuffle_zorder: bool = True):
        """

        Args:
            target_name:
            from_assay:
            cell_key:
            layout_key:
            show_target_only:
            ref_color:
            target_color:
            width:
            height:
            colormap:
            point_size:
            ax_label_size:
            frame_offset:
            spine_width:
            spine_color:
            displayed_sides:
            legend_ondata:
            legend_onside:
            legend_size:
            legends_per_col:
            marker_scale:
            lspacing:
            cspacing:
            savename:
            scatter_kwargs:
            shuffle_zorder:

        Returns:

        """
        from .plots import plot_scatter

        if from_assay is None:
            from_assay = self._defaultAssay
        t = self.z[from_assay].projections[target_name][layout_key][:]
        ref_n_cells = self.cells.table[cell_key].sum()
        t_n_cells = t.shape[0] - ref_n_cells
        x = t[:, 0]
        y = t[:, 1]
        df = pd.DataFrame({f"{layout_key}1": x, f"{layout_key}2": y})
        missing_color = target_color

        if type(ref_color) is not str and type(target_color) is not str:
            raise ValueError('ERROR: Please provide a fixed colour for one of either ref_color or target_color')
        if type(ref_color) is str and type(target_color) is str:
            c = np.hstack([np.ones(ref_n_cells), np.ones(t_n_cells) + 1]).astype(object)
            c[c == 1] = ref_color
            c[c == 2] = target_color
            df['c'] = c
        else:
            if type(ref_color) is not str:
                if len(ref_color) != ref_n_cells:
                    raise ValueError("ERROR: Number of values in `ref_color` should be same as no. of ref cells")
                df['vc'] = np.hstack([ref_color, [np.nan for _ in range(t_n_cells)]])
            else:
                if len(target_color) != t_n_cells:
                    raise ValueError("ERROR: Number of values in `target_color` should be same as no. of target cells")
                df['vc'] = np.hstack([[np.nan for _ in range(ref_n_cells)], target_color])
                missing_color = ref_color
                ref_color = missing_color
        if show_target_only:
            df = df[ref_n_cells:]
        if shuffle_zorder:
            df = df.sample(frac=1)
        return plot_scatter(df, None, None, width, height, ref_color, missing_color, colormap, point_size,
                            ax_label_size, frame_offset, spine_width, spine_color, displayed_sides, legend_ondata,
                            legend_onside, legend_size, legends_per_col, marker_scale, lspacing, cspacing, savename,
                            scatter_kwargs)

    def plot_cluster_tree(self, *, from_assay: str = None, cell_key: str = 'I', feat_key: str = None,
                          cluster_key: str = 'cluster', width: float = 1.5, lvr_factor: float = 0.5,
                          min_node_size: float = 20, node_size_expand_factor: float = 2, cmap='tab20'):
        """

        Args:
            from_assay:
            cell_key:
            feat_key:
            cluster_key:
            width:
            lvr_factor:
            min_node_size:
            node_size_expand_factor:
            cmap:

        Returns:

        """
        from .plots import plot_cluster_hierarchy
        from .dendrogram import SummarizedTree

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self.get_latest_feat_key(from_assay)
        clusts = self.cells.fetch(cluster_key)
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        dendrogram_loc = self.z[graph_loc].attrs['latest_dendrogram']
        sg = SummarizedTree(self.z[dendrogram_loc][:]).extract_ancestor_tree(clusts)
        plot_cluster_hierarchy(sg, clusts, width=width, lvr_factor=lvr_factor, min_node_size=min_node_size,
                               node_size_expand_factor=node_size_expand_factor, cmap=cmap)

    def plot_marker_heatmap(self, *, from_assay: str = None, group_key: str = None, subset_key: str = None,
                            topn: int = 5, log_transform: bool = True, vmin: float = -1, vmax: float = 2,
                            batch_size: int = None, **heatmap_kwargs):
        """

        Args:
            from_assay:
            group_key:
            subset_key:
            topn:
            log_transform:
            vmin:
            vmax:
            batch_size:
            **heatmap_kwargs:

        Returns:

        """
        from .plots import plot_heatmap

        assay = self._get_assay(from_assay)
        if group_key is None:
            raise ValueError("ERROR: Please provide a value for `group_key`")
        if subset_key is None:
            subset_key = 'I'
        if batch_size is None:
            batch_size = min(50, int(1e7 / assay.cells.N)) + 1
        if 'markers' not in self.z[assay.name]:
            raise KeyError("ERROR: Please run `run_marker_search` first")
        slot_name = f"{subset_key}__{group_key}"
        if slot_name not in self.z[assay.name]['markers']:
            raise KeyError(f"ERROR: Please run `run_marker_search` first with {group_key} as `group_key` and "
                           f"{subset_key} as `subset_key`")
        g = self.z[assay.name]['markers'][slot_name]
        goi = []
        for i in g.keys():
            if 'names' in g[i]:
                goi.extend(g[i]['names'][:][:topn])
        goi = sorted(set(goi))
        cell_idx = assay.cells.active_index(subset_key)
        cdf = []
        for i in tqdm(np.array_split(goi, len(goi) // batch_size + 1), desc="INFO: Calculating group mean values"):
            feat_idx = assay.feats.get_idx_by_ids(i)
            normed_data = assay.normed(cell_idx=cell_idx, feat_idx=feat_idx, log_transform=log_transform)
            df = pd.DataFrame(calc_computed(normed_data), columns=i)
            df['cluster'] = assay.cells.fetch(group_key, key=subset_key)
            df = df.groupby('cluster').mean().T
            df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
            cdf.append(df)
        cdf = pd.concat(cdf, axis=0)
        # noinspection PyTypeChecker
        cdf[cdf < vmin] = vmin
        # noinspection PyTypeChecker
        cdf[cdf > vmax] = vmax
        cdf.index = assay.feats.table[['ids', 'names']].set_index('ids').reindex(cdf.index)['names'].values
        plot_heatmap(cdf, **heatmap_kwargs)

    def __repr__(self):
        x = ' '.join(self.assayNames)
        return f"DataStore with {self.cells.N} cells containing {len(self.assayNames)} assays: {x}"
