"""
Contains the primary interface to interact with data (i. e. DataStore) and its superclasses.

- Classes:
    - DataStore: DataStore objects provide the primary interface to interact with the data.
"""

import os
import numpy as np
from typing import List, Iterable, Tuple, Generator, Union, Optional
import pandas as pd
import zarr
import dask.array as daskarr
from scipy.sparse import csr_matrix, coo_matrix
from .writers import create_zarr_dataset, create_zarr_obj_array
from .metadata import MetaData
from .assay import Assay, RNAassay, ATACassay, ADTassay
from .utils import (
    show_dask_progress,
    system_call,
    clean_array,
    controlled_compute,
    logger,
    tqdmbar,
)

__all__ = ["DataStore"]


def sanitize_hierarchy(z: zarr.hierarchy, assay_name: str) -> bool:
    """
    Test if an assay node in zarr object was created properly.

    Args:
        z: Zarr hierarchy object
        assay_name: String value with name of assay.

    Returns:
        True if assay_name is present in z and contains `counts` and `featureData` child nodes else raises error

    """
    if assay_name in z:
        if "counts" not in z[assay_name]:
            raise KeyError(f"ERROR: 'counts' not found in {assay_name}")
        if "featureData" not in z[assay_name]:
            raise KeyError(f"ERROR: 'featureData' not found in {assay_name}")
    else:
        raise KeyError(f"ERROR: {assay_name} not found in zarr file")
    return True


class BaseDataStore:
    """
    This is the base datastore class that deals with loading of assays from Zarr files and generating basic cell
    statistics like nCounts and nFeatures. Superclass of the other DataStores.

    Attributes:
        cells: MetaData object with cells and info about each cell (e. g. RNA_nCounts ids).
        assayNames: List of assay names in Zarr file, e. g. 'RNA' or 'ATAC'.
        nthreads: Number of threads to use for this datastore instance.
        z: The Zarr file (directory) used for for this datastore instance.
    """

    def __init__(
        self,
        zarr_loc: str,
        assay_types: dict,
        default_assay: str,
        min_features_per_cell: int,
        min_cells_per_feature: int,
        mito_pattern: str,
        ribo_pattern: str,
        nthreads: int,
        zarr_mode: str,
        synchronizer,
    ):
        """
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
            mito_pattern: Regex pattern to capture mitochondrial genes (default: 'MT-')
            ribo_pattern: Regex pattern to capture ribosomal genes (default: 'RPS|RPL|MRPS|MRPL')
            nthreads: Number of maximum threads to use in all multi-threaded functions
            zarr_mode: For read-write mode use r+' or for read-only use 'r' (Default value: 'r+')
            synchronizer: Used as `synchronizer` parameter when opening the Zarr file. Please refer to this page for
                          more details: https://zarr.readthedocs.io/en/stable/api/sync.html. By default
                          ThreadSynchronizer will be used.
        """

        self._fn: str = zarr_loc
        if type(self._fn) != str:
            self.z: zarr.hierarchy = zarr.group(self._fn, synchronizer=synchronizer)
        else:
            self.z: zarr.hierarchy = zarr.open(
                self._fn, mode=zarr_mode, synchronizer=synchronizer
            )
        self.nthreads = nthreads
        # The order is critical here:
        self.cells = self._load_cells()
        self.assayNames = self._get_assay_names()
        self._defaultAssay = self._load_default_assay(default_assay)
        self._load_assays(min_cells_per_feature, assay_types)
        # TODO: Reset all attrs, pca, dendrogram etc
        self._ini_cell_props(min_features_per_cell, mito_pattern, ribo_pattern)
        self._cachedMagicOperator = None
        self._cachedMagicOperatorLoc = None
        self._integratedGraphsLoc = "integratedGraphs"
        # TODO: Implement _caches to hold are cached data
        # TODO: Implement _defaults to hold default parameters for methods

    def _load_cells(self) -> MetaData:
        """
        This convenience function loads cellData level from the Zarr hierarchy.

        Returns:
            Metadata object

        """
        if "cellData" not in self.z:
            raise KeyError("ERROR: cellData not found in zarr file")
        return MetaData(self.z["cellData"])

    def _get_assay_names(self) -> List[str]:
        """
        Load all assay names present in the Zarr file. Zarr writers create an 'is_assay' attribute in the assay level
        and this function looks for presence of those attributes to load assay names.

        Returns:
            Names of assays present in a Zarr file

        """
        assays = []
        for i in self.z.group_keys():
            if "is_assay" in self.z[i].attrs.keys():
                sanitize_hierarchy(self.z, i)
                assays.append(i)
        return assays

    def _load_default_assay(self, assay_name: str = None) -> str:
        """
        This function sets a given assay name as defaultAssay attribute. If `assay_name` value is None then the
        top-level directory attributes in the Zarr file are looked up for presence of previously used default assay.

        Args:
            assay_name: Name of the assay to be considered for setting as default.

        Returns:
            Name of the assay to be set as default assay

        """
        if assay_name is None:
            if "defaultAssay" in self.z.attrs:
                assay_name = self.z.attrs["defaultAssay"]
            else:
                if len(self.assayNames) == 1:
                    assay_name = self.assayNames[0]
                    self.z.attrs["defaultAssay"] = assay_name
                else:
                    raise ValueError(
                        "ERROR: You have more than one assay data. "
                        f"Choose one from: {' '.join(self.assayNames)}\n using 'default_assay' parameter. "
                        "Please note that names are case-sensitive."
                    )
        else:
            if assay_name in self.assayNames:
                if "defaultAssay" in self.z.attrs:
                    if assay_name != self.z.attrs["defaultAssay"]:
                        logger.info(
                            f"Default assay changed from {self.z.attrs['defaultAssay']} to {assay_name}"
                        )
                self.z.attrs["defaultAssay"] = assay_name
            else:
                raise ValueError(
                    f"ERROR: The provided default assay name: {assay_name} was not found. "
                    f"Please Choose one from: {' '.join(self.assayNames)}\n"
                    "Please note that the names are case-sensitive."
                )
        return assay_name

    def _load_assays(self, min_cells: int, custom_assay_types: dict = None) -> None:
        """
        This function loads all the assay names present in attribute `assayNames` as Assay objects. An attempt is made
        to automatically determine the most appropriate Assay class for each assay based on following mapping:

        literal_blocks::
            {'RNA': RNAassay, 'ATAC': ATACassay, 'ADT': ADTassay, 'GeneActivity': RNAassay, 'URNA': RNAassay}

        If an assay name does not match any of the keys above then it is assigned as generic assay class. This can be
        overridden using `predefined_assays` parameter

        Args:
            min_cells: Minimum number of cells that a feature in each assay must be present to not be discarded (i.e.
                       receive False value in `I` column)
            custom_assay_types: A mapping of assay names to Assay class type to associated with.

        Returns:
        """

        preset_assay_types = {
            "RNA": RNAassay,
            "ATAC": ATACassay,
            "ADT": ADTassay,
            "GeneActivity": RNAassay,
            "URNA": RNAassay,
            "Assay": Assay,
        }
        caution_statement = (
            "%s was set as a generic Assay with no normalization. If this is unintended "
            "then please make sure that you provide a correct assay type for this assay using "
            "'assay_types' parameter."
        )
        caution_statement = (
            caution_statement
            + "\nIf you have more than one assay in the dataset then you can set "
            "assay_types={'assay1': 'RNA', 'assay2': 'ADT'} "
            "Just replace with actual assay names instead of assay1 and assay2"
        )
        if "assayTypes" not in self.z.attrs:
            self.z.attrs["assayTypes"] = {}
        z_attrs = dict(self.z.attrs["assayTypes"])
        if custom_assay_types is None:
            custom_assay_types = {}
        for i in self.assayNames:
            if i in custom_assay_types:
                if custom_assay_types[i] in preset_assay_types:
                    assay = preset_assay_types[custom_assay_types[i]]
                    assay_name = custom_assay_types[i]
                else:
                    logger.warning(
                        f"{custom_assay_types[i]} is not a recognized assay type. Has to be one of "
                        f"{', '.join(list(preset_assay_types.keys()))}\nPLease note that the names are"
                        f" case-sensitive."
                    )
                    logger.warning(caution_statement % i)
                    assay = Assay
                    assay_name = "Assay"
                if i in z_attrs and assay_name == z_attrs[i]:
                    pass
                else:
                    z_attrs[i] = assay_name
                    logger.debug(f"Setting assay {i} to assay type: {assay.__name__}")
            elif i in z_attrs:
                assay = preset_assay_types[z_attrs[i]]
            else:
                if i in preset_assay_types:
                    assay = preset_assay_types[i]
                    assay_name = i
                else:
                    logger.warning(caution_statement % i)
                    assay = Assay
                    assay_name = "Assay"
                if i in z_attrs and assay_name == z_attrs[i]:
                    pass
                else:
                    z_attrs[i] = assay_name
                    logger.debug(f"Setting assay {i} to assay type: {assay.__name__}")
            setattr(
                self,
                i,
                assay(
                    self.z,
                    i,
                    self.cells,
                    min_cells_per_feature=min_cells,
                    nthreads=self.nthreads,
                ),
            )
        if self.z.attrs["assayTypes"] != z_attrs:
            self.z.attrs["assayTypes"] = z_attrs
        return None

    def _get_assay(
        self, from_assay: str
    ) -> Union[Assay, RNAassay, ADTassay, ATACassay]:
        """
        This is a convenience function used internally to quickly obtain the assay object that is linked to a assay name.

        Args:
            from_assay: Name of the assay whose object is to be returned.

        Returns:

        """
        if from_assay is None or from_assay == "":
            from_assay = self._defaultAssay
        return self.__getattribute__(from_assay)

    def _get_latest_feat_key(self, from_assay: str) -> str:
        """
        Looks up the the value in assay level attributes for key 'latest_feat_key'.

        Args:
            from_assay: Assay whose latest feature is to be returned.

        Returns:
            Name of the latest feature that was used to run `save_normalized_data`

        """
        assay = self._get_assay(from_assay)
        return assay.attrs["latest_feat_key"]

    def _get_latest_cell_key(self, from_assay: str) -> str:
        """
        Looks up the the value in assay level attributes for key 'latest_cell_key'.

        Args:
            from_assay: Assay whose latest feature is to be returned.

        Returns:
            Name of the latest feature that was used to run `save_normalized_data`

        """
        assay = self._get_assay(from_assay)
        return assay.attrs["latest_cell_key"]

    def _ini_cell_props(
        self, min_features: int, mito_pattern: str, ribo_pattern: str
    ) -> None:
        """
        This function is called on class initialization. For each assay, it calculates per-cell statistics i.e. nCounts,
        nFeatures, percentMito and percentRibo. These statistics are then populated into the cell metadata table.

        Args:
            min_features: Minimum features that a cell must have non-zero value before being filtered out.
            mito_pattern: Regex pattern for identification of mitochondrial genes.
            ribo_pattern: Regex pattern for identification of ribosomal genes.

        Returns:

        """
        for from_assay in self.assayNames:
            assay = self._get_assay(from_assay)

            var_name = from_assay + "_nCounts"
            if var_name not in self.cells.columns:
                n_c = show_dask_progress(
                    assay.rawData.sum(axis=1),
                    f"({from_assay}) Computing nCounts",
                    self.nthreads,
                )
                self.cells.insert(var_name, n_c.astype(np.float_), overwrite=True)
                if type(assay) == RNAassay:
                    min_nc = min(n_c)
                    if min(n_c) < assay.sf:
                        logger.warning(
                            f"Minimum cell count ({min_nc}) is lower than "
                            f"size factor multiplier ({assay.sf})"
                        )
            var_name = from_assay + "_nFeatures"
            if var_name not in self.cells.columns:
                n_f = show_dask_progress(
                    (assay.rawData > 0).sum(axis=1),
                    f"({from_assay}) Computing nFeatures",
                    self.nthreads,
                )
                self.cells.insert(var_name, n_f.astype(np.float_), overwrite=True)

            if type(assay) == RNAassay:
                if mito_pattern is None:
                    mito_pattern = "MT-|mt"
                var_name = from_assay + "_percentMito"
                assay.add_percent_feature(mito_pattern, var_name)

                if ribo_pattern is None:
                    ribo_pattern = "RPS|RPL|MRPS|MRPL"
                var_name = from_assay + "_percentRibo"
                assay.add_percent_feature(ribo_pattern, var_name)

            if from_assay == self._defaultAssay:
                v = self.cells.fetch(from_assay + "_nFeatures")
                if min_features > np.median(v):
                    logger.warning(
                        f"More than of half of the less have less than {min_features} features for assay: "
                        f"{from_assay}. Will not remove low quality cells automatically."
                    )
                else:
                    bv = self.cells.sift(
                        from_assay + "_nFeatures", min_features, np.Inf
                    )
                    # Making sure that the write operation is only done if the filtering results have changed
                    cur_I = self.cells.fetch_all("I")
                    nbv = bv & cur_I
                    if all(nbv == cur_I) is False:
                        self.cells.update_key(bv, key="I")

    @staticmethod
    def _col_renamer(from_assay: str, cell_key: str, suffix: str) -> str:
        """
        A convenience function for internal usage that creates naming rule for the metadata columns.

        Args:
            from_assay: Name of the assay.
            cell_key: Cell key to use.
            suffix: Base name for the column.

        Returns:
            column name updated as per the naming rule

        """
        if cell_key == "I":
            ret_val = "_".join(list(map(str, [from_assay, suffix])))
        else:
            ret_val = "_".join(list(map(str, [from_assay, cell_key, suffix])))
        return ret_val

    def set_default_assay(self, assay_name: str) -> None:
        """
        Override assigning of default assay.

        Args:
            assay_name: Name of the assay that should be set as default.

        Returns:

        Raises:
            ValueError: if `assay_name` is not found in attribute `assayNames`

        """
        if assay_name in self.assayNames:
            self._defaultAssay = assay_name
            self.z.attrs["defaultAssay"] = assay_name
        else:
            raise ValueError(f"ERROR: {assay_name} assay was not found.")

    def get_cell_vals(
        self, *, from_assay: str, cell_key: str, k: str, clip_fraction: float = 0
    ):
        """
        Fetches data from the Zarr file.

        This convenience function allows fetching values for cells from either cell metadata table or values of a
        given feature from normalized matrix.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: One of the columns from cell metadata table that indicates the cells to be used. The values in
                      the chosen column should be boolean (Default value: 'I')
            k: A cell metadata column or name of a feature.
            clip_fraction: This value is multiplied by 100 and the percentiles are soft-clipped from either end.
                           (Default value: 0)

        Returns:
            The requested values
        """
        cell_idx = self.cells.active_index(cell_key)
        if k not in self.cells.columns:
            assay = self._get_assay(from_assay)
            feat_idx = assay.feats.get_index_by([k], "names")
            if len(feat_idx) == 0:
                raise ValueError(f"ERROR: {k} not found in {from_assay} assay.")
            else:
                if len(feat_idx) > 1:
                    logger.warning(
                        f"Plotting mean of {len(feat_idx)} features because {k} is not unique."
                    )
            vals = controlled_compute(
                assay.normed(cell_idx, feat_idx).mean(axis=1), self.nthreads
            ).astype(np.float_)
        else:
            vals = self.cells.fetch(k, cell_key)
        if clip_fraction < 0 or clip_fraction > 1:
            raise ValueError(
                "ERROR: Value for `clip_fraction` parameter should be between 0 and 1"
            )
        if clip_fraction > 0:
            if vals.dtype in [np.float_, np.uint64]:
                min_v = np.percentile(vals, 100 * clip_fraction)
                max_v = np.percentile(vals, 100 - 100 * clip_fraction)
                vals[vals < min_v] = min_v
                vals[vals > max_v] = max_v
        return vals

    def __repr__(self):
        res = (
            f"DataStore has {self.cells.active_index('I').shape[0]} ({self.cells.N}) cells with"
            f" {len(self.assayNames)} assays: {' '.join(self.assayNames)}"
        )
        htabs = " " * 3
        stabs = htabs * 2
        dtabs = stabs * 2
        res = res + f"\n{htabs}Cell metadata:"
        res += (
            "\n"
            + dtabs
            + "".join(
                [
                    f"'{x}', " if n % 5 != 0 else f"'{x}', \n{dtabs}"
                    for n, x in enumerate(self.cells.columns, start=1)
                ]
            )
        )
        res = res.rstrip("\n\t")[:-2]
        for i in self.assayNames:
            assay = self._get_assay(i)
            res += (
                f"\n{htabs}{i} assay has {assay.feats.fetch_all('I').sum()} ({assay.feats.N}) "
                f"features and following metadata:"
            )
            res += (
                "\n"
                + dtabs
                + "".join(
                    [
                        f"'{x}', " if n % 5 != 0 else f"'{x}', \n{dtabs}"
                        for n, x in enumerate(assay.feats.columns, start=1)
                    ]
                )
            )
            res = res.rstrip("\n\t")[:-2]
            if "projections" in self.z[i]:
                targets = []
                layouts = []
                for j in self.z[i]["projections"]:
                    if type(self.z[i]["projections"][j]) == zarr.Group:
                        targets.append(j)
                    else:
                        layouts.append(j)
                if len(targets) > 0:
                    res += f"\n{stabs}Projected samples:"
                    res += (
                        "\n"
                        + dtabs
                        + "".join(
                            [
                                f"'{x}', " if n % 5 != 0 else f"'{x}', \n{dtabs}"
                                for n, x in enumerate(targets, start=1)
                            ]
                        )
                    )
                    res = res.rstrip("\n\t")[:-2]
                if len(layouts) > 0:
                    res += f"\n{stabs}Co-embeddings:"
                    res += (
                        "\n"
                        + dtabs
                        + "".join(
                            [
                                f"'{x}', " if n % 5 != 0 else f"'{x}', \n{dtabs}"
                                for n, x in enumerate(layouts, start=1)
                            ]
                        )
                    )
                    res = res.rstrip("\n\t")[:-2]
        return res


# Note for the docstring: Attributes are copied from BaseDataStore docstring since the constructor is inherited.
# Meaning, for any attribute change in BaseDataStore a manual update to docstring here is needed as well. - RO
class GraphDataStore(BaseDataStore):
    """
    This class extends BaseDataStore by providing methods required to generate a cell-cell neighbourhood graph.

    It also contains all the methods that use the KNN graphs as primary input like UMAP/tSNE embedding calculation,
    clustering, down-sampling etc.

    Attributes:
        cells: List of cell barcodes.
        assayNames: List of assay names in Zarr file, e. g. 'RNA' or 'ATAC'.
        nthreads: Number of threads to use for this datastore instance.
        z: The Zarr file (directory) used for for this datastore instance.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _choose_reduction_method(assay: Assay, reduction_method: str) -> str:
        """
        This is a convenience function to determine the linear dimension reduction method to be used for a given assay.
        It is uses a predetermine rule to make this determination.

        Args:
            assay: Assay object.
            reduction_method: Name of reduction method to use. It can be one from either: 'pca', 'lsi', 'auto'.

        Returns:
            The name of dimension reduction method to be used. Either 'pca' or 'lsi'

        Raises:
            ValueError: If `reduction_method` is not one of either 'pca', 'lsi', 'auto'

        """
        reduction_method = reduction_method.lower()
        if reduction_method not in ["pca", "lsi", "auto", "custom"]:
            raise ValueError(
                "ERROR: Please choose either 'pca' or 'lsi' as reduction method"
            )
        if reduction_method == "auto":
            assay_type = str(assay.__class__).split(".")[-1][:-2]
            if assay_type == "ATACassay":
                logger.debug("Using LSI for dimension reduction")
                reduction_method = "lsi"
            else:
                logger.debug("Using PCA for dimension reduction")
                reduction_method = "pca"
        return reduction_method

    def _set_graph_params(
        self,
        from_assay,
        cell_key,
        feat_key,
        log_transform=None,
        renormalize_subset=None,
        reduction_method="auto",
        dims=None,
        pca_cell_key=None,
        ann_metric=None,
        ann_efc=None,
        ann_ef=None,
        ann_m=None,
        rand_state=None,
        k=None,
        n_centroids=None,
        local_connectivity=None,
        bandwidth=None,
    ) -> tuple:
        """
        This function allows determination of values for the parameters of `make_graph` function. This function harbours
        the default values for each parameter.  If parameter value is None, then before choosing the default, it tries
        to use the values from latest iteration of the step within the same hierarchy tree.
        Find details for parameters in the `make_graph` method

        Args:
            from_assay: Same as from_assay in make_graph
            cell_key: Same as cell_key in make_graph
            feat_key: Same as feat_key in make_graph
            log_transform: Same as log_transform in make_graph
            renormalize_subset: Same as renormalize_subset in make_graph
            reduction_method: Same as reduction_method in make_graph
            dims: Same as dims in make_graph
            pca_cell_key: Same as pca_cell_key in make_graph
            ann_metric: Same as ann_metric in make_graph
            ann_efc: Same as ann_efc in make_graph
            ann_ef: Same as ann_ef in make_graph
            ann_m: Same as ann_m in make_graph
            rand_state: Same as rand_state in make_graph
            k: Same as k in make_graph
            n_centroids: Same as n_centroids in make_graph
            local_connectivity: Same as local_connectivity in make_graph
            bandwidth: Same as bandwidth in make_graph

        Returns:
            Finalized values for the all the optional parameters in the same order

        """

        def log_message(category, name, value, custom_msg=None):
            """
            Convenience function to log variable usage messages for make_graph

            Args:
                category:
                name:
                value:
                custom_msg:

            Returns:

            """
            msg = f"No value provided for parameter `{name}`. "
            if category == "default":
                msg += f"Will use default value: {value}"
                logger.debug(msg)
            elif category == "cached":
                msg += f"Will use previously used value: {value}"
                logger.debug(msg)
            else:
                if custom_msg is None:
                    return False
                else:
                    logger.info(custom_msg)
            return True

        default_values = {
            "log_transform": True,
            "renormalize_subset": True,
            "dims": 11,
            "ann_metric": "l2",
            "rand_state": 4466,
            "k": 11,
            "n_centroids": 1000,
            "local_connectivity": 1.0,
            "bandwidth": 1.5,
        }

        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        if log_transform is None or renormalize_subset is None:
            if normed_loc in self.z and "subset_params" in self.z[normed_loc].attrs:
                # This works in coordination with save_normalized_data
                subset_params = self.z[normed_loc].attrs["subset_params"]
                c_log_transform, c_renormalize_subset = (
                    subset_params["log_transform"],
                    subset_params["renormalize_subset"],
                )
            else:
                c_log_transform, c_renormalize_subset = None, None
            if log_transform is None:
                if c_log_transform is not None:
                    log_transform = bool(c_log_transform)
                    log_message("cached", "log_transform", log_transform)
                else:
                    log_transform = default_values["log_transform"]
                    log_message("default", "log_transform", log_transform)
            if renormalize_subset is None:
                if c_renormalize_subset is not None:
                    renormalize_subset = bool(c_renormalize_subset)
                    log_message("cached", "renormalize_subset", renormalize_subset)
                else:
                    renormalize_subset = default_values["renormalize_subset"]
                    log_message("default", "renormalize_subset", renormalize_subset)
        log_transform = bool(log_transform)
        renormalize_subset = bool(renormalize_subset)

        if dims is None or pca_cell_key is None:
            if normed_loc in self.z and "latest_reduction" in self.z[normed_loc].attrs:
                reduction_loc = self.z[normed_loc].attrs["latest_reduction"]
                c_dims, c_pca_cell_key = reduction_loc.rsplit("__", 2)[1:]
            else:
                c_dims, c_pca_cell_key = None, None
            if dims is None:
                if c_dims is not None:
                    dims = int(c_dims)
                    log_message("cached", "dims", dims)
                else:
                    dims = default_values["dims"]
                    log_message("default", "dims", dims)
            if pca_cell_key is None:
                if c_pca_cell_key is not None:
                    pca_cell_key = c_pca_cell_key
                    log_message("cached", "pca_cell_key", pca_cell_key)
                else:
                    pca_cell_key = cell_key
                    log_message("default", "pca_cell_key", pca_cell_key)
            else:
                if pca_cell_key not in self.cells.columns:
                    raise ValueError(
                        f"ERROR: `pca_use_cell_key` {pca_cell_key} does not exist in cell metadata"
                    )
                if self.cells.get_dtype(pca_cell_key) != bool:
                    raise TypeError(
                        "ERROR: Type of `pca_use_cell_key` column in cell metadata should be `bool`"
                    )
        dims = int(dims)
        reduction_method = self._choose_reduction_method(
            self._get_assay(from_assay), reduction_method
        )
        reduction_loc = (
            f"{normed_loc}/reduction__{reduction_method}__{dims}__{pca_cell_key}"
        )

        if (
            ann_metric is None
            or ann_efc is None
            or ann_ef is None
            or ann_m is None
            or rand_state is None
        ):
            if reduction_loc in self.z and "latest_ann" in self.z[reduction_loc].attrs:
                ann_loc = self.z[reduction_loc].attrs["latest_ann"]
                (
                    c_ann_metric,
                    c_ann_efc,
                    c_ann_ef,
                    c_ann_m,
                    c_rand_state,
                ) = ann_loc.rsplit("/", 1)[1].split("__")[1:]
            else:
                c_ann_metric, c_ann_efc, c_ann_ef, c_ann_m, c_rand_state = (
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            if ann_metric is None:
                if c_ann_metric is not None:
                    ann_metric = c_ann_metric
                    log_message("cached", "ann_metric", ann_metric)
                else:
                    ann_metric = default_values["ann_metric"]
                    log_message("default", "ann_metric", ann_metric)
            if ann_efc is None:
                if c_ann_efc is not None:
                    ann_efc = int(c_ann_efc)
                    log_message("cached", "ann_efc", ann_efc)
                else:
                    ann_efc = None  # Will be set after value for k is determined
                    log_message("default", "ann_efc", f"min(100, max(k * 3, 50))")
            if ann_ef is None:
                if c_ann_ef is not None:
                    ann_ef = int(c_ann_ef)
                    log_message("cached", "ann_ef", ann_ef)
                else:
                    ann_ef = None  # Will be set after value for k is determined
                    log_message("default", "ann_ef", f"min(100, max(k * 3, 50))")
            if ann_m is None:
                if c_ann_m is not None:
                    ann_m = int(c_ann_m)
                    log_message("cached", "ann_m", ann_m)
                else:
                    ann_m = min(max(48, int(dims * 1.5)), 64)
                    log_message("default", "ann_m", ann_m)
            if rand_state is None:
                if c_rand_state is not None:
                    rand_state = int(c_rand_state)
                    log_message("cached", "rand_state", rand_state)
                else:
                    rand_state = default_values["rand_state"]
                    log_message("default", "rand_state", rand_state)
        ann_metric = str(ann_metric)
        ann_m = int(ann_m)
        rand_state = int(rand_state)

        if k is None:
            if reduction_loc in self.z and "latest_ann" in self.z[reduction_loc].attrs:
                ann_loc = self.z[reduction_loc].attrs["latest_ann"]
                knn_loc = self.z[ann_loc].attrs["latest_knn"]
                k = int(knn_loc.rsplit("__", 1)[1])
                log_message("cached", "k", k)
            else:
                k = default_values["k"]
                log_message("default", "k", k)
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
            if (
                reduction_loc in self.z
                and "latest_kmeans" in self.z[reduction_loc].attrs
            ):
                kmeans_loc = self.z[reduction_loc].attrs["latest_kmeans"]
                n_centroids = int(
                    kmeans_loc.split("/")[-1].split("__")[1]
                )  # depends on param_joiner
                log_message("default", "n_centroids", n_centroids)
            else:
                # n_centroids = min(data.shape[0]/10, max(500, data.shape[0]/100))
                n_centroids = default_values["n_centroids"]
                log_message("default", "n_centroids", n_centroids)
        n_centroids = int(n_centroids)

        if local_connectivity is None or bandwidth is None:
            if knn_loc in self.z and "latest_graph" in self.z[knn_loc].attrs:
                graph_loc = self.z[knn_loc].attrs["latest_graph"]
                c_local_connectivity, c_bandwidth = map(
                    float, graph_loc.rsplit("/")[-1].split("__")[1:]
                )
            else:
                c_local_connectivity, c_bandwidth = None, None
            if local_connectivity is None:
                if c_local_connectivity is not None:
                    local_connectivity = c_local_connectivity
                    log_message("cached", "local_connectivity", local_connectivity)
                else:
                    local_connectivity = default_values["local_connectivity"]
                    log_message("default", "local_connectivity", local_connectivity)
            if bandwidth is None:
                if c_bandwidth is not None:
                    bandwidth = c_bandwidth
                    log_message("cached", "bandwidth", bandwidth)
                else:
                    bandwidth = default_values["bandwidth"]
                    log_message("default", "bandwidth", bandwidth)
        local_connectivity = float(local_connectivity)
        bandwidth = float(bandwidth)

        return (
            log_transform,
            renormalize_subset,
            reduction_method,
            dims,
            pca_cell_key,
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            rand_state,
            k,
            n_centroids,
            local_connectivity,
            bandwidth,
        )

    def _get_latest_keys(
        self, from_assay: str, cell_key: str, feat_key: str
    ) -> Tuple[str, str, str]:
        if from_assay is None:
            from_assay = self._defaultAssay
        if cell_key is None:
            cell_key = self._get_latest_cell_key(from_assay)
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        return from_assay, cell_key, feat_key

    def _get_latest_graph_loc(
        self, from_assay: str, cell_key: str, feat_key: str
    ) -> str:
        """
        Convenience function to identify location of latest graph in the Zarr hierarchy.

        Args:
            from_assay: Name of the assay.
            cell_key: Cell key used to create the graph.
            feat_key: Feature key used to create the graph.

        Returns:
            Path of graph in the Zarr hierarchy

        """
        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = self.z[normed_loc].attrs["latest_reduction"]
        ann_loc = self.z[reduction_loc].attrs["latest_ann"]
        knn_loc = self.z[ann_loc].attrs["latest_knn"]
        return self.z[knn_loc].attrs["latest_graph"]

    def _get_ini_embed(
        self, from_assay: str, cell_key: str, feat_key: str, n_comps: int
    ) -> np.ndarray:
        """
        Runs PCA on kmeans cluster centers and ascribes the PC values to individual cells based on their cluster
        labels. This is used in `run_umap` and `run_tsne` for initial embedding of cells. Uses `rescale_array` to
        to reduce the magnitude of extreme values.

        Args:
            from_assay: Name fo the assay for which Kmeans was fit.
            cell_key: Cell key used.
            feat_key: Feature key used.
            n_comps: Number of PC components to use

        Returns:
            Matrix with n_comps dimensions representing initial embedding of cells.

        """
        from sklearn.decomposition import PCA
        from .utils import rescale_array

        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = self.z[normed_loc].attrs["latest_reduction"]
        kmeans_loc = self.z[reduction_loc].attrs["latest_kmeans"]
        pc = PCA(n_components=n_comps).fit_transform(
            self.z[kmeans_loc]["cluster_centers"][:]
        )
        for i in range(n_comps):
            pc[:, i] = rescale_array(pc[:, i])
        clusters = self.z[kmeans_loc]["cluster_labels"][:].astype(np.uint32)
        return np.array([pc[x] for x in clusters]).astype(np.float32, order="C")

    def _get_graph_ncells_k(self, graph_loc: str) -> Tuple[int, int]:
        """

        Args:
            graph_loc:

        Returns:

        """
        if graph_loc.startswith(self._integratedGraphsLoc):
            attrs = self.z[graph_loc].attrs
            return attrs["n_cells"], attrs["n_neighbors"]
        knn_loc = self.z[graph_loc.rsplit("/", 1)[0]]
        return knn_loc["indices"].shape

    def _store_to_sparse(
        self, graph_loc: str, sparse_format: str = "csr", use_k: int = None
    ) -> tuple:
        """

        Args:
            graph_loc:
            sparse_format:
            use_k:

        Returns:

        """
        logger.debug(f"Loading graph from location: {graph_loc}")
        store = self.z[graph_loc]
        n_cells, k = self._get_graph_ncells_k(graph_loc)
        # TODO: can we have a progress bar for graph loading. Append to coo matrix?
        if use_k is None:
            use_k = k
        if use_k > k:
            use_k = k
        if use_k < 1:
            use_k = 1
        if use_k != k:
            indexer = np.tile([True] * use_k + [False] * (k - use_k), n_cells)
        else:
            indexer = None
        w, e = store["weights"][:], store["edges"][:]
        if indexer is not None:
            w, e = w[indexer], e[indexer]
        if sparse_format == "csr":
            return n_cells, csr_matrix(
                (w, (e[:, 0], e[:, 1])), shape=(n_cells, n_cells)
            )
        else:
            return n_cells, coo_matrix(
                (w, (e[:, 0], e[:, 1])), shape=(n_cells, n_cells)
            )

    def make_graph(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        feat_key: str = None,
        pca_cell_key: str = None,
        reduction_method: str = "auto",
        dims: int = None,
        k: int = None,
        ann_metric: str = None,
        ann_efc: int = None,
        ann_ef: int = None,
        ann_m: int = None,
        ann_parallel: bool = False,
        rand_state: int = None,
        n_centroids: int = None,
        batch_size: int = None,
        log_transform: bool = None,
        renormalize_subset: bool = None,
        local_connectivity: float = None,
        bandwidth: float = None,
        update_keys: bool = True,
        return_ann_object: bool = False,
        custom_loadings: np.array = None,
        feat_scaling: bool = True,
        show_elbow_plot: bool = False,
    ):
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
            ann_parallel: If True, then ANN graph is created in parallel mode using DataStore.nthreads number of
                          threads. Results obtained in parallel mode will not be reproducible. (Defaul: False)
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
            update_keys: If True (default) then `latest_feat_key` zarr attribute of the assay will be updated.
                         Choose False if you are experimenting with a `feat_key` do not want to override existing
                         `latest_feat_key` and by extension `latest_graph`.
            return_ann_object: If True then returns the ANNStream object. This allows one to directly interact with the
                               PCA transformer and HNSWlib index. Check out ANNStream documentation to know more.
                               (Default: False)
            custom_loadings: Custom loadings/transformer for linear dimension reduction. If provided, should have a form
                             (d x p) where d is same the number of active features in feat_key and p is the number of
                             reduced dimensions. `dims` parameter is ignored when this is provided.
                             (Default value: None)
            feat_scaling: If True (default) then the feature will be z-scaled otherwise not. It is highly recommended to
                          keep this as True unless you know what you are doing. `feat_scaling` is internally turned off
                          when during cross sample mapping using CORAL normalized values are being used. Read more about
                          this in `run_mapping` method.
            show_elbow_plot: If True, then an elbow plot is shown when PCA is fitted to the data. Not shown when using
                            existing PCA loadings or custom loadings. (Default value: False)

        Returns:
            Either None or `AnnStream` object
        """
        from .ann import AnnStream

        if from_assay is None:
            from_assay = self._defaultAssay
        assay = self._get_assay(from_assay)
        if batch_size is None:
            batch_size = assay.rawData.chunksize[0]
        if cell_key is None:
            cell_key = "I"
        if feat_key is None:
            bool_cols = [
                x.split("__", 1)
                for x in assay.feats.columns
                if assay.feats.get_dtype(x) == bool and x != "I"
            ]
            bool_cols = [f"{x[1]}({x[0]})" for x in bool_cols]
            bool_cols = " ".join(map(str, bool_cols))
            raise ValueError(
                "ERROR: You have to choose which features that should be used for graph construction. "
                "Ideally you should have performed a feature selection step before making this graph. "
                "Feature selection step adds a column to your feature table. \n"
                "You have following boolean columns in the feature "
                f"metadata of assay {from_assay} which you can choose from: {bool_cols}\n The values in "
                f"brackets indicate the cell_key for which the feat_key is available. Choosing 'I' "
                f"as `feat_key` means that you will use all the genes for graph creation."
            )
        if custom_loadings is not None:
            reduction_method = "custom"
            dims = custom_loadings.shape[1]
            logger.info(
                f"`dims` parameter and its default value ignored as using custom loadings "
                f"with {dims} dims"
            )

        (
            log_transform,
            renormalize_subset,
            reduction_method,
            dims,
            pca_cell_key,
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            rand_state,
            k,
            n_centroids,
            local_connectivity,
            bandwidth,
        ) = self._set_graph_params(
            from_assay,
            cell_key,
            feat_key,
            log_transform,
            renormalize_subset,
            reduction_method,
            dims,
            pca_cell_key,
            ann_metric,
            ann_efc,
            ann_ef,
            ann_m,
            rand_state,
            k,
            n_centroids,
            local_connectivity,
            bandwidth,
        )
        normed_loc = f"{from_assay}/normed__{cell_key}__{feat_key}"
        reduction_loc = (
            f"{normed_loc}/reduction__{reduction_method}__{dims}__{pca_cell_key}"
        )
        ann_loc = f"{reduction_loc}/ann__{ann_metric}__{ann_efc}__{ann_ef}__{ann_m}__{rand_state}"
        ann_idx_loc = f"{self._fn}/{ann_loc}/ann_idx"
        knn_loc = f"{ann_loc}/knn__{k}"
        kmeans_loc = f"{reduction_loc}/kmeans__{n_centroids}__{rand_state}"
        graph_loc = f"{knn_loc}/graph__{local_connectivity}__{bandwidth}"

        data = assay.save_normalized_data(
            cell_key,
            feat_key,
            batch_size,
            normed_loc.split("/")[-1],
            log_transform,
            renormalize_subset,
            update_keys,
        )
        if custom_loadings is not None and data.shape[1] != custom_loadings.shape[0]:
            raise ValueError(
                f"Provided custom loadings has {custom_loadings.shape[0]} features while the data "
                f"has {data.shape[1]} features."
            )
        loadings = None
        fit_kmeans = True
        mu, sigma = np.ndarray([]), np.ndarray([])
        use_for_pca = self.cells.fetch(pca_cell_key, key=cell_key)
        if reduction_loc in self.z:
            # TODO: In future move 'mu' and 'sigma' to normed_loc rather than reduction_loc. This may however introduce
            # breaking changes.
            if "mu" in self.z[reduction_loc]:
                mu = self.z[reduction_loc]["mu"][:]
            if "sigma" in self.z[reduction_loc]:
                sigma = self.z[reduction_loc]["sigma"][:]
            if "reduction" in self.z[reduction_loc]:
                loadings = self.z[reduction_loc]["reduction"][:]
                if data.shape[1] != loadings.shape[0]:
                    logger.warning(
                        "Consistency breached in loading pre-cached loadings. Will perform fresh reduction."
                    )
                    loadings = None

                    del self.z[reduction_loc]
        else:
            if reduction_method in ["pca", "manual"]:
                mu = clean_array(
                    show_dask_progress(
                        data.mean(axis=0),
                        "Calculating mean of norm. data",
                        self.nthreads,
                    )
                )
                sigma = clean_array(
                    show_dask_progress(
                        data.std(axis=0),
                        "Calculating std. dev. of norm. data",
                        self.nthreads,
                    ),
                    1,
                )

        if custom_loadings is None:
            if loadings is None:
                pass  # Will compute fresh loadings
            else:
                logger.info(
                    f"Using existing loadings for {reduction_method} with {dims} dims"
                )
        else:
            if loadings is not None and np.array_equal(loadings, custom_loadings):
                logger.info("Custom loadings same as used before. Loading from cache")
            else:
                loadings = custom_loadings
                logger.info(
                    f"Using custom loadings with {dims} dims. Will overwrite any "
                    f"previously used custom loadings"
                )
                if reduction_loc in self.z:
                    del self.z[reduction_loc]

        if ann_loc in self.z:
            import hnswlib

            ann_idx = hnswlib.Index(space=ann_metric, dim=dims)
            ann_idx.load_index(ann_idx_loc)
            logger.info(f"Using existing ANN index")
        else:
            ann_idx = None
        if kmeans_loc in self.z:
            fit_kmeans = False
            logger.info(f"using existing kmeans cluster centers")
        disable_scaling = True if feat_scaling is False else False
        # TODO: expose LSImodel parameters
        ann_obj = AnnStream(
            data=data,
            k=k,
            n_cluster=n_centroids,
            reduction_method=reduction_method,
            dims=dims,
            loadings=loadings,
            use_for_pca=use_for_pca,
            mu=mu,
            sigma=sigma,
            ann_metric=ann_metric,
            ann_efc=ann_efc,
            ann_ef=ann_ef,
            ann_m=ann_m,
            nthreads=self.nthreads,
            ann_parallel=ann_parallel,
            rand_state=rand_state,
            do_kmeans_fit=fit_kmeans,
            disable_scaling=disable_scaling,
            ann_idx=ann_idx,
            lsi_params={},
        )

        if reduction_loc not in self.z:
            logger.debug(f"Saving loadings to {reduction_loc}")
            self.z.create_group(reduction_loc, overwrite=True)
            if ann_obj.loadings is not None:
                # can be None when no dimred is performed
                g = create_zarr_dataset(
                    self.z[reduction_loc],
                    "reduction",
                    (1000, 1000),
                    "f8",
                    ann_obj.loadings.shape,
                )
                g[:, :] = ann_obj.loadings
            # TODO: This belongs better in normed_loc
            if reduction_method in ["pca", "manual"]:
                g = create_zarr_dataset(
                    self.z[reduction_loc], "mu", (100000,), "f8", mu.shape
                )
                g[:] = mu
                g = create_zarr_dataset(
                    self.z[reduction_loc], "sigma", (100000,), "f8", sigma.shape
                )
                g[:] = sigma
        if ann_loc not in self.z:
            logger.debug(f"Saving ANN index to {ann_loc}")
            self.z.create_group(ann_loc, overwrite=True)
            ann_obj.annIdx.save_index(ann_idx_loc)
        if fit_kmeans:
            logger.debug(f"Saving kmeans clusters to {kmeans_loc}")
            self.z.create_group(kmeans_loc, overwrite=True)
            g = create_zarr_dataset(
                self.z[kmeans_loc],
                "cluster_centers",
                (1000, 1000),
                "f8",
                ann_obj.kmeans.cluster_centers_.shape,
            )
            g[:, :] = ann_obj.kmeans.cluster_centers_
            g = create_zarr_dataset(
                self.z[kmeans_loc],
                "cluster_labels",
                (100000,),
                "f8",
                ann_obj.clusterLabels.shape,
            )
            g[:] = ann_obj.clusterLabels
        if knn_loc in self.z and graph_loc in self.z:
            logger.info(f"KNN graph already exists will not recompute.")
        else:
            from .knn_utils import self_query_knn, smoothen_dists

            recall = None
            if knn_loc not in self.z:
                recall = self_query_knn(
                    ann_obj,
                    self.z.create_group(knn_loc, overwrite=True),
                    batch_size,
                    self.nthreads,
                )
                recall = "%.2f" % recall

            smoothen_dists(
                self.z.create_group(graph_loc, overwrite=True),
                self.z[knn_loc]["indices"],
                self.z[knn_loc]["distances"],
                local_connectivity,
                bandwidth,
                batch_size,
            )
            if recall is not None:
                logger.info(f"ANN recall: {recall}%")

        self.z[normed_loc].attrs["latest_reduction"] = reduction_loc
        self.z[reduction_loc].attrs["latest_ann"] = ann_loc
        self.z[reduction_loc].attrs["latest_kmeans"] = kmeans_loc
        self.z[ann_loc].attrs["latest_knn"] = knn_loc
        self.z[knn_loc].attrs["latest_graph"] = graph_loc
        if return_ann_object:
            return ann_obj
        if show_elbow_plot:
            from .plots import plot_elbow

            try:
                var_exp = 100 * ann_obj._pca.explained_variance_ratio_
            except AttributeError:
                logger.warning("PCA was not fitted so not showing an Elbow plot")
            else:
                plot_elbow(var_exp)
        return None

    def load_graph(
        self,
        *,
        from_assay: Optional[str] = None,
        cell_key: Optional[str] = None,
        feat_key: Optional[str] = None,
        symmetric: Optional[bool] = None,
        upper_only: Optional[bool] = None,
        use_k: Optional[int] = None,
        graph_loc: Optional[str] = None,
    ) -> csr_matrix:
        """
        Load the cell neighbourhood as a scipy sparse matrix

        Args:
            from_assay: Name of the assay. If None then the default assay is used.
            cell_key: Cell key used to create the graph. If None then the latest feature key used for creating a
                      KNN graph is used.
            feat_key: Feature key used to create the graph. If None then the latest feature key used for creating a
                      KNN graph is used.
            symmetric: If True, makes the graph symmetric by adding it to its transpose.
            upper_only: If True, then only the values from upper triangular of the matrix are returned. This is only
                       used when symmetric is True.
            use_k: Number of top k-nearest neighbours to keep in the graph. This value must be greater than 0 and less
                   the parameter k used. By default all neighbours are used. (Default value: None)
            graph_loc: Zarr hierarchy where the graph is stored. If no value is provided then graph location is
                       obtained from `_get_latest_graph_loc` method.

        Returns:
            A scipy sparse matrix representing cell neighbourhood graph.

        """

        def symmetrize(g):
            t = g + g.T
            t = t - g.multiply(g.T)
            return t

        from scipy.sparse import triu

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )

        if graph_loc is None:
            graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        if graph_loc not in self.z:
            raise ValueError(
                f"{graph_loc} not found in zarr location {self._fn}. "
                f"Run `make_graph` for assay {from_assay}"
            )
        n_cells, graph = self._store_to_sparse(graph_loc, "csr", use_k)
        if symmetric is True:
            graph = symmetrize(graph)
            if upper_only is True:
                graph = triu(graph)
        return graph
        # idx = None
        # if min_edge_weight > 0:
        #     idx = graph.data > min_edge_weight
        # # Following if-else block is for purpose for improving performance when no filtering is performed.
        # if idx is None:
        #     return graph
        # elif idx.sum() == graph.data.shape[0]:
        #     return graph
        # else:
        #     graph = graph.tocoo()
        #     return csr_matrix((graph.data[idx], (graph.row[idx], graph.col[idx])), shape=(n_cells, n_cells))

    def run_tsne(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        feat_key: str = None,
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        ini_embed: np.ndarray = None,
        tsne_dims: int = 2,
        lambda_scale: float = 1.0,
        max_iter: int = 500,
        early_iter: int = 200,
        alpha: int = 10,
        box_h: float = 0.7,
        temp_file_loc: str = ".",
        label: str = "tSNE",
        verbose: bool = True,
        parallel: bool = False,
        nthreads: int = None,
    ) -> None:
        """
        Run SGtSNE-pi (Read more here: https://github.com/fcdimitr/sgtsnepi/tree/v1.0.1). This is an implementation of
        tSNE that runs directly on graph structures. We use the graphs generated by `make_graph` method to create a
        layout of cells using tSNE algorithm. This function makes a system call to sgtSNE binary.
        To get a better understanding of how the parameters affect the embedding, check this out:
        http://t-sne-pi.cs.duke.edu/

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
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
            parallel: Whether to run tSNE in parallel mode. Setting value to True will use `nthreads` threads.
                      The results are not reproducible in parallel mode. (Default value: False)
            nthreads: If parallel=True then this number of threads will be used to run tSNE. By default the `nthreads`
                      attribute of the class is used. (Default value: None)

        Returns:

        """
        from uuid import uuid4
        from .knn_utils import export_knn_to_mtx
        from pathlib import Path
        import sys

        if sys.platform not in ["posix", "linux"]:
            logger.error(f"{sys.platform} operating system is currently not supported.")
            return None

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )

        uid = str(uuid4())
        knn_mtx_fn = Path(temp_file_loc, f"{uid}.mtx").resolve()
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=symmetric_graph,
            upper_only=graph_upper_only,
        )
        export_knn_to_mtx(knn_mtx_fn, graph)

        ini_emb_fn = Path(temp_file_loc, f"{uid}.txt").resolve()
        with open(ini_emb_fn, "w") as h:
            if ini_embed is None:
                ini_embed = self._get_ini_embed(
                    from_assay, cell_key, feat_key, tsne_dims
                ).flatten()
            else:
                if ini_embed.shape != (graph.shape[0], tsne_dims):
                    raise ValueError(
                        "ERROR: Provided initial embedding does not shape required shape: "
                        f"{(graph.shape[0], tsne_dims)}"
                    )
            h.write("\n".join(map(str, ini_embed)))
        out_fn = Path(temp_file_loc, f"{uid}_output.txt").resolve()
        if parallel:
            if nthreads is None:
                nthreads = self.nthreads
            else:
                assert type(nthreads) == int
        else:
            nthreads = 1
        cmd = (
            f"sgtsne -m {max_iter} -l {lambda_scale} -d {tsne_dims} -e {early_iter} -p {nthreads} -a {alpha}"
            f" -h {box_h} -i {ini_emb_fn} -o {out_fn} {knn_mtx_fn}"
        )
        if verbose:
            system_call(cmd)
        else:
            os.system(cmd)
        try:
            emb = pd.read_csv(out_fn, header=None, sep=" ")[
                list(range(tsne_dims))
            ].values.T
            for i in range(tsne_dims):
                self.cells.insert(
                    self._col_renamer(from_assay, cell_key, f"{label}{i + 1}"),
                    emb[i],
                    key=cell_key,
                    overwrite=True,
                )
            for fn in [out_fn, knn_mtx_fn, ini_emb_fn]:
                Path.unlink(fn)
        except FileNotFoundError:
            logger.error(
                "SG-tSNE failed, possibly due to missing libraries or file permissions. SG-tSNE currently "
                "fails on readthedocs"
            )
            for fn in [knn_mtx_fn, ini_emb_fn]:
                Path.unlink(fn)

    def run_umap(
        self,
        *,
        from_assay: Optional[str] = None,
        cell_key: Optional[str] = None,
        feat_key: Optional[str] = None,
        symmetric_graph: Optional[bool] = False,
        graph_upper_only: Optional[bool] = False,
        ini_embed: np.ndarray = None,
        umap_dims: int = 2,
        spread: float = 2.0,
        min_dist: float = 1,
        n_epochs: int = 300,
        repulsion_strength: float = 1.0,
        initial_alpha: float = 1.0,
        negative_sample_rate: float = 5,
        use_density_map: bool = False,
        dens_lambda: float = 2.0,
        dens_frac: float = 0.3,
        dens_var_shift: float = 0.1,
        random_seed: int = 4444,
        label: str = "UMAP",
        integrated_graph: Optional[str] = None,
        parallel: bool = False,
        nthreads: Optional[int] = None,
    ) -> None:
        """
        Runs UMAP algorithm using the precomputed cell-neighbourhood graph. The calculated UMAP coordinates are saved
        in the cell metadata table.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
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
            n_epochs: Same as n_epochs in UMAP package. The number of epochs to be used in optimizing the
                      low dimensional embedding. Larger values may result in more accurate embeddings.
                      (Default value: 300)
            repulsion_strength: Same as repulsion_strength in UMAP package. Weighting applied to negative samples in
                                low dimensional embedding optimization. Values higher than one will result in greater
                                weight being given to negative samples. (Default value: 1.0)
            initial_alpha: Same as learning_rate in UMAP package. The initial learning rate for the embedding
                           optimization. (Default value: 1.0)
            negative_sample_rate: Same as negative_sample_rate in UMAP package. The number of negative samples to
                                  select per positive sample in the optimization process. Increasing this value will
                                  result in greater repulsive force being applied, greater optimization cost, but
                                  slightly more accuracy. (Default value: 5)
            use_density_map:
            dens_lambda:
            dens_frac:
            dens_var_shift:
            random_seed: (Default value: 4444)
            label: base label for UMAP dimensions in the cell metadata column (Default value: 'UMAP')
            integrated_graph:
            parallel: Whether to run UMAP in parallel mode. Setting value to True will use `nthreads` threads.
                      The results are not reproducible in parallel mode. (Default value: False)
            nthreads: If parallel=True then this number of threads will be used to run UMAP. By default the `nthreads`
                      attribute of the class is used. (Default value: None)

        Returns:

        """
        from .umap import fit_transform
        from .utils import get_log_level

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        graph_loc = None
        if integrated_graph is not None:
            graph_loc = f"{self._integratedGraphsLoc}/{integrated_graph}"
            if graph_loc not in self.z:
                raise KeyError(
                    f"ERROR: An integrated graph with label: {integrated_graph} does not exist"
                )
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=symmetric_graph,
            upper_only=graph_upper_only,
            graph_loc=graph_loc,
        )

        if ini_embed is None:
            ini_embed = self._get_ini_embed(from_assay, cell_key, feat_key, umap_dims)
        if nthreads is None:
            nthreads = self.nthreads
        verbose = False
        if get_log_level() <= 20:
            verbose = True

        if use_density_map:
            if integrated_graph is not None:
                logger.warning(
                    "DensMap is not available for integrated graphs. Will run without UMAP without DensMap"
                )
            graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
            knn_loc = graph_loc.rsplit("/", 1)[0]
            logger.trace(f"Loading KNN dists and indices from {knn_loc}")
            dists = self.z[knn_loc].distances[:]
            indices = self.z[knn_loc].indices[:]
            dmat = csr_matrix(
                (
                    dists.flatten(),
                    (
                        np.repeat(range(indices.shape[0]), indices.shape[1]),
                        indices.flatten(),
                    ),
                ),
                shape=(indices.shape[0], indices.shape[0]),
            )
            # dmat = dmat.maximum(dmat.transpose()).todok()
            logger.trace(f"Created sparse KNN dists and indices")
            densmap_kwds = {
                "lambda": dens_lambda,
                "frac": dens_frac,
                "var_shift": dens_var_shift,
                "n_neighbors": dists.shape[1],
                "knn_dists": dmat,
            }
        else:
            densmap_kwds = {}

        t, a, b = fit_transform(
            graph=graph.tocoo(),
            ini_embed=ini_embed,
            spread=spread,
            min_dist=min_dist,
            n_epochs=n_epochs,
            random_seed=random_seed,
            repulsion_strength=repulsion_strength,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            densmap_kwds=densmap_kwds,
            parallel=parallel,
            nthreads=nthreads,
            verbose=verbose,
        )

        if integrated_graph is not None:
            from_assay = integrated_graph
        for i in range(umap_dims):
            self.cells.insert(
                self._col_renamer(from_assay, cell_key, f"{label}{i + 1}"),
                t[:, i],
                key=cell_key,
                overwrite=True,
            )
        return None

    def run_leiden_clustering(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        feat_key: str = None,
        resolution: int = 1,
        integrated_graph: Optional[str] = None,
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        label: str = "leiden_cluster",
        random_seed: int = 4444,
    ) -> None:
        """
        Executes Leiden graph clustering algorithm on the cell-neighbourhood graph and saves cluster identities in the
        cell metadata column.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key:  Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            resolution: Resolution parameter for `RBConfigurationVertexPartition` configuration
            integrated_graph:
            symmetric_graph: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            graph_upper_only: This parameter is forwarded to `load_graph` and is same as there. (Default value: True)
            label: base label for cluster identity in the cell metadata column (Default value: 'leiden_cluster')
            random_seed: (Default value: 4444)

        Returns:

        """
        try:
            # noinspection PyPackageRequirements
            import leidenalg
        except ImportError:
            raise ImportError(
                "ERROR: 'leidenalg' package is not installed. Please find the installation instructions "
                "here: https://github.com/vtraag/leidenalg#installation. Also, consider running Paris "
                "instead of Leiden clustering using `run_clustering` method"
            )
        # noinspection PyPackageRequirements
        import igraph  # python-igraph

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        graph_loc = None
        if integrated_graph is not None:
            graph_loc = f"{self._integratedGraphsLoc}/{integrated_graph}"
            if graph_loc not in self.z:
                raise KeyError(
                    f"ERROR: An integrated graph with label: {integrated_graph} does not exist"
                )
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=symmetric_graph,
            upper_only=graph_upper_only,
            graph_loc=graph_loc,
        )
        sources, targets = graph.nonzero()
        g = igraph.Graph()
        g.add_vertices(graph.shape[0])
        g.add_edges(list(zip(sources, targets)))
        g.es["weight"] = graph[sources, targets].A1
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=random_seed,
        )

        if integrated_graph is not None:
            from_assay = integrated_graph
        self.cells.insert(
            self._col_renamer(from_assay, cell_key, label),
            np.array(part.membership) + 1,
            fill_value=-1,
            key=cell_key,
            overwrite=True,
        )
        return None

    def run_clustering(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        feat_key: str = None,
        n_clusters: int = None,
        integrated_graph: Optional[str] = None,
        symmetric_graph: bool = False,
        graph_upper_only: bool = False,
        balanced_cut: bool = False,
        max_size: int = None,
        min_size: int = None,
        max_distance_fc: float = 2,
        force_recalc: bool = False,
        label: str = "cluster",
    ) -> None:
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
            integrated_graph:
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
            None
        """
        import sknetwork as skn

        if balanced_cut is False:
            if n_clusters is None:
                raise ValueError(
                    "ERROR: Please provide a value for n_clusters parameter. We are working on making "
                    "this parameter free"
                )
        else:
            if n_clusters is not None:
                logger.info(
                    "Using balanced cut method for cutting dendrogram. `n_clusters` will be ignored."
                )
            if max_size is None or min_size is None:
                raise ValueError(
                    "ERROR: Please provide value for max_size and min_size"
                )

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )

        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        if integrated_graph is not None:
            graph_loc = f"{self._integratedGraphsLoc}/{integrated_graph}"
            if graph_loc not in self.z:
                raise KeyError(
                    f"ERROR: An integrated graph with label: {integrated_graph} does not exist"
                )

        dendrogram_loc = f"{graph_loc}/dendrogram"
        # tuple are changed to list when saved as zarr attrs
        if dendrogram_loc in self.z and force_recalc is False:
            dendrogram = self.z[dendrogram_loc][:]
            logger.info("Using existing dendrogram")
        else:
            paris = skn.hierarchy.Paris()
            graph = self.load_graph(
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                symmetric=symmetric_graph,
                upper_only=graph_upper_only,
                graph_loc=graph_loc,
            )
            dendrogram = paris.fit_transform(graph)
            dendrogram[dendrogram == np.Inf] = 0
            g = create_zarr_dataset(
                self.z[graph_loc],
                dendrogram_loc.rsplit("/", 1)[1],
                (5000,),
                "f8",
                (graph.shape[0] - 1, 4),
            )
            g[:] = dendrogram
        self.z[graph_loc].attrs["latest_dendrogram"] = dendrogram_loc

        if balanced_cut:
            from .dendrogram import BalancedCut

            labels = BalancedCut(
                dendrogram, max_size, min_size, max_distance_fc
            ).get_clusters()
            logger.info(f"{len(set(labels))} clusters found")
        else:
            labels = skn.hierarchy.cut_straight(dendrogram, n_clusters=n_clusters) + 1

        if integrated_graph is not None:
            from_assay = integrated_graph
        self.cells.insert(
            self._col_renamer(from_assay, cell_key, label),
            labels,
            fill_value=-1,
            key=cell_key,
            overwrite=True,
        )

    def run_topacedo_sampler(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        feat_key: str = None,
        cluster_key: str = None,
        use_k: int = None,
        density_depth: int = 2,
        density_bandwidth: float = 5.0,
        max_sampling_rate: float = 0.05,
        min_sampling_rate: float = 0.01,
        min_cells_per_group: int = 3,
        snn_bandwidth: float = 5.0,
        seed_reward: float = 3.0,
        non_seed_reward: float = 0,
        edge_cost_multiplier: float = 1.0,
        edge_cost_bandwidth: float = 10.0,
        save_sampling_key: str = "sketched",
        save_density_key: str = "cell_density",
        save_mean_snn_key: str = "snn_value",
        save_seeds_key: str = "sketch_seeds",
        rand_state: int = 4466,
        return_edges: bool = False,
    ) -> Union[None, List]:
        """
        Perform sub-sampling (aka sketching) of cells using TopACeDo algorithm. Sub-sampling required
        that cells are partitioned in cluster already. Since, sub-sampling is dependent on cluster information, having,
        large number of homogeneous and even sized cluster improves sub-sampling results.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            cluster_key: Name of the column in cell metadata table where cluster information is stored.
            use_k: Number of top k-nearest neighbours to retain in the graph over which downsampling is performed.
                   BY default all neighbours are used. (Default value: None)
            density_depth: Same as 'search_depth' parameter in `calc_neighbourhood_density`. (Default value: 2)
            density_bandwidth: This value is used to scale the penalty affected by neighbourhood density. Higher values
                               will lead to to larger penalty. (Default value: 5.0)
            max_sampling_rate: Maximum fraction of cells to sample from each group. The effective sampling rate is lower
                               than this value depending on the neighbourhood degree and SNN density of cells.
                               Should be greater than 0 and less than 1. (Default value: 0.1)
            min_sampling_rate: Minimum sampling rate. Effective sampling rate is not allowed to be lower than this
                               value. Should be greater than 0 and less than 1. (Default value: 0.01)
            min_cells_per_group: Minimum number of cells to sample from each group. (Default value: 3)
            snn_bandwidth: Bandwidth for the shared nearest neighbour award. Clusters with higher mean SNN values get
                           lower sampling penalty. This value, is raised to mean SNN value of the cluster to obtain
                           sampling reward of the cluster. (Default value: 5.0)
            seed_reward: Reward/prize value for seed nodes. (Default value: 3.0)
            non_seed_reward: Reward/prize for non-seed nodes. (Default value: 0.1)
            edge_cost_multiplier: This value is multiplier to each edge's cost. Higher values will make graph traversal
                                  costly and might lead to removal of poorly connected nodes (Default value: 1.0)
            edge_cost_bandwidth: This value is raised to edge cost to get an adjusted edge cost (Default value: 1.0)
            save_sampling_key: base label for marking the cells that were sampled into a cell metadata column
                               (Default value: 'sketched')
            save_density_key: base label for saving the cell neighbourhood densities into a cell metadata column
                              (Default value: 'cell_density')
            save_mean_snn_key: base label for saving the SNN value for each cells (identified by topacedo sampler) into
                               a cell metadata column (Default value: 'snn_value')
            save_seeds_key: base label for saving the seed cells (identified by topacedo sampler) into a cell
                            metadata column (Default value: 'sketch_seeds')
            rand_state: A random values to set seed while sampling cells from a cluster randomly. (Default value: 4466)
            return_edges: If True, then steiner nodes and edges are returned. (Default value: False)

        Returns:

        """

        try:
            from topacedo import TopacedoSampler
        except ImportError:
            logger.error("Could not find topacedo package")
            return None

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if cluster_key is None:
            raise ValueError("ERROR: Please provide a value for cluster key")
        clusters = pd.Series(self.cells.fetch(cluster_key, cell_key))
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=False,
            upper_only=False,
            use_k=use_k,
        )
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        dendrogram = self.z[f"{graph_loc}/dendrogram"][:]

        if len(clusters) != graph.shape[0]:
            raise ValueError(
                f"ERROR: cluster information exists for {len(clusters)} cells while graph has "
                f"{graph.shape[0]} cells."
            )
        sampler = TopacedoSampler(
            graph,
            clusters.values,
            dendrogram,
            density_depth,
            density_bandwidth,
            max_sampling_rate,
            min_sampling_rate,
            min_cells_per_group,
            snn_bandwidth,
            seed_reward,
            non_seed_reward,
            edge_cost_multiplier,
            edge_cost_bandwidth,
            rand_state,
        )
        nodes, edges = sampler.run()
        a = np.zeros(self.cells.fetch_all(cell_key).sum()).astype(bool)
        a[nodes] = True
        key = self._col_renamer(from_assay, cell_key, save_sampling_key)
        self.cells.insert(key, a, fill_value=False, key=cell_key, overwrite=True)
        logger.debug(f"Sketched cells saved under column '{key}'")

        key = self._col_renamer(from_assay, cell_key, save_density_key)
        self.cells.insert(key, sampler.densities, key=cell_key, overwrite=True)
        logger.debug(f"Cell neighbourhood densities saved under column: '{key}'")

        key = self._col_renamer(from_assay, cell_key, save_mean_snn_key)
        self.cells.insert(key, sampler.meanSnn, key=cell_key, overwrite=True)
        logger.debug(f"Mean SNN values saved under column: '{key}'")

        a = np.zeros(self.cells.fetch_all(cell_key).sum()).astype(bool)
        a[sampler.seeds] = True
        key = self._col_renamer(from_assay, cell_key, save_seeds_key)
        self.cells.insert(key, a, fill_value=False, key=cell_key, overwrite=True)
        logger.debug(f"Seed cells saved under column: '{key}'")

        if return_edges:
            return edges

    def get_imputed(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        feature_name: str = None,
        feat_key: str = None,
        t: int = 2,
        cache_operator: bool = True,
    ) -> np.ndarray:
        """

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feature_name: Name of the feature to be imputed
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            t: Same as the t parameter in MAGIC. Higher values lead to larger diffusion of values. Too large values
               can slow down the algorithm and cause over-smoothening. (Default value: 2)
            cache_operator: Whether to keep the diffusion operator in memory after the method returns. Can be useful
                            to set to True if many features are to imputed in a batch but can lead to increased memory
                            usage. (Default value: True)

        Returns:
            An array of imputed values for the given feature

        """

        def calc_diff_operator(g: csr_matrix, to_power: int) -> coo_matrix:
            d = np.ravel(g.sum(axis=1))
            d[d != 0] = 1 / d[d != 0]
            n = g.shape[0]
            d = csr_matrix((d, (range(n), range(n))), shape=[n, n])
            return d.dot(g).__pow__(to_power).tocoo()

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        if feature_name is None:
            raise ValueError(
                "ERROR: Please provide name for the feature to be imputed. It can, for example, "
                "be a gene name."
            )
        data = self.get_cell_vals(
            from_assay=from_assay, cell_key=cell_key, k=feature_name
        )

        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        magic_loc = f"{graph_loc}/magic_{t}"
        if magic_loc in self.z:
            logger.info("Using existing MAGIC diffusion operator")
            if self._cachedMagicOperatorLoc == magic_loc:
                diff_op = self._cachedMagicOperator
            else:
                n_cells, _ = self._get_graph_ncells_k(graph_loc)
                store = self.z[magic_loc]
                diff_op = coo_matrix(
                    (store["data"][:], (store["row"][:], store["col"][:])),
                    shape=(n_cells, n_cells),
                )
                if cache_operator:
                    self._cachedMagicOperator = diff_op
                    self._cachedMagicOperatorLoc = magic_loc
                else:
                    self._cachedMagicOperator = None
                    self._cachedMagicOperatorLoc = None
        else:
            graph = self.load_graph(
                from_assay=from_assay,
                cell_key=cell_key,
                feat_key=feat_key,
                symmetric=True,
                upper_only=False,
            )
            diff_op = calc_diff_operator(graph, t)
            shape = diff_op.data.shape
            store = self.z.create_group(magic_loc, overwrite=True)
            for i, j in zip(["row", "col", "data"], ["uint32", "uint32", "float32"]):
                zg = create_zarr_dataset(store, i, (1000000,), j, shape)
                zg[:] = diff_op.__getattribute__(i)
            self.z[graph_loc].attrs["latest_magic"] = magic_loc
            if cache_operator:
                self._cachedMagicOperator = diff_op
                self._cachedMagicOperatorLoc = magic_loc
            else:
                self._cachedMagicOperator = None
                self._cachedMagicOperatorLoc = None
        return diff_op.dot(data)

    def run_pseudotime_scoring(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        feat_key: str = None,
        k_singular: int = 20,
        r_vec: np.ndarray = None,
        label: str = "pseudotime",
    ) -> None:
        """
        Calculate differentiation potential of cells. This function is a reimplementation of population balance
        analysis (PBA) approach published in Weinreb et al. 2017, PNAS. This function computes the random walk
        normalized Laplacian matrix of the reference graph, L_rw = I-A/D and then calculates a Moore-Penrose
        pseudoinverse of L_rw. The method takes an optional but recommended parameter 'r' which represents the
        relative rates of proliferation and loss in different gene expression states (R). If not provided then a vector
        with ones is used. The differentiation potential is the dot product of inverse L_rw and R

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default the latest
                        used feature for the given assay will be used.
            k_singular: Number of smallest singular values to save.
            r_vec: Same as parameter R in the above said reference.
            label:

        Returns:

        """

        from scipy.sparse.linalg import svds

        def inverse_degree(g):
            d = np.ravel(g.sum(axis=1))
            n = g.shape[0]
            d[d != 0] = 1 / d[d != 0]
            return csr_matrix((d, (range(n), range(n))), shape=[n, n])

        def laplacian(g, inv_deg):
            n = g.shape[0]
            identity = csr_matrix((np.ones(n), (range(n), range(n))), shape=[n, n])
            return identity - graph.dot(inv_deg)

        def pseudo_inverse(lap):
            u, s, vt = svds(lap, k=k_singular, which="SM")
            return vt.T @ np.diag(np.linalg.pinv([s]).reshape(1, -1)[0]) @ u.T

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        graph = self.load_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            symmetric=True,
            upper_only=False,
        )
        inv_lap = pseudo_inverse(laplacian(graph, inverse_degree(graph)))
        if r_vec is None:
            r_vec = np.ones(inv_lap.shape[0])
        v = np.dot(inv_lap, r_vec)
        self.cells.insert(
            self._col_renamer(from_assay, cell_key, label),
            v,
            key=cell_key,
            overwrite=True,
        )
        return None

    def integrate_assays(
        self,
        assays: List[str],
        label: str,
        chunk_size: int = 10000,
    ) -> None:
        """
        Merges KNN graphs of two or more assays from within the same DataStore.
        The input KNN graphs should have been constructed on the same set of cells and
        should each have been constructed with equal number of neighbours (parameter: k)
        The merged KNN graph has the same size and shape as the input graphs.

        Args:
            assays: Name of the input assays. The latest constructed graph from each assay is used.

        Returns: None

        """
        from .knn_utils import merge_graphs

        merged_graph = []
        for assay in assays:
            if assay not in self.assayNames:
                raise ValueError(f"ERROR: Assay {assay} was not found.")
            merged_graph.append(
                self.load_graph(
                    from_assay=assay,
                    cell_key=None,
                    feat_key=None,
                    symmetric=False,
                    upper_only=False,
                ).tocsr()
            )
        merged_graph = merge_graphs(merged_graph)

        n_cells = merged_graph.shape[0]
        n_neighbors = int(merged_graph.size / n_cells)

        ig_loc = self._integratedGraphsLoc
        if ig_loc not in self.z:
            self.z.create_group(ig_loc)
        if label in self.z[ig_loc]:
            del self.z[f"{ig_loc}/{label}"]
        store = self.z.create_group(f"{ig_loc}/{label}")
        store.attrs["n_cells"] = n_cells
        store.attrs["n_neighbors"] = n_neighbors

        zge = create_zarr_dataset(
            store, f"edges", (chunk_size,), ("u8", "u8"), (n_cells * n_neighbors, 2)
        )
        zgw = create_zarr_dataset(
            store, f"weights", (chunk_size,), "f8", (n_cells * n_neighbors)
        )

        zge[:, 0] = merged_graph.row
        zge[:, 1] = merged_graph.col
        zgw[:] = merged_graph.data


# Note for the docstring: Attributes are copied from BaseDataStore docstring since the constructor is inherited.
# Meaning, for any attribute change in BaseDataStore a manual update to docstring here is needed as well. - RO
class MappingDatastore(GraphDataStore):
    """
    This class extends GraphDataStore by providing methods for mapping/ projection of cells from one DataStore
    onto another.

    It also contains the methods required for label transfer, mapping score generation and co-embedding.

    Attributes:
        cells: List of cell barcodes.
        assayNames: List of assay names in Zarr file, e. g. 'RNA' or 'ATAC'.
        nthreads: Number of threads to use for this datastore instance.
        z: The Zarr file (directory) used for for this datastore instance.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_mapping(
        self,
        *,
        target_assay: Assay,
        target_name: str,
        target_feat_key: str,
        from_assay: str = None,
        cell_key: str = "I",
        feat_key: str = None,
        save_k: int = 3,
        batch_size: int = 1000,
        ref_mu: bool = True,
        ref_sigma: bool = True,
        run_coral: bool = False,
        exclude_missing: bool = False,
        filter_null: bool = False,
        feat_scaling: bool = True,
    ) -> None:
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
            None

        """
        from .mapping_utils import align_features, coral

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )
        source_assay = self._get_assay(from_assay)

        if type(target_assay) != type(source_assay):
            raise TypeError(
                f"ERROR: Source assay ({type(source_assay)}) and target assay "
                f"({type(target_assay)}) are of different types. "
                f"Mapping can only be performed between same assay types"
            )
        if type(target_assay) == RNAassay:
            if target_assay.sf != source_assay.sf:
                logger.info(
                    f"Resetting target assay's size factor from {target_assay.sf} to {source_assay.sf}"
                )
                target_assay.sf = source_assay.sf

        if target_feat_key == feat_key:
            raise ValueError(
                f"ERROR: `target_feat_key` cannot be sample as `feat_key`: {feat_key}"
            )

        feat_idx = align_features(
            source_assay,
            target_assay,
            cell_key,
            feat_key,
            target_feat_key,
            filter_null,
            exclude_missing,
            self.nthreads,
        )
        logger.debug(f"{len(feat_idx)} features being used for mapping")
        if np.all(
            source_assay.feats.active_index(cell_key + "__" + feat_key) == feat_idx
        ):
            ann_feat_key = feat_key
        else:
            ann_feat_key = f"{feat_key}_common_{target_name}"
            a = np.zeros(source_assay.feats.N).astype(bool)
            a[feat_idx] = True
            source_assay.feats.insert(
                cell_key + "__" + ann_feat_key, a, fill_value=False, overwrite=True
            )
        if run_coral:
            feat_scaling = False
        ann_obj = self.make_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=ann_feat_key,
            return_ann_object=True,
            update_keys=False,
            feat_scaling=feat_scaling,
        )
        if save_k > ann_obj.k:
            logger.warning(f"`save_k` was decreased to {ann_obj.k}")
            save_k = ann_obj.k
        target_data = daskarr.from_zarr(
            target_assay.z[f"normed__I__{target_feat_key}/data"], inline_array=True
        )
        if run_coral is True:
            # Reversing coral here to correct target data
            coral(
                target_data, ann_obj.data, target_assay, target_feat_key, self.nthreads
            )
            target_data = daskarr.from_zarr(
                target_assay.z[f"normed__I__{target_feat_key}/data_coral"],
                inline_array=True,
            )
        if ann_obj.method == "pca" and run_coral is False:
            if ref_mu is False:
                mu = show_dask_progress(
                    target_data.mean(axis=0),
                    "Calculating mean of target norm. data",
                    self.nthreads,
                )
                ann_obj.mu = clean_array(mu)
            if ref_sigma is False:
                sigma = show_dask_progress(
                    target_data.std(axis=0),
                    "Calculating std. dev. of target norm. data",
                    self.nthreads,
                )
                ann_obj.sigma = clean_array(sigma, 1)
        if "projections" not in source_assay.z:
            source_assay.z.create_group("projections")
        store = source_assay.z["projections"].create_group(target_name, overwrite=True)
        nc, nk = target_assay.cells.fetch_all("I").sum(), save_k
        zi = create_zarr_dataset(store, "indices", (batch_size,), "u8", (nc, nk))
        zd = create_zarr_dataset(store, "distances", (batch_size,), "f8", (nc, nk))
        entry_start = 0
        for i in tqdmbar(
            target_data.blocks,
            desc=f"Mapping cells from {target_name}",
            total=target_data.numblocks[0],
        ):
            a: np.ndarray = controlled_compute(i, self.nthreads)
            ki, kd = ann_obj.transform_ann(ann_obj.reducer(a), k=save_k)
            entry_end = entry_start + len(ki)
            zi[entry_start:entry_end, :] = ki
            zd[entry_start:entry_end, :] = kd
            entry_start = entry_end
        return None

    def get_mapping_score(
        self,
        *,
        target_name: str,
        target_groups: np.ndarray = None,
        from_assay: str = None,
        cell_key: str = "I",
        log_transform: bool = True,
        multiplier: float = 1000,
        weighted: bool = True,
        fixed_weight: float = 0.1,
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Yields the mapping scores that were a result of a mapping.

        Mapping scores are an indication of degree of similarity of reference cells in the graph to the target cells.
        The more often a reference cell is found in the nearest neighbour list of the target cells, the higher
        the mapping score will be for that cell.

        Args:
            target_name: Name of target data. This used to keep track of projections in the Zarr hierarchy
            target_groups: Group/cluster identity of target cells. This will then be used to calculate mapping score
                           for each group separately.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            log_transform: If True (default) then the mapping scores will be log transformed
            multiplier: A scaling factor for mapping scores. All scores al multiplied this value. This mostly intended
                        for visualization of mapping scores (Default: 1000)
            weighted: Use distance weights when calculating mapping scores (default: True). If False then the actual
                      distances between the reference and target cells are ignored.
            fixed_weight: Used when `weighted` is False. This is the value that is added to mapping score of each
                          reference cell for every projected target cell. Can be any value >0.

        Yields:
            A tuple of group name and mapping score of reference cells for that target group.

        """
        if from_assay is None:
            from_assay = self._defaultAssay
        store_loc = f"{from_assay}/projections/{target_name}"
        if store_loc not in self.z:
            raise KeyError(
                f"ERROR: Projections have not been computed for {target_name} in th latest graph. Please"
                f" run `run_mapping` or update latest_graph by running `make_graph` with desired parameters"
            )
        store = self.z[store_loc]

        indices = store["indices"][:]
        dists = store["distances"][:]
        # TODO: add more robust options for distance calculation here
        dists = 1 / (np.log1p(dists) + 1)
        n_cells = indices.shape[0]

        if target_groups is not None:
            if len(target_groups) != n_cells:
                raise ValueError(
                    f"ERROR: Length of target_groups {len(target_groups)} not same as number of target "
                    f"cells in the projection {n_cells}"
                )
            groups = pd.Series(target_groups)
        else:
            groups = pd.Series(np.zeros(n_cells))

        ref_n_cells = self.cells.fetch_all(cell_key).sum()
        for group in sorted(groups.unique()):
            coi = {x: None for x in groups[groups == group].index.values}
            ms = np.zeros(ref_n_cells)
            for n, i, j in zip(range(len(indices)), indices, dists):
                if n in coi:
                    for x, y in zip(i, j):
                        if weighted:
                            ms[x] += y
                        else:
                            ms[x] += fixed_weight
            ms = multiplier * ms / len(coi)
            if log_transform:
                ms = np.log1p(ms)
            yield group, ms

    def get_target_classes(
        self,
        *,
        target_name: str,
        from_assay: str = None,
        cell_key: str = "I",
        reference_class_group: str = None,
        threshold_fraction: int = 0.5,
        target_subset: List[int] = None,
        na_val="NA",
    ) -> pd.Series:
        """
        Perform classification of target cells using a reference group.

        Args:
            target_name: Name of target data. This value should be the same as that used for `run_mapping` earlier.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            reference_class_group: Group/cluster identity of the reference cells. These are the target labels for the
                                   classifier. The value here should be a column from cell metadata table. For
                                   example, to use default clustering identity one could use `RNA_cluster`
            threshold_fraction: The threshold for deciding if a cell belongs to a group or not.
                                Constrained between 0 and 1. (Default value: 0.5)
            target_subset: Choose only a subset of target cells to be classified. The value should be a list of
                           indices of the target cells. (Default: None)
            na_val: Value to be used if a cell is not classified to any of the `reference_class_group`.
                    (Default value: 'NA')

        Returns: A pandas Series containing predicted class for each cell in the projected sample (`target_name`).

        """
        if from_assay is None:
            from_assay = self._defaultAssay
        store_loc = f"{from_assay}/projections/{target_name}"
        if store_loc not in self.z:
            raise KeyError(
                f"ERROR: Projections have not been computed for {target_name} in th latest graph. Please"
                f" run `run_mapping` or update latest_graph by running `make_graph` with desired parameters"
            )
        if reference_class_group is None:
            raise ValueError(
                "ERROR: A value is required for the parameter `reference_class_group`. "
                "This can be any cell metadata column. Please choose the value that contains cluster or "
                "group information"
            )
        ref_groups = self.cells.fetch(reference_class_group, key=cell_key)
        if threshold_fraction < 0 or threshold_fraction > 1:
            raise ValueError(
                "ERROR: `threshold_fraction` should have a value between 0 and 1"
            )
        if target_subset is not None:
            if type(target_subset) != list:
                raise TypeError("ERROR:  `target_subset` should be <list> type")
            target_subset = {x: None for x in target_subset}

        store = self.z[store_loc]
        indices = store["indices"][:]
        dists = store["distances"][:]
        preds = []
        weights = 1 - (dists / dists.max(axis=1).reshape(-1, 1))
        for n in range(indices.shape[0]):
            if target_subset is not None and n not in target_subset:
                continue
            wd = {}
            for i, j in zip(indices[n, :-1], weights[n, :-1]):
                k = ref_groups[i]
                if k not in wd:
                    wd[k] = 0
                wd[k] += j
            temp = na_val
            s = weights[n, :-1].sum()
            for i, j in wd.items():
                if j / s > threshold_fraction:
                    if temp == na_val:
                        temp = i
                    else:
                        temp = na_val
                        break
            preds.append(temp)
        return pd.Series(preds)

    def load_unified_graph(
        self,
        *,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        target_names: List[str],
        use_k: int,
        target_weight: float,
    ) -> Tuple[List[int], csr_matrix]:
        """
        This is similar to ``load_graph`` but includes projected cells and their edges.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            target_names: Name of target datasets to be included in the unified graph
            use_k: Number of nearest neighbour edges of each projected cell to be included. If this value is larger than
                   than `save_k` parameter while running mapping for the `target_name` target then `use_k` is reset to
                   'save_k'
            target_weight: A constant uniform weight to be ascribed to each target-reference edge.

        Returns:

        """
        # TODO:  allow loading multiple targets

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        edges = self.z[graph_loc].edges[:]
        weights = self.z[graph_loc].weights[:]
        ref_n_cells = self.cells.fetch_all(cell_key).sum()
        store = self.z[from_assay].projections
        pidx = np.vstack([store[x].indices[:, :use_k] for x in target_names])
        n_cells = [ref_n_cells] + [store[x].indices.shape[0] for x in target_names]
        ne = []
        nw = []
        for n, i in enumerate(pidx):
            for j in i:
                ne.append([ref_n_cells + n, j])
                # TODO: Better way to weigh the target edges
                nw.append(target_weight)
        me = np.vstack([edges, ne]).astype(int)
        mw = np.hstack([weights, nw])
        tot_cells = ref_n_cells + pidx.shape[0]
        graph = csr_matrix((mw, (me[:, 0], me[:, 1])), shape=(tot_cells, tot_cells))
        return n_cells, graph

    def _get_uni_ini_embed(
        self,
        from_assay: str,
        cell_key: str,
        feat_key: str,
        graph: csr_matrix,
        ini_embed_with: str,
        ref_n_cells: int,
    ) -> np.ndarray:
        if ini_embed_with == "kmeans":
            ini_embed = self._get_ini_embed(from_assay, cell_key, feat_key, 2)
        else:
            x = self.cells.fetch(f"{ini_embed_with}1", cell_key)
            y = self.cells.fetch(f"{ini_embed_with}2", cell_key)
            ini_embed = np.array([x, y]).T.astype(np.float32, order="C")
        targets_best_nn = np.array(np.argmax(graph, axis=1)).reshape(1, -1)[0][
            ref_n_cells:
        ]
        return np.vstack([ini_embed, ini_embed[targets_best_nn]])

    def _save_embedding(
        self,
        from_assay: str,
        cell_key: str,
        label: str,
        embedding: np.ndarray,
        n_cells: List[int],
        target_names: List[str],
    ) -> None:
        g = create_zarr_dataset(
            self.z[from_assay].projections, label, (1000, 2), "float64", embedding.shape
        )
        g[:] = embedding
        g.attrs["n_cells"] = [
            int(x) for x in n_cells
        ]  # forcing int type here otherwise json raises TypeError
        g.attrs["target_names"] = target_names
        for i in range(2):
            self.cells.insert(
                self._col_renamer(from_assay, cell_key, f"{label}{i + 1}"),
                embedding[: n_cells[0], i],
                key=cell_key,
                overwrite=True,
            )
        return None

    def run_unified_umap(
        self,
        *,
        target_names: List[str],
        from_assay: str = None,
        cell_key: str = "I",
        feat_key: str = None,
        use_k: int = 3,
        target_weight: float = 0.1,
        spread: float = 2.0,
        min_dist: float = 1,
        n_epochs: int = 200,
        repulsion_strength: float = 1.0,
        initial_alpha: float = 1.0,
        negative_sample_rate: float = 5,
        random_seed: int = 4444,
        ini_embed_with: str = "kmeans",
        label: str = "unified_UMAP",
        parallel: bool = False,
        nthreads: int = None,
    ) -> None:
        """
        Calculates the UMAP embedding for graph obtained using ``load_unified_graph``.

        The loaded graph is processed the same way as the graph as in ``run_umap``.

        Args:
            target_names: Names of target datasets to be included in the unified UMAP.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            use_k: Number of nearest neighbour edges of each projected cell to be included. If this value is larger than
                   than `save_k` parameter while running mapping for the `target_name` target then `use_k` is reset to
                   'save_k'
            target_weight: A constant uniform weight to be ascribed to each target-reference edge.
            spread: Same as spread in UMAP package.  The effective scale of embedded points. In combination with
                    ``min_dist`` this determines how clustered/clumped the embedded points are.
            min_dist: Same as min_dist in UMAP package. The effective minimum distance between embedded points.
                      Smaller values will result in a more clustered/clumped embedding where nearby points on the
                      manifold are drawn closer together, while larger values will result on a more even dispersal of
                      points. The value should be set relative to the ``spread`` value, which determines the scale at
                      which embedded points will be spread out. (Default value: 1)
            n_epochs: Same as n_epochs in UMAP package. The number of training epochs to be used in optimizing the
                      low dimensional embedding. Larger values result in more accurate embeddings.
                      (Default value: 200)
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
            ini_embed_with: either 'kmeans' or a column from cell metadata to be used as initial embedding coordinates
            label: base label for UMAP dimensions in the cell metadata column (Default value: 'UMAP')
            parallel: Whether to run UMAP in parallel mode. Setting value to True will use `nthreads` threads.
                      The results are not reproducible in parallel mode. (Default value: False)
            nthreads: If parallel=True then this number of threads will be used to run UMAP. By default the `nthreads`
                      attribute of the class is used. (Default value: None)

        Returns:
            None
        """
        from .umap import fit_transform
        from .utils import get_log_level

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        n_cells, graph = self.load_unified_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            target_names=target_names,
            use_k=use_k,
            target_weight=target_weight,
        )
        ini_embed = self._get_uni_ini_embed(
            from_assay, cell_key, feat_key, graph, ini_embed_with, n_cells[0]
        )
        if nthreads is None:
            nthreads = self.nthreads
        verbose = False
        if get_log_level() <= 20:
            verbose = True
        t, a, b = fit_transform(
            graph=graph.tocoo(),
            ini_embed=ini_embed,
            spread=spread,
            min_dist=min_dist,
            n_epochs=n_epochs,
            random_seed=random_seed,
            repulsion_strength=repulsion_strength,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            densmap_kwds={},
            parallel=parallel,
            nthreads=nthreads,
            verbose=verbose,
        )
        self._save_embedding(from_assay, cell_key, label, t, n_cells, target_names)
        return None

    def run_unified_tsne(
        self,
        *,
        target_names: List[str],
        from_assay: str = None,
        cell_key: str = "I",
        feat_key: str = None,
        use_k: int = 3,
        target_weight: float = 0.5,
        lambda_scale: float = 1.0,
        max_iter: int = 500,
        early_iter: int = 200,
        alpha: int = 10,
        box_h: float = 0.7,
        temp_file_loc: str = ".",
        verbose: bool = True,
        ini_embed_with: str = "kmeans",
        label: str = "unified_tSNE",
    ) -> None:
        """
        Calculates the tSNE embedding for graph obtained using ``load_unified_graph``. The loaded graph is processed
        the same way as the graph as in ``run_tsne``.

        Args:
            target_names: Names of target datasets to be included in the unified tSNE.
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in the desired graph. By default the latest
                       used feature for the given assay will be used.
            use_k: Number of nearest neighbour edges of each projected cell to be included. If this value is larger than
                   than `save_k` parameter while running mapping for the `target_name` target then `use_k` is reset to
                   'save_k'.
            target_weight: A constant uniform weight to be ascribed to each target-reference edge.
            lambda_scale: λ rescaling parameter. (Default value: 1.0)
            max_iter: Maximum number of iterations. (Default value: 500)
            early_iter: Number of early exaggeration iterations. (Default value: 200)
            alpha: Early exaggeration multiplier. (Default value: 10)
            box_h: Grid side length (accuracy control). Lower values might drastically slow down
                   the algorithm (Default value: 0.7)
            temp_file_loc: Location of temporary file. By default these files will be created in the current working
                           directory. These files are deleted before the method returns.
            verbose: If True (default) then the full log from SGtSNEpi algorithm is shown.
            ini_embed_with: Initial embedding coordinates for the cells in cell_key. Should have same number of columns
                            as tsne_dims. If not value is provided then the initial embedding is obtained using
                            `get_ini_embed`.
            label: Base label for tSNE dimensions in the cell metadata column. (Default value: 'tSNE')

        Returns:

        """
        from uuid import uuid4
        from .knn_utils import export_knn_to_mtx
        from pathlib import Path

        if from_assay is None:
            from_assay = self._defaultAssay
        if feat_key is None:
            feat_key = self._get_latest_feat_key(from_assay)
        n_cells, graph = self.load_unified_graph(
            from_assay=from_assay,
            cell_key=cell_key,
            feat_key=feat_key,
            target_names=target_names,
            use_k=use_k,
            target_weight=target_weight,
        )
        ini_embed = self._get_uni_ini_embed(
            from_assay, cell_key, feat_key, graph, ini_embed_with, n_cells[0]
        )

        uid = str(uuid4())
        ini_emb_fn = Path(temp_file_loc, f"{uid}.txt").resolve()
        with open(ini_emb_fn, "w") as h:
            h.write("\n".join(map(str, ini_embed.flatten())))
        del ini_embed
        knn_mtx_fn = Path(temp_file_loc, f"{uid}.mtx").resolve()
        export_knn_to_mtx(knn_mtx_fn, graph)
        out_fn = Path(temp_file_loc, f"{uid}_output.txt").resolve()
        cmd = (
            f"sgtsne -m {max_iter} -l {lambda_scale} -d {2} -e {early_iter} -p 1 -a {alpha}"
            f" -h {box_h} -i {ini_emb_fn} -o {out_fn} {knn_mtx_fn}"
        )
        if verbose:
            system_call(cmd)
        else:
            os.system(cmd)
        t = pd.read_csv(out_fn, header=None, sep=" ")[[0, 1]].values
        self._save_embedding(from_assay, cell_key, label, t, n_cells, target_names)
        for fn in [out_fn, knn_mtx_fn, ini_emb_fn]:
            Path.unlink(fn)
        return None

    def plot_unified_layout(
        self,
        *,
        from_assay: str = None,
        layout_key: str = None,
        show_target_only: bool = False,
        ref_name: str = "reference",
        target_groups: list = None,
        width: float = 6,
        height: float = 6,
        cmap=None,
        color_key: dict = None,
        mask_color: str = "k",
        point_size: float = 10,
        ax_label_size: float = 12,
        frame_offset: float = 0.05,
        spine_width: float = 0.5,
        spine_color: str = "k",
        displayed_sides: tuple = ("bottom", "left"),
        legend_ondata: bool = False,
        legend_onside: bool = True,
        legend_size: float = 12,
        legends_per_col: int = 20,
        cbar_shrink: float = 0.6,
        marker_scale: float = 70,
        lspacing: float = 0.1,
        cspacing: float = 1,
        savename: str = None,
        save_dpi: int = 300,
        ax=None,
        force_ints_as_cats: bool = True,
        n_columns: int = 1,
        w_pad: float = 1,
        h_pad: float = 1,
        scatter_kwargs: dict = None,
        shuffle_zorder: bool = True,
        show_fig: bool = True,
    ):
        """
        Plots the reference and target cells in their unified space.

        This function helps plotting the reference and target cells the coordinates for which were obtained from
        either `run_unified_tsne` or `run_unified_umap`. Since the coordinates are not saved in the cell metadata
        but rather in the projections slot of the Zarr hierarchy, this function is needed to correctly fetch the values
        for reference and target cells. Additionally this function provides a way to colour target cells by bringing in
        external annotations for those cells.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            layout_key: Should be same as the parameter value for `label` in `run_unified_umap` or `run_unified_tsne`
                        (Default value: 'UMAP')
            show_target_only: If True then the reference cells are not shown (Default value: False)
            ref_name: A label for reference cells to be used in the legend. (Default value: 'reference')
            target_groups: Categorical values to be used to colourmap target cells. (Default value: None)
            width: Figure width (Default value: 6)
            height: Figure height (Default value: 6)
            cmap: A matplotlib colourmap to be used to colour categorical or continuous values plotted on the cells.
                  (Default value: tab20 for categorical variables and cmocean.deep for continuous variables)
            color_key: A custom colour map for cells. These can be used for categorical variables only. The keys in this
                       dictionary should be the category label as present in the `color_by` column and values should be
                       valid matplotlib colour names or hex codes of colours. (Default value: None)
            mask_color: Color to be used for masked values. This should be a valid matplotlib named colour or a hexcode
                        of a colour. (Default value: 'k')
            point_size: Size of each scatter point. This is overridden if `size_vals` is provided. Has no effect if
                        `do_shading` is True. (Default value: 10)
            ax_label_size: Font size for the x and y axis labels. (Default value: 12)
            frame_offset: Extend the x and y axis limits by this fraction (Default value: 0.05)
            spine_width: Line width of the displayed spines (Default value: 0.5)
            spine_color: Colour of the displayed spines.  (Default value: 'k')
            displayed_sides: Determines which figure spines are chosen. The spines to be shown can be supplied as a
                             tuple. The options are: top, bottom, left and right. (Default value: ('bottom', 'left) )
            legend_ondata: Whether to show category labels on the data (scatter points). The position of the label is
                           the centroid of the corresponding values.
                           (Default value: True)
            legend_onside: Whether to draw a legend table on the side of the figure. (Default value: True)
            legend_size: Font size of the legend text. (Default value: 12)
            legends_per_col: Number of legends to be used on each legend column. This value determines how many legend
                             legend columns will be drawn (Default value: 20)
            cbar_shrink: Shrinking factor for the width of color bar (Default value: 0.6)
            marker_scale: The relative size of legend markers compared with the originally drawn ones.
                          (Default value: 70)
            lspacing: The vertical space between the legend entries. Measured in font-size units. (Default value: 0.1)
            cspacing: The spacing between columns. Measured in font-size units. (Default value: 1)
            savename: Path where the rendered figure is to be saved. The format of the saved image depends on the
                      the extension present in the parameter value. (Default value: None)
            save_dpi: DPI when saving figure (Default value: 300)
            ax: An instance of Matplotlib's Axes object. This can be used to to plot the figure into an already
                created axes. (Default value: None)
            force_ints_as_cats: Force integer labels in `color_by` as categories. If False, then integer will be
                                treated as continuous variables otherwise as categories. This effects how colormaps
                                are chosen and how legends are rendered. Set this to False if you are large number of
                                unique integer entries (Default: True)
            n_columns: Number of columns in the grid
            w_pad: When plotting in multiple plots in a grid this decides the width padding between the plots.
                   If None is provided the padding will be automatically added to avoid overlap.
                   Ignored if only plotting one scatterplot.
            h_pad: When plotting in multiple plots in a grid this decides the height padding between the plots.
                   If None is provided the padding will be automatically added to avoid overlap.
                   Ignored if only plotting one scatterplot.
            scatter_kwargs: Keyword argument to be passed to matplotlib's scatter command
            shuffle_zorder: Whether to shuffle the plot order of data points in the figure. (Default value: True)
            show_fig: Whether to render the figure and display it using plt.show() (Default value: True)

        Returns:
            None
        """

        from .plots import plot_scatter

        if from_assay is None:
            from_assay = self._defaultAssay
        if layout_key is None:
            raise ValueError(
                "ERROR: Please provide a value for the `layout_key` parameter. This should be same as "
                "that for either `run_unified_umap` or `run_unified_tsne`. Please see the default values "
                "for `label` parameter in those functions if unsure."
            )
        t = self.z[from_assay].projections[layout_key][:]
        attrs = dict(self.z[from_assay].projections[layout_key].attrs)
        t_names = attrs["target_names"]
        ref_n_cells = attrs["n_cells"][0]
        t_n_cells = attrs["n_cells"][1:]
        x = t[:, 0]
        y = t[:, 1]
        df = pd.DataFrame({f"{layout_key}1": x, f"{layout_key}2": y})
        if target_groups is None:
            if color_key is not None:
                temp_raise_error = False
                if ref_name not in color_key:
                    temp_raise_error = True
                for i in t_names:
                    if i not in color_key:
                        temp_raise_error = True
                if temp_raise_error:
                    temp = " ".join(t_names)
                    raise KeyError(
                        f"ERROR: `color_key` must contain these keys: '{ref_name}' and '{temp}'"
                    )
            else:
                import seaborn as sns

                temp_cmap = sns.color_palette("hls", n_colors=len(t_names) + 1).as_hex()
                color_key = {k: v for k, v in zip(t_names, temp_cmap[1:])}
                color_key[ref_name] = temp_cmap[0]
            target_groups = []
            for i, j in zip(t_names, t_n_cells):
                target_groups.extend([i for _ in range(j)])
            target_groups = np.array(target_groups).astype(object)
            mask_values = None
            mask_name = "NA"
        else:
            if len(target_groups) == len(t_names):
                temp = []
                for i in target_groups:
                    temp.extend(list(i))
                target_groups = list(temp)
            color_key = None
            mask_values = [ref_name]
            mask_name = ref_name
            target_groups = np.array(target_groups).astype(object)
        if len(target_groups) != sum(t_n_cells):
            raise ValueError(
                "ERROR: Number of values in `target_groups` should be same as no. of target cells"
            )
        # Turning array to object forces np.NaN to 'nan'
        if any(target_groups == "nan"):
            raise ValueError("ERROR: `target_groups` cannot contain nan values")
        df["vc"] = np.hstack(
            [[ref_name for _ in range(ref_n_cells)], target_groups]
        ).astype(object)
        if show_target_only:
            df = df[ref_n_cells:]
        if shuffle_zorder:
            df = df.sample(frac=1)
        return plot_scatter(
            [df],
            ax,
            width,
            height,
            mask_color,
            cmap,
            color_key,
            mask_values,
            mask_name,
            mask_color,
            point_size,
            ax_label_size,
            frame_offset,
            spine_width,
            spine_color,
            displayed_sides,
            legend_ondata,
            legend_onside,
            legend_size,
            legends_per_col,
            cbar_shrink,
            marker_scale,
            lspacing,
            cspacing,
            savename,
            save_dpi,
            force_ints_as_cats,
            n_columns,
            w_pad,
            h_pad,
            show_fig,
            scatter_kwargs,
        )


# Note for the docstring: Attributes are copied from BaseDataStore docstring since the constructor is inherited.
# Meaning, for any attribute change in BaseDataStore a manual update to docstring here is needed as well. - RO
class DataStore(MappingDatastore):
    """
    This class extends MappingDatastore and consequently inherits methods of all the other DataStore classes.

    This class is the main user facing class as it provides most of the plotting functions.
    It also contains methods for cell filtering, feature selection, marker features identification,
    subsetting and aggregating cells. This class also contains methods that perform in-memory data exports.
    In other words, DataStore objects provide the primary interface to interact with the data.

    Attributes:

    """

    def __init__(
        self,
        zarr_loc: str,
        assay_types: dict = None,
        default_assay: str = None,
        min_features_per_cell: int = 10,
        min_cells_per_feature: int = 20,
        mito_pattern: str = None,
        ribo_pattern: str = None,
        nthreads: int = 2,
        zarr_mode: str = "r+",
        synchronizer=None,
    ):
        """
        Args:
            zarr_loc: Path to Zarr file created using one of writer functions of Scarf.
            assay_types: A dictionary with keys as assay names present in the Zarr file and values as either one of:
                         'RNA', 'ADT', 'ATAC' or 'GeneActivity'.
            default_assay: Name of assay that should be considered as default. It is mandatory to provide this value
                           when DataStore loads a Zarr file for the first time.
            min_features_per_cell: Minimum number of non-zero features in a cell. If lower than this then the cell
                                   will be filtered out.
            min_cells_per_feature: Minimum number of cells where a feature has a non-zero value. Genes with values
                                   less than this will be filtered out.
            mito_pattern: Regex pattern to capture mitochondrial genes. (default: 'MT-')
            ribo_pattern: Regex pattern to capture ribosomal genes. (default: 'RPS|RPL|MRPS|MRPL')
            nthreads: Number of maximum threads to use in all multi-threaded functions
            zarr_mode: For read-write mode use r+' or for read-only use 'r'. (Default value: 'r+')
            synchronizer: Used as `synchronizer` parameter when opening the Zarr file. Please refer to this page for
                          more details: https://zarr.readthedocs.io/en/stable/api/sync.html. By default
                          ThreadSynchronizer will be used.
        """
        if zarr_mode not in ["r", "r+"]:
            raise ValueError(
                "ERROR: Zarr file can only be accessed using either 'r' ot 'r+' mode"
            )
        if synchronizer is None:
            synchronizer = zarr.ThreadSynchronizer()
        super().__init__(
            zarr_loc=zarr_loc,
            assay_types=assay_types,
            default_assay=default_assay,
            min_features_per_cell=min_features_per_cell,
            min_cells_per_feature=min_cells_per_feature,
            mito_pattern=mito_pattern,
            ribo_pattern=ribo_pattern,
            nthreads=nthreads,
            zarr_mode=zarr_mode,
            synchronizer=synchronizer,
        )

    def filter_cells(
        self,
        *,
        attrs: Iterable[str],
        lows: Iterable[int],
        highs: Iterable[int],
        reset_previous: bool = False,
    ) -> None:
        """
        Filter cells based on the cell metadata column values. Filtering triggers `update` method on  'I' column of
        cell metadata which uses 'and' operation. This means that cells that are not within the filtering thresholds
        will have value set as False in 'I' column of cell metadata table. When performing filtering repeatedly, the
        cells that were previously filtered out remain filtered out and 'I' column is updated only for those cells that
        are filtered out due to the latest filtering attempt.

        Args:
            attrs: Names of columns to be used for filtering
            lows: Lower bounds of thresholds for filtering. Should be in same order as the names in `attrs` parameter
            highs: Upper bounds of thresholds for filtering. Should be in same order as the names in `attrs` parameter
            reset_previous: If True, then results of previous filtering will be undone completely.
                            (Default value: False)

        Returns:

        """
        new_bool = np.ones(self.cells.N).astype(bool)
        for i, j, k in zip(attrs, lows, highs):
            # Checking here to avoid hard error from metadata class
            if i not in self.cells.columns:
                logger.warning(
                    f"{i} not found in cell metadata. Will ignore {i} for filtering"
                )
                continue
            if j is None:
                j = -np.Inf
            if k is None:
                k = np.Inf
            x = self.cells.sift(i, j, k)
            logger.info(
                f"{len(x) - x.sum()} cells flagged for filtering out using attribute {i}"
            )
            new_bool = new_bool & x
        if reset_previous:
            self.cells.reset_key(key="I")
        self.cells.update_key(new_bool, key="I")

    def auto_filter_cells(
        self,
        *,
        attrs: Iterable[str] = None,
        min_p: float = 0.01,
        max_p: float = 0.99,
        show_qc_plots: bool = True,
    ) -> None:
        """
        Automatically filter cells based on columns of the cell metadata table.

        This is a wrapper function for `filer_cells` and determines the threshold values to be used for each column.
        For each cell metadata column, the function models a normal distribution using the median value and standard
        deviation of the column and then determines the point estimates of values at `min_p` and `max_p`
        fraction of densities.

        Args:
            attrs: Column names to be used for filtering.
            min_p: Fractional density point to be used for calculating lower bounds of threshold.
            max_p: Fractional density point to be used for calculating lower bounds of threshold.
            show_qc_plots: If True then violin plots with per cell distribution of features will be shown. This does
                       not have an effect if `auto_filter` is False.

        Returns:
            None
        """
        from scipy.stats import norm

        if attrs is None:
            attrs = []
            for i in ["nCounts", "nFeatures", "percentMito", "percentRibo"]:
                i = f"{self._defaultAssay}_{i}"
                if i in self.cells.columns:
                    attrs.append(i)

        attrs_used = []
        for i in attrs:
            if i not in self.cells.columns:
                logger.warning(
                    f"{i} not found in cell metadata. Will ignore {i} for filtering"
                )
                continue
            a = self.cells.fetch_all(i)
            dist = norm(np.median(a), np.std(a))
            self.filter_cells(
                attrs=[i], lows=[dist.ppf(min_p)], highs=[dist.ppf(max_p)]
            )
            attrs_used.append(i)

        if show_qc_plots:
            self.plot_cells_dists(
                cols=attrs_used, sup_title="Pre-filtering distribution"
            )
            self.plot_cells_dists(
                cols=attrs_used,
                cell_key="I",
                color="coral",
                sup_title="Post-filtering distribution",
            )

    def mark_hvgs(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        min_cells: int = None,
        top_n: int = 500,
        min_var: float = -np.Inf,
        max_var: float = np.Inf,
        min_mean: float = -np.Inf,
        max_mean: float = np.Inf,
        n_bins: int = 200,
        lowess_frac: float = 0.1,
        blacklist: str = "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
        show_plot: bool = True,
        hvg_key_name: str = "hvgs",
        **plot_kwargs,
    ) -> None:
        """
        Identify and mark genes as highly variable genes (HVGs). This is a critical and required feature selection step
        and is only applicable to RNAassay type of assays.

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for HVG selection. By default all cells with True value in 'I' will be used.
                      The provided value for `cell_key` should be a column in cell metadata table with boolean values.
            min_cells: Minimum number of cells where a gene should have non-zero expression values for it to be
                       considered a candidate for HVG selection. Large values for this parameter might make it difficult
                       to identify rare populations of cells. Very small values might lead to higher signal to noise
                       ratio in the selected features. By default, a value is set assuming smallest population has no
                       less than 1% of all cells. So for example, if you have 1000 cells (as per cell_key parameter)
                       then `min-cells` will be set to 10.
            top_n: Number of top most variable genes to be set as HVGs. This value is ignored if a value is provided
                   for `min_var` parameter. (Default: 500)
            min_var: Minimum variance threshold for HVG selection. (Default: -Infinity)
            max_var: Maximum variance threshold for HVG selection. (Default: Infinity)
            min_mean: Minimum mean value of expression threshold for HVG selection. (Default: -Infinity)
            max_mean: Maximum mean value of expression threshold for HVG selection. (Default: Infinity)
            n_bins: Number of bins into which the mean expression is binned. (Default: 200)
            lowess_frac: Between 0 and 1. The fraction of the data used when estimating the fit between mean and
                         variance. This is same as `frac` in statsmodels.nonparametric.smoothers_lowess.lowess
                         (Default: 0.1)
            blacklist: This is a regular expression (regex) string that can be used to exclude genes from being marked
                       as HVGs. By default we exclude mitochondrial, ribosomal, some cell-cycle related, histone and
                       HLA genes. (Default: '^MT- | ^RPS | ^RPL | ^MRPS | ^MRPL | ^CCN | ^HLA- | ^H2- | ^HIST' )
            show_plot: If True then a diagnostic scatter plot is shown with HVGs highlighted. (Default: True)
            hvg_key_name: Base label for HVGs in the features metadata column. The value for
                          'cell_key' parameter is prepended to this value. (Default value: 'hvgs')
            plot_kwargs: These named parameters are passed to plotting.plot_mean_var

        Returns:
            None
        """

        if cell_key is None:
            cell_key = "I"
        assay: RNAassay = self._get_assay(from_assay)
        if type(assay) != RNAassay:
            raise TypeError(
                f"ERROR: This method of feature selection can only be applied to RNAassay type of assay. "
                f"The provided assay is {type(assay)} type"
            )
        if min_cells is None:
            min_cells = int(0.01 * self.cells.N)
            logger.info(
                f"Setting `min_cells` to {min_cells}. Only those genes that are present in atleast this number "
                f"of cells will be considered HVGs."
            )
        assay.mark_hvgs(
            cell_key,
            min_cells,
            top_n,
            min_var,
            max_var,
            min_mean,
            max_mean,
            n_bins,
            lowess_frac,
            blacklist,
            hvg_key_name,
            show_plot,
            **plot_kwargs,
        )

    def mark_prevalent_peaks(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        top_n: int = 10000,
        prevalence_key_name: str = "prevalent_peaks",
    ) -> None:
        """
        Feature selection method for ATACassay type assays.

        This method first calculates prevalence of each peak by computing sum of TF-IDF normalized values for each peak
        and then marks `top_n` peaks with highest prevalence as prevalent peaks.

        Args:
            from_assay: Assay to use for graph creation. If no value is provided then `defaultAssay` will be used
            cell_key: Cells to use for selection of most prevalent peaks. By default all cells with True value in
                      'I' will be used. The provided value for `cell_key` should be a column in cell metadata table
                      with boolean values.
            top_n: Number of top prevalent peaks to be selected. This value is ignored if a value is provided
                   for `min_var` parameter. (Default: 500)
            prevalence_key_name: Base label for marking prevalent peaks in the features metadata column. The value for
                                'cell_key' parameter is prepended to this value. (Default value: 'prevalent_peaks')

        Returns:
            None
        """
        if cell_key is None:
            cell_key = "I"
        assay: ATACassay = self._get_assay(from_assay)
        if type(assay) != ATACassay:
            raise TypeError(
                f"ERROR: This method of feature selection can only be applied to ATACassay type of assay. "
                f"The provided assay is {type(assay)} type"
            )
        assay.mark_prevalent_peaks(cell_key, top_n, prevalence_key_name)

    def run_marker_search(
        self,
        *,
        from_assay: str = None,
        group_key: str = None,
        cell_key: str = None,
        threshold: float = 0.25,
        gene_batch_size: int = 50,
    ) -> None:
        """
        Identifies group specific features for a given assay.

        Please check out the ``find_markers_by_rank`` function for further details of how marker features for groups
        are identified. The results are saved into the Zarr hierarchy under `markers` group.

        Args:
            from_assay: Name of the assay to be used. If no value is provided then the default assay will be used.
            group_key: Required parameter. This has to be a column name from cell metadata table. This column dictates
                       how the cells will be grouped. Usually this would be a column denoting cell clusters.
            cell_key: To run the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table. (Default value: 'I')
            threshold: This value dictates how specific the feature value has to be in a group before it is considered a
                       marker for that group. The value has to be greater than 0 but less than or equal to 1
                       (Default value: 0.25)
            gene_batch_size: Number of genes to be loaded in memory at a time. All cells (from ell_key) are loaded for
                             these number of cells at a time.

        Returns:
            None
        """
        from .markers import find_markers_by_rank

        if group_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `group_key`. This should be the name of a column from "
                "cell metadata object that has information on how cells should be grouped."
            )
        if cell_key is None:
            cell_key = "I"
        assay = self._get_assay(from_assay)
        markers = find_markers_by_rank(
            assay, group_key, cell_key, self.nthreads, threshold, gene_batch_size
        )
        z = self.z[assay.name]
        slot_name = f"{cell_key}__{group_key}"
        if "markers" not in z:
            z.create_group("markers")
        group = z["markers"].create_group(slot_name, overwrite=True)
        for i in markers:
            g = group.create_group(i)
            vals = markers[i]
            if len(vals) != 0:
                create_zarr_obj_array(g, "names", list(vals.index))
                g_s = create_zarr_dataset(
                    g, "scores", (10000,), float, vals.values.shape
                )
                g_s[:] = vals.values
        return None

    def get_markers(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        group_key: str = None,
        group_id: Union[str, int] = None,
    ) -> pd.DataFrame:
        """
        Returns a table of markers features obtained through `run_marker_search` for a given group.

        The table contains names of marker features and feature ids are used as table index.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run run the the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table.
            group_key: Required parameter. This has to be a column name from cell metadata table.
                       Usually this would be a column denoting cell clusters. Please use the same value as used
                       when ran `run_marker_search`
            group_id: This is one of the value in `group_key` column of cell metadata.
                      Results are returned for this group

        Returns:
            Pandas dataframe with marker feature names and scores
        """

        if cell_key is None:
            cell_key = "I"
        if group_key is None:
            raise ValueError(
                f"ERROR: Please provide a value for group_key. "
                f"This should be same as used for `run_marker_search`"
            )
        assay = self._get_assay(from_assay)
        try:
            g = assay.z["markers"][f"{cell_key}__{group_key}"]
        except KeyError:
            raise KeyError(
                "ERROR: Couldnt find the location of markers. Please make sure that you have already called "
                "`run_marker_search` method with same value of `cell_key` and `group_key`"
            )
        if group_id is None:
            raise ValueError(
                f"ERROR: Please provide a value for `group_id` parameter. The value can be one of these: "
                f"{list(g.keys())}"
            )
        df = pd.DataFrame(
            [g[group_id]["names"][:], g[group_id]["scores"][:]], index=["ids", "score"]
        ).T.set_index("ids")
        id_idx = assay.feats.get_index_by(df.index, "ids")
        if len(id_idx) != df.shape[0]:
            logger.warning("Internal error in fetching names of the features IDs")
            return df
        df["names"] = assay.feats.fetch_all("names")[id_idx]
        return df

    def export_markers_to_csv(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        group_key: str = None,
        csv_filename: str = None,
    ) -> None:
        """
        Export markers of each cluster/group to a CSV file where each column contains the marker names sorted by
        score (descending order, highest first). This function does not export the scores of markers as they can be
        obtained using `get_markers` function.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: To run run the the test on specific subset of cells, provide the name of a boolean column in
                        the cell metadata table.
            group_key: Required parameter. This has to be a column name from cell metadata table.
                       Usually this would be a column denoting cell clusters. Please use the same value as used
                       when ran `run_marker_search`
            csv_filename: Required parameter. Name, with path, of CSV file where the marker table is to be saved.

        Returns:

        """
        # Not testing the values of from_assay and cell_key because they will be tested in `get_markers`
        if group_key is None:
            raise ValueError(
                f"ERROR: Please provide a value for group_key. "
                f"This should be same as used for `run_marker_search`"
            )
        if csv_filename is None:
            raise ValueError(
                "ERROR: Please provide a value for parameter `csv_filename`"
            )
        clusters = self.cells.fetch(group_key)
        markers_table = {}
        for group_id in sorted(set(clusters)):
            try:
                m = self.get_markers(
                    from_assay=from_assay,
                    cell_key=cell_key,
                    group_key=group_key,
                    group_id=group_id,
                )
                markers_table[group_id] = m["names"].reset_index(drop=True)
            except KeyError:
                markers_table[group_id] = pd.Series([])
        pd.DataFrame(markers_table).fillna("").to_csv(csv_filename, index=False)
        return None

    def run_cell_cycle_scoring(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        s_genes: List[str] = None,
        g2m_genes: List[str] = None,
        n_bins: int = 50,
        rand_seed: int = 4466,
        s_score_label: str = "S_score",
        g2m_score_label: str = "G2M_score",
        phase_label: str = "cell_cycle_phase",
    ):
        """
        Computes S and G2M phase scores by taking into account the average expression of S and G2M phase genes
        respectively. Following steps are taken for each phase:
        - Average expression of all the genes in across `cell_key` cells is calculated
        - The log average expression is divided in `n_bins` bins
        - A control set of genes is identified by sampling genes from same expression bins where phase's genes are
        present.
        - The average expression of phase genes (Ep) and control genes (Ec) is calculated per cell.
        - A phase score is calculated as: Ep-Ec
        Cell cycle phase is assigned to each cell based on following rule set:
        - G1 phase: S score < -1 > G2M sore
        - S phase: S score > G2M score
        - G2M phase: G2M score > S score

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Cell key. Should be same as the one that was used in the desired graph. (Default value: 'I')
            s_genes: A list of S phase genes. If not provided then Scarf loads pre-saved genes accessible at
                     `scarf.bio_data.s_phase_genes`
            g2m_genes: A list of G2M phase genes. If not provided then Scarf loads pre-saved genes accessible at
                     `scarf.bio_data.g2m_phase_genes`
            n_bins: Number of bins into which average expression of genes is divided.
            rand_seed: A random values to set seed while sampling cells from a cluster randomly. (Default value: 4466)
            s_score_label: A base label for saving the S phase scores into a cell metadata column
                           (Default value: 'S_score')
            g2m_score_label: A base label for saving the G2M phase scores into a cell metadata column
                           (Default value: 'G2M_score')
            phase_label: A base label for saving the inferred cell cycle phase into a cell metadata column
                           (Default value: 'cell_cycle_phase')

        Returns:

        """
        if from_assay is None:
            from_assay = self._defaultAssay
        assay = self._get_assay(from_assay)
        if cell_key is None:
            cell_key = "I"
        if s_genes is None:
            from .bio_data import s_phase_genes

            s_genes = list(s_phase_genes)
        if g2m_genes is None:
            from .bio_data import g2m_phase_genes

            g2m_genes = list(g2m_phase_genes)
        control_size = min(len(s_genes), len(g2m_genes))

        s_score = assay.score_features(
            s_genes, cell_key, control_size, n_bins, rand_seed
        )
        s_score_label = self._col_renamer(from_assay, cell_key, s_score_label)
        self.cells.insert(s_score_label, s_score, key=cell_key, overwrite=True)

        g2m_score = assay.score_features(
            g2m_genes, cell_key, control_size, n_bins, rand_seed
        )
        g2m_score_label = self._col_renamer(from_assay, cell_key, g2m_score_label)
        self.cells.insert(g2m_score_label, g2m_score, key=cell_key, overwrite=True)

        phase = pd.Series(["S" for _ in range(self.cells.fetch(cell_key).sum())])
        phase[g2m_score > s_score] = "G2M"
        phase[(g2m_score < 0) & (s_score < 0)] = "G1"
        phase_label = self._col_renamer(from_assay, cell_key, phase_label)
        self.cells.insert(phase_label, phase.values, key=cell_key, overwrite=True)

    def make_bulk(
        self,
        from_assay: str = None,
        group_key: str = None,
        pseudo_reps: int = 3,
        null_vals: list = None,
        random_seed: int = 4466,
    ) -> pd.DataFrame:
        """
        Merge data from cells to create a bulk profile.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            group_key: Name of the column in cell metadata table to be used for grouping cells.
            pseudo_reps: Within each group, cells will randomly be split into `pseudo_reps` partitions. Each partition
                         is considered a pseudo-replicate. (Default value: 3)
            null_vals: Values to be considered as missing values in the `group_key` column. These values will be
            random_seed: A random values to set seed while creating `pseudo_reps` partitions cells randomly.

        Returns:

        """

        def make_reps(v, n_reps: int, seed: int):
            v = list(v)
            np.random.seed(seed)
            shuffled_idx = np.random.choice(v, len(v), replace=False)
            rep_idx = np.array_split(shuffled_idx, n_reps)
            return [sorted(x) for x in rep_idx]

        if pseudo_reps < 1:
            pseudo_reps = 1
        if null_vals is None:
            null_vals = [-1]
        assay = self._get_assay(from_assay)
        if group_key is None:
            raise ValueError("ERROR: Please provide a value for `group_key` parameter")
        groups = self.cells.fetch_all(group_key)

        vals = {}
        for g in tqdmbar(sorted(set(groups))):
            if g in null_vals:
                continue
            rep_indices = make_reps(np.where(groups == g)[0], pseudo_reps, random_seed)
            for n, idx in enumerate(rep_indices):
                vals[f"{g}_Rep{n + 1}"] = controlled_compute(
                    assay.rawData[idx].sum(axis=0), self.nthreads
                )
        vals = pd.DataFrame(vals)
        vals = vals[(vals.sum(axis=1) != 0)]
        vals["names"] = (
            pd.Series(assay.feats.fetch_all("names")).reindex(vals.index).values
        )
        vals.index = pd.Series(assay.feats.fetch_all("ids")).reindex(vals.index).values
        return vals

    def to_anndata(
        self, from_assay: str = None, cell_key: str = None, layers: dict = None
    ):
        """
        Writes an assay of the Zarr hierarchy to AnnData file format.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: Name of column from cell metadata that has boolean values. This is used to subset cells
            layers: A mapping of layer names to assay names. Ex. {'spliced': 'RNA', 'unspliced': 'URNA'}. The raw data
                    from the assays will be stored as sparse arrays in the corresponding layer in anndata.

        Returns: anndata object

        """
        try:

            # noinspection PyPackageRequirements
            from anndata import AnnData
        except ImportError:
            logger.error(
                "Package anndata is not installed because its an optional dependency. "
                "Install via `pip install anndata` or `conda install anndata -c conda-forge`"
            )
            return None

        if cell_key is None:
            cell_key = "I"
        assay = self._get_assay(from_assay)
        df = self.cells.to_pandas_dataframe(self.cells.columns, key=cell_key)
        obs = df.reset_index(drop=True).set_index("ids")
        df = assay.feats.to_pandas_dataframe(assay.feats.columns)
        var = df.set_index("names").rename(columns={"ids": "gene_ids"})
        adata = AnnData(assay.to_raw_sparse(cell_key), obs=obs, var=var)
        if layers is not None:
            for layer, assay_name in layers.items():
                adata.layers[layer] = self._get_assay(assay_name).to_raw_sparse(
                    cell_key
                )
        return adata

    def show_zarr_tree(self, start="/", depth=None) -> None:
        """
        Prints the Zarr hierarchy of the DataStore.

        Args:
            start:
            depth:

        Returns:
            None

        """
        if depth is None:
            print(self.z[start].tree(expand=True))
        else:
            print(self.z[start].tree(expand=True, level=depth))

    def smart_label(
        self,
        to_relabel: str,
        base_label: str,
        cell_key: str = "I",
        new_col_name: Optional[str] = None,
    ) -> Union[None, List[str]]:
        """
        A convenience function to relabel the values in a cell attribure column (A) based on the values
        in another cell attribute column (B). For each unique value in A, the most frequently occuring
        value in B is found. If two or more values in A have maximum overlap with the same value in B, then
        they all get the same label as B along with different suffixes like, 'a', 'b', etc. The suffixes are
        ordered based on where the largest fraction of the B label lies. If one label from A takes up multiple
        labels from B then all the labels from B are included and they are separated by hyphens.

        Args:
            to_relabel: Cell attributes column to relabel
            base_label: Cell attributes column to relabel
            cell_key: Cell key fetching column values
            new_col_name: Name of new column where relabeled values will be saved. If None then values
                          are returned and not saved in cell attributes table

        Returns: None or a list of relabelled values

        """
        df = pd.crosstab(
            self.cells.fetch(base_label, key=cell_key),
            self.cells.fetch(to_relabel, key=cell_key),
        )
        normed_frac = df.divide(df.sum(axis=1), axis="index")
        idxmax = df.idxmax()
        missing_vals = list(set(df.index).difference(idxmax.unique()))
        new_names = {}
        for i in sorted(idxmax.unique()):
            j = normed_frac[idxmax[idxmax == i].index].iloc[i]
            j = j.sort_values(ascending=False).index
            for n, k in enumerate(j, start=1):
                a = chr(ord("@") + n)
                new_names[k] = f"{i}{a.lower()}"
        miss_idxmax = df.iloc[missing_vals].idxmax(axis=1).to_dict()
        for k, v in miss_idxmax.items():
            new_names[v] = f"{new_names[v][:-1]}-{k}{new_names[v][-1]}"

        ret_val = [new_names[x] for x in self.cells.fetch(to_relabel, key=cell_key)]
        if new_col_name is None:
            return ret_val
        else:
            self.cells.insert(new_col_name, ret_val, overwrite=True)

    def plot_cells_dists(
        self,
        from_assay: str = None,
        cols: List[str] = None,
        cell_key: str = None,
        group_key: str = None,
        color: str = "steelblue",
        cmap: str = "tab20",
        fig_size: tuple = None,
        label_size: float = 10.0,
        title_size: float = 10.0,
        sup_title: str = None,
        sup_title_size: float = 12.0,
        scatter_size: float = 1.0,
        max_points: int = 10000,
        show_on_single_row: bool = True,
        show_fig: bool = True,
    ) -> None:
        """
        Makes violin plots of the distribution of values present in cell metadata. This method is designed to
        distribution of nCounts, nFeatures, percentMito and percentRibo cell attrbutes.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cols: Column names from cell metadata table to be used for plotting. Be default, nCounts, nFeatures,
                  percentMito and percentRibo columns are chosen.
            cell_key: One of the columns from cell metadata table that indicates the cells to be used for plotting.
                      The values in the chosen column should be boolean (Default value: 'I')
            group_key: A column name from cell metadata table that indicates how cells should be grouped. This can be
                       any column that has either boolean or categorical values. By default, no grouping will be
                       performed (Default value: None)
            color: Face color of the violin plots. The value can be valid matplotlib named colour. This is used only
                   when there is a single group. (Default value: 'steelblue')
            cmap: A matplotlib colormap to be used to color different groups. (Default value: 'tab20')
            fig_size: A tuple of figure width and figure height (Default value:  Automatically determined by `plot_qc`)
            label_size: The font size of y-axis labels (Default value: 10.0)
            title_size: The font size of title. Median value is printed as title of each violin plot
                        (Default value: 10.0)
            sup_title: The title for complete figure panel (Default value: 12.0 )
            sup_title_size: The font size of title for complete figure panel (Default value: 12.0 )
            scatter_size: Size of each point in the violin plot (Default value: 1.0)
            max_points: Maximum number of points to display over violin plot. Random uniform sampling will be performed
                        to bring down the number of datapoints to this value. This does not effect the violin plot.
                        (Default value: 10000)
            show_on_single_row: Show all subplots in a single row. It might be useful to set this to False if you have
                                too many groups within each subplot (Default value: True)
            show_fig: Whether to render the figure and display it using plt.show() (Default value: True)

        Returns:
            None

        """

        from .plots import plot_qc

        if from_assay is None:
            from_assay = self._defaultAssay
        if cell_key is None:
            # Show all cells
            pass

        if cols is not None:
            if type(cols) != list:
                raise ValueError("ERROR: 'cols' argument must be of type list")
            plot_cols = []
            for i in cols:
                if i in self.cells.columns:
                    if i not in plot_cols:
                        plot_cols.append(i)
                else:
                    logger.warning(f"{i} not found in cell metadata")
        else:
            cols = ["nCounts", "nFeatures", "percentMito", "percentRibo"]
            cols = [f"{from_assay}_{x}" for x in cols]
            plot_cols = [x for x in cols if x in self.cells.columns]

        debug_print_cols = "\n".join(plot_cols)
        logger.debug(
            f"(plot_cells_dists): Will plot following columns: {debug_print_cols}"
        )

        df = self.cells.to_pandas_dataframe(plot_cols)
        if group_key is not None:
            df["groups"] = self.cells.to_pandas_dataframe([group_key])
        else:
            df["groups"] = np.zeros(len(df))
        if cell_key is not None:
            idx = self.cells.active_index(cell_key)
            df = df.reindex(idx)

        plot_qc(
            df,
            color=color,
            cmap=cmap,
            fig_size=fig_size,
            label_size=label_size,
            title_size=title_size,
            sup_title=sup_title,
            sup_title_size=sup_title_size,
            scatter_size=scatter_size,
            max_points=max_points,
            show_on_single_row=show_on_single_row,
            show_fig=show_fig,
        )
        return None

    def plot_layout(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        layout_key: str = None,
        color_by: str = None,
        subselection_key: str = None,
        size_vals=None,
        clip_fraction: float = 0.01,
        width: float = 6,
        height: float = 6,
        default_color: str = "steelblue",
        cmap=None,
        color_key: dict = None,
        mask_values: list = None,
        mask_name: str = "NA",
        mask_color: str = "k",
        point_size: float = 10,
        do_shading: bool = False,
        shade_npixels: int = 1000,
        shade_min_alpha: int = 10,
        spread_pixels: int = 1,
        spread_threshold: float = 0.2,
        ax_label_size: float = 12,
        frame_offset: float = 0.05,
        spine_width: float = 0.5,
        spine_color: str = "k",
        displayed_sides: tuple = ("bottom", "left"),
        legend_ondata: bool = True,
        legend_onside: bool = True,
        legend_size: float = 12,
        legends_per_col: int = 20,
        cbar_shrink: float = 0.6,
        marker_scale: float = 70,
        lspacing: float = 0.1,
        cspacing: float = 1,
        shuffle_df: bool = False,
        sort_values: bool = False,
        savename: str = None,
        save_dpi: int = 300,
        ax=None,
        force_ints_as_cats: bool = True,
        n_columns: int = 4,
        w_pad: float = 1,
        h_pad: float = 1,
        show_fig: bool = True,
        scatter_kwargs: dict = None,
    ):
        """
        Create a scatter plot with a chosen layout. The methods fetches the coordinates based from
        the cell metadata columns with `layout_key` prefix. DataShader library is used to draw fast
        rasterized image is `do_shading` is True. This can be useful when large number of cells are
        present to quickly render the plot and avoid over-plotting.
        The description of shading parameters has mostly been copied from the Datashader API that can be found here:
        https://holoviews.org/_modules/holoviews/operation/datashader.html

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: One of the columns from cell metadata table that indicates the cells to be used.
                      The values in the chosen column should be boolean (Default value: 'I')
            layout_key: A prefix to cell metadata columns that contains the coordinates for the 2D layout of the cells.
                        For example, 'RNA_UMAP' or 'RNA_tSNE'. If a list of prefixes is provided a grid of plots will be
                        made.
            color_by: One (or a list) of the columns of the metadata table or a feature name (for example gene, GATA2).
                      If a list of names is provided a grid of plots will be made.
                      (Default: None)
            subselection_key: A column from cell metadata table to be used to show only a subselection of cells. This
                              key can be used to hide certain cells from a 2D layout. (Default value: None)
            size_vals: An array of values to be used to set sizes of each cell's datapoint in the layout.
                       By default all cells are of same size determined by `point_size` parameter.
                       Has no effect if `do_shading` is True (Default value: None)
            clip_fraction: Same as `clip_fraction` parameter of 'get_cell_vals' method. This value is multiplied by 100
                           and the percentiles are soft-clipped from either end. (Default value: 0)
            width: Figure width (Default value: 6)
            height: Figure height (Default value: 6)
            default_color: A default color for the cells. (Default value: steelblue)
            cmap: A matplotlib colourmap to be used to colour categorical or continuous values plotted on the cells.
                  (Default value: tab20 for categorical variables and cmocean.deep for continuous variables)
            color_key: A custom colour map for cells. These can be used for categorical variables only. The keys in this
                       dictionary should be the category label as present in the `color_by` column and values should be
                       valid matplotlib colour names or hex codes of colours. (Default value: None)
            mask_values: These can a subset of categorical variables that are present in `color_by` which you would like
                         to mask away. These values would be combined under a same label (`mask_name`) and will be given
                         same colour (`mask_color`)
            mask_name: Label to replace the masked value labels. (Default value : None)
            mask_color: Color to be used for masked values. This should be a valid matplotlib named colour or a hexcode
                        of a colour. (Default value: 'k')
            point_size: Size of each scatter point. This is overridden if `size_vals` is provided. Has no effect if
                        `do_shading` is True. (Default value: 10)
            do_shading: Sets shading mode on/off. If shading mode is off (default) then matplotlib's scatter function is
                        is used otherwise a rasterized image is generated using datashader library. Turn this on if you
                        have more than 100K cells to improve render time and also to avoid issues with overplotting.
                        (Default value: False)
            shade_npixels: Number of pixels to rasterize (for both height and width). This controls the resolution of
                           the figure. Adjust this according to the size of the image you want to generate.
                           (Default value: 1000)
            shade_min_alpha: The minimum alpha value to use for non-empty pixels when doing colormapping, in [0, 255].
                             Use a higher value to avoid undersaturation, i.e. poorly visible low-value datapoints, at
                             the expense of the overall dynamic range. (Default value: 10)
            spread_pixels: Maximum number of pixels to spread on all sides (Default value: 1)
            spread_threshold:  When spreading, determines how far to spread. Spreading starts at 1 pixel, and stops
                               when the fraction of adjacent non-empty pixels reaches this threshold. Higher values
                               give more spreading, up to the `spread_pixels` allowed. (Default value: 0.2)
            ax_label_size: Font size for the x and y axis labels. (Default value: 12)
            frame_offset: Extend the x and y axis limits by this fraction (Default value: 0.05)
            spine_width: Line width of the displayed spines (Default value: 0.5)
            spine_color: Colour of the displayed spines.  (Default value: 'k')
            displayed_sides: Determines which figure spines are chosen. The spines to be shown can be supplied as a
                             tuple. The options are: top, bottom, left and right. (Default value: ('bottom', 'left) )
            legend_ondata: Whether to show category labels on the data (scatter points). The position of the label is
                           the centroid of the corresponding values. Has no effect if `color_by` has continuous values.
                           (Default value: True)
            legend_onside: Whether to draw a legend table on the side of the figure. (Default value: True)
            legend_size: Font size of the legend text. (Default value: 12)
            legends_per_col: Number of legends to be used on each legend column. This value determines how many legend
                             legend columns will be drawn (Default value: 20)
            cbar_shrink: Shrinking factor for the width of color bar (Default value: 0.6)
            marker_scale: The relative size of legend markers compared with the originally drawn ones.
                          (Default value: 70)
            lspacing: The vertical space between the legend entries. Measured in font-size units. (Default value: 0.1)
            cspacing: The spacing between columns. Measured in font-size units. (Default value: 1)
            savename: Path where the rendered figure is to be saved. The format of the saved image depends on the
                      the extension present in the parameter value. (Default value: None)
            save_dpi: DPI when saving figure (Default value: 300)
            shuffle_df: Shuffle the order of cells in the plot (Default value: False)
            sort_values: Sort the values before plotting. Setting True will cause the datapoints with
                         (cells) with larger values to be plotted over the ones with lower values.
                         (Default value: False)
            ax: An instance of Matplotlib's Axes object. This can be used to to plot the figure into an already
                created axes. It is ignored if `do_shading` is set to True. (Default value: None)
            force_ints_as_cats: Force integer labels in `color_by` as categories. If False, then integer will be
                                treated as continuous variables otherwise as categories. This effects how colourmaps
                                are chosen and how legends are rendered. Set this to False if you are large number of
                                unique integer entries (Default: True)
            n_columns: If plotting several plots in a grid this argument decides the layout by how many columns in the
                       grid. Defaults to 4 but if the total amount of plots are less than 4 it will default to that
                       number.
            w_pad: When plotting in multiple plots in a grid this decides the width padding between the plots.
                   If None is provided the padding will be automatically added to avoid overlap.
                   Ignored if only plotting one scatterplot.
            h_pad: When plotting in multiple plots in a grid this decides the height padding between the plots.
                   If None is provided the padding will be automatically added to avoid overlap.
                   Ignored if only plotting one scatterplot.
            show_fig: Whether to render the figure and display it using plt.show() (Default value: True)
            scatter_kwargs: Keyword argument to be passed to matplotlib's scatter command

        Returns:
            None

        """

        # TODO: add support for subplots
        # TODO: add support for providing a list of subselections, from_assay and cell_keys
        # TODO: add support for different kinds of point markers
        # TODO: add support for cell zorder randomization

        from .plots import shade_scatter, plot_scatter

        if from_assay is None:
            from_assay = self._defaultAssay
        if cell_key is None:
            cell_key = "I"
        if layout_key is None:
            raise ValueError("Please provide a value for `layout_key` parameter.")
        if clip_fraction >= 0.5:
            raise ValueError(
                "ERROR: clip_fraction cannot be larger than or equal to 0.5"
            )
        if isinstance(layout_key, str):
            layout_key = [layout_key]
        # If a list of layout keys and color_by (e.g. layout_key=['UMAP', 'tSNE'], color_by=['gene1', 'gene2'] the
        # grid layout will be: plot1: UMAP + gene1, plot2: UMAP + gene2, plot3: tSNE + gene1, plot4: tSNE + gene2
        dfs = []
        for lk in layout_key:
            x = self.cells.fetch(f"{lk}1", cell_key)
            y = self.cells.fetch(f"{lk}2", cell_key)
            if color_by is None:
                color_by = ""
            if isinstance(color_by, str):
                color_by = [color_by]
            for c in color_by:
                if c == "":
                    c = "vc"
                    v = np.ones(len(x)).astype(int)
                else:
                    v = self.get_cell_vals(
                        from_assay=from_assay,
                        cell_key=cell_key,
                        k=c,
                        clip_fraction=clip_fraction,
                    )
                df = pd.DataFrame({f"{lk} 1": x, f"{lk} 2": y, c: v})
                if size_vals is not None:
                    if len(size_vals) != len(x):
                        raise ValueError(
                            "ERROR: `size_vals` is not of same size as layout_key"
                        )
                    df["s"] = size_vals
                if subselection_key is not None:
                    idx = self.cells.fetch(subselection_key, cell_key)
                    if idx.dtype != bool:
                        logger.warning(
                            f"`subselection_key` {subselection_key} is not bool type. Will not sub-select"
                        )
                    else:
                        df = df[idx]
                if shuffle_df:
                    df = df.sample(frac=1)
                if sort_values:
                    df = df.sort_values(by=c)
                dfs.append(df)

        if n_columns > len(dfs):
            n_columns = len(dfs)

        if do_shading:
            return shade_scatter(
                dfs,
                ax,
                width,
                shade_npixels,
                spread_pixels,
                spread_threshold,
                shade_min_alpha,
                cmap,
                color_key,
                mask_values,
                mask_name,
                mask_color,
                ax_label_size,
                frame_offset,
                spine_width,
                spine_color,
                displayed_sides,
                legend_ondata,
                legend_onside,
                legend_size,
                legends_per_col,
                cbar_shrink,
                marker_scale,
                lspacing,
                cspacing,
                savename,
                save_dpi,
                force_ints_as_cats,
                n_columns,
                w_pad,
                h_pad,
                show_fig,
            )
        else:
            return plot_scatter(
                dfs,
                ax,
                width,
                height,
                default_color,
                cmap,
                color_key,
                mask_values,
                mask_name,
                mask_color,
                point_size,
                ax_label_size,
                frame_offset,
                spine_width,
                spine_color,
                displayed_sides,
                legend_ondata,
                legend_onside,
                legend_size,
                legends_per_col,
                cbar_shrink,
                marker_scale,
                lspacing,
                cspacing,
                savename,
                save_dpi,
                force_ints_as_cats,
                n_columns,
                w_pad,
                h_pad,
                show_fig,
                scatter_kwargs,
            )

    def plot_cluster_tree(
        self,
        *,
        from_assay: str = None,
        cell_key: str = None,
        feat_key: str = None,
        cluster_key: str = None,
        fill_by_value: str = None,
        force_ints_as_cats: bool = True,
        width: float = 1,
        lvr_factor: float = 0.5,
        vert_gap: float = 0.2,
        min_node_size: float = 10,
        node_size_multiplier: float = 1e4,
        node_power: float = 1.2,
        root_size: float = 100,
        non_leaf_size: float = 10,
        show_labels: bool = True,
        fontsize: float = 10,
        root_color: str = "#C0C0C0",
        non_leaf_color: str = "k",
        cmap="tab20",
        color_key: dict = None,
        edgecolors: str = "k",
        edgewidth: float = 1,
        alpha: float = 0.7,
        figsize=(5, 5),
        ax=None,
        show_fig: bool = True,
        savename: str = None,
        save_dpi: int = 300,
    ):
        """
        Plots a hierarchical layout of the clusters detected using `run_clustering` in a binary tree form. This helps
        evaluate the relationships between the clusters. This figure can complement embeddings likes tSNE where
        global distances are not preserved. The plot shows clusters as coloured nodes and the nodes are sized
        proportionally to the number of cells within the clusters. Root and branching nodes are shown to visually
        track the branching pattern of the tree. This figure is not scaled, i.e. the distances between the nodes are
        meaningless and only the branching pattern of the nodes must be evaluated.

        https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.hierarchy_pos.html

        Args:
            color_key: A custom colour map for cells. These can be used for categorical variables only. The keys in this
                       dictionary should be the category label as present in the `color_by` column and values should be
                       valid matplotlib colour names or hex codes of colours. (Default value: None)
            force_ints_as_cats: Force integer labels in `color_by` as categories. If False, then integer will be
                                treated as continuous variables otherwise as categories. This effects how colourmaps
                                are chosen and how legends are rendered. Set this to False if you are large number of
                                unique integer entries (Default: True)
            fill_by_value: ..
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: One of the columns from cell metadata table that indicates the cells to be used.
                      Should be same as the one that was used in one of the `run_clustering` calls for the given assay.
                      The values in the chosen column should be boolean (Default value: 'I')
            feat_key: Feature key. Should be same as the one that was used in `run_clustering` calls for the
                      given assay. By default the latest used feature for the given assay will be used.
            cluster_key: Should be one of the columns from cell metadata table that contains the output of
                         `run_clustering` method. For example if chosen assay is `RNA` and default value for `label`
                         parameter was used in `run_clustering` then `cluster_key` can be 'RNA_cluster'
            width: Horizontal space allocated for the branches. Larger values may disrupt the hierarchical layout of
                   the cells (Default value: 1)
            lvr_factor: Leaf vs root factor. Controls the relative nodes horizontal spacing between as one moves up or
                        down the tree. Higher values will cause terminal nodes to be more spread out at cost of nodes
                        closer to the root and vice versa. (Default value: 0.5)
            vert_gap: Gap between levels of hierarchy (Default value: 0.2)
            min_node_size: Minimum size of a node (Default value: 10 )
            node_size_multiplier: Size of each leaf node is increased by this factor (Default value: 1e4)
            node_power: The number of cells within each cluster is raised to this value to scale up the node size.
                        (Default value: 1.2)
            root_size: Size of the root node (Default value: 100)
            non_leaf_size: Size of the nodes that represent branch points in the tree (Default value: 10)
            show_labels: Whether to show the cluster labels on the cluster nodes (Default value: True)
            fontsize: Font size of cluster labels. Only used when `do_label` is True (Default value: 10)
            root_color: Colour for root node. Acceptable values are  Matplotlib named colours or hexcodes for colours.
                        (Default value: '#C0C0C0')
            non_leaf_color: Colour for branchpoint nodes. Acceptable values are  Matplotlib named colours or hexcodes
                            for colours. (Default value: 'k')
            cmap: A colormap to be used to colour cluster nodes. Should be one of Matplotlib colourmaps.
                  (Default value: 'tab20')
            edgecolors: Edge colour of circles representing nodes in the hierarchical tree (Default value: 'k)
            edgewidth:  Line width of the edges circles representing nodes in the hierarchical tree  (Default value: 1)
            alpha: Alpha level (Opacity) of the displayed nodes in the figure. (Default value: 0.7)
            figsize: A tuple with describing figure width and height (Default value: (5, 5))
            ax: An instance of Matplotlib's Axes object. This can be used to to plot the figure into an already
                created axes. (Default value: None)
            show_fig: If, False then axes object is returned rather then rendering the plot (Default value: True)
            savename: Path where the rendered figure is to be saved. The format of the saved image depends on the
                      the extension present in the parameter value. (Default value: None)
            save_dpi: DPI when saving figure (Default value: 300)

        Returns:
            None
        """

        from .plots import plot_cluster_hierarchy
        from .dendrogram import CoalesceTree, make_digraph
        from networkx import to_pandas_edgelist, DiGraph

        from_assay, cell_key, feat_key = self._get_latest_keys(
            from_assay, cell_key, feat_key
        )

        if cluster_key is None:
            raise ValueError(
                "ERROR: Please provide a value for `cluster_key` parameter"
            )
        clusts = self.cells.fetch(cluster_key, key=cell_key)
        graph_loc = self._get_latest_graph_loc(from_assay, cell_key, feat_key)
        dendrogram_loc = self.z[graph_loc].attrs["latest_dendrogram"]
        n_clusts = len(set(clusts))
        coalesced_loc = dendrogram_loc + f"_coalesced_{n_clusts}"
        if coalesced_loc in self.z:
            subgraph = DiGraph()
            subgraph.add_edges_from(self.z[coalesced_loc + "/edgelist"][:])
            for i, j in zip(
                self.z[coalesced_loc + "/nodelist"][:],
                self.z[coalesced_loc + "/partition_id"][:],
            ):
                node = int(i[0])
                subgraph.nodes[node]["nleaves"] = int(i[1])
                if j != "-1":
                    subgraph.nodes[node]["partition_id"] = j
        else:
            subgraph = CoalesceTree(make_digraph(self.z[dendrogram_loc][:]), clusts)
            edge_list = to_pandas_edgelist(subgraph).values
            store = create_zarr_dataset(
                self.z, coalesced_loc + "/edgelist", (100000,), "u8", edge_list.shape
            )
            store[:] = edge_list
            node_list = []
            partition_id_list = []
            for i in subgraph.nodes():
                d = subgraph.nodes[i]
                p = d["partition_id"] if "partition_id" in d else -1
                node_list.append((i, d["nleaves"]))
                partition_id_list.append(str(p))

            node_list = np.array(node_list)
            store = create_zarr_dataset(
                self.z,
                coalesced_loc + "/nodelist",
                (100000,),
                node_list.dtype,
                node_list.shape,
            )
            store[:] = node_list

            store = create_zarr_dataset(
                self.z,
                coalesced_loc + "/partition_id",
                (100000,),
                str,
                (len(partition_id_list),),
            )
            store[:] = partition_id_list

        if fill_by_value is not None:
            color_values = self.get_cell_vals(
                from_assay=from_assay, cell_key=cell_key, k=fill_by_value
            )
        else:
            color_values = None
        plot_cluster_hierarchy(
            subgraph,
            clusts,
            color_values,
            force_ints_as_cats=force_ints_as_cats,
            width=width,
            lvr_factor=lvr_factor,
            vert_gap=vert_gap,
            min_node_size=min_node_size,
            node_size_multiplier=node_size_multiplier,
            node_power=node_power,
            root_size=root_size,
            non_leaf_size=non_leaf_size,
            show_labels=show_labels,
            fontsize=fontsize,
            root_color=root_color,
            non_leaf_color=non_leaf_color,
            cmap=cmap,
            color_key=color_key,
            edgecolors=edgecolors,
            edgewidth=edgewidth,
            alpha=alpha,
            figsize=figsize,
            ax=ax,
            show_fig=show_fig,
            savename=savename,
            save_dpi=save_dpi,
        )

    def plot_marker_heatmap(
        self,
        *,
        from_assay: str = None,
        group_key: str = None,
        cell_key: str = None,
        topn: int = 5,
        log_transform: bool = True,
        vmin: float = -1,
        vmax: float = 2,
        savename: str = None,
        save_dpi: int = 300,
        show_fig: bool = True,
        **heatmap_kwargs,
    ):
        """
        Displays a heatmap of top marker gene expression for the chosen groups (usually cell clusters).

        Z-scores are calculated for each marker gene before plotting them. The groups are subjected to hierarchical
        clustering to bring groups with similar expression pattern in proximity.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            group_key: Required parameter. This has to be a column name from cell metadata table. This column dictates
                       how the cells will be grouped. This value should be same as used for `run_marker_search`
            cell_key: One of the columns from cell metadata table that indicates the cells to be used.
                     Should be same as the one that was used in one of the `run_marker_search` calls for the given
                     assay. The values in the chosen column should be boolean (Default value: 'I')
            topn: Number of markers to be displayed for each group in `group_key` column. The markers are sorted based
                  on obtained scores by `run_marker_search`. (Default value: 5)
            log_transform: Whether to log-transform the values before displaying them in the heatmap.
                           (Default value: True)
            vmin: z-scores lower than this value are ceiled to this value. (Default value: -1)
            vmax: z-scores higher than this value are floored to this value. (Default value: 2)
            savename: Path where the rendered figure is to be saved. The format of the saved image depends on the
                      the extension present in the parameter value. (Default value: None)
            save_dpi: DPI when saving figure. (Default value: 300)
            show_fig: Whether to render the figure and display it using plt.show() (Default value: True)
            **heatmap_kwargs: Keyword arguments to be forwarded to seaborn.clustermap.

        Returns:
            None
        """
        from .plots import plot_heatmap

        assay = self._get_assay(from_assay)
        if group_key is None:
            raise ValueError("ERROR: Please provide a value for `group_key`")
        if cell_key is None:
            cell_key = "I"
        if "markers" not in self.z[assay.name]:
            raise KeyError("ERROR: Please run `run_marker_search` first")
        slot_name = f"{cell_key}__{group_key}"
        if slot_name not in self.z[assay.name]["markers"]:
            raise KeyError(
                f"ERROR: Please run `run_marker_search` first with {group_key} as `group_key` and "
                f"{cell_key} as `cell_key`"
            )
        g = self.z[assay.name]["markers"][slot_name]
        goi = []
        for i in g.keys():
            if "names" in g[i]:
                goi.extend(g[i]["names"][:][:topn])
        goi = np.array(sorted(set(goi)))
        cell_idx = np.array(assay.cells.active_index(cell_key))
        feat_idx = np.array(assay.feats.get_index_by(goi, "ids"))
        feat_argsort = np.argsort(feat_idx)
        normed_data = assay.normed(
            cell_idx=cell_idx,
            feat_idx=feat_idx[feat_argsort],
            log_transform=log_transform,
        )
        nc = normed_data.chunks[0]
        # FIXME: avoid conversion to dask dataframe here
        normed_data = normed_data.to_dask_dataframe()
        groups = daskarr.from_array(
            assay.cells.fetch(group_key, cell_key), chunks=nc
        ).to_dask_dataframe()
        df = controlled_compute(normed_data.groupby(groups).mean(), 4)
        df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        df.columns = goi[feat_argsort]
        df = df.T
        df.index = (
            assay.feats.to_pandas_dataframe(["ids", "names"])
            .set_index("ids")
            .reindex(df.index)["names"]
            .values
        )
        # noinspection PyTypeChecker
        df[df < vmin] = vmin
        # noinspection PyTypeChecker
        df[df > vmax] = vmax
        plot_heatmap(
            df,
            savename=savename,
            save_dpi=save_dpi,
            show_fig=show_fig,
            **heatmap_kwargs,
        )
