from typing import List, Union, Optional
import numpy as np
import zarr
from loguru import logger
from ..utils import show_dask_progress, controlled_compute
from ..assay import RNAassay, ATACassay, ADTassay, Assay
from ..metadata import MetaData


def sanitize_hierarchy(z: zarr.hierarchy, assay_name: str) -> bool:
    """Test if an assay node in zarr object was created properly.

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
    """This is the base datastore class that deals with loading of assays from
    Zarr files and generating basic cell statistics like nCounts and nFeatures.
    Superclass of the other DataStores.

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

    Attributes:
        cells: MetaData object with cells and info about each cell (e. g. RNA_nCounts ids).
        nthreads: Number of threads to use for this datastore instance.
        z: The Zarr file (directory) used for this datastore instance.
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
        if type(zarr_loc) != str:
            self.z: zarr.hierarchy = zarr.group(zarr_loc, synchronizer=synchronizer)
        else:
            self.z: zarr.hierarchy = zarr.open(
                zarr_loc, mode=zarr_mode, synchronizer=synchronizer
            )
        self.nthreads = nthreads
        # The order is critical here:
        self.cells = self._load_cells()
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
        """This convenience function loads cellData level from the Zarr
        hierarchy.

        Returns:
            Metadata object
        """
        if "cellData" not in self.z:
            raise KeyError("ERROR: cellData not found in zarr file")
        return MetaData(self.z["cellData"])

    @property
    def assay_names(self) -> List[str]:
        """Load all assay names present in the Zarr file. Zarr writers create
        an 'is_assay' attribute in the assay level and this function looks for
        presence of those attributes to load assay names.

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
        """This function sets a given assay name as defaultAssay attribute. If
        `assay_name` value is None then the top-level directory attributes in
        the Zarr file are looked up for presence of previously used default
        assay.

        Args:
            assay_name: Name of the assay to be considered for setting as default.

        Returns:
            Name of the assay to be set as default assay
        """
        if assay_name is None:
            if "defaultAssay" in self.z.attrs:
                assay_name = self.z.attrs["defaultAssay"]
            else:
                if len(self.assay_names) == 1:
                    assay_name = self.assay_names[0]
                    self.z.attrs["defaultAssay"] = assay_name
                else:
                    raise ValueError(
                        "ERROR: You have more than one assay data. "
                        f"Choose one from: {' '.join(self.assay_names)}\n using 'default_assay' parameter. "
                        "Please note that names are case-sensitive."
                    )
        else:
            if assay_name in self.assay_names:
                if "defaultAssay" in self.z.attrs:
                    if assay_name != self.z.attrs["defaultAssay"]:
                        logger.info(
                            f"Default assay changed from {self.z.attrs['defaultAssay']} to {assay_name}"
                        )
                self.z.attrs["defaultAssay"] = assay_name
            else:
                raise ValueError(
                    f"ERROR: The provided default assay name: {assay_name} was not found. "
                    f"Please Choose one from: {' '.join(self.assay_names)}\n"
                    "Please note that the names are case-sensitive."
                )
        return assay_name

    def _load_assays(self, min_cells: int, custom_assay_types: dict = None) -> None:
        """This function loads all the assay names present in attribute
        `assayNames` as Assay objects. An attempt is made to automatically
        determine the most appropriate Assay class for each assay based on
        following mapping:

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
            "HTO": ADTassay,
            "GeneActivity": RNAassay,
            "GeneScores": RNAassay,
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
        for i in self.assay_names:
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
        """This is a convenience function used internally to quickly obtain the
        assay object that is linked to an assay name.

        Args:
            from_assay: Name of the assay whose object is to be returned.

        Returns:
        """
        if from_assay is None or from_assay == "":
            from_assay = self._defaultAssay
        return self.__getattribute__(from_assay)

    def _get_latest_feat_key(self, from_assay: str) -> str:
        """Looks up the value in assay level attributes for key
        'latest_feat_key'.

        Args:
            from_assay: Assay whose latest feature is to be returned.

        Returns:
            Name of the latest feature that was used to run `save_normalized_data`
        """
        assay = self._get_assay(from_assay)
        return assay.attrs["latest_feat_key"]

    def _get_latest_cell_key(self, from_assay: str) -> str:
        """Looks up the value in assay level attributes for key
        'latest_cell_key'.

        Args:
            from_assay: Assay whose latest feature is to be returned.

        Returns:
            Name of the latest feature that was used to run `save_normalized_data`
        """
        assay = self._get_assay(from_assay)
        return assay.attrs["latest_cell_key"]

    def _ini_cell_props(
        self,
        min_features: int,
        mito_pattern: Optional[str],
        ribo_pattern: Optional[str],
    ) -> None:
        """This function is called on class initialization. For each assay, it
        calculates per-cell statistics i.e. nCounts, nFeatures, percentMito and
        percentRibo. These statistics are then populated into the cell metadata
        table.

        Args:
            min_features: Minimum features that a cell must have non-zero value before being filtered out.
            mito_pattern: Regex pattern for identification of mitochondrial genes.
            ribo_pattern: Regex pattern for identification of ribosomal genes.

        Returns:
        """
        for from_assay in self.assay_names:
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
                if mito_pattern == "":
                    pass
                else:
                    if mito_pattern is None:
                        mito_pattern = "MT-|mt"
                    var_name = from_assay + "_percentMito"
                    assay.add_percent_feature(mito_pattern, var_name)

                if ribo_pattern == "":
                    pass
                else:
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
        """A convenience function for internal usage that creates naming rule
        for the metadata columns.

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
        """Override assigning of default assay.

        Args:
            assay_name: Name of the assay that should be set as default.

        Returns:

        Raises:
            ValueError: if `assay_name` is not found in attribute `assayNames`
        """
        if assay_name in self.assay_names:
            self._defaultAssay = assay_name
            self.z.attrs["defaultAssay"] = assay_name
        else:
            raise ValueError(f"ERROR: {assay_name} assay was not found.")

    def get_cell_vals(
        self,
        from_assay: str,
        cell_key: str,
        k: str,
        clip_fraction: float = 0,
        use_precached: bool = True,
    ):
        """Fetches data from the Zarr file.

        This convenience function allows fetching values for cells from either cell metadata table or values of a
        given feature from normalized matrix.

        Args:
            from_assay: Name of assay to be used. If no value is provided then the default assay will be used.
            cell_key: One of the columns from cell metadata table that indicates the cells to be used. The values in
                      the chosen column should be boolean (Default value: 'I')
            k: A cell metadata column or name of a feature.
            clip_fraction: This value is multiplied by 100 and the percentiles are soft-clipped from either end.
                           (Default value: 0)
            use_precached: Whether to use pre-calculated values from 'prenormed' slot. Used only if 'prenormed' is
                           present (Default value: True)

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
            vals = None
            cache_key = "prenormed"
            if use_precached and cache_key in assay.z:
                g = assay.z[cache_key]
                vals = np.zeros(assay.cells.N)
                n_feats = 0
                for i in feat_idx:
                    if i in g:
                        vals += assay.z[cache_key][i][:]
                        n_feats += 1
                if n_feats == 0:
                    logger.debug(f"Could not find prenormed values for feat: {k}")
                    vals = None
                elif n_feats > 1:
                    vals = vals / n_feats
                else:
                    pass
                if vals is not None:
                    vals = vals[cell_idx]
            if vals is None:
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
            f" {len(self.assay_names)} assays: {' '.join(self.assay_names)}"
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
        for i in self.assay_names:
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
