"""
- Classes:
    - Assay: A generic Assay class that contains methods to calculate feature level statistics.
    - RNAassay: This assay is designed for feature selection and normalization of scRNA-Seq data.
    - ATACassay: This assay is designed for ATAC-Seq data. It uses TF-IDF normalization and
                 performs feature selection by marking most prevalent peaks.
    - ADTassay: This assay is designed for ADT data (surface antibodies) obtained from CITE-Seq
                experiments. It performs CLR normalization of the data but does not have any
                method for feature selection.
"""


import numpy as np
import dask.array as daskarr
import zarr
from .metadata import MetaData
from .utils import show_dask_progress, controlled_compute, logger
from scipy.sparse import csr_matrix, vstack
from typing import Tuple, List
import pandas as pd

__all__ = ["Assay", "RNAassay", "ATACassay", "ADTassay"]


def norm_dummy(_, counts: daskarr) -> daskarr:
    """
    A dummy normalizer. Doesn't perform any normalization.
    This is useful when the 'raw data' is already normalized

    Args:
        _:
        counts: A dask array with 'raw' counts data

    Returns: Dask array

    """
    return counts


def norm_lib_size(assay, counts: daskarr) -> daskarr:
    """
    Performs library size normalization on the data.
    This is the default method for RNA assays

    Args:
        assay: An instance of the assay object
        counts: A dask array with raw counts data

    Returns:  A dask array (delayed matrix) containing normalized data.

    """
    return assay.sf * counts / assay.scalar.reshape(-1, 1)


def norm_lib_size_log(assay, counts: daskarr) -> daskarr:
    """
    Performs library size normalization and then transforms the
    values into log scale.

    Args:
        assay: An instance of the assay object
        counts: A dask array with raw counts data

    Returns: A dask array (delayed matrix) containing normalized data.

    """
    return np.log1p(assay.sf * counts / assay.scalar.reshape(-1, 1))


def norm_clr(_, counts: daskarr) -> daskarr:
    """
    Performs centered log-ratio normalization (ADT).
    This is the default method for ADT assays

    Args:
        _:
        counts: A dask array with raw counts data

    Returns: A dask array (delayed matrix) containing normalized data.

    """
    f = np.exp(np.log1p(counts).sum(axis=0) / len(counts))
    return np.log1p(counts / f)


def norm_tf_idf(assay, counts: daskarr) -> daskarr:
    """
    Performs TF-IDF normalization
    This is the default method for ATAC assays

    Args:
        assay: An instance of the assay object
        counts: A dask array with raw counts data

    Returns: A dask array (delayed matrix) containing normalized data.

    """
    tf = counts / assay.n_term_per_doc.reshape(-1, 1)
    # TODO: Split TF and IDF functionality to make it similar to norml_lib and zscaling
    idf = np.log2(1 + (assay.n_docs / (assay.n_docs_per_term + 1)))
    return tf * idf


class Assay:
    """
    A generic Assay class that contains methods to calculate feature level statistics.
    It also provides a method for saving normalized subset of data for later KNN graph construction.

    Attributes:
        name: A label for the assay instance
        z: Zarr group that contains the assay
        cells: A Metadata class object for cell attributes
        nthreads: number of threads to use for computations
        rawData: dask array containing the raw data
        feats: a MetaData class object for feature attributes
        attrs: Zarr attributes for the zarr group of the assay
        normMethod: normalization method to use.
        sf: scaling factor for doing library-size normalization
    """

    def __init__(
        self,
        z: zarr.group,
        name: str,
        cell_data: MetaData,
        nthreads: int,
        min_cells_per_feature: int = 10,
    ):
        """
        Args:
            z (zarr.Group): Zarr hierarchy where raw data is located
            name (str): A label/name for assay.
            cell_data: Metadata class object for the cell attributes.
            nthreads: number for threads to use for dask parallel computations
            min_cells_per_feature:
        """
        self.name = name
        self.z = z[self.name]
        self.cells = cell_data
        self.nthreads = nthreads
        self.rawData = daskarr.from_zarr(self.z["counts"], inline_array=True)
        self.feats = MetaData(self.z["featureData"])
        self.attrs = self.z.attrs
        if "percentFeatures" not in self.attrs:
            self.attrs["percentFeatures"] = {}
        self.normMethod = norm_dummy
        self.sf = None
        self._ini_feature_props(min_cells_per_feature)

    def normed(
        self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs
    ) -> daskarr:
        """
        This function normalizes the raw and returns a delayed dask array of the normalized
        data.

        Args:
            cell_idx: Indices of cells to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of cell
                      attribute table)
            feat_idx: Indices of features to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of
                      feature attribute table)
            **kwargs:

        Returns: A dask array (delayed matrix) containing normalized data.

        """
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = self.feats.active_index("I")
        counts = self.rawData[:, feat_idx][cell_idx, :]
        return self.normMethod(self, counts)

    def to_raw_sparse(self, cell_key) -> csr_matrix:
        """

        Args:
            cell_key: A column from cell attribute table. This column must be a boolean
                      type. The data will be exported for only those that have a True value
                      in this column.

        Returns: A sparse matrix containing raw data.

        """
        from .utils import tqdmbar

        sm = None
        for i in tqdmbar(
            self.rawData[self.cells.active_index(cell_key), :].blocks,
            total=self.rawData.numblocks[0],
            desc=f"INFO: Converting raw data from {self.name} assay into CSR format",
        ):
            s = csr_matrix(controlled_compute(i, self.nthreads))
            if sm is None:
                sm = s
            else:
                sm = vstack([sm, s])
        return sm

    def _ini_feature_props(self, min_cells: int) -> None:
        """

        Args:
            min_cells: Minimum number of cells per feature. Features below this
                       number are marked invalid.

        Returns:

        """
        if "nCells" in self.feats.columns and "dropOuts" in self.feats.columns:
            pass
        else:
            ncells = show_dask_progress(
                (self.rawData > 0).sum(axis=0),
                f"({self.name}) Computing nCells and dropOuts",
                self.nthreads,
            )
            self.feats.insert("nCells", ncells, overwrite=True)
            self.feats.insert(
                "dropOuts",
                abs(self.cells.N - self.feats.fetch("nCells")),
                overwrite=True,
            )
            self.feats.update_key(ncells > min_cells, "I")

    def add_percent_feature(self, feat_pattern: str, name: str) -> None:
        """

        Args:
            feat_pattern: A regular expression pattern to identify the features of interest
            name: This will be used as the name of column under which the percentages will
                  be saved

        Returns:

        """
        if name in self.attrs["percentFeatures"]:
            if self.attrs["percentFeatures"][name] == feat_pattern:
                return None
            else:
                logger.info(f"Pattern for percentage feature {name} updated.")
        self.attrs["percentFeatures"] = {
            **{k: v for k, v in self.attrs["percentFeatures"].items()},
            **{name: feat_pattern},
        }
        feat_idx = sorted(
            self.feats.get_index_by(self.feats.grep(feat_pattern), "names")
        )
        if len(feat_idx) == 0:
            logger.warning(
                f"No matches found for pattern {feat_pattern}."
                f" Will not add/update percentage feature"
            )
            return None
        total = show_dask_progress(
            self.rawData[:, feat_idx].sum(axis=1),
            f"({self.name}) Computing {name}",
            self.nthreads,
        )
        if total.sum() == 0:
            logger.warning(
                f"Percentage feature {name} not added because not detected in any cell"
            )
            return None
        self.cells.insert(
            name,
            100 * total / self.cells.fetch_all(self.name + "_nCounts"),
            overwrite=True,
        )

    def _verify_keys(self, cell_key: str, feat_key: str) -> None:
        """
        Checks if provided key names are present in cells and feature attribute tables
        and that they are of boolean types.

        Args:
            cell_key: Name of the key (column) from cell attribute table
            feat_key: Name of the key (column) from feature attribute table

        Returns: None

        """
        if cell_key not in self.cells.columns or self.cells.get_dtype(cell_key) != bool:
            raise ValueError(
                f"ERROR: Either {cell_key} does not exist or is not bool type"
            )
        if feat_key not in self.feats.columns or self.feats.get_dtype(feat_key) != bool:
            raise ValueError(
                f"ERROR: Either {feat_key} does not exist or is not bool type"
            )

    def _get_cell_feat_idx(
        self, cell_key: str, feat_key: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Verifies the provided key by calling _verify_keys and fetches the indices of
        rows that have True value in respective column.

        Args:
            cell_key: Name of the key (column) from cell attribute table
            feat_key: Name of the key (column) from feature attribute table

        Returns: A tuple of two numpy arrays corresponding to cell and feature indices
                 respectively.

        """

        self._verify_keys(cell_key, feat_key)
        cell_idx = self.cells.active_index(cell_key)
        feat_idx = self.feats.active_index(feat_key)
        return cell_idx, feat_idx

    @staticmethod
    def _create_subset_hash(cell_idx: np.ndarray, feat_idx: np.ndarray) -> int:
        """
        Takes two index list and hashes them individually and then computes hash of
        the resulting tuple of two hashes.
        The objective of this function is to generate a unique state identifier for
        the cell and feature indices.

        Args:
            cell_idx: Cell row indices
            feat_idx: Feature row indices

        Returns: Returns the final hash

        """
        return hash(tuple([hash(tuple(cell_idx)), hash(tuple(feat_idx))]))

    @staticmethod
    def _get_summary_stats_loc(cell_key: str) -> Tuple[str, str]:
        """
        A convenience method that returns the location of feature-wise summary statistics
        Currently summaries are stored under pattern: summary_stats_{cell_key}

        Args:
            cell_key: Name of the key (column) from cell attribute table

        Returns: A tuple of two strings. First is the text that will be prepended to column
                 names when summary statistics are loaded onto the feature attributes table. The
                 second is the location of the summary statistics group in the zarr hierarchy of
                 the assay.

        """
        return f"stats_{cell_key}", f"summary_stats_{cell_key}"

    def _validate_stats_loc(
        self,
        stats_loc: str,
        cell_idx: np.ndarray,
        feat_idx: np.ndarray,
        delete_on_fail: bool = True,
    ) -> bool:
        """
        Check whether the feature-wise summary statistics was previously calculated on the same
        set of features and cells as preset in the cell_idx and feat_idx parameters.

        Args:
            stats_loc: Location where the feature summary statistics are saved
            cell_idx: The indices of the cell attribute table
            feat_idx: The indices of the feature attribute table
            delete_on_fail: Whether to delete the summary statistics group if the validity check
                            fails (Default: True).

        Returns: True is the validity test passes otherwise False

        """
        subset_hash = self._create_subset_hash(cell_idx, feat_idx)
        if stats_loc in self.z:
            attrs = self.z[stats_loc].attrs
            if "subset_hash" in attrs and attrs["subset_hash"] == subset_hash:
                return True
            else:
                # Reset stats loc
                if delete_on_fail:
                    del self.z[stats_loc]
                return False
        else:
            return False

    def _load_stats_loc(self, cell_key: str) -> str:
        """
        Loads the feature-wise summary statistics calculated on the cells that are True in the
        'cell_key' column.

        Args:
            cell_key: Name of the key (column) from cell attribute table

        Returns: Location of the group group that contains feature-wise summary statistics

        """
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, "I")
        identifier, stats_loc = self._get_summary_stats_loc(cell_key)
        if self._validate_stats_loc(stats_loc, cell_idx, feat_idx) is False:
            raise KeyError(
                f"Summary statistics have not been calculated for cell key: {cell_key}"
            )
        if identifier not in self.feats.locations:
            self.feats.mount_location(self.z[stats_loc], identifier)
        else:
            logger.debug(f"Location ({stats_loc}) already mounted")
        return identifier

    def save_normalized_data(
        self,
        cell_key: str,
        feat_key: str,
        batch_size: int,
        location: str,
        log_transform: bool,
        renormalize_subset: bool,
        update_keys: bool,
    ) -> daskarr:
        """
        Create a new zarr group and saves the normalized data in the group for the selected features
        only.

        Args:
            cell_key: Name of the key (column) from cell attribute table. The data will be saved
                      for only those cells that have a True value in this column.
            feat_key: Name of the key (column) from feature attribute table. The data will be saved
                      for only those features that have a True value in this column
            batch_size: Number of cells to store in a single chunk. Higher values lead to larger
                        memory consumption
            location: Zarr group wherein to save the normalized values
            log_transform: Whether to log transform the values. Is only used if the 'normed' method
                           takes this parameter, ex. RNAassay
            renormalize_subset: Only used if the 'normed' method takes this parameter. Please refer
                                to the documentation of the 'normed' method of the RNAassay for
                                further description of this parameter.
            update_keys: Whether to update the keys. If True then the 'latest_feat_key' and
                         'latest_feat_key' attributes of the assay will be updated. It can be useful
                         to set False in case where you only need to save the normalized data but
                         don't intend to use it directly. For example, when mapping onto a different
                         dataset and aligning features to that dataset.

        Returns: Dask array containing the normalized data

        """

        from .writers import dask_to_zarr

        # FIXME: Extensive documentation needed to justify the naming strategy of slots here
        # Because HVGs and other feature selections have cell key appended in their metadata
        if feat_key != "I":
            feat_key = cell_key + "__" + feat_key
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        subset_hash = self._create_subset_hash(cell_idx, feat_idx)
        subset_params = {
            "log_transform": log_transform,
            "renormalize_subset": renormalize_subset,
        }
        if location in self.z:
            if (
                subset_hash == self.z[location].attrs["subset_hash"]
                and subset_params == self.z[location].attrs["subset_params"]
            ):
                logger.info(
                    f"Using existing normalized data with cell key {cell_key} and feat key {feat_key}"
                )
                if update_keys:
                    self.attrs["latest_feat_key"] = (
                        feat_key.split("__", 1)[1] if feat_key != "I" else "I"
                    )
                    self.attrs["latest_cell_key"] = cell_key
                return daskarr.from_zarr(self.z[location + "/data"], inline_array=True)
            else:
                # Creating group here to overwrite all children
                self.z.create_group(location, overwrite=True)
        vals = self.normed(
            cell_idx,
            feat_idx,
            log_transform=log_transform,
            renormalize_subset=renormalize_subset,
        )
        dask_to_zarr(vals, self.z, location + "/data", batch_size, self.nthreads)
        self.z[location].attrs["subset_hash"] = subset_hash
        self.z[location].attrs["subset_params"] = subset_params
        if update_keys:
            self.attrs["latest_feat_key"] = (
                feat_key.split("__", 1)[1] if feat_key != "I" else "I"
            )
            self.attrs["latest_cell_key"] = cell_key
        return daskarr.from_zarr(self.z[location + "/data"], inline_array=True)

    def score_features(
        self,
        feature_names: List[str],
        cell_key: str,
        ctrl_size: int,
        n_bins: int,
        rand_seed: int,
    ) -> np.ndarray:
        """
        Calculates the scores (mean values) of selection of features over a randomly sampled
        selected feature set in given cells (as marked by cell_key)

        Args:
            feature_names: Names (as in 'names' column of the feature attribute table) of features to
                           be used for scoring
            cell_key: Name of the key (column) from cell attribute table.
            ctrl_size: Number of reference features to be sampled from each bin.
            n_bins: Number of bins for sampling.
            rand_seed: The seed to use for the random number generation.

        Returns: Numpy array of the calculated scores

        """

        from .feat_utils import binned_sampling

        def _names_to_idx(i):
            return self.feats.get_index_by(i, "names", None)

        def _calc_mean(i):
            return (
                self.normed(cell_idx=cell_idx, feat_idx=np.array(sorted(i)))
                .mean(axis=1)
                .compute()
            )

        feature_idx = _names_to_idx(feature_names)
        if len(feature_idx) == 0:
            raise ValueError(
                f"ERROR: No feature ids found for any of the provided {len(feature_names)} features"
            )

        identifier = self._load_stats_loc(cell_key)
        obs_avg = pd.Series(self.feats.fetch_all(f"{identifier}_avg"))
        control_idx = binned_sampling(
            obs_avg, list(feature_idx), ctrl_size, n_bins, rand_seed
        )
        cell_idx, _ = self._get_cell_feat_idx(cell_key, "I")
        return _calc_mean(feature_idx) - _calc_mean(control_idx)

    def __repr__(self):
        f = self.feats.fetch_all("I")
        assay_name = str(self.__class__).split(".")[-1][:-2]
        return f"{assay_name} {self.name} with {f.sum()}({len(f)}) features"


class RNAassay(Assay):
    """
    This subclass of Assay is designed for feature selection and normalization of scRNA-Seq data.
    """

    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, **kwargs):
        """

        Args:
            z (zarr.Group): Zarr hierarchy where raw data is located
            name (str): A label/name for assay.
            cell_data: Metadata class object for the cell attributes.
            **kwargs:
        """
        super().__init__(z, name, cell_data, **kwargs)
        self.normMethod = norm_lib_size
        if "size_factor" in self.attrs:
            self.sf = int(self.attrs["size_factor"])
        else:
            self.sf = 1000
            self.attrs["size_factor"] = self.sf
        self.scalar = None

    def normed(
        self,
        cell_idx: np.ndarray = None,
        feat_idx: np.ndarray = None,
        renormalize_subset: bool = False,
        log_transform: bool = False,
        **kwargs,
    ) -> daskarr:
        """
        This function normalizes the raw and returns a delayed dask array of the normalized
        data. Unlike the `normed` method in the generic Assay class this method is optimized for scRNA-Seq data and
        takes additional parameters that will be used by `norm_lib_size` (default normalization
        method for this class).

        Args:
            cell_idx: Indices of cells to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of cell
                      attribute table)
            feat_idx: Indices of features to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of
                      feature attribute table)
            renormalize_subset: If True, then the data is normalized using only those features that are True in
                                `feat_key` column rather using total expression of all features in a cell
                                 (Default value: False)
            log_transform: If True, then the normalized data is log-transformed (Default value: False).
            **kwargs:

        Returns: A dask array (delayed matrix) containing normalized data.

        """
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = self.feats.active_index("I")
        counts = self.rawData[:, feat_idx][cell_idx, :]
        norm_method_cache = self.normMethod
        if log_transform:
            self.normMethod = norm_lib_size_log
        if renormalize_subset:
            a = show_dask_progress(
                counts.sum(axis=1), "Normalizing with feature subset", self.nthreads
            )
            a[a == 0] = 1
            self.scalar = a
        else:
            self.scalar = self.cells.fetch_all(self.name + "_nCounts")[cell_idx]
        val = self.normMethod(self, counts)
        self.normMethod = norm_method_cache
        return val

    def set_feature_stats(self, cell_key: str, min_cells: int) -> None:
        """
        Calculates summary statistics for the features of the assay using only cells that are marked True by the
        'cell_key' parameter.

        Args:
            cell_key: Name of the key (column) from cell attribute table.
            min_cells: Minimum number of cells across which a given feature should be present. If a feature is present
                       (has non zero un-normalized value) in fewer cells that it is ignored and summary statistics
                       are not calculated for that feature. Also, such features will be disabled and `I` value of these
                       features in the feature attribute table will be set to False

        Returns: None

        """
        feat_key = "I"  # Here we choose to calculate stats for all the features
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        identifier, stats_loc = self._get_summary_stats_loc(cell_key)
        if self._validate_stats_loc(stats_loc, cell_idx, feat_idx) is True:
            logger.info(f"Using cached feature stats for cell_key {cell_key}")
            return None
        n_cells = show_dask_progress(
            (self.normed(cell_idx, feat_idx) > 0).sum(axis=0),
            f"({self.name}) Computing nCells",
            self.nthreads,
        )
        tot = show_dask_progress(
            self.normed(cell_idx, feat_idx).sum(axis=0),
            f"({self.name}) Computing normed_tot",
            self.nthreads,
        )
        sigmas = show_dask_progress(
            self.normed(cell_idx, feat_idx).var(axis=0),
            f"({self.name}) Computing sigmas",
            self.nthreads,
        )
        idx = n_cells > min_cells
        self.feats.update_key(idx, key=feat_key)
        n_cells, tot, sigmas = n_cells[idx], tot[idx], sigmas[idx]

        self.z.create_group(stats_loc, overwrite=True)
        self.feats.mount_location(self.z[stats_loc], identifier)
        self.feats.insert(
            "normed_tot", tot.astype(float), overwrite=True, location=identifier
        )
        self.feats.insert(
            "avg",
            (tot / self.cells.N).astype(float),
            overwrite=True,
            location=identifier,
        )
        self.feats.insert(
            "nz_mean",
            (tot / n_cells).astype(float),
            overwrite=True,
            location=identifier,
        )
        self.feats.insert(
            "sigmas", sigmas.astype(float), overwrite=True, location=identifier
        )
        self.feats.insert(
            "normed_n", n_cells.astype(float), overwrite=True, location=identifier
        )
        self.z[stats_loc].attrs["subset_hash"] = self._create_subset_hash(
            cell_idx, self.feats.active_index(feat_key)
        )
        self.feats.unmount_location(identifier)
        return None

    # maybe we should return plot here? If one wants to modify it. /raz
    def mark_hvgs(
        self,
        cell_key: str,
        min_cells: int,
        top_n: int,
        min_var: float,
        max_var: float,
        min_mean: float,
        max_mean: float,
        n_bins: int,
        lowess_frac: float,
        blacklist: str,
        hvg_key_name: str,
        show_plot: bool,
        **plot_kwargs,
    ) -> None:
        """
        Identifies highly variable genes in the dataset.

        The parameters govern the min/max variance (corrected) and mean expression threshold for calling genes highly
        variable. The variance is corrected by first dividing genes into bins based on their mean expression values.
        Genes with minimum variance is selected from each bin and a Lowess curve is fitted to
        the mean-variance trend of these genes. mark_hvgs will by default run on the default assay.
        See `utils.fit_lowess` for further details.

        *Modifies the feats table*: adds a column named `<cell_key>__hvgs` to the feature table,
        which contains a True value for genes marked HVGs. The prefix comes from the `cell_key` parameter,
        the naming rule in Scarf dictates that cells used to identify HVGs are prepended to the column name
        (with a double underscore delimiter).

        Args:
            cell_key: Specify which cells to use to identify the HVGs. (Default value 'I' use all non-filtered out
                      cells).
            min_cells: Minimum number of cells where a gene should have non-zero expression values for it to be
                       considered a candidate for HVG selection. Large values for this parameter might make it difficult
                       to identify rare populations of cells. Very small values might lead to higher signal to noise
                       ratio in the selected features.
            top_n: Number of top most variable genes to be set as HVGs. This value is ignored if a value is provided
                   for `min_var` parameter.
            min_var: Minimum variance threshold for HVG selection.
            max_var: Maximum variance threshold for HVG selection.
            min_mean: Minimum mean value of expression threshold for HVG selection.
            max_mean: Maximum mean value of expression threshold for HVG selection.
            n_bins: Number of bins into which the mean expression is binned.
            lowess_frac: Between 0 and 1. The fraction of the data used when estimating the fit between mean and
                         variance. This is same as `frac` in statsmodels.nonparametric.smoothers_lowess.lowess
            blacklist: A regular expression string pattern. Gene names matching to this pattern will be excluded from
                       the final highly variable genes list
            hvg_key_name: The label for highly variable genes. This label will be used to mark the HVGs in the
                          feature attribute table. The value for 'cell_key' parameter is prepended to this value.
            show_plot: If True, a plot is produced, that for each gene shows the corrected variance on the y-axis and
                       the non-zero mean (means from cells where the gene had a non-zero value) on the x-axis. The
                       genes are colored in two gradients which indicate the number of cells where the gene was
                       expressed. The colors are yellow to dark red for HVGs, and blue to green for non-HVGs.
            **plot_kwargs: Keyword arguments for matplotlib.pyplot.scatter function
        """

        def col_renamer(x):
            return f"{identifier}_{x}"

        self.set_feature_stats(cell_key, min_cells)
        identifier = self._load_stats_loc(cell_key)
        c_var_col = f"c_var__{n_bins}__{lowess_frac}"
        if col_renamer(c_var_col) in self.feats.columns:
            logger.info("Using existing corrected dispersion values")
        else:
            slots = ["normed_tot", "avg", "nz_mean", "sigmas", "normed_n"]
            for i in slots:
                i = col_renamer(i)
                if i not in self.feats.columns:
                    raise KeyError("ERROR: {i} not found in feature metadata")
            c_var = self.feats.remove_trend(
                col_renamer("avg"), col_renamer("sigmas"), n_bins, lowess_frac
            )
            self.feats.insert(c_var_col, c_var, overwrite=True, location=identifier)

        if max_mean != np.Inf:
            max_mean = 2 ** max_mean
        if max_var != np.Inf:
            max_var = 2 ** max_var
        if min_mean != -np.Inf:
            min_mean = 2 ** min_mean
        if min_var != -np.Inf:
            min_var = 2 ** min_var

        bl = self.feats.index_to_bool(
            self.feats.get_index_by(self.feats.grep(blacklist), "names"), invert=True
        )
        if min_var == -np.Inf:
            if top_n < 1:
                raise ValueError(
                    "ERROR: Please provide a value greater than 0 for `top_n` parameter"
                )
            idx = self.feats.multi_sift(
                [col_renamer("normed_n"), col_renamer("nz_mean")],
                [min_cells, min_mean],
                [np.Inf, max_mean],
            )
            idx = idx & self.feats.fetch_all("I") & bl
            n_valid_feats = idx.sum()
            if top_n > n_valid_feats:
                logger.warning(
                    f"WARNING: Number of valid features are less then value "
                    f"of parameter `top_n`: {top_n}. Resetting `top_n` to {n_valid_feats}"
                )
                top_n = n_valid_feats - 1
            min_var = (
                pd.Series(self.feats.fetch_all(col_renamer(c_var_col)))[idx]
                .sort_values(ascending=False)
                .values[top_n]
            )
        hvgs = self.feats.multi_sift(
            [col_renamer(x) for x in ["normed_n", "nz_mean", c_var_col]],
            [min_cells, min_mean, min_var],
            [np.Inf, max_mean, max_var],
        )
        hvgs = hvgs & self.feats.fetch_all("I") & bl
        hvg_key_name = cell_key + "__" + hvg_key_name
        logger.info(f"{sum(hvgs)} genes marked as HVGs")
        self.feats.insert(hvg_key_name, hvgs, fill_value=False, overwrite=True)

        if show_plot:
            from .plots import plot_mean_var

            nzm, vf, nc = [
                self.feats.fetch(x)
                for x in [col_renamer("nz_mean"), col_renamer(c_var_col), "nCells"]
            ]
            plot_mean_var(nzm, vf, nc, self.feats.fetch(hvg_key_name), **plot_kwargs)

        return None


class ATACassay(Assay):
    """
    This subclass of Assay is designed for feature selection and normalization of scATAC-Seq data.
    """

    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, **kwargs):
        """
        This Assay subclass is designed for feature selection and normalization of scATAC-Seq data

        Args:
            z (zarr.Group): Zarr hierarchy where raw data is located
            name (str): A label/name for assay.
            cell_data: Metadata class object for the cell attributes.
            **kwargs:
        """
        super().__init__(z, name, cell_data, **kwargs)
        self.normMethod = norm_tf_idf
        self.n_term_per_doc = None
        self.n_docs = None
        self.n_docs_per_term = None

    def normed(
        self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs
    ) -> daskarr:
        """
        This function normalizes the raw and returns a delayed dask array of the normalized
        data. Unlike the `normed` method in the generic Assay class this method is optimized for scATAC-Seq data.
        This method uses the the normalization indicated by attribute self.normMethod which by default is set to
        `norm_tf_idf`. The TF-IDF normalization is performed using only the cells and features indicated by the
        'cell_idx' and 'feat_idx' parameters.

        Args:
            cell_idx: Indices of cells to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of cell
                      attribute table)
            feat_idx: Indices of features to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of
                      feature attribute table)
            **kwargs:

        Returns: A dask array (delayed matrix) containing normalized data.

        """
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = self.feats.active_index("I")
        counts = self.rawData[:, feat_idx][cell_idx, :]
        self.n_term_per_doc = self.cells.fetch_all(self.name + "_nFeatures")[cell_idx]
        self.n_docs = len(cell_idx)
        self.n_docs_per_term = self.feats.fetch_all("nCells")[feat_idx]
        return self.normMethod(self, counts)

    def set_feature_stats(self, cell_key: str) -> None:
        """
        Calculates prevalence of each valid feature of the assay using only cells that are marked True by the
        'cell_key' parameter. Prevalence of a feature is the sum of all its TF-IDF normalized values across cells.

        Args:
            cell_key: Name of the key (column) from cell attribute table.

        Returns: None

        """
        feat_key = "I"  # Here we choose to calculate stats for all the features
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        identifier, stats_loc = self._get_summary_stats_loc(cell_key)
        if self._validate_stats_loc(stats_loc, cell_idx, feat_idx) is True:
            logger.info(f"Using cached feature stats for cell_key {cell_key}")
            return None
        prevalence = show_dask_progress(
            self.normed(cell_idx, feat_idx).sum(axis=0),
            f"({self.name}) Calculating peak prevalence across cells",
            self.nthreads,
        )
        self.z.create_group(stats_loc, overwrite=True)
        self.feats.mount_location(self.z[stats_loc], identifier)
        self.feats.insert(
            "prevalence", prevalence.astype(float), overwrite=True, location=identifier
        )
        self.z[stats_loc].attrs["subset_hash"] = self._create_subset_hash(
            cell_idx, feat_idx
        )
        self.feats.unmount_location(identifier)
        return None

    def mark_prevalent_peaks(
        self, cell_key: str, top_n: int, prevalence_key_name: str
    ) -> None:
        """
        Marks `top_n` peaks with highest prevalence as prevalent peaks.

        Args:
           cell_key: Cells to use for selection of most prevalent peaks. The provided value for `cell_key` should be a
                     column in cell attributes table with boolean values.
           top_n: Number of top prevalent peaks to be selected. This value is ignored if a value is provided
                   for `min_var` parameter.
           prevalence_key_name: Base label for marking prevalent peaks in the features attributes column. The value for
                                'cell_key' parameter is prepended to this value.

        Returns: None

        """
        if top_n >= self.feats.N:
            raise ValueError(
                f"ERROR: n_top should be less than total number of features ({self.feats.N})]"
            )
        if type(top_n) != int:
            raise TypeError("ERROR: n_top must a positive integer value")
        self.set_feature_stats(cell_key)
        identifier = self._load_stats_loc(cell_key)
        idx = (
            pd.Series(self.feats.fetch_all(f"{identifier}_prevalence"))
            .sort_values(ascending=False)[:top_n]
            .index
        )
        prevalence_key_name = cell_key + "__" + prevalence_key_name
        self.feats.insert(
            prevalence_key_name,
            self.feats.index_to_bool(idx),
            fill_value=False,
            overwrite=True,
        )
        return None


class ADTassay(Assay):
    """
    This subclass of Assay is designed for normalization of ADT/HTO (feature-barcodes library) data from
    CITE-Seq experiments.
    """

    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, **kwargs):
        """
        This subclass of Assay is designed for normalization of ADT/HTO (feature-barcodes library) data from
        CITE-Seq experiments.

        Args:
            z (zarr.Group): Zarr hierarchy where raw data is located
            name (str): A label/name for assay.
            cell_data: Metadata class object for the cell attributes.
            **kwargs:
        """
        super().__init__(z, name, cell_data, **kwargs)
        self.normMethod = norm_clr

    def normed(
        self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs
    ) -> daskarr:
        """
        This function normalizes the raw and returns a delayed dask array of the normalized
        data. This method uses the the normalization indicated by attribute self.normMethod which by default is set to
        `norm_clr`. The centered log-ratio normalization is performed using only the cells and features indicated by the
        'cell_idx' and 'feat_idx' parameters.

        Args:
            cell_idx: Indices of cells to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of cell
                      attribute table)
            feat_idx: Indices of features to be included in the normalized matrix
                      (Default value: All those marked True in 'I' column of
                      feature attribute table)
            **kwargs:

        Returns: A dask array (delayed matrix) containing normalized data.

        """
        if cell_idx is None:
            cell_idx = self.cells.active_index("I")
        if feat_idx is None:
            feat_idx = self.feats.active_index("I")
        counts = self.rawData[:, feat_idx][cell_idx, :]
        return self.normMethod(self, counts)
