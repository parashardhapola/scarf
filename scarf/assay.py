"""
- Classes:
    - Assay: A generic Assay class that contains methods to calculate feature level statistics.
    - RNAassay: This assay is designed for feature selection and normalization of scRNA-Seq data.
    - ATACassay:
    - ADTassay:
"""
# TODO: add description to docstring

import numpy as np
import dask.array as daskarr
import zarr
from .metadata import MetaData
from .utils import show_progress, controlled_compute
from scipy.sparse import csr_matrix, vstack
from .logging_utils import logger
from typing import Tuple, List
import pandas as pd

__all__ = ['Assay', 'RNAassay', 'ATACassay', 'ADTassay']


def norm_dummy(_, counts: daskarr) -> daskarr:
    return counts


def norm_lib_size(assay, counts: daskarr) -> daskarr:
    return assay.sf * counts / assay.scalar.reshape(-1, 1)


def norm_lib_size_log(assay, counts: daskarr) -> daskarr:
    return np.log1p(assay.sf * counts / assay.scalar.reshape(-1, 1))


def norm_clr(_, counts: daskarr) -> daskarr:
    f = np.exp(np.log1p(counts).sum(axis=0) / len(counts))
    return np.log1p(counts / f)


def norm_tf_idf(assay, counts: daskarr) -> daskarr:
    tf = counts / assay.n_term_per_doc.reshape(-1, 1)
    # TODO: Split TF and IDF functionality to make it similar to norml_lib and zscaling
    idf = np.log2(1 + (assay.n_docs / (assay.n_docs_per_term + 1)))
    return tf * idf


class Assay:
    """
    A generic Assay class that contains methods to calculate feature level statistics.

    It also provides a method for saving normalized subset of data for later KNN graph construction.

    Attributes:
        name:
        z:
        cells:
        nthreads:
        rawData:
        feats: a MetaData object with info about each feature in the dataset
        attrs:
        normMethod: Which normalization method to use.
        sf: scaling factor for doing library-size normalization
    """
    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData,
                 nthreads: int, min_cells_per_feature: int = 10):
        """
        Args:
            z (zarr.hierarchy): Zarr hierarchy to use.
            name (str): Name for assay.
            cell_data: Metadata for the cells.
            nthreads:
            min_cells_per_feature:
        """
        self.name = name
        self.z = z[self.name]
        self.cells = cell_data
        self.nthreads = nthreads
        self.rawData = daskarr.from_zarr(self.z['counts'], inline_array=True)
        self.feats = MetaData(self.z['featureData'])
        self.attrs = self.z.attrs
        if 'percentFeatures' not in self.attrs:
            self.attrs['percentFeatures'] = {}
        self.normMethod = norm_dummy
        self.sf = None
        self._ini_feature_props(min_cells_per_feature)

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs):
        """

        Args:
            cell_idx:
            feat_idx:
            **kwargs:

        Returns:

        """
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        counts = self.rawData[:, feat_idx][cell_idx, :]
        return self.normMethod(self, counts)

    def to_raw_sparse(self, cell_key):
        """

        Args:
            cell_key:

        Returns:

        """
        from tqdm import tqdm

        sm = None
        for i in tqdm(self.rawData[self.cells.active_index(cell_key), :].blocks, total=self.rawData.numblocks[0],
                      desc=f"INFO: Converting raw data from {self.name} assay into CSR format"):
            s = csr_matrix(controlled_compute(i, self.nthreads))
            if sm is None:
                sm = s
            else:
                sm = vstack([sm, s])
        return sm

    def _ini_feature_props(self, min_cells: int) -> None:
        """

        Args:
            min_cells:

        Returns:

        """
        if 'nCells' in self.feats.columns and 'dropOuts' in self.feats.columns:
            pass
        else:
            ncells = show_progress((self.rawData > 0).sum(axis=0),
                                   f"({self.name}) Computing nCells and dropOuts", self.nthreads)
            self.feats.insert('nCells', ncells, overwrite=True)
            self.feats.insert('dropOuts', abs(self.cells.N - self.feats.fetch('nCells')), overwrite=True)
            self.feats.update_key(ncells > min_cells, 'I')

    def add_percent_feature(self, feat_pattern: str, name: str) -> None:
        """

        Args:
            feat_pattern:
            name:

        Returns:

        """
        if name in self.attrs['percentFeatures']:
            if self.attrs['percentFeatures'][name] == feat_pattern:
                return None
            else:
                logger.info(f"Pattern for percentage feature {name} updated.")
        self.attrs['percentFeatures'] = {**{k: v for k, v in self.attrs['percentFeatures'].items()},
                                         **{name: feat_pattern}}
        feat_idx = sorted(self.feats.get_index_by(self.feats.grep(feat_pattern), 'names'))
        if len(feat_idx) == 0:
            logger.warning(f"No matches found for pattern {feat_pattern}."
                           f" Will not add/update percentage feature")
            return None
        total = show_progress(self.rawData[:, feat_idx].sum(axis=1),
                              f"Computing percentage of {name}", self.nthreads)
        if total.sum() == 0:
            logger.warning(f"Percentage feature {name} not added because not detected in any cell")
            return None
        self.cells.insert(name, 100 * total / self.cells.fetch_all(self.name + '_nCounts'), overwrite=True)

    def _verify_keys(self, cell_key: str, feat_key: str) -> None:
        if cell_key not in self.cells.columns or self.cells.get_dtype(cell_key) != bool:
            raise ValueError(f"ERROR: Either {cell_key} does not exist or is not bool type")
        if feat_key not in self.feats.columns or self.feats.get_dtype(feat_key) != bool:
            raise ValueError(f"ERROR: Either {feat_key} does not exist or is not bool type")

    def _get_cell_feat_idx(self, cell_key: str, feat_key: str) -> Tuple[np.ndarray, np.ndarray]:
        self._verify_keys(cell_key, feat_key)
        cell_idx = self.cells.active_index(cell_key)
        feat_idx = self.feats.active_index(feat_key)
        return cell_idx, feat_idx

    @staticmethod
    def _create_subset_hash(cell_idx: np.ndarray, feat_idx: np.ndarray) -> int:
        return hash(tuple([hash(tuple(cell_idx)), hash(tuple(feat_idx))]))

    @staticmethod
    def _get_summary_stats_loc(cell_key: str) -> Tuple[str, str]:
        return f"stats_{cell_key}", f"summary_stats_{cell_key}"

    def _validate_stats_loc(self, stats_loc: str, cell_idx: np.ndarray,
                            feat_idx: np.ndarray, delete_on_fail: bool = True) -> bool:
        subset_hash = self._create_subset_hash(cell_idx, feat_idx)
        if stats_loc in self.z:
            attrs = self.z[stats_loc].attrs
            if 'subset_hash' in attrs and attrs['subset_hash'] == subset_hash:
                return True
            else:
                # Reset stats loc
                if delete_on_fail:
                    del self.z[stats_loc]
                return False
        else:
            return False

    def _load_stats_loc(self, cell_key: str) -> str:
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, 'I')
        identifier, stats_loc = self._get_summary_stats_loc(cell_key)
        if self._validate_stats_loc(stats_loc, cell_idx, feat_idx) is False:
            raise KeyError(f"Summary statistics have not been calculated for cell key: {cell_key}")
        if identifier not in self.feats.locations:
            self.feats.mount_location(self.z[stats_loc], identifier)
        else:
            logger.debug(f"Location ({stats_loc}) already mounted")
        return identifier

    def save_normalized_data(self, cell_key: str, feat_key: str, batch_size: int,
                             location: str, log_transform: bool, renormalize_subset: bool,
                             update_keys: bool) -> daskarr:
        """

        Args:
            cell_key:
            feat_key:
            batch_size:
            location:
            log_transform:
            renormalize_subset:
            update_keys:

        Returns:

        """

        from .writers import dask_to_zarr

        # FIXME: Extensive documentation needed to justify the naming strategy of slots here
        # Because HVGs and other feature selections have cell key appended in their metadata
        if feat_key != 'I':
            feat_key = cell_key + '__' + feat_key
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        subset_hash = self._create_subset_hash(cell_idx, feat_idx)
        subset_params = {'log_transform': log_transform, 'renormalize_subset': renormalize_subset}
        if location in self.z:
            if subset_hash == self.z[location].attrs['subset_hash'] and \
                    subset_params == self.z[location].attrs['subset_params']:
                logger.info(f"Using existing normalized data with cell key {cell_key} and feat key {feat_key}")
                if update_keys:
                    self.attrs['latest_feat_key'] = feat_key.split('__', 1)[1] if feat_key != 'I' else 'I'
                    self.attrs['latest_cell_key'] = cell_key
                return daskarr.from_zarr(self.z[location + '/data'], inline_array=True)
            else:
                # Creating group here to overwrite all children
                self.z.create_group(location, overwrite=True)
        vals = self.normed(cell_idx, feat_idx, log_transform=log_transform,
                           renormalize_subset=renormalize_subset)
        dask_to_zarr(vals, self.z, location + '/data', batch_size, self.nthreads)
        self.z[location].attrs['subset_hash'] = subset_hash
        self.z[location].attrs['subset_params'] = subset_params
        if update_keys:
            self.attrs['latest_feat_key'] = feat_key.split('__', 1)[1] if feat_key != 'I' else 'I'
            self.attrs['latest_cell_key'] = cell_key
        return daskarr.from_zarr(self.z[location + '/data'], inline_array=True)

    def score_features(self, feature_names: List[str], cell_key: str,
                       ctrl_size: int, n_bins: int, rand_seed: int) -> np.ndarray:
        """

        Args:
            feature_names:
            cell_key:
            ctrl_size:
            n_bins:
            rand_seed:

        Returns:

        """

        from .feat_utils import binned_sampling

        def _names_to_idx(i):
            return self.feats.get_index_by(i, 'names', None)

        def _calc_mean(i):
            return self.normed(cell_idx=cell_idx, feat_idx=np.array(sorted(i))).mean(axis=1).compute()

        feature_idx = _names_to_idx(feature_names)
        if len(feature_idx) == 0:
            raise ValueError(f"ERROR: No feature ids found for any of the provided {len(feature_names)} features")

        identifier = self._load_stats_loc(cell_key)
        obs_avg = pd.Series(self.feats.fetch_all(f"{identifier}_avg"))
        control_idx = binned_sampling(obs_avg, list(feature_idx), ctrl_size, n_bins, rand_seed)
        cell_idx, _ = self._get_cell_feat_idx(cell_key, 'I')
        return _calc_mean(feature_idx) - _calc_mean(control_idx)

    def __repr__(self):
        f = self.feats.fetch_all('I')
        assay_name = str(self.__class__).split('.')[-1][:-2]
        return f"{assay_name} {self.name} with {f.sum()}({len(f)}) features"


class RNAassay(Assay):
    """
    This assay is designed for feature selection and normalization of scRNA-Seq data.

    Subclass of Assay.
    """
    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, **kwargs):
        """

        Args:
            z:
            name:
            cell_data:
            **kwargs:
        """
        super().__init__(z, name, cell_data, **kwargs)
        self.normMethod = norm_lib_size
        if 'size_factor' in self.attrs:
            self.sf = int(self.attrs['size_factor'])
        else:
            self.sf = 1000
            self.attrs['size_factor'] = self.sf
        self.scalar = None

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None,
               renormalize_subset: bool = False, log_transform: bool = False, **kwargs):
        """
        Args:
            cell_idx:
            feat_idx:
            renormalize_subset:
            log_transform:
            **kwargs:

        Returns:

        """
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        counts = self.rawData[:, feat_idx][cell_idx, :]
        norm_method_cache = self.normMethod
        if log_transform:
            self.normMethod = norm_lib_size_log
        if renormalize_subset:
            a = show_progress(counts.sum(axis=1), "Normalizing with feature subset", self.nthreads)
            a[a == 0] = 1
            self.scalar = a
        else:
            self.scalar = self.cells.fetch_all(self.name + '_nCounts')[cell_idx]
        val = self.normMethod(self, counts)
        self.normMethod = norm_method_cache
        return val

    def set_feature_stats(self, cell_key: str, min_cells: int) -> None:
        """

        Args:
            cell_key:
            min_cells:

        Returns:

        """
        feat_key = 'I'  # Here we choose to calculate stats for all the features
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        identifier, stats_loc = self._get_summary_stats_loc(cell_key)
        if self._validate_stats_loc(stats_loc, cell_idx, feat_idx) is True:
            logger.info(f"Using cached feature stats for cell_key {cell_key}")
            return None
        n_cells = show_progress((self.normed(cell_idx, feat_idx) > 0).sum(axis=0),
                                f"({self.name}) Computing nCells", self.nthreads)
        tot = show_progress(self.normed(cell_idx, feat_idx).sum(axis=0),
                            f"({self.name}) Computing normed_tot", self.nthreads)
        sigmas = show_progress(self.normed(cell_idx, feat_idx).var(axis=0),
                               f"({self.name}) Computing sigmas", self.nthreads)
        idx = n_cells > min_cells
        self.feats.update_key(idx, key=feat_key)
        n_cells, tot, sigmas = n_cells[idx], tot[idx], sigmas[idx]

        self.z.create_group(stats_loc, overwrite=True)
        self.feats.mount_location(self.z[stats_loc], identifier)
        self.feats.insert('normed_tot', tot.astype(float), overwrite=True, location=identifier)
        self.feats.insert('avg', (tot / self.cells.N).astype(float), overwrite=True, location=identifier)
        self.feats.insert('nz_mean', (tot / n_cells).astype(float), overwrite=True, location=identifier)
        self.feats.insert('sigmas', sigmas.astype(float), overwrite=True, location=identifier)
        self.feats.insert('normed_n', n_cells.astype(float), overwrite=True, location=identifier)
        self.z[stats_loc].attrs['subset_hash'] = self._create_subset_hash(
            cell_idx, self.feats.active_index(feat_key))
        self.feats.unmount_location(identifier)
        return None

    # maybe we should return plot here? If one wants to modify it. /raz
    def mark_hvgs(self, cell_key: str, min_cells: int, top_n: int,
                  min_var: float, max_var: float, min_mean: float, max_mean: float,
                  n_bins: int, lowess_frac: float, blacklist: str, hvg_key_name: str,
                  show_plot: bool, **plot_kwargs) -> None:
        """
        Identifies highly variable genes in the dataset.
        
        The parameters govern the min/max variance (corrected) and mean expression threshold for calling genes highly
        variable. The variance is corrected by first dividing genes into bins based on their mean expression values.
        Genes with minimum variance is selected from each bin and a Lowess curve is fitted to
        the mean-variance trend of these genes. mark_hvgs will by default run on the default assay.

        A plot is produced, that for each gene shows the corrected variance on the y-axis and the non-zero mean
        (means from cells where the gene had a non-zero value) on the x-axis. The genes are colored in two gradients
        which indicate the number of cells where the gene was expressed. The colors are yellow to dark red for HVGs,
        and blue to green for non-HVGs.

        The mark_hvgs function has a parameter cell_key that dictates which cells to use to identify the HVGs.
        The default value of this parameter is I, which means it will use all the cells that were not filtered out.
        
        *Modifies the feats table*: adds a column named `<cell_key>__hvgs` to the feature table,
        which contains a True value for genes marked HVGs. The prefix comes from the `cell_key` parameter,
        the naming rule in Scarf dictates that cells used to identify HVGs are prepended to the column name
        (with a double underscore delimiter).

        Args:
            cell_key: Specify which cells to use to identify the HVGs. Default value `I` (use all non-filtered out cells).
            min_cells:
            top_n:
            min_var:
            max_var:
            min_mean:
            max_mean:
            n_bins: Number of bins
            lowess_frac:
            blacklist:
            hvg_key_name:
            show_plot:
            **plot_kwargs:
        """
        self.set_feature_stats(cell_key, min_cells)
        identifier = self._load_stats_loc(cell_key)
        col_renamer = lambda x: f"{identifier}_{x}"
        c_var_col = f"c_var__{n_bins}__{lowess_frac}"
        if col_renamer(c_var_col) in self.feats.columns:
            logger.info("Using existing corrected dispersion values")
        else:
            slots = ['normed_tot', 'avg', 'nz_mean', 'sigmas', 'normed_n']
            for i in slots:
                i = col_renamer(i)
                if i not in self.feats.columns:
                    raise KeyError("ERROR: {i} not found in feature metadata")
            c_var = self.feats.remove_trend(col_renamer('avg'), col_renamer('sigmas'),
                                            n_bins, lowess_frac)
            self.feats.insert(c_var_col, c_var, overwrite=True, location=identifier)

        if max_mean != np.Inf:
            max_mean = 2 ** max_mean
        if max_var != np.Inf:
            max_var = 2 ** max_var
        if min_mean != -np.Inf:
            min_mean = 2 ** min_mean
        if min_var != -np.Inf:
            min_var = 2 ** min_var

        bl = self.feats.index_to_bool(self.feats.get_index_by(self.feats.grep(blacklist), 'names'), invert=True)
        if min_var == -np.Inf:
            if top_n < 1:
                raise ValueError("ERROR: Please provide a value greater than 0 for `top_n` parameter")
            idx = self.feats.multi_sift(
                [col_renamer('normed_n'), col_renamer('nz_mean')], [min_cells, min_mean], [np.Inf, max_mean])
            idx = idx & self.feats.fetch_all('I') & bl
            n_valid_feats = idx.sum()
            if top_n > n_valid_feats:
                logger.warning(f"WARNING: Number of valid features are less then value "
                               f"of parameter `top_n`: {top_n}. Resetting `top_n` to {n_valid_feats}")
                top_n = n_valid_feats - 1
            min_var = pd.Series(self.feats.fetch_all(col_renamer(c_var_col))
                                )[idx].sort_values(ascending=False).values[top_n]
        hvgs = self.feats.multi_sift(
            [col_renamer(x) for x in ['normed_n', 'nz_mean', c_var_col]],
            [min_cells, min_mean, min_var], [np.Inf, max_mean, max_var])
        hvgs = hvgs & self.feats.fetch_all('I') & bl
        hvg_key_name = cell_key + '__' + hvg_key_name
        logger.info(f"{sum(hvgs)} genes marked as HVGs")
        self.feats.insert(hvg_key_name, hvgs, fill_value=False, overwrite=True)

        if show_plot:
            from .plots import plot_mean_var
            nzm, vf, nc = [self.feats.fetch(x) for x in [col_renamer('nz_mean'), col_renamer(c_var_col), 'nCells']]
            plot_mean_var(nzm, vf, nc, self.feats.fetch(hvg_key_name), **plot_kwargs)

        return None


class ATACassay(Assay):
    # TODO: add docstring
    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, **kwargs):
        """

        Args:
            z:
            name:
            cell_data:
            **kwargs:
        """
        super().__init__(z, name, cell_data, **kwargs)
        self.normMethod = norm_tf_idf
        self.n_term_per_doc = None
        self.n_docs = None
        self.n_docs_per_term = None

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs):
        """

        Args:
            cell_idx:
            feat_idx:
            **kwargs:

        Returns:

        """
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        counts = self.rawData[:, feat_idx][cell_idx, :]
        self.n_term_per_doc = self.cells.fetch_all(self.name + '_nFeatures')[cell_idx]
        self.n_docs = len(cell_idx)
        self.n_docs_per_term = self.feats.fetch_all('nCells')[feat_idx]
        return self.normMethod(self, counts)

    def set_feature_stats(self, cell_key: str) -> None:
        """

        Args:
            cell_key:

        Returns:

        """
        feat_key = 'I'  # Here we choose to calculate stats for all the features
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        identifier, stats_loc = self._get_summary_stats_loc(cell_key)
        if self._validate_stats_loc(stats_loc, cell_idx, feat_idx) is True:
            logger.info(f"Using cached feature stats for cell_key {cell_key}")
            return None
        prevalence = show_progress(self.normed(cell_idx, feat_idx).sum(axis=0),
                                   f"({self.name}) Calculating peak prevalence across cells", self.nthreads)
        self.z.create_group(stats_loc, overwrite=True)
        self.feats.mount_location(self.z[stats_loc], identifier)
        self.feats.insert('prevalence', prevalence.astype(float), overwrite=True, location=identifier)
        self.z[stats_loc].attrs['subset_hash'] = self._create_subset_hash(cell_idx, feat_idx)
        self.feats.unmount_location(identifier)
        return None

    def mark_prevalent_peaks(self, cell_key: str, top_n: int, prevalence_key_name: str) -> None:
        """

        Args:
            cell_key:
            top_n:
            prevalence_key_name:

        Returns:

        """
        if top_n >= self.feats.N:
            raise ValueError(f"ERROR: n_top should be less than total number of features ({self.feats.N})]")
        if type(top_n) != int:
            raise TypeError("ERROR: n_top must a positive integer value")
        self.set_feature_stats(cell_key)
        identifier = self._load_stats_loc(cell_key)
        col_renamer = lambda x: f"{identifier}_{x}"
        idx = pd.Series(self.feats.fetch_all(col_renamer('prevalence'))).sort_values(ascending=False)[:top_n].index
        prevalence_key_name = cell_key + '__' + prevalence_key_name
        self.feats.insert(prevalence_key_name, self.feats.index_to_bool(idx), fill_value=False, overwrite=True)
        return None


class ADTassay(Assay):
    # TODO: add docstring
    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, **kwargs):
        """

        Args:
            z:
            name:
            cell_data:
            **kwargs:
        """
        super().__init__(z, name, cell_data, **kwargs)
        self.normMethod = norm_clr

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs):
        """

        Args:
            cell_idx:
            feat_idx:
            **kwargs:

        Returns:

        """
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        counts = self.rawData[:, feat_idx][cell_idx, :]
        return self.normMethod(self, counts)
