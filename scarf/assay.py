import numpy as np
import dask.array as daskarr
import zarr
from .metadata import MetaData
from .utils import show_progress, controlled_compute
from .writers import create_zarr_dataset
from scipy.sparse import csr_matrix, vstack
from .logging_utils import logger
from typing import Tuple, Union, List

__all__ = ['Assay', 'RNAassay', 'ATACassay', 'ADTassay']


def norm_dummy(assay, counts: daskarr) -> daskarr:
    return counts


def norm_lib_size(assay, counts: daskarr) -> daskarr:
    return assay.sf * counts / assay.scalar.reshape(-1, 1)


def norm_lib_size_log(assay, counts: daskarr) -> daskarr:
    return np.log1p(assay.sf * counts / assay.scalar.reshape(-1, 1))


def norm_clr(assay, counts: daskarr) -> daskarr:
    f = np.exp(np.log1p(counts).sum(axis=0) / len(counts))
    return np.log1p(counts / f)


def norm_tf_idf(assay, counts: daskarr) -> daskarr:
    tf = counts / assay.n_term_per_doc.reshape(-1, 1)
    # TODO: Split TF and IDF functionality to make it similar to norml_lib and zscaling
    idf = np.log2(assay.n_docs / (assay.n_docs_per_term + 1))
    return tf * idf


class Assay:
    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData,
                 nthreads: int, min_cells_per_feature: int = 10):
        self.name = name
        self.z = z[self.name]
        self.cells = cell_data
        self.nthreads = nthreads
        self.rawData = daskarr.from_zarr(self.z['counts'])
        self.feats = MetaData(self.z['featureData'])
        self.attrs = self.z.attrs
        if 'percentFeatures' not in self.attrs:
            self.attrs['percentFeatures'] = {}
        self.normMethod = norm_dummy
        self.sf = None
        self._ini_feature_props(min_cells_per_feature)

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs):
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        counts = self.rawData[:, feat_idx][cell_idx, :]
        return self.normMethod(self, counts)

    def to_raw_sparse(self, cell_key):
        from tqdm import tqdm

        sm = None
        for i in tqdm(self.rawData[self.cells.active_index(cell_key), :].blocks, total=self.rawData.numblocks[0],
                      desc="INFO: Converting raw data from {self.name} assay into CSR format"):
            s = csr_matrix(controlled_compute(i, self.nthreads))
            if sm is None:
                sm = s
            else:
                sm = vstack([sm, s])
        return sm

    def _ini_feature_props(self, min_cells: int) -> None:
        if 'nCells' in self.feats.table.columns and 'dropOuts' in self.feats.table.columns:
            pass
        else:
            ncells = show_progress((self.rawData > 0).sum(axis=0),
                                   f"({self.name}) Computing nCells and dropOuts", self.nthreads)
            self.feats.add('nCells', ncells, overwrite=True)
            self.feats.add('dropOuts', abs(self.cells.N - self.feats.fetch('nCells')), overwrite=True)
            self.feats.update(ncells > min_cells)

    def add_percent_feature(self, feat_pattern: str, name: str) -> None:
        if name in self.attrs['percentFeatures']:
            if self.attrs['percentFeatures'][name] == feat_pattern:
                return None
            else:
                logger.info(f"Pattern for percentage feature {name} updated.")
        self.attrs['percentFeatures'] = {**{k: v for k, v in self.attrs['percentFeatures'].items()},
                                         **{name: feat_pattern}}
        feat_idx = sorted(self.feats.get_idx_by_names(self.feats.grep(feat_pattern)))
        if len(feat_idx) == 0:
            logger.warning(f"No matches found for pattern {feat_pattern}."
                           f" Will not add/update percentage feature")
            return None
        total = show_progress(self.rawData[:, feat_idx].sum(axis=1),
                              f"Computing percentage of {name}", self.nthreads)
        if total.sum() == 0:
            logger.warning(f"Percentage feature {name} not added because not detected in any cell")
            return None
        self.cells.add(name, 100 * total / self.cells.table[self.name+'_nCounts'], overwrite=True)

    def _verify_keys(self, cell_key: str, feat_key: str) -> None:
        if cell_key not in self.cells.table or self.cells.table[cell_key].dtype != bool:
            raise ValueError(f"ERROR: Either {cell_key} does not exist or is not bool type")
        if feat_key not in self.feats.table or self.feats.table[feat_key].dtype != bool:
            raise ValueError(f"ERROR: Either {feat_key} does not exist or is not bool type")

    def _get_cell_feat_idx(self, cell_key: str, feat_key: str) -> Tuple[np.ndarray, np.ndarray]:
        self._verify_keys(cell_key, feat_key)
        cell_idx = self.cells.active_index(cell_key)
        feat_idx = self.feats.active_index(feat_key)
        return cell_idx, feat_idx

    @staticmethod
    def _create_subset_hash(cell_idx: np.ndarray, feat_idx: np.ndarray) -> int:
        return hash(tuple([hash(tuple(cell_idx)), hash(tuple(feat_idx))]))

    def _validate_stats_loc(self, cell_key: str, cell_idx: np.ndarray,
                            feat_idx: np.ndarray) -> Union[str, None]:
        subset_hash = self._create_subset_hash(cell_idx, feat_idx)
        stats_loc = f"summary_stats_{cell_key}"
        if stats_loc in self.z:
            attrs = self.z[stats_loc].attrs
            if 'subset_hash' in attrs and attrs['subset_hash'] == subset_hash:
                return None
        return stats_loc

    def save_normalized_data(self, cell_key: str, feat_key: str, batch_size: int,
                             location: str, log_transform: bool, renormalize_subset: bool,
                             update_feat_key: bool) -> daskarr:

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
                if update_feat_key:
                    self.attrs['latest_feat_key'] = feat_key.split('__', 1)[1] if feat_key != 'I' else 'I'
                return daskarr.from_zarr(self.z[location + '/data'])
            else:
                # Creating group here to overwrite all children
                self.z.create_group(location, overwrite=True)
        vals = self.normed(cell_idx, feat_idx, log_transform=log_transform,
                           renormalize_subset=renormalize_subset)
        dask_to_zarr(vals, self.z, location + '/data', batch_size, self.nthreads)
        self.z[location].attrs['subset_hash'] = subset_hash
        self.z[location].attrs['subset_params'] = subset_params
        if update_feat_key:
            self.attrs['latest_feat_key'] = feat_key.split('__', 1)[1] if feat_key != 'I' else 'I'
        return daskarr.from_zarr(self.z[location + '/data'])

    def score_features(self, feature_names: List[str], cell_key: str,
                       ctrl_size: int, n_bins: int, rand_seed: int) -> np.ndarray:

        from .feat_utils import binned_sampling
        import pandas as pd

        def _name_to_ids(i):
            x = self.feats.table.reindex(self.feats.get_idx_by_names(i))
            x = x[x.I]
            return x.ids.values

        def _calc_mean(i):
            idx = np.array(sorted(self.feats.get_idx_by_ids(i)))
            return self.normed(cell_idx=cell_idx, feat_idx=idx).mean(axis=1).compute()

        feature_ids = _name_to_ids(feature_names)
        if len(feature_ids) == 0:
            raise ValueError(f"ERROR: No feature ids found for any of the provided {len(feature_names)} features")

        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, 'I')
        stats_loc = self._validate_stats_loc(cell_key, cell_idx, feat_idx)
        if stats_loc is None:
            stats_loc = f"summary_stats_{cell_key}"
            if 'avg' not in self.z[stats_loc]:
                raise KeyError(f"ERROR: 'avg' key not found in {stats_loc}. The internal file structure might be "
                               f"corrupted. Please call `set_feature_stats` with the same cell key first.")
        else:
            raise ValueError(f"ERROR: Feature statistics not set for this cell key: {cell_key}. Please call "
                             f"`set_feature_stats` with the same cell key first.")
        obs_avg = pd.Series(np.log(self.z[stats_loc]['avg'][:]), index=self.feats.fetch('ids'))
        control_ids = binned_sampling(obs_avg, feature_ids, ctrl_size, n_bins, rand_seed)
        return _calc_mean(feature_ids) - _calc_mean(control_ids)

    def __repr__(self):
        f = self.feats.table['I']
        assay_name = str(self.__class__).split('.')[-1][:-2]
        return f"{assay_name} {self.name} with {f.sum()}({len(f)}) features"


class RNAassay(Assay):
    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, **kwargs):
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
            self.scalar = self.cells.table[self.name+'_nCounts'].values[cell_idx]
        val = self.normMethod(self, counts)
        self.normMethod = norm_method_cache
        return val

    def set_feature_stats(self, cell_key: str, min_cells: int) -> None:
        feat_key = 'I'  # Here we choose to calculate stats for all the features
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        stats_loc = self._validate_stats_loc(cell_key, cell_idx, feat_idx)
        if stats_loc is None:
            logger.info(f"Using cached feature stats for cell_key {cell_key}")
            return None
        n_cells = show_progress((self.normed(cell_idx, feat_idx) > 0).sum(axis=0),
                                f"({self.name}) Computing nCells", self.nthreads)
        tot = show_progress(self.normed(cell_idx, feat_idx).sum(axis=0),
                            f"({self.name}) Computing normed_tot", self.nthreads)
        sigmas = show_progress(self.normed(cell_idx, feat_idx).var(axis=0),
                               f"({self.name}) Computing sigmas", self.nthreads)
        idx = n_cells > min_cells
        self.feats.update(idx, key=feat_key)
        n_cells, tot, sigmas = n_cells[idx], tot[idx], sigmas[idx]

        group = self.z.create_group(stats_loc, overwrite=True)
        g = create_zarr_dataset(group, 'normed_tot', (50000,), float, tot.shape)
        g[:] = tot
        g = create_zarr_dataset(group, 'avg', (50000,), float, tot.shape)
        g[:] = tot / self.cells.N
        g = create_zarr_dataset(group, 'nz_mean', (50000,), float, tot.shape)
        g[:] = tot / n_cells
        g = create_zarr_dataset(group, 'sigmas', (50000,), float, tot.shape)
        g[:] = sigmas
        g = create_zarr_dataset(group, 'normed_n', (50000,), float, tot.shape)
        g[:] = n_cells

        self.z[stats_loc].attrs['subset_hash'] = self._create_subset_hash(cell_idx,
                                                                          self.feats.active_index(feat_key))
        return None

    def mark_hvgs(self, cell_key: str, min_cells: int, top_n: int,
                  min_var: float, max_var: float, min_mean: float, max_mean: float,
                  n_bins: int, lowess_frac: float, blacklist: str, hvg_key_name: str,
                  clear_from_table: bool, show_plot: bool, **plot_kwargs) -> None:

        self.set_feature_stats(cell_key, min_cells)
        stats_loc = f"summary_stats_{cell_key}"
        c_var_loc = f"c_var__{n_bins}__{lowess_frac}"
        slots = ['normed_tot', 'avg', 'nz_mean', 'sigmas', 'normed_n']
        for i in slots:
            self.feats.add(i, self.z[stats_loc + '/' + i], key='I', overwrite=True)
        if c_var_loc in self.z[stats_loc]:
            logger.info("Using existing corrected dispersion values")
        else:
            c_var = self.feats.remove_trend('avg', 'sigmas', n_bins, lowess_frac)
            g = create_zarr_dataset(self.z[stats_loc], c_var_loc, (50000,), float, c_var.shape)
            g[:] = c_var
        self.feats.add(c_var_loc, self.z[stats_loc + '/' + c_var_loc], key='I', overwrite=True)

        bl = self.feats.idx_to_bool(self.feats.get_idx_by_names(self.feats.grep(blacklist)), invert=True)
        if min_var == -np.Inf:
            if top_n < 1:
                raise ValueError("ERROR: Please provide a value greater than 0 for `top_n` parameter")
            idx = self.feats.multi_sift(
                ['normed_n', 'nz_mean'], [min_cells, min_mean], [np.Inf, max_mean])
            idx = idx & self.feats.table['I'] & bl
            n_valid_feats = idx.sum()
            if top_n > n_valid_feats:
                logger.warning(f"WARNING: Number of valid features are less then value "
                               f"of parameter `top_n`: {top_n}. Resetting `top_n` to {n_valid_feats}")
                top_n = n_valid_feats - 1
            min_var = self.feats.table[idx][c_var_loc].sort_values(ascending=False).values[top_n]
        hvgs = self.feats.multi_sift(
            ['normed_n', 'nz_mean', c_var_loc], [min_cells, min_mean, min_var], [np.Inf, max_mean, max_var])
        hvgs = hvgs & self.feats.table['I'] & bl
        hvg_key_name = cell_key + '__' + hvg_key_name
        logger.info(f"{sum(hvgs)} genes marked as HVGs")
        self.feats.add(hvg_key_name, hvgs, fill_val=False, overwrite=True)

        if show_plot:
            from .plots import plot_mean_var
            nzm, vf, nc = [self.feats.fetch(x).astype('float') for x in ['nz_mean', c_var_loc, 'nCells']]
            plot_mean_var(nzm, vf, nc, self.feats.fetch(hvg_key_name), **plot_kwargs)

        if clear_from_table:
            for i in slots:
                self.feats.remove(i)
            self.feats.remove(c_var_loc)
        return None


class ATACassay(Assay):
    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, **kwargs):
        super().__init__(z, name, cell_data, **kwargs)
        self.normMethod = norm_tf_idf
        self.n_term_per_doc = None
        self.n_docs = None
        self.n_docs_per_term = None

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs):
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        counts = self.rawData[:, feat_idx][cell_idx, :]
        self.n_term_per_doc = self.cells.table[self.name+'_nFeatures'].values[cell_idx]
        self.n_docs = len(cell_idx)
        self.n_docs_per_term = self.feats.table['nCells'].values[feat_idx]
        return self.normMethod(self, counts)

    def set_feature_stats(self, cell_key: str) -> None:
        feat_key = 'I'
        cell_idx, feat_idx = self._get_cell_feat_idx(cell_key, feat_key)
        stats_loc = self._validate_stats_loc(cell_key, cell_idx, feat_idx)
        if stats_loc is None:
            logger.info(f"Using cached feature stats for cell_key {cell_key}")
            return None
        prevalence = show_progress(self.normed(cell_idx, feat_idx).sum(axis=0),
                                   f"({self.name}) Calculating peak prevalence across cells", self.nthreads)
        group = self.z.create_group(stats_loc, overwrite=True)
        g = create_zarr_dataset(group, 'prevalence', (50000,), float, prevalence.shape)
        g[:] = prevalence
        self.z[stats_loc].attrs['subset_hash'] = self._create_subset_hash(cell_idx, feat_idx)
        return None

    def mark_prevalent_peaks(self, cell_key: str, top_n: int, prevalence_key_name: str, clear_from_table: bool):
        self.set_feature_stats(cell_key)
        if top_n >= self.feats.N:
            raise ValueError(f"ERROR: n_top should be less than total number of features ({self.feats.N})]")
        if type(top_n) != int:
            raise TypeError("ERROR: n_top must a positive integer value")
        stats_loc = f"summary_stats_{cell_key}"

        self.feats.add('prevalence', self.z[stats_loc + '/prevalence'], key='I', overwrite=True)
        idx = self.feats.table['prevalence'].sort_values(ascending=False)[:top_n].index
        prevalence_key_name = cell_key + '__' + prevalence_key_name
        self.feats.add(prevalence_key_name, self.feats.idx_to_bool(idx), fill_val=False, overwrite=True)
        if clear_from_table:
            self.feats.remove('prevalence')


class ADTassay(Assay):
    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, **kwargs):
        super().__init__(z, name, cell_data, **kwargs)
        self.normMethod = norm_clr

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs):
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        counts = self.rawData[:, feat_idx][cell_idx, :]
        return self.normMethod(self, counts)
