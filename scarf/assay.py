import numpy as np
import dask.array as daskarr
import zarr
from .metadata import MetaData
from .utils import calc_computed
from .writers import create_zarr_dataset

__all__ = ['Assay', 'RNAassay', 'ATACassay', 'ADTassay']


def norm_dummy(assay, counts: daskarr) -> daskarr:
    return counts


def norm_lib_size(assay, counts: daskarr) -> daskarr:
    return assay.sf * counts / assay.scalar.reshape(-1, 1)


def norm_clr(assay, counts: daskarr) -> daskarr:
    f = np.exp(np.log1p(counts).sum(axis=0) / len(counts))
    return np.log1p(counts / f)


def norm_tf_idf(assay, counts: daskarr) -> daskarr:
    tf = counts / assay.n_term_per_doc.reshape(-1, 1)
    # TODO: Split TF and IDF functionality to make it similar to norml_lib and zscaling
    idf = np.log2(assay.n_docs / (assay.n_docs_per_term + 1))
    return tf * idf


class Assay:
    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, min_cells_per_feature: int = 10):
        self.name = name
        self.z = z[self.name]
        self.cells = cell_data
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

    def _ini_feature_props(self, min_cells: int) -> None:
        if 'nCells' in self.feats.table.columns and 'dropOuts' in self.feats.table.columns:
            pass
        else:
            ncells = calc_computed((self.rawData > 0).sum(axis=0),
                                   f"INFO: ({self.name}) Computing nCells and dropOuts")
            self.feats.add('nCells', ncells, overwrite=True)
            self.feats.add('dropOuts', abs(self.cells.N - self.feats.fetch('nCells')), overwrite=True)
            self.feats.update(ncells > min_cells)

    def _verify_keys(self, cell_key: str, feat_key: str) -> None:
        if cell_key not in self.cells.table or self.cells.table[cell_key].dtype != bool:
            raise ValueError(f"ERROR: Either {cell_key} does not exist or is not bool type")
        if feat_key not in self.feats.table or self.feats.table[feat_key].dtype != bool:
            raise ValueError(f"ERROR: Either {feat_key} does not exist or is not bool type")

    def add_percent_feature(self, feat_pattern: str, name: str, verbose: bool = True) -> None:
        if name in self.attrs['percentFeatures']:
            if self.attrs['percentFeatures'][name] == feat_pattern:
                if verbose:
                    print(f"INFO: Percentage feature {name} already exists. Not adding again")
                return None
        feat_idx = sorted(self.feats.get_idx_by_names(self.feats.grep(feat_pattern)))
        if len(feat_idx) == 0:
            print(f"WARNING: No matches found for pattern {feat_pattern}. Will not add/update percentage feature")
            return None
        total = calc_computed(self.rawData[:, feat_idx].sum(axis=1),
                              f"Computing percentage of {name}")
        if total.sum() == 0:
            print(f"WARNING: Percentage feature {name} not added because not detected in any cell", flush=True)
            return None
        self.cells.add(name, 100 * total / self.cells.table['nCounts'], overwrite=True)
        self.attrs['percentFeatures'] = {**{k: v for k, v in self.attrs['percentFeatures'].items()},
                                         **{name: feat_pattern}}

    def create_subset_hash(self, cell_key: str, feat_key: str):
        cell_idx = self.cells.active_index(cell_key)
        feat_idx = self.feats.active_index(feat_key)
        return hash(tuple([hash(tuple(cell_idx)), hash(tuple(feat_idx))]))

    def save_normalized_data(self, cell_key: str, feat_key: str, batch_size: int,
                             location: str, log_transform: bool, renormalize_subset: bool) -> daskarr:
        # Because HVGs and other feature selections have cell key appended in their metadata
        from .writers import dask_to_zarr

        if feat_key != 'I':
            feat_key = cell_key + '__' + feat_key
        self._verify_keys(cell_key, feat_key)

        subset_hash = self.create_subset_hash(cell_key, feat_key)
        subset_params = {'log_transform': log_transform, 'renormalize_subset': renormalize_subset}
        if location in self.z:
            if subset_hash == self.z[location].attrs['subset_hash'] and \
                    subset_params == self.z[location].attrs['subset_params']:
                print(f"INFO: Using existing normalized data with cell key {cell_key} and feat key {feat_key}",
                      flush=True)
                self.attrs['latest_feat_key'] = feat_key.split('__', 1)[1] if feat_key != 'I' else 'I'
                return daskarr.from_zarr(self.z[location + '/data'])
            else:
                # Creating group here to overwrite all children
                self.z.create_group(location, overwrite=True)
        cell_idx = self.cells.active_index(cell_key)
        feat_idx = self.feats.active_index(feat_key)
        vals = self.normed(cell_idx, feat_idx, log_transform=log_transform,
                           renormalize_subset=renormalize_subset)
        dask_to_zarr(vals, self.z, location + '/data', batch_size)
        self.z[location].attrs['subset_hash'] = subset_hash
        self.z[location].attrs['subset_params'] = subset_params
        self.attrs['latest_feat_key'] = feat_key.split('__', 1)[1] if feat_key != 'I' else 'I'
        return daskarr.from_zarr(self.z[location + '/data'])

    def __repr__(self):
        f = self.feats.table['I']
        assay_name = str(self.__class__).split('.')[-1][:-2]
        return f"{assay_name} {self.name} with {f.sum()}({len(f)}) features"


class RNAassay(Assay):
    def __init__(self, z: zarr.hierarchy, name: str, cell_data: MetaData, **kwargs):
        super().__init__(z, name, cell_data, **kwargs)
        self.normMethod = norm_lib_size
        self.sf = 10000
        self.scalar = None

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None,
               renormalize_subset: bool = False, log_transform: bool = False, **kwargs):
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        counts = self.rawData[:, feat_idx][cell_idx, :]
        if log_transform:
            counts = np.log1p(counts)
        if renormalize_subset:
            print("INFO: Renormalizing normed data", flush=True)
            a = counts.sum(axis=1).compute()
            a[a == 0] = 1
            self.scalar = a
        else:
            self.scalar = self.cells.table['nCounts'].values[cell_idx]
        return self.normMethod(self, counts)

    def set_feature_stats(self, cell_key: str, min_cells: int = 10) -> None:
        feat_key = 'I'  # Here we choose to calculate stats for all the features
        self._verify_keys(cell_key, feat_key)
        subset_hash = self.create_subset_hash(cell_key, feat_key)
        stats_loc = f"summary_stats_{cell_key}"
        if stats_loc in self.z:
            attrs = self.z[stats_loc].attrs
            if 'subset_hash' in attrs and attrs['subset_hash'] == subset_hash:
                print(f"INFO: Using cached feature stats for cell_key {cell_key}")
                return None
        cell_idx = self.cells.active_index(cell_key)
        feat_idx = self.feats.active_index(feat_key)

        n_cells = calc_computed((self.normed(cell_idx, feat_idx) > 0).sum(axis=0),
                                f"INFO: ({self.name}) Computing nCells")
        tot = calc_computed(self.normed(cell_idx, feat_idx).sum(axis=0),
                            f"INFO: ({self.name}) Computing normed_tot")
        sigmas = calc_computed(self.normed(cell_idx, feat_idx).var(axis=0),
                               f"INFO: ({self.name}) Computing sigmas")
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

        self.z[stats_loc].attrs['subset_hash'] = self.create_subset_hash(cell_key, feat_key)
        return None

    def mark_hvgs(self, cell_key: str = 'I', min_cells: int = 20, top_n: int = 500,
                  min_var: float = -np.Inf, max_var: float = np.Inf,
                  min_mean: float = -np.Inf, max_mean: float = np.Inf,
                  n_bins: int = 200, lowess_frac: float = 0.1,
                  blacklist: str = "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST",
                  show_plot: bool = True, hvg_key_name: str = 'hvgs', **plot_kwargs) -> None:
        self.set_feature_stats(cell_key)
        stats_loc = f"summary_stats_{cell_key}"
        c_var_loc = f"c_var__{n_bins}__{lowess_frac}"
        slots = ['normed_tot', 'avg', 'nz_mean', 'sigmas', 'normed_n']
        for i in slots:
            self.feats.add(i, self.z[stats_loc + '/' + i], key='I', overwrite=True)
        if c_var_loc in self.z[stats_loc]:
            print("INFO: Using existing corrected dispersion values")
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
            min_var = self.feats.table[idx][c_var_loc].sort_values(ascending=False).values[top_n]
        hvgs = self.feats.multi_sift(
            ['normed_n', 'nz_mean', c_var_loc], [min_cells, min_mean, min_var], [np.Inf, max_mean, max_var])
        hvgs = hvgs & self.feats.table['I'] & bl
        hvg_key_name = cell_key + '__' + hvg_key_name
        print(f"INFO: {sum(hvgs)} genes marked as HVGs", flush=True)
        self.feats.add(hvg_key_name, hvgs, fill_val=False, overwrite=True)

        if show_plot:
            from .plots import plot_mean_var
            nzm, vf, nc = [self.feats.fetch(x).astype('float') for x in ['nz_mean', c_var_loc, 'nCells']]
            plot_mean_var(nzm, vf, nc, self.feats.fetch(hvg_key_name), **plot_kwargs)

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
        self.n_term_per_doc = self.cells.table['nFeatures'].values[cell_idx]
        self.n_docs = len(cell_idx)
        self.n_docs_per_term = self.feats.table['nCells'].values[feat_idx]
        return self.normMethod(self, counts)

    def set_feature_stats(self, cell_key: str = 'I') -> None:
        feat_key = 'I'
        self._verify_keys(cell_key, feat_key)
        subset_hash = self.create_subset_hash(cell_key, feat_key)
        stats_loc = f"summary_stats_{cell_key}"
        if stats_loc in self.z:
            attrs = self.z[stats_loc].attrs
            if 'subset_hash' in attrs and attrs['subset_hash'] == subset_hash:
                print(f"INFO: Using cached feature stats for cell_key {cell_key}")
                return None

        cell_idx = self.cells.active_index(cell_key)
        feat_idx = self.feats.active_index(feat_key)
        prevalence = calc_computed(self.normed(cell_idx, feat_idx).sum(axis=0),
                                   f"INFO: ({self.name}) Calculating peak prevalence across cells")
        group = self.z.create_group(stats_loc, overwrite=True)
        g = create_zarr_dataset(group, 'prevalence', (50000,), float, prevalence.shape)
        g[:] = prevalence
        self.z[stats_loc].attrs['subset_hash'] = self.create_subset_hash(cell_key, feat_key)
        return None

    def mark_top_prevalent_peaks(self, cell_key: str = 'I', n_top: int = 1000):
        self.set_feature_stats(cell_key)
        if n_top >= self.feats.N:
            raise ValueError(f"ERROR: n_top should be less than total number of features ({self.feats.N})]")
        if type(n_top) != int:
            raise TypeError("ERROR: n_top must a positive integer value")
        stats_loc = f"summary_stats_{cell_key}"
        self.feats.add('prevalence', self.z[stats_loc + '/prevalence'], key='I', overwrite=True)
        idx = self.feats.table['prevalence'].sort_values(ascending=False)[:n_top].index
        self.feats.add(cell_key+'__top_peaks', self.feats.idx_to_bool(idx), fill_val=False, overwrite=True)
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
