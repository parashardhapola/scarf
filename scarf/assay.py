import numpy as np
import dask.array as daskarr
import zarr
from .metadata import MetaData
from .utils import show_progress
from .writers import dask_to_zarr


__all__ = ['Assay', 'RNAassay', 'ATACassay', 'ADTassay']


def dummy_norm(counts: daskarr, **kwargs) -> daskarr:
    return counts


def norm_lib_size(counts: daskarr, sf: float, n_counts: np.ndarray) -> daskarr:
    return sf * counts / (n_counts.reshape(-1, 1)+1)


def norm_clr(counts: daskarr) -> daskarr:
    f = np.exp(np.log1p(counts).sum(axis=0) / len(counts))
    return np.log1p(counts / f)


def norm_tf_idf(counts: daskarr, n_docs: int, n_feats_per_cell: np.ndarray,
                n_cells_per_feat: np.ndarray) -> daskarr:
    tf = counts / n_feats_per_cell.reshape(-1, 1)
    idf = np.log2(n_docs / (n_cells_per_feat+1))
    return tf * idf


class Assay:
    def __init__(self, fn: str, name: str, cell_data: MetaData):
        self._fn = fn
        self._z = zarr.open(fn, 'r+')
        self.name = name
        self.cells = cell_data
        self.rawData = daskarr.from_zarr(self._z[self.name]['counts'])
        self.feats = MetaData(self._z[self.name]['featureData'])
        self.attrs = self._z[self.name].attrs
        if 'percentFeatures' not in self.attrs:
            self.attrs['percentFeatures'] = {}
        self.annObj = None  # Can be dynamically attached for debugging purposes
        self.normMethod = dummy_norm
        self.sf = None
        self._ini_feature_props()

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs):
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        return self.normMethod(self.rawData[:, feat_idx][cell_idx, :].rechunk(self.rawData.chunksize))

    @show_progress
    def _ini_feature_props(self, force_recalc: bool = False) -> None:
        if 'nCells' in self.feats.table.columns and  \
                'dropOuts' in self.feats.table.columns and force_recalc is False:
            pass
        else:
            print(f"INFO: ({self.name}) Computing nCells and dropOuts", flush=True)
            self.feats.add('nCells', (self.rawData > 0).sum(axis=0).compute(), overwrite=True)
            self.feats.add('dropOuts', abs(self.cells.N - self.feats.fetch('nCells')), overwrite=True)

    @show_progress
    def add_percent_feature(self, feat_pattern: str, name: str, verbose: bool = True) -> None:
        if name in self.attrs['percentFeatures']:
            if self.attrs['percentFeatures'][name] == feat_pattern:
                if verbose:
                    print(f"INFO: Percentage feature {name} already exists. Not adding again")
                return None
        feat_idx = self.feats.get_idx_by_names(self.feats.grep(feat_pattern))
        if len(feat_idx) == 0:
            print(f"WARNING: No matches found for pattern {feat_pattern}. Will not add/update percentage feature")
            return None
        print(f"Computing percentage of {name}", flush=True)
        total = self.rawData[:, feat_idx].rechunk(self.rawData.chunksize).sum(axis=1).compute()
        self.cells.add(name, 100 * total / self.cells.table['nCounts'], overwrite=True)
        self.attrs['percentFeatures'] = {**{k: v for k, v in self.attrs['percentFeatures'].items()},
                                         **{name: feat_pattern}}

    def create_subset_hash(self, cell_key: str, feat_key: str):
        cell_idx = self.cells.active_index(cell_key)
        feat_idx = self.feats.active_index(feat_key)
        return hash(tuple([hash(tuple(cell_idx)), hash(tuple(feat_idx))]))

    def select_and_normalize(self, cell_key: str, feat_key: str, batch_size: int) -> daskarr:
        if cell_key not in self.cells.table or self.cells.table[cell_key].dtype != bool:
            raise ValueError(f"ERROR: Either {cell_key} does not exist or is not bool type")
        if feat_key not in self.feats.table or self.feats.table[feat_key].dtype != bool:
            raise ValueError(f"ERROR: Either {feat_key} does not exist or is not bool type")
        cell_idx = self.cells.active_index(cell_key)
        feat_idx = self.feats.active_index(feat_key)
        subset_hash = self.create_subset_hash(cell_key, feat_key)
        subset_name = f"subset_{cell_key}_{feat_key}"
        loc = f"{self.name}/{subset_name}"
        if subset_name in self.attrs and self.attrs[subset_name] == subset_hash and loc in self._z:
            print(f"INFO: Exact features already selected for assay {self.name}")
        else:
            # renormed parameter will not have an effect unless the norMethod uses it. But this is hard coded
            dask_to_zarr(self.normed(cell_idx, feat_idx, renormed=True), self._z, loc, batch_size)
            self.attrs[subset_name] = subset_hash
        return daskarr.from_zarr(self._z[loc])

    def __repr__(self):
        f = self.feats.table['I']
        assay_name = str(self.__class__).split('.')[-1][:-2]
        return f"{assay_name} {self.name} with {f.sum()}({len(f)}) features"


class RNAassay(Assay):
    def __init__(self, fn: str, name: str, cell_data: MetaData):
        super().__init__(fn, name, cell_data)
        self.normMethod = norm_lib_size
        self.sf = 10000

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, renormed: bool = False):
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        counts = self.rawData[:, feat_idx][cell_idx, :].rechunk(self.rawData.chunksize)
        if renormed:
            scalar = counts.sum(axis=1).reshape(-1, 1) + 1
        else:
            scalar = self.cells.table['nCounts'].values[cell_idx]
        return self.normMethod(counts, self.sf, scalar)

    def correct_var(self, n_bins: int = 200, lowess_frac: float = 0.1) -> None:
        self.feats.remove_trend('c_var', 'avg', 'sigmas', n_bins, lowess_frac)

    def correct_dropouts(self, n_bins: int = 200, lowess_frac: float = 0.1) -> None:
        self.feats.remove_trend('c_dropOuts', 'avg', 'dropOuts', n_bins, lowess_frac)

    @show_progress
    def set_feature_stats(self, cell_key: str = 'I', feat_key: str = 'I', min_cells: int = 10) -> None:
        subset_name = f"subset_{cell_key}_{feat_key}"
        cell_idx = self.cells.active_index(cell_key)

        feat_idx = self.feats.active_index(feat_key)
        subset_hash = hash(tuple([hash(tuple(cell_idx)), hash(tuple(feat_idx))]))
        if subset_name in self.attrs and self.attrs[subset_name] == subset_hash:
            print("INFO: Using cached feature stats data")
            return None

        print(f"INFO: ({self.name}) Cells per features (1/3)", flush=True)
        n_cells = (self.normed(cell_idx, feat_idx) > 0).sum(axis=0).compute()
        self.feats.add('normed_n', n_cells, key=feat_key, overwrite=True)
        self.feats.update(n_cells > min_cells, key=feat_key)
        n_cells = self.feats.fetch('normed_n', key=feat_key)
        feat_idx = self.feats.active_index(feat_key)
        subset_hash = hash(tuple([hash(tuple(cell_idx)), hash(tuple(feat_idx))]))

        print(f"INFO: ({self.name}) Feature sum (2/3)", flush=True)
        tot = self.normed(cell_idx, feat_idx).sum(axis=0).compute()
        self.feats.add('normed_tot', tot, key=feat_key, overwrite=True)
        self.feats.add('avg', tot / self.cells.N,  key=feat_key, overwrite=True)
        self.feats.add('nz_mean', tot / n_cells, key=feat_key, overwrite=True)

        print(f"INFO: ({self.name}) Feature variance (3/3)", flush=True)
        sigmas = self.normed(cell_idx, feat_idx).var(axis=0).compute()
        self.feats.add('sigmas', sigmas, key=feat_key, overwrite=True)

        self.attrs[subset_name] = subset_hash
        return None

    def mark_hvgs(self, min_cells: int = 20, top_n: int = 500,
                  min_var: float = -np.Inf, max_var: float = np.Inf,
                  min_mean: float = -np.Inf, max_mean: float = np.Inf,
                  blacklist: str = "^MT-|^RPS|^RPL|^MRPS|^MRPL|^CCN|^HLA-|^H2-|^HIST"):
        bl = self.feats.idx_to_bool(self.feats.get_idx_by_names(self.feats.grep(blacklist)), invert=True)
        if min_var == -np.Inf:
            if top_n < 1:
                raise ValueError("ERROR: Please provide a value greater than 0 for `top_n` parameter")
            idx = self.feats.multi_sift(
                ['normed_n', 'nz_mean'], [min_cells, min_mean], [np.Inf, max_mean])
            idx = idx & self.feats.table['I'] & bl
            min_var = self.feats.table[idx]['c_var'].sort_values(ascending=False).values[top_n]
        hvgs = self.feats.multi_sift(
            ['normed_n', 'nz_mean', 'c_var'], [min_cells, min_mean, min_var], [np.Inf, max_mean, max_var])
        hvgs = hvgs & self.feats.table['I'] & bl
        print(f"INFO: {sum(hvgs)} genes marked as HVGs", flush=True)
        self.feats.add('hvgs', hvgs, fill_val=False, overwrite=True)


class ATACassay(Assay):
    def __init__(self, fn: str, name: str, cell_data: MetaData):
        super().__init__(fn, name, cell_data)
        self.normMethod = norm_tf_idf

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs):
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        return self.normMethod(self.rawData[cell_idx, :][:, feat_idx], n_docs=len(cell_idx),
                               n_feats_per_cell=self.cells.table['nFeatures'].values[cell_idx],
                               n_cells_per_feat=self.feats.table['nCells'].values[feat_idx])

    @show_progress
    def set_feature_stats(self, cell_key: str = 'I', feat_key: str = 'I') -> None:
        subset_name = f"subset_{cell_key}_{feat_key}"
        cell_idx = self.cells.active_index(cell_key)

        feat_idx = self.feats.active_index(feat_key)
        subset_hash = hash(tuple([hash(tuple(cell_idx)), hash(tuple(feat_idx))]))
        if subset_name in self.attrs and self.attrs[subset_name] == subset_hash:
            print("INFO: Using cached feature stats data", flush=True)
            return None

        print(f"INFO: ({self.name}) Calculating peak prevalence across cells", flush=True)
        prevalence = self.normed(cell_idx, feat_idx).sum(axis=0).compute()
        self.feats.add('prevalence', prevalence, fill_val=False, overwrite=True)

        self.attrs[subset_name] = subset_hash
        return None

    @show_progress
    def mark_top_prevalent_peaks(self, n_top: int = 1000):
        if 'prevalence' not in self.feats.table:
            raise ValueError("ERROR: Please 'run set_feature_stats' first")
        if n_top >= self.feats.N:
            raise ValueError(f"ERROR: n_top should be less than total number of features ({self.feats.N})]")
        if type(n_top) != int:
            raise TypeError("ERROR: n_top must a positive integer value")
        idx = self.feats.table['prevalence'].sort_values(ascending=False)[:n_top].index
        self.feats.add('top_peaks', self.feats.idx_to_bool(idx), fill_val=False, overwrite=True)


class ADTassay(Assay):
    def __init__(self, fn: str, name: str, cell_data: MetaData):
        super().__init__(fn, name, cell_data)
        self.normMethod = norm_clr

    def normed(self, cell_idx: np.ndarray = None, feat_idx: np.ndarray = None, **kwargs):
        if cell_idx is None:
            cell_idx = self.cells.active_index('I')
        if feat_idx is None:
            feat_idx = self.feats.active_index('I')
        return self.normMethod(self.rawData[cell_idx, :][:, feat_idx])
