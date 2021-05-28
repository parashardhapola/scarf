import zarr
from typing import Any, Tuple, List, Union
import numpy as np
from tqdm import tqdm
from .readers import CrReader, H5adReader, NaboH5Reader, LoomReader
import os
import pandas as pd
from .utils import controlled_compute
from .logging_utils import logger
from scipy.sparse import csr_matrix

__all__ = ['create_zarr_dataset', 'create_zarr_obj_array', 'create_zarr_count_assay',
           'subset_assay_zarr', 'dask_to_zarr', 'ZarrMerge',
           'CrToZarr', 'H5adToZarr', 'MtxToZarr', 'NaboH5ToZarr', 'LoomToZarr', 'SparseToZarr']


"""
Methods and classes for writing data to disk.

Methods:
    create_zarr_dataset:
    create_zarr_obj_array:
    create_zarr_count_assay:
    subset_assay_zarr:
    dask_to_zarr:

Classes:
    ZarrMerge:
    CrToZarr:
    H5adToZarr:
    MtxToZarr:
    NaboH5ToZarr:
    LoomToZarr:
"""


def create_zarr_dataset(g: zarr.hierarchy, name: str, chunks: tuple,
                        dtype: Any, shape: Tuple, overwrite: bool = True) -> zarr.hierarchy:
    # TODO: add description in docstring
    """
    Returns:
        A Zarr Array.
    """
    from numcodecs import Blosc

    compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE)
    return g.create_dataset(name, chunks=chunks, dtype=dtype,
                            shape=shape, compressor=compressor, overwrite=overwrite)


def create_zarr_obj_array(g: zarr.hierarchy, name: str, data,
                          dtype: Union[str, Any] = None, overwrite: bool = True) -> zarr.hierarchy:
    # TODO: add docstring
    data = np.array(data)
    if dtype is None or dtype == object:
        dtype = 'U' + str(max([len(str(x)) for x in data]))
    if np.issubdtype(data.dtype, np.dtype('S')):
        data = data.astype('U')
    return g.create_dataset(name, data=data, chunks=(100000,),
                            shape=len(data), dtype=dtype, overwrite=overwrite)


def create_zarr_count_assay(z: zarr.hierarchy, assay_name: str, chunk_size: Tuple[int, int], n_cells: int,
                            feat_ids: Union[np.ndarray, List[str]], feat_names: Union[np.ndarray, List[str]],
                            dtype: str = 'uint32') -> zarr.hierarchy:
    # TODO: add docstring
    g = z.create_group(assay_name, overwrite=True)
    g.attrs['is_assay'] = True
    g.attrs['misc'] = {}
    create_zarr_obj_array(g, 'featureData/ids', feat_ids)
    create_zarr_obj_array(g, 'featureData/names', feat_names)
    create_zarr_obj_array(g, 'featureData/I',
                          [True for _ in range(len(feat_ids))], 'bool')
    return create_zarr_dataset(g, 'counts', chunk_size, dtype,
                               (n_cells, len(feat_ids)), overwrite=True)


class CrToZarr:
    def __init__(self, cr: CrReader, zarr_fn: str, chunk_size=(1000, 1000), dtype: str = 'uint32'):
        # TODO: add docstring
        self.cr = cr
        self.fn = zarr_fn
        self.chunkSizes = chunk_size
        self.z = zarr.open(self.fn, mode='w')
        self._ini_cell_data()
        for assay_name in self.cr.assayFeats.columns:
            create_zarr_count_assay(self.z, assay_name, chunk_size, self.cr.nCells,
                                    self.cr.feature_ids(assay_name), self.cr.feature_names(assay_name), dtype)

    def _ini_cell_data(self):
        g = self.z.create_group('cellData')
        create_zarr_obj_array(g, 'ids', self.cr.cell_names())
        create_zarr_obj_array(g, 'names', self.cr.cell_names())
        create_zarr_obj_array(g, 'I', [True for _ in range(self.cr.nCells)], 'bool')

    def dump(self, batch_size: int = 1000, lines_in_mem: int = 100000) -> None:
        # TODO: add docstring
        stores = [self.z["%s/counts" % x] for x in self.cr.assayFeats.columns]
        spidx = [0] + list(self.cr.assayFeats.T.nFeatures.cumsum().values)
        spidx = [(spidx[x - 1], spidx[x]) for x in range(1, len(spidx))]
        s, e, = 0, 0
        n_chunks = self.cr.nCells//batch_size + 1
        for a in tqdm(self.cr.consume(batch_size, lines_in_mem), total=n_chunks):
            e += a.shape[0]
            a = a.todense()
            for j in range(len(stores)):
                stores[j][s:e] = a[:, spidx[j][0]:spidx[j][1]]
            s = e
        if e != self.cr.nCells:
            raise AssertionError("ERROR: This is a bug in CrToZarr. All cells might not have been successfully "
                                 "written into the zarr file. Please report this issue")


class MtxToZarr:
    def __init__(self, cr: CrReader, zarr_fn: str, chunk_size=(1000, 1000), dtype: str = 'uint32'):
        # TODO: add docstring
        self.cr = cr
        self.fn = zarr_fn
        self.chunkSizes = chunk_size
        self.z = zarr.open(self.fn, mode='w')
        self._ini_cell_data()
        for assay_name in set(self.cr.assayFeats.columns):
            create_zarr_count_assay(self.z, assay_name, chunk_size, self.cr.nCells,
                                    self.cr.feature_ids(assay_name), self.cr.feature_names(assay_name), dtype)

    def _ini_cell_data(self):
        g = self.z.create_group('cellData')
        create_zarr_obj_array(g, 'ids', self.cr.cell_names())
        create_zarr_obj_array(g, 'names', self.cr.cell_names())
        create_zarr_obj_array(g, 'I', [True for _ in range(self.cr.nCells)], 'bool')

    def _prep_assay_ranges(self):
        ret_val = {}
        for assay in set(self.cr.assayFeats.columns):
            temp = []
            if len(self.cr.assayFeats[assay].shape) == 2:
                for i in self.cr.assayFeats[assay].values[1:3].T:
                    temp.append([i[0], i[1]])
            else:
                idx = self.cr.assayFeats[assay]
                temp = [[idx.start, idx.end]]
            ret_val[assay] = temp
        return ret_val

    def dump(self, batch_size: int = 1000, lines_in_mem: int = 100000) -> None:
        # TODO: add docstring
        stores = {x: self.z["%s/counts" % x] for x in set(self.cr.assayFeats.columns)}
        assay_ranges = self._prep_assay_ranges()
        s, e, = 0, 0
        n_chunks = self.cr.nCells // batch_size + 1
        for a in tqdm(self.cr.consume(batch_size, lines_in_mem), total=n_chunks):
            e += a.shape[0]
            a = a.todense()
            b = {x: [] for x in stores.keys()}
            for store_name in stores.keys():
                ar = assay_ranges[store_name]
                temp = []
                for i in ar:
                    temp.append(a[:, i[0]:i[1]])
                if len(temp) > 1:
                    b[store_name] = np.hstack(temp)
                else:
                    b[store_name] = temp[0]
            for store_name in stores.keys():
                stores[store_name][s:e] = b[store_name]
            s = e
        if e != self.cr.nCells:
            raise AssertionError("ERROR: This is a bug in MtxToZarr. All cells might not have been successfully "
                                 "written into the zarr file. Please report this issue")


class H5adToZarr:
    def __init__(self, h5ad: H5adReader, zarr_fn: str, assay_name: str = None,
                 chunk_size=(1000, 1000), dtype: str = 'uint32'):
        # TODO: support for multiple assay. One of the `var` datasets can be used to group features in separate assays
        # TODO: add docstring
        self.h5ad = h5ad
        self.fn = zarr_fn
        self.chunkSizes = chunk_size
        if assay_name is None:
            logger.info(f"No value provided for assay names. Will use default value: 'RNA'")
            self.assayName = 'RNA'
        else:
            self.assayName = assay_name
        self.z = zarr.open(self.fn, mode='w')
        self._ini_cell_data()
        create_zarr_count_assay(self.z, self.assayName, chunk_size, self.h5ad.nCells,
                                self.h5ad.feat_ids(), self.h5ad.feat_names(), dtype)
        for i, j in self.h5ad.get_feat_columns():
            if i not in self.z[self.assayName]['featureData']:
                create_zarr_obj_array(self.z[self.assayName]['featureData'], i, j, j.dtype)

    def _ini_cell_data(self):
        g = self.z.create_group('cellData')
        ids = self.h5ad.cell_ids()
        create_zarr_obj_array(g, 'ids', ids, ids.dtype)
        create_zarr_obj_array(g, 'names', ids, ids.dtype)
        create_zarr_obj_array(g, 'I', [True for _ in range(self.h5ad.nCells)], 'bool')
        for i, j in self.h5ad.get_cell_columns():
            create_zarr_obj_array(g, i, j, j.dtype)

    def dump(self, batch_size: int = 1000) -> None:
        # TODO: add docstring
        store = self.z["%s/counts" % self.assayName]
        s, e, = 0, 0
        n_chunks = self.h5ad.nCells//batch_size + 1
        for a in tqdm(self.h5ad.consume(batch_size), total=n_chunks):
            e += a.shape[0]
            store[s:e] = a
            s = e
        if e != self.h5ad.nCells:
            raise AssertionError("ERROR: This is a bug in H5adToZarr. All cells might not have been successfully "
                                 "written into the zarr file. Please report this issue")


class NaboH5ToZarr:
    def __init__(self, h5: NaboH5Reader, zarr_fn: str, assay_name: str = None,
                 chunk_size=(1000, 1000), dtype: str = 'uint32'):
        # TODO: add docstring
        self.h5 = h5
        self.fn = zarr_fn
        self.chunkSizes = chunk_size
        if assay_name is None:
            logger.info(f"No value provided for assay names. Will use default value: 'RNA'")
            self.assayName = 'RNA'
        else:
            self.assayName = assay_name
        self.z = zarr.open(self.fn, mode='w')
        self._ini_cell_data()
        create_zarr_count_assay(self.z, self.assayName, chunk_size, self.h5.nCells,
                                self.h5.feat_ids(), self.h5.feat_names(), dtype)

    def _ini_cell_data(self):
        g = self.z.create_group('cellData')
        create_zarr_obj_array(g, 'ids', self.h5.cell_ids())
        create_zarr_obj_array(g, 'names', self.h5.cell_ids())
        create_zarr_obj_array(g, 'I', [True for _ in range(self.h5.nCells)], 'bool')

    def dump(self, batch_size: int = 500) -> None:
        store = self.z["%s/counts" % self.assayName]
        s, e, = 0, 0
        n_chunks = self.h5.nCells // batch_size + 1
        for a in tqdm(self.h5.consume(batch_size), total=n_chunks):
            e += a.shape[0]
            store[s:e] = a
            s = e
        if e != self.h5.nCells:
            raise AssertionError("ERROR: This is a bug in NaboH5ToZarr. All cells might not have been successfully "
                                 "written into the zarr file. Please report this issue")


class LoomToZarr:
    def __init__(self, loom: LoomReader, zarr_fn: str, assay_name: str = None,
                 chunk_size=(1000, 1000)):
        """
        Converts Loom file read using scarf.LoomReader into Scarf's Zarr format

        Args:
            loom: LoomReader object used to open Loom format file
            zarr_fn: Output Zarr filename with path
            assay_name: Name for the output assay. If not provided then automatically set to RNA
            chunk_size: Chunk size for the count matrix saved in Zarr file.
        """
        # TODO: support for multiple assay. Data from within individual layers can be treated as separate assays
        self.loom = loom
        self.fn = zarr_fn
        self.chunkSizes = chunk_size
        if assay_name is None:
            logger.info(f"No value provided for assay names. Will use default value: 'RNA'")
            self.assayName = 'RNA'
        else:
            self.assayName = assay_name
        self.z = zarr.open(self.fn, mode='w')
        self._ini_cell_data()
        create_zarr_count_assay(self.z, self.assayName, chunk_size, self.loom.nCells,
                                self.loom.feature_ids(), self.loom.feature_names(), self.loom.matrixDtype)
        for i, j in self.loom.get_feature_attrs():
            create_zarr_obj_array(self.z[self.assayName]['featureData'], i, j, j.dtype)

    def _ini_cell_data(self):
        g = self.z.create_group('cellData')
        create_zarr_obj_array(g, 'ids', self.loom.cell_ids())
        create_zarr_obj_array(g, 'names', self.loom.cell_ids())
        create_zarr_obj_array(g, 'I', [True for _ in range(self.loom.nCells)], 'bool')
        for i, j in self.loom.get_cell_attrs():
            create_zarr_obj_array(g, i, j, j.dtype)

    def dump(self, batch_size: int = 1000) -> None:
        # TODO: add docstring
        store = self.z["%s/counts" % self.assayName]
        s, e, = 0, 0
        n_chunks = self.loom.nCells//batch_size + 1
        for a in tqdm(self.loom.consume(batch_size), total=n_chunks):
            e += a.shape[0]
            store[s:e] = a
            s = e
        if e != self.loom.nCells:
            raise AssertionError("ERROR: This is a bug in LoomToZarr. All cells might not have been successfully "
                                 "written into the zarr file. Please report this issue")


class SparseToZarr:
    def __init__(self, csr_mat: csr_matrix, zarr_fn: str, cell_ids: List[str], feature_ids: List[str],
                 assay_name: str = None, chunk_size=(1000, 1000), ):
        # TODO: add docstring
        self.mat = csr_mat
        self.fn = zarr_fn
        self.chunkSizes = chunk_size
        if assay_name is None:
            logger.info(f"No value provided for assay names. Will use default value: 'RNA'")
            self.assayName = 'RNA'
        else:
            self.assayName = assay_name
        self.nFeatures, self.nCells = self.mat.shape
        if len(cell_ids) != self.nCells:
            raise ValueError("ERROR: Number of cell ids are not same as number of cells in the matrix")
        if len(feature_ids) != self.nFeatures:
            raise ValueError("ERROR: Number of feature ids are not same as number of features in the matrix")

        self.z = zarr.open(self.fn, mode='w')
        self._ini_cell_data(cell_ids)
        create_zarr_count_assay(self.z, self.assayName, chunk_size, self.nCells,
                                feature_ids, feature_ids, 'int64')

    def _ini_cell_data(self, cell_ids):
        g = self.z.create_group('cellData')
        create_zarr_obj_array(g, 'ids', cell_ids)
        create_zarr_obj_array(g, 'names', cell_ids)
        create_zarr_obj_array(g, 'I', [True for _ in range(self.nCells)], 'bool')

    def dump(self, batch_size: int = 1000) -> None:
        # TODO: add docstring
        store = self.z["%s/counts" % self.assayName]
        s, e, = 0, 0
        n_chunks = self.nCells//batch_size + 1
        for e in tqdm(range(batch_size, self.nCells+batch_size, batch_size), total=n_chunks):
            if s == self.nCells:
                raise ValueError("Unexpected error encountered in writing to Zarr. The last iteration has failed. "
                                 "Please report this issue.")
            if e > self.nCells:
                e = self.nCells
            store[s:e] = self.mat[:, s:e].todense().T
            s = e
        if e != self.nCells:
            raise AssertionError("ERROR: This is a bug in SparseToZarr. All cells might not have been successfully "
                                 "written into the zarr file. Please report this issue")


def subset_assay_zarr(zarr_fn: str, in_grp: str, out_grp: str,
                      cells_idx: np.ndarray, feat_idx: np.ndarray,
                      chunk_size: tuple):
    # TODO: add docstring
    z = zarr.open(zarr_fn, 'r+')
    ig = z[in_grp]
    og = create_zarr_dataset(z, out_grp, chunk_size, 'uint32', (len(cells_idx), len(feat_idx)))
    pos_start, pos_end = 0, 0
    for i in tqdm(np.array_split(cells_idx, len(cells_idx) // chunk_size[0] + 1)):
        pos_end += len(i)
        og[pos_start:pos_end, :] = ig.get_orthogonal_selection((i, feat_idx))
        pos_start = pos_end
    return None


def dask_to_zarr(df, z, loc, chunk_size, nthreads: int, msg: str = None):
    # TODO: add docstring
    if msg is None:
        msg = f"Writing data to {loc}"
    og = create_zarr_dataset(z, loc, chunk_size, 'float64', df.shape)
    pos_start, pos_end = 0, 0
    for i in tqdm(df.blocks, total=df.numblocks[0], desc=msg):
        pos_end += i.shape[0]
        og[pos_start:pos_end, :] = controlled_compute(i, nthreads)
        pos_start = pos_end
    return None


class ZarrMerge:

    def __init__(self, zarr_path: str, assays: list, names: List[str], merge_assay_name: str,
                 chunk_size=(1000, 1000), dtype: str = None, overwrite: bool = False,
                 reset_cell_filter: bool = True):
        """
        Merge multiple Zarr files into a single Zarr file

        Args:
            zarr_path: Name of the new, merged Zarr file with path
            assays: List of assay objects to be merged. For example, [ds1.RNA, ds2.RNA]
            names: Names of the each assay objects in the `assays` parameter. They should be in the same order as in
                   `assays` parameter.
            merge_assay_name: Name of assay in the merged Zarr file. For example, for scRNA-Seq it could be simply,
                              'RNA'
            chunk_size: Tuple of cell and feature chunk size. (Default value: (1000, 1000))
            dtype: Dtype of the raw values in the assay. Dtype is automatically inferred from the provided assays. If
                   assays have different dtypes then a float type is used.
            overwrite: If True, then overwrites previously created assay in the Zarr file. (Default value: False)
            reset_cell_filter: If True, then the cell filtering information is removed, i.e. even the filtered out cells
                               are set as True as in the 'I' column. To keep the filtering information set the value for
                               this parameter to False. (Default value: True)
        """
        self.assays = assays
        self.names = names
        self.mergedCells = self._merge_cell_table(reset_cell_filter)
        self.nCells = self.mergedCells.shape[0]
        self.featCollection = self._get_feat_ids(assays)
        self.mergedFeats = self._merge_order_feats()
        self.nFeats = self.mergedFeats.shape[0]
        self.featOrder = self._ref_order_feat_idx()
        self.z = self._use_existing_zarr(zarr_path, merge_assay_name, overwrite)
        self._ini_cell_data()
        if dtype is None:
            if len(set([str(x.rawData.dtype) for x in self.assays])) == 1:
                dtype = str(self.assays[0].rawData.dtype)
            else:
                dtype = 'float'
        self.assayGroup = create_zarr_count_assay(
            self.z['/'], merge_assay_name, chunk_size, self.nCells, list(self.mergedFeats.index),
            list(self.mergedFeats.names.values), dtype
        )

    def _merge_cell_table(self, reset):
        ret_val = []
        if len(self.assays) != len(set(self.names)):
            raise ValueError("ERROR: A unique name should be provided for each of the assay")
        for assay, name in zip(self.assays, self.names):
            a = pd.DataFrame({
                'ids': [f"{name}__{x}" for x in assay.cells.fetch_all('ids')],
                'names': assay.cells.fetch_all('names')
            })
            for i in assay.cells.columns:
                if i not in ['ids', 'I', 'names']:
                    a[f"orig_{i}"] = assay.cells.fetch_all(i)
            if reset:
                a['I'] = np.ones(len(a['ids'])).astype(bool)
            else:
                a['I'] = assay.cells.fetch_all('I')
            ret_val.append(a)
        return pd.concat(ret_val).reset_index(drop=True)

    @staticmethod
    def _get_feat_ids(assays):
        ret_val = []
        for i in assays:
            ret_val.append(i.feats.to_pandas_dataframe(['names', 'ids']).set_index('ids')['names'].to_dict())
        return ret_val

    def _merge_order_feats(self):
        union_set = {}
        for ids in self.featCollection:
            for i in ids:
                if i not in union_set:
                    union_set[i] = ids[i]
        return pd.DataFrame({'idx': range(len(union_set)),
                             'names': list(union_set.values()),
                             'ids': list(union_set.keys())}).set_index('ids')

    def _ref_order_feat_idx(self):
        ret_val = []
        for ids in self.featCollection:
            ret_val.append(self.mergedFeats['idx'].reindex(ids).values)
        return ret_val

    def _use_existing_zarr(self, zarr_path, merge_assay_name, overwrite):
        try:
            z = zarr.open(zarr_path, mode='r')
            if 'cellData' not in z:
                raise ValueError(
                    f"ERROR: Zarr file with name {zarr_path} exists but seems corrupted. Either delete the "
                    "existing file or choose another path")
            if merge_assay_name in z:
                if overwrite is False:
                    raise ValueError(
                        f"ERROR: Zarr file `{zarr_path}` already contains {merge_assay_name} assay. Choose "
                        "a different zarr path or a different assay name. Otherwise set overwrite to True")
            try:
                if not all(z['cellData']['ids'][:] == np.array(self.mergedCells['ids'].values)):
                    raise ValueError(f"ERROR: order of cells does not match the one in existing file: {zarr_path}")
            except KeyError:
                raise ValueError(f"ERROR: 'cell data' in Zarr file {zarr_path} seems corrupted. Either delete the "
                                 "existing file or choose another path")
            return zarr.open(zarr_path, mode='r+')
        except ValueError:
            # So no zarr file with same name exists. Check if a non zarr folder with the same name exists
            if os.path.exists(zarr_path):
                raise ValueError(
                    f"ERROR: Directory/file with name `{zarr_path}`exists. Either delete it or use another name")
            # creating a new zarr file
            return zarr.open(zarr_path, mode='w')

    def _ini_cell_data(self):
        if 'cellData' not in self.z:
            g = self.z.create_group('cellData')
            for i in self.mergedCells:
                vals = self.mergedCells[i].values
                create_zarr_obj_array(g, i, vals, vals.dtype)
        else:
            logger.info(f"cellData already exists so skipping _ini_cell_data")

    def write(self, nthreads=2):
        # TODO: add docstring
        pos_start, pos_end = 0, 0
        for assay, feat_order in zip(self.assays, self.featOrder):
            for i in tqdm(assay.rawData.blocks, total=assay.rawData.numblocks[0],
                          desc=f"Writing data to merged file"):
                pos_end += i.shape[0]
                a = np.ones((i.shape[0], self.nFeats))
                a[:, feat_order] = controlled_compute(i, nthreads)
                self.assayGroup[pos_start:pos_end, :] = a
                pos_start = pos_end


def to_h5ad(assay, h5ad_filename: str) -> None:
    """

    Args:
        assay:
        h5ad_filename:

    Returns:

    """
    import h5py

    def save_attr(group, col, scarf_col, md):
        d = md.fetch_all(scarf_col)
        h5[group].create_dataset(col, data=d.astype(h5py.special_dtype(vlen=str)))

    h5 = h5py.File(h5ad_filename, 'w')
    for i in ['X', 'obs', 'var']:
        h5.create_group(i)

    n_feats_per_cell = assay.cells.fetch_all(f"{assay.name}_nFeatures").astype(int)
    tot_counts = int(n_feats_per_cell.sum())

    for i, s in zip(['indptr', 'indices', 'data'],
                    [assay.cells.N + 1, tot_counts, tot_counts]):
        h5['X'].create_dataset(i, (s,), chunks=True, compression='gzip', dtype=int)
    h5['X/indptr'][:] = np.array([0] + list(n_feats_per_cell.cumsum())).astype(int)
    s, e = 0, 0
    for i in tqdm(assay.rawData.blocks, total=assay.rawData.numblocks[0]):
        i = csr_matrix(i.compute()).astype(int)
        e += i.data.shape[0]
        h5['X/data'][s:e] = i.data
        h5['X/indices'][s:e] = i.indices
        s = e
    save_attr('obs', '_index', 'ids', assay.cells)
    save_attr('var', '_index', 'ids', assay.feats)
    save_attr('var', 'gene_short_name', 'names', assay.feats)

    attrs = {
        'encoding-type': 'csr_matrix',
        'encoding-version': '0.1.0',
        'shape': np.array([assay.cells.N, assay.feats.N])
    }
    for i, j in attrs.items():
        h5['X'].attrs[i] = j

    attrs = {
        '_index': '_index',
        'column-order': np.array(['_index'], dtype=object),
        'encoding-type': 'dataframe',
        'encoding-version': '0.1.0'
    }
    for i, j in attrs.items():
        h5['obs'].attrs[i] = j

    attrs = {
        '_index': '_index',
        'column-order': np.array(['_index', 'gene_short_name'], dtype=object),
        'encoding-type': 'dataframe',
        'encoding-version': '0.1.0'
    }
    for i, j in attrs.items():
        h5['var'].attrs[i] = j

    h5.close()
    return None


def to_mtx(assay, mtx_directory: str, compress: bool = False):
    """

    Args:
        assay: Scarf assay. For example: `ds.RNA`
        mtx_directory: Out directory where MTX file will be saved along with barcodes and features file
        compress: If True, then the files are compressed and saved with .gz extension. (Default value: False).

    Returns:

    """
    from scipy.sparse import coo_matrix
    import gzip

    if os.path.isdir(mtx_directory) is False:
        os.mkdir(mtx_directory)

    n_feats_per_cell = assay.cells.fetch_all(f"{assay.name}_nFeatures").astype(int)
    tot_counts = int(n_feats_per_cell.sum())
    if compress:
        barcodes_fn = 'barcodes.tsv.gz'
        features_fn = 'features.tsv.gz'
        h = gzip.open(os.path.join(mtx_directory, 'matrix.mtx.gz'), 'wt')
    else:
        barcodes_fn = 'barcodes.tsv.gz'
        features_fn = 'genes.tsv'
        h = open(os.path.join(mtx_directory, 'matrix.mtx'), 'w')
    h.write("%%MatrixMarket matrix coordinate integer general\n% Generated by Scarf\n")
    h.write(f"{assay.feats.N} {assay.cells.N} {tot_counts}\n")
    s = 0
    for i in tqdm(assay.rawData.blocks, total=assay.rawData.numblocks[0]):
        i = coo_matrix((i.compute()))
        df = pd.DataFrame({'col': i.col + 1, 'row': i.row + s + 1, 'd': i.data})
        df.to_csv(h, sep=' ', header=False, index=False, mode='a')
        s += i.shape[0]
    h.close()
    assay.cells.to_pandas_dataframe(['ids']).to_csv(
        os.path.join(mtx_directory, barcodes_fn),
        sep='\t', header=False, index=False)

    assay.feats.to_pandas_dataframe(['ids', 'names']).to_csv(
        os.path.join(mtx_directory, features_fn),
        sep='\t', header=False, index=False)
