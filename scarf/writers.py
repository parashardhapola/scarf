import zarr
from typing import Any, Tuple, List
import numpy as np
from tqdm import tqdm
from .readers import CrReader
import os
import pandas as pd
# from .assay import Assay  # Disabled because of circular dependency

__all__ = ['CrToZarr', 'create_zarr_dataset', 'create_zarr_obj_array', 'create_zarr_count_assay',
           'subset_assay_zarr', 'dask_to_zarr', 'ZarrMerge', 'AnndataToZarr']


def create_zarr_dataset(g: zarr.hierarchy, name: str, chunks: tuple,
                        dtype: Any, shape: Tuple, overwrite: bool = True) -> zarr.hierarchy:
    from numcodecs import Blosc

    compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE)
    return g.create_dataset(name, chunks=chunks, dtype=dtype,
                            shape=shape, compressor=compressor, overwrite=overwrite)


def create_zarr_obj_array(g: zarr.hierarchy, name: str, data,
                          dtype: str = None, overwrite: bool = True) -> zarr.hierarchy:
    if dtype is None or dtype == object:
        dtype = 'U' + str(max([len(x) for x in data]))
    return g.create_dataset(name, data=data, chunks=False,
                            shape=len(data), dtype=dtype, overwrite=overwrite)


def create_zarr_count_assay(z: zarr.hierarchy, assay_name: str, chunk_size: Tuple[int, int], n_cells: int,
                            feat_ids: List[str], feat_names: List[str], dtype: str = 'uint32') -> zarr.hierarchy:
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


class AnndataToZarr:
    def __init__(self, h5ad, zarr_fn: str, assay_locations: list = None, assay_names: list = None,
                 chunk_size=(1000, 1000), dtype: str = 'uint32'):
        self.h5ad = h5ad
        self.fn = zarr_fn
        self.chunkSizes = chunk_size
        if assay_locations is None:
            print(f"INFO: no value provided for assay locations. Will use default value: ['X']")
            self.assayLocations = ['X']
        else:
            self.assayLocations = assay_locations
        if assay_names is None:
            print(f"INFO: no value provided for assay names. Will use default value: ['RNA']")
            self.assayNames = ['RNA']
        else:
            self.assayNames = assay_names
        if len(self.assayLocations) != len(self.assayNames):
            raise ValueError("ERROR: Number of entries in parameters `assay_locations` and "
                             "`assay_names` should be equal")
        self.z = zarr.open(self.fn, mode='w')
        self._ini_cell_data()
        for i in self.assayNames:
            create_zarr_count_assay(self.z, i, chunk_size, self.h5ad.nCells,
                                    self.h5ad.feat_ids(), self.h5ad.feat_names(), dtype)

    def _ini_cell_data(self):
        g = self.z.create_group('cellData')
        create_zarr_obj_array(g, 'ids', self.h5ad.cell_names())
        create_zarr_obj_array(g, 'names', self.h5ad.cell_names())
        create_zarr_obj_array(g, 'I', [True for _ in range(self.h5ad.nCells)], 'bool')
        for i, j in self.h5ad.get_cell_columns():
            # TODO: use `_categories` under `uns` slot to the values rather than indices
            create_zarr_obj_array(g, i, j, j.dtype)

    def dump(self, batch_size: int = 1000) -> None:
        for assay_name, assay_loc in zip(self.assayNames, self.assayLocations):
            store = self.z["%s/counts" % assay_name]
            s, e, = 0, 0
            n_chunks = self.h5ad.nCells//batch_size + 1
            for a in tqdm(self.h5ad.consume(batch_size, data_loc=assay_loc), total=n_chunks):
                e += a.shape[0]
                a = a.todense()
                store[s:e] = a
                s = e


def subset_assay_zarr(zarr_fn: str, in_grp: str, out_grp: str,
                      cells_idx: np.ndarray, feat_idx: np.ndarray,
                      chunk_size: tuple):
    z = zarr.open(zarr_fn, 'r+')
    ig = z[in_grp]
    og = create_zarr_dataset(z, out_grp, chunk_size, 'uint32', (len(cells_idx), len(feat_idx)))
    pos_start, pos_end = 0, 0
    for i in tqdm(np.array_split(cells_idx, len(cells_idx) // chunk_size[0] + 1)):
        pos_end += len(i)
        og[pos_start:pos_end, :] = ig.get_orthogonal_selection((i, feat_idx))
        pos_start = pos_end
    return None


def dask_to_zarr(df, z, loc, chunk_size, msg: str = None):
    if msg is None:
        msg = f"Writing data to {loc}"
    og = create_zarr_dataset(z, loc, chunk_size, 'float64', df.shape)
    pos_start, pos_end = 0, 0
    for i in tqdm(df.blocks, total=df.numblocks[0], desc=msg):
        pos_end += i.shape[0]
        og[pos_start:pos_end, :] = i.compute()
        pos_start = pos_end
    return None


class ZarrMerge:

    def __init__(self, zarr_path: str, assays: list, names: List[str], merge_assay_name: str,
                 chunk_size=(1000, 1000), dtype: str = 'uint32', overwrite: bool = False):
        self.assays = assays
        self.names = names
        self.mergedCells = self._merge_cell_table()
        self.nCells = self.mergedCells.shape[0]
        self.featCollection = self._get_feat_ids(assays)
        self.mergedFeats = self._merge_order_feats()
        self.nFeats = self.mergedFeats.shape[0]
        self.featOrder = self._ref_order_feat_idx()
        self.z = self._use_existing_zarr(zarr_path, merge_assay_name, overwrite)
        self._ini_cell_data()
        self.assayGroup = create_zarr_count_assay(
            self.z['/'], merge_assay_name, chunk_size, self.nCells, list(self.mergedFeats.index),
            list(self.mergedFeats.names.values), dtype
        )

    def _merge_cell_table(self):
        ret_val = []
        if len(self.assays) != len(set(self.names)):
            raise ValueError("ERROR: A unique name should be provided for each of the assay")
        for assay, name in zip(self.assays, self.names):
            a = assay.cells.table[['names']].copy()
            a['ids'] = [f"{name}__{x}" for x in assay.cells.table['ids']]
            ret_val.append(a)
        return pd.concat(ret_val).reset_index().drop(columns='index')

    @staticmethod
    def _get_feat_ids(assays):
        ret_val = []
        for i in assays:
            ret_val.append(i.feats.table[['names', 'ids']].set_index('ids')['names'].to_dict())
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
            create_zarr_obj_array(g, 'ids', list(self.mergedCells['ids'].values))
            create_zarr_obj_array(g, 'names', list(self.mergedCells['names']))
            create_zarr_obj_array(g, 'I', [True for _ in range(self.mergedCells.shape[0])], 'bool')
        else:
            print(f"INFO: cellData already exists so skipping _ini_cell_data", flush=True)

    def write(self):
        pos_start, pos_end = 0, 0
        for assay, feat_order in zip(self.assays, self.featOrder):
            for i in tqdm(assay.rawData.blocks, total=assay.rawData.numblocks[0],
                          desc=f"Writing aligned normed target data"):
                pos_end += i.shape[0]
                a = np.ones((i.shape[0], self.nFeats))
                a[:, feat_order] = i.compute()
                self.assayGroup[pos_start:pos_end, :] = a
                pos_start = pos_end
