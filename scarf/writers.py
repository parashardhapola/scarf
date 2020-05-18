import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    import zarr
    from numcodecs import Blosc
    from typing import Any, Tuple
    import numpy as np
    from tqdm import tqdm
    from .readers import CrReader

__all__ = ['CrToZarr', 'subset_assay_zarr', 'create_zarr_dataset', 'create_zarr_obj_array', 'dask_to_zarr']


def create_zarr_dataset(g: zarr.hierarchy, name: str, chunks: tuple,
                        dtype: Any, shape: Tuple, overwrite: bool = True) -> zarr.hierarchy:
    compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE)
    return g.create_dataset(name, chunks=chunks, dtype=dtype,
                            shape=shape, compressor=compressor, overwrite=overwrite)


def create_zarr_obj_array(g: zarr.hierarchy, name: str, data,
                          dtype: str = None, overwrite: bool = True) -> zarr.hierarchy:
    if dtype is None or dtype == object:
        dtype = 'U' + str(max([len(x) for x in data]))
    return g.create_dataset(name, data=data, chunks=False,
                            shape=len(data), dtype=dtype, overwrite=overwrite)


class CrToZarr:
    def __init__(self, cr: CrReader, zarr_fn: str, chunk_sizes=(1000, 1000), dtype: str = 'uint32'):
        self.cr = cr
        self.fn = zarr_fn
        self.chunkSizes = chunk_sizes
        self.z = zarr.open(self.fn, mode='w')
        self._ini_cell_data()
        for assay_name in self.cr.assayFeats.columns:
            g = self.z.create_group(assay_name)
            g.attrs['is_assay'] = True
            g.attrs['misc'] = {}
            create_zarr_dataset(g, 'counts', chunk_sizes, dtype,
                                (self.cr.nCells, self.cr.assayFeats[assay_name]['nFeatures']))
            self._ini_feature_data(assay_name)

    def _ini_cell_data(self):
        g = self.z.create_group('cellData')
        create_zarr_obj_array(g, 'ids', self.cr.cell_names())
        create_zarr_obj_array(g, 'names', self.cr.cell_names())
        create_zarr_obj_array(g, 'I', [True for _ in range(self.cr.nCells)], 'bool')

    def _ini_feature_data(self, assay_name):
        g = self.z[assay_name]
        create_zarr_obj_array(g, 'featureData/ids', self.cr.feature_ids(assay_name))
        create_zarr_obj_array(g, 'featureData/names', self.cr.feature_names(assay_name))
        is_active_data = [True for _ in range(self.cr.assayFeats[assay_name].nFeatures)]
        create_zarr_obj_array(g, 'featureData/I', is_active_data, 'bool')

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


def subset_assay_zarr(zarr_fn: str, in_grp: str, out_grp: str,
                      cells_idx: np.ndarray, feat_idx: np.ndarray,
                      chunk_size: tuple):
    z = zarr.open(zarr_fn, 'r+')
    ig = z[in_grp]
    og = z.create_dataset(
        out_grp, chunks=chunk_size, dtype='uint32', shape=(len(cells_idx), len(feat_idx)),
        compressor=Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE), overwrite=True)
    pos_start, pos_end = 0, 0
    for i in tqdm(np.array_split(cells_idx, len(cells_idx) // chunk_size[0] + 1)):
        pos_end += len(i)
        og[pos_start:pos_end, :] = ig.get_orthogonal_selection((i, feat_idx))
        pos_start = pos_end
    return None


def dask_to_zarr(df, z, loc, chunk_size):
    re_df = df.rechunk(chunks=(chunk_size, df.shape[0]))
    og = z.create_dataset(
        loc, overwrite=True, chunks=(chunk_size, None),
        shape=re_df.shape, dtype='float64',
        compressor=Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE))
    pos_start, pos_end = 0, 0
    for i in tqdm(re_df.blocks, total=len(re_df.chunks[0]), desc=f"Writing data to {loc}"):
        pos_end += len(i)
        og[pos_start:pos_end, :] = i.compute()
        pos_start = pos_end
    return None
