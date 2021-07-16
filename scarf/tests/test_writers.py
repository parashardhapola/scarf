from . import full_path, remove


def test_crtozarr(crh5_reader):
    from ..writers import CrToZarr

    fn = full_path('dummy_1K_pbmc_citeseq.zarr')
    writer = CrToZarr(crh5_reader, zarr_fn=fn)
    writer.dump()
    remove(fn)


def test_crtozarr_fromdir(crdir_reader):
    from ..writers import CrToZarr

    fn = full_path('1K_pbmc_citeseq_dir.zarr')
    writer = CrToZarr(crdir_reader, zarr_fn=fn)
    writer.dump()
    remove(fn)


def test_h5adtozarr(h5ad_reader):
    from ..writers import H5adToZarr

    fn = full_path('bastidas.zarr')
    writer = H5adToZarr(h5ad_reader, zarr_fn=fn)
    writer.dump()
    remove(fn)


def test_to_h5ad(datastore):
    # TODO: Evaluate the resulting H5ad file
    from ..writers import to_h5ad

    fn = full_path('test_1K_pbmc_citeseq.h5ad')
    to_h5ad(datastore.RNA, fn)
    remove(fn)


def test_to_mtx(datastore):
    # TODO: Evaluate the resulting MTX directory
    from ..writers import to_mtx

    fn = full_path('test_1K_pbmc_citeseq_dir')
    to_mtx(datastore.RNA, fn)
    remove(fn)


def test_zarr_merge(datastore):
    # TODO: Evaluate the resulting merged file
    from ..writers import ZarrMerge

    fn = full_path('merged_zarr.zarr')
    writer = ZarrMerge(zarr_path=fn, assays=[datastore.RNA, datastore.RNA],
                       names=['self1', 'self2'], merge_assay_name='RNA', prepend_text='')
    writer.dump()
    remove(fn)


def test_zarr_subset(datastore):
    # TODO: Evaluate the resulting subsetted file

    from ..writers import SubsetZarr

    in_fn = full_path('1K_pbmc_citeseq.zarr')
    out_fn = full_path('subset.zarr')

    writer = SubsetZarr(in_zarr=in_fn, out_zarr=out_fn, cell_idx=[1, 10, 100, 500])
    writer.dump()
    remove(out_fn)
