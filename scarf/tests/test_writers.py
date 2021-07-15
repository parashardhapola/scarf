import pytest
import os
import shutil


def full_path(fn):
    return os.path.join('scarf', 'tests', 'datasets', fn)


@pytest.fixture(scope='session')
def pbmc_writer(pbmc_reader):
    from ..writers import CrToZarr
    fn = full_path('dummy_1K_pbmc_citeseq.zarr')
    yield CrToZarr(pbmc_reader, zarr_fn=fn)
    shutil.rmtree(fn)


def test_cr_to_zarr(pbmc_writer):
    pbmc_writer.dump()


@pytest.fixture
def h5ad_writer(h5ad_reader):
    from ..writers import H5adToZarr

    fn = full_path('bastidas.zarr')
    yield H5adToZarr(h5ad_reader, zarr_fn=fn)
    shutil.rmtree(fn)


def test_h5ad_to_zarr(h5ad_writer):
    h5ad_writer.dump()


def test_to_h5ad(datastore):
    # TODO: Evaluate the resulting H5ad file
    from ..writers import to_h5ad

    fn = full_path('test_1K_pbmc_citeseq.h5ad')
    to_h5ad(datastore.RNA, fn)
    os.unlink(fn)


def test_to_mtx(datastore):
    # TODO: Evaluate the resulting MTX directory
    from ..writers import to_mtx

    fn = full_path('test_1K_pbmc_citeseq_dir')
    to_mtx(datastore.RNA, fn)
    shutil.rmtree(fn)


def test_zarr_merge(datastore):
    # TODO: Evaluate the resulting merged file
    from ..writers import ZarrMerge

    fn = full_path('merged_zarr.zarr')
    writer = ZarrMerge(zarr_path=fn, assays=[datastore.RNA, datastore.RNA],
                       names=['self1', 'self2'], merge_assay_name='RNA', prepend_text='')
    writer.dump()
    shutil.rmtree(fn)


def test_zarr_subset(datastore):
    # TODO: Evaluate the resulting subsetted file

    from ..writers import SubsetZarr

    in_fn = full_path('1K_pbmc_citeseq.zarr')
    out_fn = full_path('subset.zarr')

    writer = SubsetZarr(in_zarr=in_fn, out_zarr=out_fn, cell_idx=[1, 10, 100, 500])
    writer.dump()
    shutil.rmtree(out_fn)
