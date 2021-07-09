import pytest
import os
import shutil


@pytest.fixture(scope='session')
def pbmc_writer(pbmc_reader):
    from ..writers import CrToZarr

    fn = os.path.join('scarf', 'tests', 'datasets', 'dummy_1K_pbmc_citeseq.zarr')
    yield CrToZarr(pbmc_reader, zarr_fn=fn)
    shutil.rmtree(fn)


def test_cr_to_zarr(pbmc_writer):
    pbmc_writer.dump()


@pytest.fixture
def h5ad_writer(h5ad_reader):
    from ..writers import H5adToZarr

    fn = os.path.join('scarf', 'tests', 'datasets', 'bastidas.zarr')
    yield H5adToZarr(h5ad_reader, zarr_fn=fn)
    shutil.rmtree(fn)


def test_h5ad_to_zarr(h5ad_writer):
    h5ad_writer.dump()
