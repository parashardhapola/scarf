import pytest
import os
import shutil


@pytest.fixture
def pbmc_writer(pbmc_reader):
    from ..writers import CrToZarr

    fn = os.path.join('scarf', 'tests', 'datasets', 'dummy_1K_pbmc_citeseq.zarr')
    yield CrToZarr(pbmc_reader, zarr_fn=fn)
    shutil.rmtree(fn)


def test_write_pbmc(pbmc_writer):
    pbmc_writer.dump()
