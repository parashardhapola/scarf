import pytest
from . import full_path, remove


@pytest.fixture(scope='session')
def crh5_reader():
    from ..readers import CrH5Reader

    return CrH5Reader(full_path('1K_pbmc_citeseq.h5'), 'rna')


@pytest.fixture(scope='session')
def crdir_reader(datastore):
    from ..readers import CrDirReader
    from ..writers import to_mtx

    fn = full_path('1K_pbmc_citeseq_dir')
    # TODO: This is not ideal. Chaining and creating test
    # dependencies like this is not a good idea.
    to_mtx(datastore.RNA, fn, compress=True)
    reader = CrDirReader(fn, 'rna')
    yield reader
    remove(fn)


@pytest.fixture(scope='session')
def h5ad_reader(bastidas_ponce_data):
    from ..readers import H5adReader

    reader = H5adReader(bastidas_ponce_data)
    yield reader
    reader.h5.close()
