import pytest
from . import full_path, remove


@pytest.fixture(scope='session')
def crh5_reader():
    from ..readers import CrH5Reader

    return CrH5Reader(full_path('1K_pbmc_citeseq.h5'), 'rna')


@pytest.fixture(scope='session')
def mtx_dir(datastore):
    from ..writers import to_mtx

    fn = full_path('1K_pbmc_citeseq_dir')
    to_mtx(datastore.RNA, fn, compress=True)
    yield fn
    remove(fn)


@pytest.fixture(scope='session')
def crdir_reader(datastore, mtx_dir):
    from ..readers import CrDirReader

    reader = CrDirReader(mtx_dir, 'rna')
    yield reader


@pytest.fixture(scope='session')
def mtx_reader(datastore, mtx_dir):
    from ..readers import MtxDirReader

    reader = MtxDirReader(mtx_dir, 'rna')
    yield reader


@pytest.fixture(scope='session')
def h5ad_reader(bastidas_ponce_data):
    from ..readers import H5adReader

    reader = H5adReader(bastidas_ponce_data)
    yield reader
    reader.h5.close()


@pytest.fixture(scope='session')
def loom_reader():
    from ..readers import LoomReader

    reader = LoomReader(full_path('sympathetic.loom'), cell_names_key='Cell_id',
                        feature_names_key='Gene')
    yield reader
    reader.h5.close()
