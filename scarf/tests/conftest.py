import pytest
import os
import shutil


@pytest.fixture(scope='session')
def pbmc_reader():
    from ..readers import CrH5Reader

    fn = os.path.join('scarf', 'tests', 'datasets', '1K_pbmc_citeseq.h5')
    return CrH5Reader(fn, 'rna')


@pytest.fixture(scope='session')
def h5ad_reader():
    from ..readers import H5adReader
    from ..downloader import fetch_dataset

    sample = 'bastidas-ponce_4K_pancreas-d15_rnaseq'
    out_dir = os.path.join('scarf', 'tests', 'datasets')
    fetch_dataset(sample, save_path=out_dir)
    h5ad_fn = os.path.join(out_dir, sample, 'data.h5ad')
    reader = H5adReader(h5ad_fn)
    yield reader
    reader.h5.close()
    shutil.rmtree(os.path.join(out_dir, sample))
