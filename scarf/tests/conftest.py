import pytest
import os


@pytest.fixture
def pbmc_reader(scope="module"):
    from ..readers import CrH5Reader

    fn = os.path.join('scarf', 'tests', 'datasets', '1K_pbmc_citeseq.h5')
    return CrH5Reader(fn, 'rna')
