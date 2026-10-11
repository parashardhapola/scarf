import pytest

from . import full_path, remove


@pytest.fixture(scope="session")
def bastidas_ponce_data():
    from ..downloader import fetch_dataset

    sample = "bastidas-ponce_4K_pancreas-d15_rnaseq"
    fetch_dataset(sample, full_path(None))
    yield full_path(sample, "data.h5ad")
    remove(full_path(sample))
