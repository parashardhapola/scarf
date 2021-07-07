import pytest
import os
import numpy as np
import shutil


@pytest.fixture
def dummy_metadata():
    import zarr
    from ..metadata import MetaData

    fn = os.path.join('scarf', 'tests', 'datasets', 'dummy_metadata.zarr')
    if os.path.isdir(fn):
        shutil.rmtree(fn)
    g = zarr.open(fn)
    data = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1]).astype(bool)
    g.create_dataset('I', data=data, chunks=(100000,),
                     shape=len(data), dtype=data.dtype)
    yield MetaData(g)
    shutil.rmtree(fn)


def test_metadata_attrs(dummy_metadata):
    assert dummy_metadata.N == 9
    assert np.all(dummy_metadata.index == np.array(range(9)))


def test_metadata_fetch(dummy_metadata):
    assert len(dummy_metadata.fetch('I')) == 7
    assert len(dummy_metadata.fetch_all('I')) == 9


def test_metadata_verify_bool(dummy_metadata):
    assert dummy_metadata._verify_bool('I') is True


def test_metadata_active_index(dummy_metadata):
    a = np.array([0, 1, 2, 3, 6, 7, 8])
    assert np.all(dummy_metadata.active_index(key='I') == a)
