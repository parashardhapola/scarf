import scarf
import pytest



def test_H5adReader( ):
    assert scarf.H5adReader( 'testData/tiny2.h5ad')
