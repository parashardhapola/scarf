import scarf
import pytest



def test_H5adReader( ):
    reader = scarf.H5adReader( '/home/med-sal/BMC_Projects/MattiasMagnusson/Pavan_Prabhala/nextseq5.S11.h5ad')
    writer = scarf.H5adToZarr(reader, zarr_fn='testData/TestZarr', chunk_size=(1000, 1000))
    writer.dump(batch_size=1000)
    dr = scarf.DataStore( f'testData/TestZarr' )
    dr
