import numpy as np
import zarr
from . import full_path, remove


def test_assay_merge(datastore):
    # TODO: Evaluate the resulting merged file
    from ..merge import AssayMerge

    fn = full_path("merged_zarr.zarr")
    writer = AssayMerge(
        zarr_path=fn,
        assays=[datastore.RNA, datastore.RNA],
        names=["self1", "self2"],
        merge_assay_name="RNA",
        prepend_text="",
    )
    writer.dump()
    tmp = zarr.open(fn+"/RNA/counts")
    assert tmp.shape[0] == 2*datastore.cells.N
    assert int(tmp[...].sum()) == int(datastore.RNA.rawData.compute().sum()*2)
    remove(fn)

def test_dataset_merge(datastore):
    from ..merge import DatasetMerge
    fn = full_path("merged_zarr.zarr")
    writer = DatasetMerge(
        zarr_path=fn,
        datasets=[datastore, datastore],
        names=["self1", "self2"],
        prepend_text="",
    )
    writer.dump()
    tmp1 = zarr.open(fn+"/RNA/counts")
    tmp2 = zarr.open(fn+"/assay2/counts")
    assert tmp1.shape[0] == 2*datastore.cells.N
    assert tmp2.shape[0] == 2*datastore.cells.N
    assert int(tmp1[...].sum()) == int(datastore.RNA.rawData.compute().sum()*2)
    assert int(tmp2[...].sum()) == int(datastore.assay2.rawData.compute().sum()*2)
    remove(fn)