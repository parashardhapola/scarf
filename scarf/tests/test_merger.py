import zarr

from . import full_path, remove


def test_assay_merge(datastore):
    # TODO: Evaluate the resulting merged file
    """
    Test the AssayMerge class by merging two assays of same type from the same dataset.
    """
    from ..merge import AssayMerge

    fn = full_path("merged_zarr.zarr")
    writer = AssayMerge(
        zarr_path=fn,
        assays=[datastore.RNA, datastore.RNA],
        names=["self1", "self2"],
        merge_assay_name="RNA",
        prepend_text="",
        overwrite=True,
    )
    writer.dump()
    tmp = zarr.open(fn + "/RNA/counts")
    assert tmp.shape[0] == 2 * datastore.cells.N
    assert int(tmp[...].sum()) == int(datastore.RNA.rawData.compute().sum() * 2)
    remove(fn)


def test_dataset_merge_2(datastore):
    """
    Test the DatasetMerge class by merging two datasets. This will test the merging of all assays in the dataset.
    """
    from ..merge import DatasetMerge

    fn = full_path("merged_zarr.zarr")
    writer = DatasetMerge(
        zarr_path=fn,
        datasets=[datastore, datastore],
        names=["self1", "self2"],
        prepend_text="",
        overwrite=True
    )
    writer.dump()
    # Check if the merged file has the correct shape and counts
    # Load the merged counts file
    rna_count = zarr.open(fn + "/RNA/counts")
    assay2_count = zarr.open(fn + "/assay2/counts")
    # We merged only two datasets, so the shape should be 2*original_shape
    # Check the number of cells
    assert rna_count.shape[0] == 2 * datastore.cells.N
    assert assay2_count.shape[0] == 2 * datastore.cells.N
    # Check the count values through sum
    assert int(rna_count[...].sum()) == int(datastore.RNA.rawData.compute().sum() * 2)
    assert int(assay2_count[...].sum()) == int(
        datastore.assay2.rawData.compute().sum() * 2
    )
    remove(fn)


def test_dataset_merge_3(datastore):
    """
    Test the DatasetMerge class by merging two datasets. This will test the merging of all assays in the dataset.
    If merging is successful for more than two datasets, it should work for any number of datasets.
    """
    from ..merge import DatasetMerge

    fn = full_path("merged_zarr.zarr")
    writer = DatasetMerge(
        zarr_path=fn,
        datasets=[datastore, datastore, datastore],
        names=["self1", "self2", "self3"],
        prepend_text="",
        overwrite=True
    )
    writer.dump()
    # Check if the merged file has the correct shape and counts
    # Load the merged counts file
    rna_count = zarr.open(fn + "/RNA/counts")
    assay2_count = zarr.open(fn + "/assay2/counts")
    # We merged only three datasets, so the shape should be 3*original_shape
    # Check the number of cells
    assert rna_count.shape[0] == 3 * datastore.cells.N
    assert assay2_count.shape[0] == 3 * datastore.cells.N
    # Check the count values through sum
    assert int(rna_count[...].sum()) == int(datastore.RNA.rawData.compute().sum() * 3)
    assert int(assay2_count[...].sum()) == int(
        datastore.assay2.rawData.compute().sum() * 3
    )
    remove(fn)

def test_dataset_merge_cells(datastore):
    from ..merge import DatasetMerge
    from ..datastore.datastore import DataStore

    fn = full_path("merged_zarr.zarr")
    writer = DatasetMerge(
        zarr_path=fn,
        datasets=[datastore, datastore],
        names=["self1", "self2"],
        prepend_text="orig",
        overwrite=True,
    )
    writer.dump()

    ds = DataStore(
        fn,
        default_assay="RNA",
    )
    
    df = ds.cells.to_pandas_dataframe(ds.cells.columns)
    df_diff = df[df['orig_RNA_nCounts'] != df['RNA_nCounts']]
    assert len(df_diff) == 0