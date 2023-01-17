from . import full_path, remove
import numpy as np

def test_crtozarr(crh5_reader):
    from ..writers import CrToZarr

    fn = full_path("dummy_1K_pbmc_citeseq.zarr")
    writer = CrToZarr(crh5_reader, zarr_loc=fn)
    writer.dump()
    remove(fn)


def test_crtozarr_fromdir(crdir_reader):
    from ..writers import CrToZarr

    fn = full_path("1K_pbmc_citeseq_dir.zarr")
    writer = CrToZarr(crdir_reader, zarr_loc=fn)
    writer.dump()
    remove(fn)


def test_h5adtozarr(h5ad_reader):
    from ..writers import H5adToZarr

    fn = full_path("bastidas.zarr")
    writer = H5adToZarr(h5ad_reader, zarr_loc=fn)
    writer.dump()
    remove(fn)


def test_loomtozarr(loom_reader):
    from ..writers import LoomToZarr

    fn = full_path("sympathetic.zarr")
    writer = LoomToZarr(loom_reader, zarr_fn=fn)
    writer.dump()
    remove(fn)


def test_sparsetozarr():
    from ..writers import SparseToZarr
    from scipy.sparse import csr_matrix

    cols = [1, 3, 8, 2, 3, 1, 2, 8, 9]
    rows = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    data = [1, 10, 15, 10, 20, 2, 3, 1, 5]
    mat = (data, (rows, cols))
    mat = csr_matrix(mat, shape=(3, 10))

    fn = full_path("dummy_sparse.zarr")

    writer = SparseToZarr(
        mat,
        zarr_fn=fn,
        cell_ids=[f"cell_{x}" for x in range(3)],
        feature_ids=[f"feat_{x}" for x in range(10)],
    )
    writer.dump()
    remove(fn)


def test_to_h5ad(datastore):
    # TODO: Evaluate the resulting H5ad file
    from ..writers import to_h5ad

    fn = full_path("test_1K_pbmc_citeseq.h5ad")
    to_h5ad(datastore.RNA, fn)
    remove(fn)


def test_to_mtx(datastore):
    # TODO: Evaluate the resulting MTX directory
    from ..writers import to_mtx

    fn = full_path("test_1K_pbmc_citeseq_dir")
    to_mtx(datastore.RNA, fn)
    remove(fn)


def test_zarr_merge(datastore):
    # TODO: Evaluate the resulting merged file
    from ..writers import ZarrMerge

    fn = full_path("merged_zarr.zarr")
    writer = ZarrMerge(
        zarr_path=fn,
        assays=[datastore.RNA, datastore.RNA],
        names=["self1", "self2"],
        merge_assay_name="RNA",
        prepend_text="",
    )
    writer.dump()
    remove(fn)


def test_zarr_subset(datastore):
    # TODO: Evaluate the resulting subsetted file

    from ..writers import SubsetZarr

    zarr_path = full_path("subset.zarr")
    writer = SubsetZarr(
        zarr_loc=zarr_path, assays=[datastore.RNA], cell_idx=np.array([1, 10, 100, 500])
    )
    writer.dump()
    remove(zarr_path)
