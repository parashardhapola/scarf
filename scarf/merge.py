"""
Methods and classes for merging datasets

"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import zarr
from dask.array import from_array
from dask.array.core import Array as daskArrayType
from scipy.sparse import coo_matrix

from .assay import (
    ADTassay,
    ATACassay,
    RNAassay,
)
from .datastore.datastore import DataStore
from .metadata import MetaData
from .utils import (
    ZARRLOC,
    controlled_compute,
    load_zarr,
    logger,
    permute_into_chunks,
    tqdmbar,
)
from .writers import create_zarr_count_assay, create_zarr_obj_array

__all__ = [
    "DatasetMerge",
    "AssayMerge",
    "ZarrMerge",
]


# Creating a dummy Assay object
class DummyAssay:
    """
    A dummy assay object to be used in the AssayMerge class when an assay is missing in a dataset.
    """

    def __init__(
        self,
        ds: DataStore,
        counts: daskArrayType,
        feats: MetaData,
    ):
        self.rawData = counts
        self.feats = feats
        self.cells = ds.cells


class AssayMerge:
    # class ZarrMerge is renamed to AssayMerge for better understanding
    """Merge multiple Zarr files into a single Zarr file.

    Args:
        zarr_path: Name of the new, merged Zarr file with path.
        assays: List of assay objects to be merged. For example, [ds1.RNA, ds2.RNA].
        names: Names of each of the assay objects in the `assays` parameter. They should be in the same order as in
               `assays` parameter.
        merge_assay_name: Name of assay in the merged Zarr file. For example, for scRNA-Seq it could be simply,
                          'RNA'.
        chunk_size: Tuple of cell and feature chunk size. (Default value: (1000, 1000)).
        dtype: Dtype of the raw values in the assay. Dtype is automatically inferred from the provided assays. If
               assays have different dtypes then a float type is used.
        overwrite: If True, then overwrites previously created assay in the Zarr file. (Default value: False).
        prepend_text: This text is pre-appended to each column name (Default value: 'orig').
        reset_cell_filter: If True, then the cell filtering information is removed, i.e. even the filtered out cells
                           are set as True as in the 'I' column. To keep the filtering information set the value for
                           this parameter to False. (Default value: True)

    Attributes:
        assays: List of assay objects to be merged. For example, [ds1.RNA, ds2.RNA].
        names: Names of each assay objects in the `assays` parameter.
        mergedCells:
        nCells: Number of cells in dataset.
        featCollection:
        mergedFeats:
        nFeats: Number of features in the dataset.
        featOrder:
        z: The merged Zarr file.
        assayGroup:
    """

    def __init__(
        self,
        zarr_path: ZARRLOC,
        assays: List[Union[RNAassay, ATACassay, ADTassay]],
        names: List[str],
        merge_assay_name: str,
        in_workspaces: Union[list[str], None] = None,
        out_workspace: Union[str, None] = None,
        chunk_size=(1000, 1000),
        dtype: Optional[str] = None,
        overwrite: bool = False,
        prepend_text: Optional[str] = "orig",
        reset_cell_filter: bool = True,
        seed: Optional[int] = 42,
    ):
        self.assays = assays
        self.names = names
        self.inWorkspaces = in_workspaces
        self.outWorkspace = out_workspace
        self.merge_assay_name = merge_assay_name
        self.chunk_size = chunk_size
        (
            self.permutations_rows,
            self.permutations_rows_offset,
            self.coordinates_permutations,
        ) = self.perfrom_randomization_rows(seed)
        self.mergedCells: pl.DataFrame = self._merge_cell_table(
            reset_cell_filter, prepend_text
        )
        self.nCells: int = self.mergedCells.shape[0]
        self.featCollection: List[Dict[str, str]] = self._get_feat_ids(assays)
        self.mergedFeats: pl.DataFrame = self._merge_order_feats()
        self.nFeats: int = self.mergedFeats.shape[0]
        self.featOrder: List[np.ndarray] = self._ref_order_feat_idx()
        self.cellOrder: Dict[int, Dict[int, np.ndarray]] = self._ref_order_cell_idx()
        self.z: zarr.Group = self._use_existing_zarr(
            zarr_path, merge_assay_name, overwrite
        )
        self._ini_cell_data(overwrite)
        if dtype is None:
            if len(set([str(x.rawData.dtype) for x in self.assays])) == 1:
                dtype = str(self.assays[0].rawData.dtype)
            else:
                dtype = "float"

        self.assayGroup = create_zarr_count_assay(
            z=self.z,
            assay_name=merge_assay_name,
            workspace=self.outWorkspace,
            chunk_size=chunk_size,
            n_cells=self.nCells,
            feat_ids=np.array(self.mergedFeats["ids"]),
            feat_names=np.array(self.mergedFeats["names"]),
            dtype=dtype,
        )

    def perfrom_randomization_rows(
        self, seed: Optional[int] = 42
    ) -> Tuple[
        Dict[int, Dict[int, np.ndarray]], Dict[int, Dict[int, np.ndarray]], np.ndarray
    ]:
        """
        Perform randomization of rows in the assays.
        Args:
            seed: Seed for randomization
        Returns:
        """
        np.random.seed(seed)
        chunkSize = np.array([x.rawData.chunksize[0] for x in self.assays])
        nCells = np.array([x.rawData.shape[0] for x in self.assays])
        permutations = {
            i: permute_into_chunks(nCells[i], chunkSize[i])
            for i in range(len(self.assays))
        }

        permutations_rows = {}
        for key, arrays in permutations.items():
            in_dict = {i: x for i, x in enumerate(arrays)}
            permutations_rows[key] = in_dict

        permutations_rows_offset = {}
        for i in range(len(permutations)):
            in__dict: dict[int, np.ndarray] = {}
            last_key = i - 1 if i > 0 else 0
            offset = nCells[last_key] + offset if i > 0 else 0
            for j, arr in enumerate(permutations[i]):
                in__dict[j] = arr + offset
            permutations_rows_offset[i] = in__dict

        coordinates = []
        extra = []
        for i in range(len(self.assays)):
            for j in range(len(permutations[i])):
                if j == len(permutations[i]) - 1:  # if j is last, append extra
                    extra.append([i, j])
                    continue
                coordinates.append([i, j])

        coordinates_permutations = np.random.permutation(coordinates)
        if len(coordinates_permutations) > 0:
            coordinates_permutations = np.concatenate(
                [coordinates_permutations, extra], axis=0
            )
        else:
            coordinates_permutations = np.array(extra)

        try:
            assert permutations_rows_offset[0][0].min() == 0
        except AssertionError:
            raise AssertionError(
                "ERROR: Randomization of rows failed. The first row should be at 0.",
                "Please report this issue.",
            )
        try:
            assert (
                permutations_rows_offset[list(permutations_rows_offset.keys())[-1]][
                    list(
                        permutations_rows_offset[
                            list(permutations_rows_offset.keys())[-1]
                        ].keys()
                    )[-1]
                ].max()
                == nCells.sum() - 1
            )
        except AssertionError:
            raise AssertionError(
                "ERROR: Randomization of rows failed. The last row should be at the end of the dataset.",
                "Please report this issue.",
            )
        return permutations_rows, permutations_rows_offset, coordinates_permutations

    def _ref_order_cell_idx(self) -> Dict[int, Dict[int, np.ndarray]]:
        new_cells = {}
        for i in range(len(self.assays)):
            in_dict: dict[int, np.ndarray] = {}
            for j in range(len(self.permutations_rows[i])):
                in_dict[j] = np.array([])
            new_cells[i] = in_dict

        offset = 0
        for i, (x, y) in enumerate(self.coordinates_permutations):
            arr = self.permutations_rows_offset[x][y]
            arr = np.array(range(len(arr)))
            arr = arr + offset
            new_cells[x][y] = arr
            offset = arr.max() + 1

        return new_cells

    def _merge_cell_table(
        self, reset: bool, prepend_text: Optional[str] = None
    ) -> pl.DataFrame:
        """Merges the cell metadata table for each sample.

        Args:
            reset: whether to remove filtering information
            prepend_text: string to add as prefix for each cell column

        Returns:
        """
        # TODO: This method is not very memory efficient
        # Update: Implemented Polars for better memory efficiency

        if len(self.assays) != len(set(self.names)):
            raise ValueError(
                "ERROR: A unique name should be provided for each of the assay"
            )
        if prepend_text == "":
            prepend_text = None
        ret_val = []
        for assay, name in zip(self.assays, self.names):
            a = assay.cells.to_polars_dataframe(assay.cells.columns)
            a = a.with_columns(
                [pl.Series("ids", np.array([f"{name}__{x}" for x in a["ids"]]))]
            )
            for i in a.columns:
                if i not in ["ids", "I", "names"] and prepend_text is not None:
                    a = a.with_columns(
                        [pl.Series(f"{prepend_text}_{i}", assay.cells.fetch_all(i))]
                    )
                    a = a.drop([i])
            if reset:
                a = a.with_columns(
                    [pl.Series("I", np.ones(len(a["ids"])).astype(bool))]
                )
            ret_val.append(a)

        ret_val_df = pl.concat(
            ret_val,
            how="diagonal",  # Finds a union between the column schemas and fills missing column values with null
        )

        # Randomize the rows in chunks
        compiled_idx = [
            self.permutations_rows_offset[i][j]
            for i, j in self.coordinates_permutations
        ]
        compiled_idx = np.concatenate(compiled_idx)
        ret_val_df = ret_val_df[
            compiled_idx
        ]  # Polars does not support iloc so we have to use this method
        if sum([x.cells.N for x in self.assays]) != ret_val_df.shape[0]:
            raise AssertionError(
                "Unexpected number of cells in the merged table. This is unexpected, "
                " please report this bug"
            )
        return ret_val_df

    @staticmethod
    def _get_feat_ids(assays) -> List[Dict[str, str]]:
        """Fetches ID->names mapping of features from each assay.

        Args:
            assays: List of Assay objects

        Returns:
            A list of dictionaries. Each dictionary is a id to name
            mapping for each feature in the corresponding assay
        """
        ret_val = []
        for i in assays:
            df = i.feats.to_polars_dataframe(["names", "ids"])
            ret_val.append(dict(zip(df["ids"].to_numpy(), df["names"].to_numpy())))
        return ret_val

    def _merge_order_feats(self) -> pl.DataFrame:
        """Merge features from all the assays and determine their order.

        Returns:
        """
        union_set = {}
        for ids in self.featCollection:
            for i in ids:
                if i not in union_set:
                    union_set[i] = ids[i]
        ret_val = pl.DataFrame(
            {
                "idx": range(len(union_set)),
                "names": list(union_set.values()),
                "ids": list(union_set.keys()),
            }
        )

        r = ret_val.shape[0] / sum([x.feats.N for x in self.assays])
        if r == 1:
            raise ValueError(
                "No overlapping features found! Will not merge the files. Please check the features ids "
                " are comparable across the assays"
            )
        if r > 0.9:
            logger.warning("The number overlapping features is very low.")
        return ret_val

    def _ref_order_feat_idx(self) -> List[np.ndarray]:
        ret_val = []
        for ids in self.featCollection:
            vals = self.mergedFeats.filter(pl.col("ids").is_in(list(ids.keys())))["idx"]
            ret_val.append(np.array(vals))
        return ret_val

    def _use_existing_zarr(
        self, zarr_loc: ZARRLOC, merge_assay_name, overwrite
    ) -> zarr.Group:
        if self.outWorkspace is None:
            cell_slot = "cellData"
            assay_slot = merge_assay_name
        else:
            cell_slot = f"{self.outWorkspace}/cellData"
            assay_slot = f"{self.outWorkspace}/merge_assay_name"

        try:
            z = load_zarr(zarr_loc, mode="r")
            if cell_slot not in z:
                raise ValueError(
                    f"ERROR: Zarr file exists but seems corrupted. Either delete the "  # noqa: F541
                    "existing file or choose another path"
                )
            if assay_slot in z:
                if overwrite is False:
                    raise ValueError(
                        f"ERROR: Zarr file already contains {merge_assay_name} assay. Choose "
                        "a different zarr path or a different assay name. Otherwise set overwrite to True"
                    )
            try:
                if not all(
                    z[cell_slot]["ids"][:]
                    == np.array(np.array(self.mergedCells["ids"]))  # type: ignore
                ):
                    raise ValueError(
                        f"ERROR: order of cells does not match the one in existing file"  # noqa: F541
                    )
            except KeyError:
                raise ValueError(
                    f"ERROR: 'cell data seems corrupted. Either delete the "  # noqa: F541
                    "existing file or choose another path"
                )
            return load_zarr(zarr_loc, mode="r+")
        except ValueError:
            # So no zarr file with same name exists. Check if a non zarr folder with the same name exists
            if isinstance(zarr_loc, str) and os.path.exists(zarr_loc):
                raise ValueError(
                    f"ERROR: Directory/file with name `{zarr_loc}`exists. "
                    f"Either delete it or use another name"
                )
            # creating a new zarr file
            return load_zarr(zarr_loc, mode="w")

    def _ini_cell_data(self, overwrite) -> None:
        """Save cell attributes to Zarr.

        Returns:
            None
        """
        if self.outWorkspace is None:
            cell_slot = "cellData"
        else:
            cell_slot = f"{self.outWorkspace}/cellData"

        if (cell_slot in self.z and overwrite is True) or cell_slot not in self.z:
            g = self.z.create_group(cell_slot, overwrite=True)
            for i in self.mergedCells.columns:
                vals = np.array(self.mergedCells[i])
                create_zarr_obj_array(g, str(i), vals, vals.dtype, overwrite=True)
        else:
            logger.info(
                f"cellData already exists so skipping _ini_cell_data"  # noqa: F541
            )

    def _dask_to_coo(self, d_arr, order: np.ndarray, n_threads: int) -> coo_matrix:
        mat = np.zeros((d_arr.shape[0], self.nFeats))
        mat[:, order] = controlled_compute(d_arr, n_threads)
        return coo_matrix(mat)

    def dump(self, nthreads=2):
        """Copy the values from individual assays to the merged assay.

        Args:
            nthreads: Number of compute threads to use. (Default value: 2)

        Returns:
        """
        counter = 0
        for i, (assay, feat_order) in enumerate(zip(self.assays, self.featOrder)):
            for j, block in tqdmbar(
                enumerate(assay.rawData.blocks),
                total=assay.rawData.numblocks[0],
                desc=f"Writing data to merged file",  # noqa: F541
            ):
                a = self._dask_to_coo(block, feat_order, nthreads)
                row_idx = self.cellOrder[i][j]
                self.assayGroup.set_coordinate_selection(
                    (a.row + row_idx.min(), a.col), a.data.astype(self.assayGroup.dtype)
                )
                counter += a.shape[0]
        try:
            assert counter == self.nCells
        except AssertionError:
            raise AssertionError(
                "ERROR: Mismatch in number of cells in the merged assay. Please report this issue."
            )


# Alias for ZarrMerge
class ZarrMerge(AssayMerge):
    """
    Alias for AssayMerge for backward compatibility.
    """

    def __init__(self, *args, **kwargs):
        logger.warning(
            "The 'ZarrMerge' class is deprecated and will be removed in a future release. Please use 'AssayMerge' instead."
        )
        super().__init__(*args, **kwargs)


class DatasetMerge:
    """
    Merge multiple datastores, handling different assay types and generating missing assays on the fly.

    Args:
        datasets: List of DataStore objects to be merged.
        zarr_path: Name of the new, merged Zarr file with path.
        names: Names of each of the dataset objects in the `datasets` parameter. They should be in the same order as in
               `datasets` parameter.
        in_workspaces: List of workspaces to be merged. If None, all workspaces are merged.
        out_workspace: Name of the workspace in the merged Zarr file. If None, the name of the first workspace is used.
        chunk_size: Tuple of cell and feature chunk size. (Default value: (1000, 1000)).
        dtype: Dtype of the raw values in the assay. Dtype is automatically inferred from the provided assays. If
               assays have different dtypes then a float type is used.
        overwrite: If True, then overwrites previously created assay in the Zarr file. (Default value: False).
        prepend_text: This text is pre-appended to each column name (Default value: 'orig').
        reset_cell_filter: If True, then the cell filtering information is removed, i.e. even the filtered out cells
                           are set as True as in the 'I' column. To keep the filtering information set the value for
                           this parameter to False. (Default value: True)
        seed: Seed for randomization of rows in the assays.

    Example:
        >>> # Assuming ds1, ds2 and ds3 are DataStore objects
        >>> # ds1 has RNA and ADT assays. ds2 has RNA assay. ds3 has ADT assay.
        >>> # Merge RNA and ADT assays from all the datastores
        >>> merge = DatasetMerge(
        >>>     datasets=[ds1, ds2, ds3],
        >>>     zarr_path="merged.zarr",
        >>>     names=["ds1", "ds2", "ds3"],
        >>>     overwrite = True
        >>> )
        >>> merge.dump()
        >>> # The merged.zarr file will have RNA and ADT assays from all the datastores
    """

    def __init__(
        self,
        datasets: List[DataStore],
        zarr_path: ZARRLOC,
        names: List[str],
        in_workspaces: Union[list[str], None] = None,
        out_workspace: Union[str, None] = None,
        chunk_size=(1000, 1000),
        dtype: Optional[str] = None,
        overwrite: bool = False,
        prepend_text: Optional[str] = "orig",
        reset_cell_filter: bool = True,
        seed: Optional[int] = 42,
    ):
        self.datasets = datasets
        self.names = names
        self.zarr_path = zarr_path
        self.in_workspaces = in_workspaces
        self.out_workspace = out_workspace
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.overwrite = overwrite
        self.prepend_text = prepend_text
        self.reset_cell_filter = reset_cell_filter
        self.seed = seed
        self.unique_assays = self.get_unique_assays()
        self.n_unique_assays = len(self.unique_assays)
        self.merge_generators = self.create_merge_generators()

    def get_unique_assays(self) -> List[str]:
        """
        Get unique assays from both datasets
        """
        unique_assays = set()
        for ds in self.datasets:
            unique_assays.update(ds.assay_names)
        return list(unique_assays)

    def create_merge_generators(self) -> List[AssayMerge]:
        """
        Create AssayMerge objects for each unique assay
        """
        gens = []
        for assay in self.unique_assays:
            assay_list = []
            for ds in self.datasets:
                if assay in ds.assay_names:
                    assay_list.append(ds.get_assay(assay))
                else:
                    # Generate a dummy assay on the fly
                    dummy_assay: DummyAssay = self.generate_dummy_assay(ds, assay)
                    assay_list.append(dummy_assay)
            gens.append(
                AssayMerge(
                    zarr_path=self.zarr_path,
                    assays=assay_list,
                    names=self.names,
                    merge_assay_name=assay,
                    in_workspaces=self.in_workspaces,
                    out_workspace=self.out_workspace,
                    chunk_size=self.chunk_size,
                    dtype=self.dtype,
                    overwrite=self.overwrite,
                    prepend_text=self.prepend_text,
                    reset_cell_filter=self.reset_cell_filter,
                    seed=self.seed,
                )
            )
        return gens

    def generate_dummy_assay(self, ds: DataStore, assay_name: str) -> DummyAssay:
        """
        Generate a dummy assay for a datastore that doesn't have the specified assay
        """
        # Find a datastore that has this assay to get feature information
        reference_ds = next(ds for ds in self.datasets if assay_name in ds.assay_names)
        reference_assay = reference_ds.get_assay(assay_name)
        # Create a dummy assay with zero counts and matching features
        dummy_shape = (ds.cells.N, reference_assay.feats.N)
        dummy_counts = np.zeros(dummy_shape, dtype=reference_assay.rawData.dtype)
        dummy_counts = from_array(
            dummy_counts, chunks=reference_assay.rawData.chunksize
        )
        dummy_assay = DummyAssay(ds, dummy_counts, reference_assay.feats)
        logger.info(f"Generated dummy {assay_name} assay for datastore {ds}")
        return dummy_assay

    def dump(self, nthreads=2) -> None:
        """
        Dump the merged data to the zarr file
        """
        for gen in self.merge_generators:
            print(f"Dumping {gen.merge_assay_name}")
            gen.dump(nthreads)
        print("Merging complete")
        return None