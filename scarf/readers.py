"""A collection of classes for reading in different data formats.

- Classes:
    - CrH5Reader: A class to read in CellRanger (Cr) data, in the form of an H5 file.
    - CrDirReader: A class to read in CellRanger (Cr) data, in the form of a directory.
    - CrReader: A class to read in CellRanger (Cr) data.
    - H5adReader: A class to read in data in the form of a H5ad file (h5 file with AnnData information).
    - NaboH5Reader: A class to read in data in the form of a Nabo H5 file.
    - LoomReader: A class to read in data in the form of a Loom file.
"""

from abc import ABC, abstractmethod
from typing import Generator, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import os
from scipy.sparse import coo_matrix
from typing import IO
import h5py
from .utils import logger, tqdmbar

__all__ = [
    "CrH5Reader",
    "CrDirReader",
    "CrReader",
    "H5adReader",
    "NaboH5Reader",
    "LoomReader",
    "CSVReader",
]


def get_file_handle(fn: str) -> IO:
    """Returns a file object for the given file name.

    Args:
        fn: The path to the file (file name).
    """
    import gzip

    try:
        if fn.rsplit(".", 1)[-1] == "gz":
            return gzip.open(fn, mode="rt")
        else:
            return open(fn, "r")
    except (OSError, IOError, FileNotFoundError):
        raise FileNotFoundError("ERROR: FILE NOT FOUND: %s" % fn)


def read_file(fn: str):
    """Yields the lines from the file the given file name points to.

    Args:
        fn: The path to the file (file name).
    """
    fh = get_file_handle(fn)
    for line in fh:
        yield line.rstrip()


class CrReader(ABC):
    """A class to read in CellRanger (Cr) data.

    Args:
        grp_names (Dict): A dictionary that specifies where to find the matrix, features and barcodes.

    Attributes:
        autoNames: Specifies if the data is from RNA or ATAC sequencing.
        grpNames: A dictionary that specifies where to find the matrix, features and barcodes.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        assayFeats: A DataFrame with information about the features in the assay.
    """

    def __init__(self, grp_names):
        self.autoNames = {
            "Gene Expression": "RNA",
            "Peaks": "ATAC",
            "Antibody Capture": "ADT",
        }
        self.grpNames: Dict = grp_names
        self.nFeatures: int = len(self.feature_names())
        self.nCells: int = len(self.cell_names())
        self.assayFeats = self._make_feat_table()
        self._auto_rename_assay_names()

    @abstractmethod
    def _handle_version(self):
        pass

    @abstractmethod
    def _read_dataset(self, key: Optional[str] = None) -> List:
        pass

    @abstractmethod
    def consume(self, batch_size: int, lines_in_mem: int):
        """Returns a generator that yield chunks of data."""
        pass

    def _subset_by_assay(self, v, assay) -> List:
        if assay is None:
            return v
        elif assay not in self.assayFeats:
            raise ValueError(f"ERROR: Assay ID {assay} is not valid")
        if len(self.assayFeats[assay].shape) == 2:
            ret_val = []
            for i in self.assayFeats[assay].values[1:3].T:
                ret_val.extend(list(v[i[0] : i[1]]))
            return ret_val
        elif len(self.assayFeats[assay].shape) == 1:
            idx = self.assayFeats[assay]
            return v[idx.start : idx.end]
        else:
            raise ValueError(
                "ERROR: assay feats is 3D. Something went really wrong. Create a github issue"
            )

    def _make_feat_table(self) -> pd.DataFrame:
        s = self.feature_types()
        span: List[Tuple] = []
        last = s[0]
        last_n: int = 0
        for n, i in enumerate(s[1:], 1):
            if i != last:
                span.append((last, last_n, n))
                last_n = n
            elif n == len(s) - 1:
                span.append((last, last_n, n + 1))
            last = i
        df = pd.DataFrame(span, columns=["type", "start", "end"])
        df.index = ["ASSAY%s" % str(x + 1) for x in df.index]
        df["nFeatures"] = df.end - df.start
        return df.T

    def _auto_rename_assay_names(self):
        new_names = []
        for k, v in self.assayFeats.T["type"].to_dict().items():
            if v in self.autoNames:
                new_names.append(self.autoNames[v])
            else:
                new_names.append(k)
        self.assayFeats.columns = new_names

    def rename_assays(self, name_map: Dict[str, str]) -> None:
        """Renames specified assays in the Reader.

        Args:
            name_map: A Dictionary containing current name as key and new name as value.
        """
        self.assayFeats.rename(columns=name_map, inplace=True)

    def feature_ids(self, assay: str = None) -> List[str]:
        """Returns a list of feature IDs in a specified assay.

        Args:
            assay: Select which assay to retrieve feature IDs from.
        """
        return self._subset_by_assay(self._read_dataset("feature_ids"), assay)

    def feature_names(self, assay: str = None) -> List[str]:
        """Returns a list of features in the dataset.

        Args:
            assay: Select which assay to retrieve features from.
        """
        vals = self._read_dataset("feature_names")
        if vals is None:
            logger.warning("Feature names extraction failed using feature IDs")
            vals = self._read_dataset("feature_ids")
        return self._subset_by_assay(vals, assay)

    def feature_types(self) -> List[str]:
        """Returns a list of feature types in the dataset."""
        if self.grpNames["feature_types"] is not None:
            ret_val = self._read_dataset("feature_types")
            if ret_val is not None:
                return ret_val
        default_name = list(self.autoNames.keys())[0]
        return [default_name for _ in range(self.nFeatures)]

    def cell_names(self) -> List[str]:
        """Returns a list of names of the cells in the dataset."""
        return self._read_dataset("cell_names")


class CrH5Reader(CrReader):
    # noinspection PyUnresolvedReferences
    """A class to read in CellRanger (Cr) data, in the form of an H5 file.

    Subclass of CrReader.

    Args:
        h5_fn: File name for the h5 file.

    Attributes:
        autoNames: Specifies if the data is from RNA or ATAC sequencing.
        grpNames: A dictionary that specifies where to find the matrix, features and barcodes.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        assayFeats: A DataFrame with information about the features in the assay.
        h5obj: A File object from the h5py package.
        grp: Current active group in the hierarchy.
    """

    def __init__(self, h5_fn, is_filtered: bool = True, filtering_cutoff: int = 500):
        self.h5obj = h5py.File(h5_fn, mode="r")
        self.grp = None
        self.validBarcodeIdx = None
        super().__init__(self._handle_version())
        if is_filtered:
            self.validBarcodeIdx = np.array(range(self.nCells))
        else:
            self.validBarcodeIdx = self._get_valid_barcodes(filtering_cutoff)
        self.nCells = len(self.validBarcodeIdx)

    def _handle_version(self):
        root_key = list(self.h5obj.keys())[0]
        self.grp = self.h5obj[root_key]
        if root_key == "matrix":
            grps = {
                "feature_ids": "features/id",
                "feature_names": "features/name",
                "feature_types": "features/feature_type",
                "cell_names": "barcodes",
            }
        else:
            grps = {
                "feature_ids": "genes",
                "feature_names": "gene_names",
                "feature_types": None,
                "cell_names": "barcodes",
            }
        return grps

    def _get_valid_barcodes(
        self, filtering_cutoff: int, batch_size: int = 1000
    ) -> np.ndarray:
        valid_idx = []
        test_counter = 0
        indptr = self.grp["indptr"][:]
        for s in tqdmbar(
            range(0, len(indptr) - 1, batch_size),
            desc=f"Filtering out background barcodes",
        ):
            idx = indptr[s : s + batch_size + 1]
            data = self.grp["data"][idx[0] : idx[-1]]
            indices = self.grp["indices"][idx[0] : idx[-1]]
            cell_idx = np.repeat(range(len(idx) - 1), np.diff(idx))
            mat = coo_matrix((data, (cell_idx, indices)))
            valid_idx.append(np.array(mat.sum(axis=1)).T[0] > filtering_cutoff)
            test_counter += data.shape[0]
        assert test_counter == self.grp["data"].shape[0]
        assert len(indptr) == (s + len(idx))
        return np.where(np.hstack(valid_idx))[0]

    def _read_dataset(self, key: Optional[str] = None):
        return [x.decode("UTF-8") for x in self.grp[self.grpNames[key]][:]]

    def cell_names(self) -> List[str]:
        """Returns a list of names of the cells in the dataset."""
        vals = np.array(self._read_dataset("cell_names"))
        if self.validBarcodeIdx is not None:
            vals = vals[self.validBarcodeIdx]
        return list(vals)

    # noinspection DuplicatedCode
    def consume(
        self, batch_size: int, lines_in_mem: int = None
    ) -> Generator[coo_matrix, None, None]:
        indptr = self.grp["indptr"][:]
        for s in range(0, len(self.validBarcodeIdx), batch_size):
            v_pos = self.validBarcodeIdx[s : s + batch_size]
            idx = [np.arange(x, y) for x, y in zip(indptr[v_pos], indptr[v_pos + 1])]
            cell_idx = np.repeat(np.arange(len(idx)), [len(x) for x in idx])
            idx = np.hstack(idx)
            data = self.grp["data"][idx[0] : idx[-1] + 1]
            data = data[idx - idx[0]]
            indices = self.grp["indices"][idx[0] : idx[-1] + 1]
            indices = indices[idx - idx[0]]
            yield coo_matrix((data, (cell_idx, indices)))

    def close(self) -> None:
        """Closes file connection."""
        self.h5obj.close()


class CrDirReader(CrReader):
    """A class to read in CellRanger (Cr) data, in the form of a directory.

    Subclass of CrReader.

    Args:
        loc (str): Path for the directory containing the cellranger output.
        mtx_separator (str): Column delimiter in the MTX file (Default value: ' ')
        index_offset (int): This value is added to each feature index (Default value: -1)

    Attributes:
        loc: Path for the directory containing the cellranger output.
        matFn: The file name for the matrix file.
        sep (str): Column delimiter in the MTX file (Default value: ' ')
        indexOffset (int): This value is added to each feature index (Default value: -1)
    """

    def __init__(
        self,
        loc,
        mtx_separator: str = " ",
        index_offset: int = -1,
    ):
        self.loc: str = loc.rstrip("/") + "/"
        self.matFn = None
        self.sep = mtx_separator
        self.indexOffset = index_offset
        super().__init__(self._handle_version())

    def _handle_version(self):
        show_error = False
        if os.path.isfile(self.loc + "matrix.mtx.gz"):
            self.matFn = self.loc + "matrix.mtx.gz"
        elif os.path.isfile(self.loc + "matrix.mtx"):
            self.matFn = self.loc + "matrix.mtx"
        else:
            show_error = True
        if os.path.isfile(self.loc + "features.tsv.gz"):
            feat_fn = "features.tsv.gz"
        elif os.path.isfile(self.loc + "features.tsv"):
            feat_fn = "features.tsv"
        elif os.path.isfile(self.loc + "genes.tsv.gz"):
            feat_fn = "genes.tsv.gz"
        elif os.path.isfile(self.loc + "genes.tsv"):
            feat_fn = "genes.tsv"
        elif os.path.isfile(self.loc + "peaks.bed"):
            feat_fn = "peaks.bed"
        elif os.path.isfile(self.loc + "peaks.bed.gz"):
            feat_fn = "peaks.bed.gz"
        else:
            feat_fn = None
            show_error = True
        if os.path.isfile(self.loc + "barcodes.tsv.gz"):
            cell_fn = "barcodes.tsv.gz"
        elif os.path.isfile(self.loc + "barcodes.tsv"):
            cell_fn = "barcodes.tsv"
        else:
            cell_fn = None
            show_error = True
        if show_error:
            raise IOError(
                "ERROR: Couldn't find either of these expected combinations of files:\n"
                "\t- matrix.mtx, barcodes.tsv and genes.tsv\n"
                "\t- matrix.mtx.gz, barcodes.tsv.gz and features.tsv.gz\n"
                "Please make sure that you have not compressed or uncompressed the Cellranger output files "
                "manually"
            )
        return {
            "feature_ids": (feat_fn, 0),
            "feature_names": (feat_fn, 1),
            "feature_types": (feat_fn, 2),
            "cell_names": (cell_fn, 0),
        }

    def _read_dataset(self, key: Optional[str] = None):
        try:
            vals = [
                x.split("\t")[self.grpNames[key][1]]
                for x in read_file(self.loc + self.grpNames[key][0])
            ]
        except IndexError:
            logger.warning(
                f"{key} extraction failed from {self.grpNames[key][0]} "
                f"in column {self.grpNames[key][1]}",
                flush=True,
            )
            vals = None
        return vals

    def to_sparse(self, a: np.ndarray) -> coo_matrix:
        """Returns the input data as a sparse (COO) matrix.

        Args:
            a: Sparse matrix, contains a chunk of data from the MTX file.
        """
        return coo_matrix(
            (
                a[:, 2],
                (
                    (a[:, 1] - a[0, 1]).astype(int),
                    (a[:, 0] + self.indexOffset).astype(int),
                ),
            )
        )

    # noinspection DuplicatedCode
    def consume(
        self, batch_size: int, lines_in_mem: int = int(1e5)
    ) -> Generator[coo_matrix, None, None]:
        stream = pd.read_csv(
            self.matFn, skiprows=3, sep=self.sep, header=None, chunksize=lines_in_mem
        )
        start = 1
        dfs = []
        for df in stream:
            if df.iloc[-1, 1] - start >= batch_size:
                idx = df[1] < batch_size + start
                dfs.append(df[idx])
                yield self.to_sparse(np.vstack(dfs))
                dfs = [df[~idx]]
                start += batch_size
            else:
                dfs.append(df)
        yield self.to_sparse(np.vstack(dfs))


class H5adReader:
    """A class to read in data from a H5ad file (h5 file with AnnData
    information).

    Args:
        h5ad_fn: Path to H5AD file
        cell_attrs_key: H5 group under which cell attributes are saved.(Default value: 'obs')
        feature_attrs_key: H5 group under which feature attributes are saved.(Default value: 'var')
        cell_ids_key: Key in `obs` group that contains unique cell IDs. By default the index will be used.
        feature_ids_key: Key in `var` group that contains unique feature IDs. By default the index will be used.
        feature_name_key: Key in `var` group that contains feature names. (Default: gene_short_name)
        matrix_key: Group where in the sparse matrix resides (default: 'X')
        category_names_key: Looks up this group and replaces the values in `var` and 'obs' child datasets with the
                            corresponding index value within this group.
        dtype: Numpy dtype of the matrix data. This dtype is enforced when streaming the data through `consume`
               method. (Default value: Automatically determined)

    Attributes:
        h5: A File object from the h5py package.
        matrixKey: Group where in the sparse matrix resides (default: 'X')
        cellAttrsKey: Group wherein the cell attributes are present
        featureAttrsKey: Group wherein the feature attributes are present
        groupCodes: Used to ensure compatibility with different AnnData versions.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        cellIdsKey: Key in `obs` group that contains unique cell IDs. By default the index will be used.
        featIdsKey: Key in `var` group that contains unique feature IDs. By default the index will be used.
        featNamesKey: Key in `var` group that contains feature names. (Default: gene_short_name)
        catNamesKey: Looks up this group and replaces the values in `var` and 'obs' child datasets with the
                     corresponding index value within this group.
        matrixDtype: dtype of the matrix containing the data (as indicated by matrix_key)
    """

    def __init__(
        self,
        h5ad_fn: str,
        cell_attrs_key: str = "obs",
        cell_ids_key: str = "_index",
        feature_attrs_key: str = "var",
        feature_ids_key: str = "_index",
        feature_name_key: str = "gene_short_name",
        matrix_key: str = "X",
        obsm_attrs_key: str = "obsm",
        category_names_key: str = "__categories",
        dtype: str = None,
    ):
        self.h5 = h5py.File(h5ad_fn, mode="r")
        self.matrixKey = matrix_key
        self.cellAttrsKey, self.featureAttrsKey, self.obsmAttrsKey = (
            cell_attrs_key,
            feature_attrs_key,
            obsm_attrs_key,
        )
        self.groupCodes = {
            self.cellAttrsKey: self._validate_group(self.cellAttrsKey),
            self.featureAttrsKey: self._validate_group(self.featureAttrsKey),
            self.obsmAttrsKey: self._validate_group(self.obsmAttrsKey),
            self.matrixKey: self._validate_group(self.matrixKey),
        }
        self.nCells, self.nFeatures = self._get_n(self.cellAttrsKey), self._get_n(
            self.featureAttrsKey
        )
        self.cellIdsKey = self._fix_name_key(self.cellAttrsKey, cell_ids_key)
        self.featIdsKey = self._fix_name_key(self.featureAttrsKey, feature_ids_key)
        self.featNamesKey = feature_name_key
        self.catNamesKey = category_names_key
        self.matrixDtype = self._get_matrix_dtype() if dtype is None else dtype

    def _validate_group(self, group: str) -> int:
        if group not in self.h5:
            logger.warning(f"`{group}` group not found in the H5ad file")
            ret_val = 0
        elif type(self.h5[group]) == h5py.Dataset:
            ret_val = 1
        elif type(self.h5[group]) == h5py.Group:
            ret_val = 2
        else:
            logger.warning(
                f"`{group}` slot in H5ad file is not of Dataset or Group type. "
                f"Due to this, no information in `{group}` can be used"
            )
            ret_val = 0
        if ret_val == 2:
            if len(self.h5[group].keys()) == 0:
                logger.warning(f"`{group}` slot in H5ad file is empty.")
                ret_val = 0
            elif (
                len(
                    set(
                        [
                            self.h5[group][x].shape[0]
                            for x in self.h5[group].keys()
                            if type(self.h5[group][x]) == h5py.Dataset
                        ]
                    )
                )
                > 1
            ):
                if sorted(self.h5[group].keys()) != ["data", "indices", "indptr"]:
                    logger.info(
                        f"`{group}` slot in H5ad file has unequal sized child groups"
                    )
        return ret_val

    def _get_matrix_dtype(self):
        if self.groupCodes[self.matrixKey] == 1:
            return self.h5[self.matrixKey].dtype
        elif self.groupCodes[self.matrixKey] == 2:
            return self.h5[self.matrixKey]["data"].dtype
        else:
            raise ValueError(
                f"ERROR: {self.matrixKey} is neither Dataset or Group type. Will not consume data"
            )

    def _check_exists(self, group: str, key: str) -> bool:
        if group in self.groupCodes:
            group_code = self.groupCodes[group]
        else:
            group_code = self._validate_group(group)
            self.groupCodes[group] = group_code
        if group_code == 1:
            if key in list(self.h5[group].dtype.names):
                return True
        if group_code == 2:
            if key in self.h5[group].keys():
                return True
        return False

    def _fix_name_key(self, group: str, key: str) -> str:
        if self._check_exists(group, key) is False:
            if key.startswith("_"):
                temp_key = key[1:]
                if self._check_exists(group, temp_key):
                    return temp_key
        return key

    def _get_n(self, group: str) -> int:
        if self.groupCodes[group] == 0:
            if self._check_exists(self.matrixKey, "shape"):
                return self.h5[self.matrixKey]["shape"][0]
            else:
                raise KeyError(
                    f"ERROR: `{group}` not found and `shape` key is missing in the {self.matrixKey} group. "
                    f"Aborting read process."
                )
        elif self.groupCodes[group] == 1:
            return self.h5[group].shape[0]
        else:
            for i in self.h5[group].keys():
                if type(self.h5[group][i]) == h5py.Dataset:
                    return self.h5[group][i].shape[0]
            raise KeyError(
                f"ERROR: `{group}` key doesn't contain any child node of Dataset type."
                f"Aborting because unexpected H5ad format."
            )

    def cell_ids(self) -> np.ndarray:
        """Returns a list of cell IDs."""
        if self._check_exists(self.cellAttrsKey, self.cellIdsKey):
            if self.groupCodes[self.cellAttrsKey] == 1:
                return self.h5[self.cellAttrsKey][self.cellIdsKey]
            else:
                return self.h5[self.cellAttrsKey][self.cellIdsKey][:]
        logger.warning(f"Could not find cells ids key: {self.cellIdsKey} in `obs`.")
        return np.array([f"cell_{x}" for x in range(self.nCells)])

    # noinspection DuplicatedCode
    def feat_ids(self) -> np.ndarray:
        """Returns a list of feature IDs."""
        if self._check_exists(self.featureAttrsKey, self.featIdsKey):
            if self.groupCodes[self.featureAttrsKey] == 1:
                return self.h5[self.featureAttrsKey][self.featIdsKey]
            else:
                return self.h5[self.featureAttrsKey][self.featIdsKey][:]
        logger.warning(
            f"Could not find feature ids key: {self.featIdsKey} in {self.featureAttrsKey}."
        )
        return np.array([f"feature_{x}" for x in range(self.nFeatures)])

    # noinspection DuplicatedCode
    def feat_names(self) -> np.ndarray:
        """Returns a list of feature names."""
        if self._check_exists(self.featureAttrsKey, self.featNamesKey):
            if self.groupCodes[self.featureAttrsKey] == 1:
                values = self.h5[self.featureAttrsKey][self.featNamesKey]
            else:
                values = self.h5[self.featureAttrsKey][self.featNamesKey][:]
            return self._replace_category_values(
                values, self.featNamesKey, self.featureAttrsKey
            ).astype(object)
        logger.warning(
            f"Could not find feature names key: {self.featNamesKey} in self.featureAttrsKey."
        )
        return self.feat_ids()

    def _replace_category_values(self, v: np.ndarray, key: str, group: str):
        if self.catNamesKey is not None:
            if self._check_exists(group, self.catNamesKey):
                cat_g = self.h5[group][self.catNamesKey]
                if type(cat_g) == h5py.Group:
                    if key in cat_g:
                        c = cat_g[key][:]
                        try:
                            return np.array([c[x] for x in v])
                        except (IndexError, TypeError):
                            return v
        if "uns" in self.h5:
            if key + "_categories" in self.h5["uns"]:
                c = self.h5["uns"][key + "_categories"][:]
                try:
                    return np.array([c[x] for x in v])
                except (IndexError, TypeError):
                    return v
        return v

    def _get_col_data(
        self, group: str, ignore_keys: List[str]
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        if self.groupCodes[group] == 1:
            for i in tqdmbar(
                self.h5[group].dtype.names,
                desc=f"Reading attributes from group {group}",
            ):
                if i in ignore_keys:
                    continue
                yield i, self._replace_category_values(self.h5[group][i][:], i, group)
        if self.groupCodes[group] == 2:
            for i in tqdmbar(
                self.h5[group].keys(), desc=f"Reading attributes from group {group}"
            ):
                if i in ignore_keys:
                    continue
                if type(self.h5[group][i]) == h5py.Dataset:
                    yield i, self._replace_category_values(
                        self.h5[group][i][:], i, group
                    )

    def _get_obsm_data(
        self, group: str
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        if self.groupCodes[group] == 2:
            for i in tqdmbar(
                self.h5[group].keys(), desc=f"Reading attributes from group {group}"
            ):
                g = self.h5[group][i]
                if g.shape[0] != self.nCells:
                    logger.error(
                        f"Dimension of {i}({g.shape}) is not correct."
                        f" Will not save this specific slot into Zarr."
                    )
                    continue
                if type(g) == h5py.Dataset:
                    for j in range(g.shape[1]):
                        yield f"{i}{j+1}", g[:, j]
        else:
            logger.warning(
                f"Reading of obsm failed because it either does not exist or is not in expected format"
            )

    def get_cell_columns(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Creates a Generator that yields the cell columns."""
        for i, j in self._get_col_data(self.cellAttrsKey, [self.cellIdsKey]):
            yield i, j
        for i, j in self._get_obsm_data(self.obsmAttrsKey):
            yield i, j

    def get_feat_columns(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Creates a Generator that yields the feature columns."""
        for i, j in self._get_col_data(
            self.featureAttrsKey, [self.featIdsKey, self.featNamesKey]
        ):
            yield i, j

    # noinspection DuplicatedCode
    def consume_dataset(
        self, batch_size: int = 1000
    ) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        dset = self.h5[self.matrixKey]
        s = 0
        for e in range(batch_size, dset.shape[0] + batch_size, batch_size):
            if e > dset.shape[0]:
                e = dset.shape[0]
            yield coo_matrix(dset[s:e])
            s = e

    def consume_group(self, batch_size: int) -> Generator[coo_matrix, None, None]:
        """Returns a generator that yield chunks of data."""
        grp = self.h5[self.matrixKey]
        s = 0
        for ind_n in range(0, self.nCells, batch_size):
            i = grp["indptr"][ind_n : ind_n + batch_size]
            e = i[-1]
            if s != 0:
                idx = np.array([s] + list(i))
                idx = idx - idx[0]
            else:
                idx = np.array(i)
            n = idx.shape[0] - 1
            nidx = np.repeat(range(n), np.diff(idx).astype("int32"))
            yield coo_matrix((grp["data"][s:e], (nidx, grp["indices"][s:e])))
            s = e

    def consume(self, batch_size: int):
        """Returns a generator that yield chunks of data."""
        if self.groupCodes[self.matrixKey] == 1:
            return self.consume_dataset(batch_size)
        elif self.groupCodes[self.matrixKey] == 2:
            return self.consume_group(batch_size)


class NaboH5Reader:
    """A class to read in data in the form of a Nabo H5 file.

    Args:
        h5_fn: Path to H5 file.

    Attributes:
        h5: A File object from the h5py package.
        nCells: Number of cells in dataset.
        nFeatures: Number of features in dataset.
    """

    def __init__(self, h5_fn: str):
        self.h5 = h5py.File(h5_fn, mode="r")
        self._check_integrity()
        self.nCells = self.h5["names"]["cells"].shape[0]
        self.nFeatures = self.h5["names"]["genes"].shape[0]

    def _check_integrity(self) -> bool:
        for i in ["cell_data", "gene_data", "names"]:
            if i not in self.h5:
                raise KeyError(f"ERROR: Expected group: {i} is missing in the H5 file")
        return True

    def cell_ids(self) -> List[str]:
        """Returns a list of cell IDs."""
        return [x.decode("UTF-8") for x in self.h5["names"]["cells"][:]]

    def feat_ids(self) -> np.ndarray:
        """Returns a list of feature IDs."""
        return np.array([f"feature_{x}" for x in range(self.nFeatures)])

    def feat_names(self) -> List[str]:
        """Returns a list of feature names."""
        return [
            x.decode("UTF-8").rsplit("_", 1)[0] for x in self.h5["names"]["genes"][:]
        ]

    def consume(self, batch_size: int = 100) -> Generator[np.ndarray, None, None]:
        """Returns a generator that yield chunks of data."""
        batch = []
        for i in self.h5["cell_data"]:
            a = np.zeros(self.nFeatures).astype(int)
            v = self.h5["cell_data"][i][:][::-1]
            a[v["idx"]] = v["val"]
            batch.append(a)
            if len(batch) >= batch_size:
                batch = np.array(batch)
                yield batch
                batch = []
        if len(batch) > 0:
            yield np.array(batch)


class LoomReader:
    """A class to read in data in the form of a Loom file.

    Args:
        loom_fn: Path to loom format file.
        matrix_key: Child node under HDF5 file root wherein the chunked matrix is stored. (Default value: matrix).
                    This matrix is expected to be of form (nFeatures x nCells)
        cell_attrs_key: Child node under the HDF5 file wherein the cell attributes are stored.
                        (Default value: col_attrs)
        cell_names_key: Child node under the `cell_attrs_key` wherein the cell names are stored.
                        (Default value: obs_names)
        feature_attrs_key: Child node under the HDF5 file wherein the feature/gene attributes are stored.
                           (Default value: row_attrs)
        feature_names_key: Child node under the `feature_attrs_key` wherein the feature/gene names are stored.
                           (Default value: var_names)
        feature_ids_key: Child node under the `feature_attrs_key` wherein the feature/gene ids are stored.
                         (Default value: None)
        dtype: Numpy dtype of the matrix data. This dtype is enforced when streaming the data through `consume`
               method. (Default value: Automatically determined)

    Attributes:
        h5: A File object from the h5py package.
        matrixKey: Child node under HDF5 file root wherein the chunked matrix is stored.
        cellAttrsKey: Child node under the HDF5 file wherein the cell attributes are stored.
        featureAttrsKey: Child node under the HDF5 file wherein the feature/gene attributes are stored.
        cellNamesKey: Child node under the `cell_attrs_key` wherein the cell names are stored.
        featureNamesKey: Child node under the `feature_attrs_key` wherein the feature/gene names are stored.
        featureIdsKey: Child node under the `feature_attrs_key` wherein the feature/gene ids are stored.
        matrixDtype: Numpy dtype of the matrix data.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
    """

    def __init__(
        self,
        loom_fn: str,
        matrix_key: str = "matrix",
        cell_attrs_key="col_attrs",
        cell_names_key: str = "obs_names",
        feature_attrs_key: str = "row_attrs",
        feature_names_key: str = "var_names",
        feature_ids_key: str = None,
        dtype: str = None,
    ) -> None:
        self.h5 = h5py.File(loom_fn, mode="r")
        self.matrixKey = matrix_key
        self.cellAttrsKey, self.featureAttrsKey = cell_attrs_key, feature_attrs_key
        self.cellNamesKey, self.featureNamesKey = cell_names_key, feature_names_key
        self.featureIdsKey = feature_ids_key
        self.matrixDtype = self.h5[self.matrixKey].dtype if dtype is None else dtype
        self._check_integrity()
        self.nFeatures, self.nCells = self.h5[self.matrixKey].shape

    def _check_integrity(self) -> bool:
        if self.matrixKey not in self.h5:
            raise KeyError(
                f"ERROR: Matrix key (location): {self.matrixKey} is missing in the H5 file"
            )
        if self.cellAttrsKey not in self.h5:
            logger.warning(
                f"Cell attributes are missing. Key {self.cellAttrsKey} was not found"
            )
        if self.featureAttrsKey not in self.h5:
            logger.warning(
                f"Feature attributes are missing. Key {self.featureAttrsKey} was not found"
            )
        return True

    def cell_names(self) -> List[str]:
        """Returns a list of names of the cells in the dataset."""
        if self.cellAttrsKey not in self.h5:
            pass
        elif self.cellNamesKey not in self.h5[self.cellAttrsKey]:
            logger.warning(
                f"Cell names/ids key ({self.cellNamesKey}) is missing in attributes"
            )
        else:
            return self.h5[self.cellAttrsKey][self.cellNamesKey][:]
        return [f"cell_{x}" for x in range(self.nCells)]

    def cell_ids(self) -> List[str]:
        """Returns a list of cell IDs."""
        return self.cell_names()

    def _stream_attrs(
        self, key, ignore
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        if key in self.h5:
            for i in tqdmbar(self.h5[key].keys(), desc=f"Reading {key} attributes"):
                if i in [ignore]:
                    continue
                vals = self.h5[key][i][:]
                if vals.dtype.names is None:
                    yield i, vals
                else:
                    # Attribute is a structured array
                    for j in vals.dtype.names:
                        yield i + "_" + str(j), vals[j]

    def get_cell_attrs(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Returns a Generator that yields the cells' attributes."""
        return self._stream_attrs(self.cellAttrsKey, [self.cellNamesKey])

    def feature_names(self) -> List[str]:
        """Returns a list of feature names."""
        if self.featureAttrsKey not in self.h5:
            pass
        elif self.featureNamesKey not in self.h5[self.featureAttrsKey]:
            logger.warning(
                f"Feature names key ({self.featureNamesKey}) is missing in attributes"
            )
        else:
            return self.h5[self.featureAttrsKey][self.featureNamesKey][:]
        return [f"feature_{x}" for x in range(self.nFeatures)]

    def feature_ids(self) -> List[str]:
        """Returns a list of feature IDs."""
        if self.featureAttrsKey not in self.h5:
            pass
        elif self.featureIdsKey is None:
            pass
        elif self.featureIdsKey not in self.h5[self.featureAttrsKey]:
            logger.warning(
                f"Feature names key ({self.featureIdsKey}) is missing in attributes"
            )
        else:
            return self.h5[self.featureAttrsKey][self.featureIdsKey][:]
        return [f"feature_{x}" for x in range(self.nFeatures)]

    def get_feature_attrs(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """Returns a Generator that yields the features' attributes."""
        return self._stream_attrs(
            self.featureAttrsKey, [self.featureIdsKey, self.featureNamesKey]
        )

    def consume(self, batch_size: int = 1000) -> Generator[np.ndarray, None, None]:
        """Returns a generator that yield chunks of data."""
        dset = self.h5[self.matrixKey]
        s = 0
        for e in range(batch_size, dset.shape[1] + batch_size, batch_size):
            if e > dset.shape[1]:
                e = dset.shape[1]
            yield coo_matrix(dset[:, s:e]).T.astype(self.matrixDtype)
            s = e


class CSVReader:
    """A class to read in data from a CSV file.

    Args:
        csv_fn: Path to the CSV file
        has_header: Does the CSV file has a header. (Default value: True)
        id_column: The column number which contains row name. (Default value: None)
        rows_are_cells: If True then each row represents a cell and hence each column is a feature. If False then each
                        row is feature and each column in a cell
        sep: The column separator in the CSV file (Default value: ',')
        skip_rows: Number of rows to skip from the top of the file. (Default value: 0)
        skip_cols: Names of columns to skip. Must be provided as a list even if just one column.
                        (Default value: None)
        cell_data_cols: Names of columns to include in cell metadata rather than count matrix. Must be provided as a
                       list even if just one column. (Default value: None)
        batch_size: Number of lines to read at a time. Decrease this value if you have too many columns.
                    (Default value: 50,000)
        pandas_kwargs: A dictionary of keyword arguments to be passed to Pandas read_csv function.

    Attributes:
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
    """

    def __init__(
        self,
        csv_fn: str,
        has_header: bool = True,
        id_column: Optional[int] = None,
        rows_are_cells: bool = True,
        sep: str = ",",
        skip_rows: int = 0,
        skip_cols: Optional[List[str]] = None,
        cell_data_cols: Optional[List[str]] = None,
        batch_size=10000,
        pandas_kwargs: Optional[dict] = None,
    ):
        self._fn = csv_fn
        if rows_are_cells is False:
            raise NotImplementedError(
                "Currently Scarf supports only those CSV files where cells are along the rows"
            )
        if pandas_kwargs is None:
            pandas_kwargs = {}
        else:
            if type(pandas_kwargs) != dict:
                logger.error("")
        if has_header is False:
            has_header = None
        else:
            has_header = 0
        self.pandas_kwargs = pandas_kwargs
        self.pandas_kwargs["sep"] = sep
        self.pandas_kwargs["header"] = has_header
        self.pandas_kwargs["skiprows"] = skip_rows
        self.pandas_kwargs["chunksize"] = batch_size
        self.pandas_kwargs["index_col"] = id_column

        if skip_cols is None:
            self.skipCols = []
        else:
            self.skipCols = skip_cols
        if cell_data_cols is None:
            self.cellDataCols = []
        else:
            self.cellDataCols = cell_data_cols
        (
            self.nCells,
            self.nFeatures,
            self.cellIds,
            self.featureIds,
            self.keepCols,
            self.cellDataDtypes,
            self.cellDataIdx,
        ) = self._consistency_check()

    def _get_streamer(self) -> Generator:
        return pd.read_csv(self._fn, **self.pandas_kwargs)

    def _consistency_check(
        self,
    ) -> Tuple[
        int,
        int,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[List[int]],
        Optional[List[np.dtype]],
        Optional[List[int]],
    ]:
        stream = self._get_streamer()
        n_cells = 0
        n_features = 0
        feature_ids = None
        cell_data_dtypes = None
        cell_data_idx = None
        if self.pandas_kwargs["index_col"] is None:
            cell_ids = None
        else:
            cell_ids = []
        for df in tqdmbar(stream, desc="Performing CSV file consistency check"):
            n_cells += df.shape[0]
            if n_features == 0:
                n_features = df.shape[1]
                if self.pandas_kwargs["header"] is not None:
                    feature_ids = df.columns.values
                    if len(feature_ids) != n_features:
                        raise ValueError(
                            "Header length not same as number of features. This can happen if you did not"
                            " skip the right number of rows."
                        )
                    if len(self.cellDataCols) > 0:
                        cell_data_dtypes = list(df[self.cellDataCols].dtypes.values)
                        cell_data_idx = [
                            n
                            for n, x in enumerate(feature_ids)
                            if x in self.cellDataCols
                        ]
            else:
                if n_features != df.shape[1]:
                    raise ValueError(
                        "Number of columns changed in the CSV during consistency check."
                        " Maybe a problem with the delimiter."
                    )
        if cell_ids is not None:
            cell_ids = np.ndarray(cell_ids)
        keep_cols = None
        if feature_ids is not None:
            skip_names = list(set(self.skipCols).union(self.cellDataCols))
            if len(skip_names) > 0:
                keep_cols = [
                    n for n, x in enumerate(feature_ids) if x not in skip_names
                ]
                feature_ids = feature_ids[keep_cols]
                n_features = len(keep_cols)
        return (
            n_cells,
            n_features,
            cell_ids,
            feature_ids,
            keep_cols,
            cell_data_dtypes,
            cell_data_idx,
        )

    def _n_features(self):
        return np.array([f"feature_{x}" for x in range(self.nFeatures)])

    def cell_ids(self) -> np.ndarray:
        """Returns a list of cell IDs."""
        if self.cellIds is None:
            return np.array([f"cell_{x}" for x in range(self.nCells)])
        else:
            return self.cellIds

    def feature_ids(self) -> np.ndarray:
        """Returns a list of feature IDs."""
        if self.featureIds is None:
            return np.array([f"feature_{x}" for x in range(self.nFeatures)])
        else:
            return self.featureIds

    def consume(self) -> Generator[Tuple[np.ndarray, Optional[np.ndarray]], None, None]:
        """Returns a generator that yield chunks of data."""
        stream = self._get_streamer()
        if self.keepCols is None:
            for df in stream:
                yield df.values, None
        else:
            if self.cellDataIdx is not None:
                for df in stream:
                    yield df.values[:, self.keepCols], df.values[:, self.cellDataIdx]
            else:
                for df in stream:
                    yield df.values[:, self.keepCols], None
