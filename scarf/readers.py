"""
A collection of classes for reading in different data formats.

- Classes:
    - CrH5Reader: A class to read in CellRanger (Cr) data, in the form of an H5 file.
    - CrDirReader: A class to read in CellRanger (Cr) data, in the form of a directory.
    - CrReader: A class to read in CellRanger (Cr) data.
    - H5adReader: A class to read in data in the form of a H5ad file (h5 file with AnnData information).
    - MtxDirReader: A class to read in data in the form of a directory containing a Matrix Market file + accompanying files.
    - NaboH5Reader: A class to read in data in the form of a Nabo H5 file.
    - LoomReader: A class to read in data in the form of a Loom file.
"""

from abc import ABC, abstractmethod
from typing import Generator, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import os
import sparse
from typing import IO
import h5py
from tqdm import tqdm
from .logging_utils import logger

__all__ = ['CrH5Reader', 'CrDirReader', 'CrReader', 'H5adReader', 'MtxDirReader', 'NaboH5Reader', 'LoomReader']


def get_file_handle(fn: str) -> IO:
    """
    Returns a file object for the given file name.

    Args:
        fn: The path to the file (file name).
    """
    import gzip

    try:
        if fn.rsplit('.', 1)[-1] == 'gz':
            return gzip.open(fn, mode='rt')
        else:
            return open(fn, 'r')
    except (OSError, IOError, FileNotFoundError):
        raise FileNotFoundError("ERROR: FILE NOT FOUND: %s" % fn)


def read_file(fn: str):
    """
    Yields the lines from the file the given file name points to.

    Args:
        fn: The path to the file (file name).
    """
    fh = get_file_handle(fn)
    for line in fh:
        yield line.rstrip()


class CrReader(ABC):
    """
    A class to read in CellRanger (Cr) data.

    Attributes:
        autoNames: Specifies if the data is from RNA or ATAC sequencing.
        grpNames: A dictionary that specifies where to find the matrix, features and barcodes.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        assayFeats: A DataFrame with information about the features in the assay.
    """
    def __init__(self, grp_names, file_type):
        """
        Args:
            grp_names (Dict): A dictionary that specifies where to find the matrix, features and barcodes.
            file_type (str): Type of sequencing data ('rna' | 'atac')
        """
        if file_type == 'rna':
            self.autoNames = {'Gene Expression': 'RNA'}
        elif file_type == 'atac':
            self.autoNames = {'Peaks': 'ATAC'}
        else:
            raise ValueError("ERROR: Please provide a value for parameter 'file_type'.\n"
                             "The value can be either 'rna' or 'atac' depending on whether is is scRNA-Seq or "
                             "scATAC-Seq data")
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
        """
        Returns a generator that yield chunks of data.
        """
        pass

    def _subset_by_assay(self, v, assay) -> List:
        if assay is None:
            return v
        elif assay not in self.assayFeats:
            raise ValueError("ERROR: Assay ID %s is not valid" % assay)
        idx = self.assayFeats[assay]
        return v[idx.start: idx.end]

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
        df = pd.DataFrame(span, columns=['type', 'start', 'end'])
        df.index = ["assay%s" % str(x + 1) for x in df.index]
        df['nFeatures'] = df.end - df.start
        return df.T

    def _auto_rename_assay_names(self):
        anames = list(map(str.upper, self.assayFeats.columns))
        main_name_k = list(self.autoNames.keys())[0]
        main_name_v = list(self.autoNames.values())[0]
        if main_name_v in anames:
            logger.info(f"{main_name_v} already present")
            # Making sure that column name is in uppercase 'RNA'
            newnames = list(self.assayFeats.columns)
            newnames[anames.index(main_name_v)] = main_name_v
            self.assayFeats.columns = newnames
        else:
            at = self.assayFeats.T['type'] == main_name_k
            if at.sum() == 1:
                main_assay = at[at].index[0]
            else:
                # FIXME: raise warning here
                main_assay = self.assayFeats.T[at].nFeatures.astype(int).idxmax()
            self.rename_assays({main_assay: main_name_v})

    def rename_assays(self, name_map: Dict[str, str]) -> None:
        """
        Renames specified assays in the Reader.

        Args:
            name_map: A Dictionary containing current name as key and new name as value.
        """
        self.assayFeats.rename(columns=name_map, inplace=True)

    def feature_ids(self, assay: str = None) -> List[str]:
        """
        Returns a list of feature IDs in a specified assay.

        Args:
            assay: Select which assay to retrieve feature IDs from.
        """
        return self._subset_by_assay(
            self._read_dataset('feature_ids'), assay)

    def feature_names(self, assay: str = None) -> List[str]:
        """
        Returns a list of features in the dataset.

        Args:
            assay: Select which assay to retrieve features from.
        """
        vals = self._read_dataset('feature_names')
        if vals is None:
            logger.warning('Feature names extraction failed using feature IDs')
            vals = self._read_dataset('feature_ids')
        return self._subset_by_assay(vals, assay)

    def feature_types(self) -> List[str]:
        """
        Returns a list of feature types in the dataset.
        """
        if self.grpNames['feature_types'] is not None:
            ret_val = self._read_dataset('feature_types')
            if ret_val is not None:
                return ret_val
        default_name = list(self.autoNames.keys())[0]
        return [default_name for _ in range(self.nFeatures)]

    def cell_names(self) -> List[str]:
        """
        Returns a list of names of the cells in the dataset.
        """
        return self._read_dataset('cell_names')


class CrH5Reader(CrReader):
    """
    A class to read in CellRanger (Cr) data, in the form of an H5 file.

    Subclass of CrReader.

    Attributes:
        autoNames: Specifies if the data is from RNA or ATAC sequencing.
        grpNames: A dictionary that specifies where to find the matrix, features and barcodes.
        nFeatures: Number of features in dataset.
        nCells: Number of cells in dataset.
        assayFeats: A DataFrame with information about the features in the assay.
        h5obj: A File object from the h5py package.
        grp: Current active group in the hierarchy.
    """
    def __init__(self, h5_fn, file_type: str = None):
        """
        Args:
            h5_fn: File name for the h5 file.
            file_type (str): Type of sequencing data ('rna' | 'atac')
        """
        self.h5obj = h5py.File(h5_fn, mode='r')
        self.grp = None
        super().__init__(self._handle_version(), file_type)

    def _handle_version(self):
        root_key = list(self.h5obj.keys())[0]
        self.grp = self.h5obj[root_key]
        if root_key == 'matrix':
            grps = {'feature_ids': 'features/id',
                    'feature_names': 'features/name',
                    'feature_types': 'features/feature_type',
                    'cell_names': 'barcodes'}
        else:
            grps = {'feature_ids': 'genes', 'feature_names': 'gene_names',
                    'feature_types': None, 'cell_names': 'barcodes'}
        return grps

    def _read_dataset(self, key: Optional[str] = None):
        return [x.decode('UTF-8') for x in self.grp[self.grpNames[key]][:]]

    # noinspection DuplicatedCode
    def consume(self, batch_size: int, lines_in_mem: int):
        s = 0
        for ind_n in range(0, self.nCells, batch_size):
            i = self.grp['indptr'][ind_n:ind_n + batch_size]
            e = i[-1]
            if s != 0:
                idx = np.array([s] + list(i))
                idx = idx - idx[0]
            else:
                idx = np.array(i)
            n = idx.shape[0] - 1
            nidx = np.repeat(range(n), np.diff(idx).astype('int32'))
            yield sparse.COO([nidx, self.grp['indices'][s: e]], self.grp['data'][s: e],
                             shape=(n, self.nFeatures))
            s = e

    def close(self) -> None:
        """
        Closes file connection.
        """
        self.h5obj.close()


class CrDirReader(CrReader):
    """
    A class to read in CellRanger (Cr) data, in the form of a directory.

    Subclass of CrReader.

    Attributes:
        loc: Path for the directory containing the cellranger output.
        matFn: The file name for the matrix file.
    """
    def __init__(self, loc, file_type: str = None):
        """
        Args:
            loc (str): Path for the directory containing the cellranger output.
            file_type (str): Type of sequencing data ('rna' | 'atac')
        """
        self.loc: str = loc.rstrip('/') + '/'
        self.matFn = None
        super().__init__(self._handle_version(), file_type)

    def _handle_version(self):
        if os.path.isfile(self.loc + 'features.tsv.gz'):
            self.matFn = self.loc + 'matrix.mtx.gz'
            grps = {'feature_ids': ('features.tsv.gz', 0),
                    'feature_names': ('features.tsv.gz', 1),
                    'feature_types': ('features.tsv.gz', 2),
                    'cell_names': ('barcodes.tsv.gz', 0)}
        elif os.path.isfile(self.loc + 'genes.tsv'):
            self.matFn = self.loc + 'matrix.mtx'
            grps = {'feature_ids': ('genes.tsv', 0),
                    'feature_names': ('genes.tsv', 1),
                    'feature_types': None,
                    'cell_names': ('barcodes.tsv', 0)}
        else:
            raise IOError("ERROR: Couldn't find either of these expected combinations of files:\n"
                          "\t- matrix.mtx, barcodes.tsv and genes.tsv\n"
                          "\t- matrix.mtx.gz, barcodes.tsv.gz and features.tsv.gz\n"
                          "Please make sure that you have not compressed or uncompressed the Cellranger output files "
                          "manually")
        return grps

    def _read_dataset(self, key: Optional[str] = None):
        try:
            vals = [x.split('\t')[self.grpNames[key][1]] for x in
                    read_file(self.loc + self.grpNames[key][0])]
        except IndexError:
            logger.warning(f"{key} extraction failed from {self.grpNames[key][0]} "
                           f"in column {self.grpNames[key][1]}", flush=True)
            vals = None
        return vals

    # noinspection DuplicatedCode
    def to_sparse(self, a: np.ndarray) -> sparse.COO:
        """
        Returns the input data as a sparse (COO) matrix.

        Args:
            a: Sparse matrix, contains a chunk of data from the MTX file.
        """
        idx = np.where(np.diff(a[:, 1]) == 1)[0] + 1
        return sparse.COO([(a[:, 1] - a[0, 1]).astype(int),
                           (a[:, 0] - 1).astype(int)],
                          a[:, 2], shape=(len(idx) + 1, self.nFeatures))

    # noinspection DuplicatedCode
    def consume(self, batch_size: int, lines_in_mem: int = int(1e5)) -> \
            Generator[List[np.ndarray], None, None]:
        stream = pd.read_csv(self.matFn, skiprows=3, sep=' ',
                             header=None, chunksize=lines_in_mem)
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


class MtxDirReader(CrReader):
    """
    A class to read a directory with a Matrix Market file and its accompanying files.

    Subclass of CrReader.

    Attributes:
        loc: Path for the directory containing the cellranger output.
        matFn: The file name for the matrix file.
    """
    def __init__(self, loc, file_type: str = None):
        """
        Args:
            loc (str): Path for the directory containing the cellranger output.
            file_type (str): Type of sequencing data ('rna' | 'atac')
        """
        self.loc: str = loc.rstrip('/') + '/'
        self.matFn = None
        super().__init__(self._handle_version(), file_type)

    def _handle_version(self):
        if os.path.isfile(self.loc + 'features.tsv.gz'):
            self.matFn = self.loc + 'matrix.mtx.gz'
            grps = {'feature_ids': ('features.tsv.gz', 0),
                    'feature_names': ('features.tsv.gz', 1),
                    'feature_types': ('features.tsv.gz', 2),
                    'cell_names': ('barcodes.tsv.gz', 0)}
        elif os.path.isfile(self.loc + 'features.tsv.gz'):  # sometimes somebody might have gunziped these files...
            self.matFn = self.loc + 'matrix.mtx'
            grps = {'feature_ids': ('features.tsv', 0),
                    'feature_names': ('features.tsv', 1),
                    'feature_types': None,
                    'cell_names': ('barcodes.tsv', 0)}
        elif os.path.isfile(self.loc + 'genes.tsv'):
            self.matFn = self.loc + 'matrix.mtx'
            grps = {'feature_ids': ('genes.tsv', 0),
                    'feature_names': ('genes.tsv', 1),
                    'feature_types': None,
                    'cell_names': ('barcodes.tsv', 0)}
        elif os.path.isfile(self.loc + 'genes.tsv.gz'):
            self.matFn = self.loc + 'matrix.mtx.gz'
            grps = {'feature_ids': ('genes.tsv.gz', 0),
                    'feature_names': ('genes.tsv.gz', 1),
                    'feature_types': ('genes.tsv.gz', 2),
                    'cell_names': ('barcodes.tsv.gz', 0)}
        else:
            raise IOError("ERROR: Couldn't find either of these expected combinations of files:\n"
                          "\t- matrix.mtx, barcodes.tsv and genes.tsv\n"
                          "\t- matrix.mtx.gz, barcodes.tsv.gz and features.tsv.gz\n"
                          "Please make sure that you have not compressed or uncompressed the Cellranger output files "
                          "manually")
        return grps

    def _read_dataset(self, key: Optional[str] = None):
        try:
            vals = [x.split('\t')[self.grpNames[key][1]] for x in
                    read_file(self.loc + self.grpNames[key][0])]
        except IndexError:
            vals = None
        return vals

    # noinspection DuplicatedCode
    def to_sparse(self, a):
        """
        Returns the input data as a sparse (COO) matrix.

        Args:
            a: a dense numpy matrix
        """
        idx = np.where(np.diff(a[:, 1]) == 1)[0] + 1
        return sparse.COO([(a[:, 1] - a[0, 1]).astype(int), (a[:, 0] - 1).astype(int)],
                          a[:, 2], shape=(len(idx) + 1, self.nFeatures))

    def _subset_by_assay(self, v, assay) -> List:
        if assay is None:
            return v
        elif assay not in self.assayFeats:
            raise ValueError("ERROR: Assay ID %s is not valid" % assay)
        if len(self.assayFeats[assay].shape) == 2:
            ret_val = []
            for i in self.assayFeats[assay].values[1:3].T:
                ret_val.extend(list(v[i[0]: i[1]]))
            return ret_val
        elif len(self.assayFeats[assay].shape) == 1:
            idx = self.assayFeats[assay]
            return v[idx.start: idx.end]
        else:
            raise ValueError("ERROR: assay feats is 3D. Something went really wrong. Create a github issue")

    # noinspection DuplicatedCode
    def consume(self, batch_size: int, lines_in_mem: int = int(1e5)) -> \
            Generator[List[np.ndarray], None, None]:
        stream = pd.read_csv(self.matFn, skiprows=3, sep='\t',
                             header=None, chunksize=lines_in_mem)
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
    """
    A class to read in data from a H5ad file (h5 file with AnnData information).

    Attributes:
        h5: A File object from the h5py package.
        matrix_key: Group where in the sparse matrix resides (default: 'X')
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
    def __init__(self, h5ad_fn: str, cell_attrs_key: str = 'obs', cell_ids_key: str = '_index',
                 feature_attrs_key: str = 'var', feature_ids_key: str = '_index',
                 feature_name_key: str = 'gene_short_name',  matrix_key: str = 'X',
                 category_names_key: str = '__categories', dtype: str = None):
        """
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
        """

        self.h5 = h5py.File(h5ad_fn, mode='r')
        self.matrixKey = matrix_key
        self.cellAttrsKey, self.featureAttrsKey = cell_attrs_key, feature_attrs_key
        self.groupCodes = {self.cellAttrsKey: self._validate_group(self.cellAttrsKey),
                           self.featureAttrsKey: self._validate_group(self.featureAttrsKey),
                           self.matrixKey: self._validate_group(self.matrixKey)}
        self.nCells, self.nFeatures = self._get_n(self.cellAttrsKey), self._get_n(self.featureAttrsKey)
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
            logger.warning(f"`{group}` slot in H5ad file is not of Dataset or Group type. "
                           f"Due to this, no information in `{group}` can be used")
            ret_val = 0
        if ret_val == 2:
            if len(self.h5[group].keys()) == 0:
                logger.warning(f"`{group}` slot in H5ad file is empty.")
                ret_val = 0
            elif len(set([self.h5[group][x].shape[0] for x in self.h5[group].keys() if
                          type(self.h5[group][x]) == h5py.Dataset])) > 1:
                if sorted(self.h5[group].keys()) != ['data', 'indices', 'indptr']:
                    logger.info(f"`{group}` slot in H5ad file has unequal sized child groups")
        return ret_val

    def _get_matrix_dtype(self):
        if self.groupCodes[self.matrixKey] == 1:
            return self.h5[self.matrixKey].dtype
        elif self.groupCodes[self.matrixKey] == 2:
            return self.h5[self.matrixKey]['data'].dtype
        else:
            raise ValueError(f"ERROR: {self.matrixKey} is neither Dataset or Group type. Will not consume data")

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
            if key.startswith('_'):
                temp_key = key[1:]
                if self._check_exists(group, temp_key):
                    return temp_key
        return key

    def _get_n(self, group: str) -> int:
        if self.groupCodes[group] == 0:
            if self._check_exists(self.matrixKey, 'shape'):
                return self.h5[self.matrixKey]['shape'][0]
            else:
                raise KeyError(f"ERROR: `{group}` not found and `shape` key is missing in the {self.matrixKey} group. "
                               f"Aborting read process.")
        elif self.groupCodes[group] == 1:
            return self.h5[group].shape[0]
        else:
            for i in self.h5[group].keys():
                if type(self.h5[group][i]) == h5py.Dataset:
                    return self.h5[group][i].shape[0]
            raise KeyError(f"ERROR: `{group}` key doesn't contain any child node of Dataset type."
                           f"Aborting because unexpected H5ad format.")

    def cell_ids(self) -> np.ndarray:
        """
        Returns a list of cell IDs.
        """
        if self._check_exists(self.cellAttrsKey, self.cellIdsKey):
            if self.groupCodes[self.cellAttrsKey] == 1:
                return self.h5[self.cellAttrsKey][self.cellIdsKey]
            else:
                return self.h5[self.cellAttrsKey][self.cellIdsKey][:]
        logger.warning(f"Could not find cells ids key: {self.cellIdsKey} in `obs`.")
        return np.array([f'cell_{x}' for x in range(self.nCells)])

    # noinspection DuplicatedCode
    def feat_ids(self) -> np.ndarray:
        """
        Returns a list of feature IDs.
        """
        if self._check_exists(self.featureAttrsKey, self.featIdsKey):
            if self.groupCodes[self.featureAttrsKey] == 1:
                return self.h5[self.featureAttrsKey][self.featIdsKey]
            else:
                return self.h5[self.featureAttrsKey][self.featIdsKey][:]
        logger.warning(f"Could not find feature ids key: {self.featIdsKey} in {self.featureAttrsKey}.")
        return np.array([f'feature_{x}' for x in range(self.nFeatures)])

    # noinspection DuplicatedCode
    def feat_names(self) -> np.ndarray:
        """
        Returns a list of feature names.
        """
        if self._check_exists(self.featureAttrsKey, self.featNamesKey):
            if self.groupCodes[self.featureAttrsKey] == 1:
                values = self.h5[self.featureAttrsKey][self.featNamesKey]
            else:
                values = self.h5[self.featureAttrsKey][self.featNamesKey][:]
            return self._replace_category_values(values, self.featNamesKey, self.featureAttrsKey).astype(object)
        logger.warning(f"Could not find feature names key: {self.featNamesKey} in self.featureAttrsKey.")
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
        if 'uns' in self.h5:
            if key+'_categories' in self.h5['uns']:
                c = self.h5['uns'][key+'_categories'][:]
                try:
                    return np.array([c[x] for x in v])
                except (IndexError, TypeError):
                    return v
        return v

    def _get_col_data(self, group: str, ignore_keys: List[str]) -> Generator[Tuple[str, np.ndarray], None, None]:
        if self.groupCodes[group] == 1:
            for i in tqdm(self.h5[group].dtype.names, desc=f"Reading attributes from group {group}"):
                if i in ignore_keys:
                    continue
                yield i, self._replace_category_values(self.h5[group][i][:], i, group)
        if self.groupCodes[group] == 2:
            for i in tqdm(self.h5[group].keys(), desc=f"Reading attributes from group {group}"):
                if i in ignore_keys:
                    continue
                if type(self.h5[group][i]) == h5py.Dataset:
                    yield i, self._replace_category_values(self.h5[group][i][:], i, group)

    def get_cell_columns(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Creates a Generator that yields the cell columns.
        """
        for i, j in self._get_col_data(self.cellAttrsKey, [self.cellIdsKey]):
            yield i, j

    def get_feat_columns(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Creates a Generator that yields the feature columns.
        """
        for i, j in self._get_col_data(self.featureAttrsKey, [self.featIdsKey, self.featNamesKey]):
            yield i, j

    # noinspection DuplicatedCode
    def consume_dataset(self, batch_size: int = 1000) -> Generator[sparse.COO, None, None]:
        """
        Returns a generator that yield chunks of data.
        """
        dset = self.h5[self.matrixKey]
        s = 0
        for e in range(batch_size, dset.shape[0] + batch_size, batch_size):
            if e > dset.shape[0]:
                e = dset.shape[0]
            yield dset[s:e]
            s = e

    def consume_group(self, batch_size: int) -> Generator[sparse.COO, None, None]:
        """
        Returns a generator that yield chunks of data.
        """
        grp = self.h5[self.matrixKey]
        s = 0
        for ind_n in range(0, self.nCells, batch_size):
            i = grp['indptr'][ind_n:ind_n + batch_size]
            e = i[-1]
            if s != 0:
                idx = np.array([s] + list(i))
                idx = idx - idx[0]
            else:
                idx = np.array(i)
            n = idx.shape[0] - 1
            nidx = np.repeat(range(n), np.diff(idx).astype('int32'))
            yield sparse.COO([nidx, grp['indices'][s: e]], grp['data'][s: e],
                             shape=(n, self.nFeatures)).todense()
            s = e

    def consume(self, batch_size: int = 1000):
        """
        Returns a generator that yield chunks of data.
        """
        if self.groupCodes[self.matrixKey] == 1:
            return self.consume_dataset(batch_size)
        elif self.groupCodes[self.matrixKey] == 2:
            return self.consume_group(batch_size)


class NaboH5Reader:
    """
    A class to read in data in the form of a Nabo H5 file.

    Attributes:
        h5: A File object from the h5py package.
        nCells: Number of cells in dataset.
        nFeatures: Number of features in dataset.
    """
    def __init__(self, h5_fn: str):
        """
        Args:
            h5_fn: Path to H5 file.
        """
        self.h5 = h5py.File(h5_fn, mode='r')
        self._check_integrity()
        self.nCells = self.h5['names']['cells'].shape[0]
        self.nFeatures = self.h5['names']['genes'].shape[0]

    def _check_integrity(self) -> bool:
        for i in ['cell_data', 'gene_data', 'names']:
            if i not in self.h5:
                raise KeyError(f"ERROR: Expected group: {i} is missing in the H5 file")
        return True

    def cell_ids(self) -> List[str]:
        """
        Returns a list of cell IDs.
        """
        return [x.decode('UTF-8') for x in self.h5['names']['cells'][:]]

    def feat_ids(self) -> np.ndarray:
        """
        Returns a list of feature IDs.
        """
        return np.array([f'feature_{x}' for x in range(self.nFeatures)])

    def feat_names(self) -> List[str]:
        """
        Returns a list of feature names.
        """
        return [x.decode('UTF-8').rsplit('_', 1)[0] for x in self.h5['names']['genes'][:]]

    def consume(self, batch_size: int = 100) -> Generator[np.ndarray, None, None]:
        """
        Returns a generator that yield chunks of data.
        """
        batch = []
        for i in self.h5['cell_data']:
            a = np.zeros(self.nFeatures).astype(int)
            v = self.h5['cell_data'][i][:][::-1]
            a[v['idx']] = v['val']
            batch.append(a)
            if len(batch) >= batch_size:
                batch = np.array(batch)
                yield batch
                batch = []
        if len(batch) > 0:
            yield np.array(batch)


class LoomReader:
    """
    A class to read in data in the form of a Loom file.

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
    def __init__(self, loom_fn: str, matrix_key: str = 'matrix',
                 cell_attrs_key='col_attrs', cell_names_key: str = 'obs_names',
                 feature_attrs_key: str = 'row_attrs', feature_names_key: str = 'var_names',
                 feature_ids_key: str = None, dtype: str = None) -> None:
        """
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
        """
        self.h5 = h5py.File(loom_fn, mode='r')
        self.matrixKey = matrix_key
        self.cellAttrsKey, self.featureAttrsKey = cell_attrs_key, feature_attrs_key
        self.cellNamesKey, self.featureNamesKey = cell_names_key, feature_names_key
        self.featureIdsKey = feature_ids_key
        self.matrixDtype = self.h5[self.matrixKey].dtype if dtype is None else dtype
        self._check_integrity()
        self.nFeatures, self.nCells = self.h5[self.matrixKey].shape

    def _check_integrity(self) -> bool:
        if self.matrixKey not in self.h5:
            raise KeyError(f"ERROR: Matrix key (location): {self.matrixKey} is missing in the H5 file")
        if self.cellAttrsKey not in self.h5:
            logger.warning(f"Cell attributes are missing. Key {self.cellAttrsKey} was not found")
        if self.featureAttrsKey not in self.h5:
            logger.warning(f"Feature attributes are missing. Key {self.featureAttrsKey} was not found")
        return True

    def cell_names(self) -> List[str]:
        """
        Returns a list of names of the cells in the dataset.
        """
        if self.cellAttrsKey not in self.h5:
            pass
        elif self.cellNamesKey not in self.h5[self.cellAttrsKey]:
            logger.warning(f"Cell names/ids key ({self.cellNamesKey}) is missing in attributes")
        else:
            return self.h5[self.cellAttrsKey][self.cellNamesKey][:]
        return [f"cell_{x}" for x in range(self.nCells)]

    def cell_ids(self) -> List[str]:
        """
        Returns a list of cell IDs.
        """
        return self.cell_names()

    def _stream_attrs(self, key, ignore) -> Generator[Tuple[str, np.ndarray], None, None]:
        if key in self.h5:
            for i in tqdm(self.h5[key].keys(), desc=f"Reading {key} attributes"):
                if i in [ignore]:
                    continue
                vals = self.h5[key][i][:]
                if vals.dtype.names is None:
                    yield i, vals
                else:
                    # Attribute is a structured array
                    for j in vals.dtype.names:
                        yield i + '_' + str(j), vals[j]

    def get_cell_attrs(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Returns a Generator that yields the cells' attributes.
        """
        return self._stream_attrs(self.cellAttrsKey, [self.cellNamesKey])

    def feature_names(self) -> List[str]:
        """
        Returns a list of feature names.
        """
        if self.featureAttrsKey not in self.h5:
            pass
        elif self.featureNamesKey not in self.h5[self.featureAttrsKey]:
            logger.warning(f"Feature names key ({self.featureNamesKey}) is missing in attributes")
        else:
            return self.h5[self.featureAttrsKey][self.featureNamesKey][:]
        return [f"feature_{x}" for x in range(self.nFeatures)]

    def feature_ids(self) -> List[str]:
        """
        Returns a list of feature IDs.
        """
        if self.featureAttrsKey not in self.h5:
            pass
        elif self.featureIdsKey is None:
            pass
        elif self.featureIdsKey not in self.h5[self.featureAttrsKey]:
            logger.warning(f"Feature names key ({self.featureIdsKey}) is missing in attributes")
        else:
            return self.h5[self.featureAttrsKey][self.featureIdsKey][:]
        return [f"feature_{x}" for x in range(self.nFeatures)]

    def get_feature_attrs(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Returns a Generator that yields the features' attributes.
        """
        return self._stream_attrs(self.featureAttrsKey,
                                  [self.featureIdsKey, self.featureNamesKey])

    def consume(self, batch_size: int = 1000) -> Generator[np.ndarray, None, None]:
        """
        Returns a generator that yield chunks of data.
        """
        last_n = 0
        for i in range(batch_size, self.nCells, batch_size):
            yield self.h5[self.matrixKey][:, last_n:i].astype(self.matrixDtype).T
            last_n = i
        if last_n < self.nCells:
            yield self.h5[self.matrixKey][:, last_n:].astype(self.matrixDtype).T
