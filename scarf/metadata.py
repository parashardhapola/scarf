from zarr import hierarchy as zarr_hierarchy
import numpy as np
import re
import pandas as pd
from typing import List, Iterable, Any
from .feat_utils import fit_lowess
from .writers import create_zarr_obj_array
from .logging_utils import logger

__all__ = ['MetaData']


def _all_true(bools: np.ndarray) -> np.ndarray:
    """

    Args:
        bools:

    Returns:

    """
    a = bools.sum(axis=0)
    a[a < bools.shape[0]] = 0
    return a.astype(bool)


class MetaData:

    def __init__(self, zgrp: zarr_hierarchy):
        """
        This class provides an interface to perform CRUD operations on metadata, saved in the Zarr hierarchy.
         All the changes ot the metadata are synchronized on disk.

        Args:
            zgrp: Zarr hierarchy object wherein metadata arrays are saved
        """

        self._zgrp = zgrp
        self.N = self._get_size()
        self.index = np.array(range(self.N))

    @property
    def columns(self) -> List[str]:
        """

        Returns:

        """
        c = ['I', 'ids', 'names']
        c = c + [x for x in sorted(self._zgrp.keys()) if x not in c]
        return c

    def _get_size(self):
        sizes = []
        for i in self.columns:
            sizes.append(self._zgrp[i].shape[0])
        if len(set(sizes)) != 1:
            raise ValueError("ERROR: Metadata table is corrupted. Not all columns are of same length")
        return sizes[0]

    def head(self, n: int = 5) -> pd.DataFrame:
        """

        Args:
            n:

        Returns:

        """
        df = pd.DataFrame({
            x: self._zgrp[x][:n] for x in self.columns
        })
        return df

    def to_pandas_dataframe(self, columns: List[str], key: str = None) -> pd.DataFrame:
        """

        Args:
            columns:
            key:

        Returns:

        """
        valid_cols = self.columns
        df = pd.DataFrame({x: self._zgrp[x][:] for x in columns if x in valid_cols})
        if key is not None:
            df = df.reindex(self.active_index(key))
        return df

    def get_dtype(self, column: str) -> type:
        """

        Args:
            column: Column name of the table

        Returns:

        """
        return self._zgrp[column].dtype

    def _verify_bool(self, key: str) -> None:
        """
        Validates if a give table column (parameter 'key') is bool type

        Args:
            key: Name of the column to query

        Returns: None

        """

        if self.get_dtype(key) != bool:
            raise TypeError("ERROR: `key` should be name of a boolean type column in Metadata table")
        return None

    def _fill_to_index(self, values: np.array, fill_value, key: str) -> np.ndarray:
        """
        Makes sure that the array being added to the table is of the same shape.
        If the array has same shape as the table then the input array is added straightaway.
        It is assumed that the input array is in same order as table rows.

        Args:
            values:
            fill_value:
            key:

        Returns:

        """

        if len(values) == self.N:
            return values
        else:
            self._verify_bool(key)
            k = self.fetch_all(key)
            if len(values) != k.sum():
                raise ValueError(f"ERROR: `values`  are of incorrect length ({len(values)}). "
                                 f" Chosen key ({key}) has {len(k)} active rows")
            else:
                a = np.empty(self.N).astype(type(values[0]))
                a[k] = values
                a[~k] = fill_value
                return a

    def _save(self, column_name: str, values: np.ndarray) -> None:
        """

        Args:
            column_name:
            values:

        Returns:

        """
        create_zarr_obj_array(self._zgrp, column_name, values, values.dtype)
        return None

    def active_index(self, key: str) -> np.ndarray:
        """

        Args:
            key:

        Returns:

        """
        self._verify_bool(key)
        return self.index[self.fetch_all(key)]

    def get_index_by(self, value_targets: List[Any], column: str, key: str = None) -> np.ndarray:
        """

        Args:
            value_targets:
            column:
            key:
        Returns:

        """
        if isinstance(value_targets, Iterable) and type(value_targets) != str:
            if key is None:
                values = self.fetch_all(column)
            else:
                values = self.fetch(column, key)
            value_map = {}
            for n, x in enumerate(values):
                x = x.upper()
                if x not in value_map:
                    value_map[x] = []
                value_map[x].append(n)
            ret_val = []
            missing_count = 0
            for i in value_targets:
                i = i.upper()
                if i in value_map:
                    ret_val.extend(value_map[i])
                else:
                    missing_count += 1
            if missing_count > 0:
                logger.warning(f"{missing_count} values were not found in the table column {column}")
            return np.array(ret_val)
        else:
            raise TypeError("ERROR: Please provide the `value_targets` as list")

    def index_to_bool(self, idx: np.ndarray, invert: bool = False) -> np.ndarray:
        """

        Args:
            idx:
            invert:

        Returns:

        """
        a = np.zeros(self.N, dtype=bool)
        a[idx] = True
        if invert:
            a = ~a
        return a

    def fetch_all(self, column: str) -> np.ndarray:
        """

        Args:
            column:

        Returns:

        """

        if column is None or column not in self.columns:
            raise KeyError(f"ERROR: '{column}' not found in the MetaData table")
        return self._zgrp[column][:].astype(self.get_dtype(column))

    def fetch(self, column: str, key: str = 'I') -> np.ndarray:
        """
        Get column values for only valid rows

        Args:
            column:
            key:

        Returns:

        """

        return self.fetch_all(column)[self.active_index(key)]

    def insert(self, column_name: str, values: np.array, fill_value: Any = np.NaN,
               key: str = 'I', overwrite: bool = False) -> None:
        """
        add

        Args:
            column_name:
            values:
            fill_value:
            key:
            overwrite:

        Returns:

        """
        if column_name in ['I', 'ids']:
            raise ValueError(f"ERROR: {column_name} is a protected column name in MetaData class.")
        if column_name in self.columns and overwrite is False:
            raise ValueError(f"ERROR: {column_name} already exists. Please use `update` method instead.")
        if type(values) == list:
            logger.warning("'values' parameter is of `list` type and not `np.ndarray` as expected. The correct dtype "
                           "may not be assigned to the the column")
            values = np.array(values)
        v = self._fill_to_index(values, fill_value, key)
        self._save(column_name, v.astype(values.dtype))
        return None

    def update_key(self, values: np.array, key) -> None:
        """


        Args:
            values:
            key:

        Returns:

        """
        v = self._fill_to_index(values, False, key)
        v = _all_true(np.array([v, self.fetch_all(key)]))
        self._save(key, v)
        return None

    def reset_key(self, key: str) -> None:
        """

        Args:
            key:

        Returns:

        """
        values = np.array([True for _ in range(self.N)]).astype(bool)
        self._save(key, values)
        return None

    def drop(self, column: str) -> None:
        """

        Args:
            column:

        Returns:

        """
        if column in ['I', 'ids', 'names']:
            raise ValueError(f"ERROR: {column} is a protected name in MetaData class. Cannot be deleted")
        if column not in self.columns:
            raise KeyError(f"{column} does not exist. Nothing to remove")
        if column not in self._zgrp:
            logger.warning(f"Unexpected inconsistency found: {column} is not present in the Zarr hierarchy")
        else:
            del self._zgrp[column]

    def sift(self, column: str, min_v: float = -np.Inf, max_v: float = np.Inf) -> np.ndarray:
        """

        Args:
            column:
            min_v:
            max_v:

        Returns:

        """
        values = self.fetch_all(column)
        return (values > min_v) & (values < max_v)

    def multi_sift(self, columns: List[str], lows: Iterable, highs: Iterable) -> np.ndarray:
        """

        Args:
            columns:
            lows:
            highs:

        Returns:

        """
        ret_val = _all_true(np.array([self.sift(i, j, k) for i, j, k
                                      in zip(columns, lows, highs)]))
        return ret_val

    def grep(self, pattern: str, only_valid=False) -> List[str]:
        """

        Args:
            pattern:
            only_valid:

        Returns:

        """
        names = np.array(list(map(str.upper, self.fetch_all('names'))))
        if only_valid:
            names = names[self.active_index('I')]
        return sorted(set([x for x in names
                           if re.match(pattern.upper(), x) is not None]))

    def remove_trend(self, x: str, y: str, n_bins: int = 200,
                     lowess_frac: float = 0.1) -> np.ndarray:
        """

        Args:
            x:
            y:
            n_bins:
            lowess_frac:

        Returns:

        """
        a = fit_lowess(self.fetch(x).astype(float),
                       self.fetch(y).astype(float),
                       n_bins, lowess_frac)
        return a

    def __repr__(self):
        return f"MetaData of {self.fetch_all('I').sum()}({self.N}) elements"
