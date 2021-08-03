"""
Contains the MetaData class, which is used for storing metadata about cells and features.
"""
from zarr import hierarchy as zarr_hierarchy
from zarr import array as zarr_array
import numpy as np
import re
import pandas as pd
from typing import List, Iterable, Any, Dict, Tuple
from .feat_utils import fit_lowess
from .writers import create_zarr_obj_array
from .utils import logger

__all__ = ["MetaData"]


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
    """
    MetaData class for cells and features.

    This class provides an interface to perform CRUD operations on metadata, saved in the Zarr hierarchy.
    All the changes at the metadata are synchronized on disk.

    Attributes:
        locations: The locations for where the metadata is stored.
        N: The size of the primary data.
        index: A numpy array with the indices of the cells/features.
    """

    def __init__(self, zgrp: zarr_hierarchy):
        """
        Args:
            zgrp: Zarr hierarchy object wherein metadata arrays are saved.
        """
        self.locations: Dict[str, zarr_hierarchy] = {"primary": zgrp}
        self.N = self._get_size(self.locations["primary"], strict_mode=True)
        self.index = np.array(range(self.N))

    def _get_size(self, zgrp: zarr_hierarchy, strict_mode: bool = False) -> int:
        """

        Args:
            zgrp:
            strict_mode:

        Returns:

        """
        sizes = []
        for i in zgrp.keys():
            sizes.append(zgrp[i].shape[0])
        if len(sizes) > 0:
            if len(set(sizes)) != 1:
                raise ValueError(
                    "ERROR: Metadata table is corrupted. Not all columns are of same length"
                )
            return sizes[0]
        else:
            if strict_mode:
                raise ValueError("Attempted to get size of empty zarr group")
            else:
                return self.N

    @staticmethod
    def _col_renamer(loc: str, col: str) -> str:
        """

        Args:
            loc:
            col:

        Returns:

        """
        if loc != "primary":
            return f"{loc}_{col}"
        return col

    def _column_map(self) -> Dict[str, Tuple[str, str]]:
        """

        Returns:

        """
        reserved_cols = ["I", "ids", "names"]
        col_map = {x: "primary" for x in reserved_cols}
        for loc, zgrp in self.locations.items():
            for i in zgrp.keys():
                j = self._col_renamer(loc, i)
                if j in col_map and j not in reserved_cols:
                    logger.warning(
                        f" {i} is duplicate in metadata loc {loc}. This means something has failed "
                        f"upstream. This is quite unexpected. Please report this issue."
                    )
                col_map[j] = (loc, i)
        return col_map

    def _get_loc(self, column: str) -> Tuple[str, str]:
        """

        Args:
            column:

        Returns:

        """
        col_map = self._column_map()
        if column not in col_map:
            raise KeyError(f"{column} does not exist in the metadata columns.")
        loc, col = col_map[column]
        return loc, col

    def _get_array(self, column: str) -> zarr_array:
        """

        Args:
            column:

        Returns:

        """
        loc, col = self._get_loc(column)
        return self.locations[loc][col]

    def get_dtype(self, column: str) -> type:
        """
        Returns the dtype for the given column.

        Args:
            column: Column name of the table.
        """
        return self._get_array(column).dtype

    def _verify_bool(self, key: str) -> bool:
        """
        Validates if a give table column (parameter 'key') is bool type

        Args:
            key: Name of the column to query

        Returns: None

        """

        if self.get_dtype(key) != bool:
            raise TypeError(
                "ERROR: `key` should be name of a boolean type column in Metadata table"
            )
        return True

    def mount_location(self, zgrp: zarr_hierarchy, identifier: str) -> None:
        """

        Args:
            zgrp:
            identifier:

        Returns:

        """
        if identifier in self.locations:
            raise ValueError(
                f"ERROR: a location with identifier '{identifier}' already mounted"
            )
        size = self._get_size(zgrp)
        if size != self.N:
            raise ValueError(
                f"ERROR: The index size of the mount location ({size}) is not same as primary ({self.N})"
            )
        new_cols = [self._col_renamer(identifier, x) for x in zgrp.keys()]
        cols = self.columns
        conflict_names = [x for x in new_cols if x in cols]
        if len(conflict_names) > 0:
            conflict_names = " ".join(conflict_names)
            raise ValueError(
                f"ERROR: These names in location conflict with existing names: {conflict_names}\n. "
                f"Please try with a different identifier value."
            )
        self.locations[identifier] = zgrp

    def unmount_location(self, identifier: str) -> None:
        """

        Args:
            identifier:

        Returns:

        """
        if identifier == "primary":
            raise ValueError("Cannot unmount the primary location")
        if identifier not in self.locations:
            logger.warning(f"{identifier} is not mounted. Nothing to unmount")
            return None
        self.locations.pop(identifier)

    @property
    def columns(self) -> List[str]:
        """

        Returns:

        """
        return list(self._column_map().keys())

    def fetch_all(self, column: str) -> np.ndarray:
        """

        Args:
            column:

        Returns:

        """
        return self._get_array(column)[:]

    def active_index(self, key: str) -> np.ndarray:
        """

        Args:
            key:

        Returns:

        """
        if self._verify_bool(key):
            return self.index[self.fetch_all(key)]
        else:
            raise ValueError(
                "ERROR: Unexpected error when verifying boolean key. Please report this issue"
            )

    def fetch(self, column: str, key: str = "I") -> np.ndarray:
        """
        Get column values for only valid rows

        Args:
            column:
            key:

        Returns:

        """

        return self.fetch_all(column)[self.active_index(key)]

    def _save(
        self, column_name: str, values: np.ndarray, location: str = "primary"
    ) -> None:
        """

        Args:
            column_name:
            values:

        Returns:

        """
        if location not in self.locations:
            raise KeyError(
                f"ERROR: '{location}' has not been mounted. Save data request failed!"
            )
        if values.shape != (self.N,):
            raise ValueError(
                f"ERROR: Values are of shape: {values.shape}. Expected shape is: ({self.N},)"
            )
        create_zarr_obj_array(
            self.locations[location], column_name, values, values.dtype
        )
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
                raise ValueError(
                    f"ERROR: `values`  are of incorrect length ({len(values)}). "
                    f" Chosen key ({key}) has {len(k)} active rows"
                )
            else:
                a = np.empty(self.N).astype(type(values[0]))
                a[k] = values
                a[~k] = fill_value
                return a

    def get_index_by(
        self, value_targets: List[Any], column: str, key: str = None
    ) -> np.ndarray:
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
                logger.warning(
                    f"{missing_count} values were not found in the table column {column}"
                )
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
        if len(idx) > 0:
            a[idx] = True
        if invert:
            a = ~a
        return a

    def insert(
        self,
        column_name: str,
        values: np.array,
        fill_value: Any = np.NaN,
        key: str = "I",
        overwrite: bool = False,
        location: str = "primary",
        force: bool = False,
    ) -> None:
        """
        Insert a column into the table.

        Args:
            column_name (str): Name of column to modify.
            values (np.array): Values the column should contain.
            fill_value (Any = np.NaN): Value to fill unassigned slots with.
            key (str = 'I'):
            overwrite (bool = False): Should function overwrite column if it already exists?
            location (str = 'primary'):
            force (bool = False): Enforce change to column, even if column is a protected column name ('I' or 'ids').

        Returns:
            None
        """
        col = self._col_renamer(location, column_name)
        if col in ["I", "ids"] and force is False:
            raise ValueError(
                f"ERROR: {col} is a protected column name in MetaData class."
            )
        if col in self.columns and overwrite is False:
            raise ValueError(
                f"ERROR: {col} already exists. Please set `overwrite` to True to overwrite."
            )
        if type(values) == list:
            logger.warning(
                "'values' parameter is of `list` type and not `np.ndarray` as expected. The correct dtype "
                "may not be assigned to the column"
            )
            values = np.array(values)
        v = self._fill_to_index(values, fill_value, key)
        self._save(column_name, v.astype(values.dtype), location=location)
        return None

    def update_key(self, values: np.array, key) -> None:
        """
        Modify a column in the metadata table, specified with `key`.

        Args:
            values: The values to update the column with.
            key: Which column in the metadata table to update.

        Returns:
            None
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
        if column in ["I", "ids", "names"]:
            raise ValueError(
                f"ERROR: {column} is a protected name in MetaData class. Cannot be deleted"
            )
        # noinspection PyUnusedLocal
        loc, col = self._get_loc(column)
        del self.locations[loc][col]
        return None

    def sift(
        self, column: str, min_v: float = -np.Inf, max_v: float = np.Inf
    ) -> np.ndarray:
        """

        Args:
            column:
            min_v:
            max_v:

        Returns:

        """
        values = self.fetch_all(column)
        return (values > min_v) & (values < max_v)

    def multi_sift(
        self, columns: List[str], lows: Iterable, highs: Iterable
    ) -> np.ndarray:
        """

        Args:
            columns:
            lows:
            highs:

        Returns:

        """
        ret_val = _all_true(
            np.array([self.sift(i, j, k) for i, j, k in zip(columns, lows, highs)])
        )
        return ret_val

    def head(self, n: int = 5) -> pd.DataFrame:
        """

        Args:
            n:

        Returns:

        """
        df = pd.DataFrame({x: self.fetch_all(x)[:n] for x in self.columns})
        return df

    def to_pandas_dataframe(self, columns: List[str], key: str = None) -> pd.DataFrame:
        """
        Returns the requested columns as a Pandas dataframe, sorted on key.
        """
        valid_cols = self.columns
        df = pd.DataFrame({x: self.fetch_all(x) for x in columns if x in valid_cols})
        if key is not None:
            df = df.reindex(self.active_index(key))
        return df

    def grep(self, pattern: str, only_valid=False) -> List[str]:
        """

        Args:
            pattern:
            only_valid:

        Returns:

        """
        names = np.array(list(map(str.upper, self.fetch_all("names"))))
        if only_valid:
            names = names[self.active_index("I")]
        return sorted(
            set([x for x in names if re.match(pattern.upper(), x) is not None])
        )

    def remove_trend(
        self, x: str, y: str, n_bins: int = 200, lowess_frac: float = 0.1
    ) -> np.ndarray:
        """

        Args:
            x:
            y:
            n_bins:
            lowess_frac:

        Returns:

        """
        a = fit_lowess(
            self.fetch(x).astype(float),
            self.fetch(y).astype(float),
            n_bins,
            lowess_frac,
        )
        return a

    def __repr__(self):
        return f"MetaData of {self.fetch_all('I').sum()}({self.N}) elements"
