import numpy as np
import re
import pandas as pd
from typing import List, Iterable
from .utils import fit_lowess
from .writers import create_zarr_obj_array
from .logging_utils import logger

__all__ = ['MetaData']


def load_zarr_table(zgrp) -> pd.DataFrame:
    keys = ['I', 'ids', 'names']
    keys = keys + [x for x in zgrp.keys() if x not in keys]
    return pd.DataFrame({x: zgrp[x][:] for x in keys})


class MetaData:

    def __init__(self, zgrp):
        self._zgrp = zgrp
        # TODO: think of a strategy that the datframe columns (except I) are made immutable
        self.table = load_zarr_table(self._zgrp)
        self.N = len(self.table)

    def active_index(self, key: str) -> np.ndarray:
        if self.table[key].dtype != bool:
            raise ValueError(f"ERROR: {key} is not of bool type")
        return self.table.index[self.table[key]].values

    def fetch(self, k: str, key: str = 'I') -> np.ndarray:
        """
        Get column values for only valid rows
        """
        if k not in self.table.columns or key not in self.table.columns:
            raise KeyError(f"ERROR: '{k}' not found in MetaData")
        return self.table[k].values[self.table[key].values]

    def add(self, k: str, v: np.array, fill_val=np.NaN, key: str = 'I', overwrite: bool = False) -> None:
        """
        Add new column
        """
        if k in ['I', 'ids']:
            raise ValueError(f"ERROR: {k} is a protected name in MetaData class."
                             "Please choose any other name")
        if k in self.table.columns and overwrite is False:
            raise ValueError(f"ERROR: attribute {k} already exists. Set overwrite to True")
        if len(v) == self.N:
            self.table[k] = v
        elif len(v) == self.table[key].sum():
            self.table[k] = self._expand(v, fill_val, key)
        else:
            raise ValueError("ERROR: Trying to add attr of incorrect length")
        self._save(k)
        return None

    def remove(self, k: str) -> None:
        if k in ['I', 'ids', 'names']:
            raise ValueError(f"ERROR: {k} is a protected name in MetaData class. Cannot be deleted")
        if k in self.table.columns:
            self.table.drop(columns=k, inplace=True)
            self._del(k)
        else:
            logger.warning(f"{k} does not exist. Nothing to remove")

    def update(self, bool_arr: np.array, key: str = 'I') -> None:
        """
        Update valid rows using a boolean array and 'and' operation
        """
        if len(bool_arr) != self.N:
            bool_arr = self._expand(bool_arr, False, key)
        self.table[key] = self.table[key] & bool_arr
        self._save(key)
        return None

    @staticmethod
    def sift(a: np.ndarray, min_v: float = -np.Inf, max_v: float = np.Inf) -> np.ndarray:
        ret_val = ((a > min_v) & (a < max_v))
        return ret_val

    def multi_sift(self, cols: List[str], lows: Iterable, highs: Iterable) -> np.ndarray:
        ret_val = self._and_bools([self.sift(self.table[i], j, k) for i, j, k
                                   in zip(cols, lows, highs)])
        return ret_val

    def get_idx_by_names(self, names: List[str], all_names: bool = True) -> List[int]:
        return self._get_idx([x.upper() for x in names], 'names', all_names)

    def get_idx_by_ids(self, ids: List[str], all_ids: bool = True) -> List[int]:
        return self._get_idx([x.upper() for x in ids], 'ids', all_ids)

    def grep(self, pattern: str, only_valid=False) -> List[str]:
        names = np.array(list(map(str.upper, self.table['names'].values)))
        if only_valid:
            names = names[self.table['I']]
        return sorted(set([x for x in names
                           if re.match(pattern.upper(), x) is not None]))

    def remove_trend(self, x: str, y: str, n_bins: int = 200,
                     lowess_frac: float = 0.1) -> np.ndarray:
        a = fit_lowess(self.fetch(x).astype(float),
                       self.fetch(y).astype(float),
                       n_bins, lowess_frac)
        return a

    def idx_to_bool(self, idx, invert: bool = False) -> np.ndarray:
        a = np.zeros(self.N, dtype=bool)
        a[idx] = True
        if invert:
            a = ~a
        return a

    @staticmethod
    def _and_bools(bools: List[np.ndarray]) -> np.ndarray:
        a = sum(bools)
        a[a < len(bools)] = 0
        return a.astype(bool)

    def _get_idx(self, k, d, f) -> List[int]:
        if isinstance(k, Iterable) and type(k) != str:
            if f:
                d = self.table[d].values
            else:
                d = self.fetch(d)
            d = np.array([x.upper() for x in d])
            return sum([list(np.where(d == x)[0]) for x in k], [])
        else:
            raise TypeError("ERROR: Please provide the ID/names as list")

    def _expand(self, v: np.array, fill_val, key: str = 'I') -> np.ndarray:
        """
        Makes sure that the array being added to the table is of the same shape.
        If the array has same shape as the table then the input array is added straightaway.
        It is assumed that the input array is in same order as table rows.
        """
        a = np.empty(self.N).astype(type(v[0]))
        k = self.table[key].values
        a[k] = v
        a[~k] = fill_val
        return a

    def reset(self) -> None:
        """
        Set all rows to active
        """
        self.table['I'] = np.array([True for _ in range(self.N)]).astype(bool)
        self._zgrp['I'] = self.table['I'].values

    def _make_data_table(self) -> pd.DataFrame:
        keys = ['I', 'ids', 'names']
        keys = keys + [x for x in self._zgrp.keys() if x not in keys]
        return pd.DataFrame({x: self._zgrp[x][:] for x in keys})

    def _save(self, key: str = None) -> None:
        if key is not None and key in self.table:
            create_zarr_obj_array(self._zgrp, key, self.table[key].values, self.table[key].dtype)
        # FIXME: Why is the following line here?
        self._zgrp['I'] = self.table['I'].values

    def _del(self, key: str = None) -> None:
        if key is not None and key in self._zgrp:
            del self._zgrp[key]

    def __repr__(self):
        return f"MetaData of {self.table['I'].sum()}({self.N}) elements"
