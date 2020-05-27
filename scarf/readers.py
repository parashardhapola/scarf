from abc import ABC, abstractmethod
from typing import Generator, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import h5py
import os
import sparse
import gzip
from typing import IO


__all__ = ['CrH5Reader', 'CrDirReader', 'CrReader']


def get_file_handle(fn: str) -> IO:
    try:
        if fn.rsplit('.', 1)[-1] == 'gz':
            return gzip.open(fn, mode='rt')
        else:
            return open(fn, 'r')
    except (OSError, IOError, FileNotFoundError):
        raise FileNotFoundError("ERROR: FILE NOT FOUND: %s" % fn)


def read_file(fn: str):
    fh = get_file_handle(fn)
    for line in fh:
        yield line.rstrip()


class CrReader(ABC):
    def __init__(self, grp_names, file_type):
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
            print(f"INFO: {main_name_v} already present")
            # Making sure that column name is in uppercase 'RNA'
            newnames = list(self.assayFeats.columns)
            newnames[anames.index(main_name_v)] = main_name_v
            self.assayFeats.columns = newnames
        else:
            at = self.assayFeats.T['type'] == main_name_k
            if at.sum() == 1:
                main_assay = at[at].index[0]
            else:
                print('WARNING:')
                main_assay = self.assayFeats.T[at].nFeatures.astype(int).idxmax()
            self.rename_assays({main_assay: main_name_v})

    def rename_assays(self, name_map: Dict[str, str]) -> None:
        self.assayFeats.rename(columns=name_map, inplace=True)

    def feature_ids(self, assay: str = None) -> List[str]:
        return self._subset_by_assay(
            self._read_dataset('feature_ids'), assay)

    def feature_names(self, assay: str = None) -> List[str]:
        vals = self._read_dataset('feature_names')
        if vals is None:
            print('WARNING: Feature names extraction failed using feature IDs', flush=True)
            vals = self._read_dataset('feature_ids')
        return self._subset_by_assay(vals, assay)

    def feature_types(self) -> List[str]:
        if self.grpNames['feature_types'] is not None:
            return self._read_dataset('feature_types')
        else:
            default_name = list(self.autoNames.keys())[0]
            return [default_name for _ in range(self.nFeatures)]

    def cell_names(self) -> List[str]:
        return self._read_dataset('cell_names')


class CrH5Reader(CrReader):
    def __init__(self, h5_fn, file_type: str = None):
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
        self.h5obj.close()


class CrDirReader(CrReader):
    def __init__(self, loc, file_type: str = None):
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
            raise IOError("ERROR: Couldn't find files")
        return grps

    def _read_dataset(self, key: Optional[str] = None):
        try:
            vals = [x.split('\t')[self.grpNames[key][1]] for x in
                    read_file(self.loc + self.grpNames[key][0])]
        except IndexError:
            vals = None
        return vals

    def to_sparse(self, a):
        idx = np.where(np.diff(a[:, 1]) == 1)[0] + 1
        return sparse.COO([a[:, 1] - a[0, 1], a[:, 0] - 1], a[:, 2], shape=(len(idx) + 1, self.nFeatures))

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
