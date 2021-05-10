"""
====================================
Scarf - Single-cell atlases reformed
====================================

Scarf is a Python package that performs memory-efficient analysis of single-cell genomics data.

- Analyze atlas scale datasets on your laptop (tested with up to 4 million cells)
- Perform analysis of scATAC-Seq data (datasets with up to 700K cells with 1 million peaks tested) under 10 GB RAM
- Make parallel implementations of UMAP and tSNE (SG-tSNE) for quick cell embedding
- Perform hierarchical clustering that gives interpretable cluster relationships
- Sub-sample highly representative cells using state-of-the-art TopACeDo method
- Perform quick and accurate projections of cells from one dataset onto another or integrate multiple datasets.

Exports:
--------

- assay:
    - Assay
    - RNAassay
    - ATACassay
    - ADTassay
- datastore
    - `DataStore`: DataStore objects provide primary interface to interact with the data.
- readers
    - `CrH5Reader`: A class to read in CellRanger (Cr) h5 data.
    - CrDirReader
    - `CrReader`: A class to read in CellRanger (Cr) data.
    - H5adReader
    - MtxDirReader
    - NaboH5Reader
    - LoomReader
- utils
    - fetch_dataset
    - system_call
    - rescale_array
    - clean_array
    - show_progress
    - controlled_compute
- writers
    - create_zarr_dataset
    - create_zarr_obj_array
    - create_zarr_count_assay
    - subset_assay_zarr
    - dask_to_zarr
    - ZarrMerge
    - CrToZarr
    - H5adToZarr
    - MtxToZarr
    - NaboH5ToZarr
    - LoomToZarr
    - SparseToZarr
- meld_assay
    - meld_assay
    - make_bed_from_gff

GitHub: https://github.com/parashardhapola/scarf

Documentation: https://scarf.readthedocs.io/en/latest/index.html

Pre-print: https://www.biorxiv.org/content/10.1101/2021.05.02.441899v1

PyPI: https://pypi.org/project/scarf-toolkit/
"""

import warnings
from dask.array import PerformanceWarning
from importlib_metadata import version

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)

try:
    __version__ = version('scarf-toolkit')
except ImportError:
    print("Scarf is not installed", flush=True)

from .datastore import *
from .readers import *
from .writers import *
from .meld_assay import *
from .utils import *
from .downloader import *

