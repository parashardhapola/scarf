---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Data storage in Scarf

In this notebook, we provide a more detailed exploration of how the data is organized in Scarf. This can be useful for users who want to customize certain aspects of Scarf or want to extend its functionality. 

```python
%load_ext autotime
%config InlineBackend.figure_format = 'retina'

import scarf
scarf.__version__
```

We download the CITE-Seq dataset that we [analyzed earlier](https://scarf.readthedocs.io/en/latest/vignettes/multiple_modalities.html)

```python
scarf.fetch_dataset('tenx_8K_pbmc_citeseq', save_path='scarf_datasets', as_zarr=True)
```

```python
ds = scarf.DataStore('scarf_datasets/tenx_8K_pbmc_citeseq/data.zarr', nthreads=4)
```

---
### 1) Zarr trees

Scarf uses [Zarr format](https://zarr.readthedocs.io/en/stable/) to store raw counts as dense, chunked matrices. The Zarr file is in fact a directory that organizes the data in form of a tree. The count matrices, cell and features attributes, and all the data is stored in this Zarr hierarchy. Some of the key benefits of using Zarr over other formats like HDF5 are:
- Parallel read and write access
- Availability of compression algorithms like [LZ4](https://github.com/lz4/lz4) that provides very high compression and decompression speeds.
- Automatic storage of intermediate and processed data

In Scarf, the data is always synchronized between hard disk and RAM and as such there is no need to save data manually or synchronize the

We can inspect how the data is organized within the Zarr file using `show_zarr_tree` method of the `DataStore`. 

```python
ds.show_zarr_tree(depth=1)
```

By setting `depth=1` we get a look of the top-levels in the Zarr hierarchy. `cellData` will always be present in the top level, the rest will be the the assays present in the dataset. In this case, we have two assays present, `ADT` and `RNA`. T

The three top levels here are: 'RNA', 'ADT' and 'cellData'. The top levels are hence composed of the two assays from above, and the cellData level, which will be explained below. Scarf attempts to store most of the data on disk immediately after it is processed. Below we can see that the calculated cell attributes can now be found under the 'cellData' level.

```python

```

```python
ds.show_zarr_tree(start='cellData')
```

```python
ds.show_zarr_tree(start='RNA', depth=1)
```

```python
ds.show_zarr_tree(start='RNA/featureData', depth=1)
```

---
### 2) Dask arrays and data normalization in Scarf



Scarf uses Zarr format so that data can be stored in rectangular chunks. The raw data is saved in the `counts` level within each assay level in the Zarr hierarchy. It can easily be accessed as a [Dask](https://dask.org/) array using the `rawData` attribute of the assay. Note that for a standard analysis one would not interact with the raw data directly. Scarf internally optimizes the use of this Dask array to minimize the memory requirement of all operations.

```python
ds.RNA.rawData
```

```python
ds.RNA.normed()
```

---
### 3) Data caching during graph creation


All the results of `make_graph` step are saved under a name on the form '*normed\_\_{cell key}\_\_{feature key}*' (placeholders used in brackets here). In this case, since we did not provide a cell key it takes default value of `I`, which means all the non-filtered out cells. The feature key (`feat_key`) was set to `hvgs`. The Zarr directory is organized such that all the intermediate data is also saved. The intermediate data is organized in a hierarchy which triggers recomputation when upstream changes are detected. The parameter values are also saved in hierarchy level names. For example, 'reduction_pca_31_I' means that PCA linear dimension reduction with 31 PC axes was used and the PCA was fit across all the cells that have `True` value in column **I**.

```python
ds.show_zarr_tree(start='RNA/normed__I__hvgs')
```

The graph calculated by `make_graph` can be easily loaded using the `load_graph` method, like below. The graph is loaded as a sparse matrix of the cells that were used for creating a graph.

Next, we show how the graph can be accessed if required. However, as stated above, normally Scarf handles the graph loading internally where required. 

Because Scarf saves all the intermediate data, it might be the case that a lot of graphs are stored in the Zarr hierachy. `load_graph` will load only the latest graph that was computed (for the given assay, cell key and feat key). 

```python
ds.load_graph(from_assay='RNA', cell_key='I', feat_key='hvgs', symmetric=False, upper_only=False)
```

The location of the latest graph can be accessed by `_get_latest_graph_loc` method. The latest graph location is set using the parameters used in the latest call to `make_graph`. If one needs to set the latest graph to one that was previously calculated then one needs to call `make_graph` with the corresponding parameters.

```python
ds._get_latest_graph_loc(from_assay='RNA', cell_key='I', feat_key='hvgs')
```

---
### 4) Working with attribute tables

The cell and feature level attributes can be accessed through `DataStore`, for example, using `ds.cells` and `ds.RNA.feats` respectively. In this section we dive deeper into these attribute tables and try to perform [CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete) operations on them.


`head` provides a quick look at the attribute tables. 

```python
ds.cells.head()
```

```python
ds.RNA.feats.head()
```

```python
ds.cells.columns
```

```python
ds.cells.to_pandas_dataframe(['ids', 'RNA_UMAP1', 'RNA_UMAP2']).set_index('ids')
```

```python
ds.cells.fetch('ids')
```

```python
len(ds.cells.fetch_all('ids'))
```

```python
len(ds.cells.fetch('ids', key='I'))
```

```python
clusters = ds.cells.fetch_all('RNA_leiden_cluster')
clusters
```

```python
len(clusters)
```

```python
set(clusters)
```

```python
clusters == 1
```

```python
ds.cells.insert(column_name='is_clust1', values=clusters==1)
```

```python
ds.cells.head()
```

```python
clust1_ids = ds.cells.fetch('ids', key='is_clust1')
clust1_ids
```

```python
len(clust1_ids)
```

```python
ds.cells.update_key(values=clusters==1, key='I')
ds.cells
```

```python
ds.cells.insert(column_name='backup_I',
                values=ds.cells.fetch_all('I'))
```

```python
ds.cells.reset_key(key='I')
ds.cells
```

```python
ds.cells.update_key(values=ds.cells.fetch_all('backup_I'), key='I')
ds.cells
```

---
### 5) Working with marker features

```python
ds.show_zarr_tree(start='RNA/markers')
```

```python
ds.get_markers(group_key='RNA_cluster', group_id=7)
```

```python

```

```python
ds.export_markers_to_csv(group_key='RNA_cluster', csv_filename='test.csv')
```

```python

```

```python

```

```python
import pandas as pd
```

```python
pd.read_csv('test.csv').fillna('')
```

```python

```

```python

```
