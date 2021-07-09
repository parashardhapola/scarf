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

## Understanding how data is organized in Scarf

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

By setting `depth=1` we get a look of the top-levels in the Zarr hierarchy: 'RNA', 'ADT' and 'cellData'. The top levels here are hence composed of two assays 
and the `cellData` level, which will be explained below. Scarf attempts to store most of the data on disk immediately after it is processed. Since, that data that we have loaded is already preprocessed, below we can see that the calculated cell attributes can now be found under the `cellData` level. The cell-wise statistics that were calculated using 'RNA' assay have 'RNA' preppended to the name of the column. Similarly the columns stating with 'ADT' were created using 'ADT' assay data.

```python
ds.show_zarr_tree(start='cellData')
```

**The `I` column** : This column is used to keep track of valid and invalid cells. The values in this column are boolean indicating which cells where filtered out (hence False) during filtering process. One of can think of column `I` as the default subset of cells that will be used for analysis if any other subset is not specifially chosen explicitly by the users. Most methods of `DataStore` object accept a parameter called `cell_key` and the default value for this paramter is `I`, which means that only cells with True in `I` column will be used.


If we inspect one of the assay, we will see the following zarr groups. The raw data matrix is stored under `counts` group (more on it in the next section). `featureData` is the feature level equivalent of `cellData`. `featureData` is explained in further detail in the next section. The `markers` level will be explained further in the last section. `summary_stats_I` contains statistics about features. These statistics are generally created during feature selection step. 'normed__I__hvgs' is explained in detail in the fourth section of this documentation.

```python
ds.show_zarr_tree(start='RNA', depth=1)
```

The feature wise attributes for each assay are stored under the assays `featureData` level. The output below shows some of the feature level statistics.

```python
ds.show_zarr_tree(start='RNA/featureData', depth=1)
```

---
### 2) Cell and feature attributes

The cell and feature level attributes can be accessed through `DataStore`, for example, using `ds.cells` and `ds.RNA.feats` respectively (both are objects of `Metadata` class). In this section we dive deeper into these attribute tables and try to perform [CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete) operations on them. 


`head` provides a quick look at the attribute tables. 

```python
ds.cells.head()
```

The feature attribute table from any assay can similarly be quickly inspected

```python
ds.RNA.feats.head()
```

Even though the above 'head' command may make you think that `ds.cells` and `ds.RNA.feats` are Pandas dataframe, they in fact are not. If you wish to obtain a full table as Pandas dataframe, then you can export the columns of your choice as shown below.

```python
ds.cells.to_pandas_dataframe(['ids', 'RNA_UMAP1', 'RNA_UMAP2', 'RNA_cluster']).set_index('ids')
```

If you wish to export all the columns as a dataframe then simply provide `ds.cells.columns` or `ds.RNA.feats.columns` as an argument rather than a list of columns. If you are interested in one of the columns then it can fetched using either `fetch` or `fetch_all` command.

`fetch` will provide values for a subset of cells (by default, only for those that have True value in column `I`, but any other boolean column can be given).

```python
clusters = ds.cells.fetch('RNA_cluster')
clusters.shape
```

`fetch_all` will return values for all the cells.

```python
clusters_all = ds.cells.fetch_all('RNA_cluster')
clusters_all.shape
```

If you wish to add a new column then `insert` command can be used for either cell or feature attributes. `insert` method will take care of inserting the values in the right rows even when a subset of values are provided. The default value for `key` parameter is `I` in `insert`method, so it will add values in same order as cells that have True value for `I`

```python
is_clust_1 = clusters == 1   # is_clust_1 has just 7549 elements (same as the number of cells with value True in `I`)
ds.cells.insert(column_name='is_clust1', values=is_clust_1)
```

If we try to add values for a column that already exists then Scarf will throw an error. For example, if we simply rerun the command above, we should get an error

```python
ds.cells.insert(column_name='is_clust1', values=is_clust_1)
```

To override this behaviour, you can use 'overwrite' parameter and set it to True

```python
ds.cells.insert(column_name='is_clust1', values=is_clust_1, overwrite=True)
```

Please checkout the API of the `Metadata` class to get information on how to perform other operations like delete and update columns.


---
### 3) Count matrices and data normalization



Scarf uses Zarr format so that data can be stored in rectangular chunks. The raw data is saved in the `counts` level within each assay level in the Zarr hierarchy. It can easily be accessed as a [Dask](https://dask.org/) array using the `rawData` attribute of the assay. Note that for a standard analysis one would not interact with the raw data directly. Scarf internally optimizes the use of this Dask array to minimize the memory requirement of all operations.

```python
ds.RNA.rawData
```

The normalized data can be accessed through the `normed` method of the assay. In Scarf a user doesn't need to perform normalization step manually, Scarf only stores the raw data and generates normalized data whenever needed. This means that Scarf may need to perform normalization several times. However, in practise we have noted that time spent in normalizing the data is only a small fraction of routine workflows. 

```python
ds.RNA.normed()
```

Users can override how the normalization is performed in Scarf. Normalization is performed using the `normMethod` attribute which references the function responsible for performing the normalization.

```python
ds.RNA.normMethod
```

Let's checkout the source of function that is referenced by ds.RNA.normMethod

```python
import inspect 

print (inspect.getsource(ds.RNA.normMethod))
```

Following is an example of how one can override the method of normalization

```python
def my_cool_normalization_method(assay, counts):
    import numpy as np

    lib_size = counts.sum(axis=1).reshape(-1, 1) # Calculate total counts for each cell
    return np.log2(counts/lib_size)  # Library size normalization followed by log2 transformation

ds.RNA.normMethod = my_cool_normalization_method
```

Now whenever Scarf internally requires normalized values, this function will be used. Scarf provides a dummy normalization function (`scarf.assay.norm_dummy`) that does not perform normalization. This function can be useful if you have pre-normalized data and need to disable default normalization.

```python
ds.RNA.normMethod = scarf.assay.norm_dummy
```

Please note: If you are using a custom function or disabling normalization, then everytime you load a DataStore object you will need to reassign `normMethod` to the function of your choice.


---
### 4) Data caching during graph creation


All the results of `make_graph` step are saved under a name on the form '*normed\_\_{cell key}\_\_{feature key}*' (placeholders used in brackets here). In this case, since we did not provide a cell key it takes default value of `I`, which means all the cells that were not filtered out. The feature key (`feat_key`) was set to `hvgs`. The Zarr directory is organized such that all the intermediate data is also saved. The intermediate data is organized in a hierarchy which triggers recomputation when upstream changes are detected. The parameter values are also saved in hierarchy level names. For example, 'reduction_pca_31_I' means that PCA linear dimension reduction with 31 PC axes was used and the PCA was fit across all the cells that have `True` value in column **I**.

```python
ds.show_zarr_tree(start='RNA/normed__I__hvgs')
```

The graph calculated by `make_graph` can be easily loaded using the `load_graph` method, like below. The graph is loaded as a sparse matrix of the cells that were used for creating a graph.

Next, we show how the graph can be accessed if required. However, as stated above, normally Scarf handles the graph loading internally where required. 

Because Scarf saves all the intermediate data, it might be the case that a lot of graphs are stored in the Zarr hierarchy. `load_graph` will load only the latest graph that was computed (for the given assay, cell key and feat key). 

```python
ds.load_graph(from_assay='RNA', cell_key='I', feat_key='hvgs', symmetric=False, upper_only=False)
```

The sparse matrix above will load the If you would like to load a graph generated using some other parameter, then simply run `make_graph` again using those parameters. `make_graph` will not recalculate the graph if it already exists and will simply set it as the latest graph. This graph can then be loaded using `load_graph`


---
### 5) Fetching marker features


Marker features (ex. marker genes) for a selection of cells can be calculated using `run_marker_search`. The markers for each group are stored under 'marker' group under the assay's group. Within the '{assay}/markers' level there are sublevels (only 1 here) created based on which cell set and cell groups were used: {cell_key}__{group_key}. Stored below this Zarr level are the individual group labels that contain the maker feature ids and their corresponding scores.

```python
ds.show_zarr_tree(start='RNA/markers', depth=2)
```

Marker list for any group (e.x. cell cluster) can be fetched like below

```python
ds.get_markers(group_key='RNA_cluster', group_id=7)
```

One can also export the names of markers for all the groups to a CSV file like below:

```python
ds.export_markers_to_csv(group_key='RNA_cluster', csv_filename='test.csv')
```

---
That is all for this vignette.
