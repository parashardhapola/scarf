---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Basic workflow of Scarf

This workflow is meant to familiarize users with the Scarf API and how data is internally handled in Scarf. Please checkout the quick start guide if you are interested in the minimal steps required to run the analysis.

```python
%load_ext autotime
%config InlineBackend.figure_format = 'retina'

import scarf
scarf.__version__
```


Download the data from 10x's website using the `fetch_dataset` function. This is a convenience function that stores URLs of datasets that can be downloaded. The `save_path` parameter allows the data to be saved to a location of choice.

```python
scarf.fetch_dataset('tenx_10k_pbmc_citeseq', save_path='scarf_data')
```

---
### 1) Format conversion

The first step of the analysis workflow is to convert the file into the Zarr format that is supported by Scarf. We read in the data using `CrH5Reader` (stands for cellranger H5 reader). The reader object allows quick investigation of the file before the format is converted.

```python
reader = scarf.CrH5Reader('scarf_data/tenx_10k_pbmc_citeseq/data.h5', 'rna')
```

We can quickly check the number of cells and features (genes as well as ADT features in this case) present in the file.

```python
reader.nCells, reader.nFeatures
```
We can also quickly check the different kinds of assays present in the file and the number of features from each of them.

```python
reader.assayFeats
```
The **nFeatures** column shows the number of features present in each assay. `CrH5Reader` will automatically pull this information from `features.tsv` and will rename the 'Gene Expression' assay to **RNA**. Here it also found another assay: 'Antibody Capture' and named it to **assay2**. We will rename this to **ADT**.

```python
reader.rename_assays({'assay2': 'ADT'})
reader.assayFeats
```
Next we convert the data to the Zarr format that will later on be used by Scarf. For this we use Scarf's `CrToZarr` class. This class will first quickly ascertain the type of data to be written and then create a Zarr format file for the data to be written into. `CrToZarr` takes two mandatory arguments. The first is the cellranger reader, and the other is the name of the output file.


<div class="alert alert-block alert-info">
NOTE: When we say zarr file, we actually mean zarr directory  because, unlike HDF5, Zarr hierarchy is organized as a directory structure.
</div>

```python
writer = scarf.CrToZarr(reader, zarr_fn='scarf_data/tenx_10k_pbmc_citeseq/data.zarr',
                        chunk_size=(2000, 1000))
```

We can inspect the Zarr hierarchy of the output file:

```python
print (writer.z.tree(expand=True))
```

The three top levels here are: 'RNA', 'ADT' and 'cellData'. The top levels are hence composed of the two assays from above, and the cellData level, which will be explained below. 

Finally, we dump the data into a Zarr object.

```python
writer.dump(batch_size=1000)
```

The next step is to create a Scarf `DataStore` object. This object will be the primary way to interact with the data and all its constituent assays. The first time a Zarr file is loaded, we need to set the default assay. Here we set the 'RNA' assay as the default assay. When a Zarr file is loaded, Scarf checks if some per-cell statistics have been calculated. If not, then **nFeatures** (number of features per cell) and **nCounts** (total sum of feature counts per cell) are calculated. Scarf will also attempt to calculate the percent of mitochondrial and ribosomal content per cell.

```python
ds = scarf.DataStore('scarf_data/tenx_10k_pbmc_citeseq/data.zarr',
                     default_assay='RNA',
                     nthreads=4)
```

Scarf uses Zarr format so that data can be stored in rectangular chunks. The raw data is saved in the `counts` level within each assay level in the Zarr hierarchy. It can easily be accessed as a [Dask](https://dask.org/) array using the `rawData` attribute of the assay. Note that for a standard analysis one would not interact with the raw data directly. Scarf internally optimizes the use of this Dask array to minimize the memory requirement of all operations.

```python
ds.RNA.rawData
```
---
### 2) Cell filtering

We can visualize the per-cell statistics in [violin plots](https://datavizcatalogue.com/methods/violin_plot.html) before we start filtering cells out.

```python
ds.plot_cells_dists()
```


We can filter cells based on these cell attributes by providing upper and lower threshold values.

```python
ds.filter_cells(attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'],
                highs=[15000, 4000, 15],
                lows=[1000, 500, 0])
```

Now we visualize the attributes again after filtering the values. 

*Note: the 'I' value given as the `cell_key` attribute signifies the column of the table that is set to `False` for cells that were filtered out or `True` for cells that are kept.*


```python
ds.plot_cells_dists(cell_key='I', color='coral')
```


Scarf attempts to store most of the data on disk immediately after it is processed. Below we can see that the calculated cell attributes can now be found under the 'cellData' level.

```python
ds.show_zarr_tree()
```

The data stored under the 'cellData' level can easily be accessed using the `cells` attribute of the `DataStore` object.

```python
ds.cells.head()
```
<div class="alert alert-block alert-info">
NOTE: We strongly discourage directly adding or removing the data from this table as Scarf will not be able to synchronize the changes to the disk. Instead use the methods of the <code>cells</code> attribute. Please refer to the <code>insert</code>, <code>fetch</code>, <code>fetch_all</code>, <code>drop</code> and <code>update_key</code> methods.
</div>


---
### 3) Feature selection

Similar to the cell table and the 'cellData' Zarr level, Scarf also saves the feature level data under 'featureData' that is located within each assay. For example, for the RNA assay the feature can be accessed as below:

```python
ds.RNA.feats.head()
```
The feature selection step is performed on the normalized data. The default normalization method for `RNAassay`-type data is library-size normalization, wherein the count values are divided by the sum of total values for a cell. These values are then multiplied by a scalar factor. The default value of this scalar factor is 1000. However, if the total counts in a cell are less than this value, then on multiplication with this scalar factor the values will be 'scaled up' (which is not a desired behaviour). In the filtering step above, we set the `low` threshold for `RNA_nCounts` at 1000, and hence it is safe to use 1000 as a scalar factor. The scalar factor can be set by modifying the `sf` attribute of the assay. Let's print the default value of `sf`

```python
ds.RNA.sf
```

Now the next step is to identify the highly variable genes in the dataset (for the RNA assay). This can be done using the `mark_hvgs` method of the assay. The parameters govern the min/max variance (corrected) and mean expression threshold for calling genes highly variable. 

The variance is corrected by first dividing genes into bins based on their mean expression values. Genes with minimum variance is selected from each bin and a Lowess curve is fitted to the mean-variance trend of these genes. `mark_hvgs` will by default run on the default assay.

A plot is produced, that for each gene shows the corrected variance on the y-axis and the non-zero mean (means from cells where the gene had a non-zero value) on the x-axis. The genes are colored in two gradients which indicate the number of cells where the gene was expressed. The colors are yellow to dark red for HVGs, and blue to green for non-HVGs.

The `mark_hvgs` function has a parameter `cell_key` that dictates which cells to use to identify the HVGs. The default value of this parameter is `I`, which means it will use all the cells that were not filtered out.

```python
ds.mark_hvgs(min_cells=20, top_n=2000, max_mean=1.9, min_mean=-2, max_var=7)
```

As a result of running `mark_hvgs`, the feature table now has an extra column **I__hvgs** which contains a `True` value for genes marked HVGs. The naming rule in Scarf dictates that cells used to identify HVGs are prepended to the column name (with a double underscore delimiter). Since we did not provide any `cell_key` parameter the default value was used, i.e. the filtered cells. This resulted in **I** becoming the prefix.

```python
ds.RNA.feats.head()
```
### 4) Graph creation

Creating a neighbourhood graph of cells is the most critical step in any Scarf workflow. This step internally involves multiple substeps: 

- data normalization for selected features
- linear dimensionality reduction using PCA
- creating an approximate nearest neighbour graph index (using the HNSWlib library)
- querying cells against this index to identify nearest neighbours for each cell
- edge weight computation using the `compute_membership_strengths` function from the UMAP package
- fitting MiniBatch Kmeans (The kmeans centers are used later, for UMAP initialization)

`make_graph` method is responsible for graph construction. It method takes a mandatory parameter: `feat_key`. This should be a column in the feature metadata table that indicates which genes to use to create the graph. Since, we have already identified the `hvgs` in the step above, we use those genes. Note that we do not need to write *I__hvgs* but just *hvgs* as the value of the parameter. We also supply values for two very important parameters here: `k` (number of nearest neighbours to be queried for each cell) and `dims` (number of PCA dimensions to use for graph construction). `n_centroids` parameter controls number of clusters to create for the data using Kmeans algorithm. We perform a more accurate clustering of data in the later steps.

```python
ds.make_graph(feat_key='hvgs', k=21, dims=30, n_centroids=100)
```

All the results of `make_graph` step are saved under a name on the form '*normed\_\_{cell key}\_\_{feature key}*' (placeholders used in brackets here). In this case, since we did not provide a cell key it takes default value of `I`, which means all the non-filtered out cells. The feature key (`feat_key`) was set to `hvgs`. The Zarr directory is organized such that all the intermediate data is also saved. The intermediate data is organized in a hierarchy which triggers recomputation when upstream changes are detected. The parameter values are also saved in hierarchy level names. For example, 'reduction_pca_31_I' means that PCA linear dimension reduction with 31 PC axes was used and the PCA was fit across all the cells that have `True` value in column **I**.

```python
ds.show_zarr_tree()
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
### 5) Low dimensional embedding and clustering

Next we run UMAP on the graph calculated above. Here we will not provide which assay, cell key or feature key to be used, because we want the UMAP to run on the default assay with all the non-filtered out cells and with the feature key used to calculate the latest graph. We can provide the parameter values for the UMAP algorithm here.

```python
ds.run_umap(fit_n_epochs=200, spread=5, min_dist=2, parallel=True)
```

The UMAP results are saved in the cell metadata table as seen below in columns: **RNA_UMAP1** and **RNA_UMAP2**

```python
ds.cells.head()
```
`plot_layout` is a versatile method to create a [scatter plot](https://datavizcatalogue.com/methods/scatterplot.html) using Scarf. Here we can plot the UMAP coordinates of all the non-filtered out cells.

```python
ds.plot_layout(layout_key='RNA_UMAP')
```


`plot_layout` can be used to easily visualize data from any column of the cell metadata table. Next, we visualize the number of genes expressed in each cell.

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_nCounts', cmap='coolwarm')
```


Identifying clusters of cells is one of the central tenets of single cell approaches. Scarf includes two graph clustering methods and any (or even both) can be used on the dataset. The methods start with the same graph as the UMAP algorithm above to minimize the disparity between the UMAP and clustering results. The two clustering methods are:

- **Paris**: This is the default clustering algorithm.
- **Leiden**: Leiden is a widely used graph clustering algorithm in single-cell genomics.

Paris is the default algorithm in Scarf due to its ability to highlight cluster relationships. [Paris](https://github.com/tbonald/paris) is a hierarchical graph clustering algorithm that is based on node pair sampling. Paris creates a dendrogram of cells which can then be cut to obtain desired number of clusters. The advantage of using Paris, especially in the larger datasets, is that once the dendrogram has been created one can change the desired number of clusters with minimal computation overhead.

```python
ds.run_clustering(n_clusters=20)
```

The results of the clustering algorithm are saved in the cell metadata table. In this case, they have been saved under the column name **RNA_cluster**.

```python
ds.cells.head()
```
We can visualize the results using the `plot_layout` method again:

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster')
```


Leiden clustering provides very accurate results

```python
ds.run_leiden_clustering(resolution=2)
```

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_leiden_cluster')
```

There has been a lot of discussion over the choice of non-linear dimensionality reduction for single-cell data. tSNE was initially considered an excellent solution, but has gradually lost out to UMAP because the magnitude of relations between the clusters cannot easily be discerned in a tSNE plot. Scarf contains an implementation of tSNE that runs directly on the graph structure of cells. So, essentially the same data that was used to create the UMAP and clustering is used. Additionally, to minimize the differences between the UMAP and tSNE, we use the same initial coordinates of tSNE that were used for UMAP, i.e. the first two (in case of 2D) PC axes of PCA of kmeans cluster centers. We have found that tSNE is actually a complementary technique to UMAP. While UMAP focuses on highlighting the cluster relationship, tSNE highlights the heterogeneity of the dataset. As we show in the sample integration vignette, using tSNE can be better at visually accessing the extent of heterogeneity compared to UMAP. The biggest reason, however, to run Scarf's implementation of graph tSNE could be the runtime. It can be an order of magnitude faster than UMAP on large datasets.

```python
ds.run_tsne(alpha=10, box_h=1, early_iter=250, max_iter=500, parallel=True)
```

```python
# Here we run plot_layout under exception catching because if you are not on Linux then the `run_tnse` would have failed.
try:
    ds.plot_layout(layout_key='RNA_tSNE', color_by='RNA_cluster')
except KeyError:
    print ("'RNA_tSNE1' not found in MetaData")
```


We saw an over 2x speedup compared to UMAP using tSNE with the given parameters. It is harder to compare the distances between the clusters here but easier to visually gauge the size of clusters and intra-cluster heterogeneity.

Discerning similarity between clusters can be difficult from visual inspection alone, especially for tSNE plots. `plot_cluster_tree` function plots the relationship between clusters as a binary tree. This tree is simply a condensation of the dendrogram obtained using Paris clustering.

```python
ds.plot_cluster_tree(cluster_key='RNA_cluster', width=1)
```

The tree is free form (i.e the position of clusters doesn't convey any meaning) but allows inspection of cluster similarity based on branching pattern. The sizes of clusters indicate the number of cells present in each cluster. The tree starts from the root node (black dot with no incoming edges). 


---
### 6) Marker gene identification

Now we can identify the genes that are differentially expressed between the clusters using the `run_marker_search` method. The method to identify the differentially expressed genes in Scarf is optimized to obtain quick results. We have not compared the sensitivity of our method compared to other differential expression detecting methods and expect specialized methods to be more sensitive and accurate to varying degrees. Our method is designed to quickly obtain key marker genes for populations from a large dataset. For each gene individually, following steps are carried out:

- Expression values are converted to ranks (dense format) across cells.
- A mean of ranks is calculated for each group of cells
- The mean value for each group is divided by the sum of mean values to obtain the 'specificity score'
- The gene is saved as a marker gene if it's specificity score is higher than a given threshold.

This method does not perform any statistical test of significance and uses 'specificity score' as a measure of importance of each gene for a cluster.

```python
ds.run_marker_search(group_key='RNA_cluster', threshold=0.25)
```

Using the `plot_marker_heatmap` method, we can also plot a heatmap with the top marker genes from each cluster. The method will calculate the mean expression value for each gene from each cluster.


```python
ds.plot_marker_heatmap(group_key='RNA_cluster', topn=3)
```

We can directly visualize the expression values for a gene of interest. It is usually a good idea to visually confirm the the gene expression pattern across the cells atleast this way.

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='CD14')
```


---
### 7) Working with non-default assays: surface protein expression (ADT) data

Here, we show how to work with non-default assays. We have surface protein data present, in the ADT assay. Let's check out the feature table for this assay:


```python
ds.ADT.feats.head()
```
We can manually filter out the control antibodies by updating **I** to be False for those features.

```python
ds.ADT.feats.update_key(~ds.ADT.feats.to_pandas_dataframe(['names'])['names'].str.contains('control').values, 'I')
ds.ADT.feats.head(n=ds.ADT.feats.N)
```
Assays named ADT are automatically created as objects of the `ADTassay` class, which uses CLR (centred log ratio) normalization as the default normalization method.

```python
print (ds.ADT)
print (ds.ADT.normMethod.__name__)
```

Now we create a new graph of cells using just the `ADT` data.

```python
ds.make_graph(from_assay='ADT', feat_key='I', k=11, dims=11, n_centroids=100, log_transform=False)
```

Run UMAP on the `ADT` graph:


```python
ds.run_umap(from_assay='ADT', fit_n_epochs=200, parallel=True)
```

One can check concordance between the RNA and ADT assays by visualizing the RNA cluster information on ADT data:

```python
ds.plot_layout(layout_key='ADT_UMAP', color_by='RNA_cluster')
```


We can also run the clustering directly on the `ADT` graph and visualize it on the UMAP plot:

```python
ds.run_leiden_clustering(from_assay='ADT', resolution=1.4)
ds.plot_layout(layout_key='ADT_UMAP', color_by='ADT_leiden_cluster')
```

Or another way, visualize the ADT clusters on the RNA UMAP:

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='ADT_leiden_cluster')
```


Individual ADT expression can be visualized in both UMAPs easily.

```python
ds.plot_layout(from_assay='ADT', layout_key='ADT_UMAP', color_by='CD3_TotalSeqB')
ds.plot_layout(from_assay='ADT', layout_key='RNA_UMAP', color_by='CD3_TotalSeqB')
```


---
That is all for this vignette.
