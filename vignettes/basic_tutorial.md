## Scarf workflow using 10K PBMC CITE-Seq data

This workflow is meant to familiarize users with Scarf API and how data is internally handled in Scarf. Please checkout the quick start guide if you are interested in the minimal steps required to run the analysis.

```python
%load_ext autotime
%config InlineBackend.figure_format = 'retina'

import scarf
```


Download data from 10x's website.

```python
!mkdir -p data
!wget http://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5
!mv pbmc_10k_protein_v3_filtered_feature_bc_matrix.h5 ./data/pbmc_10k_rna_prot.h5
```

### 1) Format conversion

The first step of the analysis workflow is to convert the file into Zarr format that is support by Scarf. So we read in the data using CrH5Reader (stands for cellranger H5 reader). The reader object allows quick investigation of the file before the format is converted.

```python
reader = scarf.CrH5Reader(f'./data/pbmc_10k_rna_prot.h5', 'rna')
```

We can quickly check the number of cells and features (genes as well ADT features in this case) present in the file.

```python
reader.nCells, reader.nFeatures
```
We can also quickly check the different kinds of assays present in the file and the number of features from each of them.

```python
reader.assayFeats
```
nFeature column shows the number of features present in each assay. CrH5Reader will automatically pull this information from `features.tsv` and will rename the 'Gene Expression' assay to 'RNA'. Here it also found another assay: 'Antibody Capture' and named it to 'assay2'. We will rename this to 'ADT'.

```python
reader.rename_assays({'assay2': 'ADT'})
reader.assayFeats
```
Convert data to Zarr format that will be later on used by Scarf. For this we use Scarf's CrToZarr class. This class will first quickly ascertain the data to be written and will create a zarr format file for data to be written into. CrToZarr takes two mandatory arguments. The first is the cellranger reader an other is the name of the output file.


<div class="alert alert-block alert-info">
NOTE: When we say zarr file, we actually mean zarr directory  because, unlike HDF5, Zarr hierarchy is organized as a directory structure.
</div>

```python
writer = scarf.CrToZarr(reader, zarr_fn=f'./data/pbmc_10k_rna_prot.zarr', chunk_size=(1000, 1000))
```

We can inspect the Zarr hierarchy of the output file.

```python
print (writer.z.tree(expand=True))
```

The three top levels here are: 'RNA', 'ADT' and 'cellData'. The top levels are hence composed of two assays and the cellData level which is explained below. And finally, we dump the data into a zarr object.


```python
writer.dump(batch_size=1000)
```

The next step is to create a Scarf DataStore object. This object will be the primary way to interact with the data and all its constituent assays. The first time a Zarr file is loaded, we need to set the default assay. Here we set the 'RNA' assay as the default assay. When a Zarr file is loaded, scarf checks if some per cell statistics have been calculated. If not then nFeatures (no. of features per cell) and nCounts (total sum of feature counts per cell) are calculated. Scarf will also attempt to calculate % mitochondrial and ribosomal content per cell.

```python
ds = scarf.DataStore('./data/pbmc_10k_rna_prot.zarr', default_assay='RNA')
```

Scarf uses Zarr format so that data can be stored in rectangular chunks. The raw data is saved in the `counts` level within each assay level in the Zarr hierarchy. It can easily be accessed as a Dask array using `rawData` attribute of the assay. For a standard analysis one would not to interact with the raw data directly. Scarf internally optimizes the use of this Dask array to minimize the memory requirement of all operations.

```python
ds.RNA.rawData
```
### 2) Cell filtering

We can visualize the per cell statistics in violin plots before we start filtering cells out

```python
ds.plot_cells_dists(cols=['percent*'])
```


We can filter cells based on these cell attributes by providing upper and lower threshold values.

```python
ds.filter_cells(attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'], highs=[20000, 5000, 25], lows=[1000, 500, 0])
```

Now we visualize the attributes again after filtering the values.


```python
ds.plot_cells_dists(cols=['percent*'])
```


Scarf attempts to store most of the data into the disk immediately after it is calculated. Below we can see that the calculated cell attributes can now be found under the 'cellData' level.

```python
print (ds.z.tree(expand=True))
```

The data stored under the cellData level can easily be accessed using the `cells.table` attribute of the DataStore object.


```python
ds.cells.table
```
### 3) Feature selection

The 'I' column is set to False for cells that were filtered out.

Similar to the cell table and 'cellData' Zarr level. Scarf also saves the feature level data under 'featureData' that is located within each assay. For example, for RNA assay the feature can be accessed as below:


<div class="alert alert-block alert-info">
NOTE: We strongly discourage directly adding or removing the data from this table as Scarf will not be able to synchronize the changes to the disk. Instead use methods of the cell attribute. Please refer to `add`, `fetch`, `remove` and `update` methods.
</div>

```python
ds.RNA.feats.table
```
Now the next step is to identify the highly variable genes in the dataset (RNA assay). This can be done using `mark_hvgs` method of the assay. The parameters govern the min/max variance (corrected) and mean expression threshold for calling genes highly variable. 

The variance is corrected by first dividing genes into bins based on their mean expression values. Gene with minimum variance is selected from each bin and a Lowess curve to the mean-variance trend of these genes. `mark_hvgs` will by default run on the default assay

A plot is produced showing, for each gene, the corrected variance on the y-axis and the non-zero mean (mean from cells where the gene had a non-zero value) on the x-axis. The genes are colored in two gradients which indicate the number of cells were the gene expressed. Yellow to dark red for HVGs and blue to green for non-HVGs.

The `mark_hvgs` function has a parameter `cell_key` that dictates which cells to use to identify the HVGs. The default value of this parameter is `I` which means it will use the filtered cells.

```python
ds.RNA.mark_hvgs(min_cells=20, top_n=2000, n_bins=50)
```

As a result of running `mark_hvgs`, the feature table now has an extra column `I__hvgs` which contains a True value for genes marked HVGs. The naming rule in Scarf dictates that cells used to identify HVGs are prepended to column name (with double underscore delimiter). Since we did not provide any `cell_key` parameter the default value was used i.e. the filtered cells so 'I' was prepended.

```python
ds.RNA.feats.table
```
### 4) Graph creation

Creating a neighbourhood graph of cells is the most critical step in any Scarf workflow. This step internally involves multiple substeps: 
- data normalization for selected features
- linear dimension reduction using PCA
- creating an approximate nearest neighbour graph index (using HNSWlib library)
- querying cells against this index to identify nearest neighbours for each cell
- Edge weight computation using compute_membership_strengths function from UMAP package
- Fitting MiniBatch Kmeans (The kmeans centers are used later for UMAP initialization)

```python
ds.make_graph(feat_key='hvgs', k=21, dims=30, n_centroids=100)
```

All the results of `make_graph` are saved under 'normed\__{cell key}__{feature key}'. in this case since we did not provide a cell key, it takes default value of `I` which means all the filtered cells and feature key (`feat_key`) was set to `hvgs`. The directory is organized such that all the intermediate data is also saved. The intermediate data is organized in a hierarchy which triggers recomputation when upstream changes are detected. The parameter values are also saved in hierarchy level names. For example, 'reduction_pca_31_I' means that PCA linear dimension reduction with 31 PC axes was used and the PCA was fit across all the cells that have True value in column 'I'. 

```python
print (ds.RNA.z.tree(expand=True))
```

The graph calculated by `make_graph` can be easily loaded using `load_graph` method like below. The graph is loaded as a sparse matrix of cells that were used for creating a graph. Following we show how the graph can be accessed if required, but normally Scarf handles the graph loading internally where required. 

Because Scarf saves all the intermediate data, it might be the case that a lot of graphs are stored in Zarr hierachy. `load_graph` will load only the latest graph that was computed (for the given assay, cell key and feat key). 

```python
ds.load_graph(from_assay='RNA', cell_key='I', feat_key='hvgs', graph_format='csr', min_edge_weight=-1, symmetric=False, upper_only=False)
```
The location of the latest graph can be accessed by `_get_latest_graph_loc` method. The latest graph location is set using the parameters used in the latest call to `make_graph`. If one needs to set the latest graph to one that was previously calculated then one needs to call `make_graph` with the corresponding parameters.

```python
ds._get_latest_graph_loc(from_assay='RNA', cell_key='I', feat_key='hvgs')
```
### 5) Low dimensional embedding and clustering

Next we run UMAP on the graph calculated above. Here we will not provide which assay, cell key or feature key to be used because we want the UMAP to run on the default assay with all the filtered cells and with the feature key used to calculate the latest graph. We can also provide the parameters values for the UMAP algorithm here

```python
ds.run_umap(fit_n_epochs=500, min_dist=0.5)
```

The UMAP results are saved in the cell metadata table as seen below in columns: 'RNA_UMAP1' and 'RNA_UMAP2'

```python
ds.cells.table
```
`plot_layout` is a versatile method to create a scatter plot using Scarf. Here we can plot the UMAP coordinates of all the filtered cells.

```python
ds.plot_layout(layout_key='RNA_UMAP')
```


`plot_layout` can be used to easily visualize data from any column of the cell metadata table. Following we visualize the number of genes expressed in each cell

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_nCounts', colormap='coolwarm')
```


Identifying clusters of cells is one of the central tenets of single cell approaches. Scarf includes two graph clustering methods and any (or even both) can be used on the dataset. The methods start with the same graph as the UMAP algorithm above to minimize the disparity between the UMAP and clustering results. The two clustering methods are:
- **Paris**: This is the default clustering algorithm that scales very well to millions of cells (yielding results in less than 10 mins for a million cells)
- **Leiden**: Leiden is a widely used graph clustering algorithm in single-cell genomics and provides very good results but is slower to run larger datasets.

In this vignette we demonstrate clustering using Paris, the default algorithm. Paris is hierarchical graph clustering algorithm that is based on node pair sampling. Paris creates a dendrogram of cells which can then be cut to obtain desired number of clusters. The advantage of using Paris, especially in the larger datasets, is that once the dendrogram has been created one can change the desired number of clusters with minimal computation overhead.

```python
ds.run_clustering(n_clusters=25)
```

The results of clustering algorithm are saved in the cell metadata table. In this case, they have been saved under 'RNA_cluster' column.

```python
ds.cells.table
```
We can visualize the results using the `plot_layout` method again

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster')
```


There has been a lot of discussion over the choice of non-linear dimension reduction for single-cell data. tSNE was initially considered an excellent solution but has gradually lost out to UMAP because the magnitude of relation between the clusters cannot easily be discerned in a tSNE plot. Scarf contains an implementation of tSNE that runs directly on the graph structure of cells. So essentially the same data that was used to create the UMAP and clustering is used. Additionally, to minimize the differences between the UMAP and tSNE, we use the same initial coordinates of tSNE as were used for UMAP, i.e. the first two (in case of 2D) PC axis of PCA of kmeans cluster centers. We have found that tSNE is actually a complementary technique to UMAP. While UMAP focuses on highlighting the cluster relationship, tSNE highlights the heterogeneity of the dataset. As we show in the 1M cell vignette, using tSNE can be better at visually accessing the extent of heterogeneity than UMAP. The biggest reason, however to run Scarf's implementation of graph tSNE could be the runtime which can be an order of magnitude faster than UMAP on large datasets.

```python
ds.run_tsne(sgtsne_loc='../bin/sgtsne', alpha=20, box_h=1)
```

```python
ds.plot_layout(layout_key='RNA_tSNE', color_by='RNA_cluster')
```


We saw a 7x speedup compared to UMAP using tSNE with the given parameters. It is harder to compare the distances between the clusters here but easier to visually gauge the size of clusters and intra-cluster heterogeneity.

Discerning similarity between clusters can be difficult from visual inspection alone, especially for tSNE plots. `plot_cluster_tree` function plots the relationship between clusters as a binary tree. This tree is simply a condensation of the dendrogram obtained using Paris clustering.

```python
ds.plot_cluster_tree(cluster_key='RNA_cluster', width=1)
```

The tree is free form (i.e the position of clusters doesn't convey any meaning) but allows inspection of cluster similarity based on branching pattern. The sizes of clusters indicate the number of cells present in each cluster. The tree starts from the root node (black dot with no incoming edges). As an example, one can observe by looking at the branching pattern that cluster 1 is closer to cluster 8 than it is to cluster 4 since 1 and 8 share parent node whereas 4 is part of another branch. Cluster 4 is in turn closest to cluster 17. 

### 6) Marker gene identification

Now we can identify the genes that are differentially expressed between the clusters using `run_marker_search` method. The method to identify the differentially expressed genes in Scarf is optimized to obtain quick results. We have not compared the sensitivity of our method compared to other differential expression detecting methods and expect specialized methods to be more sensitive and accurate to varying degrees. Our method is designed to quickly obtain key marker genes for populations from a large dataset. For each gene individually following steps are carried out:
- Expression values are converted to ranks (dense format) across cells.
- A mean of ranks is calculated for each group of cells
- The mean value for each group is divided by the sum of mean values to obtain the 'specificity score'
- The gene is saved as a marker gene if it's specificity score is higher than a given threshold.

This method does not perform any statistical test of significance and uses 'specificity score' as a measure of importance of each gene for a cluster.

```python
ds.run_marker_search(group_key='RNA_cluster', threshold=0.2)
```

Using `plot_marker_heatmap` we can also plot a heatmap with top marker genes from each cluster. The method will calculate the mean expression value for each gene from each cluster.


```python
ds.plot_marker_heatmap(group_key='RNA_cluster', topn=5)
```

We can directly visualize the expression values for a gene of interest. It is usually a good idea to visually confirm the the gene expression pattern across the cells atleast this way.

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='PPBP')
```


### 7) Down sampling

UMAP, clustering and marker identification together allow a good understanding of cellular diversity. However, one can still choose from a plethora of other analysis on the data. For example, identification of cell differentiation trajectories. One of the major challenges to run these analysis could be the size of the data. Scarf performs a topology conserving downsampling of the data based on the cell neighbourhood graph. This downsampling aims to maximize the heterogeneity while sampling the cells from the the data. Before the actual downsampling step, two key steps must be performed.

The first step is the micro-clustering of the cells. Micro-clustering is performed using the dendrogram generated by the Paris algorithm. Rather than using a fixed distance value to cut the dendrogram to obtain cluster, a balanced cut is performed such the size of obtained clusters are bounded within the given limits. Below we perform balanced micro clustering and visualize the results

```python
ds.run_clustering(balanced_cut=True, min_size=20, max_size=100, label='b_cluster')
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_b_cluster', legend_onside=False, legend_ondata=False)
ds.plot_cluster_tree(cluster_key='RNA_b_cluster', width=1, do_label=False)
```

It is good idea to make sure that small populations are divided into smaller clusters to facilitate comprehensive downsampling of even smaller clusters. The next is to calculate the neighbourhood density of nodes. A degree of a node (i.e. a cell in the graph) is the number of nodes it is connected to, the two step degree (aka 1 neighbourhood degree)of a cell is the sum of degrees of cells that a cell is connected to. We calculate the two neighbourhood degree of cells to obtain an estimate of how densely connected the cells are in each region of the graph. The more densely connected the cells are, the less the heterogeneity across them. These values are saved in the cell metadata table, here as 'RNA_node_density'. We can visualize these values using `plot_layout` method.

```python
ds.calc_node_density(neighbourhood_degree=2)
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_node_density', clip_fraction=0.1, colormap='coolwarm')
```

Now we are ready to perform down-sampling of cells. The extent of down sampling is primarily governed by the number of micro clusters, i.e. atleast 1 cell from each micro-cluster (*seed cells*) will be present in the down sampled data. However, this might not be sufficient to ensure that these will conserve the topology, i.e. are connected to each other. Hence, the `run_subsampling` method will run a prize-collecting Steiner graph search to ensure that *seed cells* are connected (to the extent that the full graph is connected). In order to do this we need to set a reward on each seed and non-seed cells. This is done using the parameter `rewards` which is provided a tuple with values for seed and non-seed cells. Low reward on seed cells might lead to them being excluded from the subsample (something that we should try to avoid). High reward on non-seed cells will lead to inflation of number of cells in the sample. We also set a value for parameter `seed_frac` which is the fraction of cells that should be randomly sampled from each micro-cluster. This value is dynamically increased to a maximum of double the `seed_frac` value based on the relative mean value of node density for that cluster. Hence, in effect we increase the sampling rate for micro clusters that have lower overall connectivity.

```python
ds.run_subsampling(seed_frac=0.05, rewards=(2, 0), min_nodes=1, min_edge_weight=0.1,
                   cluster_key='RNA_b_cluster', density_key='RNA_node_density')
```

As a result of subsampling the sub sampled cells are marked True under the cell metadata column `RNA_sketched`. We can visualize these cells using `plot_layout`


```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster', subselection_key='RNA_sketched')
```


As can be seen from the plot above the subsample containing 16.49% of the cells has cells from all regions of the UMAP landscape.


### 8) Working with non-default assays: surface protein expression (ADT) data

Hereon, we show how to work with non-default assay. Here we have surface protein data present in ADT assay. Let's check out the feature table for this assay.


```python
ds.ADT.feats.table
```
We can manually filter out the control antibodies by updating `I` to be False for those features

```python
ds.ADT.feats.update((~ds.ADT.feats.table.names.str.contains('control')).values)
ds.ADT.feats.table
```
Assays named ADT are automatically created as objects of ADTassay class that uses CLR (centred log ratio) normalization as the default normalization method.

```python
print (ds.ADT)
print (ds.ADT.normMethod.__name__)
```

Now we create a new graph of cells using just the `ADT` data.

```python
ds.make_graph(from_assay='ADT', feat_key='I', k=11, dims=11, n_centroids=100, log_transform=False)
```

Run UMAP on `ADT` graph


```python
ds.run_umap(from_assay='ADT', fit_n_epochs=500)
```

One can check concordance between RNA and ADT assays by visualizing the RNA cluster information on ADT data

```python
ds.plot_layout(layout_key='ADT_UMAP', color_by='RNA_cluster')
```


We can also run the clustering directly on `ADT` graph and visualize it on UMAP

```python
ds.run_clustering(from_assay='ADT', n_clusters=20)
ds.plot_layout(layout_key='ADT_UMAP', color_by='ADT_cluster')
```

Or the other way, visualize ADT clusters on RNA UMAP

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='ADT_cluster')
```


Individual ADT expression can be visualized in both UMAPs easily.

```python
ds.plot_layout(from_assay='ADT', layout_key='ADT_UMAP', color_by='CD3_TotalSeqB')
ds.plot_layout(from_assay='ADT', layout_key='RNA_UMAP', color_by='CD3_TotalSeqB')
```


##### End of vignette
