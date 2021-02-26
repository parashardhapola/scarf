---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%load_ext autotime
%config InlineBackend.figure_format = 'retina'

import scarf
```

```python
cd ~
```

```python
# Loading preanalyzed dataset that processed in `basic_tutorial` vignette
ds = scarf.DataStore('scarf_data/tenx_10k_pbmc_citeseq/data.zarr')
ds.run_clustering(n_clusters=21)
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster')
```

### Down sampling

UMAP, clustering and marker identification together allow a good understanding of cellular diversity. However, one can still choose from a plethora of other analysis on the data. For example, identification of cell differentiation trajectories. One of the major challenges to run these analysis could be the size of the data. Scarf performs a topology conserving downsampling of the data based on the cell neighbourhood graph. This downsampling aims to maximize the heterogeneity while sampling the cells from the the data. Before the actual downsampling step, two key steps must be performed.


Now we are ready to perform down-sampling of cells. The extent of down sampling is primarily governed by the number of micro clusters, i.e. atleast 1 cell from each micro-cluster (*seed cells*) will be present in the down sampled data. However, this might not be sufficient to ensure that these will conserve the topology, i.e. are connected to each other. Hence, the `run_topacedo_sampler` method will our TopACeDo (Topology assisted cell downsampling) algorithm which employs a prize-collecting Steiner graph traveral (PCST) to ensure that *seed cells* are connected (to the extent that the full graph is connected). In order to do this we need to set a reward on each seed and non-seed cells. This is done using the parameter `seed_reward` and `non_seed_reward`. Low reward on seed cells might lead to them being excluded from the subsample (something that we should try to avoid). High reward on non-seed cells will lead to inflation of number of cells in the sample. We also set a value for parameter `sampling_rate` which is the fraction of cells that should be randomly sampled from each micro-cluster.

```python
ds.run_topacedo_sampler(cluster_key='RNA_cluster', max_sampling_rate=0.2)
```

As a result of subsampling the sub sampled cells are marked True under the cell metadata column `RNA_sketched`. We can visualize these cells using `plot_layout`


```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster', subselection_key='RNA_sketched')
```

It may also be interesting to visualize the cells that were marked as `seed cells` useing using PCST was run. These cells are marked under the column `RNA_sketch_seeds`

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster', subselection_key='RNA_sketch_seeds')
```

Internally, Topacedo is trying to acheive two goals:
- Reduce redundancy in data (select fewer cells that are very similar)
- Ensure that the complete monfil remains connected after downsampling (achieved using the means of PCST algorithm)

To reduce the redundancy, Topacedo modifies `sampling_rate` for each cluster by mulitplying it to a coefficient, called Density based Sampling Rate Adjusted (DSRA). 

DSRA is derived from cell neighbourhood density which provides an estimate of how densely connected the cells are in each region of the graph. The more densely connected the cells are, the less the heterogeneity across them. A degree of a node (i.e. a cell in the graph) is the number of nodes it is connected to, the two step degree (aka 1 neighbourhood degree) of a cell is the sum of degrees of cells that a cell is connected to. Hence in this example we defined neighbourhood density at value of 2.

To calculate DSRA, we min-max normalize the neighbourhood density and for each cell substract these value from 1. Lower DSRA values will lead to decrease in effective sampling rate from clusters. These values are, by default, saved in the cell metadata table with 'cell_density' suffix. We can visualize these values using `plot_layout` method.


As can be seen from the plot above the subsample containing much smaller fraction of the cells has cells from all regions of the UMAP landscape.

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cell_density')
```

##### End of vignette
