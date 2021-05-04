---
jupyter:
  jupytext:
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

```python
%load_ext autotime
%config InlineBackend.figure_format = 'retina'

import scarf
scarf.__version__
```

Need to install the TopACeDo algorithm to perform subsampling

```python
!pip install -U topacedo
```

```python
# Loading preanalyzed dataset that was processed in the `basic_tutorial` vignette
scarf.fetch_dataset('tenx_10k_pbmc_citeseq', as_zarr=True)
```

```python
ds = scarf.DataStore('tenx_10k_pbmc_citeseq/data.zarr')

ds.run_clustering(n_clusters=21)
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster')
```

### Cell subsampling

UMAP, clustering and marker identification together allow a good understanding of cellular diversity. However, one can still choose from a plethora of other analysis on the data. For example, identification of cell differentiation trajectories. One of the major challenges to run these analysis could be the size of the data. Scarf performs a topology conserving downsampling of the data based on the cell neighbourhood graph. This downsampling aims to maximize the heterogeneity while sampling the cells from the the data.

Here we TopACeDo downsampling algorithm that leverages Scarf's KNN graph to perform a manifold preserving subsampling of cells. The subsampler can be invoked directly from Scarf's DataStore object.

```python
ds.run_topacedo_sampler(cluster_key='RNA_cluster', max_sampling_rate=0.1)
```

As a result of subsampling the sub sampled cells are marked True under the cell metadata column `RNA_sketched`. We can visualize these cells using `plot_layout`

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster', subselection_key='RNA_sketched')
```

It may also be interesting to visualize the cells that were marked as `seed cells` useing using PCST was run. These cells are marked under the column `RNA_sketch_seeds`

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster', subselection_key='RNA_sketch_seeds')
```

To identify the seed cells, the subsampling algorithm calculates cell densities based on neighbourhood degrees. Regions of higher cell density get a sampling penalty. The neighbourhood degree of individual cells are stored under the column `RNA_cell_density`

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cell_density')
```

The dowsampling algorithm also identififes regions of graph where cells form tightly connected groups by calculating mean shared nearest neighbours of each cell's nieghbours. The tightly connected regions get a sampling award. These values can be accessed from under the cell metadata column `RNA_snn_value`

```python
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_snn_value')
```

##### End of vignette
