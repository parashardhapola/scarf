---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## Cell subsampling using TopACeDo

```{code-cell} ipython3
%load_ext autotime

import scarf
scarf.__version__
```

---
### 1) Installing dependencies

+++

We need to install the TopACeDo algorithm to perform subsampling:

```{code-cell} ipython3
!pip install git+https://github.com/fraenkel-lab/pcst_fast.git@deb3236cc26ee9fee77d5af40fac3f12bb753850
!pip install -U topacedo
```

---
### 2) Fetching pre-processed data

```{code-cell} ipython3
# Loading preanalyzed dataset that was processed in the `basic_tutorial` vignette
scarf.fetch_dataset('tenx_5K_pbmc_rnaseq', as_zarr=True, save_path='scarf_datasets')
```

```{code-cell} ipython3
ds = scarf.DataStore('scarf_datasets/tenx_5K_pbmc_rnaseq/data.zarr')
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster')
```

---
### 3) Run TopACeDo downsampler

+++

UMAP, clustering and marker identification together allow a good understanding of cellular diversity. However, one can still choose from a plethora of other analysis on the data. For example, identification of cell differentiation trajectories. One of the major challenges to run these analysis could be the size of the data. Scarf performs a topology conserving downsampling of the data based on the cell neighbourhood graph. This downsampling aims to maximize the heterogeneity while sampling cells from the data.

Here we run the TopACeDo downsampling algorithm that leverages Scarf's KNN graph to perform a manifold preserving subsampling of cells. The subsampler can be invoked directly from Scarf's DataStore object.

```{code-cell} ipython3
ds.run_topacedo_sampler(cluster_key='RNA_cluster', max_sampling_rate=0.1)
```

As a result of subsampling the subsampled cells are marked True under the cell metadata column `RNA_sketched`. We can visualize these cells using `plot_layout`

```{code-cell} ipython3
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster', subselection_key='RNA_sketched')
```

It may also be interesting to visualize the cells that were marked as `seed cells` used when PCST was run. These cells are marked under the column `RNA_sketch_seeds`.

```{code-cell} ipython3
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster', subselection_key='RNA_sketch_seeds')
```

---
### 4) Intermediate parameters of downsampling

+++

To identify the seed cells, the subsampling algorithm calculates cell densities based on neighbourhood degrees. Regions of higher cell density get a sampling penalty. The neighbourhood degree of individual cells are stored under the column `RNA_cell_density`.

```{code-cell} ipython3
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cell_density')
```

The dowsampling algorithm also identifies regions of the graph where cells form tightly connected groups by calculating mean shared nearest neighbours of each cell's nieghbours. The tightly connected regions get a sampling award. These values can be accessed from under the cell metadata column `RNA_snn_value`.

```{code-cell} ipython3
ds.plot_layout(layout_key='RNA_UMAP', color_by='RNA_snn_value')
```

---
That is all for this vignette.
