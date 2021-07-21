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

## Workflow for scATAC-Seq data

```python
%load_ext autotime
%config InlineBackend.figure_format = 'retina'

import scarf
scarf.__version__
```

---
### 1) Fetch and convert data

```python
scarf.fetch_dataset('tenx_10K_pbmc_atacseq', save_path='scarf_datasets')
reader = scarf.CrH5Reader('scarf_datasets/tenx_10K_pbmc_atacseq/data.h5', 'atac')
reader.assayFeats
```

```python
writer = scarf.CrToZarr(reader, zarr_fn=f'scarf_datasets/tenx_10K_pbmc_atacseq/data.zarr', chunk_size=(1000, 2000))
writer.dump(batch_size=1000)
```

---
### 2) Create DataStore and filter cells

```python
ds = scarf.DataStore('scarf_datasets/tenx_10K_pbmc_atacseq/data.zarr', nthreads=4)
```

```python
ds.auto_filter_cells()
```

---
### 3) Feature selection

For scATAC-Seq data, the features are ranked by their [TF-IDF](https://en.wikipedia.org/wiki/Tf-idf) normalized values, summed across all cells. The top n features are marked as `prevalent_peaks` and are used for downstream steps.

```python
ds.mark_prevalent_peaks(top_n=20000)
```

---
### 4) KNN graph creation

For scATAC-Seq datasets, Scarf uses TF-IDF normalization. The normalization is automatically performed during the graph building step. The selected features, marked as `prevalent_peaks` in feature metadata, are used for graph creation. For the dimension reduction step, LSI (latent semantic indexing) is used rather than PCA. The rest of the steps are same as for scRNA-Seq data.

```python
ds.make_graph(feat_key='prevalent_peaks', k=11, dims=21, n_centroids=1000)
```

<!-- #region -->
---
### 5) UMAP reduction and clustering


Non-linear dimension reduction using UMAP and tSNE are performed in the same way as for scRNA-Seq data. Similarly the clustering step is also performed in the same way as for scRNA-Seq data.
<!-- #endregion -->

```python
ds.run_umap(fit_n_epochs=250, min_dist=0.5, parallel=True)
```

```python
ds.run_leiden_clustering(resolution=1)
```

```python
ds.plot_layout(layout_key='ATAC_UMAP', color_by='ATAC_leiden_cluster')
```

---
### 6) Calculating gene scores

This feature is coming soon..

```python
ds.ATAC.feats.head()
```

---
That is all for this vignette.
