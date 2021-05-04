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

## Workflow for scATAC-Seq data

```python
%load_ext autotime
%config InlineBackend.figure_format = 'retina'

import scarf
scarf.__version__
```

```python
scarf.fetch_dataset('tenx_10k_pbmc_atacseq')
reader = scarf.CrH5Reader('tenx_10k_pbmc_atacseq/data.h5', 'atac')
reader.assayFeats
```

```python
writer = scarf.CrToZarr(reader, zarr_fn=f'tenx_10k_pbmc_atacseq/data.zarr', chunk_size=(500, 1000))
writer.dump(batch_size=500)
```

```python
ds = scarf.DataStore('tenx_10k_pbmc_atacseq/data.zarr', nthreads=1)
```

```python
# ds.auto_filter_cells()
# ds.mark_prevalent_peaks(top_n=20000)
# ds.make_graph(feat_key='prevalent_peaks', k=11, dims=21, n_centroids=1000)
# ds.run_umap(fit_n_epochs=250, min_dist=0.5, parallel=True)
# ds.run_leiden_clustering(resolution=1)
# ds.plot_layout(layout_key='ATAC_UMAP', color_by='ATAC_leiden_cluster')
```
