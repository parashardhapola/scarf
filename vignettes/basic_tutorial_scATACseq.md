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
scarf.fetch_dataset('tenx_10k_pbmc_atacseq', save_path='scarf_data')
reader = scarf.CrH5Reader('scarf_data/tenx_10k_pbmc_atacseq/data.h5', 'atac')
reader.assayFeats
```

```python
writer = scarf.CrToZarr(reader, zarr_fn=f'scarf_data/tenx_10k_pbmc_atacseq/data.zarr', chunk_size=(1000, 2000))
writer.dump(batch_size=1000)
```

```python
ds = scarf.DataStore('tenx_10k_pbmc_atacseq/data.zarr', nthreads=4)
```

```python
ds.auto_filter_cells()
```

```python
ds.mark_prevalent_peaks(top_n=20000)
```

```python
ds.make_graph(feat_key='prevalent_peaks', k=11, dims=21, n_centroids=1000)
```

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
That is all for this vignette.
