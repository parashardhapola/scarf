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
```

```python
cd ~
```

```python
scarf.fetch_dataset('tenx_10k_pbmc_atacseq')
```

```python
reader = scarf.CrH5Reader('tenx_10k_pbmc_atacseq/data.h5', 'atac')
reader.assayFeats
```

```python
writer = scarf.CrToZarr(reader, zarr_fn=f'tenx_10k_pbmc_atacseq/data.zarr', chunk_size=(5000, 2000))
writer.dump(batch_size=1000)
```

```python
ds = scarf.DataStore('tenx_10k_pbmc_atacseq/data.zarr', nthreads=4)
```

```python
ds.auto_filter_cells()
```

```python
ds.mark_prevalent_peaks(top_n=10000)
```

```python
ds
```

```python
ds.make_graph(feat_key='prevalent_peaks', k=7, dims=51, n_centroids=1000)
```

```python
ds.run_umap(fit_n_epochs=500, min_dist=0.5)
```

```python
ds.run_clustering(n_clusters=10)
```

```python
ds.plot_layout(layout_key='ATAC_UMAP', color_by='ATAC_cluster')
```

```python

```
