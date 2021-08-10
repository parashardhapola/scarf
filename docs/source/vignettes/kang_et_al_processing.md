---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'
%load_ext autotime

import scarf
scarf.__version__
```

```{code-cell} ipython3
scarf.fetch_dataset('kang_15K_pbmc_rnaseq', save_path='scarf_datasets')
scarf.fetch_dataset('kang_14K_ifnb-pbmc_rnaseq', save_path='scarf_datasets')
```

```{code-cell} ipython3
def scarf_pipeline(in_dir=None, zarr_fn=None, pca_cell_key='I',
                   umap_label='UMAP', feat_key='hvgs', n_cluster=20):
    if in_dir is not None:
        zarr_fn = in_dir + '/data.zarr'
        reader = scarf.CrDirReader(in_dir, 'rna')
        scarf.CrToZarr(reader, zarr_fn=zarr_fn, chunk_size=(2000, 2000)).dump(batch_size=4000)
    if zarr_fn is None:
        raise ValueError("Please provide a Zarr file")
    ds = scarf.DataStore(zarr_fn, nthreads=8)
    ds.filter_cells(attrs=['RNA_nCounts', 'RNA_nFeatures', 'RNA_percentMito'], highs=[6000, 1500, 1], lows=[500, 100, 0])
    ds.filter_cells(attrs=['RNA_nCounts'], highs=[None], lows=[1000])
    ds.mark_hvgs(min_cells=10, top_n=2000)
    ds.make_graph(feat_key=feat_key, k=11, dims=25, n_centroids=100,
                  log_transform=True, renormalize_subset=True, pca_cell_key=pca_cell_key)
    ds.run_leiden_clustering(resolution=2)
    ds.run_umap(fit_n_epochs=250, min_dist=0.5, label=umap_label)
    return ds
```

```{code-cell} ipython3
# Control PBMC data
ds_ctrl = scarf_pipeline(in_dir='scarf_datasets/kang_15K_pbmc_rnaseq')
```

```{code-cell} ipython3
ds_ctrl.run_marker_search(group_key='RNA_leiden_cluster')
ds_ctrl.plot_marker_heatmap(group_key='RNA_leiden_cluster', topn=5, figsize=(8,12))
```

```{code-cell} ipython3
ctrl_cluster_labels = {
    19: 'CD4 naive T', 3: 'CD4 naive T', 2: 'CD4 naive T', 24: 'CD4 naive T',
    1: 'CD4 Memory T', 14: 'T activated', 7: 'CD8 T', 12: 'NK',
    5: 'B', 16: 'B activated', 11: 'CD16 Mono', 8: 'CD16 Mono',
    10: 'CD 14 Mono', 13: 'CD 14 Mono', 20: 'CD 14 Mono', 6: 'CD 14 Mono',
    9: 'CD 14 Mono', 4: 'CD 14 Mono', 18: 'CD 14 Mono',
    15: 'DC', 22: 'pDC', 17: 'Mk', 21: 'Mk', 23: 'Eryth'
}
ds_ctrl.cells.insert('cluster_labels',
                     [ctrl_cluster_labels[x] for x in ds_ctrl.cells.fetch('RNA_leiden_cluster')],
                     overwrite=True)
ds_ctrl.plot_layout(layout_key='RNA_UMAP', color_by='cluster_labels',
                    save_dpi=300)
```

```{code-cell} ipython3
# Interferon beta stimulated PBMC data
ds_stim = scarf_pipeline(in_dir='scarf_datasets/kang_14K_ifnb-pbmc_rnaseq')
```

```{code-cell} ipython3
ds_stim.run_marker_search(group_key='RNA_leiden_cluster')
ds_stim.plot_marker_heatmap(group_key='RNA_leiden_cluster', topn=5, figsize=(8,12))
```

```{code-cell} ipython3
stim_cluster_labels = {
    24: 'CD4 naive T', 23: 'CD4 naive T', 19: 'CD4 naive T',
    5: 'CD4 naive T', 2: 'CD4 naive T', 14: 'CD4 naive T',
    1: 'CD4 Memory T', 13: 'T activated', 6: 'CD8 T', 11: 'NK',
    4: 'B', 16: 'B activated', 10: 'CD16 Mono', 9: 'CD16 Mono',
    3: 'CD 14 Mono', 7: 'CD 14 Mono', 8: 'CD 14 Mono', 12: 'CD 14 Mono',
    22: 'CD 14 Mono', 18: 'CD 14 Mono',
    15: 'DC', 21: 'pDC', 17: 'Mk', 22: 'Mk', 20: 'Eryth'
}
ds_stim.cells.insert('cluster_labels',
                     [stim_cluster_labels[x] for x in ds_stim.cells.fetch('RNA_leiden_cluster')],
                     overwrite=True)
ds_stim.plot_layout(layout_key='RNA_UMAP', color_by='cluster_labels', save_dpi=300)
```

```{code-cell} ipython3

```
