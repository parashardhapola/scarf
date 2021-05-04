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

```python
%config InlineBackend.figure_format = 'retina'
%load_ext autotime

import scarf
scarf.__version__
```

```python
cd ~
```

```python
scarf.fetch_dataset('kang_ctrl_pbmc_rnaseq', save_path='scarf_data')
scarf.fetch_dataset('kang_stim_pbmc_rnaseq', save_path='scarf_data')
```

```python
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

```python
# Control PBMC data
ds_ctrl = scarf_pipeline(in_dir='scarf_data/kang_ctrl_pbmc_rnaseq')

# Interferon beta stimulated PBMC data
ds_stim = scarf_pipeline(in_dir='scarf_data/kang_stim_pbmc_rnaseq')
```

```python
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
                    savename='./scarf_data/kang_umap_control.svg', save_dpi=300)
```

```python
ds_ctrl.run_marker_search(group_key='cluster_labels')
ds_ctrl.plot_marker_heatmap(group_key='cluster_labels', topn=5, figsize=(8,12))
```

```python
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
ds_stim.plot_layout(layout_key='RNA_UMAP', color_by='cluster_labels', savename='./scarf_data/kang_umap_ifnb.svg', save_dpi=300)
```

As done before for control cells, we make sense of stimulated cells' clusters by running marker search and visualizing top 5 marker genes for each cluster in form of a heatmap.

```python
ds_stim.run_marker_search(group_key='cluster_labels')
ds_stim.plot_marker_heatmap(group_key='cluster_labels', topn=5, figsize=(8,12))
```

```python
ds_ctrl.run_mapping(target_assay=ds_stim.RNA, target_name='stim',
                    target_feat_key='hvgs_ctrl', save_k=5, run_coral=True)
```

```python
ds_ctrl.run_unified_umap(target_names=['stim'], ini_embed_with='RNA_UMAP', target_weight=1,
                         use_k=5, fit_n_epochs=100, tx_n_epochs=10)
```

```python
ds_ctrl.plot_layout(layout_key='RNA_unified_UMAP', color_by='cluster_labels',
                    savename='./scarf_data/kang_uni_umap_control_labels.svg', save_dpi=300)
```

```python
ds_ctrl.plot_unified_layout(layout_key='unified_UMAP', show_target_only=True, legend_ondata=True,
                            target_groups=ds_stim.cells.fetch('cluster_labels'),
                           savename='./scarf_data/kang_uni_umap_ifn_labels.svg', save_dpi=300)
```

```python
ds_stim, ds_ctrl
```

```python
import pandas as pd
```

```python
ds_stim.cells.insert('predicted_labels', 
        ds_ctrl.get_target_classes(target_name='stim', reference_class_group='cluster_labels').values,
        overwrite=True)
```

```python
df = pd.crosstab(ds_stim.cells.fetch('cluster_labels'),
                 ds_stim.cells.fetch('predicted_labels'))
df.to_csv('kang_ifnb_preds.csv')
```

```python
(100 * df / df.sum(axis=0)).round(1)
```

```python
ds_ctrl.plot_unified_layout(layout_key='unified_UMAP', show_target_only=True, legend_ondata=True,
                            target_groups=ds_stim.cells.fetch('predicted_labels'),
                            savename='./scarf_data/kang_uni_umap_pred_labels.svg', save_dpi=300)
```

```python
ds_stim.plot_layout(layout_key='RNA_UMAP', color_by='predicted_labels', savename='./scarf_data/kang_umap_pred_labels.svg', save_dpi=300)
```

```python
ds_ctrl.plot_unified_layout(layout_key='unified_UMAP', point_size=4, scatter_kwargs={'lw': 0}, shuffle_zorder=True,
                            savename='./scarf_data/kang_uni_umap_sample.svg', save_dpi=300)
```
