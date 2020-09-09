---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

### Data integration strategies available through Scarf

In this vignette we will try to integrate data that was obtained [from Kang et. al.](https://www.nature.com/articles/nbt.4042). The data consists of stimulated (IFN-beta) and unstimulated PBMCs. In the article, the authors highlight how IFN-beta treatment induces gene expression changes resulting in separation of two samples on low dimension embedding plots. We will try to integrate the two datasets such that a equivalent cell types from both the conditions can be compared. This vignette is inspired by Seurat's 'immune_alignment' vignette available [here](https://satijalab.org/seurat/v3.0/immune_alignment.html). It should be noted though that Scarf's strategy of data integration is quite different from Seurat and we <ins>do not</ins> attempt to compare the results obtained through Scarf vs Seurat.

The data was downloaded from [this GEO repo](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583) of the article. The vignette assumes that the data was stored in a directory called `kang_stim_pbmc` which contains two subdirectories `control` (for control cells) and `stim` (IFN-beta stimulated cells). Both of these directories contain the 10x format barcodes, genes and matrix files.


Import Scarf. We also run autotime Ipython magic command that will help track the runtime duration of each cell.

```python
%load_ext autotime

import scarf
```

Change directory to the location where `kang_stim_pbmc` is located

```python
cd ../../data
```

Since in this vignette multiple datasets will be handled, we created a function ``scarf_pipeline`` that contains the basic steps of Scarf workflow: loading Zarr file, marking HVGs, creation of cell-cell neighbourhood graph, clustering and UMAP embedding of the data. For further information on these steps please check the [basic tutorial vignette](./basic_tutorial.md). Here we have designed the pipeline in a way that will allow us to update required parameters easily for pedagogic purposes.

```python
def scarf_pipeline(zarr_fn, in_dir=None, pca_cell_key='I', umap_label='UMAP', feat_key='hvgs', n_cluster=20):
    if in_dir is not None:
        reader = scarf.CrDirReader(in_dir, 'rna')
        scarf.CrToZarr(reader, zarr_fn=zarr_fn, chunk_size=(2000, 2000)).dump(batch_size=2000)
    ds = scarf.DataStore(zarr_fn, auto_filter=False)
    ds.RNA.mark_hvgs(min_cells=20, top_n=2000)
    ds.make_graph(feat_key=feat_key, k=21, dims=21, n_centroids=100, log_transform=True, renormalize_subset=True, pca_cell_key=pca_cell_key)
    ds.run_clustering(n_clusters=n_cluster, min_edge_weight=0.1)
    ds.run_umap(fit_n_epochs=500, min_dist=0.5, label=umap_label)
    return ds
```

We start by creating DataStore objects for both control and stimulated cells.


### 1) Independent analysis of the two samples

```python
# Control PBMC data
ds_ctrl = scarf_pipeline('pbmc_kang_control.zarr', 'kang_stim_pbmc/control')

# Interferon beta stimulated PBMC data
ds_stim = scarf_pipeline('pbmc_kang_stimulated.zarr', 'kang_stim_pbmc/stim')
```

Let's visualize the UMAP embedding of control PBMCs along with clustering information

```python
ds_ctrl.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster')
```

To make sense of these cluster we can run marker search and visualize top 5 marker genes for each cluster in form of a heatmap

```python
ds_ctrl.run_marker_search(group_key='RNA_cluster')
ds_ctrl.plot_marker_heatmap(group_key='RNA_cluster', topn=5, figsize=(8,12))
```

So we see three distinct clusters which have varying levels internal heterogeneity. Let's visualize the UMAP embedding of IFN-beta stimulated PBMCs along with clustering information

```python
ds_stim.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster')
```

As done before for control cells, we make sense of stimulated cells' clusters by running marker search and visualizing top 5 marker genes for each cluster in form of a heatmap

```python
ds_stim.run_marker_search(group_key='RNA_cluster')
ds_stim.plot_marker_heatmap(group_key='RNA_cluster', topn=5, figsize=(8,12))
```

UMAP embedding of stimulated cells show that overall heterogeneity of the cells is quite similar to control cells. However, because clustering was performed independently it is hard to infer how the cells are related across the two samples. 


### 2) KNN mapping approach to link cells between two (or more) datasets.


``run_mapping`` method of DataStore class performs KNN mapping/projection of target cells over reference cells. Reference cells are the one that are present in the object where `run_mapping` is being called. The `Assay` object of target cells is provided as an argument to `run_mapping`. This step will load the latest graph of the reference cells and query the ANN index of the reference cells for nearest neighbours of each target cells. Since the ANN index doesn't contain any target cells, nearest neighbours of target cells will exclusively be reference cells. Under the hood, `run_mapping` method makes sure that the features order in the target cells is same as that in reference cells. By default, `run_mapping` will impute zero values for missing features in the target order to preserve the feature order. Here we have set `run_coral` parameter to True which activates CORAL normalization of target data. CORAL aligns the the feature distribution between reference and target cells thus removing systemic difference between reference and target cells. Read more about CORAL [here](https://arxiv.org/pdf/1612.01939.pdf). Here we use control PMBCs as reference because we invoke `run_mapping` on control PBMCs' DataStore object and provide stimulated PBMC's `RNA` assay as target.

```python
ds_ctrl.run_mapping(target_assay=ds_stim.RNA, target_name='stim',
                    target_feat_key='hvgs_ctrl', save_k=5, run_coral=True)
```

We can now extend the cell-cell neighbourhood graph of reference cells (control PMBCs) by including the target cells (stimulated PBMCs) based on their nearest reference cells. A 'unified' UMAP embedding of this extended graph can then be generated to visualize the reference and target cells together. To encourage similarity between UMAP of control only cells and this 'unified' UMAP, we use the UMAP coordinates of control only cells for initialization of UMAP. ``run_unified_umap`` will take the name of the target cells to be included in the extended graph.

```python
ds_ctrl.run_unified_umap(target_name='stim', ini_embed_with='RNA_UMAP', use_k=5, fit_n_epochs=100, tx_n_epochs=10)
```

We fist visualize the unified UMAP embedding of control cells. These embedding were automatically saved in cell metadata columns starting with `RNA_UMAP_stim`. When compared to independent control PMBC UMAP, this UMAP looks strikingly similar. This mostly due to the fact that independent control PMBC UMAP coordinates were used for initialization, but also because the inclusion of target cells in the UMAP did not have a major impact.

```python
ds_ctrl.plot_layout(layout_key='RNA_UMAP_stim', color_by='RNA_cluster')
```

Because the UMAP coordinates of target cells cannot be stored in cell metadata object of the reference, ``plot_layout`` method cannot be used to visualize the 'unified' UMAP. Hence, we use a specialized method, ``plot_unified_layout``. By default the reference cells (control PBMCs) will be shown in `coral` colour while the target cells (stimulated cells here) will be shown in `black`

```python
ds_ctrl.plot_unified_layout(target_name='stim')
```

One can clearly see that the target cells have integrated more or less evenly throughout the reference UMAP landscape.

``plot_unified_layout`` can also be used to visualize target cells only. Here we colour the target cells based on the cluster information in independent analysis of stimulated cells.

```python
ds_ctrl.plot_unified_layout(target_name='stim', show_target_only=True, target_color=ds_stim.cells.fetch('RNA_cluster'))
```

Quite clearly, the target cells are <ins>not</ins> located randomly on the unified UMAP but mostly are aggregated based on their own cluster information. This allows for a quick visual inspection of cell types. For example, cluster 3 from control PBMCs is likely the same cell type as cluster 5 from stimulated PBMCs.

As an alternative to unified UMAPs, we can use `mapping scores` to perform cross dataset cluster similarity inspection. `mapping scores` are scores assigned to each reference cell based on how frequently it was identified as one of the nearest neighbour of the target cells. ``get_mapping_score`` method allows generating these scores. We use an optional parameter of `get_mapping_score`, `target_groups`. `target_groups` takes grouping information for target cells such that mapping scores are calculated for one group at a time. Here we provide the cluster information of stimulated cells as group information and mapping scores will be obtained for each target cluster independently.

```python
# Here we will generate plots for target clusters 5, 7 and 8 for sake of brevity

for g, ms in ds_ctrl.get_mapping_score(target_name='stim', target_groups=ds_stim.cells.fetch('RNA_cluster'), log_transform=False):
    if g in [5, 7, 8]:
        print (f"Target cluster {g}")
        ds_ctrl.plot_layout(layout_key='RNA_UMAP', color_by='RNA_cluster', size_vals=ms*10, height=4, width=4, legend_onside=False)
```

<!-- #region -->
Again, one can clearly see, as an example, that projection from stimulated cell cluster 5 has led to highest mapping scores in control cell cluster 3.


The mapping can be performed both ways. It is highly recommended that users try to map to the cells both ways in order to completely evaluate the similarities and differences between the samples. We repeat the steps above but this time use stimulated PBMCs as reference and control PBMCs as target.
<!-- #endregion -->

```python
ds_stim.run_mapping(target_assay=ds_ctrl.RNA, target_name='ctrl',
                    target_feat_key='hvgs_stim', save_k=5, run_coral=True)
ds_stim.run_unified_umap(target_name='ctrl', ini_embed_with='RNA_UMAP', use_k=5, fit_n_epochs=100, tx_n_epochs=10)
```

```python
ds_stim.plot_layout(layout_key='RNA_UMAP_ctrl', color_by='RNA_cluster')
```

```python
ds_stim.plot_unified_layout(target_name='ctrl')
```

```python
ds_stim.plot_unified_layout(target_name='ctrl', show_target_only=True, target_color=ds_ctrl.cells.fetch('RNA_cluster'))
```

### 3) Integration through merging of Zarr files


A simple for integrating two or more datasets is to simple merge the Zarr files and process all the cells as if they were one sample. ``ZarrMerge`` class is used to merge two or more assays from different DataStores into a single Zarr file

```python
scarf.ZarrMerge('pbmc_kang_merged.zarr', [ds_ctrl.RNA, ds_stim.RNA], ['ctrl', 'stim'], 'RNA', overwrite=True).write()
```

The merged Zarr can then be used like any other Scarf Zarr file and loaded as a DataStore object. Here we run the `scarf_pipeline` that was created in the beginning of this vignette on the merged Zarr file

```python
ds_merged = scarf_pipeline('pbmc_kang_merged.zarr')
```

One can see that in the cell metadata table, the names of samples are prepended to cell ids.

```python
ds_merged.cells.table
```

We can extract these sample names and cell ids and add them as separate column

```python
ds_merged.cells.add(k='sample_id', v=[x.split('__')[0] for x in ds_merged.cells.table.ids.values], overwrite=True)
```

Additionally, we can create two more columns that essentially have same information as the `sample_id` column but are in boolean format. One column for control cells and other for stimulated cells such that the values will be True in respective columns if a cell belong to that sample otherwise False. Having sample information organized this way will make few steps below more convenient for us.

```python
for i in ['ctrl', 'stim']:
    ds_merged.cells.add(k=f'sample_{i}', v=ds_merged.cells.table['sample_id'].isin([i]).values, overwrite=True)
```

```python
ds_merged.cells.table
```

Let's visualize the UMAP of this merged dataset and colur the cells by `sample_id` column created above

```python
ds_merged.plot_layout(layout_key='RNA_UMAP', color_by='sample_id', colormap='RdBu')
```

We can also visualize the cells from two samples separately using `subselection_key` in `plot_layout`

```python
ds_merged.plot_layout(layout_key='RNA_UMAP', subselection_key='sample_stim')
ds_merged.plot_layout(layout_key='RNA_UMAP', subselection_key='sample_ctrl')
```

It is quite clear that the cells from two samples are systematically different and simply merging the Zarr files is not sufficient to merge them. Therefore taking the KNN mapping approach that we demonstrated might be useful.


### 4) Integration through usage of common HVGs and union HVGs


Another approach that is commonly used is supervised selection of features. Merging together highly variable genes (HVGs) that were identified independently in each dataset has proven to be useful in certain cases of sample integration.


#### Integration using union HVGs

We can consider a gene to be HVG if it was detected as HVG in <ins>any</ins> one of the two datasets. We start by first extracting the gene id for HVGs from both the samples

```python
ctrl_hvgs = ds_ctrl.RNA.feats.table.ids[ds_ctrl.RNA.feats.table['I__hvgs']].values
stim_hvgs = ds_stim.RNA.feats.table.ids[ds_stim.RNA.feats.table['I__hvgs']].values
len(ctrl_hvgs), len(stim_hvgs)
```

We can see that each sample has 2000 HVGs marked. A 'union' set operation can be used to merge together the HVGs

```python
union_hvgs = list(set(ctrl_hvgs).union(stim_hvgs))
len(union_hvgs)
```

Now we identify the indices of these 3191 features in the feature table and them as a new boolean column in the feature metadata table

```python
union_hvgs_idx = ds_merged.RNA.feats.get_idx_by_ids(union_hvgs)
ds_merged.RNA.feats.add(k='I__union_hvgs', v=ds_merged.RNA.feats.idx_to_bool(union_hvgs_idx), overwrite=True)
```

```python
ds_merged.RNA.feats.table
```

Rerun the scarf pipeline using `union_hvgs` as `feat_key`. We will make sure that the previous UMAP data is not overwritten and hence will provide a new label to save UMAP coordinates. Please note that the pipeline will attempt to find HVGs again but these will not be used as the pipeline will directly use the column `I__common_hvgs` of features.

```python
ds_merged = scarf_pipeline('pbmc_kang_merged.zarr', feat_key='union_hvgs', umap_label='UMAP_union_hvgs')
```

Plot the UMAP layout for the graph generated using union HVGs

```python
ds_merged.plot_layout(layout_key='RNA_UMAP_union_hvgs', color_by='sample_id', colormap='RdBu')
```

We can clearly see that using union HVGs has led to improved merging between the two datasets. This is most likely due to inclusion of cell type discriminatory genes.


#### Integration using common (intersection) HVGs


Now we try a slight variant of the above approach where rather than taking genes which are HVGs in any of the two samples, we consider only those genes that are HVGs in both the samples. This can easily be done using `intersection` set operation

```python
common_hvgs = list(set(ctrl_hvgs).intersection(stim_hvgs))
len(common_hvgs)
```

We now follow similar steps as for union HVGs and run the pipeline using `common_hvgs`

```python
common_hvgs_idx = ds_merged.RNA.feats.get_idx_by_ids(common_hvgs)
ds_merged.RNA.feats.add(k='I__common_hvgs', v=ds_merged.RNA.feats.idx_to_bool(common_hvgs_idx), overwrite=True)
ds_merged = scarf_pipeline('pbmc_kang_merged.zarr', feat_key='common_hvgs', umap_label='UMAP_common_hvgs')
```

Now we plot the UMAP embedding obtained for graph generated using only the common HVGs between the two samples.

```python
ds_merged.plot_layout(layout_key='RNA_UMAP_common_hvgs', color_by='sample_id', colormap='RdBu')
```

We clearly see here that using the intersection of genes has improved overlap of clusters that were already integrating well. However we still see strong sample wise separation of cells.


### 5) Integration through partial PCA training


An alternate strategy to HVG selection is to use all the HVGs that were found in the merged data but rather than training PCA on all the cells, we train PCA on only one of the samples. Training PCA on just one sample will decrease the variance of sample specific genes and hence they will be included in later PCs and hence will have decreased effect on distance calculation between cells. Once the PCA is trained on cells from one of the sample, all the cells (from both samples) are projected onto PCA space as usual. We will provide `pca_cell_key` to our pipeline as argument, which then is used in ``make_graph``method.

```python
ds_merged = scarf_pipeline('pbmc_kang_merged.zarr', pca_cell_key='sample_ctrl', umap_label='UMAP_ctrl_trained', n_cluster=10)
```

Visualization as UMAP

```python
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='sample_id', colormap='RdBu')
```

This approach successfully integrates the cells from the two samples. We will now investigate the difference between these clusters, as well as the difference between the two samples i.e. gene expression changes induced by interferon beta. Next we visualize the clusters in this integrated UMAP

```python
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='RNA_cluster')
```

We try to identify cluster markers in order to annotate each cluster

```python
ds_merged.run_marker_search(group_key='RNA_cluster')
ds_merged.plot_marker_heatmap(group_key='RNA_cluster', topn=5, figsize=(5,10))
```

We can now individually visualize some these cluster markers and plot them separately for two samples in the integrated UMAP

```python
# B cell marker
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='CD79A', subselection_key='sample_ctrl', height=4, width=4)
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='CD79A', subselection_key='sample_stim', height=4, width=4)
```

```python
# Monocyte marker
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='FCGR2A', subselection_key='sample_ctrl', height=4, width=4)
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='FCGR2A', subselection_key='sample_stim', height=4, width=4)
```

```python
# NK/CD8 T cell marker
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='GNLY', subselection_key='sample_ctrl', height=4, width=4)
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='GNLY', subselection_key='sample_stim', height=4, width=4)
```

Now let's run a marker search for the differences between the cells from two samples.

```python jupyter={"outputs_hidden": true}
ds_merged.run_marker_search(group_key='sample_id')
ds_merged.plot_marker_heatmap(group_key='sample_id', topn=20, figsize=(3, 8), cmap='coolwarm')
```

Let's verify that expression of some of these genes is indeed different between the two samples

```python
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='IFIT3', subselection_key='sample_ctrl', height=4, width=4)
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='IFIT3', subselection_key='sample_stim', height=4, width=4)
```

```python
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='IL8', subselection_key='sample_ctrl', height=4, width=4)
ds_merged.plot_layout(layout_key='RNA_UMAP_ctrl_trained', color_by='IL8', subselection_key='sample_stim', height=4, width=4)
```

End of vignette.
