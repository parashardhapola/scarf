# FAQs

## How does Scarf compare to Scanpy?

Scarf performs many of the essential steps of scRNA-Seq analysis that can be
performed in Scanpy as well.

Benefits of Scarf over Scanpy:
- Low memory requirement, so one can analyze large datasets or many small- to
  medium-sized datasets in parallel.
- Performs topology-conserving data downsampling
- Supports multiple single-cell genomics, like: scRNA-Seq, CITE-Seq, scATAC-Seq

Benefits of Scanpy over Scarf:
- Faster performance on small- and medium-sized datasets
- Has multiple external tools integrated into its API and provides seamless access
  to those tools, like sample integration and trajectory inference
- Anndata format is supported much more widely

We see Scarf as a complementary tool to Scanpy and an analysis workflow for large
data should make use of both these tools. For example, make UMAP and clustering on
Scarf and then bring the downsampled cells into Scanpy to perform other downstream
analyses.

## Should I use tSNE or UMAP?
tSNE and UMAP are complementary data visualization tools. tSNE focuses on highlighting
the largest differences in the dataset while tSNE highlights the even smaller ones. UMAP
preserves the global structure which may sometimes come at cost of local resolution. We
suggest using tSNE for large (>50k cells) and atlas scale datasets because of its quick
runtime and ability to reveal the cellular diversity. UMAP is favoured when identification
of relationship of clusters is important. UMAP runtime can span in hours on atlas scale
datasets. In Scarf, UMAP and tSNE use the same initial embedding by default and have the
same input graph.


## Which clustering should we use, Paris or Leiden?
The Leiden clustering method is faster than Paris, especially when it comes to large scale
datasets. On small datasets that we have tested, Leiden clustering results seem to be more
concordant with UMAP clustering. Paris, however, clearly shows relationship between clusters
using the `plot_cluster_tree` method of the DataStore class. Due to low computational
requirements of both the methods we suggest that you run both the clustering methods and
visualize them together using `plot_cluster_tree` like this::

    ds.plot_cluster_tree(cluster_key='RNA_cluster',
                         fill_by_value='RNA_leiden_cluster')

This will allow you test the robustness of clusters and visualize the relationship between
Leiden clusters as well.

## How do I create a count matrix for my single-cell data?
Generating count matrices is the primary step of single-cell data analysis. For scRNA-Seq one can
tools like [STARsolo], [alevin-fry] or [kallisto|bustools]. If your data was generated using
10x's commercial solution then you can use [Cell Ranger]. In the case of single-cell ATAC-Seq data,
there you may try following protocol from [Yan et al] or [Cusanovich et al]. [Cell Ranger ATAC] can
be used if your data was generated using 10x's kit.

[STARsolo]: https://github.com/alexdobin/STAR/blob/master/docs/STARsolo.md
[alevin-fry]: https://alevin-fry.readthedocs.io/en/stable/
[kallisto|bustools]: https://www.kallistobus.tools/
[Cell Ranger]: https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/what-is-cell-ranger
[Yan et al]: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-1929-3
[Cusanovich et al]: https://www.cell.com/cell/fulltext/S0092-8674(18)30855-9
[Cell Ranger ATAC]: https://support.10xgenomics.com/single-cell-atac/software/pipelines/latest/what-is-cell-ranger-atac

## Can I use Scarf from R?
Unfortunately, not yet! Please let the developers know if you would like to create an R API for Scarf.

## Can I try Scarf without installing anything on my computer?
Yes, you may try Scarf on Google Colab, an online notebook environment that allows running any
Python code. {ref}`Check this out <colab>` for more details.

## What does Scarf's logo signify?
Scarf's logo is highly inspired by the Human Cell Atlas's logo.
Scarf's logo is composed of three circular fields, each composed of Voronoi cells representing
atlas-scale datasets composed of multiple cell types. The arrangement of these 'atlases' symbolizes
Scarf's ability to downsample and integrate them.
