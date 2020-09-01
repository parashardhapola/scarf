==========================
Frequently asked questions
==========================

How does Scarf compare to Scanpy
--------------------------------
Scarf performs many of the essential steps of scRNA-Seq analysis that can be performed in Scanpy as well.

Benefits of Scarf over Scanpy:

- Low memory requirement so one can analyze large datasets or many small to medium sized datasets in parallel.
- Performs topology conserving data downsampling
- Supports multiple single-cell genomics like, scRNA-Seq, CITE-Seq, scATAC-Seq

Benefits of Scanpy over Scarf:

- Faster performance on small and medium sized datasets
- Has mutiple external tools integrated into API and provides seemless access to those tools, like sample integration and trajectory inference
- Anndata format is supported much more widely

We see Scarf as a complementary tool to Scanpy and an analysis workflow for large data should make use of both these tools.
For example make UMAP and clustering on Scarf and then bring the downsampled cells into Scanpy to perform other downstream analyses.

Can I use Scarf from R
----------------------
Unfortunately, not yet! Please let the developers know if you would like to create an R API for Scarf. For scATAC-Seq analysis archR is good alternative to Scarf
