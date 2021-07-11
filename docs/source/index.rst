Scarf
=====

Scarf is a Python package that performs memory-efficient analysis of single-cell genomics data.

- Analyze atlas scale datasets on your laptop (tested with up to 4 million cells)
- Perform analysis of scATAC-Seq data (datasets with up to 700K cells with 1 million peaks tested) under 10 GB RAM
- Make parallel implementations of UMAP and tSNE (SG-tSNE) for quick cell embedding
- Perform hierarchical clustering that gives interpretable cluster relationships
- Sub-sample highly representative cells using state-of-the-art TopACeDo method
- Perform quick and accurate projections of cells from one dataset onto another or integrate multiple datasets.

.. toctree::
    :maxdepth: 2

    install
    vignettes
    faq
    api
    license
    genindex
