=====
Scarf
=====

|IMG1|

.. |IMG1| image:: docs/source/_static/scarf_logo.svg
    :width: 75%

Scarf is a Python package to performs memory efficient analysis of single-cell genomics data.

- Analyze atlas scale datasets on your laptop (tested with upto 4 million cells)
- Perform analysis of scATAC-Seq data on laptop computers (test on dataset with 700K cells and 1 million peaks)
- Make parallel implementations of UMAP and tSNE for quick cell embedding
- Perform hierarchical clustering that gives interpretable cluster relationships
- Down-sample cells using state-of-the-art TopACeDo method
- Perform quick and accurate projections of cells from one dataset onto another

Install Scarf with::

    pip install scarf-toolkit

Read documentation here: `scarf.rtfd.io`_ or jump to a `basic workflow of Scarf`_

.. _scarf.rtfd.io: scarf.rtfd.io
.. _basic workflow of Scarf: scarf.rtfd.io/en/latest/vignettes/public/basic_tutorial.html
