|PyPI| |Docs| |Tests| |Gitter| |commits-latest| |pypi-downloads|

|IMG1|


.. |PyPI| image:: https://img.shields.io/pypi/v/scarf.svg
         :target: https://pypi.org/project/scarf
.. |Docs| image:: https://readthedocs.org/projects/scarf/badge/?version=latest
         :target: https://scarf.readthedocs.io
.. |Tests| image:: https://circleci.com/gh/parashardhapola/scarf/tree/master.svg?style=svg
          :target: https://circleci.com/gh/parashardhapola/scarf/tree/master
.. |commits-latest| image:: https://img.shields.io/github/last-commit/parashardhapola/scarf
                   :target: https://github.com/parashardhapola/scarf/commit/master
                   :alt: GitHub last commit
.. |pypi-downloads| image:: https://img.shields.io/pypi/dm/scarf
                   :target: https://pypi.org/project/scarf/
                   :alt: PyPI - Downloads

.. |IMG1| image:: docs/source/_static/scarf_logo.svg
         :width: 75%


Scarf is a Python package that performs memory-efficient analysis of single-cell genomics data.

- Analyze atlas scale datasets on your laptop (tested with up to 4 million cells)
- Perform analysis of scATAC-Seq data (datasets with up to 700K cells with 1 million peaks tested) under 10 GB RAM
- Make parallel implementations of UMAP and tSNE (SG-tSNE) for quick cell embedding
- Perform hierarchical clustering that gives interpretable cluster relationships
- Sub-sample highly representative cells using state-of-the-art TopACeDo method
- Perform quick and accurate projections of cells from one dataset onto another or integrate multiple datasets.

Preprint describing Scarf is out on `Biorxiv`_

Install Scarf with::

    pip install scarf

Read the documentation here: `scarf.rtfd.io`_ or jump to a `basic workflow of Scarf`_

.. _scarf.rtfd.io: http://scarf.rtfd.io
.. _basic workflow of Scarf: https://scarf.readthedocs.io/en/latest/vignettes/basic_tutorial_scRNAseq.html
.. _Biorxiv: https://www.biorxiv.org/content/10.1101/2021.05.02.441899v1

**Known issues**:

`DataStore.run_tsne()` does not work on Windows and Mac: We currently ship a pre-compiled version SG-tSNE. This will be
fixed in future updates.

High memory consumption: If you are using a version of Scarf less than 0.7.0 and have dask version >2021.03.1
then you might face high memory consumption issues. The solution is to install dask==2021.03.1. The
latest version of Scarf automatically solves this issue.
