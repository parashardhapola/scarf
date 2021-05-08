|PyPI| |Docs| |Gitter|

|IMG1|


.. |PyPI| image:: https://img.shields.io/pypi/v/scarf-toolkit.svg
         :target: https://pypi.org/project/scarf-toolkit
.. |Docs| image:: https://readthedocs.org/projects/scarf/badge/?version=latest
         :target: https://scarf.readthedocs.io
.. |Gitter| image:: https://badges.gitter.im/scarf-toolkit/community.svg
           :target: https://gitter.im/scarf-toolkit/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |IMG1| image:: docs/source/_static/scarf_logo.svg
    :width: 75%

Scarf is a Python package to performs memory efficient analysis of single-cell genomics data.

- Analyze atlas scale datasets on your laptop (tested with upto 4 million cells)
- Perform analysis of scATAC-Seq data (datasets with upto 700K cells with 1 million peaks tested) under 10 GB RAM
- Make parallel implementations of UMAP and tSNE (SG-tSNE) for quick cell embedding
- Perform hierarchical clustering that gives interpretable cluster relationships
- Sub-sample highly representative cells using state-of-the-art TopACeDo method
- Perform quick and accurate projections of cells from one dataset onto another or integrate multiple datasets.

Install Scarf with::

    pip install scarf-toolkit

Read the documentation here: `scarf.rtfd.io`_ or jump to a `basic workflow of Scarf`_

.. _scarf.rtfd.io: http://scarf.rtfd.io
.. _basic workflow of Scarf: https://scarf.readthedocs.io/en/latest/vignettes/basic_tutorial_scRNAseq.html


**Known issues**:

`DataStore.run_tsne()` does not work on Windows and Mac: We currently ship a pre-compiled version SG-tSNE. This will be
fixed in future updates.

High memory consumption: If you are using a version of Scarf less than 0.7.0 and have dask version >2021.03.1
then you might face high memory consumption issues. The solution is to install dask==2021.03.1. The
latest version of Scarf automatically solves this issue.
