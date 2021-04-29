===
API
===

BaseDataStore
-------------

This is the base datastore class that deals with loading of assays from Zarr files and generating basic cell
statistics like nCounts and nFeatures.

.. autoclass:: scarf.BaseDataStore
    :members:

GraphDataStore
--------------

This class extends BaseDataStore by providing methods required to generate a cell-cell neighbourhood graph. It also
contains all the methods that use the KNN graphs as primary input like UMAP/tSNE embedding calculation, clustering,
down-sampling etc.

.. autoclass:: scarf.GraphDataStore
    :members:

MappingDatastore
----------------

This class extends GraphDataStore by providing methods for mapping/ projection of cells from one DataStore onto another.
It also contains the methods reuqired for label transfer, mapping score generation and co-embedding.

.. autoclass:: scarf.MappingDatastore
    :members:

DataStore
---------

This class extends MappingDatastore and consequently inherits methods of all the above DataStore classes. This class is
the main user facing class as it provides most of the plotting functions. It also contains methods for cell filtering,
feature selection, marker features identification, subsetting and aggregating cells. This class also contains methods
that perform in-memory data exports.

.. autoclass:: scarf.DataStore
    :members:

Assay
-----

A generic Assay class that contains methods to calculate feature level statistics. It also provides method for saving
normalized subset of data for later KNN graph construction.

.. autoclass:: scarf.assay.Assay
    :members:

RNAassay
--------

This assay is designed for feature selection and normalization of scRNA-Seq data

.. autoclass:: scarf.assay.RNAassay
    :members:

ATACassay
---------

This assay is designed for feature selection and normalization of scATAC-Seq data

.. autoclass:: scarf.assay.ATACassay
    :members:


ADTassay
--------

This assay is designed for feature selection and normalization of ADTs from CITE-Seq data

.. autoclass:: scarf.assay.ADTassay
    :members:


MetaData
--------

.. autoclass:: scarf.metadata.MetaData
    :members:
