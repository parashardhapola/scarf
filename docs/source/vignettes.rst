=========
Vignettes
=========

.. toctree::
    :maxdepth: 1

    vignettes/basic_tutorial_scRNAseq
    vignettes/basic_tutorial_scATACseq
    vignettes/multiple_modalities
    vignettes/cell_subsampling_tutorial
    vignettes/data_projection
    vignettes/merging_datasets
    vignettes/zarr_explanation
    vignettes/download_conversion

Running notebooks live on Google Colab
--------------------------------------

Paste the following code on the top of the notebook before running any other cell on Colab notebook

::

    !pip install ipython-autotime
    !pip install scarf
    !pip install -U numpy scipy

Google Colab has older versions of Numpy and Scipy which are not compatible with Scarf.
Once `scipy` and `numpy` have updated you will see a `RESTART RUNTIME` button. Click on it to activate latest versions.
Now you are free to execute rest of the notebook.

Live notebook links:

- `Basic workflow of Scarf using scRNA-Seq data <https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/basic_tutorial_scRNAseq.ipynb>`_
- `Workflow for analysis of scATAC-Seq count matrices <https://colab.research.google.com/github//parashardhapola/scarf_vignettes/blob/main/basic_tutorial_scATACseq.ipynb>`_
- `Handling datasets with multiple modalities <https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/multiple_modalities.ipynb>`_
- `Cell subsampling using TopACeDo <https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/cell_subsampling_tutorial.ipynb>`_
- `Projection of cells across datasets <https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/data_projection.ipynb>`_
- `Merging datasets and partial training <https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/merging_datasets.ipynb>`_
- `Understanding how data is organized in Scarf <https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/zarr_explanation.ipynb>`_
- `Getting data in and out of Scarf <https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/download_conversion.ipynb>`_
