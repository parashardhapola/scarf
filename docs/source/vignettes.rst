=========
Vignettes
=========

.. toctree::
    :maxdepth: 2

    vignettes/basic_tutorial
    vignette/download_conversion
    vignettes/basic_tutorial_scATACseq
    vignettes/cell_subsampling_tutorial


Running notebooks live on Google Colab
--------------------------------------

Paste the following code on the top of the notebook before running any other cell. 

::

    !pip install ipython-autotime
    !pip install scarf-toolkit
    !pip install -U numpy scipy

Google Colab has older versions of Numpy and Scipy which are not compatible with Scarf.
Once `scipy` and `numpy` have updated you will see a `RESTART RUNTIME` button. Click on it to activate latest versions.
Now are free to execute rest of the notebook.

Links:

- `Basic workflow of Scarf: Using 10K PBMC data (CITE-Seq) <https://colab.research.google.com/github//parashardhapola/scarf_vignettes/blob/main/basic_tutorial.ipynb>`_
- `Getting data in and out of Scarf <https://colab.research.google.com/github//parashardhapola/scarf_vignettes/blob/main/download_conversion.ipynb)>`_
- `Workflow for analysis of sc-ATACSeq count matrices <https://colab.research.google.com/github//parashardhapola/scarf_vignettes/blob/main/basic_tutorial_scATACseq.ipynb>`_
- `Cell subsampling using TopACeDo <https://colab.research.google.com/github//parashardhapola/scarf_vignettes/blob/main/cell_subsampling_tutorial.ipynb)>`_
- Data projection
- Pseudotime ordering and imputation
