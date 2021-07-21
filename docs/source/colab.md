(colab)=
# Run tutorials on Google Colab

## Wish to try Scarf without installation?

Google Colab allows running Python code directly on Google's server through a notebook interface.
With the following links you can try running any of the vignettes on Colab. This is a quick way to
try Scarf without installing it.

## Before you run notebooks on Colab

Paste the following code on the top of the notebook before running any other cell on Colab notebook

    !pip install ipython-autotime
    !pip install scarf
    !pip install -U numpy scipy

Google Colab has older versions of Numpy and Scipy which are not compatible with Scarf.
Once `scipy` and `numpy` have updated you will see a `RESTART RUNTIME` button.
Click on it to activate latest versions.
Now you are free to execute rest of the notebook.

## Colab links

- [Basic workflow of Scarf using scRNA-Seq data](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/basic_tutorial_scRNAseq.ipynb)
- [Workflow for analysis of scATAC-Seq count matrices](https://colab.research.google.com/github//parashardhapola/scarf_vignettes/blob/main/basic_tutorial_scATACseq.ipynb)
- [Handling datasets with multiple modalities](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/multiple_modalities.ipynb)
- [Cell subsampling using TopACeDo](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/cell_subsampling_tutorial.ipynb)
- [Projection of cells across datasets](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/data_projection.ipynb)
- [Merging datasets and partial training](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/merging_datasets.ipynb)
- [Understanding how data is organized in Scarf](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/zarr_explanation.ipynb)
- [Getting data in and out of Scarf](https://colab.research.google.com/github/parashardhapola/scarf_vignettes/blob/main/download_conversion.ipynb)
