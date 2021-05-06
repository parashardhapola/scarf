## Vignettes

#### Links to vignettes:

   - Basic workflow of Scarf: Using 10K PBMC data (CITE-Seq): [Static](https://github.com/parashardhapola/scarf_vignettes/blob/main/basic_tutorial.ipynb) | [Live](https://colab.research.google.com/github//parashardhapola/scarf_vignettes/blob/main/basic_tutorial.ipynb)
   - Workflow for working with sc-ATACSeq count matrices: [Static](https://github.com/parashardhapola/scarf_vignettes/blob/main/basic_tutorial_scATACseq.ipynb) | [Live](https://colab.research.google.com/github//parashardhapola/scarf_vignettes/blob/main/basic_tutorial_scATACseq.ipynb)
   - Performing cell subsampling with TopACeDo: [Static](https://github.com/parashardhapola/scarf_vignettes/blob/main/cell_subsampling_tutorial.ipynb) | [Live](https://colab.research.google.com/github//parashardhapola/scarf_vignettes/blob/main/cell_subsampling_tutorial.ipynb)
   - Data projection
   - Pseudotime ordering and imputation


#### When running live on Colab:

Paste the following code on the top of the notebook before running any other cell. 
```
!pip install ipython-autotime
!pip install scarf-toolkit
!pip install -U numpy scipy
```

Google Colab has older versions of Numpy and Scipy which are not compatible with Scarf.
Once `scipy` and `numpy` have updated you will see a `RESTART RUNTIME` button. Click on it to activate latest versions.
Now are free to execute rest of the notebook.
