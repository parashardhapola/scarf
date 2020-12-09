bash ../vignettes/convert_to_notebook.bash
cp ../vignettes/*.ipynb -t ./source/
make html
rm ./source/*.ipynb