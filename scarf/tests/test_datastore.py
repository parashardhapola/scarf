import tarfile
import pytest
import os
import shutil
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def pbmc_datastore(pbmc_reader):
    from ..datastore import DataStore

    fn = os.path.join('scarf', 'tests', 'datasets', '1K_pbmc_citeseq.zarr.tar.gz')
    out_fn = fn.replace('.tar.gz', '')
    if os.path.isdir(out_fn):
        shutil.rmtree(out_fn)
    tar = tarfile.open(fn, "r:gz")
    tar.extractall(out_fn)
    yield DataStore(out_fn, default_assay='RNA')
    shutil.rmtree(out_fn)

@pytest.fixture(scope="class")
def test_run_auto_filter_cells(pbmc_datastore):
    pbmc_datastore.auto_filter_cells(show_qc_plots=False)

@pytest.fixture(scope="class")
def test_run_mark_hvgs(test_run_auto_filter_cells, pbmc_datastore):
    pbmc_datastore.mark_hvgs(top_n=100, show_plot=False)

@pytest.fixture(scope="class")
def test_run_make_graph(test_run_mark_hvgs, pbmc_datastore):
    pbmc_datastore.make_graph(feat_key='hvgs')

@pytest.fixture(scope="class")
def test_run_leiden_clustering(test_run_make_graph, pbmc_datastore):
    pbmc_datastore.run_leiden_clustering()

@pytest.fixture(scope="class")
def test_run_paris_clustering(test_run_make_graph, pbmc_datastore):
    pbmc_datastore.run_clustering(n_clusters=10)

@pytest.fixture(scope="class")
def test_run_umap(test_run_make_graph, pbmc_datastore):
   pbmc_datastore.run_umap(fit_n_epochs=50, tx_n_epochs=20)

class TestDataStore:
    def test_leiden_values(self, test_run_leiden_clustering, pbmc_datastore):
        assert pbmc_datastore.cells.to_pandas_dataframe(['RNA_leiden_cluster'], key='I').nunique()[0] == 10

    def test_paris_values(self, test_run_paris_clustering, pbmc_datastore):
        assert pbmc_datastore.cells.to_pandas_dataframe(['RNA_cluster'], key='I').nunique()[0] == 10

    def test_umap_values(self, test_run_umap, pbmc_datastore):
        values = pd.DataFrame({
            'UMAP1': pbmc_datastore.cells.fetch('RNA_UMAP1'),
            'UMAP2': pbmc_datastore.cells.fetch('RNA_UMAP2')
        })
        ground_truth_values = pd.read_csv('precalc_umap.csv')
        tolerance = 0.01
        assert (np.abs(values.UMAP1 - ground_truth_values.UMAP1) < tolerance).all()
        assert (np.abs(values.UMAP2 - ground_truth_values.UMAP2) < tolerance).all()

    
