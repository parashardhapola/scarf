import tarfile
import pytest
import os
import shutil
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def datastore():
    from ..datastore import DataStore

    fn = os.path.join('scarf', 'tests', 'datasets', '1K_pbmc_citeseq.zarr.tar.gz')
    out_fn = fn.replace('.tar.gz', '')
    if os.path.isdir(out_fn):
        shutil.rmtree(out_fn)
    tar = tarfile.open(fn, "r:gz")
    tar.extractall(out_fn)
    yield DataStore(out_fn, default_assay='RNA')
    shutil.rmtree(out_fn)


@pytest.fixture(scope="module")
def auto_filter_cells(datastore):
    datastore.auto_filter_cells(show_qc_plots=False)


@pytest.fixture(scope="module")
def mark_hvgs(auto_filter_cells, datastore):
    datastore.mark_hvgs(top_n=100, show_plot=False)


@pytest.fixture(scope="module")
def make_graph(mark_hvgs, datastore):
    datastore.make_graph(feat_key='hvgs')


@pytest.fixture(scope="module")
def leiden_clustering(make_graph, datastore):
    datastore.run_leiden_clustering()
    yield datastore.cells.fetch('RNA_leiden_cluster')


@pytest.fixture(scope="module")
def paris_clustering(make_graph, datastore):
    datastore.run_clustering(n_clusters=10)
    yield datastore.cells.fetch('RNA_cluster')


@pytest.fixture(scope="module")
def umap(make_graph, datastore):
    datastore.run_umap(fit_n_epochs=50, tx_n_epochs=20)
    yield np.array([datastore.cells.fetch('RNA_UMAP1'),
                    datastore.cells.fetch('RNA_UMAP2')]).T


@pytest.fixture(scope="module")
def graph_indices(make_graph, datastore):
    fn = os.path.join('scarf', 'tests', 'datasets', 'knn_indices.npy')
    return np.load(fn)


@pytest.fixture(scope="module")
def graph_distances(make_graph, datastore):
    fn = os.path.join('scarf', 'tests', 'datasets', 'knn_distances.npy')
    return np.load(fn)


@pytest.fixture(scope="module")
def graph_weights(make_graph, datastore):
    fn = os.path.join('scarf', 'tests', 'datasets', 'knn_weights.npy')
    return np.load(fn)


@pytest.fixture(scope="module")
def cell_attrs():
    fn = os.path.join('scarf', 'tests', 'datasets', 'cell_attributes.csv')
    return pd.read_csv(fn, index_col=0)


GRAPH_LOC = 'RNA/normed__I__hvgs/reduction__pca__11__I/ann__l2__50__50__48__4466/knn__11'


class TestDataStore:
    def test_graph_indices(self, graph_indices, datastore):
        a = datastore.z[GRAPH_LOC]['indices'][:]
        assert np.array_equal(a, graph_indices)

    def test_graph_distances(self, graph_distances, datastore):
        a = datastore.z[GRAPH_LOC]['distances'][:]
        assert np.alltrue((a - graph_distances) < 0.1)

    def test_graph_weights(self, graph_weights, datastore):
        a = datastore.z[GRAPH_LOC]['graph__1.0__1.5']['weights'][:]
        assert np.alltrue((a - graph_weights) < 0.1)

    def test_leiden_values(self, leiden_clustering, cell_attrs):
        assert np.array_equal(leiden_clustering, cell_attrs['RNA_leiden_cluster'].values)

    def test_paris_values(self, paris_clustering, cell_attrs):
        assert np.array_equal(paris_clustering, cell_attrs['RNA_cluster'].values)

    def test_umap_values(self, umap, cell_attrs):
        precalc_umap = cell_attrs[['RNA_UMAP1', 'RNA_UMAP2']].values
        assert np.alltrue((umap - precalc_umap) < 0.1)
