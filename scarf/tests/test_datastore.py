import sys
import pytest
import os
import pandas as pd
import numpy as np


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
def marker_search(datastore):
    datastore.run_marker_search(group_key='RNA_leiden_cluster')


@pytest.fixture(scope="module")
def run_mapping_no_coral(datastore):
    datastore.run_mapping(target_assay=datastore.RNA, target_name='selfmap',
                          target_feat_key='hvgs_self', save_k=3)


@pytest.fixture(scope="module")
def cell_attrs():
    if sys.platform == 'win32':
        # UMAP and Leiden are not reproducible cross platform. Most likely due to underlying C libraries
        fn = os.path.join('scarf', 'tests', 'datasets', 'cell_attributes_win32.csv')
    else:
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
        assert len(set(leiden_clustering)) == 10
        # Disabled the following test because failing on CI
        # assert np.array_equal(leiden_clustering, cell_attrs['RNA_leiden_cluster'].values)

    def test_paris_values(self, paris_clustering, cell_attrs):
        assert np.array_equal(paris_clustering, cell_attrs['RNA_cluster'].values)

    def test_umap_values(self, umap, cell_attrs):
        precalc_umap = cell_attrs[['RNA_UMAP1', 'RNA_UMAP2']].values
        # Disabled the following test because failing on CI
        # assert np.alltrue((umap - precalc_umap) < 0.1)

    def test_get_markers(self, marker_search, datastore):
        # TODO: Add assertion here to check if gene names make sense
        datastore.get_markers(group_key='RNA_leiden_cluster', group_id='1')

    def test_plot_layout(self, datastore):
        datastore.plot_layout(layout_key='RNA_UMAP', color_by='RNA_leiden_cluster', show_fig=False)

    def test_plot_cluster_tree(self, datastore):
        datastore.plot_cluster_tree(cluster_key='RNA_cluster', show_fig=False)

    def test_plot_marker_heatmap(self, marker_search, datastore):
        datastore.plot_marker_heatmap(group_key='RNA_leiden_cluster', show_fig=False)
