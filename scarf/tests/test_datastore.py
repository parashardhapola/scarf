import tarfile
import pytest
import os
import shutil


@pytest.fixture
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


def test_datastore(pbmc_datastore):
    pbmc_datastore.auto_filter_cells(show_qc_plots=False)
    pbmc_datastore.mark_hvgs(top_n=100, show_plot=False)
    pbmc_datastore.make_graph(feat_key='hvgs')
    pbmc_datastore.run_umap(fit_n_epochs=50, tx_n_epochs=20)
    pbmc_datastore.run_leiden_clustering()
    pbmc_datastore.run_clustering(n_clusters=10)
