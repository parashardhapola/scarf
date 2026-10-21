import sys

import numpy as np
import pandas as pd
import pytest


def lisi_inputs():
    rng = np.random.default_rng(0)
    n_cells, k = 60, 10
    # The first neighbour of each cell is the cell itself
    indices = np.column_stack(
        [np.arange(n_cells), rng.integers(0, n_cells, size=(n_cells, k - 1))]
    )
    distances = np.column_stack(
        [np.zeros(n_cells), np.sort(rng.uniform(0.1, 2, size=(n_cells, k - 1)), axis=1)]
    )
    metadata = pd.DataFrame(
        {
            "batch": rng.choice(["a", "b"], n_cells),
            "group": rng.choice(["x", "y", "z"], n_cells),
        }
    )
    return distances, indices, metadata


def test_compute_lisi_values():
    from ..metrics import compute_lisi

    lisi = compute_lisi(*lisi_inputs(), label_colnames=["batch", "group"])
    # Scores from Scarf's own copy of the LISI calculation, which harmonypy replaced
    np.testing.assert_allclose(
        lisi[:4],
        [
            [1.9953871345, 2.4537543381],
            [1.9078710478, 2.2636513304],
            [1.1463416367, 2.251591697],
            [1.9317173482, 2.2924103756],
        ],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        lisi.sum(axis=0), [100.8081195527, 143.9920147827], atol=1e-8
    )


def test_compute_lisi_without_harmonypy(monkeypatch):
    from ..metrics import compute_lisi

    monkeypatch.setitem(sys.modules, "harmonypy", None)
    with pytest.raises(ImportError, match=r"pip install scarf\[harmony\]"):
        compute_lisi(*lisi_inputs(), label_colnames=["batch"])


def test_metric_lisi(datastore, make_graph):
    lables = np.random.randint(0, 2, datastore.cells.N)
    datastore.cells.insert(
        column_name="samples_id",
        values=lables,
        overwrite=True,
    )
    lisi = datastore.metric_lisi(
        label_colnames=["samples_id"], save_result=False, return_lisi=True
    )
    assert len(lisi[0][1]) == len(datastore.cells.active_index("I"))


def test_metric_silhouette(datastore, make_graph, leiden_clustering):
    _ = datastore.metric_silhouette()


def test_metric_integration_ari(datastore, make_graph, leiden_clustering):
    lables1 = np.random.randint(0, 2, datastore.cells.N)
    lables2 = np.random.randint(0, 2, datastore.cells.N)
    datastore.cells.insert(
        column_name="lables1",
        values=lables1,
        overwrite=True,
    )
    datastore.cells.insert(
        column_name="lables2",
        values=lables2,
        overwrite=True,
    )
    _ = datastore.metric_integration(batch_labels=["lables1", "lables2"], metric="ari")

def test_metric_integration_nmi(datastore, make_graph, leiden_clustering):
    lables1 = np.random.randint(0, 2, datastore.cells.N)
    lables2 = np.random.randint(0, 2, datastore.cells.N)
    datastore.cells.insert(
        column_name="lables1",
        values=lables1,
        overwrite=True,
    )
    datastore.cells.insert(
        column_name="lables2",
        values=lables2,
        overwrite=True,
    )
    _ = datastore.metric_integration(batch_labels=["lables1", "lables2"], metric="nmi")