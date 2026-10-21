import sys

import numpy as np
import pandas as pd
import pytest


def harmony_inputs():
    # Three well-separated cell types, spread over three shifted batches
    rng = np.random.default_rng(0)
    cell_types = np.repeat([0, 1, 2], 30)
    centers = np.array([[0, 0, 0, 0], [10, 0, 0, 0], [0, 10, 0, 0]], dtype=float)
    batches = pd.DataFrame({"batch": rng.choice(["a", "b", "c"], cell_types.size)})
    shifts = batches["batch"].map({"a": 0.0, "b": 1.5, "c": -1.5}).to_numpy()
    data = centers[cell_types] + shifts[:, None] + rng.normal(scale=0.5, size=(90, 4))
    return data.T, batches


def test_run_harmony_values():
    from ..harmony import run_harmony

    data, batches = harmony_inputs()
    corrected = run_harmony(data, batches)
    means = pd.DataFrame(corrected.T).groupby(batches["batch"].to_numpy()).mean()
    # Batch means from Scarf's own copy of Harmony, which harmonypy replaced
    np.testing.assert_allclose(
        means.to_numpy(),
        [
            [4.1379818704, 3.0491853419, 0.0229670732, -0.0131465323],
            [3.2475417026, 3.3034803212, 0.179513129, 0.1262123236],
            [2.7801583742, 3.0352016185, -0.059180436, -0.117505863],
        ],
        atol=1e-8,
    )


def test_run_harmony_without_harmonypy(monkeypatch):
    from ..harmony import run_harmony

    monkeypatch.setitem(sys.modules, "harmonypy", None)
    with pytest.raises(ImportError, match=r"pip install scarf\[harmony\]"):
        run_harmony(*harmony_inputs())


def test_make_graph_harmonize_without_harmonypy(target_datastore, monkeypatch):
    ds = target_datastore
    ds.mark_hvgs(top_n=100, show_plot=False)
    ds.cells.insert("batch", np.arange(ds.cells.N) % 2, overwrite=True)
    ds.cells.insert("batch2", np.arange(ds.cells.N) % 3, overwrite=True)
    ds.make_graph(feat_key="hvgs", harmonize=True, batch_columns=["batch"])

    monkeypatch.setitem(sys.modules, "harmonypy", None)
    # Harmonized data that the store holds for these batches needs no harmonypy
    ds.make_graph(feat_key="hvgs", harmonize=True, batch_columns=["batch"])
    with pytest.raises(ImportError, match=r"pip install scarf\[harmony\]"):
        ds.make_graph(feat_key="hvgs", harmonize=True, batch_columns=["batch2"])
    # The failed call leaves the store usable
    ds.make_graph(feat_key="hvgs", harmonize=True, batch_columns=["batch"])
