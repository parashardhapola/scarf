import numpy as np
import pytest
from scipy.sparse import block_diag, csr_matrix

from scarf.agent.tuning_diagnostics import (
    _cross_unit_support,
    _subsample_partition_stability,
)


def test_cross_unit_support_requires_replication_per_cluster() -> None:
    labels = np.asarray([1, 1, 2, 2])
    units = np.asarray(["a", "b", "a", "a"])

    assert _cross_unit_support(labels, units) == 0.5
    assert _cross_unit_support(labels, np.asarray(["a"] * 4)) is None


def test_subsample_partition_stability_reclusters_induced_graph() -> None:
    community = csr_matrix(np.ones((8, 8)) - np.eye(8))
    graph = block_diag((community, community), format="csr")
    labels = np.asarray([1] * 8 + [2] * 8)

    assert _subsample_partition_stability(graph, labels, 0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="does not align"):
        _subsample_partition_stability(graph, labels[:-1], 0.5)
