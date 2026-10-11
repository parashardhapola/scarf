import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from scarf.markers import mannwhitneyu_from_ranks


@pytest.mark.parametrize(
    "groups",
    [
        np.array(["a", "a", "a", "b", "b", "b"]),
        np.array(["a", "a", "b", "b", "b", "b"]),
        np.array(["a", "a", "b", "b", "c", "c"]),
    ],
)
def test_mannwhitneyu_matches_scipy(groups):
    values = pd.DataFrame(
        {
            "increasing": [1, 2, 3, 4, 5, 6],
            "decreasing": [6, 5, 4, 3, 2, 1],
            "tied": [0, 0, 1, 0, 2, 2],
            "constant": [1, 1, 1, 1, 1, 1],
            "balanced": [1, 3, 5, 2, 4, 6],
            "equal_ranks": [1, 2, 3, 1, 2, 3],
        }
    )
    group_set = np.unique(groups)
    actual = mannwhitneyu_from_ranks(values.rank(), groups, group_set)
    expected = pd.DataFrame(
        [
            [
                mannwhitneyu(
                    values.loc[groups == group, feature],
                    values.loc[groups != group, feature],
                    alternative="two-sided",
                    method="asymptotic",
                ).pvalue
                for feature in values.columns
            ]
            for group in group_set
        ],
        index=group_set,
        columns=values.columns,
    )
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=0)
    if len(group_set) == 2:
        np.testing.assert_array_equal(actual.loc["a"], actual.loc["b"])


def test_mannwhitneyu_large_tie_group():
    values = np.zeros(2_100_006)
    values[-6:] = np.arange(1, 7)
    midpoint = len(values) // 2
    groups = np.repeat([0, 1], midpoint)
    ranked = pd.DataFrame({"feature": values}).rank()

    actual = mannwhitneyu_from_ranks(ranked, groups, np.array([0, 1]))
    expected = mannwhitneyu(
        values[:midpoint],
        values[midpoint:],
        alternative="two-sided",
        method="asymptotic",
    ).pvalue

    np.testing.assert_allclose(actual["feature"], expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("n_cells", [416_134, 1_000_000])
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_mannwhitneyu_large_constant_feature(n_cells):
    groups = np.repeat([0, 1], n_cells // 2)
    ranked = pd.DataFrame({"constant": np.ones(n_cells)}).rank()

    actual = mannwhitneyu_from_ranks(ranked, groups, np.array([0, 1]))

    np.testing.assert_array_equal(actual["constant"], np.ones(2))
