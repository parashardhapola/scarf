"""Utility functions for features."""
import pandas as pd
import numpy as np
from typing import List, Sequence

__all__ = ["fit_lowess", "binned_sampling"]


def fit_lowess(a, b, n_bins: int, lowess_frac: float) -> np.ndarray:
    """Fits a LOWESS (Locally Weighted Scatterplot Smoothing) curve.

    Args:
        a:
        b:
        n_bins:
        lowess_frac:

    Returns:
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    stats = pd.DataFrame({"a": a, "b": b}).apply(np.log)
    bin_edges = np.histogram(stats.a, bins=n_bins)[1]
    bin_edges[-1] += 0.1  # For including last gene
    bin_idx = []
    for i in range(n_bins):
        idx = pd.Series((stats.a >= bin_edges[i]) & (stats.a < bin_edges[i + 1]))
        if sum(idx) > 0:
            bin_idx.append(list(idx[idx].index))
    bin_vals = []
    for idx in bin_idx:
        temp_stat = stats.reindex(idx)
        temp_gene = temp_stat.idxmin().b
        bin_vals.append([temp_stat.b[temp_gene], temp_stat.a[temp_gene]])
    bin_vals = np.array(bin_vals).T
    bin_cor_fac = lowess(
        bin_vals[0], bin_vals[1], return_sorted=False, frac=lowess_frac, it=100
    ).T
    fixed_var = {}
    for bcf, indices in zip(bin_cor_fac, bin_idx):
        for idx in indices:
            fixed_var[idx] = np.e ** (stats.b[idx] - bcf)
    return np.array([fixed_var[x] for x in range(len(a))])


def binned_sampling(
    values: pd.Series,
    feature_list: List[str],
    ctrl_size: int,
    n_bins: int,
    rand_seed: int,
) -> List[str]:
    """Score a set of genes [Satija15]_. The score is the average expression of
    a set of genes subtracted with the average expression of a reference set of
    genes. The reference set is randomly sampled from the `gene_pool` for each
    binned expression value.

    This reproduces the approach in Seurat [Satija15]_ and has been implemented
    for Scanpy by Davide Cittaro.

    This function is adapted from Scanpy's `score_genes`.

    Args:
        values: The values for the features.
        feature_list: The list of features to use for score calculation.
        ctrl_size: Number of reference features to be sampled from each bin.
        n_bins: Number of bins for sampling.
        rand_seed: The seed to use for the random number generation.

    Returns:
        A list of sampled features.
    """
    n_items = int(np.round(len(values) / (n_bins - 1)))
    feature_list = set(feature_list)
    # Made following more linter friendly
    # obs_cut = obs_avg.rank(method='min') // n_items
    obs_cut: pd.Series = values.fillna(0).rank(method="min").divide(n_items).astype(int)

    control_genes = set()
    for cut in np.unique(obs_cut[feature_list]):
        # Replaced np.random.shuffle with pandas' sample method
        r_genes = (
            obs_cut[obs_cut == cut].sample(n=ctrl_size, random_state=rand_seed).index
        )
        control_genes.update(set(r_genes))
    return list(control_genes - feature_list)
