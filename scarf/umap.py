# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Modified file
#
# License: BSD 3 clause

import locale
locale.setlocale(locale.LC_NUMERIC, "C")


__all__ = ['fit', 'transform', 'fit_transform']


def simplicial_set_embedding(
        g, embedding, n_epochs, a, b, random_seed, gamma,
        initial_alpha, negative_sample_rate, parallel, nthreads):
    import numba
    from threadpoolctl import threadpool_limits
    from umap.umap_ import make_epochs_per_sample
    from umap.layouts import optimize_layout_euclidean
    from sklearn.utils import check_random_state
    import numpy as np

    g.data[g.data < (g.data.max() / float(n_epochs))] = 0.0
    g.eliminate_zeros()
    epochs_per_sample = make_epochs_per_sample(g.data, n_epochs)
    head = g.row
    tail = g.col
    rng_state = check_random_state(random_seed).randint(
        np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1, 3).astype(np.int64)
    # Since threadpool_limits doesnt work well with numba. We will use numba's set_num_threads to limit threads
    if numba.config.NUMBA_NUM_THREADS > nthreads:
        numba.set_num_threads(nthreads)
    with threadpool_limits(limits=nthreads):
        embedding = optimize_layout_euclidean(
            embedding, embedding, head, tail, n_epochs, g.shape[1],
            epochs_per_sample, a, b, rng_state, gamma, initial_alpha,
            negative_sample_rate, parallel=parallel, verbose=True)
    return embedding


def fuzzy_simplicial_set(g, set_op_mix_ratio):
    tg = g.transpose()
    prod = g.multiply(tg)
    res = (
        set_op_mix_ratio * (g + tg - prod)
        + (1.0 - set_op_mix_ratio) * prod
    )
    res.eliminate_zeros()
    return res.tocoo()


def fit(graph, embedding, spread, min_dist,
        set_op_mix_ratio, n_epochs, random_seed,
        repulsion_strength, initial_alpha, negative_sample_rate, parallel, nthreads):
    from umap.umap_ import find_ab_params

    a, b = find_ab_params(spread, min_dist)
    sym_graph = fuzzy_simplicial_set(graph, set_op_mix_ratio)
    embedding = simplicial_set_embedding(
        sym_graph, embedding, n_epochs, a, b, random_seed, repulsion_strength,
        initial_alpha, negative_sample_rate, parallel, nthreads)
    return embedding, a, b


def transform(graph, embedding,
              a, b, n_epochs, random_seed,
              repulsion_strength, initial_alpha, negative_sample_rate, parallel, nthreads):
    return simplicial_set_embedding(
        graph, embedding, n_epochs, a, b, random_seed,
        repulsion_strength, initial_alpha, negative_sample_rate, parallel, nthreads)


def fit_transform(graph, ini_embed, spread: float, min_dist: float, tx_n_epochs: int, fit_n_epochs: int,
                  random_seed: int, set_op_mix_ratio: float = 1.0, repulsion_strength: float = 1.0,
                  initial_alpha: float = 1.0, negative_sample_rate: float = 5, parallel: bool = False,
                  nthreads: int = 1):
    e, a, b = fit(graph, ini_embed,
                  spread=spread, min_dist=min_dist, set_op_mix_ratio=set_op_mix_ratio,
                  n_epochs=fit_n_epochs, random_seed=random_seed, repulsion_strength=repulsion_strength,
                  initial_alpha=initial_alpha, negative_sample_rate=negative_sample_rate,
                  parallel=parallel, nthreads=nthreads)
    t = transform(graph, e, a, b, n_epochs=tx_n_epochs, random_seed=random_seed,
                  repulsion_strength=repulsion_strength, initial_alpha=initial_alpha,
                  negative_sample_rate=negative_sample_rate, parallel=parallel, nthreads=nthreads)
    return t
