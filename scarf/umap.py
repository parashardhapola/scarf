# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Modified file
#
# License: BSD 3 clause

import locale
from .utils import logger

locale.setlocale(locale.LC_NUMERIC, "C")

__all__ = ["fit_transform"]


def simplicial_set_embedding(
    g,
    embedding,
    n_epochs,
    a,
    b,
    random_seed,
    gamma,
    initial_alpha,
    negative_sample_rate,
    parallel,
    nthreads,
    verbose,
):
    import numba
    from threadpoolctl import threadpool_limits
    from umap.umap_ import make_epochs_per_sample
    from umap.layouts import optimize_layout_euclidean
    from sklearn.utils import check_random_state
    import numpy as np
    import warnings
    from .utils import tqdm_params

    # g.data[g.data < (g.data.max() / float(n_epochs))] = 0.0
    # g.eliminate_zeros()
    epochs_per_sample = make_epochs_per_sample(g.data, n_epochs)
    logger.trace("calculated epochs_per_sample")
    rng_state = (
        check_random_state(random_seed)
        .randint(np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1, 3)
        .astype(np.int64)
    )
    # Since threadpool_limits doesnt work well with numba. We will use numba's set_num_threads to limit threads
    if numba.config.NUMBA_NUM_THREADS > nthreads:
        numba.set_num_threads(nthreads)

    # tqdm will be activated if https://github.com/lmcinnes/umap/pull/739
    # is merged and when it is released
    tqdm_params = dict(tqdm_params)
    tqdm_params["desc"] = "Training UMAP"

    with threadpool_limits(limits=nthreads):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            embedding = optimize_layout_euclidean(
                embedding,
                embedding,
                g.row,
                g.col,
                n_epochs,
                g.shape[1],
                epochs_per_sample,
                a,
                b,
                rng_state,
                gamma,
                initial_alpha,
                negative_sample_rate,
                parallel=parallel,
                verbose=verbose,
                # tqdm_kwds=tqdm_params,
            )
    return embedding


# def fuzzy_simplicial_set(g, set_op_mix_ratio):
#     tg = g.transpose()
#     prod = g.multiply(tg)
#     res = set_op_mix_ratio * (g + tg - prod) + (1.0 - set_op_mix_ratio) * prod
#     res.eliminate_zeros()
#     return res.tocoo()


def fit_transform(
    graph,
    ini_embed,
    spread,
    min_dist,
    # set_op_mix_ratio,
    n_epochs,
    random_seed,
    repulsion_strength,
    initial_alpha,
    negative_sample_rate,
    parallel,
    nthreads,
    verbose,
):
    from umap.umap_ import find_ab_params

    a, b = find_ab_params(spread, min_dist)
    logger.trace("Found ab params")
    # sym_graph = fuzzy_simplicial_set(graph, set_op_mix_ratio)
    embedding = simplicial_set_embedding(
        graph,
        ini_embed,
        n_epochs,
        a,
        b,
        random_seed,
        repulsion_strength,
        initial_alpha,
        negative_sample_rate,
        parallel,
        nthreads,
        verbose,
    )
    return embedding, a, b
