# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Modified file
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import make_epochs_per_sample, find_ab_params, optimize_layout
import numba
import locale
from umap.utils import tau_rand_int
from tqdm import tqdm
locale.setlocale(locale.LC_NUMERIC, "C")


__all__ = ['fit', 'transform', 'fit_transform']


def simplicial_set_embedding(
        g, embedding, n_epochs, a, b, random_seed, repulsion_strength,
        initial_alpha, negative_sample_rate, parallel):
    g.data[g.data < (g.data.max() / float(n_epochs))] = 0.0
    g.eliminate_zeros()
    epochs_per_sample = make_epochs_per_sample(g.data, n_epochs)
    head = g.row
    tail = g.col
    rng_state = check_random_state(random_seed).randint(
        np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1, 3).astype(np.int64)
    if parallel:
        embedding = optimize_layout_euclidean(
            embedding, embedding, head, tail, n_epochs,
            g.shape[1], epochs_per_sample, a, b, rng_state,
            repulsion_strength, initial_alpha, negative_sample_rate)
    else:
        embedding = optimize_layout(
            embedding, embedding, head, tail, n_epochs, g.shape[1],
            epochs_per_sample, a, b, rng_state, repulsion_strength, initial_alpha,
            negative_sample_rate, verbose=True)
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
        repulsion_strength, initial_alpha, negative_sample_rate, parallel):
    a, b = find_ab_params(spread, min_dist)
    sym_graph = fuzzy_simplicial_set(graph, set_op_mix_ratio)
    embedding = simplicial_set_embedding(
        sym_graph, embedding, n_epochs, a, b, random_seed, repulsion_strength,
        initial_alpha, negative_sample_rate, parallel)
    return embedding, a, b


def transform(graph, embedding,
              a, b, n_epochs, random_seed,
              repulsion_strength, initial_alpha, negative_sample_rate, parallel):
    return simplicial_set_embedding(
        graph, embedding, n_epochs, a, b, random_seed,
        repulsion_strength, initial_alpha, negative_sample_rate, parallel)


def fit_transform(graph, ini_embed, spread: float, min_dist: float, tx_n_epochs: int, fit_n_epochs: int,
                  random_seed: int, set_op_mix_ratio: float = 1.0, repulsion_strength: float = 1.0,
                  initial_alpha: float = 1.0, negative_sample_rate: float = 5, parallel: bool = False):
    e, a, b = fit(graph, ini_embed,
                  spread=spread, min_dist=min_dist, set_op_mix_ratio=set_op_mix_ratio,
                  n_epochs=fit_n_epochs, random_seed=random_seed, repulsion_strength=repulsion_strength,
                  initial_alpha=initial_alpha, negative_sample_rate=negative_sample_rate, parallel=parallel)
    t = transform(graph, e, a, b, n_epochs=tx_n_epochs, random_seed=random_seed,
                  repulsion_strength=repulsion_strength, initial_alpha=initial_alpha,
                  negative_sample_rate=negative_sample_rate, parallel=parallel)
    return t


@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)
    Parameters
    ----------
    val: float
        The value to be clamped.
    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.int32,
    },
)
def rdist(x, y):
    """Reduced Euclidean distance.
    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)
    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


def _optimize_layout_euclidean_single_epoch(head_embedding, tail_embedding, head, tail,
                                            n_vertices, epochs_per_sample, a, b, rng_state,
                                            gamma, dim, move_other, alpha, epochs_per_negative_sample,
                                            epoch_of_next_negative_sample, epoch_of_next_sample, n):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]
            current = head_embedding[j]
            other = tail_embedding[k]
            dist_squared = rdist(current, other)
            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0
            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha
            epoch_of_next_sample[i] += epochs_per_sample[i]
            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )
            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices
                other = tail_embedding[k]
                dist_squared = rdist(current, other)
                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0
                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 4.0
                    current[d] += grad_d * alpha
            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def optimize_layout_euclidean(head_embedding, tail_embedding, head, tail, n_epochs,
                              n_vertices, epochs_per_sample, a, b, rng_state,
                              gamma=1.0, initial_alpha=1.0, negative_sample_rate=5.0):
    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    optimize_fn = numba.njit(_optimize_layout_euclidean_single_epoch, fastmath=True)
    for n in tqdm(range(n_epochs)):
        optimize_fn(head_embedding, tail_embedding, head, tail,
                    n_vertices, epochs_per_sample, a, b, rng_state,
                    gamma, dim, move_other, alpha, epochs_per_negative_sample,
                    epoch_of_next_negative_sample, epoch_of_next_sample, n)
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
    return head_embedding
