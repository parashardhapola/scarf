import numpy as np
import pandas as pd
import zarr
from numba import jit

from .utils import tqdmbar
from .writers import create_zarr_count_assay, create_cell_data


def get_simulated_pair_idx(
    n_obs: int,
    random_seed: int = 1,
    sim_doublet_ratio: float = 2.0,
    cell_range: int = 100,
):
    """
    sim_doublet_ratio: ratio of simulated doublets. Default 2
    """
    n_sim = int(n_obs * sim_doublet_ratio)

    rng_state = np.random.RandomState(random_seed)
    pair_1 = rng_state.randint(0, n_obs, size=n_sim)
    # If we assume that the order of cells (indices) is truly random in the source dataset ie iid
    # Then it doesn't matter what the search space for random cell is
    pair_2 = np.array(
        [
            rng_state.randint(
                max(0, x - cell_range), min(n_obs - 1, x + cell_range), size=1
            )[0]
            for x in pair_1
        ]
    )

    idx = np.array([pair_1, pair_2]).T
    idx = pd.DataFrame(idx).sort_values(by=[0, 1]).values

    return idx


def get_simulated_doublets(ds, indexes: np.ndarray):
    return ds.RNA.rawData[indexes[:, 0]] + ds.RNA.rawData[indexes[:, 1]]


def save_sim_doublets(
    simulated_doublets, assay, idx, sim_dset_path: str, rechunk=False
) -> None:
    zarr_path = zarr.open(sim_dset_path, "w")

    g = create_zarr_count_assay(
        z=zarr_path,
        assay_name=assay.name,
        workspace=None,
        chunk_size=assay.rawData.chunksize,
        n_cells=simulated_doublets.shape[0],
        feat_ids=assay.feats.fetch_all("ids"),
        feat_names=assay.feats.fetch_all("names"),
    )
    sim_cell_ids = np.array([f"Sim_{x[0]}-{x[1]}" for x in idx]).astype(object)
    create_cell_data(z=zarr_path, workspace=None, ids=sim_cell_ids, names=sim_cell_ids)
    if rechunk:
        simulated_doublets = simulated_doublets.rechunk(
            1000, simulated_doublets.shape[1]
        )

    compute_write(simulated_doublets, g)


# TODO: is there built-in scarf function to replace this
def compute_write(simulated, zarr_array):
    s, e = 0, 0
    batch = None

    # TODO: do we need tqdm here
    for i in tqdmbar(simulated.blocks, total=simulated.numblocks[0]):
        if batch is None:
            batch = i.compute()
        else:
            batch = np.vstack([batch, i.compute()])
        if len(batch) > 1000:
            e += batch.shape[0]
            zarr_array[s:e] = batch
            batch = None
            s = e

    if batch is not None:
        e += batch.shape[0]
        zarr_array[s:e] = batch

    assert e == simulated.shape[0]


@jit(cache=True, nopython=True)
def average_signal_by_neighbour(inds, data, signal, t: int):
    out = signal.copy()
    n = out.shape[0]
    for _ in range(t):
        temp = np.zeros(n)
        for i in range(n):
            neighbour_w_mean = (out[inds[i]] * data[i]).mean()
            temp[i] = (out[i] + neighbour_w_mean) / 2
        out = temp.copy()

    return out


def process_sim_ds(sim_ds):
    sim_ds.mark_hvgs(min_cells=20, top_n=500, min_mean=-3, max_mean=2, max_var=6)

    sim_ds.make_graph(
        feat_key="hvgs", k=11, dims=15, n_centroids=100, local_connectivity=0.9
    )
