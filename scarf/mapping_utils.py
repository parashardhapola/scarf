import dask.array as daskarr
import numpy as np
from typing import Tuple
from .assay import Assay
from .utils import controlled_compute, show_progress

__all__ = ['align_features', 'coral']


def _cov_diaged(da: daskarr) -> daskarr:
    a = daskarr.cov(da, rowvar=0)
    a[a == np.inf] = 0
    a[a == np.nan] = 0
    return a + np.eye(a.shape[0])


def _correlation_alignment(s: daskarr, t: daskarr, nthreads: int) -> daskarr:
    from scipy.linalg import fractional_matrix_power as fmp
    from threadpoolctl import threadpool_limits

    s_cov = show_progress(_cov_diaged(s), f"CORAL: Computing source covariance", nthreads)
    t_cov = show_progress(_cov_diaged(t), f"CORAL: Computing target covariance", nthreads)
    print("INFO: Calculating fractional power of covariance matrices. This might take a while... ", flush=True, end='')
    with threadpool_limits(limits=nthreads):
        a_coral = np.dot(fmp(s_cov, -0.5), fmp(t_cov, 0.5))
    print("Done", flush=True)
    return daskarr.dot(s, a_coral)


def coral(source_data, target_data, assay, feat_key: str, nthreads: int):
    from .writers import dask_to_zarr
    from .utils import clean_array

    sm = clean_array(show_progress(source_data.mean(axis=0),
                                   'INFO: (CORAL) Calculating source feature means', nthreads))
    sd = clean_array(show_progress(source_data.std(axis=0),
                                   'INFO: (CORAL) Calculating source feature stdev', nthreads), 1)
    tm = clean_array(show_progress(target_data.mean(axis=0),
                                   'INFO: (CORAL) Calculating target feature means', nthreads))
    td = clean_array(show_progress(target_data.std(axis=0),
                                   'INFO: (CORAL) Calculating target feature stdev', nthreads), 1)
    data = _correlation_alignment((source_data - sm) / sd, (target_data - tm) / td, nthreads)
    dask_to_zarr(data, assay.z['/'], f"{assay.name}/normed__I__{feat_key}/data_coral", 1000, nthreads,
                 msg="Writing out coral corrected data")


def _order_features(s_assay, t_assay, s_feat_ids: np.ndarray, filter_null: bool,
                    exclude_missing: bool, nthreads: int) -> Tuple[np.ndarray, np.ndarray]:
    t_idx = t_assay.feats.table.ids.isin(s_feat_ids)
    if filter_null:
        if exclude_missing is False:
            print("WARNING: `filter_null` has not effect because `exclude_missing` is False", flush=True)
        else:
            t_idx[t_idx] = controlled_compute(
                t_assay.rawData[:, list(t_idx[t_idx].index)][t_assay.cells.active_index('I'), :].sum(axis=0),
                nthreads) != 0
    t_idx = t_idx[t_idx].index
    if exclude_missing:
        s_idx = s_assay.feats.table.ids.isin(t_assay.feats.table.ids[t_idx].values)
    else:
        s_idx = s_assay.feats.table.ids.isin(s_feat_ids)
    s_idx = s_idx[s_idx].index
    t_idx_map = {v: k for k, v in t_assay.feats.table.ids[t_idx].to_dict().items()}
    t_re_idx = np.array([t_idx_map[x] if x in t_idx_map else -1 for x in s_assay.feats.table.ids[s_idx].values])
    if len(s_idx) != len(t_re_idx):
        raise AssertionError("ERROR: Feature ordering failed. Please report this issue. "
                             f"This is an unexpected scenario. Source has {len(s_idx)} features while target has "
                             f"{len(t_re_idx)} features")
    return s_idx.values, t_re_idx


def align_features(source_assay: Assay, target_assay: Assay, source_cell_key: str,
                   source_feat_key: str, target_feat_key: str, filter_null: bool,
                   exclude_missing: bool, nthreads: int) -> np.ndarray:
    from .writers import create_zarr_dataset
    from tqdm import tqdm

    source_feat_ids = source_assay.feats.table.ids[source_assay.feats.table[
        source_cell_key + '__' + source_feat_key]].values
    s_idx, t_idx = _order_features(source_assay, target_assay, source_feat_ids, filter_null, exclude_missing, nthreads)
    print(f"INFO: {(t_idx == -1).sum()} features missing in target data", flush=True)
    normed_loc = f"normed__{source_cell_key}__{source_feat_key}"
    norm_params = source_assay.z[normed_loc].attrs['subset_params']
    sorted_t_idx = np.array(sorted(t_idx[t_idx != -1]))
    normed_data = target_assay.normed(target_assay.cells.active_index('I'), sorted_t_idx, **norm_params)
    loc = f"{target_assay.name}/normed__I__{target_feat_key}/data"
    og = create_zarr_dataset(target_assay.z['/'], loc, (1000,), 'float64', (normed_data.shape[0], len(t_idx)))
    pos_start, pos_end = 0, 0
    unsorter_idx = np.argsort(np.argsort(t_idx[t_idx != -1]))
    for i in tqdm(normed_data.blocks, total=normed_data.numblocks[0],
                  desc=f"Writing aligned normed target data to {loc}"):
        pos_end += i.shape[0]
        a = np.ones((i.shape[0], len(t_idx)))
        a[:, np.where(t_idx != -1)[0]] = controlled_compute(i, nthreads)[:, unsorter_idx]
        og[pos_start:pos_end, :] = a
        pos_start = pos_end
    return s_idx
