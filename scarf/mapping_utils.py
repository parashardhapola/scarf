import dask.array as daskarr
import numpy as np
from scipy.linalg import fractional_matrix_power as fmp
from .writers import dask_to_zarr
from .assay import Assay

__all__ = ['align_common_features', 'coral']


def _cov_diaged(da: daskarr) -> daskarr:
    a = daskarr.cov(da, rowvar=0)
    a[a == np.inf] = 0
    a[a == np.nan] = 0
    return a + np.eye(a.shape[0])


def _correlation_alignment(s: daskarr, t: daskarr) -> daskarr:
    s_cov = _cov_diaged(s)
    t_cov = _cov_diaged(t)
    # No dask hereforth. Hence possibly memory inefficiency.
    a_coral = np.dot(fmp(s_cov, -0.5), fmp(t_cov, 0.5))
    return daskarr.dot(s, a_coral)


def _add_feats_to_target(s_assay, t_assay, s_feat_ids):
    t_idx = t_assay.feats.table.ids.isin(s_feat_ids)
    t_idx[t_idx] = t_assay.rawData[:, list(t_idx[t_idx].index)].sum(axis=0).compute() != 0
    t_idx = t_idx[t_idx].index

    s_idx = s_assay.feats.table.ids.isin(t_assay.feats.table.ids.reindex(t_idx).values)
    s_idx = s_idx[s_idx].index

    t_idx_map = {v: k for k, v in t_assay.feats.table.ids.reindex(t_idx).to_dict().items()}
    t_re_idx = np.array([t_idx_map[x] for x in s_assay.feats.table.ids.reindex(s_idx).values])
    return s_idx, t_re_idx


def align_common_features(source_assay: Assay, target_assay: Assay, source_cell_key: str,
                          source_feat_key: str, target_feat_key: str):
    source_feat_ids = source_assay.feats.table.ids[source_assay.feats.table[
        source_cell_key + '__' + source_feat_key]].values

    s_idx, t_idx = _add_feats_to_target(source_assay, target_assay, source_feat_ids)

    normed_loc = f"{source_assay.name}/normed__{source_cell_key}__{source_feat_key}"
    norm_params = dict(zip(['log_transform', 'renormalize_subset'],
                           source_assay.z['/'][normed_loc].attrs['subset_params']))
    dask_to_zarr(target_assay.normed(target_assay.cells.active_index('I'), t_idx, **norm_params),
                 target_assay.z['/'], f"{target_assay.name}/normed__I__{target_feat_key}/data", 1000)

    x = np.zeros(source_assay.feats.N).astype(bool)
    x[s_idx] = True
    source_assay.feats.add(k='I__' + target_feat_key, v=x, fill_val=False, overwrite=True)


def coral(source_data, target_data, target_assay, target_feat_key):
    coral_target_data = _correlation_alignment((target_data - target_data.mean(axis=0)) / target_data.std(axis=0),
                                               (source_data - source_data.mean(axis=0)) / source_data.std(axis=0))
    dask_to_zarr(coral_target_data, target_assay.z['/'],
                 f"{target_assay.name}/normed__I__{target_feat_key}/data_coral", 1000)
