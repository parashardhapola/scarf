import pandas as pd
from .writers import create_zarr_dataset
from tqdm import tqdm

__all__ = ['Mapper']


class Mapper:
    def __init__(self, ref, ref_cell_key, ref_feat_key, target, name, assay_z, chunk_size,
                 reducer, ann_idx, save_k: int = None):
        self.ref = ref
        self.target = target
        self.refFeatKey = ref_cell_key
        self.saveK = save_k
        self.targetFeatKey = name + '_' + ref_feat_key
        self.featIds = self._add_target_feats()
        self.reducer = reducer
        self.annIdx = ann_idx
        # TODO: renormalize the data to account for missing feats
        # normed_data_loc =
        #
        # data = assay.save_normalized_data(cell_key, feat_key, batch_size, param_joiner('normed'),
        #                                   log_transform, renormalize_subset)

        self.data = self.target.save_normalized_data(
            cell_key='I', feat_key=self.targetFeatKey, batch_size=1000, **ref.attrs['selection_kwargs'])
        self.zi, self.zd = self._prep_store(assay_z, name, chunk_size)

    def _add_target_feats(self):
        feat_ids = self.ref.feats.table.ids[
            self.ref.feats.table[self.refFeatKey]].values
        self.target.feats.add(k=self.targetFeatKey,
                              v=self.target.feats.table.ids.isin(feat_ids).values,
                              fill_val=False, overwrite=True)
        return feat_ids
    
    def _prep_store(self, assay_z, name, chunk_size):
        z_key = f'projections/{name}'
        if z_key in assay_z:
            del assay_z[z_key]
        if self.saveK is None:
            self.saveK = self.ref.annObj.k
        nc, nk = self.target.cells.table.I.sum(), self.saveK 
        store = assay_z.create_group(z_key)
        zi = create_zarr_dataset(store, 'indices', (chunk_size,),
                                 'u8', (nc, nk))
        zd = create_zarr_dataset(store, 'distances', (chunk_size,),
                                 'f8', (nc, nk))
        return zi, zd

    def aligned_feat_data(self):
        colnames = self.target.feats.table.ids[
            self.target.feats.table[self.targetFeatKey]].values
        if len(colnames) == 0:
            raise ValueError("No common features found between the two datasets")
        else:
            print(f"INFO: {len(colnames)} common features from {len(self.featIds)} "
                  f"reference features will be used", flush=True)
        for i in self.data.blocks:
            yield pd.DataFrame(i.compute(), columns=colnames).T.reindex(
                self.featIds).fillna(0).T.values

    def run(self):
        entry_start = 0
        for i in tqdm(self.aligned_feat_data(), desc='Mapping'):
            ki, kd = self.annIdx.knn_query(self.reducer(i), k=self.saveK)
            entry_end = entry_start + len(ki)
            self.zi[entry_start:entry_end, :] = ki
            self.zd[entry_start:entry_end, :] = kd
            entry_start = entry_end
        return None
