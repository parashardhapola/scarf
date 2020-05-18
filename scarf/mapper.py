import pandas as pd
import numpy as np
from .writers import create_zarr_dataset

__all__ = ['Mapper']


class Mapper():
    def __init__(self, ref, target, feat_key, name, assay_z, chunk_size, save_k = None):
        self.ref = ref
        self.target = target
        self.refFeatKey = feat_key
        self.saveK = save_k
        self.targetFeatKey = name + '_' + feat_key
        self.featIds = self.add_target_feats()
        # TODO: renormalize the data to account for missing feats
        self.data = self.target.select_and_normalize(
            cell_key='I', feat_key=self.targetFeatKey, batch_size=1000)
        self.zi, self.zd = self.prep_store(assay_z, name, chunk_size)

    def add_target_feats(self):
        feat_ids = self.ref.feats.table.ids[
            self.ref.feats.table[self.refFeatKey]].values
        self.target.feats.add(k=self.targetFeatKey,
                              v=self.target.feats.table.ids.isin(feat_ids).values,
                              fill_val=False, overwrite=True)
        return feat_ids
    
    def prep_store(self, assay_z, name, chunk_size):
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
            raise ValueError("No common featrures found between the two datasets")
        else:
            print (f"INFO: {len(colnames)} common features from {len(self.featIds)} "
                    "reference features will be used")
        for i in self.data.blocks:
            yield pd.DataFrame(i.compute(), columns=colnames).T.reindex(
                self.featIds).fillna(0).T.values

    def run(self):
        entry_start = 0
        for i in self.aligned_feat_data():
            ki, kd = self.ref.annObj.transform_ann(self.ref.annObj.reducer(i))
            entry_end = entry_start + len(ki)
            self.zi[entry_start:entry_end, :] = ki[:, :self.saveK]
            self.zd[entry_start:entry_end, :] = kd[:, :self.saveK]
            entry_start = entry_end
        return None
