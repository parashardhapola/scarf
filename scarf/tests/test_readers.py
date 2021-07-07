def test_read_pbmc(pbmc_reader):
    assert pbmc_reader.nCells == 892
    assert pbmc_reader.nFeatures == 36611
    n_assay_feats = list(pbmc_reader.assayFeats.T.nFeatures.values)
    assert n_assay_feats == [36601, 10]


def test_h5ad_reader(h5ad_reader):
    assert h5ad_reader.nCells == 3696 == len(h5ad_reader.cell_ids())
    assert h5ad_reader.nFeatures == 27998 == len(h5ad_reader.feat_names())
