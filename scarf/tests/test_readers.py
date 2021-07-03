def test_read_pbmc(pbmc_reader):
    assert pbmc_reader.nCells == 892
    assert pbmc_reader.nFeatures == 36611
    n_assay_feats = list(pbmc_reader.assayFeats.T.nFeatures.values)
    assert n_assay_feats == [36601, 10]
