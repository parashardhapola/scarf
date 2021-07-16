def test_crh5reader(crh5_reader):
    assert crh5_reader.nCells == 892
    assert crh5_reader.nFeatures == 36611
    n_assay_feats = list(crh5_reader.assayFeats.T.nFeatures.values)
    assert n_assay_feats == [36601, 10]


def test_crdir_reader(crdir_reader):
    assert crdir_reader.nCells == 892
    assert crdir_reader.nFeatures == 36601  # Does not contain 10 ADTs


def test_h5ad_reader(h5ad_reader):
    assert h5ad_reader.nCells == 3696 == len(h5ad_reader.cell_ids())
    assert h5ad_reader.nFeatures == 27998 == len(h5ad_reader.feat_names())
