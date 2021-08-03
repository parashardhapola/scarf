def test_crh5reader(crh5_reader):
    assert crh5_reader.nCells == 892
    assert crh5_reader.nFeatures == 36611
    n_assay_feats = list(crh5_reader.assayFeats.T.nFeatures.values)
    assert n_assay_feats == [36601, 10]


def test_crdir_reader(crdir_reader):
    assert crdir_reader.nCells == 892
    assert crdir_reader.nFeatures == 36601  # Does not contain 10 ADTs


def test_mtx_reader(mtx_reader):
    assert mtx_reader.nCells == 892
    assert mtx_reader.nFeatures == 36601  # Does not contain 10 ADTs


def test_h5ad_reader(h5ad_reader):
    assert h5ad_reader.nCells == 3696 == len(h5ad_reader.cell_ids())
    assert h5ad_reader.nFeatures == 27998 == len(h5ad_reader.feat_names())


def test_loom_reader(loom_reader):
    assert loom_reader.nCells == 298 == len(loom_reader.cell_ids())
    assert loom_reader.nFeatures == 16892 == len(loom_reader.feature_names())
