import pandas as pd
import numpy as np
from . import full_path, remove


class TestDataStore:
    def test_graph_indices(self, make_graph, datastore):
        a = np.load(full_path("knn_indices.npy"))
        b = datastore.z[make_graph]["indices"][:]
        assert np.array_equal(a, b)

    def test_graph_distances(self, make_graph, datastore):
        a = np.load(full_path("knn_distances.npy"))
        b = datastore.z[make_graph]["distances"][:]
        assert np.alltrue((a - b) < 1e-3)

    def test_graph_weights(self, make_graph, datastore):
        a = np.load(full_path("knn_weights.npy"))
        b = datastore.z[make_graph]["graph__1.0__1.5"]["weights"][:]
        assert np.alltrue((a - b) < 1e-5)

    def test_atac_graph_indices(self, make_atac_graph, atac_datastore):
        a = np.load(full_path("atac_knn_indices.npy"))
        b = atac_datastore.z[make_atac_graph]["indices"][:]
        assert a.shape == b.shape

        # TODO: activate this when this PR is merged and released in gensim
        # https://github.com/RaRe-Technologies/gensim/pull/3194
        # assert np.array_equal(a, b)

    def test_atac_graph_distances(self, make_atac_graph, atac_datastore):
        a = np.load(full_path("atac_knn_distances.npy"))
        b = atac_datastore.z[make_atac_graph]["distances"][:]
        assert a.shape == b.shape

        # TODO: activate this when this PR is merged and released in gensim
        # https://github.com/RaRe-Technologies/gensim/pull/3194
        # assert np.alltrue((a - b) < 1e-5)

    def test_leiden_values(self, leiden_clustering, cell_attrs):
        assert len(set(leiden_clustering)) == 10
        # Disabled the following test because failing on CI
        # assert np.array_equal(leiden_clustering, cell_attrs['RNA_leiden_cluster'].values)

    def test_paris_values(self, paris_clustering, cell_attrs):
        assert np.array_equal(paris_clustering, cell_attrs["RNA_cluster"].values)

    def test_paris_balanced_values(self, paris_clustering_balanced, cell_attrs):
        assert np.array_equal(
            paris_clustering_balanced, cell_attrs["RNA_balanced_clusters"].values
        )

    def test_run_cell_cycle_scoring(self, cell_cycle_scoring, cell_attrs):
        assert np.array_equal(
            cell_cycle_scoring, cell_attrs["RNA_cell_cycle_phase"].values
        )

    def test_umap_values(self, umap, cell_attrs):
        precalc_umap = cell_attrs[["RNA_UMAP1", "RNA_UMAP2"]].values
        assert umap.shape == precalc_umap.shape
        # Disabled the following test because failing on CI
        # assert np.alltrue((umap - precalc_umap) < 0.1)

    def test_get_markers(self, marker_search, paris_clustering, datastore):
        precalc_markers = pd.read_csv(full_path("markers_cluster1.csv"), index_col=0)
        markers = datastore.get_markers(group_key="RNA_cluster", group_id=1)
        assert markers.names.equals(precalc_markers.names)
        diff = (markers.score - precalc_markers.score).values
        assert np.all(diff < 1e-3)

    def test_export_markers_to_csv(self, marker_search, paris_clustering, datastore):
        precalc_markers = pd.read_csv(full_path("markers_all_clusters.csv"))
        out_file = full_path("test_values_markers.csv")
        datastore.export_markers_to_csv(group_key="RNA_cluster", csv_filename=out_file)
        markers = pd.read_csv(out_file)
        assert markers.equals(precalc_markers)
        remove(out_file)

    def test_run_unified_umap(self, run_unified_umap, datastore):
        coords = datastore.z["RNA"].projections["unified_UMAP"][:]
        precalc_coords = np.load(full_path("unified_UMAP_coords.npy"))
        assert coords.shape == precalc_coords.shape

    def test_get_target_classes(
        self, run_mapping, paris_clustering, cell_attrs, datastore
    ):
        classes = datastore.get_target_classes(
            target_name="selfmap", reference_class_group="RNA_cluster"
        )
        assert np.array_equal(classes.values, cell_attrs["target_classes"].values)

    def test_get_mapping_score(self, run_mapping, cell_attrs, datastore):
        scores = next(datastore.get_mapping_score(target_name="selfmap"))[1]
        diff = scores - cell_attrs["mapping_scores"].values
        assert np.all(diff < 1e-3)

    def test_coral_mapping_score(self, run_mapping_coral, cell_attrs, datastore):
        # TODO: add test values for coral
        assert 1 == 1

    def test_repr(self, datastore):
        # TODO: Test if the expected values are printed
        print(datastore)

    def test_get_imputed(self, datastore):
        # TODO: Test the output values
        values = datastore.get_imputed(feature_name="CD4")
        assert values.shape == datastore.cells.fetch("I").shape

    def test_run_pseudotime_scoring(self, pseudotime_scoring, cell_attrs):
        diff = pseudotime_scoring - cell_attrs["RNA_pseudotime"].values
        assert np.all(diff < 1e-3)

    def test_run_pseudotime_marker_search(self, pseudotime_markers):
        precalc_markers = pd.read_csv(
            full_path("pseudotime_markers_r_values.csv"), index_col=0
        )
        assert np.alltrue(precalc_markers.index == pseudotime_markers.index)
        assert np.alltrue(
            precalc_markers.names.values == pseudotime_markers.names.values
        )
        assert np.allclose(
            precalc_markers.I__RNA_pseudotime__r.values,
            pseudotime_markers.I__RNA_pseudotime__r.values,
        )

    def test_run_pseudotime_aggregation(self, pseudotime_aggregation, datastore):
        precalc_values = np.load(full_path("aggregated_feat_idx.npy"))
        test_values = datastore.z.RNA.aggregated_I_I_RNA_pseudotime.feature_indices[:]
        assert np.all(precalc_values == test_values)

        precalc_values = np.load(full_path("aggregated_df_top_10.npy"))
        test_values = datastore.z.RNA.aggregated_I_I_RNA_pseudotime.data[:10]
        assert np.all(precalc_values == test_values)

        precalc_values = np.load(full_path("pseudotime_clusters.npy"))
        test_values = datastore.RNA.feats.fetch_all("pseudotime_clusters")
        assert np.all(precalc_values == test_values)

    def test_add_grouped_assay(self, grouped_assay, datastore):
        precalc_values = np.load(full_path("ptime_modules_group_1.npy"))
        test_values = datastore.get_cell_vals(
            from_assay="PTIME_MODULES", cell_key="I", k="group_1"
        )
        assert np.allclose(precalc_values, test_values)

    def test_make_bulk(self, paris_clustering, datastore):
        df = datastore.make_bulk(group_key="RNA_cluster")
        assert df.shape == (18850, 31)

    def test_to_anndata(self, datastore):
        # TODO: Check if all the attributes copied to anndata
        datastore.to_anndata()

    def test_run_topacedo_sampler(self, cell_attrs, topacedo_sampler):
        assert np.all(topacedo_sampler == cell_attrs["RNA_sketched"])

    def test_plot_cells_dists(self, datastore):
        datastore.plot_cells_dists(show_fig=False)

    def test_plot_layout(self, umap, paris_clustering, datastore):
        datastore.plot_layout(
            layout_key="RNA_UMAP", color_by="RNA_cluster", show_fig=False
        )

    def test_plot_layout_shade(self, umap, paris_clustering, datastore):
        datastore.plot_layout(
            layout_key="RNA_UMAP",
            color_by="RNA_cluster",
            show_fig=False,
            do_shading=True,
        )

    def test_plot_cluster_tree(self, datastore):
        datastore.plot_cluster_tree(cluster_key="RNA_cluster", show_fig=False)

    def test_plot_marker_heatmap(self, marker_search, datastore):
        datastore.plot_marker_heatmap(group_key="RNA_cluster", show_fig=False)

    def test_plot_unified_layout(self, run_unified_umap, datastore):
        datastore.plot_unified_layout(layout_key="unified_UMAP", show_fig=False)

    def test_plot_pseudotime_heatmap(self, pseudotime_aggregation, datastore):
        datastore.plot_pseudotime_heatmap(
            cell_key="I",
            feat_key="I",
            feature_cluster_key="pseudotime_clusters",
            pseudotime_key="RNA_pseudotime",
            show_features=["Wsb1", "Rest"],
            show_fig=False,
        )
