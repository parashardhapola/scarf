import numpy as np
import pytest

from scarf.agent.parameter_tuning.hvg import (
    HvgGroupVariability,
    aggregate_hvg_rankings,
    effective_hvg_candidate_counts,
    run_hvg_diagnostic_artifacts,
)
from scarf.storage.artifacts import inspect_artifact


def test_hvg_candidate_counts_are_capped_and_unique() -> None:
    assert effective_hvg_candidate_counts(2500) == (1000, 2000, 2500)
    assert effective_hvg_candidate_counts(8, (3, 5, 10, 10)) == (3, 5, 8)
    with pytest.raises(ValueError, match="greater than 0"):
        effective_hvg_candidate_counts(0)


def test_batch_aware_hvg_ranking_keeps_nested_registered_candidates() -> None:
    corrected = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
    eligible = np.asarray([True, True, True, True, False])
    groups = [
        HvgGroupVariability(
            group_id="str:a",
            cell_count=10,
            corrected_variance=np.asarray([5.0, 1.0, 4.0, 3.0, 2.0]),
            detected_features=np.ones(5, dtype=bool),
        ),
        HvgGroupVariability(
            group_id="str:b",
            cell_count=10,
            corrected_variance=np.asarray([1.0, 5.0, 4.0, 3.0, 2.0]),
            detected_features=np.ones(5, dtype=bool),
        ),
    ]

    ranking = aggregate_hvg_rankings(
        corrected,
        eligible,
        groups,
        valid_group_count=2,
        candidate_targets=(2, 3),
    )

    assert ranking.ranking_mode == "batchAware"
    assert ranking.recurrence.tolist() == [1, 1, 2, 2, 0]
    narrow = ranking.candidate_mask(2)
    broad = ranking.candidate_mask(3)
    assert np.all(~narrow | broad)
    assert int(narrow.sum()) == 2
    assert int(broad.sum()) == 3


def test_hvg_diagnostic_artifacts_reuse_exact_pooled_candidates(
    datastore_ephemeral: object,
) -> None:
    store = datastore_ephemeral
    cell_selection = store.snapshot_cell_selection("I")
    all_features = store.select_all_features(from_assay="RNA")

    first = run_hvg_diagnostic_artifacts(
        store.zw,
        store.RNA,
        cell_selection=cell_selection,
        eligible_features=all_features,
        all_features=all_features,
        technical_group_column=None,
        min_group_cells=2,
        min_cells=0,
        n_bins=20,
        lowess_frac=0.2,
        invalidate_cache=False,
        candidate_targets=(3, 5),
    )
    second = run_hvg_diagnostic_artifacts(
        store.zw,
        store.RNA,
        cell_selection=cell_selection,
        eligible_features=all_features,
        all_features=all_features,
        technical_group_column=None,
        min_group_cells=2,
        min_cells=0,
        n_bins=20,
        lowess_frac=0.2,
        invalidate_cache=False,
        candidate_targets=(3, 5),
    )

    assert first == second
    assert len(first) == 1
    global_ranking = first[0]
    assert global_ranking.ranking_mode == "global"
    assert [candidate.top_n for candidate in global_ranking.candidates] == [3, 5]
    assert inspect_artifact(store.zw, global_ranking.diagnostic).operation == (
        "diagnose_hvg_candidates"
    )
    masks = [
        np.asarray(store.load_artifact(candidate.features)["values"][:], dtype=bool)
        for candidate in global_ranking.candidates
    ]
    assert int(masks[0].sum()) == 3
    assert int(masks[1].sum()) == 5
    assert np.all(~masks[0] | masks[1])


def test_hvg_diagnostics_persist_global_and_batch_aware_rankings(
    datastore_ephemeral: object,
) -> None:
    store = datastore_ephemeral
    midpoint = store.cells.N // 2
    store.cells.insert(
        "technical_batch",
        np.asarray(["a"] * midpoint + ["b"] * (store.cells.N - midpoint)),
        overwrite=True,
    )
    rankings = run_hvg_diagnostic_artifacts(
        store.zw,
        store.RNA,
        cell_selection=store.snapshot_cell_selection("I"),
        eligible_features=store.select_all_features(from_assay="RNA"),
        all_features=store.select_all_features(from_assay="RNA"),
        technical_group_column="technical_batch",
        min_group_cells=2,
        min_cells=0,
        n_bins=20,
        lowess_frac=0.2,
        invalidate_cache=False,
        candidate_targets=(3, 5),
    )

    assert [ranking.ranking_mode for ranking in rankings] == [
        "global",
        "batchAware",
    ]
    assert rankings[0].diagnostic != rankings[1].diagnostic
    assert len(rankings[1].valid_groups) == 2
    for ranking in rankings:
        group = store.load_artifact(ranking.diagnostic)
        assert group.attrs["ranking_mode"] == ranking.ranking_mode
        assert [candidate.top_n for candidate in ranking.candidates] == [3, 5]
