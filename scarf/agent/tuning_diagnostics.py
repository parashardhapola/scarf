"""Deterministic representation and partition evidence for RNA decisions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..clustering.leiden import leiden_membership
from ..metadata.rows import read_metadata_rows_chunkwise
from ..storage.arrays import create_zarr_dataset
from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ..storage.artifacts import fingerprint_stored_arrays
from ..storage.feature_selection import read_feature_selection_indices
from ..storage.refs import ArtifactRef
from ..storage.selections import read_stored_selection_indices
from ..storage.types import as_zarr_array
from .parameter_tuning import (
    ArtifactRecord,
    ParameterCandidateEvaluation,
)

_PCA_DIAGNOSTIC_ARRAYS = (
    "component_variance",
    "top_loading_feature_indices",
    "top_loading_values",
    "family_enrichment",
    "covariate_association",
    "adjacent_neighbor_overlap",
)


@dataclass(frozen=True, slots=True)
class AdvisoryDoubletScores:
    """Advisory score artifacts and their exact cell-axis selections."""

    scores: tuple[ArtifactRef, ...]
    cell_selections: tuple[ArtifactRef, ...]
    native_graph: ArtifactRef
    native_clusters: ArtifactRef
    limitations: tuple[str, ...] = ()


def _artifact_ref(
    evaluation: ParameterCandidateEvaluation,
    name: str,
) -> ArtifactRef:
    artifact = evaluation.artifacts.get(name)
    if artifact is None:
        raise ValueError(
            f"Candidate {evaluation.candidateId!r} lacks {name!r} evidence"
        )
    return ArtifactRef(
        scope=artifact.scope,
        assay=artifact.assay,
        kind=artifact.kind,
        artifact_id=artifact.artifactId,
    )


def _cluster_labels(store: Any, cluster_ref: ArtifactRef) -> np.ndarray:
    group = store.load_artifact(cluster_ref)
    values = as_zarr_array(group["values"], name="values")
    labels = np.asarray(values[:])
    if labels.ndim != 1:
        raise ValueError("Cluster evidence must be a one-dimensional label vector")
    return labels


def _selected_feature_names(
    store: Any,
    feature_selection: ArtifactRef,
) -> tuple[np.ndarray, np.ndarray]:
    if feature_selection.assay is None:
        raise ValueError("PCA feature selection must belong to one assay")
    indices = read_feature_selection_indices(
        store.zw,
        feature_selection.assay,
        feature_selection,
    ).astype(np.int64, copy=False)
    names = np.asarray(
        store.get_assay(feature_selection.assay).feats.fetch_all("names")
    ).astype(str)
    return indices, names[indices]


def _family_mask(names: np.ndarray, family: str) -> np.ndarray | None:
    upper = np.char.upper(names.astype(str))
    if family == "mitochondrial":
        return np.char.startswith(upper, "MT-")
    if family == "ribosomal":
        return np.asarray(
            np.logical_or.reduce(
                [
                    np.char.startswith(upper, prefix)
                    for prefix in ("RPS", "RPL", "MRPS", "MRPL")
                ]
            ),
            dtype=bool,
        )
    if family == "histone":
        return np.char.startswith(upper, "HIST")
    if family == "hemoglobin":
        return np.char.startswith(upper, "HB")
    if family == "immuneReceptor":
        return np.asarray(
            np.logical_or.reduce(
                [
                    np.char.startswith(upper, prefix)
                    for prefix in ("IGH", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG")
                ]
            ),
            dtype=bool,
        )
    return None


def _component_variance(values: Any) -> np.ndarray:
    if len(values.shape) != 2:
        raise ValueError("PCA coordinates must be a two-dimensional array")
    n_rows, n_components = values.shape
    if n_rows < 1 or n_components < 1:
        raise ValueError("PCA coordinates cannot be empty")
    totals = np.zeros(n_components, dtype=np.float64)
    totals_squared = np.zeros(n_components, dtype=np.float64)
    for start in range(0, n_rows, 65_536):
        block = np.asarray(values[start : start + 65_536], dtype=np.float64)
        totals += block.sum(axis=0)
        totals_squared += np.square(block).sum(axis=0)
    variance = totals_squared / n_rows - np.square(totals / n_rows)
    return np.asarray(np.maximum(variance, 0.0), dtype=np.float64)


def _top_loadings(
    loadings: np.ndarray,
    selected_indices: np.ndarray,
    family_masks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if loadings.ndim != 2 or loadings.shape[0] != len(selected_indices):
        raise ValueError("PCA loadings do not align with selected features")
    top_n = min(20, loadings.shape[0])
    top_indices = np.zeros((loadings.shape[1], top_n), dtype=np.int64)
    top_values = np.zeros((loadings.shape[1], top_n), dtype=np.float64)
    enrichment = np.zeros(
        (len(family_masks), loadings.shape[1]),
        dtype=np.float64,
    )
    for component in range(loadings.shape[1]):
        absolute = np.abs(loadings[:, component])
        order = np.lexsort((np.arange(len(absolute)), -absolute))[:top_n]
        top_indices[component] = selected_indices[order]
        top_values[component] = absolute[order]
        for family_index, mask in enumerate(family_masks.values()):
            background = float(mask.mean())
            enrichment[family_index, component] = (
                float(mask[order].mean()) / background if background > 0 else 0.0
            )
    return top_indices, top_values, enrichment


def _aligned_metadata_values(
    store: Any,
    cell_selection: ArtifactRef,
    column: str,
) -> np.ndarray:
    indices = read_stored_selection_indices(
        store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    values = np.asarray(read_metadata_rows_chunkwise(store.cells, column, indices))
    if values.shape != (len(indices),):
        raise ValueError(f"Metadata column {column!r} does not align with PCA")
    return values


def _numeric_association(coordinates: Any, values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric PCA covariates must be finite")
    n_rows, n_components = coordinates.shape
    total_x = np.zeros(n_components, dtype=np.float64)
    total_x2 = np.zeros(n_components, dtype=np.float64)
    total_xy = np.zeros(n_components, dtype=np.float64)
    total_y = float(numeric.sum())
    total_y2 = float(np.square(numeric).sum())
    for start in range(0, n_rows, 65_536):
        block = np.asarray(coordinates[start : start + 65_536], dtype=np.float64)
        y = numeric[start : start + len(block)]
        total_x += block.sum(axis=0)
        total_x2 += np.square(block).sum(axis=0)
        total_xy += (block * y[:, None]).sum(axis=0)
    numerator = n_rows * total_xy - total_x * total_y
    denominator = np.sqrt(
        np.maximum(n_rows * total_x2 - np.square(total_x), 0.0)
        * max(n_rows * total_y2 - total_y * total_y, 0.0)
    )
    return np.asarray(
        np.divide(
            np.abs(numerator),
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        ),
        dtype=np.float64,
    )


def _categorical_association(coordinates: Any, values: np.ndarray) -> np.ndarray:
    labels = values.astype(str)
    _levels, codes = np.unique(labels, return_inverse=True)
    n_rows, n_components = coordinates.shape
    totals = np.zeros(n_components, dtype=np.float64)
    totals_squared = np.zeros(n_components, dtype=np.float64)
    group_sums = np.zeros((int(codes.max()) + 1, n_components), dtype=np.float64)
    group_counts = np.bincount(codes, minlength=group_sums.shape[0]).astype(np.float64)
    for start in range(0, n_rows, 65_536):
        block = np.asarray(coordinates[start : start + 65_536], dtype=np.float64)
        block_codes = codes[start : start + len(block)]
        totals += block.sum(axis=0)
        totals_squared += np.square(block).sum(axis=0)
        np.add.at(group_sums, block_codes, block)
    grand_mean = totals / n_rows
    group_means = np.divide(
        group_sums,
        group_counts[:, None],
        out=np.zeros_like(group_sums),
        where=group_counts[:, None] > 0,
    )
    between = (
        group_counts[:, None] * np.square(group_means - grand_mean[None, :])
    ).sum(axis=0)
    total = totals_squared - n_rows * np.square(grand_mean)
    return np.asarray(
        np.sqrt(
            np.divide(
                between,
                total,
                out=np.zeros_like(between),
                where=total > 0,
            )
        ),
        dtype=np.float64,
    )


def _covariate_associations(
    store: Any,
    cell_selection: ArtifactRef,
    coordinates: Any,
    columns: Sequence[str],
) -> np.ndarray:
    associations = np.zeros((len(columns), coordinates.shape[1]), dtype=np.float64)
    for index, column in enumerate(columns):
        values = _aligned_metadata_values(store, cell_selection, column)
        if values.dtype.kind in {"i", "u", "f"} and len(np.unique(values)) > 10:
            associations[index] = _numeric_association(coordinates, values)
        else:
            associations[index] = _categorical_association(coordinates, values)
    return associations


def _neighbor_overlap(store: Any, left: ArtifactRef, right: ArtifactRef) -> float:
    left_values = as_zarr_array(store.load_artifact(left)["indices"], name="indices")
    right_values = as_zarr_array(store.load_artifact(right)["indices"], name="indices")
    if left_values.shape != right_values.shape or len(left_values.shape) != 2:
        raise ValueError("Adjacent PCA neighbor artifacts must align exactly")
    total = 0.0
    rows = left_values.shape[0]
    for start in range(0, rows, 4096):
        left_block = np.asarray(left_values[start : start + 4096])
        right_block = np.asarray(right_values[start : start + 4096])
        intersection = (
            (left_block[:, :, None] == right_block[:, None, :]).any(axis=2).sum(axis=1)
        )
        union = left_block.shape[1] + right_block.shape[1] - intersection
        total += float(np.divide(intersection, union).sum())
    return total / rows


def _write_pca_diagnostic(
    store: Any,
    evaluation: ParameterCandidateEvaluation,
    *,
    feature_selection: ArtifactRef,
    selected_indices: np.ndarray,
    family_masks: Mapping[str, np.ndarray],
    covariate_columns: Sequence[str],
    covariate_roles: Sequence[str],
    adjacent_overlap: float | None,
) -> tuple[ArtifactRef, np.ndarray, np.ndarray, np.ndarray]:
    reduction = _artifact_ref(evaluation, "pca")
    neighbors = _artifact_ref(evaluation, "neighbors")
    reduction_group = store.load_artifact(reduction)
    coordinates = as_zarr_array(reduction_group["data"], name="data")
    loadings = np.asarray(
        as_zarr_array(reduction_group["loadings"], name="loadings")[:],
        dtype=np.float64,
    )
    component_variance = _component_variance(coordinates)
    top_indices, top_values, family_enrichment = _top_loadings(
        loadings,
        selected_indices,
        family_masks,
    )
    associations = (
        _covariate_associations(
            store,
            ArtifactRef(
                scope=evaluation.cellSelection.scope,
                assay=evaluation.cellSelection.assay,
                kind=evaluation.cellSelection.kind,
                artifact_id=evaluation.cellSelection.artifactId,
            ),
            coordinates,
            covariate_columns,
        )
        if evaluation.cellSelection is not None
        else np.zeros((len(covariate_columns), coordinates.shape[1]), dtype=np.float64)
    )
    overlap_array = np.asarray(
        [np.nan if adjacent_overlap is None else adjacent_overlap],
        dtype=np.float64,
    )
    payload = {
        "component_variance": component_variance,
        "top_loading_feature_indices": top_indices,
        "top_loading_values": top_values,
        "family_enrichment": family_enrichment,
        "covariate_association": associations,
        "adjacent_neighbor_overlap": overlap_array,
    }
    planned = plan_artifact(
        store.zw,
        scope="assay",
        assay=reduction.assay,
        kind="feature_summary",
        operation="diagnose_pca_representation",
        parameters={
            "family_names": list(family_masks),
            "covariate_columns": list(covariate_columns),
            "covariate_roles": list(covariate_roles),
            "top_loading_count": top_indices.shape[1],
            "adjacent_neighbor_overlap": adjacent_overlap,
        },
        inputs={
            "reduction": reduction,
            "neighbors": neighbors,
            "feature_selection": feature_selection,
        },
        execution_options={},
        invalidate_cache=False,
        required_arrays=tuple(
            ArrayRequirement(name, shape=values.shape, dtype=values.dtype)
            for name, values in payload.items()
        ),
        required_attributes=(
            AttributeRequirement("family_names", expected_types=(list,)),
            AttributeRequirement("covariate_columns", expected_types=(list,)),
            AttributeRequirement("covariate_roles", expected_types=(list,)),
            AttributeRequirement("payload_fingerprint", expected_types=(str,)),
        ),
    )
    if not planned.reused:
        group = start_artifact(store.zw, planned)
        for name, values in payload.items():
            chunks = tuple(max(1, min(size, 4096)) for size in values.shape)
            array = create_zarr_dataset(
                group,
                name,
                chunks,
                values.dtype,
                values.shape,
            )
            array[:] = values
        group.attrs["family_names"] = list(family_masks)
        group.attrs["covariate_columns"] = list(covariate_columns)
        group.attrs["covariate_roles"] = list(covariate_roles)
        group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
            group,
            _PCA_DIAGNOSTIC_ARRAYS,
        )
        finish_artifact(group, planned)
    return planned.ref, component_variance, family_enrichment, associations


def augment_pca_evaluations(
    store: Any,
    evaluations: Sequence[ParameterCandidateEvaluation],
    *,
    feature_selection: ArtifactRef,
    nominated_families: Sequence[str],
    protected_families: Sequence[str],
    technical_columns: Sequence[str],
    protected_columns: Sequence[str],
    qc_columns: Sequence[str],
) -> tuple[ParameterCandidateEvaluation, ...]:
    """Attach persisted PCA loading, variance, topology, and covariate evidence."""
    selected_indices, selected_names = _selected_feature_names(
        store,
        feature_selection,
    )
    family_masks = {
        family: mask
        for family in dict.fromkeys([*nominated_families, *protected_families])
        for mask in [_family_mask(selected_names, family)]
        if mask is not None and bool(mask.any())
    }
    columns: list[str] = []
    roles: list[str] = []
    for role, values in (
        ("technical", technical_columns),
        ("protected", protected_columns),
        ("qc", qc_columns),
    ):
        for column in values:
            if column in store.cells.columns and column not in columns:
                columns.append(column)
                roles.append(role)
    completed = [
        evaluation
        for evaluation in evaluations
        if evaluation.status == "done"
        and evaluation.eligible
        and evaluation.cellSelection is not None
        and "pca" in evaluation.artifacts
        and "neighbors" in evaluation.artifacts
    ]
    previous_by_id: dict[str, float | None] = {}
    previous: ParameterCandidateEvaluation | None = None
    for evaluation in sorted(
        completed,
        key=lambda value: value.parameters.dimensions,
    ):
        overlap = (
            _neighbor_overlap(
                store,
                _artifact_ref(previous, "neighbors"),
                _artifact_ref(evaluation, "neighbors"),
            )
            if previous is not None
            else None
        )
        previous_by_id[evaluation.candidateId] = overlap
        previous = evaluation

    augmented: list[ParameterCandidateEvaluation] = []
    for evaluation in evaluations:
        if evaluation.candidateId not in previous_by_id:
            augmented.append(evaluation)
            continue
        diagnostic, variance, family_enrichment, associations = _write_pca_diagnostic(
            store,
            evaluation,
            feature_selection=feature_selection,
            selected_indices=selected_indices,
            family_masks=family_masks,
            covariate_columns=columns,
            covariate_roles=roles,
            adjacent_overlap=previous_by_id[evaluation.candidateId],
        )
        family_maxima = {
            family: float(family_enrichment[index].max(initial=0.0))
            for index, family in enumerate(family_masks)
        }
        role_associations = {
            role: {
                column: float(associations[index].max(initial=0.0))
                for index, (column, column_role) in enumerate(
                    zip(columns, roles, strict=True)
                )
                if column_role == role
            }
            for role in ("technical", "protected", "qc")
        }
        metrics = evaluation.metrics.model_copy(
            update={
                "componentVariance": variance.tolist(),
                "loadingFamilyEnrichment": family_maxima,
                "technicalPcaAssociation": role_associations["technical"],
                "protectedPcaAssociation": role_associations["protected"],
                "qcPcaAssociation": role_associations["qc"],
                "neighborPrefixOverlap": previous_by_id[evaluation.candidateId],
            }
        )
        artifact = ArtifactRecord.from_ref(diagnostic)
        augmented.append(
            evaluation.model_copy(
                update={
                    "metrics": metrics,
                    "artifacts": {
                        **evaluation.artifacts,
                        "representationDiagnostic": artifact,
                    },
                    "evidenceIds": list(
                        dict.fromkeys(
                            [
                                *evaluation.evidenceIds,
                                f"candidate:{evaluation.candidateId}:pcaVariance",
                                f"candidate:{evaluation.candidateId}:pcaLoadings",
                                f"candidate:{evaluation.candidateId}:pcaCovariates",
                                *(
                                    [
                                        f"candidate:{evaluation.candidateId}:neighborPrefixOverlap"
                                    ]
                                    if previous_by_id[evaluation.candidateId]
                                    is not None
                                    else []
                                ),
                            ]
                        )
                    ),
                }
            )
        )
    return tuple(augmented)


def _select_capture_cells(
    store: Any,
    parent: ArtifactRef,
    *,
    column: str,
    value: str,
    active_indices: np.ndarray,
    active_values: np.ndarray,
) -> tuple[ArtifactRef, int]:
    if active_indices.shape != active_values.shape:
        raise ValueError("Capture values must align with the selected cells")
    labels = active_values.astype(str)
    selected = labels == value
    expected = np.zeros(store.cells.N, dtype=bool)
    expected[active_indices] = selected
    reference = store.filter_cells(
        [column],
        [value],
        [value],
        cell_selection=parent,
        keep_bounds=True,
        invalidate_cache=False,
    )
    stored_array = cast(Any, store.load_artifact(reference)["values"])
    stored = np.asarray(stored_array[:], dtype=bool)
    if not np.array_equal(stored, expected):
        raise RuntimeError("Capture selection does not match its validated evidence")
    return reference, int(stored.sum())


def resolve_native_doublet_inputs(
    store: Any,
    selected: ParameterCandidateEvaluation,
    evaluations: Sequence[ParameterCandidateEvaluation],
) -> tuple[ArtifactRef, ArtifactRef]:
    """Return or reconstruct the parameter-matched uncorrected graph and clusters."""
    parameters = selected.parameters
    exact_native = next(
        (
            evaluation
            for evaluation in evaluations
            if evaluation.status == "done"
            and evaluation.eligible
            and not evaluation.parameters.useHarmony
            and evaluation.parameters.dimensions == parameters.dimensions
            and evaluation.parameters.neighborsK == parameters.neighborsK
            and evaluation.parameters.leidenResolution == parameters.leidenResolution
            and "clusters" in evaluation.artifacts
            and "connectivityMap" in evaluation.artifacts
        ),
        None,
    )
    if exact_native is not None:
        return (
            _artifact_ref(exact_native, "clusters"),
            _artifact_ref(exact_native, "connectivityMap"),
        )
    if not selected.parameters.useHarmony:
        return (
            _artifact_ref(selected, "clusters"),
            _artifact_ref(selected, "connectivityMap"),
        )
    reduction = _artifact_ref(selected, "pca")
    ann = store.build_ann_index(
        reduction,
        ann_metric="l2",
        ann_parallel=False,
        rand_state=4444,
        invalidate_cache=False,
    )
    neighbors = store.query_neighbors(
        ann,
        coordinates=reduction,
        k=parameters.neighborsK,
        invalidate_cache=False,
    )
    graph = store.build_connectivity_map(
        neighbors,
        local_connectivity=1.0,
        bandwidth=1.5,
        invalidate_cache=False,
    )
    clusters = store.run_leiden_clustering(
        graph,
        resolution=parameters.leidenResolution,
        backend="igraph",
        symmetric_graph=False,
        graph_upper_only=False,
        random_seed=4444,
        invalidate_cache=False,
    )
    return clusters, graph


def score_advisory_doublets(
    store: Any,
    selected: ParameterCandidateEvaluation,
    evaluations: Sequence[ParameterCandidateEvaluation],
    *,
    assay: str,
    feature_selection: ArtifactRef,
    capture_column: str | None,
) -> AdvisoryDoubletScores:
    """Score doublet evidence without making singlet or removal decisions."""
    if selected.cellSelection is None:
        raise ValueError("Doublet scoring requires an exact selected cell axis")
    parent_selection = ArtifactRef(
        scope=selected.cellSelection.scope,
        assay=selected.cellSelection.assay,
        kind=selected.cellSelection.kind,
        artifact_id=selected.cellSelection.artifactId,
    )
    native_clusters, native_graph = resolve_native_doublet_inputs(
        store,
        selected,
        evaluations,
    )
    limitations: list[str] = []
    if capture_column is None or capture_column not in store.cells.columns:
        score = store.run_doublet_detection(
            native_clusters,
            native_graph,
            from_assay=assay,
            invalidate_cache=False,
        )
        limitations.append(
            "Physical capture identity was unavailable, so advisory doublet "
            "scores were computed across the selected dataset."
        )
        return AdvisoryDoubletScores(
            scores=(score,),
            cell_selections=(parent_selection,),
            native_graph=native_graph,
            native_clusters=native_clusters,
            limitations=tuple(limitations),
        )

    active_indices = read_stored_selection_indices(
        store.zw,
        parent_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    capture_values = read_metadata_rows_chunkwise(
        store.cells,
        capture_column,
        active_indices,
    )
    capture_groups = sorted(set(capture_values.astype(str).tolist()))
    if len(capture_groups) == 1:
        score = store.run_doublet_detection(
            native_clusters,
            native_graph,
            from_assay=assay,
            invalidate_cache=False,
        )
        return AdvisoryDoubletScores(
            scores=(score,),
            cell_selections=(parent_selection,),
            native_graph=native_graph,
            native_clusters=native_clusters,
        )

    n_features = len(read_feature_selection_indices(store.zw, assay, feature_selection))
    scores: list[ArtifactRef] = []
    selections: list[ArtifactRef] = []
    for capture_value in capture_groups:
        capture_selection, capture_cells = _select_capture_cells(
            store,
            parent_selection,
            column=capture_column,
            value=capture_value,
            active_indices=active_indices,
            active_values=capture_values,
        )
        dimensions = min(
            selected.parameters.dimensions,
            capture_cells - 1,
            n_features - 1,
        )
        neighbors_k = min(selected.parameters.neighborsK, capture_cells - 1)
        if dimensions < 2 or neighbors_k < 2:
            limitations.append(
                "Advisory doublet scores were not computed for capture "
                f"{capture_value!r} because it contains only {capture_cells} "
                "selected cells."
            )
            continue
        normalized = store.run_normalization(
            capture_selection,
            features=feature_selection,
            log_transform=True,
            renormalize_subset=True,
            invalidate_cache=False,
        )
        reduction = store.run_pca(
            normalized,
            dims=dimensions,
            feat_scaling=True,
            invalidate_cache=False,
        )
        ann = store.build_ann_index(
            reduction,
            ann_metric="l2",
            ann_parallel=False,
            rand_state=4444,
            invalidate_cache=False,
        )
        neighbors = store.query_neighbors(
            ann,
            coordinates=reduction,
            k=neighbors_k,
            invalidate_cache=False,
        )
        graph = store.build_connectivity_map(
            neighbors,
            local_connectivity=1.0,
            bandwidth=1.5,
            invalidate_cache=False,
        )
        clusters = store.run_leiden_clustering(
            graph,
            resolution=selected.parameters.leidenResolution,
            backend="igraph",
            symmetric_graph=False,
            graph_upper_only=False,
            random_seed=4444,
            invalidate_cache=False,
        )
        scores.append(
            store.run_doublet_detection(
                clusters,
                graph,
                from_assay=assay,
                invalidate_cache=False,
            )
        )
        selections.append(capture_selection)
    if not scores:
        raise ValueError("No physical capture had enough cells for doublet scoring")
    return AdvisoryDoubletScores(
        scores=tuple(scores),
        cell_selections=tuple(selections),
        native_graph=native_graph,
        native_clusters=native_clusters,
        limitations=tuple(limitations),
    )


def _doublet_concentration(
    store: Any,
    labels: np.ndarray,
    cell_selection: ArtifactRef,
    evidence: AdvisoryDoubletScores,
) -> float | None:
    parent_indices = read_stored_selection_indices(
        store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    if labels.shape != parent_indices.shape:
        raise ValueError("Cluster labels do not align with advisory doublet evidence")
    positions = {int(value): index for index, value in enumerate(parent_indices)}
    high_score = np.zeros(len(parent_indices), dtype=bool)
    covered = np.zeros(len(parent_indices), dtype=bool)
    for score_ref, selection_ref in zip(
        evidence.scores,
        evidence.cell_selections,
        strict=True,
    ):
        score_values = np.asarray(
            as_zarr_array(store.load_artifact(score_ref)["values"], name="values")[:],
            dtype=np.float64,
        )
        selection_indices = read_stored_selection_indices(
            store.zw,
            selection_ref,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        if score_values.shape != selection_indices.shape:
            raise ValueError("Doublet scores do not align with their cell selection")
        local_positions = np.asarray(
            [positions[int(value)] for value in selection_indices],
            dtype=np.int64,
        )
        threshold = float(np.quantile(score_values, 0.9))
        covered[local_positions] = True
        high_score[local_positions] = score_values >= threshold
    if not covered.any() or not high_score[covered].any():
        return None
    baseline = float(high_score[covered].mean())
    enrichments = [
        float(high_score[covered & (labels == cluster)].mean()) / baseline
        for cluster in np.unique(labels[covered])
        if bool((covered & (labels == cluster)).any())
    ]
    return max(enrichments, default=0.0)


def _aligned_metadata(
    store: Any,
    cell_selection: ArtifactRef,
    column: str,
) -> np.ndarray:
    indices = read_stored_selection_indices(
        store.zw,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    values = np.asarray(read_metadata_rows_chunkwise(store.cells, column, indices))
    if values.shape != (len(indices),):
        raise ValueError(f"Metadata column {column!r} does not align with clusters")
    return values.astype(str)


def _cross_unit_support(labels: np.ndarray, units: np.ndarray) -> float | None:
    available_units = np.unique(units)
    if len(available_units) < 2:
        return None
    supported = [
        len(np.unique(units[labels == cluster])) >= 2 for cluster in np.unique(labels)
    ]
    return float(np.mean(supported)) if supported else None


def _subsample_partition_stability(
    graph: Any,
    labels: np.ndarray,
    resolution: float,
) -> float:
    if graph.shape != (len(labels), len(labels)):
        raise ValueError("Candidate graph does not align with cluster labels")
    selected = np.arange(len(labels)) % 5 != 0
    if int(selected.sum()) < 3:
        selected = np.ones(len(labels), dtype=bool)
    subsample_labels = leiden_membership(
        graph[selected][:, selected],
        resolution,
        4444,
        backend="igraph",
    )
    return float(adjusted_rand_score(labels[selected], subsample_labels))


def augment_cluster_evaluations(
    store: Any,
    evaluations: Sequence[ParameterCandidateEvaluation],
    *,
    marker_assay: str,
    marker_features: ArtifactRef,
    independent_unit_columns: Sequence[str],
    technical_columns: Sequence[str],
    nominated_families: Sequence[str] = (),
    protected_families: Sequence[str] = (),
    doublet_evidence: AdvisoryDoubletScores | None = None,
) -> tuple[ParameterCandidateEvaluation, ...]:
    """Add seed, marker, unit-support, and technical-association evidence."""
    _marker_indices, marker_feature_names = _selected_feature_names(
        store,
        marker_features,
    )
    family_masks = {
        family: mask
        for family in dict.fromkeys([*nominated_families, *protected_families])
        for mask in [_family_mask(marker_feature_names, family)]
        if mask is not None and bool(mask.any())
    }
    augmented: list[ParameterCandidateEvaluation] = []
    for evaluation in evaluations:
        if evaluation.status != "done" or evaluation.cellSelection is None:
            augmented.append(evaluation)
            continue
        graph_ref = _artifact_ref(evaluation, "connectivityMap")
        clusters_ref = _artifact_ref(evaluation, "clusters")
        labels = _cluster_labels(store, clusters_ref)
        alternative_ref = store.run_leiden_clustering(
            graph_ref,
            resolution=evaluation.parameters.leidenResolution,
            backend="igraph",
            symmetric_graph=False,
            graph_upper_only=False,
            random_seed=9173,
            invalidate_cache=False,
        )
        alternative = _cluster_labels(store, alternative_ref)
        if alternative.shape != labels.shape:
            raise ValueError("Alternate-seed clusters do not align with the candidate")
        seed_stability = float(adjusted_rand_score(labels, alternative))
        graph = store.load_graph(graph_ref)
        subsample_stability = _subsample_partition_stability(
            graph,
            labels,
            evaluation.parameters.leidenResolution,
        )

        marker_ref = store.run_marker_search(
            clusters_ref,
            from_assay=marker_assay,
            features=marker_features,
            invalidate_cache=False,
        )
        markers = store.get_markers(
            marker_ref,
            min_score=0.25,
            min_frac_exp=0.2,
        )
        marker_groups = (
            set(markers["group_id"].astype(str))
            if "group_id" in markers.columns
            else set()
        )
        cluster_count = len(np.unique(labels))
        marker_coherence = (
            float(len(marker_groups) / cluster_count) if cluster_count else 0.0
        )
        marker_names = (
            markers["feature_name"].astype(str).to_numpy()
            if "feature_name" in markers.columns
            else np.asarray([], dtype=str)
        )
        marker_family_enrichment: dict[str, float] = {}
        protected_marker_families: list[str] = []
        for family, mask in family_masks.items():
            marker_mask = _family_mask(marker_names, family)
            marker_fraction = (
                float(marker_mask.mean())
                if marker_mask is not None and marker_mask.size
                else 0.0
            )
            background = float(mask.mean())
            enrichment = marker_fraction / background if background > 0 else 0.0
            if family in nominated_families:
                marker_family_enrichment[family] = enrichment
            if family in protected_families and marker_fraction > 0:
                protected_marker_families.append(family)

        selection_ref = ArtifactRef(
            scope=evaluation.cellSelection.scope,
            assay=evaluation.cellSelection.assay,
            kind=evaluation.cellSelection.kind,
            artifact_id=evaluation.cellSelection.artifactId,
        )
        doublet_concentration = (
            _doublet_concentration(
                store,
                labels,
                selection_ref,
                doublet_evidence,
            )
            if doublet_evidence is not None
            else None
        )
        unit_scores = [
            score
            for column in independent_unit_columns
            if column in store.cells.columns
            for score in [
                _cross_unit_support(
                    labels,
                    _aligned_metadata(store, selection_ref, column),
                )
            ]
            if score is not None
        ]
        cross_unit_support = min(unit_scores) if unit_scores else None
        technical_association = {
            column: float(
                normalized_mutual_info_score(
                    labels,
                    _aligned_metadata(store, selection_ref, column),
                )
            )
            for column in technical_columns
            if column in store.cells.columns
        }

        metrics = evaluation.metrics.model_copy(
            update={
                "seedStability": seed_stability,
                "subsampleStability": subsample_stability,
                "markerCoherence": marker_coherence,
                "crossUnitSupport": cross_unit_support,
                "technicalAssociation": technical_association,
                "markerFamilyEnrichment": marker_family_enrichment,
                "protectedMarkerFamilies": protected_marker_families,
                "doubletHighScoreConcentration": doublet_concentration,
            }
        )
        evidence_ids = [
            *evaluation.evidenceIds,
            f"candidate:{evaluation.candidateId}:seedStability",
            f"candidate:{evaluation.candidateId}:subsampleStability",
            f"candidate:{evaluation.candidateId}:markerCoherence",
            f"candidate:{evaluation.candidateId}:markerFamilies",
            *(
                [f"candidate:{evaluation.candidateId}:crossUnitSupport"]
                if cross_unit_support is not None
                else []
            ),
            *[
                f"candidate:{evaluation.candidateId}:technicalAssociation:{column}"
                for column in technical_association
            ],
            *(
                [f"candidate:{evaluation.candidateId}:doubletConcentration"]
                if doublet_concentration is not None
                else []
            ),
        ]
        artifacts = {
            **evaluation.artifacts,
            "stabilityClusters": ArtifactRecord.from_ref(alternative_ref),
            "markerTable": ArtifactRecord.from_ref(marker_ref),
            **(
                {
                    f"doubletScore:{index}": ArtifactRecord.from_ref(score)
                    for index, score in enumerate(doublet_evidence.scores)
                }
                if doublet_evidence is not None
                else {}
            ),
            **(
                {
                    "doubletNativeGraph": ArtifactRecord.from_ref(
                        doublet_evidence.native_graph
                    ),
                    "doubletNativeClusters": ArtifactRecord.from_ref(
                        doublet_evidence.native_clusters
                    ),
                }
                if doublet_evidence is not None
                else {}
            ),
        }
        augmented.append(
            evaluation.model_copy(
                update={
                    "metrics": metrics,
                    "evidenceIds": list(dict.fromkeys(evidence_ids)),
                    "artifacts": artifacts,
                    "warnings": list(
                        dict.fromkeys(
                            [
                                *evaluation.warnings,
                                *(
                                    doublet_evidence.limitations
                                    if doublet_evidence is not None
                                    else ()
                                ),
                            ]
                        )
                    ),
                }
            )
        )
    return tuple(augmented)


__all__ = [
    "AdvisoryDoubletScores",
    "augment_cluster_evaluations",
    "augment_pca_evaluations",
    "resolve_native_doublet_inputs",
    "score_advisory_doublets",
]
