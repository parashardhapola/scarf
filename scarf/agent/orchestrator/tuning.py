"""Sequential RNA parameter tuning and review stages."""

import hashlib
import io
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field
from pydantic_ai.exceptions import AgentRunError

from ...datastore.datastore import DataStore
from ...metadata.rows import read_metadata_rows_chunkwise
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_indices
from ...storage.types import as_zarr_array
from ...utils.logging import logger
from .. import record_io
from ..config.agent_exec import (
    ImageEvidence,
    ImageInputUnsupportedError,
    build_visual_evidence_prompt,
    run_agent_sync,
)
from ..decisions.kernel import DecisionEvidence, DecisionSelection, EvidenceBundle
from ..decisions.rna import (
    ClusterExecutorPayload,
    ConditionalGeneFamily,
    CorrectionLicensePayload,
    CorrectionNeedPayload,
    CorrectionOutcomeExecutorPayload,
    FeaturePolicyExecutorPayload,
    GraphExecutorPayload,
    PcaPrefixExecutorPayload,
    build_cluster_partition_decision,
    build_correction_license_decision,
    build_correction_need_decision,
    build_correction_outcome_decision,
    build_feature_policy_decision,
    build_graph_k_decision,
    build_pca_prefix_decision,
    require_option_evidence,
)
from ..experimental_context.contracts import ExperimentalContextResult
from ..experimental_context.study import StudyContract
from ..parameter_tuning.agent import ParameterTuningAgent
from ..parameter_tuning.contracts import (
    ParameterCandidateEvaluation,
    ParameterSearchPlan,
    ParameterTuningDependencies,
    ParameterTuningReport,
)
from ..parameter_tuning.diagnostics import (
    SCARF_DEFAULT_DIAGNOSTIC_FAMILIES,
    augment_cluster_evaluations,
    augment_pca_evaluations,
    restore_advisory_doublets,
    score_advisory_doublets,
)
from ..parameter_tuning.execution import (
    _metadata_column_fingerprint,
    candidate_metric_cache,
)
from ..parameter_tuning.prompts import (
    parameter_search_prompt,
    parameter_search_system_prompt,
    parameter_tuning_prompt,
    parameter_tuning_system_prompt,
)
from ..parameter_tuning.selection import (
    harmony_acceptance_gate,
    finalize_parameter_tuning_selection,
    pending_parameter_tuning_report,
    validate_parameter_tuning_report,
)
from ..parameter_tuning.sequential import (
    CorrectionNeedSelection,
    ParameterPhaseEvidence,
    ParameterPhasePlan,
    ParameterPhaseSelection,
    SequentialAssayTuningEvidence,
    SequentialRnaTuningPlanner,
    execute_parameter_phase,
    execute_sequential_refinement,
    prepare_sequential_refinement_dependencies,
    sequential_evidence_to_report,
    validate_parameter_phase_selection,
    validate_sequential_refinement_plan,
)
from ..persistence.contracts import (
    AgentInvocation,
    AgentReportReference,
    AgentWorkflowRun,
)
from ..types import AgentDataModel, ArtifactReferenceModel, ExperimentalTuningHandoff
from . import journal
from .decisions import DecisionResolution, DecisionStagesMixin
from .models import (
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    PreprocessedAssayHandoff,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    WorkflowStageName,
    artifact_model_to_ref,
)
from .preprocessing import apply_feature_policy_to_plan


def _bounded_evidence_summary(summary: str) -> str:
    """Keep prompt-facing evidence within the decision-kernel contract."""
    return summary if len(summary) <= 2_000 else f"{summary[:1_997].rstrip()}..."


def _stable_phase_evaluations(
    evaluations: Sequence[ParameterCandidateEvaluation],
) -> tuple[ParameterCandidateEvaluation, ...]:
    """Give fresh and restored evidence the same persisted mapping order."""
    return tuple(
        ParameterCandidateEvaluation.model_validate_json(
            record_io.canonical_json_bytes(evaluation.model_dump(mode="json"))
        )
        for evaluation in evaluations
    )


def _cluster_review_values(
    evaluation: ParameterCandidateEvaluation,
) -> tuple[dict[str, float], dict[str, float]]:
    metrics = evaluation.metrics
    maximize = {
        "silhouette": metrics.graphSilhouetteMedian or 0.0,
        "seedStability": metrics.seedStability or 0.0,
        "subsampleStability": metrics.subsampleStability or 0.0,
        "markerCoherence": metrics.markerCoherence or 0.0,
        "markerSpecificity": metrics.markerSpecificityMedian or 0.0,
        "crossUnitSupport": metrics.crossUnitSupport or 0.0,
        "membershipStrength": metrics.membershipStrengthMean or 0.0,
        "clusterConnectivity": metrics.clusterConnectivity or 0.0,
        "minimumClusterFraction": metrics.minClusterFraction or 0.0,
    }
    minimize = {
        "technicalAssociation": max(
            metrics.technicalAssociation.values(),
            default=0.0,
        ),
        "doubletConcentration": (
            metrics.doubletHighScoreConcentration
            if metrics.doubletHighScoreConcentration is not None
            else 1.0
        ),
        "protectedAssociation": max(
            metrics.protectedPcaAssociation.values(),
            default=0.0,
        ),
    }
    return maximize, minimize


def _changed_analysis_checkpoint(
    candidate: ParameterCandidateEvaluation,
    selected: ParameterCandidateEvaluation,
) -> (
    Literal[
        "pcaPrefix",
        "correctionOutcome",
        "graphK",
        "clusterPartition",
    ]
    | None
):
    changed = [
        checkpoint
        for checkpoint, differs in (
            (
                "pcaPrefix",
                candidate.parameters.dimensions != selected.parameters.dimensions,
            ),
            (
                "correctionOutcome",
                candidate.parameters.useHarmony != selected.parameters.useHarmony,
            ),
            (
                "graphK",
                candidate.parameters.neighborsK != selected.parameters.neighborsK,
            ),
            (
                "clusterPartition",
                candidate.parameters.leidenResolution
                != selected.parameters.leidenResolution,
            ),
        )
        if differs
    ]
    return cast(Any, changed[0]) if len(changed) == 1 else None


def _analysis_parameter_value(
    checkpoint: str,
    evaluation: ParameterCandidateEvaluation,
) -> int | float | bool:
    if checkpoint == "pcaPrefix":
        return evaluation.parameters.dimensions
    if checkpoint == "correctionOutcome":
        return evaluation.parameters.useHarmony
    if checkpoint == "graphK":
        return evaluation.parameters.neighborsK
    if checkpoint == "clusterPartition":
        return evaluation.parameters.leidenResolution
    raise ValueError(f"Unknown analysis checkpoint {checkpoint!r}")


def _dominates_analysis_choice(
    candidate: ParameterCandidateEvaluation,
    selected: ParameterCandidateEvaluation,
    *,
    tolerance: float = 0.02,
    material: float = 0.05,
) -> bool:
    """Admit only a one-checkpoint alternative with independently better evidence."""
    checkpoint = _changed_analysis_checkpoint(candidate, selected)
    if (
        candidate.candidateId == selected.candidateId
        or candidate.status != "done"
        or not candidate.eligible
        or checkpoint is None
        or (candidate.parameters.useHarmony and not selected.parameters.useHarmony)
    ):
        return False
    candidate_max, candidate_min = _cluster_review_values(candidate)
    selected_max, selected_min = _cluster_review_values(selected)
    no_worse = all(
        candidate_max[name] >= selected_max[name] - tolerance for name in candidate_max
    ) and all(
        candidate_min[name] <= selected_min[name] + tolerance for name in candidate_min
    )
    independently_better = sum(
        candidate_max[name] > selected_max[name] + material for name in candidate_max
    ) + sum(
        candidate_min[name] < selected_min[name] - material for name in candidate_min
    )
    return no_worse and independently_better >= 2


class AnalysisVisualAdjudication(AgentDataModel):
    """Bounded interpretation of the supplied analysis diagnostics."""

    status: Literal["acceptable", "concern"] = "concern"
    selectedCandidateId: str = ""
    featureLevelFindings: list[str] = Field(default_factory=list)
    rationale: str = ""


_NUMERIC_REVIEW_CANDIDATE_LIMIT = 24
_NUMERIC_REVIEW_METRICS = (
    "nClusters",
    "minClusterCells",
    "minClusterFraction",
    "graphSilhouetteMedian",
    "membershipStrengthMean",
    "membershipStrengthP10",
    "clusterConnectivity",
    "seedStability",
    "subsampleStability",
    "markerCoherence",
    "markerSpecificityMedian",
    "crossUnitSupport",
    "technicalAssociation",
    "batchMixing",
    "biologicalPreservation",
    "qcPcaAssociation",
    "doubletHighScoreConcentration",
    "doubletScoreQuantiles",
    "doubletCaptureCoverage",
    "loadingFamilyEnrichment",
    "markerFamilyEnrichment",
    "paretoOptimal",
    "dominatedByCandidateIds",
    "dominatesCandidateIds",
)


def _numeric_analysis_review_payload(
    study_objective: str,
    selected: ParameterCandidateEvaluation,
    candidates: Sequence[ParameterCandidateEvaluation],
) -> dict[str, Any]:
    comparisons = [
        candidate
        for candidate in candidates
        if candidate.candidateId != selected.candidateId
        and candidate.status == "done"
        and candidate.eligible
        and _changed_analysis_checkpoint(candidate, selected) is not None
    ]
    checkpoint_order = {
        "pcaPrefix": 0,
        "correctionOutcome": 1,
        "graphK": 2,
        "clusterPartition": 3,
    }
    comparisons.sort(
        key=lambda candidate: (
            checkpoint_order[
                cast(str, _changed_analysis_checkpoint(candidate, selected))
            ],
            float(
                _analysis_parameter_value(
                    cast(str, _changed_analysis_checkpoint(candidate, selected)),
                    candidate,
                )
            ),
            candidate.candidateId,
        )
    )

    def compact_candidate(
        candidate: ParameterCandidateEvaluation,
    ) -> dict[str, Any]:
        metrics = candidate.metrics.model_dump(mode="json", exclude_none=True)
        return {
            "candidateId": candidate.candidateId,
            "changedCheckpoint": _changed_analysis_checkpoint(candidate, selected),
            "parameters": candidate.parameters.model_dump(mode="json"),
            "effectiveDimensions": candidate.effectiveDimensions,
            "metrics": {
                name: metrics[name]
                for name in _NUMERIC_REVIEW_METRICS
                if name in metrics and metrics[name] not in ({}, [])
            },
            "warnings": list(candidate.warnings),
        }

    return {
        "studyObjective": study_objective,
        "evidenceMode": "numeric",
        "evidenceLimitation": (
            "The configured model did not accept image input. Spatial and visual "
            "distribution patterns are unavailable for this review."
        ),
        "selectedCandidate": {
            "candidateId": selected.candidateId,
            "parameters": selected.parameters.model_dump(mode="json"),
            "effectiveDimensions": selected.effectiveDimensions,
            "metrics": selected.metrics.model_dump(mode="json", exclude_none=True),
            "warnings": list(selected.warnings),
        },
        "comparisonCandidates": [
            compact_candidate(candidate)
            for candidate in comparisons[:_NUMERIC_REVIEW_CANDIDATE_LIMIT]
        ],
        "comparisonCandidateCount": len(comparisons),
        "includedComparisonCandidateCount": min(
            len(comparisons),
            _NUMERIC_REVIEW_CANDIDATE_LIMIT,
        ),
    }


def _run_analysis_adjudication(
    *,
    model: Any,
    config: AutomatedWorkflowConfig,
    study_objective: str,
    selected: ParameterCandidateEvaluation,
    candidates: Sequence[ParameterCandidateEvaluation],
    visual_content: Sequence[ImageEvidence],
) -> tuple[AnalysisVisualAdjudication, Literal["multimodal", "numeric"]]:
    def validate(value: AnalysisVisualAdjudication) -> AnalysisVisualAdjudication:
        if (
            value.selectedCandidateId != selected.candidateId
            or not value.rationale.strip()
        ):
            raise ValueError(
                "Analysis review must identify the exact selected candidate "
                "and provide a rationale"
            )
        return value

    visual_payload = {
        "studyObjective": study_objective,
        "selectedCandidateId": selected.candidateId,
        "metrics": selected.metrics.model_dump(mode="json"),
        "warnings": selected.warnings,
    }
    try:
        execution = run_agent_sync(
            model=model,
            output_type=AnalysisVisualAdjudication,
            system_prompt=(
                "Adjudicate the bounded diagnostic board together with the supplied "
                "exact metrics. Report only feature-level, partition-level, batch, "
                "QC, and doublet findings. Do not assign cell types. Mark concern "
                "only when an image shows a specific conflict with numeric evidence."
            ),
            user_prompt=build_visual_evidence_prompt(
                json.dumps(visual_payload, indent=2, sort_keys=True),
                visual_content,
            ),
            config=config.agentRunConfig,
            name="analysis_visual_review",
            output_validator=validate,
        )
        mode: Literal["multimodal", "numeric"] = "multimodal"
    except ImageInputUnsupportedError:
        logger.info(
            "The configured model does not accept image input; retrying analysis "
            "review with exact numeric evidence"
        )
        execution = run_agent_sync(
            model=model,
            output_type=AnalysisVisualAdjudication,
            system_prompt=(
                "Adjudicate the selected analysis using only the supplied exact "
                "numeric evidence. Evaluate feature, partition, batch, QC, and "
                "doublet measurements. Do not infer spatial patterns or cell types. "
                "Mark concern only when a supplied measurement conflicts with the "
                "selected analysis."
            ),
            user_prompt=json.dumps(
                _numeric_analysis_review_payload(
                    study_objective,
                    selected,
                    candidates,
                ),
                indent=2,
                sort_keys=True,
            ),
            config=config.agentRunConfig,
            name="analysis_numeric_review",
            output_validator=validate,
        )
        mode = "numeric"
    if not isinstance(execution.output, AnalysisVisualAdjudication):
        raise TypeError("Analysis review returned an unexpected output type")
    return execution.output, mode


def _analysis_visual_content(
    store: DataStore,
    selected: ParameterCandidateEvaluation,
    candidates: Sequence[ParameterCandidateEvaluation],
    *,
    qc_columns: Sequence[str] = (),
    qc_artifact_metrics: Sequence[tuple[str, ArtifactReferenceModel]] = (),
) -> list[ImageEvidence]:
    """Render one bounded diagnostic board for multimodal review."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Visual adjudication requires the installed plotting dependencies"
        ) from exc

    def image_content(figure: Any, identifier: str) -> ImageEvidence:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=120)
        plt.close(figure)
        return ImageEvidence(
            identifier=identifier,
            data=buffer.getvalue(),
            media_type="image/png",
        )

    def sampled_values(array: Any, selection: tuple[Any, ...]) -> np.ndarray:
        return np.asarray(
            cast(Any, as_zarr_array(array, name="diagnostic")).get_orthogonal_selection(
                selection
            )
        )

    coordinate_name = (
        "harmony"
        if selected.parameters.useHarmony
        else selected.parameters.reductionMethod
    )
    coordinate_record = selected.artifacts.get(coordinate_name)
    cluster_record = selected.artifacts.get("clusters")
    if coordinate_record is None or cluster_record is None:
        raise ValueError(
            "Selected candidate lacks visualizable coordinates or clusters"
        )
    coordinate_group = store.load_artifact(
        artifact_model_to_ref(
            ArtifactReferenceModel.model_validate(coordinate_record.model_dump())
        )
    )
    cluster_group = store.load_artifact(
        artifact_model_to_ref(
            ArtifactReferenceModel.model_validate(cluster_record.model_dump())
        )
    )
    coordinate_values = as_zarr_array(coordinate_group["data"], name="data")
    cluster_values = as_zarr_array(cluster_group["values"], name="values")
    if len(coordinate_values.shape) != 2 or coordinate_values.shape[1] < 2:
        raise ValueError("Selected coordinates need at least two dimensions")
    if coordinate_values.shape[0] != cluster_values.shape[0]:
        raise ValueError("Selected coordinates and clusters do not align")
    sample_count = min(5_000, coordinate_values.shape[0])
    sample_indices = np.linspace(
        0,
        coordinate_values.shape[0] - 1,
        sample_count,
        dtype=np.int64,
    )
    coordinates = sampled_values(
        coordinate_values,
        (sample_indices, slice(0, 2)),
    ).astype(np.float32, copy=False)
    labels = sampled_values(cluster_values, (sample_indices,))
    _, label_codes = np.unique(labels, return_inverse=True)

    comparison = max(
        (
            candidate
            for candidate in candidates
            if candidate.candidateId != selected.candidateId
            and candidate.status == "done"
            and candidate.eligible
            and candidate.parameters.dimensions == selected.parameters.dimensions
            and candidate.parameters.neighborsK == selected.parameters.neighborsK
            and candidate.parameters.useHarmony == selected.parameters.useHarmony
        ),
        key=lambda value: (
            value.metrics.markerCoherence or 0.0,
            value.metrics.seedStability or 0.0,
        ),
        default=None,
    )

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    variance = np.asarray(selected.metrics.componentVariance, dtype=float)
    if variance.size:
        axes[0, 0].plot(np.arange(1, variance.size + 1), variance, marker=".")
        axes[0, 0].set_title("PCA explained variance")
        axes[0, 0].set_xlabel("Component")
    else:
        axes[0, 0].text(0.5, 0.5, "Variance unavailable", ha="center")
        axes[0, 0].set_axis_off()

    axes[0, 1].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=label_codes,
        cmap="tab20",
        s=2,
        alpha=0.65,
        rasterized=True,
    )
    axes[0, 1].set_title(f"Selected partition: {selected.candidateId}")

    if comparison is not None and "clusters" in comparison.artifacts:
        comparison_group = store.load_artifact(
            artifact_model_to_ref(
                ArtifactReferenceModel.model_validate(
                    comparison.artifacts["clusters"].model_dump()
                )
            )
        )
        comparison_labels = sampled_values(
            comparison_group["values"],
            (sample_indices,),
        )
        _, comparison_codes = np.unique(
            comparison_labels,
            return_inverse=True,
        )
        axes[0, 2].scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=comparison_codes,
            cmap="tab20",
            s=2,
            alpha=0.65,
            rasterized=True,
        )
        axes[0, 2].set_title(f"Matched alternative: {comparison.candidateId}")
    else:
        axes[0, 2].text(0.5, 0.5, "No matched alternative", ha="center")
        axes[0, 2].set_axis_off()

    completed = [
        candidate
        for candidate in candidates
        if candidate.status == "done"
        and candidate.eligible
        and candidate.parameters.dimensions == selected.parameters.dimensions
        and candidate.parameters.neighborsK == selected.parameters.neighborsK
        and candidate.parameters.useHarmony == selected.parameters.useHarmony
    ]
    resolutions = [value.parameters.leidenResolution for value in completed]
    axes[1, 0].plot(
        resolutions,
        [value.metrics.markerCoherence or 0.0 for value in completed],
        marker="o",
        label="marker coherence",
    )
    axes[1, 0].plot(
        resolutions,
        [value.metrics.seedStability or 0.0 for value in completed],
        marker="o",
        label="seed stability",
    )
    axes[1, 0].plot(
        resolutions,
        [value.metrics.crossUnitSupport or 0.0 for value in completed],
        marker="o",
        label="cross-unit support",
    )
    axes[1, 0].set_title("Partition evidence")
    axes[1, 0].set_xlabel("Leiden resolution")
    axes[1, 0].legend(fontsize=8)

    doublet_records = {
        name.removeprefix("doubletScore:"): value
        for name, value in selected.artifacts.items()
        if name.startswith("doubletScore:")
    }
    doublet_selections = {
        name.removeprefix("doubletCellSelection:"): value
        for name, value in selected.artifacts.items()
        if name.startswith("doubletCellSelection:")
    }
    doublet_sample = np.full(sample_count, np.nan, dtype=np.float64)
    doublet_hist_values: list[np.ndarray] = []
    if doublet_records:
        if selected.cellSelection is None:
            raise ValueError("Doublet visuals require an exact cell selection")
        parent_selection = artifact_model_to_ref(selected.cellSelection)
        parent_indices = read_stored_selection_indices(
            store.zw,
            parent_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        sampled_global_indices = parent_indices[sample_indices]
        for score_index, record in sorted(doublet_records.items()):
            score_group = store.load_artifact(
                artifact_model_to_ref(
                    ArtifactReferenceModel.model_validate(record.model_dump())
                )
            )
            score_values = as_zarr_array(score_group["values"], name="values")
            selection_record = doublet_selections.get(score_index)
            if selection_record is None:
                if score_values.shape != parent_indices.shape:
                    raise ValueError(
                        "Doublet score artifact lacks its exact cell selection"
                    )
                local_positions = sample_indices
                matched_positions = np.arange(sample_count, dtype=np.int64)
            else:
                score_selection = artifact_model_to_ref(
                    ArtifactReferenceModel.model_validate(selection_record.model_dump())
                )
                score_indices = read_stored_selection_indices(
                    store.zw,
                    score_selection,
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                ).astype(np.int64, copy=False)
                if score_values.shape != score_indices.shape:
                    raise ValueError(
                        "Doublet scores do not align with their cell selection"
                    )
                candidate_positions = np.searchsorted(
                    score_indices,
                    sampled_global_indices,
                )
                within = candidate_positions < len(score_indices)
                matched = np.zeros(sample_count, dtype=bool)
                matched[within] = (
                    score_indices[candidate_positions[within]]
                    == sampled_global_indices[within]
                )
                matched_positions = np.flatnonzero(matched)
                local_positions = candidate_positions[matched]
            sampled_scores = sampled_values(score_values, (local_positions,)).astype(
                np.float64,
                copy=False,
            )
            doublet_sample[matched_positions] = sampled_scores
            doublet_hist_values.append(sampled_scores)
        finite_doublets = np.concatenate(doublet_hist_values)
        finite_doublets = finite_doublets[np.isfinite(finite_doublets)]
        axes[1, 1].hist(finite_doublets, bins=40)
        axes[1, 1].set_title("Advisory doublet scores")
    else:
        axes[1, 1].text(0.5, 0.5, "Doublet scores unavailable", ha="center")
        axes[1, 1].set_axis_off()

    family_values = {
        **selected.metrics.loadingFamilyEnrichment,
        **selected.metrics.markerFamilyEnrichment,
    }
    if family_values:
        ordered = sorted(
            family_values.items(),
            key=lambda item: (-item[1], item[0]),
        )[:12]
        axes[1, 2].barh(
            [name for name, _value in reversed(ordered)],
            [value for _name, value in reversed(ordered)],
        )
        axes[1, 2].set_title("Feature-family enrichment")
    else:
        axes[1, 2].text(0.5, 0.5, "Family evidence unavailable", ha="center")
        axes[1, 2].set_axis_off()

    content = [image_content(figure, "analysis-overview")]

    paired_corrections: list[
        tuple[ParameterCandidateEvaluation, ParameterCandidateEvaluation]
    ] = []
    candidates_by_parameters: dict[
        tuple[str, int, int, float],
        dict[bool, ParameterCandidateEvaluation],
    ] = {}
    for candidate in candidates:
        if candidate.status != "done" or not candidate.eligible:
            continue
        key = (
            candidate.parameters.reductionMethod,
            candidate.parameters.dimensions,
            candidate.parameters.neighborsK,
            candidate.parameters.leidenResolution,
        )
        candidates_by_parameters.setdefault(key, {})[
            candidate.parameters.useHarmony
        ] = candidate
    for correction_pair in candidates_by_parameters.values():
        if False in correction_pair and True in correction_pair:
            paired_corrections.append((correction_pair[False], correction_pair[True]))
    if paired_corrections:
        native, harmony = min(
            paired_corrections,
            key=lambda pair: (
                abs(pair[0].parameters.dimensions - selected.parameters.dimensions),
                abs(pair[0].parameters.neighborsK - selected.parameters.neighborsK),
                abs(
                    pair[0].parameters.leidenResolution
                    - selected.parameters.leidenResolution
                ),
            ),
        )
        correction_figure, correction_axes = plt.subplots(
            2,
            2,
            figsize=(11, 10),
            constrained_layout=True,
        )
        batch_columns = [
            column
            for column in (
                *native.metrics.batchMixing,
                *harmony.metrics.batchMixing,
            )
            if column in store.cells.columns
        ]
        batch_codes: np.ndarray | None = None
        batch_label = "Batch unavailable"
        if batch_columns and selected.cellSelection is not None:
            parent_selection = artifact_model_to_ref(selected.cellSelection)
            parent_indices = read_stored_selection_indices(
                store.zw,
                parent_selection,
                kind="cell_selection",
                scope="datastore",
                assay=None,
                table_path="cellData",
            ).astype(np.int64, copy=False)
            batch_label = batch_columns[0]
            batch_values = read_metadata_rows_chunkwise(
                store.cells,
                batch_label,
                parent_indices[sample_indices],
            ).astype(str)
            _, batch_codes = np.unique(batch_values, return_inverse=True)
        for column_index, candidate in enumerate((native, harmony)):
            name = (
                "harmony"
                if candidate.parameters.useHarmony
                else candidate.parameters.reductionMethod
            )
            candidate_coordinate = candidate.artifacts.get(name)
            candidate_cluster = candidate.artifacts.get("clusters")
            if candidate_coordinate is None or candidate_cluster is None:
                raise ValueError("Matched correction candidate lacks visual artifacts")
            candidate_coordinate_group = store.load_artifact(
                artifact_model_to_ref(
                    ArtifactReferenceModel.model_validate(
                        candidate_coordinate.model_dump()
                    )
                )
            )
            candidate_cluster_group = store.load_artifact(
                artifact_model_to_ref(
                    ArtifactReferenceModel.model_validate(
                        candidate_cluster.model_dump()
                    )
                )
            )
            candidate_coordinates = sampled_values(
                candidate_coordinate_group["data"],
                (sample_indices, slice(0, 2)),
            )
            candidate_labels = sampled_values(
                candidate_cluster_group["values"],
                (sample_indices,),
            )
            _, candidate_codes = np.unique(
                candidate_labels,
                return_inverse=True,
            )
            title = "Harmony" if candidate.parameters.useHarmony else "Native PCA"
            correction_axes[0, column_index].scatter(
                candidate_coordinates[:, 0],
                candidate_coordinates[:, 1],
                c=candidate_codes,
                cmap="tab20",
                s=2,
                alpha=0.65,
                rasterized=True,
            )
            correction_axes[0, column_index].set_title(f"{title}, partition colors")
            if batch_codes is not None:
                correction_axes[1, column_index].scatter(
                    candidate_coordinates[:, 0],
                    candidate_coordinates[:, 1],
                    c=batch_codes,
                    cmap="tab20",
                    s=2,
                    alpha=0.65,
                    rasterized=True,
                )
                correction_axes[1, column_index].set_title(
                    f"{title}, {batch_label} colors"
                )
            else:
                correction_axes[1, column_index].text(
                    0.5,
                    0.5,
                    batch_label,
                    ha="center",
                )
                correction_axes[1, column_index].set_axis_off()
        correction_figure.suptitle(
            "Parameter-matched native and Harmony representations"
        )
        content.append(image_content(correction_figure, "native-harmony-comparison"))

    marker_record = selected.artifacts.get("markerTable")
    marker_genes = list(
        dict.fromkeys(
            gene for genes in selected.metrics.topMarkerGenes.values() for gene in genes
        )
    )[:24]
    if marker_record is not None and marker_genes:
        marker_ref = artifact_model_to_ref(
            ArtifactReferenceModel.model_validate(marker_record.model_dump())
        )
        marker_table = store.get_markers(
            marker_ref,
            min_score=0.25,
            min_frac_exp=0.2,
        )
        required_marker_columns = {"group_id", "feature_name", "score"}
        if required_marker_columns.issubset(marker_table.columns):
            marker_groups = sorted(
                selected.metrics.topMarkerGenes,
                key=lambda value: (
                    (0, int(value)) if value.lstrip("-").isdigit() else (1, value)
                ),
            )
            marker_scores = np.zeros(
                (len(marker_groups), len(marker_genes)),
                dtype=np.float64,
            )
            group_positions = {
                value: index for index, value in enumerate(marker_groups)
            }
            gene_positions = {value: index for index, value in enumerate(marker_genes)}
            for row in marker_table.itertuples(index=False):
                group = str(getattr(row, "group_id"))
                gene = str(getattr(row, "feature_name"))
                if group not in group_positions or gene not in gene_positions:
                    continue
                score = float(getattr(row, "score"))
                if np.isfinite(score):
                    marker_scores[group_positions[group], gene_positions[gene]] = max(
                        marker_scores[
                            group_positions[group],
                            gene_positions[gene],
                        ],
                        score,
                    )

            def tagged_gene(gene: str) -> str:
                upper = gene.upper()
                if upper.startswith("MT-"):
                    return f"{gene} [mitochondrial]"
                if upper.startswith(("MRPS", "MRPL")):
                    return f"{gene} [mitoribosomal]"
                if upper.startswith(("RPS", "RPL")):
                    return f"{gene} [ribosomal]"
                if upper.startswith("CCN"):
                    return f"{gene} [CCN]"
                if upper.startswith("HLA-"):
                    return f"{gene} [HLA]"
                if upper.startswith("H2-"):
                    return f"{gene} [H2]"
                if upper.startswith("HIST"):
                    return f"{gene} [histone]"
                if upper in {
                    "XIST",
                    "DDX3Y",
                    "USP9Y",
                    "EIF1AY",
                    "KDM5D",
                    "SRY",
                    "ZFY",
                    "UTY",
                    "TMSB4Y",
                    "NLGN4Y",
                }:
                    return f"{gene} [sex-linked]"
                return gene

            marker_figure, marker_axis = plt.subplots(
                figsize=(max(9, len(marker_genes) * 0.45), 6),
                constrained_layout=True,
            )
            marker_image = marker_axis.imshow(
                marker_scores,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
            )
            marker_axis.set_xticks(
                np.arange(len(marker_genes)),
                [tagged_gene(gene) for gene in marker_genes],
                rotation=75,
                ha="right",
                fontsize=8,
            )
            marker_axis.set_yticks(
                np.arange(len(marker_groups)),
                marker_groups,
            )
            marker_axis.set_xlabel("Top marker feature")
            marker_axis.set_ylabel("Cluster")
            marker_axis.set_title("Feature-level marker score heatmap")
            marker_figure.colorbar(marker_image, ax=marker_axis, label="marker score")
            content.append(image_content(marker_figure, "marker-score-heatmap"))

    available_qc_columns = [
        column
        for column in dict.fromkeys([*qc_columns, *selected.metrics.qcPcaAssociation])
        if column in store.cells.columns
    ]
    qc_sources: list[tuple[str, ArtifactReferenceModel | None]] = [
        (column, None) for column in available_qc_columns
    ]
    known_qc_names = set(available_qc_columns)
    for qc_name, reference in qc_artifact_metrics:
        if qc_name not in known_qc_names:
            qc_sources.append((qc_name, reference))
            known_qc_names.add(qc_name)
    qc_sources = qc_sources[:4]
    if qc_sources or np.isfinite(doublet_sample).any():
        diagnostic_count = len(qc_sources) + 1
        qc_figure, qc_axes = plt.subplots(
            1,
            diagnostic_count,
            figsize=(max(5, diagnostic_count * 3.2), 4),
            constrained_layout=True,
        )
        qc_axis_list = np.atleast_1d(qc_axes).tolist()
        if selected.cellSelection is None:
            raise ValueError("QC visuals require an exact cell selection")
        parent_selection = artifact_model_to_ref(selected.cellSelection)
        parent_indices = read_stored_selection_indices(
            store.zw,
            parent_selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)
        sampled_parent_indices = parent_indices[sample_indices]
        for axis, (qc_name, artifact) in zip(
            qc_axis_list,
            qc_sources,
            strict=False,
        ):
            if artifact is None:
                values = np.asarray(
                    read_metadata_rows_chunkwise(
                        store.cells,
                        qc_name,
                        sampled_parent_indices,
                    ),
                    dtype=np.float64,
                )
            else:
                artifact_ref = artifact_model_to_ref(artifact)
                artifact_group = store.load_artifact(artifact_ref)
                artifact_values = as_zarr_array(
                    artifact_group["values"],
                    name="values",
                )
                if artifact_values.shape == parent_indices.shape:
                    artifact_positions = sample_indices
                else:
                    artifact_status = store.inspect_artifact(artifact_ref)
                    raw_selection = (
                        getattr(artifact_status, "inputs", None) or {}
                    ).get("cell_selection")
                    if not isinstance(raw_selection, Mapping):
                        raise ValueError(
                            f"QC artifact {qc_name!r} lacks its cell selection"
                        )
                    artifact_selection = ArtifactRef.from_dict(dict(raw_selection))
                    artifact_indices = read_stored_selection_indices(
                        store.zw,
                        artifact_selection,
                        kind="cell_selection",
                        scope="datastore",
                        assay=None,
                        table_path="cellData",
                    ).astype(np.int64, copy=False)
                    candidate_positions = np.searchsorted(
                        artifact_indices,
                        sampled_parent_indices,
                    )
                    if np.any(
                        candidate_positions >= len(artifact_indices)
                    ) or not np.array_equal(
                        artifact_indices[candidate_positions],
                        sampled_parent_indices,
                    ):
                        raise ValueError(
                            f"QC artifact {qc_name!r} does not cover selected cells"
                        )
                    artifact_positions = candidate_positions
                values = sampled_values(
                    artifact_values,
                    (artifact_positions,),
                ).astype(np.float64, copy=False)
            finite_values = values[np.isfinite(values)]
            axis.violinplot(finite_values, showmedians=True)
            jitter = (
                (np.arange(len(finite_values), dtype=np.float64) % 23.0) - 11.0
            ) / 115.0
            axis.scatter(
                np.ones(len(finite_values)) + jitter,
                finite_values,
                s=1,
                alpha=0.15,
                rasterized=True,
            )
            axis.set_xticks([])
            axis.set_title(qc_name)
        doublet_axis = qc_axis_list[-1]
        finite_doublet_mask = np.isfinite(doublet_sample)
        if finite_doublet_mask.any():
            doublet_plot = doublet_axis.scatter(
                coordinates[finite_doublet_mask, 0],
                coordinates[finite_doublet_mask, 1],
                c=doublet_sample[finite_doublet_mask],
                cmap="magma",
                s=2,
                alpha=0.7,
                rasterized=True,
            )
            doublet_axis.set_title("Advisory doublet score")
            qc_figure.colorbar(
                doublet_plot,
                ax=doublet_axis,
                label="score",
            )
        else:
            doublet_axis.text(
                0.5,
                0.5,
                "Doublet embedding unavailable",
                ha="center",
            )
            doublet_axis.set_axis_off()
        qc_figure.suptitle("Selected-cell QC and non-removing doublet evidence")
        content.append(image_content(qc_figure, "qc-doublet-diagnostics"))
    return content


class TuningStagesMixin(DecisionStagesMixin):
    """Execute parameter searches, integration comparisons, and graph selection."""

    model: Any

    @staticmethod
    def _tuning_evidence_bundle(
        decision_id: str,
        evidence: list[DecisionEvidence],
    ) -> EvidenceBundle:
        digest = hashlib.sha256(
            record_io.canonical_json_bytes(
                [item.model_dump(mode="json") for item in evidence]
            )
        ).hexdigest()
        return EvidenceBundle(
            bundleId=f"bundle:{decision_id}:{digest[:24]}",
            decisionId=decision_id,
            evidence=evidence,
        ).with_content_sha256()

    @staticmethod
    def _evaluation_artifacts(
        evaluation: Any,
    ) -> list[ArtifactReferenceModel]:
        references: list[ArtifactReferenceModel] = []
        identities: set[tuple[str, str | None, str, str]] = set()
        for name in sorted(evaluation.artifacts):
            reference = ArtifactReferenceModel.model_validate(
                evaluation.artifacts[name].model_dump()
            )
            identity = (
                reference.scope,
                reference.assay,
                reference.kind,
                reference.artifactId,
            )
            if identity not in identities:
                identities.add(identity)
                references.append(reference)
        return references

    def _analysis_candidate_evidence(
        self,
        checkpoint: str,
        evaluation: ParameterCandidateEvaluation,
    ) -> list[DecisionEvidence]:
        metrics = evaluation.metrics
        artifacts = self._evaluation_artifacts(evaluation)
        summaries = {
            "geometric": (
                f"candidate={evaluation.candidateId}; dimensions="
                f"{evaluation.parameters.dimensions}; neighbors="
                f"{evaluation.parameters.neighborsK}; resolution="
                f"{evaluation.parameters.leidenResolution}; silhouette="
                f"{metrics.graphSilhouetteMedian}; connectivity="
                f"{metrics.clusterConnectivity}; membership="
                f"{metrics.membershipStrengthMean}; minimum cluster fraction="
                f"{metrics.minClusterFraction}."
            ),
            "technical": (
                f"batch PC association={metrics.batchPcaAssociation}; technical "
                f"PC association={metrics.technicalPcaAssociation}; QC PC "
                f"association={metrics.qcPcaAssociation}; neighbour-prefix "
                f"overlap={metrics.neighborPrefixOverlap}."
            ),
            "batchRemoval": (
                f"Harmony={evaluation.parameters.useHarmony}; batch mixing="
                f"{metrics.batchMixing}."
            ),
            "biologicalConservation": (
                f"biological preservation={metrics.biologicalPreservation}; "
                f"marker coherence={metrics.markerCoherence}; marker specificity="
                f"{metrics.markerSpecificityMedian}; cross-unit support="
                f"{metrics.crossUnitSupport}."
            ),
            "protectedVariablePreservation": (
                f"protected PC association={metrics.protectedPcaAssociation}; "
                f"protected marker families={metrics.protectedMarkerFamilies}."
            ),
            "markerCoherence": (
                f"marker coherence={metrics.markerCoherence}; specificity="
                f"{metrics.markerSpecificityMedian}; family enrichment="
                f"{metrics.markerFamilyEnrichment}."
            ),
            "resamplingStability": (
                f"seed ARI={metrics.seedStability}; subsample ARI="
                f"{metrics.subsampleStability}."
            ),
            "crossUnitSupport": (
                f"cross-unit support={metrics.crossUnitSupport}; technical "
                f"association={metrics.technicalAssociation}."
            ),
            "qualityControl": (
                f"doublet concentration={metrics.doubletHighScoreConcentration}; "
                f"doublet capture coverage={metrics.doubletCaptureCoverage}; "
                f"score quantiles={metrics.doubletScoreQuantiles}."
            ),
        }
        return [
            DecisionEvidence(
                evidenceId=(
                    f"evidence:analysisReview:{checkpoint}:"
                    f"{evaluation.candidateId}:{evidence_class}"
                ),
                evidenceClass=cast(Any, evidence_class),
                summary=_bounded_evidence_summary(summary),
                artifactReferences=artifacts,
            )
            for evidence_class, summary in summaries.items()
        ]

    @staticmethod
    def _payload_option_id(
        definition: Any,
        payload_type: type[Any],
        field_name: str,
        value: Any,
    ) -> str:
        matches = [
            option.optionId
            for option in definition.executorOptions
            if isinstance(option.payload, payload_type)
            and getattr(option.payload, field_name) == value
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Decision {definition.spec.decisionId!r} lacks one exact "
                f"{field_name!r} option for {value!r}"
            )
        return cast(str, matches[0])

    def _restore_tuning_descendants(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        study_contract: StudyContract,
        replacement: ParameterCandidateEvaluation,
        *,
        revised_checkpoint: str,
        previous_options: Mapping[str, str],
        model_name: str | None,
    ) -> str:
        """Recreate invalidated tuning decisions from one executed replacement."""
        order = (
            "pcaPrefix",
            "correctionLicense",
            "correctionNeed",
            "correctionOutcome",
            "graphK",
            "clusterPartition",
        )
        try:
            start = order.index(revised_checkpoint) + 1
        except ValueError as exc:
            raise ValueError(
                f"Unknown revised tuning checkpoint {revised_checkpoint!r}"
            ) from exc
        latest_snapshot = ""

        def resolve(
            definition: Any,
            evidence: list[DecisionEvidence],
            option_id: str,
            *,
            rule_owned: bool = False,
        ) -> None:
            nonlocal latest_snapshot
            bundle = self._tuning_evidence_bundle(
                definition.spec.decisionId,
                evidence,
            )
            if definition.spec.evidenceBundleId != bundle.bundleId:
                raise ValueError("Successor definition has stale evidence identity")
            selection = DecisionSelection(
                selectedOptionId=option_id,
                evidenceIds=[value.evidenceId for value in evidence],
                rationale=(
                    "The upstream analysis decision changed, so this descendant "
                    "was recomputed from the exact executed replacement candidate."
                ),
                confidence="medium",
            )
            resolved = self._resolve_rna_decision(
                store,
                request_record,
                definition,
                bundle,
                {},
                **(
                    {"rule_selection": selection}
                    if rule_owned
                    else {
                        "agent_selection": selection,
                        "agent_model_name": model_name,
                    }
                ),
            )
            if resolved.compiled is None or resolved.record is None:
                raise RuntimeError(
                    f"Successor {definition.spec.decisionId!r} did not resolve"
                )
            latest_snapshot = resolved.snapshotSha256

        artifacts = self._evaluation_artifacts(replacement)
        generic = self._analysis_candidate_evidence(
            "successor",
            replacement,
        )
        if "correctionLicense" in order[start:]:
            license_evidence = [
                DecisionEvidence(
                    evidenceId=("evidence:analysisSuccessor:correctionLicense:design"),
                    evidenceClass="design",
                    summary=(
                        "The validated study contract licenses correction as "
                        f"{study_contract.correctionLicense!r} with technical "
                        f"columns {study_contract.technicalBatchColumns} and "
                        f"protected columns {study_contract.protectedColumns}."
                    ),
                    artifactReferences=artifacts,
                )
            ]
            license_bundle = self._tuning_evidence_bundle(
                "correctionLicense",
                license_evidence,
            )
            license_definition = build_correction_license_decision(
                evidence_bundle_id=license_bundle.bundleId,
                license=cast(Any, study_contract.correctionLicense),
            )
            resolve(
                license_definition,
                license_evidence,
                f"correctionLicense:{study_contract.correctionLicense}",
                rule_owned=True,
            )

        need_value: Literal["needed", "notNeeded"] | None = None
        previous_need = previous_options.get("correctionNeed")
        if study_contract.correctionLicense == "safe":
            need_value = (
                "needed"
                if previous_need == "correctionNeed:needed"
                or replacement.parameters.useHarmony
                else "notNeeded"
            )
        if "correctionNeed" in order[start:] and need_value is not None:
            need_evidence = [
                value
                for value in generic
                if value.evidenceClass in {"batchRemoval", "biologicalConservation"}
            ]
            need_bundle = self._tuning_evidence_bundle(
                "correctionNeed",
                need_evidence,
            )
            need_definition = build_correction_need_decision(
                evidence_bundle_id=need_bundle.bundleId,
                license="safe",
            )
            resolve(
                need_definition,
                need_evidence,
                f"correctionNeed:{need_value}",
            )

        if "correctionOutcome" in order[start:]:
            outcome_evidence = [
                value
                for value in generic
                if value.evidenceClass
                in {
                    "batchRemoval",
                    "biologicalConservation",
                    "protectedVariablePreservation",
                }
            ]
            outcome_bundle = self._tuning_evidence_bundle(
                "correctionOutcome",
                outcome_evidence,
            )
            outcome_definition = build_correction_outcome_decision(
                evidence_bundle_id=outcome_bundle.bundleId,
                license=cast(Any, study_contract.correctionLicense),
                need=need_value,
                harmony_eligible=True,
            )
            outcome_id = (
                "correctionOutcome:acceptHarmony"
                if replacement.parameters.useHarmony
                else "correctionOutcome:retainNative"
            )
            resolve(
                outcome_definition,
                outcome_evidence,
                outcome_id,
                rule_owned=outcome_definition.spec.allowedSources == ["rule"],
            )

        if "graphK" in order[start:]:
            graph_evidence = [
                value for value in generic if value.evidenceClass == "geometric"
            ]
            graph_bundle = self._tuning_evidence_bundle("graphK", graph_evidence)
            graph_definition = build_graph_k_decision(
                evidence_bundle_id=graph_bundle.bundleId,
                n_cells=max(
                    replacement.parameters.neighborsK + 1,
                    replacement.metrics.minClusterCells or 3,
                ),
                candidate_neighbors=[replacement.parameters.neighborsK],
            )
            graph_option = self._payload_option_id(
                graph_definition,
                GraphExecutorPayload,
                "neighborsK",
                replacement.parameters.neighborsK,
            )
            resolve(
                graph_definition,
                graph_evidence,
                graph_option,
            )

        if "clusterPartition" in order[start:]:
            cluster_evidence = [
                value for value in generic if value.evidenceClass == "geometric"
            ]
            cluster_bundle = self._tuning_evidence_bundle(
                "clusterPartition",
                cluster_evidence,
            )
            resolution = replacement.parameters.leidenResolution
            known_ids = {
                0.25: "clusterResolution:veryCoarse",
                0.5: "clusterResolution:coarse",
                0.75: "clusterResolution:balanced",
                1.0: "clusterResolution:detailed",
                1.25: "clusterResolution:fine",
                1.5: "clusterResolution:veryFine",
            }
            preferred = known_ids.get(
                resolution,
                f"clusterResolution:r{str(resolution).replace('.', 'p')}",
            )
            cluster_definition = build_cluster_partition_decision(
                evidence_bundle_id=cluster_bundle.bundleId,
                metric_preferred_option_id=preferred,
                resolution_candidates=[resolution],
            )
            resolve(
                cluster_definition,
                cluster_evidence,
                preferred,
            )
        return latest_snapshot

    @staticmethod
    def _phase_from_resolution(
        plan: ParameterPhasePlan,
        evaluations: Sequence[Any],
        resolution: DecisionResolution,
        *,
        payload_field: str,
        payload_value: Any,
    ) -> ParameterPhaseEvidence:
        if resolution.compiled is None or resolution.record is None:
            pending = resolution.pending
            selection = ParameterPhaseSelection(
                phase=plan.phase,
                status="needsInput",
                rationale=(
                    pending.reason
                    if pending is not None
                    else "The registered decision is unresolved."
                ),
            )
            return validate_parameter_phase_selection(plan, evaluations, selection)
        selected = next(
            (
                evaluation
                for evaluation in evaluations
                if getattr(evaluation.parameters, payload_field) == payload_value
                and evaluation.status == "done"
                and evaluation.eligible
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                "Audited RNA decision has no eligible exact candidate execution"
            )
        selection = ParameterPhaseSelection(
            phase=plan.phase,
            status="selected",
            selectedCandidateId=selected.candidateId,
            evidenceIds=list(resolution.record.evidenceIds),
            rationale=resolution.record.rationale,
        )
        return validate_parameter_phase_selection(plan, evaluations, selection)

    def _run_sequential_rna_tuning(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        plan: AutomatedPreprocessingPlan,
        preprocessed: Sequence[PreprocessedAssayHandoff],
        experimental_handoff: ExperimentalTuningHandoff,
        study_contract: StudyContract,
        answers: Mapping[str, Any],
        prior: SequentialAssayTuningEvidence | None,
    ) -> tuple[ParameterTuningReport, SequentialAssayTuningEvidence]:
        if len(preprocessed) != 1 or plan.pairedAssays:
            raise ValueError("Decision-driven v1 tuning accepts one RNA assay only")
        handoff = preprocessed[0]
        if (
            handoff.assayType != "RNA"
            or handoff.cellSelection is None
            or handoff.normalized is None
            or handoff.graphFeatures is None
            or handoff.markerFeatures is None
        ):
            raise ValueError("Decision-driven v1 tuning requires normalized RNA")
        if prior is not None and prior.assay != handoff.assay:
            raise ValueError("Persisted sequential evidence belongs to another assay")
        selected_cells = artifact_model_to_ref(handoff.cellSelection)
        prior_phases = (
            {value.plan.phase: value for value in prior.phases}
            if prior is not None
            else {}
        )

        def phase_evaluations(
            phase_plan: ParameterPhasePlan,
            execute: Callable[[], Sequence[ParameterCandidateEvaluation]],
        ) -> tuple[ParameterCandidateEvaluation, ...]:
            persisted = prior_phases.get(phase_plan.phase)
            if persisted is None:
                return tuple(execute())
            if persisted.plan != phase_plan:
                raise ValueError(
                    f"Persisted {phase_plan.phase!r} plan differs from the "
                    "current registered plan"
                )
            for evaluation in persisted.evaluations:
                if evaluation.cellSelection is not None and (
                    artifact_model_to_ref(evaluation.cellSelection) != selected_cells
                ):
                    raise ValueError("Persisted tuning evidence uses different cells")
                for artifact in evaluation.artifacts.values():
                    status = store.inspect_artifact(artifact_model_to_ref(artifact))
                    if not status.exists or not status.complete:
                        raise ValueError(
                            "Persisted tuning evidence contains unavailable artifacts"
                        )
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing persisted "
                f"{phase_plan.phase} executor evidence"
            )
            return tuple(persisted.evaluations)

        diagnostic_batch_candidates = (
            tuple(study_contract.technicalBatchColumns)
            if study_contract.correctionLicense == "safe"
            and experimental_handoff.batchAction == "evaluateHarmony"
            else (
                (
                    study_contract.physicalCaptureColumn,
                    *study_contract.technicalBatchColumns,
                )
                if request_record.config.runConfoundedHarmonyDiagnostic
                else tuple(study_contract.technicalBatchColumns)
            )
        )
        diagnostic_batch_columns = [
            column
            for column in dict.fromkeys(diagnostic_batch_candidates)
            if column is not None
            and column in store.cells.columns
            and len(np.unique(store.cells.fetch(column, key="I"))) > 1
        ]
        selectable_harmony = bool(
            study_contract.correctionLicense == "safe"
            and experimental_handoff.batchAction == "evaluateHarmony"
            and diagnostic_batch_columns
            and sorted(diagnostic_batch_columns)
            == sorted(experimental_handoff.batchColumns)
        )
        evaluate_harmony = bool(
            diagnostic_batch_columns
            and request_record.config.maxHarmonyCandidatesPerAssay == 1
            and (
                selectable_harmony
                or request_record.config.runConfoundedHarmonyDiagnostic
            )
        )
        tuning_handoff = experimental_handoff if selectable_harmony else None
        planner = SequentialRnaTuningPlanner(
            workflow_run_id=workflow.workflowRunId,
            assay=handoff.assay,
            n_cells=handoff.nCells,
            n_features=handoff.nFeatures,
            harmony_authorized=evaluate_harmony,
            dimension_candidates=request_record.config.pcaCandidateDimensions,
            neighbor_candidates=request_record.config.graphNeighborCandidates,
            resolution_candidates=request_record.config.leidenResolutionCandidates,
        )
        phase_evidence: list[ParameterPhaseEvidence] = []
        decision_sources: dict[
            str,
            Literal["rule", "agent", "human"],
        ] = {}
        correction_need_selection: CorrectionNeedSelection | None = None

        def build_state(
            *,
            pending_resolution: DecisionResolution | None = None,
            correction_license: str = "notApplicable",
            final_candidate_id: str | None = None,
        ) -> SequentialAssayTuningEvidence:
            pending = (
                pending_resolution.pending if pending_resolution is not None else None
            )
            if pending_resolution is not None and pending is None:
                raise ValueError(
                    "Pending tuning resolution lacks pending decision data"
                )
            return SequentialAssayTuningEvidence.model_validate(
                {
                    "assay": handoff.assay,
                    "phases": [
                        value.model_dump(mode="json") for value in phase_evidence
                    ],
                    "correctionLicense": correction_license,
                    "correctionNeed": (
                        correction_need_selection.model_dump(mode="json")
                        if correction_need_selection is not None
                        else None
                    ),
                    "decisionSources": decision_sources,
                    "pendingDecisionId": (
                        pending.decisionId if pending is not None else None
                    ),
                    "pendingOptionIds": (
                        pending.offeredOptionIds if pending is not None else []
                    ),
                    "pendingEvidenceIds": (
                        pending.availableEvidenceIds if pending is not None else []
                    ),
                    "finalCandidateId": final_candidate_id,
                }
            )

        def return_pending(
            resolution: DecisionResolution,
            *,
            correction_license: str = "notApplicable",
        ) -> tuple[ParameterTuningReport, SequentialAssayTuningEvidence]:
            state = build_state(
                pending_resolution=resolution,
                correction_license=correction_license,
            )
            return (
                sequential_evidence_to_report(
                    state,
                    marker_assay=plan.markerAssay,
                ),
                state,
            )

        normalized = artifact_model_to_ref(handoff.normalized)
        assay_plan = next(
            value for value in plan.assays if value.assay == handoff.assay
        )
        nominated_families = cast(
            list[str],
            assay_plan.featureParameters.get("proposedExcludeFamilies", []),
        )
        protected_families = cast(
            list[str],
            assay_plan.featureParameters.get("protectFamilies", []),
        )
        diagnostic_families = list(
            dict.fromkeys(
                [
                    *SCARF_DEFAULT_DIAGNOSTIC_FAMILIES,
                    *nominated_families,
                ]
            )
        )
        pca_plan = planner.pca_prefix_phase()
        raw_pca = phase_evaluations(
            pca_plan,
            lambda: execute_parameter_phase(
                store,
                normalized=normalized,
                plan=pca_plan,
                batch_columns=diagnostic_batch_columns,
                preservation_columns=experimental_handoff.preservationColumns,
                experimental_handoff=tuning_handoff,
                min_cluster_cells=request_record.config.minClusterCells,
                identity_feature_limit=request_record.config.maxIdentityFeatures,
            ),
        )
        if "pcaPrefix" not in prior_phases:
            raw_pca = augment_pca_evaluations(
                store,
                raw_pca,
                feature_selection=artifact_model_to_ref(handoff.graphFeatures),
                nominated_families=diagnostic_families,
                protected_families=protected_families,
                technical_columns=diagnostic_batch_columns,
                batch_columns=diagnostic_batch_columns,
                protected_columns=study_contract.protectedColumns,
                qc_columns=[
                    column
                    for column in plan.cellQc.attributes
                    if column in store.cells.columns
                ],
            )
        pca_items: list[DecisionEvidence] = []
        pca_evaluations: list[ParameterCandidateEvaluation] = []
        eligible_pca_dimensions: list[int] = []
        pca_evidence_by_dimensions: dict[int, list[str]] = {}
        for evaluation in _stable_phase_evaluations(raw_pca):
            evidence_ids: list[str] = []
            if evaluation.status == "done" and evaluation.eligible:
                eligible_pca_dimensions.append(evaluation.parameters.dimensions)
                technical_id = f"evidence:pca:{evaluation.candidateId}:technical"
                loading_preview = {
                    component: genes[:3]
                    for component, genes in sorted(
                        evaluation.metrics.topLoadingGenes.items(),
                        key=lambda item: int(item[0].removeprefix("PC")),
                    )[:10]
                }
                cumulative_variance = (
                    evaluation.metrics.pcaCumulativeExplainedVarianceRatio[-1]
                    if evaluation.metrics.pcaCumulativeExplainedVarianceRatio
                    else None
                )
                pca_summary = _bounded_evidence_summary(
                    f"The exact PCA candidate used "
                    f"{evaluation.effectiveDimensions} dimensions; "
                    f"first component variances="
                    f"{evaluation.metrics.componentVariance[:10]}; "
                    f"first explained variance ratios="
                    f"{evaluation.metrics.pcaExplainedVarianceRatio[:10]}; "
                    f"total cumulative explained variance={cumulative_variance}; "
                    f"top loading-gene preview={loading_preview}; "
                    "maximum default/context-family loading enrichment="
                    f"{evaluation.metrics.loadingFamilyEnrichment}; "
                    "technical PC association="
                    f"{evaluation.metrics.technicalPcaAssociation}; "
                    "protected PC association="
                    f"{evaluation.metrics.protectedPcaAssociation}; "
                    f"QC PC association={evaluation.metrics.qcPcaAssociation}; "
                    f"warnings={evaluation.warnings}."
                )
                pca_items.append(
                    DecisionEvidence(
                        evidenceId=technical_id,
                        evidenceClass="technical",
                        summary=pca_summary,
                        artifactReferences=self._evaluation_artifacts(evaluation),
                    )
                )
                evidence_ids.append(technical_id)
                geometric_id = f"evidence:pca:{evaluation.candidateId}:geometric"
                pca_items.append(
                    DecisionEvidence(
                        evidenceId=geometric_id,
                        evidenceClass="geometric",
                        summary=(
                            "PCA and graph silhouette diagnostics are "
                            f"{evaluation.metrics.pcaSilhouette} and "
                            f"{evaluation.metrics.graphSilhouetteMedian}; "
                            f"the registered graph produced "
                            f"{evaluation.metrics.nClusters} clusters; adjacent-prefix "
                            "neighbor overlap="
                            f"{evaluation.metrics.neighborPrefixOverlap}."
                        ),
                        artifactReferences=self._evaluation_artifacts(evaluation),
                    )
                )
                evidence_ids.append(geometric_id)
                pca_evidence_by_dimensions[evaluation.parameters.dimensions] = list(
                    evidence_ids
                )
            else:
                failure_id = f"evidence:pca:{evaluation.candidateId}:failure"
                pca_items.append(
                    DecisionEvidence(
                        evidenceId=failure_id,
                        evidenceClass="other",
                        summary=_bounded_evidence_summary(
                            f"The candidate was not eligible: "
                            f"{evaluation.error or evaluation.eligibilityReasons}."
                        ),
                        artifactReferences=self._evaluation_artifacts(evaluation),
                    )
                )
                evidence_ids.append(failure_id)
            pca_evaluations.append(
                evaluation.model_copy(
                    update={
                        "evidenceIds": list(
                            dict.fromkeys([*evaluation.evidenceIds, *evidence_ids])
                        )
                    }
                )
            )
        pca_bundle = self._tuning_evidence_bundle("pcaPrefix", pca_items)
        pca_definition = build_pca_prefix_decision(
            evidence_bundle_id=pca_bundle.bundleId,
            matrix_rank=min(handoff.nCells, handoff.nFeatures) - 1,
            candidate_dimensions=(
                eligible_pca_dimensions
                if eligible_pca_dimensions
                else [candidate.dimensions for candidate in pca_plan.candidates]
            ),
        )
        pca_definition = require_option_evidence(
            pca_definition,
            {
                option.optionId: pca_evidence_by_dimensions[option.payload.dimensions]
                for option in pca_definition.executorOptions
                if isinstance(option.payload, PcaPrefixExecutorPayload)
                and option.payload.dimensions in pca_evidence_by_dimensions
            },
        )
        pca_rule_selection = (
            DecisionSelection(
                selectedOptionId="pcaPrefix:defer",
                evidenceIds=[item.evidenceId for item in pca_bundle.evidence],
                rationale=(
                    "No registered PCA candidate completed with the required "
                    "technical and geometric evidence."
                ),
                confidence="notApplicable",
            )
            if not eligible_pca_dimensions
            else None
        )
        pca_resolution = self._resolve_rna_decision(
            store,
            request_record,
            pca_definition,
            pca_bundle,
            answers,
            rule_selection=pca_rule_selection,
        )
        pca_payload = (
            pca_resolution.compiled.executorPayload
            if pca_resolution.compiled is not None
            else None
        )
        if pca_payload is not None and not isinstance(
            pca_payload, PcaPrefixExecutorPayload
        ):
            raise TypeError("PCA decision compiled an unexpected payload")
        pca_phase = self._phase_from_resolution(
            pca_plan,
            pca_evaluations,
            pca_resolution,
            payload_field="dimensions",
            payload_value=(pca_payload.dimensions if pca_payload is not None else -1),
        )
        phase_evidence.append(pca_phase)
        if pca_resolution.record is not None:
            decision_sources["pcaPrefix"] = pca_resolution.record.source
        selected_pca = pca_phase.selected_evaluation()
        if selected_pca is None:
            return return_pending(pca_resolution)

        license_evidence_id = "evidence:correctionLicense:studyContract"
        license_bundle = self._tuning_evidence_bundle(
            "correctionLicense",
            [
                DecisionEvidence(
                    evidenceId=license_evidence_id,
                    evidenceClass="design",
                    summary=(
                        f"The StudyContract license is "
                        f"{study_contract.correctionLicense}; technical columns="
                        f"{study_contract.technicalBatchColumns}; protected columns="
                        f"{study_contract.protectedColumns}."
                    ),
                )
            ],
        )
        license_definition = build_correction_license_decision(
            evidence_bundle_id=license_bundle.bundleId,
            license=study_contract.correctionLicense,
        )
        license_definition = require_option_evidence(
            license_definition,
            {
                f"correctionLicense:{study_contract.correctionLicense}": [
                    license_evidence_id
                ]
            },
        )
        license_resolution = self._resolve_rna_decision(
            store,
            request_record,
            license_definition,
            license_bundle,
            answers,
            rule_selection=DecisionSelection(
                selectedOptionId=(
                    f"correctionLicense:{study_contract.correctionLicense}"
                ),
                evidenceIds=[license_evidence_id],
                rationale="Apply the exact deterministic StudyContract license.",
            ),
        )
        if license_resolution.compiled is None:
            return return_pending(
                license_resolution,
                correction_license=study_contract.correctionLicense,
            )
        if license_resolution.record is not None:
            decision_sources["correctionLicense"] = license_resolution.record.source
        license_payload = license_resolution.compiled.executorPayload
        if not isinstance(license_payload, CorrectionLicensePayload):
            raise TypeError("Correction license compiled an unexpected payload")

        correction_need: str | None = None
        if license_payload.license == "safe":
            need_items = [
                DecisionEvidence(
                    evidenceId="evidence:correctionNeed:design",
                    evidenceClass="design",
                    summary=(
                        "The design license is safe, but an indeterminate choice "
                        "remains available if representation evidence is incomplete."
                    ),
                )
            ]
            if (
                selected_pca.metrics.batchMixing
                or selected_pca.metrics.technicalPcaAssociation
            ):
                need_items.append(
                    DecisionEvidence(
                        evidenceId="evidence:correctionNeed:batch",
                        evidenceClass="batchRemoval",
                        summary=(
                            "Native representation batch-mixing metrics are "
                            f"{selected_pca.metrics.batchMixing}; per-PC technical "
                            "associations are "
                            f"{selected_pca.metrics.technicalPcaAssociation}."
                        ),
                        artifactReferences=self._evaluation_artifacts(selected_pca),
                    )
                )
            if (
                selected_pca.metrics.biologicalPreservation
                or not study_contract.protectedColumns
            ):
                need_items.append(
                    DecisionEvidence(
                        evidenceId="evidence:correctionNeed:biology",
                        evidenceClass="biologicalConservation",
                        summary=(
                            "Native protected-variable diagnostics are "
                            f"{selected_pca.metrics.biologicalPreservation}; "
                            f"declared protected columns="
                            f"{study_contract.protectedColumns}."
                        ),
                        artifactReferences=self._evaluation_artifacts(selected_pca),
                    )
                )
            need_bundle = self._tuning_evidence_bundle(
                "correctionNeed",
                need_items,
            )
            need_definition = build_correction_need_decision(
                evidence_bundle_id=need_bundle.bundleId,
                license=license_payload.license,
            )
            comparative_need_ids = [
                item.evidenceId
                for item in need_items
                if item.evidenceClass in {"batchRemoval", "biologicalConservation"}
            ]
            need_definition = require_option_evidence(
                need_definition,
                {
                    "correctionNeed:needed": comparative_need_ids,
                    "correctionNeed:notNeeded": comparative_need_ids,
                    "correctionNeed:indeterminate": ["evidence:correctionNeed:design"],
                },
            )
            need_resolution = self._resolve_rna_decision(
                store,
                request_record,
                need_definition,
                need_bundle,
                answers,
            )
            if need_resolution.compiled is None:
                pending_reason = (
                    need_resolution.pending.reason
                    if need_resolution.pending is not None
                    else "Correction need remains unresolved."
                )
                correction_need_selection = CorrectionNeedSelection(
                    status="needsInput",
                    selectedOptionId="correctionNeed:indeterminate",
                    rationale=pending_reason,
                )
                return return_pending(
                    need_resolution,
                    correction_license=license_payload.license,
                )
            if need_resolution.record is not None:
                decision_sources["correctionNeed"] = need_resolution.record.source
            need_payload = need_resolution.compiled.executorPayload
            if not isinstance(need_payload, CorrectionNeedPayload):
                raise TypeError("Correction need compiled an unexpected payload")
            correction_need = need_payload.need
            assert need_resolution.record is not None
            need_option_id: Literal[
                "correctionNeed:needed",
                "correctionNeed:notNeeded",
            ] = (
                "correctionNeed:needed"
                if need_payload.need == "needed"
                else "correctionNeed:notNeeded"
            )
            correction_need_selection = CorrectionNeedSelection(
                status="selected",
                selectedOptionId=need_option_id,
                evidenceIds=list(need_resolution.record.evidenceIds),
                rationale=need_resolution.record.rationale,
            )

        full_correction_plan = planner.batch_correction_phase(selected_pca.parameters)
        correction_candidates = list(full_correction_plan.candidates)
        if not evaluate_harmony:
            correction_candidates = [
                candidate
                for candidate in correction_candidates
                if not candidate.useHarmony
            ]
        correction_plan = ParameterPhasePlan.model_validate(
            {
                **full_correction_plan.model_dump(mode="json"),
                "candidates": [
                    candidate.model_dump(mode="json")
                    for candidate in correction_candidates
                ],
            }
        )
        correction_evaluations = list(
            phase_evaluations(
                correction_plan,
                lambda: execute_parameter_phase(
                    store,
                    normalized=normalized,
                    plan=correction_plan,
                    batch_columns=diagnostic_batch_columns,
                    preservation_columns=experimental_handoff.preservationColumns,
                    experimental_handoff=tuning_handoff,
                    min_cluster_cells=request_record.config.minClusterCells,
                    identity_feature_limit=request_record.config.maxIdentityFeatures,
                ),
            )
        )
        correction_native = next(
            (
                evaluation
                for evaluation in correction_evaluations
                if not evaluation.parameters.useHarmony
                and evaluation.status == "done"
                and evaluation.eligible
            ),
            None,
        )
        if "batchCorrection" not in prior_phases:
            correction_doublets = (
                score_advisory_doublets(
                    store,
                    correction_native,
                    correction_evaluations,
                    assay=handoff.assay,
                    feature_selection=artifact_model_to_ref(handoff.graphFeatures),
                    capture_column=study_contract.physicalCaptureColumn,
                )
                if correction_native is not None
                else None
            )
            correction_evaluations = list(
                augment_cluster_evaluations(
                    store,
                    correction_evaluations,
                    marker_assay=plan.markerAssay,
                    marker_features=artifact_model_to_ref(handoff.markerFeatures),
                    independent_unit_columns=study_contract.independentUnitColumns,
                    technical_columns=diagnostic_batch_columns,
                    nominated_families=diagnostic_families,
                    protected_families=protected_families,
                    doublet_evidence=correction_doublets,
                )
            )
        correction_evaluations = list(_stable_phase_evaluations(correction_evaluations))
        native_evaluation = next(
            (
                evaluation
                for evaluation in correction_evaluations
                if not evaluation.parameters.useHarmony
                and evaluation.status == "done"
                and evaluation.eligible
            ),
            None,
        )
        harmony_evaluation = next(
            (
                evaluation
                for evaluation in correction_evaluations
                if evaluation.parameters.useHarmony
                and evaluation.status == "done"
                and evaluation.eligible
            ),
            None,
        )
        outcome_items: list[DecisionEvidence] = []
        native_biology_id: str | None = None
        if native_evaluation is not None:
            native_biology_id = "evidence:correctionOutcome:nativeBiology"
            outcome_items.append(
                DecisionEvidence(
                    evidenceId=native_biology_id,
                    evidenceClass="biologicalConservation",
                    summary=(
                        "Native protected-variable diagnostics are "
                        f"{native_evaluation.metrics.biologicalPreservation}; "
                        "cross-unit support is "
                        f"{native_evaluation.metrics.crossUnitSupport}; marker "
                        f"coherence is {native_evaluation.metrics.markerCoherence}."
                    ),
                    artifactReferences=self._evaluation_artifacts(native_evaluation),
                )
            )

        harmony_eligible, harmony_gate_reasons = harmony_acceptance_gate(
            native_evaluation,
            harmony_evaluation,
            batch_columns=diagnostic_batch_columns,
            protected_columns=study_contract.protectedColumns,
            independent_unit_columns=study_contract.independentUnitColumns,
            require_doublet_evidence=True,
        )
        harmony_eligible = (
            harmony_eligible
            and license_payload.license == "safe"
            and correction_need == "needed"
        )
        if harmony_evaluation is not None and not selectable_harmony:
            harmony_gate_reasons = [
                *harmony_gate_reasons,
                "Harmony was executed for diagnosis but is not licensed for selection.",
            ]
        harmony_evidence_ids: list[str] = []
        if harmony_evaluation is not None:
            harmony_biology_id = "evidence:correctionOutcome:harmonyBiology"
            outcome_items.append(
                DecisionEvidence(
                    evidenceId=harmony_biology_id,
                    evidenceClass="biologicalConservation",
                    summary=(
                        "Harmony protected-variable diagnostics are "
                        f"{harmony_evaluation.metrics.biologicalPreservation}; "
                        "cross-unit support is "
                        f"{harmony_evaluation.metrics.crossUnitSupport}; marker "
                        f"coherence is {harmony_evaluation.metrics.markerCoherence}; "
                        f"acceptance gate findings are {harmony_gate_reasons}."
                    ),
                    artifactReferences=self._evaluation_artifacts(harmony_evaluation),
                )
            )
            harmony_evidence_ids.append(harmony_biology_id)
            batch_id = "evidence:correctionOutcome:batchRemoval"
            outcome_items.append(
                DecisionEvidence(
                    evidenceId=batch_id,
                    evidenceClass="batchRemoval",
                    summary=(
                        "Matched native and Harmony batch-mixing metrics are "
                        f"{native_evaluation.metrics.batchMixing if native_evaluation else {}} "
                        f"and {harmony_evaluation.metrics.batchMixing}; gate findings "
                        f"are {harmony_gate_reasons}."
                    ),
                    artifactReferences=self._evaluation_artifacts(harmony_evaluation),
                )
            )
            harmony_evidence_ids.append(batch_id)
            if harmony_eligible:
                protected_id = "evidence:correctionOutcome:protectedPreservation"
                outcome_items.append(
                    DecisionEvidence(
                        evidenceId=protected_id,
                        evidenceClass="protectedVariablePreservation",
                        summary=(
                            "Harmony improved at least one approved batch metric "
                            "beyond 0.05 and did not materially degrade protected, "
                            "cross-unit, graph-connectivity, or marker evidence."
                        ),
                        artifactReferences=self._evaluation_artifacts(
                            harmony_evaluation
                        ),
                    )
                )
                harmony_evidence_ids.append(protected_id)

        outcome_bundle = self._tuning_evidence_bundle(
            "correctionOutcome",
            outcome_items,
        )
        outcome_definition = build_correction_outcome_decision(
            evidence_bundle_id=outcome_bundle.bundleId,
            license=license_payload.license,
            need=(
                cast(Any, correction_need)
                if license_payload.license == "safe"
                else None
            ),
            harmony_eligible=harmony_eligible,
        )
        outcome_definition = require_option_evidence(
            outcome_definition,
            {
                **(
                    {"correctionOutcome:retainNative": [native_biology_id]}
                    if native_biology_id is not None
                    else {}
                ),
                **(
                    {"correctionOutcome:acceptHarmony": harmony_evidence_ids}
                    if harmony_eligible
                    else {}
                ),
            },
        )
        native_rule = None
        if not harmony_eligible:
            native_rule = DecisionSelection(
                selectedOptionId=(
                    "correctionOutcome:retainNative"
                    if native_biology_id is not None
                    else "correctionOutcome:indeterminate"
                ),
                evidenceIds=(
                    [native_biology_id]
                    if native_biology_id is not None
                    else [item.evidenceId for item in outcome_items]
                ),
                rationale=(
                    "Retain the mandatory native representation because Harmony "
                    "did not demonstrate both material batch improvement and "
                    "preserved biological evidence: "
                    f"{harmony_gate_reasons}."
                    if native_biology_id is not None
                    else "Native biological-conservation evidence is unavailable."
                ),
            )
        outcome_resolution = self._resolve_rna_decision(
            store,
            request_record,
            outcome_definition,
            outcome_bundle,
            answers,
            rule_selection=native_rule,
        )
        outcome_payload = (
            outcome_resolution.compiled.executorPayload
            if outcome_resolution.compiled is not None
            else None
        )
        if outcome_payload is not None and not isinstance(
            outcome_payload,
            CorrectionOutcomeExecutorPayload,
        ):
            raise TypeError("Correction outcome compiled an unexpected payload")
        augmented_correction: list[ParameterCandidateEvaluation] = []
        for evaluation in correction_evaluations:
            correction_extra = (
                [native_biology_id]
                if not evaluation.parameters.useHarmony
                and native_biology_id is not None
                else harmony_evidence_ids
                if evaluation.parameters.useHarmony
                else []
            )
            augmented_correction.append(
                evaluation.model_copy(
                    update={
                        "evidenceIds": list(
                            dict.fromkeys([*evaluation.evidenceIds, *correction_extra])
                        )
                    }
                )
            )
        correction_phase = self._phase_from_resolution(
            correction_plan,
            augmented_correction,
            outcome_resolution,
            payload_field="useHarmony",
            payload_value=(
                outcome_payload.useHarmony if outcome_payload is not None else False
            ),
        )
        phase_evidence.append(correction_phase)
        if outcome_resolution.record is not None:
            decision_sources["correctionOutcome"] = outcome_resolution.record.source
        selected_correction = correction_phase.selected_evaluation()
        if selected_correction is None:
            return return_pending(
                outcome_resolution,
                correction_license=license_payload.license,
            )

        graph_plan = planner.graph_phase(selected_correction.parameters)
        raw_graph = phase_evaluations(
            graph_plan,
            lambda: execute_parameter_phase(
                store,
                normalized=normalized,
                plan=graph_plan,
                batch_columns=(
                    experimental_handoff.batchColumns
                    if selected_correction.parameters.useHarmony
                    else []
                ),
                preservation_columns=experimental_handoff.preservationColumns,
                experimental_handoff=experimental_handoff,
                min_cluster_cells=request_record.config.minClusterCells,
                identity_feature_limit=request_record.config.maxIdentityFeatures,
            ),
        )
        graph_doublet_reference = next(
            (
                evaluation
                for evaluation in raw_graph
                if evaluation.status == "done" and evaluation.eligible
            ),
            None,
        )
        graph_doublet_evidence = (
            restore_advisory_doublets(
                graph_doublet_reference,
                capture_column=study_contract.physicalCaptureColumn,
            )
            if "graphK" in prior_phases and graph_doublet_reference is not None
            else score_advisory_doublets(
                store,
                graph_doublet_reference,
                raw_graph,
                assay=handoff.assay,
                feature_selection=artifact_model_to_ref(handoff.graphFeatures),
                capture_column=study_contract.physicalCaptureColumn,
            )
            if graph_doublet_reference is not None
            else None
        )
        if "graphK" not in prior_phases:
            raw_graph = augment_cluster_evaluations(
                store,
                raw_graph,
                marker_assay=plan.markerAssay,
                marker_features=artifact_model_to_ref(handoff.markerFeatures),
                independent_unit_columns=study_contract.independentUnitColumns,
                technical_columns=diagnostic_batch_columns,
                nominated_families=diagnostic_families,
                protected_families=protected_families,
                doublet_evidence=graph_doublet_evidence,
            )
        graph_items: list[DecisionEvidence] = []
        graph_evaluations: list[ParameterCandidateEvaluation] = []
        eligible_graph_values: list[int] = []
        graph_evidence_by_k: dict[int, list[str]] = {}
        for evaluation in _stable_phase_evaluations(raw_graph):
            graph_extra: list[str] = []
            if evaluation.status == "done" and evaluation.eligible:
                eligible_graph_values.append(evaluation.parameters.neighborsK)
                evidence_id = f"evidence:graph:{evaluation.candidateId}:geometry"
                graph_items.append(
                    DecisionEvidence(
                        evidenceId=evidence_id,
                        evidenceClass="geometric",
                        summary=(
                            f"The graph has k={evaluation.parameters.neighborsK}, "
                            f"{evaluation.metrics.nClusters} clusters, silhouette "
                            f"{evaluation.metrics.graphSilhouetteMedian}, and "
                            f"minimum cluster size "
                            f"{evaluation.metrics.minClusterCells}."
                        ),
                        artifactReferences=self._evaluation_artifacts(evaluation),
                    )
                )
                graph_extra.append(evidence_id)
                graph_summaries = (
                    (
                        "stability",
                        "resamplingStability",
                        (
                            f"seed ARI={evaluation.metrics.seedStability}; "
                            f"subsample ARI={evaluation.metrics.subsampleStability}; "
                            "membership strength="
                            f"{evaluation.metrics.membershipStrengthMean}; cluster "
                            f"connectivity={evaluation.metrics.clusterConnectivity}."
                        ),
                    ),
                    (
                        "markers",
                        "markerCoherence",
                        (
                            f"marker coherence={evaluation.metrics.markerCoherence}; "
                            "marker specificity="
                            f"{evaluation.metrics.markerSpecificityMedian}; "
                            "default/context-family enrichment="
                            f"{evaluation.metrics.markerFamilyEnrichment}; protected "
                            f"families={evaluation.metrics.protectedMarkerFamilies}."
                        ),
                    ),
                    (
                        "support",
                        "crossUnitSupport",
                        (
                            f"cross-unit support={evaluation.metrics.crossUnitSupport}; "
                            "technical association="
                            f"{evaluation.metrics.technicalAssociation}."
                        ),
                    ),
                    (
                        "doublets",
                        "qualityControl",
                        (
                            "advisory doublet concentration="
                            f"{evaluation.metrics.doubletHighScoreConcentration}; "
                            "score quantiles="
                            f"{evaluation.metrics.doubletScoreQuantiles}."
                        ),
                    ),
                )
                for suffix, evidence_class, summary in graph_summaries:
                    graph_evidence_id = (
                        f"evidence:graph:{evaluation.candidateId}:{suffix}"
                    )
                    graph_items.append(
                        DecisionEvidence(
                            evidenceId=graph_evidence_id,
                            evidenceClass=cast(Any, evidence_class),
                            summary=summary,
                            artifactReferences=self._evaluation_artifacts(evaluation),
                        )
                    )
                    graph_extra.append(graph_evidence_id)
                graph_evidence_by_k[evaluation.parameters.neighborsK] = list(
                    graph_extra
                )
            else:
                evidence_id = f"evidence:graph:{evaluation.candidateId}:failure"
                graph_items.append(
                    DecisionEvidence(
                        evidenceId=evidence_id,
                        evidenceClass="other",
                        summary=_bounded_evidence_summary(
                            f"The graph candidate was not eligible: "
                            f"{evaluation.error or evaluation.eligibilityReasons}."
                        ),
                        artifactReferences=self._evaluation_artifacts(evaluation),
                    )
                )
                graph_extra.append(evidence_id)
            graph_evaluations.append(
                evaluation.model_copy(
                    update={
                        "evidenceIds": list(
                            dict.fromkeys([*evaluation.evidenceIds, *graph_extra])
                        )
                    }
                )
            )
        graph_bundle = self._tuning_evidence_bundle("graphK", graph_items)
        graph_definition = build_graph_k_decision(
            evidence_bundle_id=graph_bundle.bundleId,
            n_cells=handoff.nCells,
            candidate_neighbors=(
                eligible_graph_values
                if eligible_graph_values
                else [candidate.neighborsK for candidate in graph_plan.candidates]
            ),
        )
        graph_definition = require_option_evidence(
            graph_definition,
            {
                option.optionId: graph_evidence_by_k[option.payload.neighborsK]
                for option in graph_definition.executorOptions
                if isinstance(option.payload, GraphExecutorPayload)
                and option.payload.neighborsK in graph_evidence_by_k
            },
        )
        graph_rule_selection = (
            DecisionSelection(
                selectedOptionId="graphScale:defer",
                evidenceIds=[item.evidenceId for item in graph_bundle.evidence],
                rationale=(
                    "No registered graph candidate completed with geometric evidence."
                ),
                confidence="notApplicable",
            )
            if not eligible_graph_values
            else None
        )
        graph_resolution = self._resolve_rna_decision(
            store,
            request_record,
            graph_definition,
            graph_bundle,
            answers,
            rule_selection=graph_rule_selection,
        )
        graph_payload = (
            graph_resolution.compiled.executorPayload
            if graph_resolution.compiled is not None
            else None
        )
        if graph_payload is not None and not isinstance(
            graph_payload, GraphExecutorPayload
        ):
            raise TypeError("Graph decision compiled an unexpected payload")
        graph_phase = self._phase_from_resolution(
            graph_plan,
            graph_evaluations,
            graph_resolution,
            payload_field="neighborsK",
            payload_value=(
                graph_payload.neighborsK if graph_payload is not None else -1
            ),
        )
        phase_evidence.append(graph_phase)
        if graph_resolution.record is not None:
            decision_sources["graphK"] = graph_resolution.record.source
        selected_graph = graph_phase.selected_evaluation()
        if selected_graph is None:
            return return_pending(
                graph_resolution,
                correction_license=license_payload.license,
            )

        if graph_doublet_evidence is None:
            raise ValueError(
                "Selected graph lacks the required advisory doublet evidence"
            )
        doublet_evidence = graph_doublet_evidence
        cluster_plan = planner.clustering_phase(selected_graph.parameters)
        persisted_cluster = prior_phases.get(cluster_plan.phase)
        raw_clusters = phase_evaluations(
            cluster_plan,
            lambda: execute_parameter_phase(
                store,
                normalized=normalized,
                plan=cluster_plan,
                batch_columns=(
                    experimental_handoff.batchColumns
                    if selected_graph.parameters.useHarmony
                    else []
                ),
                preservation_columns=experimental_handoff.preservationColumns,
                experimental_handoff=experimental_handoff,
                min_cluster_cells=request_record.config.minClusterCells,
                identity_feature_limit=request_record.config.maxIdentityFeatures,
            ),
        )
        if persisted_cluster is not None:
            cluster_evaluations = list(raw_clusters)
        else:
            cluster_evaluations = list(
                augment_cluster_evaluations(
                    store,
                    raw_clusters,
                    marker_assay=plan.markerAssay,
                    marker_features=artifact_model_to_ref(handoff.markerFeatures),
                    independent_unit_columns=study_contract.independentUnitColumns,
                    technical_columns=diagnostic_batch_columns,
                    nominated_families=diagnostic_families,
                    protected_families=protected_families,
                    doublet_evidence=doublet_evidence,
                )
            )
        cluster_items: list[DecisionEvidence] = []
        scored: list[tuple[float, float, ParameterCandidateEvaluation]] = []
        eligible_cluster_values: list[float] = []
        augmented_clusters: list[ParameterCandidateEvaluation] = []
        cluster_evidence_by_resolution: dict[float, list[str]] = {}
        for evaluation in _stable_phase_evaluations(cluster_evaluations):
            cluster_extra: list[str] = []
            if evaluation.status == "done" and evaluation.eligible:
                marker_auc_preview = dict(
                    list(evaluation.metrics.markerAucByCluster.items())[:20]
                )
                marker_gene_preview = {
                    cluster: genes[:3]
                    for cluster, genes in list(
                        evaluation.metrics.topMarkerGenes.items()
                    )[:20]
                }
                eligible_cluster_values.append(evaluation.parameters.leidenResolution)
                geometry_id = f"evidence:cluster:{evaluation.candidateId}:geometry"
                stability_id = f"evidence:cluster:{evaluation.candidateId}:stability"
                marker_id = f"evidence:cluster:{evaluation.candidateId}:markers"
                cluster_items.extend(
                    [
                        DecisionEvidence(
                            evidenceId=geometry_id,
                            evidenceClass="geometric",
                            summary=(
                                f"Resolution "
                                f"{evaluation.parameters.leidenResolution:g} "
                                f"has silhouette "
                                f"{evaluation.metrics.graphSilhouetteMedian}, "
                                f"{evaluation.metrics.nClusters} clusters, and "
                                f"minimum cluster size "
                                f"{evaluation.metrics.minClusterCells}; membership "
                                "strength="
                                f"{evaluation.metrics.membershipStrengthMean}; "
                                f"cluster connectivity="
                                f"{evaluation.metrics.clusterConnectivity}."
                            ),
                            artifactReferences=self._evaluation_artifacts(evaluation),
                        ),
                        DecisionEvidence(
                            evidenceId=stability_id,
                            evidenceClass="resamplingStability",
                            summary=(
                                f"Alternate-seed ARI is "
                                f"{evaluation.metrics.seedStability}; deterministic "
                                f"subsample ARI is "
                                f"{evaluation.metrics.subsampleStability}."
                            ),
                            artifactReferences=self._evaluation_artifacts(evaluation),
                        ),
                        DecisionEvidence(
                            evidenceId=marker_id,
                            evidenceClass="markerCoherence",
                            summary=_bounded_evidence_summary(
                                "The fraction of clusters with marker programs is "
                                f"{evaluation.metrics.markerCoherence}; nominated "
                                "family marker enrichment is "
                                f"{evaluation.metrics.markerFamilyEnrichment}; "
                                "median marker specificity is "
                                f"{evaluation.metrics.markerSpecificityMedian}; "
                                "per-cluster marker AUC preview is "
                                f"{marker_auc_preview}; top-feature preview is "
                                f"{marker_gene_preview}; "
                                "protected families observed among markers are "
                                f"{evaluation.metrics.protectedMarkerFamilies}."
                            ),
                            artifactReferences=self._evaluation_artifacts(evaluation),
                        ),
                    ]
                )
                cluster_extra.extend([geometry_id, stability_id, marker_id])
                if evaluation.metrics.crossUnitSupport is not None:
                    support_id = (
                        f"evidence:cluster:{evaluation.candidateId}:unitSupport"
                    )
                    cluster_items.append(
                        DecisionEvidence(
                            evidenceId=support_id,
                            evidenceClass="crossUnitSupport",
                            summary=(
                                "The fraction of clusters represented in at least "
                                "two independent units is "
                                f"{evaluation.metrics.crossUnitSupport}."
                            ),
                            artifactReferences=self._evaluation_artifacts(evaluation),
                        )
                    )
                    cluster_extra.append(support_id)
                if evaluation.metrics.biologicalPreservation:
                    protected_id = (
                        f"evidence:cluster:{evaluation.candidateId}:protected"
                    )
                    cluster_items.append(
                        DecisionEvidence(
                            evidenceId=protected_id,
                            evidenceClass="protectedVariablePreservation",
                            summary=(
                                "Protected-variable metrics are "
                                f"{evaluation.metrics.biologicalPreservation}."
                            ),
                            artifactReferences=self._evaluation_artifacts(evaluation),
                        )
                    )
                    cluster_extra.append(protected_id)
                if evaluation.metrics.technicalAssociation:
                    technical_id = (
                        f"evidence:cluster:{evaluation.candidateId}:technical"
                    )
                    cluster_items.append(
                        DecisionEvidence(
                            evidenceId=technical_id,
                            evidenceClass="technical",
                            summary=(
                                "Cluster-to-technical association is "
                                f"{evaluation.metrics.technicalAssociation}."
                            ),
                            artifactReferences=self._evaluation_artifacts(evaluation),
                        )
                    )
                    cluster_extra.append(technical_id)
                if evaluation.metrics.doubletHighScoreConcentration is not None:
                    doublet_id = f"evidence:cluster:{evaluation.candidateId}:doublet"
                    cluster_items.append(
                        DecisionEvidence(
                            evidenceId=doublet_id,
                            evidenceClass="qualityControl",
                            summary=(
                                "Maximum cluster enrichment for the top decile of "
                                "capture-aware advisory doublet scores is "
                                f"{evaluation.metrics.doubletHighScoreConcentration}."
                            ),
                            artifactReferences=self._evaluation_artifacts(evaluation),
                        )
                    )
                    cluster_extra.append(doublet_id)
                geometry = (
                    evaluation.metrics.graphSilhouetteMedian
                    if evaluation.metrics.graphSilhouetteMedian is not None
                    else -1.0
                )
                stability = (
                    (
                        evaluation.metrics.seedStability
                        + evaluation.metrics.subsampleStability
                    )
                    / 2
                    if evaluation.metrics.seedStability is not None
                    and evaluation.metrics.subsampleStability is not None
                    else -1.0
                )
                marker = evaluation.metrics.markerCoherence or 0.0
                marker_specificity = evaluation.metrics.markerSpecificityMedian or 0.0
                unit_support = evaluation.metrics.crossUnitSupport or 0.0
                membership = evaluation.metrics.membershipStrengthMean or 0.0
                connectivity = evaluation.metrics.clusterConnectivity or 0.0
                technical = max(
                    evaluation.metrics.technicalAssociation.values(),
                    default=0.0,
                )
                doublet_penalty = max(
                    0.0,
                    (evaluation.metrics.doubletHighScoreConcentration or 1.0) - 1.0,
                )
                protected_penalty = float(
                    bool(evaluation.metrics.protectedMarkerFamilies)
                )
                score = (
                    geometry
                    + 0.25 * stability
                    + 0.2 * marker
                    + 0.15 * marker_specificity
                    + 0.1 * unit_support
                    + 0.1 * membership
                    + 0.1 * connectivity
                    - 0.1 * technical
                    - 0.1 * doublet_penalty
                    - 0.1 * protected_penalty
                )
                scored.append(
                    (
                        score,
                        -evaluation.parameters.leidenResolution,
                        evaluation,
                    )
                )
                cluster_evidence_by_resolution[
                    evaluation.parameters.leidenResolution
                ] = list(cluster_extra)
            else:
                failure_id = f"evidence:cluster:{evaluation.candidateId}:failure"
                cluster_items.append(
                    DecisionEvidence(
                        evidenceId=failure_id,
                        evidenceClass="other",
                        summary=_bounded_evidence_summary(
                            f"The partition was not eligible: "
                            f"{evaluation.error or evaluation.eligibilityReasons}."
                        ),
                        artifactReferences=self._evaluation_artifacts(evaluation),
                    )
                )
                cluster_extra.append(failure_id)
            augmented_clusters.append(
                evaluation.model_copy(
                    update={
                        "evidenceIds": list(
                            dict.fromkeys([*evaluation.evidenceIds, *cluster_extra])
                        )
                    }
                )
            )

        known_cluster_ids = {
            0.25: "clusterResolution:veryCoarse",
            0.5: "clusterResolution:coarse",
            0.75: "clusterResolution:balanced",
            1.0: "clusterResolution:detailed",
            1.25: "clusterResolution:fine",
            1.5: "clusterResolution:veryFine",
        }

        def cluster_option_id(resolution: float) -> str:
            return known_cluster_ids.get(
                resolution,
                f"clusterResolution:r{str(resolution).replace('.', 'p')}",
            )

        candidate_resolutions = (
            eligible_cluster_values
            if eligible_cluster_values
            else [candidate.leidenResolution for candidate in cluster_plan.candidates]
        )
        preferred_resolution = (
            max(scored, key=lambda value: (value[0], value[1]))[
                2
            ].parameters.leidenResolution
            if scored
            else candidate_resolutions[0]
        )
        cluster_bundle = self._tuning_evidence_bundle(
            "clusterPartition",
            cluster_items,
        )
        cluster_definition = build_cluster_partition_decision(
            evidence_bundle_id=cluster_bundle.bundleId,
            metric_preferred_option_id=cluster_option_id(preferred_resolution),
            resolution_candidates=candidate_resolutions,
        )
        cluster_definition = require_option_evidence(
            cluster_definition,
            {
                option.optionId: cluster_evidence_by_resolution[
                    option.payload.leidenResolution
                ]
                for option in cluster_definition.executorOptions
                if isinstance(option.payload, ClusterExecutorPayload)
                and option.payload.leidenResolution in cluster_evidence_by_resolution
            },
        )
        cluster_rule_selection = (
            DecisionSelection(
                selectedOptionId="clusterPartition:abstain",
                evidenceIds=[item.evidenceId for item in cluster_bundle.evidence],
                rationale=(
                    "No registered cluster partition completed with the required "
                    "independent evidence."
                ),
                confidence="notApplicable",
            )
            if not eligible_cluster_values
            else None
        )
        cluster_resolution = self._resolve_rna_decision(
            store,
            request_record,
            cluster_definition,
            cluster_bundle,
            answers,
            rule_selection=cluster_rule_selection,
        )
        if cluster_resolution.compiled is None:
            cluster_phase = ParameterPhaseEvidence(
                plan=cluster_plan,
                evaluations=augmented_clusters,
                selection=ParameterPhaseSelection(
                    phase="clusteringResolution",
                    status="needsInput",
                    rationale=(
                        cluster_resolution.pending.reason
                        if cluster_resolution.pending is not None
                        else "Cluster partition remains unresolved."
                    ),
                ),
            )
            phase_evidence.append(cluster_phase)
            return return_pending(
                cluster_resolution,
                correction_license=license_payload.license,
            )
        if cluster_resolution.record is not None:
            decision_sources["clusterPartition"] = cluster_resolution.record.source
        if cluster_resolution.record is not None and (
            cluster_resolution.record.status == "abstain"
        ):
            cluster_phase = ParameterPhaseEvidence(
                plan=cluster_plan,
                evaluations=augmented_clusters,
                selection=ParameterPhaseSelection(
                    phase="clusteringResolution",
                    status="abstained",
                    evidenceIds=list(cluster_resolution.record.evidenceIds),
                    rationale=cluster_resolution.record.rationale,
                ),
            )
            phase_evidence.append(cluster_phase)
            state = build_state(
                correction_license=license_payload.license,
            )
            return (
                sequential_evidence_to_report(
                    state,
                    marker_assay=plan.markerAssay,
                ),
                state,
            )
        cluster_payload = cluster_resolution.compiled.executorPayload
        if not isinstance(cluster_payload, ClusterExecutorPayload):
            raise TypeError("Cluster decision compiled an unexpected payload")
        cluster_phase = self._phase_from_resolution(
            cluster_plan,
            augmented_clusters,
            cluster_resolution,
            payload_field="leidenResolution",
            payload_value=cluster_payload.leidenResolution,
        )
        phase_evidence.append(cluster_phase)
        selected_cluster = cluster_phase.selected_evaluation()
        if selected_cluster is None:
            raise RuntimeError("Completed cluster decision lacks an exact candidate")
        state = build_state(
            correction_license=license_payload.license,
            final_candidate_id=selected_cluster.candidateId,
        )
        if request_record.config.maxRefinedCandidatesPerAssay == 0:
            return (
                sequential_evidence_to_report(
                    state,
                    marker_assay=plan.markerAssay,
                ),
                state,
            )

        refinement_deps, initial_candidate_ids = (
            prepare_sequential_refinement_dependencies(
                store,
                normalized=normalized,
                evidence=state,
                batch_columns=diagnostic_batch_columns,
                preservation_columns=experimental_handoff.preservationColumns,
                experimental_handoff=tuning_handoff,
                min_cluster_cells=request_record.config.minClusterCells,
                identity_feature_limit=request_record.config.maxIdentityFeatures,
            )
        )
        completed_evaluations = [
            refinement_deps.evaluations[candidate_id]
            for candidate_id in initial_candidate_ids
        ]
        try:
            planning_execution = run_agent_sync(
                model=self.model,
                output_type=ParameterSearchPlan,
                system_prompt=parameter_search_system_prompt(),
                user_prompt=parameter_search_prompt(
                    from_assay=handoff.assay,
                    cell_selection=handoff.cellSelection,
                    evaluations=completed_evaluations,
                    batch_columns=diagnostic_batch_columns,
                    preservation_columns=experimental_handoff.preservationColumns,
                    harmony_authorized=selectable_harmony,
                    max_refined_candidates=1,
                ),
                deps_type=ParameterTuningDependencies,
                deps=refinement_deps,
                config=request_record.config.agentRunConfig,
                name="sequential_parameter_refinement",
                output_validator=lambda proposed: validate_sequential_refinement_plan(
                    proposed,
                    refinement_deps,
                    initial_candidate_ids,
                ),
            )
        except AgentRunError:
            pending_plan = ParameterSearchPlan(
                status="complete",
                basedOnCandidateIds=[selected_cluster.candidateId],
                rationale=(
                    "The required bounded refinement review could not be completed."
                ),
                evidenceIds=list(selected_cluster.evidenceIds),
                stoppingCriteria=[
                    "Obtain a grounded refinement decision before final selection."
                ],
            )
            return (
                pending_parameter_tuning_report(
                    refinement_deps,
                    search_plan=pending_plan,
                    agent_name="sequential_parameter_refinement_needs_input",
                ),
                state,
            )
        if not isinstance(planning_execution.output, ParameterSearchPlan):
            raise TypeError("Sequential refinement returned an unexpected output")
        refinement = execute_sequential_refinement(
            refinement_deps,
            planning_execution.output.model_copy(
                update={"runInfo": planning_execution.runInfo}
            ),
            initial_candidate_ids,
        )
        if refinement.evaluation is not None:
            refined_pca = augment_pca_evaluations(
                store,
                [refinement.evaluation],
                feature_selection=artifact_model_to_ref(handoff.graphFeatures),
                nominated_families=diagnostic_families,
                protected_families=protected_families,
                technical_columns=diagnostic_batch_columns,
                batch_columns=diagnostic_batch_columns,
                protected_columns=study_contract.protectedColumns,
                qc_columns=[
                    column
                    for column in plan.cellQc.attributes
                    if column in store.cells.columns
                ],
            )
            refined_cluster = augment_cluster_evaluations(
                store,
                refined_pca,
                marker_assay=plan.markerAssay,
                marker_features=artifact_model_to_ref(handoff.markerFeatures),
                independent_unit_columns=study_contract.independentUnitColumns,
                technical_columns=diagnostic_batch_columns,
                nominated_families=diagnostic_families,
                protected_families=protected_families,
                doublet_evidence=doublet_evidence,
            )[0]
            refined_evidence_ids = [
                f"candidate:{refined_cluster.candidateId}:refinedPca",
                f"candidate:{refined_cluster.candidateId}:refinedGraph",
                f"candidate:{refined_cluster.candidateId}:refinedMarkers",
                f"candidate:{refined_cluster.candidateId}:refinedUnitSupport",
                f"candidate:{refined_cluster.candidateId}:refinedTechnical",
                f"candidate:{refined_cluster.candidateId}:refinedDoublets",
            ]
            refined_cluster = refined_cluster.model_copy(
                update={
                    "evidenceIds": list(
                        dict.fromkeys(
                            [
                                *refined_cluster.evidenceIds,
                                *refined_evidence_ids,
                            ]
                        )
                    )
                }
            )
            refinement_deps.evaluations[refined_cluster.candidateId] = refined_cluster

        if refinement.evaluation is None:
            completed_report = sequential_evidence_to_report(
                state,
                marker_assay=plan.markerAssay,
            )
            assay_report = completed_report.assayReports[handoff.assay].model_copy(
                update={"searchPlan": refinement.plan}
            )
            return (
                completed_report.model_copy(
                    update={
                        "searchPlan": refinement.plan,
                        "assayReports": {handoff.assay: assay_report},
                        "stopReason": (
                            f"{completed_report.stopReason} Refinement review "
                            f"stopped because {refinement.plan.rationale}"
                        ),
                    }
                ),
                state,
            )

        selection_ids = [
            selected_cluster.candidateId,
            refinement.evaluation.candidateId,
        ]
        selection_deps = refinement_deps.model_copy(
            update={
                "candidates": {
                    candidate_id: refinement_deps.candidates[candidate_id]
                    for candidate_id in selection_ids
                },
                "candidatePhases": {
                    candidate_id: refinement_deps.candidatePhases[candidate_id]
                    for candidate_id in selection_ids
                },
                "evaluations": {
                    candidate_id: refinement_deps.evaluations[candidate_id]
                    for candidate_id in selection_ids
                },
                "executionOrder": selection_ids,
                "maxCandidates": len(selection_ids),
            }
        )
        selection_evaluations = [
            selection_deps.evaluations[candidate_id] for candidate_id in selection_ids
        ]
        try:
            selection_execution = run_agent_sync(
                model=self.model,
                output_type=ParameterTuningReport,
                system_prompt=parameter_tuning_system_prompt(
                    request_record.config.minClusterCells
                ),
                user_prompt=parameter_tuning_prompt(
                    from_assay=handoff.assay,
                    cell_selection=handoff.cellSelection,
                    evaluations=selection_evaluations,
                    batch_columns=diagnostic_batch_columns,
                    preservation_columns=experimental_handoff.preservationColumns,
                    search_plan=refinement.plan,
                ),
                deps_type=ParameterTuningDependencies,
                deps=selection_deps,
                config=request_record.config.agentRunConfig,
                name="sequential_parameter_selection",
                output_validator=lambda proposed: validate_parameter_tuning_report(
                    proposed,
                    selection_deps,
                    search_plan=refinement.plan,
                ),
            )
        except AgentRunError:
            return (
                pending_parameter_tuning_report(
                    selection_deps,
                    search_plan=refinement.plan,
                    agent_name="sequential_parameter_selection_needs_input",
                ),
                state,
            )
        if not isinstance(selection_execution.output, ParameterTuningReport):
            raise TypeError("Sequential final selection returned an unexpected output")
        selected_report = validate_parameter_tuning_report(
            selection_execution.output,
            selection_deps,
            search_plan=refinement.plan,
        ).model_copy(update={"runInfo": selection_execution.runInfo})
        complete_inventory = [
            refinement_deps.evaluations[candidate_id]
            for candidate_id in refinement_deps.executionOrder
            if candidate_id in refinement_deps.evaluations
        ]
        selected_report = selected_report.model_copy(
            update={
                "evaluations": complete_inventory,
                "totalCandidates": len(complete_inventory),
            }
        )
        assay_report = selected_report.model_copy(update={"assayReports": {}})
        selected_report = selected_report.model_copy(
            update={"assayReports": {handoff.assay: assay_report}}
        )
        return (
            finalize_parameter_tuning_selection(
                selected_report,
                marker_assay=plan.markerAssay,
                native_assay=handoff.assay,
            ),
            state,
        )

    def feature_policy_review_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        tuning_report: ParameterTuningReport,
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, AutomatedPreprocessingPlan, bool]:
        """Review the baseline feature policy against PCA and marker evidence."""
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "feature_policy_review",
            request_record,
            parents,
        )
        if existing is not None:
            return (
                existing,
                AutomatedPreprocessingPlan.model_validate(
                    existing.outputs["preprocessingPlan"]
                ),
                bool(existing.outputs["revised"]),
            )
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "feature_policy_review",
            request_record,
            parents,
            inputs={
                "preprocessingPlan": plan.model_dump(mode="json"),
                "tuningReportSha256": hashlib.sha256(
                    record_io.canonical_json_bytes(
                        tuning_report.model_dump(mode="json")
                    )
                ).hexdigest(),
            },
            resume_record=resume_record,
        )
        try:
            assay_plan = next(
                value for value in plan.assays if value.assay == plan.primaryAssay
            )
            assay_report = tuning_report.assayReports[plan.primaryAssay]
            selected = next(
                evaluation
                for evaluation in assay_report.evaluations
                if evaluation.candidateId == assay_report.recommendedCandidateId
            )
            loading_evaluation = next(
                (
                    evaluation
                    for evaluation in assay_report.evaluations
                    if evaluation.status == "done"
                    and evaluation.eligible
                    and evaluation.parameters.dimensions
                    == selected.parameters.dimensions
                    and evaluation.metrics.loadingFamilyEnrichment
                ),
                None,
            )
            aliases = {
                "sex": "sexLinked",
                "ribosomalProtein": "ribosomal",
                "cellCycleCcn": "cellCycle",
                "HLA": "hla",
                "H2": "h2",
            }
            allowed_families = {
                "mitochondrial",
                "ribosomal",
                "mitoribosomal",
                "histone",
                "hla",
                "h2",
                "hemoglobin",
                "immuneReceptor",
                "cellCycle",
                "stress",
                "dissociation",
                "sexLinked",
            }
            nominated = [
                aliases.get(str(value), str(value))
                for value in cast(
                    list[str],
                    assay_plan.featureParameters.get(
                        "proposedExcludeFamilies",
                        [],
                    ),
                )
            ]
            protected = [
                aliases.get(str(value), str(value))
                for value in cast(
                    list[str],
                    assay_plan.featureParameters.get("protectFamilies", []),
                )
            ]
            nominated = [
                value for value in dict.fromkeys(nominated) if value in allowed_families
            ]
            protected = [
                value for value in dict.fromkeys(protected) if value in allowed_families
            ]
            loading_enrichment = (
                loading_evaluation.metrics.loadingFamilyEnrichment
                if loading_evaluation is not None
                else {}
            )
            marker_enrichment = selected.metrics.markerFamilyEnrichment
            normalized_loading: dict[str, float] = {}
            normalized_markers: dict[str, float] = {}
            for name, value in loading_enrichment.items():
                canonical = aliases.get(name, name)
                normalized_loading[canonical] = max(
                    normalized_loading.get(canonical, 0.0),
                    value,
                )
            for name, value in marker_enrichment.items():
                canonical = aliases.get(name, name)
                normalized_markers[canonical] = max(
                    normalized_markers.get(canonical, 0.0),
                    value,
                )
            eligible = [
                cast(ConditionalGeneFamily, family)
                for family in nominated
                if family not in protected
                and (
                    normalized_loading.get(family, 0.0) >= 2.0
                    or normalized_markers.get(family, 0.0) >= 2.0
                )
            ]
            scarf_default_families = {
                "mitochondrial",
                "ribosomal",
                "mitoribosomal",
                "cellCycle",
                "hla",
                "h2",
                "histone",
                "sexLinked",
            }
            default_dominant = any(
                normalized_loading.get(family, 0.0) >= 2.0
                or normalized_markers.get(family, 0.0) >= 2.0
                for family in scarf_default_families
            )
            current_default = (
                assay_plan.featureParameters.get("useScarfDefaultBlacklist") is True
            )
            default_option = bool(
                (current_default or default_dominant)
                and not scarf_default_families.intersection(protected)
            )
            loading_id = "evidence:featurePolicyReview:pcaLoadings"
            marker_id = "evidence:featurePolicyReview:clusterMarkers"
            protected_id = "evidence:featurePolicyReview:protectedFamilies"
            evidence = [
                DecisionEvidence(
                    evidenceId=loading_id,
                    evidenceClass="technical",
                    summary=(
                        "Maximum top-loading family enrichments are "
                        f"{loading_enrichment}; the registered gate is 2.0."
                    ),
                    artifactReferences=(
                        self._evaluation_artifacts(loading_evaluation)
                        if loading_evaluation is not None
                        else []
                    ),
                ),
                DecisionEvidence(
                    evidenceId=marker_id,
                    evidenceClass="markerCoherence",
                    summary=(
                        "Selected-partition family marker enrichments are "
                        f"{marker_enrichment}; the registered gate is 2.0."
                    ),
                    artifactReferences=self._evaluation_artifacts(selected),
                ),
                DecisionEvidence(
                    evidenceId=protected_id,
                    evidenceClass="protectedVariablePreservation",
                    summary=(
                        f"Protected families are {protected}; eligible nominated "
                        f"families after the veto are {eligible}."
                    ),
                ),
            ]
            bundle = self._tuning_evidence_bundle("featurePolicy", evidence)
            definition = build_feature_policy_decision(
                evidence_bundle_id=bundle.bundleId,
                proposed_exclusion_families=eligible,
                dominant_families=eligible,
                protected_families=[
                    cast(ConditionalGeneFamily, value) for value in protected
                ],
                scarf_default_eligible=default_option,
            )
            requirements: dict[str, list[str]] = {
                "featurePolicy:keepAll": [loading_id, marker_id, protected_id]
            }
            if eligible:
                requirements["featurePolicy:excludeEligibleBundle"] = [
                    loading_id,
                    marker_id,
                    protected_id,
                ]
            if default_option:
                requirements["featurePolicy:excludeScarfDefaults"] = [
                    loading_id,
                    marker_id,
                    protected_id,
                ]
            definition = require_option_evidence(definition, requirements)
            active_excluded = [
                cast(ConditionalGeneFamily, value)
                for value in assay_plan.featureParameters.get(
                    "excludeFamilies",
                    [],
                )
                if value in allowed_families
            ]
            active_payload = (
                FeaturePolicyExecutorPayload(
                    policy="excludeScarfDefaults",
                    useScarfDefaultBlacklist=True,
                )
                if current_default
                else FeaturePolicyExecutorPayload(
                    policy="excludeEligibleBundle",
                    excludedFamilies=active_excluded,
                )
                if active_excluded
                else FeaturePolicyExecutorPayload(
                    policy="keepAll",
                    excludedFamilies=[],
                )
            )
            if eligible or default_option:
                review = self._reconsider_rna_decision(
                    store,
                    request_record,
                    definition,
                    bundle,
                    answers,
                    review_instructions=(
                        "Reconsider the active graph-feature policy among the exact "
                        "registered keep-all, Scarf-default, and context-derived "
                        "alternatives. Cite every required evidence ID and class. "
                        "Any exclusion affects representation only, never marker "
                        "testing."
                    ),
                )
                if review.question is not None:
                    outcome = journal._complete_attempt(
                        started,
                        status="needsInput",
                        outputs={
                            "decisionSnapshotSha256": review.snapshotSha256,
                            "eligibleFamilies": list(eligible),
                        },
                        needs_input=WorkflowNeedsInput(questions=[review.question]),
                        notes=[
                            "Feature-policy review requires a registered selection."
                        ],
                    )
                    journal._save_outcome(store.zw, prefix, outcome)
                    return outcome, plan, False
                if review.selection is None:
                    raise RuntimeError("Feature-policy review lacks a selection")
                review_selection = review.selection
                selected_option_id = review.selection.selectedOptionId
                revised = review.revised
                snapshot_sha256 = review.snapshotSha256
                payload = (
                    review.resolution.compiled.executorPayload
                    if review.resolution is not None
                    and review.resolution.compiled is not None
                    else active_payload
                )
            else:
                review_selection = DecisionSelection(
                    selectedOptionId="featurePolicy:keepAll",
                    evidenceIds=[loading_id, marker_id, protected_id],
                    rationale=(
                        "No nominated, unprotected family passed the registered "
                        "loading or marker-enrichment gate."
                    ),
                    confidence="notApplicable",
                )
                selected_option_id = "featurePolicy:keepAll"
                revised = False
                _workflow, snapshot_sha256 = self._load_or_create_decision_workflow(
                    store,
                    request_record,
                )
                payload = FeaturePolicyExecutorPayload(
                    policy=active_payload.policy,
                    excludedFamilies=list(active_payload.excludedFamilies),
                    useScarfDefaultBlacklist=(active_payload.useScarfDefaultBlacklist),
                )
            if not isinstance(payload, FeaturePolicyExecutorPayload):
                raise TypeError("Feature-policy review compiled an unexpected payload")
            reviewed_plan = apply_feature_policy_to_plan(plan, payload)
            artifacts: dict[str, ArtifactReferenceModel] = {}
            if (
                loading_evaluation is not None
                and "representationDiagnostic" in loading_evaluation.artifacts
            ):
                artifacts["pcaRepresentationDiagnostic"] = (
                    ArtifactReferenceModel.model_validate(
                        loading_evaluation.artifacts[
                            "representationDiagnostic"
                        ].model_dump()
                    )
                )
            if "markerTable" in selected.artifacts:
                artifacts["clusterMarkerTable"] = ArtifactReferenceModel.model_validate(
                    selected.artifacts["markerTable"].model_dump()
                )
            outcome = journal._complete_attempt(
                started,
                status="done",
                artifacts=artifacts,
                outputs={
                    "preprocessingPlan": reviewed_plan.model_dump(mode="json"),
                    "revised": revised,
                    "selectedOptionId": selected_option_id,
                    "decisionSelection": review_selection.model_dump(mode="json"),
                    "evidenceBundle": bundle.model_dump(mode="json"),
                    "eligibleFamilies": list(eligible),
                    "decisionSnapshotSha256": snapshot_sha256,
                },
                actions=[
                    "review_feature_policy",
                    ("revise_feature_policy" if revised else "retain_feature_policy"),
                ],
            )
            journal._save_outcome(store.zw, prefix, outcome)
            return outcome, reviewed_plan, revised
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
            )
            return outcome, plan, False

    def reuse_feature_policy_tuning_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        baseline_outcome: WorkflowStageAttempt,
        baseline_report: ParameterTuningReport,
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, ParameterTuningReport]:
        """Record deterministic reuse when no feature-policy revision occurred."""
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "feature_policy_tuning",
            request_record,
            parents,
        )
        if existing is not None:
            return existing, baseline_report
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "feature_policy_tuning",
            request_record,
            parents,
            inputs={
                "baselineAttemptId": baseline_outcome.attemptId,
                "baselineReportReferences": [
                    value.model_dump(mode="json")
                    for value in baseline_outcome.reportReferences
                ],
            },
            resume_record=resume_record,
        )
        outcome = journal._complete_attempt(
            started,
            status="done",
            artifacts=dict(baseline_outcome.artifacts),
            outputs={
                "reusedBaselineAttemptId": baseline_outcome.attemptId,
                "recommendedByAssay": dict(baseline_report.recommendedByAssay),
                "operations": [
                    {
                        "operation": "reuse_baseline_parameter_tuning",
                        "attemptId": baseline_outcome.attemptId,
                    }
                ],
            },
            actions=["reuse_baseline_parameter_tuning"],
        )
        journal._save_outcome(store.zw, prefix, outcome)
        return outcome, baseline_report

    def analysis_review_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        tuning_report: ParameterTuningReport,
        tuning_reference: AgentReportReference,
        study_contract: StudyContract,
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[
        WorkflowStageAttempt,
        ParameterTuningReport,
        AgentReportReference,
    ]:
        """Reconsider one dominated analysis checkpoint through the revision ledger."""
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "analysis_review",
            request_record,
            parents,
        )
        if existing is not None:
            if existing.reportReferences:
                loaded = journal.load_stage_report(
                    store,
                    existing,
                    ParameterTuningReport,
                )
                return (
                    existing,
                    cast(ParameterTuningReport, loaded),
                    existing.reportReferences[0],
                )
            return existing, tuning_report, tuning_reference
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "analysis_review",
            request_record,
            parents,
            inputs={
                "parameterReport": tuning_reference.model_dump(mode="json"),
                "studyContractSha256": hashlib.sha256(
                    record_io.canonical_json_bytes(
                        study_contract.model_dump(mode="json")
                    )
                ).hexdigest(),
                "reviewPolicy": {
                    "maximumRevisions": request_record.config.maxRevisions,
                    "dominanceTolerance": 0.02,
                    "materialDifference": 0.05,
                },
            },
            resume_record=resume_record,
        )
        try:
            assay_report = tuning_report.assayReports.get(
                plan.primaryAssay,
                tuning_report,
            )
            selected = next(
                (
                    evaluation
                    for evaluation in assay_report.evaluations
                    if evaluation.candidateId == assay_report.recommendedCandidateId
                ),
                None,
            )
            if selected is None:
                raise ValueError("Analysis review lacks the selected tuning candidate")
            visual_answer = answers.get("analysisVisualReview")
            if visual_answer is not None:
                if not isinstance(visual_answer, Mapping):
                    raise ValueError("analysisVisualReview answer must be a mapping")
                visual_review = AnalysisVisualAdjudication.model_validate(
                    dict(visual_answer)
                )
                adjudication_mode: Literal["multimodal", "numeric", "provided"] = (
                    "provided"
                )
            else:
                try:
                    visual_content = _analysis_visual_content(
                        store,
                        selected,
                        assay_report.evaluations,
                        qc_columns=plan.cellQc.attributes,
                        qc_artifact_metrics=[
                            (value.name, value.artifact)
                            for value in plan.cellQc.artifactMetrics
                        ],
                    )
                    visual_review, adjudication_mode = _run_analysis_adjudication(
                        model=self.model,
                        config=request_record.config,
                        study_objective=request_record.request.studyObjective,
                        selected=selected,
                        candidates=assay_report.evaluations,
                        visual_content=visual_content,
                    )
                except (AgentRunError, RuntimeError, ValueError) as exc:
                    evidence_ids = list(
                        dict.fromkeys(
                            [
                                *selected.evidenceIds,
                                *(
                                    f"artifact:{value.artifactId}"
                                    for value in self._evaluation_artifacts(selected)
                                ),
                                *(
                                    f"artifact:{value.artifact.artifactId}"
                                    for value in plan.cellQc.artifactMetrics
                                ),
                            ]
                        )
                    )
                    if request_record.config.inputPolicy == "unattended":
                        visual_review = AnalysisVisualAdjudication(
                            status="acceptable",
                            selectedCandidateId=selected.candidateId,
                            featureLevelFindings=[
                                "Model adjudication was unavailable; deterministic "
                                "candidate gates remained authoritative."
                            ],
                            rationale=(
                                "The selected candidate already passed the registered "
                                "geometric, stability, marker, cross-unit, technical, "
                                "protected-variable, QC, and doublet gates. The "
                                "unattended workflow retained it after model review "
                                f"failed with {type(exc).__name__}."
                            ),
                        )
                        adjudication_mode = "numeric"
                    else:
                        question = WorkflowQuestion(
                            questionId="analysisVisualReview",
                            question=(
                                "Analysis adjudication could not be completed. Review "
                                "the selected PCA, partition, marker, QC, and doublet "
                                "artifacts and provide an acceptable or concern result "
                                f"for candidate {selected.candidateId!r}. Cause: {exc}"
                            ),
                            options=["acceptable", "concern"],
                            evidenceIds=evidence_ids,
                        )
                        outcome = journal._complete_attempt(
                            started,
                            status="needsInput",
                            outputs={
                                "revised": False,
                                "reviewedCandidateId": selected.candidateId,
                                "visualEvidenceIds": evidence_ids,
                            },
                            needs_input=WorkflowNeedsInput(questions=[question]),
                            actions=[
                                "review_analysis_evidence",
                                "pause_visual_review",
                            ],
                        )
                        journal._save_outcome(store.zw, prefix, outcome)
                        return outcome, tuning_report, tuning_reference
            if visual_review.selectedCandidateId != selected.candidateId:
                raise ValueError("Visual review references a stale selected candidate")
            if not visual_review.rationale.strip():
                raise ValueError("Visual review rationale must be non-empty")
            alternatives = [
                evaluation
                for evaluation in assay_report.evaluations
                if _dominates_analysis_choice(evaluation, selected)
            ]
            if not alternatives:
                if visual_review.status == "concern":
                    concern_answer = answers.get("analysisVisualConcern")
                    if concern_answer == "stop":
                        outcome = journal._complete_attempt(
                            started,
                            status="abstained",
                            outputs={
                                "revised": False,
                                "reviewedCandidateId": selected.candidateId,
                                "adjudicationMode": adjudication_mode,
                                "visualAdjudication": visual_review.model_dump(
                                    mode="json"
                                ),
                            },
                            actions=[
                                "review_analysis_evidence",
                                "stop_on_visual_concern",
                            ],
                        )
                        journal._save_outcome(store.zw, prefix, outcome)
                        return outcome, tuning_report, tuning_reference
                    if concern_answer == "retainSelected":
                        visual_review = visual_review.model_copy(
                            update={
                                "status": "acceptable",
                                "rationale": (
                                    f"{visual_review.rationale} Human review "
                                    "retained the exact selected partition."
                                ),
                            }
                        )
                    elif concern_answer is not None:
                        raise ValueError(
                            "analysisVisualConcern must be retainSelected or stop"
                        )
                if visual_review.status == "concern":
                    if request_record.config.inputPolicy == "unattended":
                        visual_review = visual_review.model_copy(
                            update={
                                "status": "acceptable",
                                "rationale": (
                                    f"{visual_review.rationale} No executed matched "
                                    "alternative passed the deterministic dominance "
                                    "gate, so the unattended workflow retained the "
                                    "selected partition."
                                ),
                            }
                        )
                    else:
                        question = WorkflowQuestion(
                            questionId="analysisVisualConcern",
                            question=(
                                "Visual adjudication found a concern, but no executed "
                                "matched alternative passed the deterministic "
                                "dominance gate. Decide whether to retain the selected "
                                "partition or stop for a revised analysis request."
                            ),
                            options=["retainSelected", "stop"],
                            evidenceIds=list(selected.evidenceIds),
                        )
                        outcome = journal._complete_attempt(
                            started,
                            status="needsInput",
                            outputs={
                                "revised": False,
                                "reviewedCandidateId": selected.candidateId,
                                "adjudicationMode": adjudication_mode,
                                "visualAdjudication": visual_review.model_dump(
                                    mode="json"
                                ),
                            },
                            needs_input=WorkflowNeedsInput(questions=[question]),
                            actions=[
                                "review_analysis_evidence",
                                "pause_visual_concern",
                            ],
                        )
                        journal._save_outcome(store.zw, prefix, outcome)
                        return outcome, tuning_report, tuning_reference
                outcome = journal._complete_attempt(
                    started,
                    status="done",
                    outputs={
                        "revised": False,
                        "reviewedCandidateId": selected.candidateId,
                        "adjudicationMode": adjudication_mode,
                        "stoppingReason": (
                            "No eligible one-checkpoint alternative dominated the "
                            "selected candidate across geometric, stability, marker, "
                            "cross-unit, technical, protected-variable, and doublet "
                            "evidence."
                        ),
                        "visualAdjudication": visual_review.model_dump(mode="json"),
                    },
                    actions=["review_analysis_evidence", "retain_selected_analysis"],
                )
                journal._save_outcome(store.zw, prefix, outcome)
                return outcome, tuning_report, tuning_reference

            best = max(
                alternatives,
                key=lambda evaluation: (
                    sum(_cluster_review_values(evaluation)[0].values())
                    - sum(_cluster_review_values(evaluation)[1].values()),
                    -evaluation.parameters.leidenResolution,
                ),
            )
            checkpoint = _changed_analysis_checkpoint(best, selected)
            if checkpoint is None:
                raise RuntimeError(
                    "Dominating analysis alternative lacks one exact checkpoint"
                )
            raw_candidates = [
                selected,
                *[
                    value
                    for value in alternatives
                    if _changed_analysis_checkpoint(value, selected) == checkpoint
                ],
            ]
            candidates_by_value: dict[Any, ParameterCandidateEvaluation] = {}
            for candidate in raw_candidates:
                value = _analysis_parameter_value(checkpoint, candidate)
                current = candidates_by_value.get(value)
                if current is None or (
                    sum(_cluster_review_values(candidate)[0].values())
                    - sum(_cluster_review_values(candidate)[1].values())
                    > sum(_cluster_review_values(current)[0].values())
                    - sum(_cluster_review_values(current)[1].values())
                ):
                    candidates_by_value[value] = candidate
            candidates = list(candidates_by_value.values())
            evidence: list[DecisionEvidence] = []
            candidate_evidence: dict[str, list[str]] = {}
            for evaluation in candidates:
                values = self._analysis_candidate_evidence(
                    checkpoint,
                    evaluation,
                )
                evidence.extend(values)
                candidate_evidence[evaluation.candidateId] = [
                    value.evidenceId for value in values
                ]
            best = max(
                alternatives,
                key=lambda evaluation: (
                    sum(_cluster_review_values(evaluation)[0].values())
                    - sum(_cluster_review_values(evaluation)[1].values()),
                    -evaluation.parameters.leidenResolution,
                ),
            )
            bundle = self._tuning_evidence_bundle(checkpoint, evidence)
            if checkpoint == "pcaPrefix":
                definition = build_pca_prefix_decision(
                    evidence_bundle_id=bundle.bundleId,
                    matrix_rank=max(
                        evaluation.parameters.dimensions for evaluation in candidates
                    ),
                    candidate_dimensions=[
                        evaluation.parameters.dimensions for evaluation in candidates
                    ],
                )
                option_for_candidate = {
                    evaluation.candidateId: self._payload_option_id(
                        definition,
                        PcaPrefixExecutorPayload,
                        "dimensions",
                        evaluation.parameters.dimensions,
                    )
                    for evaluation in candidates
                }
            elif checkpoint == "correctionOutcome":
                definition = build_correction_outcome_decision(
                    evidence_bundle_id=bundle.bundleId,
                    license="safe",
                    need="needed",
                    harmony_eligible=True,
                )
                option_for_candidate = {
                    evaluation.candidateId: (
                        "correctionOutcome:acceptHarmony"
                        if evaluation.parameters.useHarmony
                        else "correctionOutcome:retainNative"
                    )
                    for evaluation in candidates
                }
            elif checkpoint == "graphK":
                definition = build_graph_k_decision(
                    evidence_bundle_id=bundle.bundleId,
                    n_cells=max(
                        evaluation.parameters.neighborsK for evaluation in candidates
                    )
                    + 1,
                    candidate_neighbors=[
                        evaluation.parameters.neighborsK for evaluation in candidates
                    ],
                )
                option_for_candidate = {
                    evaluation.candidateId: self._payload_option_id(
                        definition,
                        GraphExecutorPayload,
                        "neighborsK",
                        evaluation.parameters.neighborsK,
                    )
                    for evaluation in candidates
                }
            else:
                known_ids = {
                    0.25: "clusterResolution:veryCoarse",
                    0.5: "clusterResolution:coarse",
                    0.75: "clusterResolution:balanced",
                    1.0: "clusterResolution:detailed",
                    1.25: "clusterResolution:fine",
                    1.5: "clusterResolution:veryFine",
                }

                def resolution_option(value: float) -> str:
                    return known_ids.get(
                        value,
                        f"clusterResolution:r{str(value).replace('.', 'p')}",
                    )

                definition = build_cluster_partition_decision(
                    evidence_bundle_id=bundle.bundleId,
                    metric_preferred_option_id=resolution_option(
                        best.parameters.leidenResolution
                    ),
                    resolution_candidates=[
                        evaluation.parameters.leidenResolution
                        for evaluation in candidates
                    ],
                )
                option_for_candidate = {
                    evaluation.candidateId: resolution_option(
                        evaluation.parameters.leidenResolution
                    )
                    for evaluation in candidates
                }
            requirements = {
                option_for_candidate[evaluation.candidateId]: candidate_evidence[
                    evaluation.candidateId
                ]
                for evaluation in candidates
            }
            definition = require_option_evidence(definition, requirements)
            decision_workflow, _snapshot = self._load_or_create_decision_workflow(
                store,
                request_record,
            )
            previous_options = {
                record.decisionId: record.selectedOptionId
                for record in decision_workflow.active_decision_records()
            }
            review = self._reconsider_rna_decision(
                store,
                request_record,
                definition,
                bundle,
                answers,
                review_instructions=(
                    f"Reconsider the active {checkpoint} decision only because an "
                    "executed one-checkpoint alternative passed the registered "
                    "dominance gate. Cite the option-specific geometric, technical, "
                    "stability, marker, protected-variable, cross-unit, and "
                    "quality-control evidence. Keep the current option unless at "
                    "least two independent evidence classes justify replacement."
                ),
            )
            if review.question is not None:
                outcome = journal._complete_attempt(
                    started,
                    status="needsInput",
                    outputs={
                        "revised": False,
                        "adjudicationMode": adjudication_mode,
                        "decisionSnapshotSha256": review.snapshotSha256,
                        "dominatingCandidateIds": [
                            value.candidateId for value in alternatives
                        ],
                    },
                    needs_input=WorkflowNeedsInput(questions=[review.question]),
                    actions=["review_analysis_evidence"],
                )
                journal._save_outcome(store.zw, prefix, outcome)
                return outcome, tuning_report, tuning_reference
            if review.selection is None:
                raise RuntimeError("Analysis review returned no selection")
            replacement = next(
                evaluation
                for evaluation in candidates
                if option_for_candidate[evaluation.candidateId]
                == review.selection.selectedOptionId
            )
            if not review.revised:
                outcome = journal._complete_attempt(
                    started,
                    status="done",
                    outputs={
                        "revised": False,
                        "reviewedCandidateId": selected.candidateId,
                        "adjudicationMode": adjudication_mode,
                        "decisionSelection": review.selection.model_dump(mode="json"),
                        "decisionSnapshotSha256": review.snapshotSha256,
                        "visualAdjudication": visual_review.model_dump(mode="json"),
                    },
                    actions=["review_analysis_evidence", "retain_selected_analysis"],
                )
                journal._save_outcome(store.zw, prefix, outcome)
                return outcome, tuning_report, tuning_reference

            descendant_snapshot = self._restore_tuning_descendants(
                store,
                request_record,
                study_contract,
                replacement,
                revised_checkpoint=checkpoint,
                previous_options=previous_options,
                model_name=(
                    tuning_report.runInfo.modelName
                    or assay_report.runInfo.modelName
                    or None
                ),
            )
            updated_assay = assay_report.model_copy(
                update={
                    "recommendedCandidateId": replacement.candidateId,
                    "selectedArtifacts": dict(replacement.artifacts),
                    "evidenceIds": list(review.selection.evidenceIds),
                    "rationale": review.selection.rationale,
                    "tradeoffs": [
                        *assay_report.tradeoffs,
                        (
                            "The bounded analysis review superseded "
                            f"{selected.candidateId!r} with "
                            f"{replacement.candidateId!r}."
                        ),
                    ],
                }
            )
            reports = dict(tuning_report.assayReports)
            reports[plan.primaryAssay] = updated_assay
            updated_report = tuning_report.model_copy(
                update={
                    "assayReports": reports,
                    "recommendedByAssay": {
                        **tuning_report.recommendedByAssay,
                        plan.primaryAssay: replacement.candidateId,
                    },
                    **(
                        {
                            "recommendedCandidateId": replacement.candidateId,
                            "selectedArtifacts": dict(replacement.artifacts),
                            "evidenceIds": list(review.selection.evidenceIds),
                            "rationale": review.selection.rationale,
                        }
                        if tuning_report.fromAssay == plan.primaryAssay
                        else {}
                    ),
                }
            )
            final_selection = updated_report.finalSelection
            if final_selection is not None:
                final_selection = final_selection.model_copy(
                    update={
                        "selectedOptionId": (
                            f"native:{plan.primaryAssay}:{replacement.candidateId}"
                        ),
                        "nativeAssay": plan.primaryAssay,
                        "nativeCandidateId": replacement.candidateId,
                        "integrationId": None,
                        "evidenceIds": list(review.selection.evidenceIds),
                        "rationale": review.selection.rationale,
                    }
                )
            updated_report = finalize_parameter_tuning_selection(
                updated_report,
                marker_assay=plan.markerAssay,
                native_assay=plan.primaryAssay,
                final_selection=final_selection,
            )
            stage_artifacts = {
                name: ArtifactReferenceModel.model_validate(value.model_dump())
                for name, value in replacement.artifacts.items()
            }
            saved, reference = journal._save_stage_report(
                store,
                started,
                updated_report,
                invocation=AgentInvocation(
                    agentName="parameter_tuning",
                    parentReports=[journal._report_link(tuning_reference)],
                    inputs={
                        "selectedCandidateId": selected.candidateId,
                        "replacementCandidateId": replacement.candidateId,
                        "revisedCheckpoint": checkpoint,
                        "evidenceBundle": bundle.model_dump(mode="json"),
                        "adjudicationMode": adjudication_mode,
                        "visualAdjudication": visual_review.model_dump(mode="json"),
                    },
                    artifacts=stage_artifacts,
                    runConfig=request_record.config.agentRunConfig,
                ),
                expected_type=ParameterTuningReport,
            )
            updated_report = cast(ParameterTuningReport, saved)
            outcome = journal._complete_attempt(
                started,
                status="done",
                report_references=[reference],
                artifacts=stage_artifacts,
                outputs={
                    "revised": True,
                    "selectedCandidateId": selected.candidateId,
                    "replacementCandidateId": replacement.candidateId,
                    "revisedCheckpoint": checkpoint,
                    "adjudicationMode": adjudication_mode,
                    "decisionSelection": review.selection.model_dump(mode="json"),
                    "decisionSnapshotSha256": (
                        descendant_snapshot or review.snapshotSha256
                    ),
                    "visualAdjudication": visual_review.model_dump(mode="json"),
                },
                actions=[
                    "review_analysis_evidence",
                    f"revise_{checkpoint}",
                    "recompute_invalidated_tuning_decisions",
                ],
            )
            journal._save_outcome(store.zw, prefix, outcome)
            return outcome, updated_report, reference
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
            )
            return outcome, tuning_report, tuning_reference

    def parameter_tuning_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        preprocessed: Sequence[PreprocessedAssayHandoff],
        experimental: ExperimentalContextResult,
        enrichment_reference: AgentReportReference,
        experimental_reference: AgentReportReference,
        answers: Mapping[str, Any],
        *,
        study_contract: StudyContract | None = None,
        resume_record: OrchestrationResumeRecord | None = None,
        stage_name: WorkflowStageName = "parameter_tuning",
    ) -> tuple[WorkflowStageAttempt, ParameterTuningReport]:
        prefix = journal._ensure_orchestration_store(store)
        cell_selection = preprocessed[0].cellSelection if preprocessed else None
        if cell_selection is None or any(
            value.cellSelection != cell_selection for value in preprocessed
        ):
            raise ValueError("Preprocessed assays must share one exact cell selection")
        experimental_handoff = experimental.to_parameter_tuning_handoff().model_copy(
            update={"cellSelection": cell_selection}
        )
        metadata_columns = {
            *experimental_handoff.batchColumns,
            *experimental_handoff.preservationColumns,
            *plan.cellQc.attributes,
        }
        if study_contract is not None:
            metadata_columns.update(study_contract.technicalBatchColumns)
            metadata_columns.update(study_contract.protectedColumns)
            metadata_columns.update(study_contract.independentUnitColumns)
            if study_contract.physicalCaptureColumn is not None:
                metadata_columns.add(study_contract.physicalCaptureColumn)
        metadata_fingerprints = {
            column: (
                _metadata_column_fingerprint(store.cells, column)
                if column in store.cells.columns
                else None
            )
            for column in sorted(metadata_columns)
        }
        feature_metadata = store.get_assay(plan.primaryAssay).feats
        feature_metadata_fingerprints = {
            column: _metadata_column_fingerprint(feature_metadata, column)
            for column in ("ids", "names")
        }
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            stage_name,
            request_record,
            parents,
        )
        if existing is not None:
            if (
                existing.inputs.get("metadataFingerprints") != metadata_fingerprints
                or existing.inputs.get("featureMetadataFingerprints")
                != feature_metadata_fingerprints
            ):
                raise ValueError(
                    "Tuning metadata changed since the saved evidence was computed; "
                    "restore the original metadata or start a new workflow"
                )
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing Parameter Tuning report"
            )
            report = journal.load_stage_report(store, existing, ParameterTuningReport)
            return existing, cast(ParameterTuningReport, report)
        paused = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            stage_name,
            request_record,
            parents,
            required_status="needsInput",
        )
        prior_sequential: SequentialAssayTuningEvidence | None = None
        if paused is not None and paused.outputs.get("sequentialEvidence") is not None:
            prior_sequential = SequentialAssayTuningEvidence.model_validate(
                paused.outputs["sequentialEvidence"]
            )
        if paused is not None and (
            paused.inputs.get("metadataFingerprints") != metadata_fingerprints
            or paused.inputs.get("featureMetadataFingerprints")
            != feature_metadata_fingerprints
        ):
            raise ValueError(
                "Tuning metadata changed since the saved evidence was computed; "
                "restore the original metadata or start a new workflow"
            )
        tuning_answer = answers.get("parameter_tuning")
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            stage_name,
            request_record,
            parents,
            inputs={
                "preprocessedAssays": [
                    value.model_dump(mode="json") for value in preprocessed
                ],
                "experimentalTuningHandoff": experimental_handoff.model_dump(
                    mode="json"
                ),
                "cellSelection": cell_selection.model_dump(mode="json"),
                "primaryAssay": plan.primaryAssay,
                "markerAssay": plan.markerAssay,
                "pairedAssays": plan.pairedAssays,
                "finalGraphOptionId": answers.get("finalGraphOptionId"),
                "parameterTuning": tuning_answer,
                "studyObjective": request_record.request.studyObjective,
                "metadataFingerprints": metadata_fingerprints,
                "featureMetadataFingerprints": feature_metadata_fingerprints,
                "resumeFromAttempt": (paused.attemptId if paused is not None else None),
            },
            resume_record=resume_record,
        )
        report = ParameterTuningReport.get_blank()
        actions: list[str] = []
        candidate_payload: dict[str, list[dict[str, Any]]] = {}
        logger.info(
            f"Workflow {workflow.workflowRunId}: Parameter Tuning started for "
            f"{len(preprocessed)} RNA assay(s)"
        )
        try:
            agent = ParameterTuningAgent(
                self.model,
                config=request_record.config.agentRunConfig,
            )
            recovered = (
                None
                if paused is not None
                else journal._recover_persisted_stage_report(
                    store,
                    started,
                    agent_name="parameter_tuning",
                    expected_type=ParameterTuningReport,
                )
            )
            if recovered is not None:
                recovered_report, recovered_reference = recovered
                report = cast(ParameterTuningReport, recovered_report)
                if report.integrationEvaluations or report.recommendedIntegrationId:
                    raise ValueError(
                        "Saved tuning report contains unsupported integration"
                    )
                candidate_payload = {
                    assay: [
                        evaluation.parameters.model_dump(mode="json")
                        for evaluation in assay_report.evaluations
                    ]
                    for assay, assay_report in report.assayReports.items()
                }
                actions.append("recover_persisted_parameter_tuning_report")
                logger.info(
                    f"Workflow {workflow.workflowRunId}: recovering completed "
                    "Parameter Tuning provider result"
                )
                return self.save_parameter_tuning_outcome(
                    store,
                    prefix,
                    workflow,
                    request_record,
                    started,
                    report,
                    plan,
                    preprocessed,
                    candidate_payload,
                    enrichment_reference,
                    experimental_reference,
                    experimental_handoff,
                    agent,
                    actions,
                    persisted_reference=recovered_reference,
                )
            if len(preprocessed) != 1 or plan.pairedAssays:
                raise ValueError("Automated parameter tuning requires one RNA assay")
            if study_contract is None:
                raise ValueError("Decision-driven RNA tuning requires a StudyContract")
            with candidate_metric_cache():
                report, sequential_evidence = self._run_sequential_rna_tuning(
                    store,
                    workflow,
                    request_record,
                    plan,
                    preprocessed,
                    experimental_handoff,
                    study_contract,
                    answers,
                    prior_sequential,
                )
            candidate_payload = {
                sequential_evidence.assay: [
                    evaluation.parameters.model_dump(mode="json")
                    for evaluation in report.evaluations
                ]
            }
            actions.extend(
                f"adjudicate_{phase.plan.phase}" for phase in sequential_evidence.phases
            )
            if report.searchPlan is not None:
                actions.append("review_parameter_refinement")
                actions.extend(
                    f"execute_refined_candidate:{candidate.candidateId}"
                    for candidate in report.searchPlan.candidates
                )
            return self.save_parameter_tuning_outcome(
                store,
                prefix,
                workflow,
                request_record,
                started,
                report,
                plan,
                preprocessed,
                candidate_payload,
                enrichment_reference,
                experimental_reference,
                experimental_handoff,
                agent,
                actions,
                sequential_evidence=sequential_evidence,
            )
        except Exception as exc:
            failure_artifacts: dict[str, ArtifactReferenceModel] = {
                "cellSelection": cell_selection
            }
            for assay, assay_report in report.assayReports.items():
                for evaluation in assay_report.evaluations:
                    for name, artifact in evaluation.artifacts.items():
                        failure_artifacts[
                            f"{assay}_{evaluation.parameters.candidateId}_{name}"
                        ] = ArtifactReferenceModel.model_validate(artifact.model_dump())
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts=failure_artifacts,
                actions=actions,
                outputs={
                    "candidatePlan": candidate_payload,
                },
            )
            return outcome, ParameterTuningReport.get_blank()

    def save_parameter_tuning_outcome(
        self,
        store: DataStore,
        prefix: str,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        started: WorkflowStageAttempt,
        report: ParameterTuningReport,
        plan: AutomatedPreprocessingPlan,
        preprocessed: Sequence[PreprocessedAssayHandoff],
        candidate_payload: Mapping[str, list[dict[str, Any]]],
        enrichment_reference: AgentReportReference,
        experimental_reference: AgentReportReference,
        experimental_handoff: ExperimentalTuningHandoff,
        agent: ParameterTuningAgent,
        actions: Sequence[str],
        *,
        prior_tuning_reference: AgentReportReference | None = None,
        persisted_reference: AgentReportReference | None = None,
        sequential_evidence: SequentialAssayTuningEvidence | None = None,
    ) -> tuple[WorkflowStageAttempt, ParameterTuningReport]:
        if experimental_handoff.cellSelection is None:
            raise ValueError("Parameter tuning handoff lacks an exact cell selection")
        if report.cellSelection != experimental_handoff.cellSelection:
            raise ValueError("Parameter tuning report uses a different cell selection")
        invocation_artifacts: dict[str, ArtifactReferenceModel] = {}
        for value in preprocessed:
            if value.normalized is not None:
                invocation_artifacts[f"{value.assay}_normalized"] = value.normalized
        invocation_artifacts["cellSelection"] = experimental_handoff.cellSelection
        stage_artifacts = dict(invocation_artifacts)
        for assay, assay_report in report.assayReports.items():
            for name, artifact in assay_report.selectedArtifacts.items():
                stage_artifacts[f"{assay}_{name}"] = (
                    ArtifactReferenceModel.model_validate(artifact.model_dump())
                )
        if report.finalClusterArtifact is not None:
            stage_artifacts["final_clusters"] = ArtifactReferenceModel.model_validate(
                report.finalClusterArtifact.model_dump()
            )
        if report.graphAssay is not None:
            assay_reports = report.assayReports or {report.fromAssay: report}
            graph_artifact = assay_reports[report.graphAssay].selectedArtifacts[
                "connectivityMap"
            ]
            stage_artifacts["final_graph"] = ArtifactReferenceModel.model_validate(
                graph_artifact.model_dump()
            )
        if persisted_reference is None:
            saved_report, reference = journal._save_stage_report(
                store,
                started,
                report,
                invocation=AgentInvocation(
                    agentName="parameter_tuning",
                    parentReports=[
                        journal._report_link(enrichment_reference),
                        journal._report_link(experimental_reference),
                        *(
                            [journal._report_link(prior_tuning_reference)]
                            if prior_tuning_reference is not None
                            else []
                        ),
                    ],
                    inputs={
                        "assays": dict(candidate_payload),
                        "primaryAssay": plan.primaryAssay,
                        "markerAssay": plan.markerAssay,
                        "cellSelection": (
                            experimental_handoff.cellSelection.model_dump(mode="json")
                            if experimental_handoff.cellSelection is not None
                            else None
                        ),
                        "maxCandidateEvaluations": (
                            request_record.config.maxCandidateEvaluations
                        ),
                    },
                    artifacts=stage_artifacts,
                    runConfig=agent.config,
                    experimentalTuningHandoff=experimental_handoff,
                ),
                expected_type=ParameterTuningReport,
            )
            report = cast(ParameterTuningReport, saved_report)
        else:
            reference = persisted_reference
        stage_report_references = [reference]
        operations: list[dict[str, Any]] = []
        for assay, assay_report in report.assayReports.items():
            for candidate_evaluation in assay_report.evaluations:
                operations.append(
                    {
                        "operation": "execute_parameter_candidate",
                        "assay": assay,
                        "candidate": candidate_evaluation.parameters.model_dump(
                            mode="json"
                        ),
                        "phase": candidate_evaluation.phase,
                        "cellSelection": (
                            experimental_handoff.cellSelection.model_dump(mode="json")
                        ),
                        "harmonyBatchColumns": list(
                            candidate_evaluation.harmonyBatchColumns
                        ),
                        "identityFeatureLimit": (
                            request_record.config.maxIdentityFeatures
                        ),
                        "status": candidate_evaluation.status,
                        "artifacts": {
                            name: value.model_dump(mode="json")
                            for name, value in candidate_evaluation.artifacts.items()
                        },
                    }
                )
        if (
            report.status == "needsInput"
            and request_record.config.inputPolicy == "unattended"
        ):
            outcome = journal._complete_attempt(
                started,
                status="failed",
                report_references=stage_report_references,
                artifacts=stage_artifacts,
                outputs={
                    "candidateCount": report.totalCandidates,
                    "sequentialEvidence": (
                        sequential_evidence.model_dump(mode="json")
                        if sequential_evidence is not None
                        else None
                    ),
                    "operations": operations,
                },
                actions=actions,
                error=(
                    "The unattended Parameter Tuning stage returned an unresolved "
                    "decision"
                ),
                notes=report.limitations,
            )
        elif report.status == "needsInput":
            needs_input = report.needsInput
            assert needs_input is not None
            if (
                sequential_evidence is not None
                and sequential_evidence.pendingDecisionId is None
                and report.searchPlan is None
            ):
                raise ValueError(
                    "Sequential tuning needsInput lacks a pending decision ID"
                )
            outcome = journal._complete_attempt(
                started,
                status="needsInput",
                report_references=stage_report_references,
                artifacts=stage_artifacts,
                outputs={
                    "candidateCount": report.totalCandidates,
                    "sequentialEvidence": (
                        sequential_evidence.model_dump(mode="json")
                        if sequential_evidence is not None
                        else None
                    ),
                    "operations": operations,
                },
                actions=actions,
                needs_input=WorkflowNeedsInput(
                    questions=[
                        WorkflowQuestion(
                            questionId=(
                                f"decision:{sequential_evidence.pendingDecisionId}"
                                if sequential_evidence is not None
                                and sequential_evidence.pendingDecisionId is not None
                                else "finalGraphOptionId"
                                if report.finalSelection is not None
                                and report.finalSelection.status == "needsInput"
                                else "parameter_tuning"
                            ),
                            decisionId=(
                                sequential_evidence.pendingDecisionId
                                if sequential_evidence is not None
                                else None
                            ),
                            question=needs_input.question,
                            options=list(needs_input.options),
                            evidenceIds=list(needs_input.evidenceIds),
                        )
                    ]
                ),
                notes=report.limitations,
            )
        elif report.status == "abstained":
            outcome = journal._complete_attempt(
                started,
                status="abstained",
                report_references=stage_report_references,
                artifacts=stage_artifacts,
                outputs={
                    "candidateCount": report.totalCandidates,
                    "sequentialEvidence": (
                        sequential_evidence.model_dump(mode="json")
                        if sequential_evidence is not None
                        else None
                    ),
                    "operations": operations,
                },
                actions=actions,
                notes=(
                    report.limitations
                    or ["No defensible discrete clustering partition was found."]
                ),
            )
        elif report.status == "failed":
            outcome = journal._complete_attempt(
                started,
                status="failed",
                report_references=stage_report_references,
                artifacts=stage_artifacts,
                outputs={
                    "candidateCount": report.totalCandidates,
                    "sequentialEvidence": (
                        sequential_evidence.model_dump(mode="json")
                        if sequential_evidence is not None
                        else None
                    ),
                    "operations": operations,
                },
                actions=actions,
                error="; ".join(report.limitations) or "Parameter Tuning failed",
            )
        else:
            outcome = journal._complete_attempt(
                started,
                status="done",
                report_references=stage_report_references,
                artifacts=stage_artifacts,
                outputs={
                    "candidateCount": report.totalCandidates,
                    "recommendedByAssay": report.recommendedByAssay,
                    "recommendedIntegrationId": report.recommendedIntegrationId,
                    "sequentialEvidence": (
                        sequential_evidence.model_dump(mode="json")
                        if sequential_evidence is not None
                        else None
                    ),
                    "operations": operations,
                },
                actions=actions,
                notes=[*report.tradeoffs, *report.limitations],
            )
        journal._save_outcome(store.zw, prefix, outcome)
        logger.info(
            f"Workflow {workflow.workflowRunId}: Parameter Tuning outcome "
            f"status={outcome.status!r}, candidates={report.totalCandidates}"
        )
        if outcome.status == "failed":
            journal.finalize_failed(store, workflow, outcome.error or "tuning failed")
        return outcome, report
