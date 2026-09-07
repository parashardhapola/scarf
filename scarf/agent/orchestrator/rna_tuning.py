"""Objective-led RNA experiments on frozen screening and full-cohort cells."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import Field, create_model, model_validator

from ...metadata.rows import read_metadata_rows_chunkwise
from ...storage.refs import ArtifactRef
from ...storage.selections import (
    read_stored_selection_indices,
    resolve_generated_selection_artifact,
)
from ...utils.logging import logger
from .. import record_io
from ..config.agent_exec import (
    ImageInputUnsupportedError,
    build_visual_evidence_prompt,
    run_agent_sync,
)
from ..experimental_context.study import (
    StudyContract,
    unsupported_comparison_limitations,
)
from ..experimental_context.contracts import CovariateComparison
from ..parameter_tuning.agent import prepare_parameter_tuning_dependencies
from ..parameter_tuning.contracts import (
    ArtifactRecord,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterTuningNeedsInput,
    ParameterTuningReport,
)
from ..parameter_tuning.diagnostics import (
    SCARF_DEFAULT_DIAGNOSTIC_FAMILIES,
    _family_mask,
    _neighbor_overlap,
    augment_cluster_evaluations,
    augment_pca_evaluations,
    population_support_evidence,
    score_advisory_doublets,
)
from ..parameter_tuning.execution import execute_parameter_candidate
from ..parameter_tuning.hvg import (
    HvgGroupVariability,
    aggregate_hvg_rankings,
    rank_core_hvgs,
)
from ..parameter_tuning.selection import (
    finalize_parameter_tuning_selection,
    harmony_acceptance_gate,
)
from ..types import AgentDataModel, ArtifactReferenceModel
from . import journal
from .budget import CandidateBudget, CandidateBudgetExceeded, candidate_identity
from .models import (
    AutomatedPreprocessingPlan,
    OrchestrationRequestRecord,
    PreprocessedAssayHandoff,
    artifact_model_to_ref,
)


_DOMAINS = {
    "qualityControl",
    "featurePolicy",
    "hvgRankingAndCount",
    "pca",
    "batchCorrection",
    "neighbors",
    "partition",
    "rarePopulations",
}

_STRUCTURED_VISUAL_LIMITATION = (
    "The model assessed structured marker, PCA loading and diagnostic evidence; "
    "visual inspection was unavailable because the configured model does not accept images."
)


def _configured_image_input(model: Any) -> bool | None:
    """Honor an explicit input capability without inferring it from image output."""
    declared = getattr(model, "supports_image_input", None)
    if isinstance(declared, bool):
        return declared
    profile = getattr(model, "profile", None)
    if isinstance(profile, Mapping):
        declared = profile.get("supports_image_input")
        if isinstance(declared, bool):
            return declared
    return None


class TuningAction(AgentDataModel):
    """An assessment of observed evidence and at most one registered experiment."""

    action: Literal["accept", "experiment", "enlarge", "defer"] = Field(
        description=(
            "Accept supported observed evidence, request one next experiment, "
            "enlarge a screening sample, or defer an unresolved essential question."
        )
    )
    selectedCandidateId: str = Field(
        description=(
            "Copy an observed candidate ID. An experiment must keep "
            "currentCandidateId as its fixed baseline."
        )
    )
    experimentId: str | None = Field(
        default=None,
        description=(
            "For action=experiment, copy one exact key from experiments. "
            "This requests the next operation; it has not yet run. Otherwise null."
        ),
    )
    correctionNeed: Literal["needed", "notNeeded", "uncertain", "notApplicable"]
    assessedDomains: list[str]
    evidenceIds: list[str] = Field(
        min_length=1,
        description=(
            "Copy exact IDs from availableEvidenceIds, including the selected "
            "candidate's anchor or one of its supplied diagnostic evidence IDs."
        ),
    )
    quantitativeFindings: list[str] = Field(
        min_length=1,
        description="Describe supplied observed measurements, not predicted results.",
    )
    qualitativeFindings: list[str] = Field(min_length=1)
    concern: str = ""
    expectedImprovement: str = Field(
        default="",
        description="Predict what the requested experiment should improve and why.",
    )
    objectivePreservation: str = Field(min_length=1)
    rationale: str = Field(
        min_length=1,
        description=(
            "Explain why this action follows from observed evidence. "
            "Do not describe an unexecuted experiment as a completed result."
        ),
    )

    @model_validator(mode="after")
    def validate_action(self) -> "TuningAction":
        if (self.action == "experiment") != (self.experimentId is not None):
            raise ValueError("Only an experiment action names an experiment")
        if self.action == "experiment" and (
            not self.concern.strip() or not self.expectedImprovement.strip()
        ):
            raise ValueError(
                "An experiment needs an observed concern and expected improvement"
            )
        if self.action == "accept" and set(self.assessedDomains) != _DOMAINS:
            raise ValueError(
                "Acceptance requires assessment of every scientific domain"
            )
        return self


def _assessment_output_type(
    candidate_ids: Sequence[str], experiment_ids: Sequence[str], *, scope: str
) -> type[TuningAction]:
    """Constrain new model choices without changing the saved action contract."""
    if not candidate_ids:
        raise ValueError("RNA assessment requires observed candidates")
    actions = tuple(
        action
        for action in ("accept", "experiment", "enlarge", "defer")
        if (action != "enlarge" or scope != "full")
        and (action != "experiment" or experiment_ids)
    )
    return create_model(
        "ObservedRnaAssessment",
        __base__=TuningAction,
        action=(
            Literal[actions],
            Field(description=TuningAction.model_fields["action"].description),
        ),
        selectedCandidateId=(
            Literal[tuple(dict.fromkeys(candidate_ids))],
            Field(
                description=TuningAction.model_fields["selectedCandidateId"].description
            ),
        ),
        experimentId=(
            Literal[tuple(dict.fromkeys(experiment_ids))] | None
            if experiment_ids
            else type(None),
            Field(
                default=None,
                description=TuningAction.model_fields["experimentId"].description,
            ),
        ),
    )


class RnaSetting(AgentDataModel):
    parameters: ParameterCandidate
    features: ArtifactReferenceModel
    eligibleFeatures: ArtifactReferenceModel
    hvgCount: int = 1000
    ranking: Literal["global", "batchAware"] = "global"
    rankingColumn: str | None = None


def uniform_screening_selection(
    store: Any, parent: ArtifactRef, *, size: int, seed: int
) -> ArtifactRef:
    """Select nested uniform prefixes without weighting study groups differently."""
    indices = read_stored_selection_indices(
        store.zw,
        parent,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    if size >= len(indices):
        return parent
    if size < 3:
        raise ValueError("Screening requires at least three cells")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(indices))
    mask = np.zeros(store.cells.N, dtype=bool)
    mask[indices[order[:size]]] = True
    selection, _ = resolve_generated_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=mask,
        row_ids=np.asarray(store.cells.fetch_all("ids")),
        operation="agent_uniform_rna_screen",
        parameters={"size": size, "seed": seed, "generator": "PCG64"},
        inputs={"parent_selection": parent},
        source_column="agent_screening",
    )
    return selection


def screening_coverage(
    store: Any,
    parent: ArtifactRef,
    sample: ArtifactRef,
    columns: Sequence[str],
    combinations: Sequence[Sequence[str]] = (),
) -> tuple[dict[str, Any], list[str]]:
    """Report population and sample proportions without oversampling rare groups."""

    def indices(selection: ArtifactRef) -> np.ndarray:
        return read_stored_selection_indices(
            store.zw,
            selection,
            kind="cell_selection",
            scope="datastore",
            assay=None,
            table_path="cellData",
        ).astype(np.int64, copy=False)

    full_indices, sample_indices = indices(parent), indices(sample)
    evidence: dict[str, Any] = {
        "populationCells": len(full_indices),
        "screeningCells": len(sample_indices),
        "sampling": "Uniform without replacement; no group oversampling or weights.",
        "groups": {},
    }
    concerns: list[str] = []
    grouped = {}
    for column in dict.fromkeys(columns):
        full = np.asarray(
            read_metadata_rows_chunkwise(store.cells, column, full_indices)
        ).astype(str)
        sampled = np.asarray(
            read_metadata_rows_chunkwise(store.cells, column, sample_indices)
        ).astype(str)
        grouped[column] = (full, sampled)
    if combinations:
        from ..experimental_context.characterization import _SelectionBoundCells
        from ..experimental_context.comparisons import combination_labels

        full_cells = _SelectionBoundCells(store.zw, store.cells, parent)
        sample_cells = _SelectionBoundCells(store.zw, store.cells, sample)
        for group_columns in combinations:
            key = "joint:" + json.dumps(list(group_columns), separators=(",", ":"))
            grouped[key] = (
                combination_labels(full_cells, group_columns),
                combination_labels(sample_cells, group_columns),
            )
    for column, (full, sampled) in grouped.items():
        values, counts = np.unique(full, return_counts=True)
        sample_values, sample_counts = np.unique(sampled, return_counts=True)
        lookup = dict(zip(sample_values, sample_counts, strict=True))
        rows = []
        for value, count in zip(values, counts, strict=True):
            observed = int(lookup.get(value, 0))
            rows.append(
                {
                    "value": str(value),
                    "populationCells": int(count),
                    "screeningCells": observed,
                    "populationFraction": float(count / len(full)),
                    "screeningFraction": observed / len(sampled),
                }
            )
            if observed < min(int(count), 20):
                concerns.append(
                    f"{column}={value}: {observed} sampled of {int(count)} cells"
                )
        evidence["groups"][column] = rows
    return evidence, concerns


class RnaTuningRun:
    """Execute evidence-requested comparisons with one authoritative history."""

    def __init__(
        self,
        owner: Any,
        store: Any,
        workflow: Any,
        request: OrchestrationRequestRecord,
        plan: AutomatedPreprocessingPlan,
        handoff: PreprocessedAssayHandoff,
        study: StudyContract,
        answers: Mapping[str, Any],
        provenance: dict[str, Any],
        *,
        design_comparisons: Sequence[CovariateComparison] = (),
    ) -> None:
        if (
            handoff.cellSelection is None
            or handoff.graphFeatures is None
            or handoff.markerFeatures is None
        ):
            raise ValueError(
                "RNA tuning requires exact cells, graph genes and marker genes"
            )
        self.owner, self.store, self.workflow = owner, store, workflow
        self.request, self.plan, self.handoff, self.study = (
            request,
            plan,
            handoff,
            study,
        )
        self.answers, self.provenance = answers, provenance
        self.design_comparisons = tuple(design_comparisons)
        self.prefix = journal._ensure_orchestration_store(store)
        self.cells = artifact_model_to_ref(handoff.cellSelection)
        self.marker_features = artifact_model_to_ref(handoff.markerFeatures)
        self.budget = CandidateBudget(
            store, self.prefix, workflow.workflowRunId, request.config, provenance
        )
        self.settings: dict[str, RnaSetting] = {}
        self.history: list[dict[str, Any]] = []
        self.evaluations: dict[str, list[ParameterCandidateEvaluation]] = {
            "sample0": [],
            "sample1": [],
            "full": [],
        }
        self.last_action: TuningAction | None = None
        self.full_repairs = 0
        self.answer_consumed = False
        self.scope_sizes: dict[str, int] = {}
        self.feature_evidence_cache: dict[str, dict[str, Any]] = {}
        self.neighbor_comparisons: dict[tuple[str, str], float] = {}
        self.batch_columns = list(study.technicalBatchColumns)
        self.coverage_columns = [
            value
            for value in dict.fromkeys(
                [
                    study.physicalCaptureColumn,
                    *study.independentUnitColumns,
                    *study.conditionColumns,
                    *study.technicalBatchColumns,
                    *study.protectedColumns,
                ]
            )
            if value is not None and study.columnKinds.get(value) != "continuous"
        ]
        self.family_patterns = {
            str(row["family"]): str(row["pattern"])
            for row in (
                plan.assays[0].featureParameters.get("defaultFeatureInventory") or {}
            ).get("families", [])
            if isinstance(row, Mapping) and "family" in row and "pattern" in row
        }

    def baseline(self, resolution: float = 1.0) -> RnaSetting:
        assert self.handoff.graphFeatures is not None
        return RnaSetting(
            parameters=ParameterCandidate(
                candidateId="baseline",
                dimensions=min(21, self.handoff.nFeatures - 1, self.handoff.nCells - 1),
                neighborsK=min(11, self.handoff.nCells - 1),
                leidenResolution=resolution,
                useHarmony=False,
            ),
            features=self.handoff.graphFeatures,
            eligibleFeatures=self.handoff.graphFeatureCandidates["eligibleDefault"],
            hvgCount=self.handoff.nFeatures,
        )

    @staticmethod
    def execution_inputs(cells: ArtifactRef, setting: RnaSetting) -> dict[str, Any]:
        return {
            "cells": cells.to_dict(),
            "features": setting.features.model_dump(mode="json"),
            "parameters": setting.parameters.model_dump(mode="json"),
        }

    def execute(
        self, scope: str, cells: ArtifactRef, setting: RnaSetting
    ) -> ParameterCandidateEvaluation:
        inputs = self.execution_inputs(cells, setting)
        identity = candidate_identity(inputs)
        parameters = setting.parameters.model_copy(
            update={"candidateId": f"rna_{identity[:24]}"}
        )
        setting = setting.model_copy(update={"parameters": parameters})
        self.settings[parameters.candidateId] = setting
        admission = self.budget.admit(scope, inputs)
        saved = self.budget.completed(admission)
        if saved is not None:
            evaluation = ParameterCandidateEvaluation.model_validate(
                saved["evaluation"]
            )
            for artifact in evaluation.artifacts.values():
                status = self.store.inspect_artifact(artifact_model_to_ref(artifact))
                if not status.exists or not status.complete:
                    raise ValueError(
                        "Saved candidate evidence is unavailable or incomplete"
                    )
        else:
            features = artifact_model_to_ref(setting.features)
            normalized = self.store.run_normalization(
                cells, features=features, invalidate_cache=False
            )
            deps, ids = prepare_parameter_tuning_dependencies(
                self.store,
                normalized=normalized,
                candidates=[parameters],
                batch_columns=self.batch_columns,
                preservation_columns=self.study.protectedColumns,
                pair_harmony_candidates=False,
                max_candidates=1,
                min_cluster_cells=1,
            )
            deps.protectedCombinations = tuple(
                tuple(columns) for columns in self.study.protectedCombinations
            )
            deps.columnKinds = self.study.columnKinds
            evaluation = execute_parameter_candidate(deps, ids[0])
            if evaluation.status != "done":
                raise RuntimeError(
                    evaluation.error
                    or "Candidate execution failed; its admitted work can be retried on resume"
                )
            evaluation.artifacts.update(
                {
                    "normalized": ArtifactRecord.from_ref(normalized),
                    "graphFeatures": ArtifactRecord.from_ref(features),
                }
            )
            if evaluation.status == "done":
                evaluation = augment_pca_evaluations(
                    self.store,
                    [evaluation],
                    feature_selection=features,
                    nominated_families=SCARF_DEFAULT_DIAGNOSTIC_FAMILIES,
                    protected_families=self.plan.assays[0].featureParameters.get(
                        "protectFamilies", []
                    ),
                    technical_columns=self.batch_columns,
                    batch_columns=self.batch_columns,
                    protected_columns=self.study.protectedColumns,
                    qc_columns=self.plan.cellQc.attributes,
                    column_kinds=self.study.columnKinds,
                )[0]
                native = next(
                    (
                        item
                        for item in self.evaluations[scope]
                        if not item.parameters.useHarmony
                        and self.settings[item.candidateId].features == setting.features
                        and item.parameters.model_dump(
                            exclude={"candidateId", "useHarmony"}
                        )
                        == parameters.model_dump(exclude={"candidateId", "useHarmony"})
                    ),
                    None,
                )
                doublets = score_advisory_doublets(
                    self.store,
                    native or evaluation,
                    [
                        item
                        for item in [*self.evaluations[scope], evaluation]
                        if item.artifacts.get("graphFeatures")
                        == evaluation.artifacts["graphFeatures"]
                        and item.cellSelection == evaluation.cellSelection
                    ],
                    assay=self.handoff.assay,
                    feature_selection=features,
                    capture_column=self.study.physicalCaptureColumn,
                )
                evaluation = augment_cluster_evaluations(
                    self.store,
                    [evaluation],
                    marker_assay=self.handoff.assay,
                    marker_features=self.marker_features,
                    independent_unit_columns=self.study.independentUnitColumns,
                    technical_columns=self.batch_columns,
                    nominated_families=SCARF_DEFAULT_DIAGNOSTIC_FAMILIES,
                    protected_families=self.plan.assays[0].featureParameters.get(
                        "protectFamilies", []
                    ),
                    doublet_evidence=doublets,
                )[0]
            evaluation = ParameterCandidateEvaluation.model_validate_json(
                record_io.canonical_json_bytes(evaluation.model_dump(mode="json"))
            )
            self.budget.complete(
                admission, {"evaluation": evaluation.model_dump(mode="json")}
            )
        if evaluation.candidateId not in {
            item.candidateId for item in self.evaluations[scope]
        }:
            self.evaluations[scope].append(evaluation)
        return evaluation

    def execute_matched(
        self, scope: str, cells: ArtifactRef, setting: RnaSetting
    ) -> ParameterCandidateEvaluation:
        if setting.parameters.useHarmony:
            if self.study.correctionLicense != "safe" or not self.batch_columns:
                raise ValueError(
                    "Harmony requires a safe design license and approved batch columns"
                )
            native = setting.model_copy(
                update={
                    "parameters": setting.parameters.model_copy(
                        update={"useHarmony": False}
                    )
                }
            )
            self.budget.admit_many(
                scope,
                [
                    self.execution_inputs(cells, native),
                    self.execution_inputs(cells, setting),
                ],
            )
            self.execute(scope, cells, native)
        return self.execute(scope, cells, setting)

    def harmony_gate(
        self, scope: str, selected: ParameterCandidateEvaluation
    ) -> tuple[bool, list[str]]:
        if not selected.parameters.useHarmony:
            return True, []
        setting = self.settings[selected.candidateId]
        native = next(
            (
                item
                for item in self.evaluations[scope]
                if not item.parameters.useHarmony
                and self.settings[item.candidateId].features == setting.features
                and item.parameters.model_dump(exclude={"candidateId", "useHarmony"})
                == selected.parameters.model_dump(exclude={"candidateId", "useHarmony"})
            ),
            None,
        )
        return harmony_acceptance_gate(
            native,
            selected,
            batch_columns=self.batch_columns,
            protected_columns=[
                *self.study.protectedColumns,
                *(
                    "joint:" + json.dumps(columns, separators=(",", ":"))
                    for columns in self.study.protectedCombinations
                ),
            ],
            independent_unit_columns=self.study.independentUnitColumns,
            require_doublet_evidence=True,
        )

    def experiments(
        self, selected: ParameterCandidateEvaluation
    ) -> dict[str, dict[str, Any]]:
        """Offer individual interventions, without executing their numerical work."""
        setting = self.settings[selected.candidateId]
        n_cells = next(
            (
                self.scope_sizes[name]
                for name, rows in self.evaluations.items()
                if selected in rows and name in self.scope_sizes
            ),
            self.handoff.nCells,
        )
        options: dict[str, dict[str, Any]] = {}
        for field, values in (
            ("dimensions", (10, 21, 30, 50)),
            ("neighborsK", (11, 21, 41)),
            ("leidenResolution", (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)),
        ):
            for value in values:
                if value != getattr(setting.parameters, field):
                    if field == "dimensions" and value >= min(
                        setting.hvgCount, n_cells
                    ):
                        continue
                    if field == "neighborsK" and value >= n_cells:
                        continue
                    options[f"{field}:{value}"] = {"parameter": field, "value": value}
        for count in (1000, 2000, 4000):
            if count != setting.hvgCount:
                options[f"hvgCount:{count}"] = {"parameter": "hvgCount", "value": count}
        for column in self.batch_columns:
            if self.study.columnKinds.get(column) == "continuous":
                continue
            if setting.ranking != "batchAware" or setting.rankingColumn != column:
                options[f"hvgRanking:batchAware:{column}"] = {
                    "parameter": "hvgRanking",
                    "value": "batchAware",
                    "column": column,
                }
        if setting.ranking == "batchAware":
            options["hvgRanking:global"] = {
                "parameter": "hvgRanking",
                "value": "global",
            }
        for family in self.family_patterns:
            for operation in ("includeFamily", "excludeFamily"):
                options[f"{operation}:{family}"] = {
                    "parameter": operation,
                    "value": family,
                }
        feature_policy = self.plan.assays[0].featureParameters
        for feature in dict.fromkeys(
            [
                *feature_policy.get("proposedExcludeFeatures", []),
                *feature_policy.get("protectFeatures", []),
            ]
        ):
            for operation in ("includeFeature", "excludeFeature"):
                if operation == "excludeFeature" and feature in feature_policy.get(
                    "protectFeatures", []
                ):
                    continue
                options[f"{operation}:{feature}"] = {
                    "parameter": operation,
                    "value": feature,
                }
        if self.study.correctionLicense == "safe" and self.batch_columns:
            options["useHarmony:true"] = {"parameter": "useHarmony", "value": True}
        if setting.parameters.useHarmony:
            options["useHarmony:false"] = {"parameter": "useHarmony", "value": False}
        return options

    def batch_ranking(
        self, eligible: ArtifactRef, count: int, column: str
    ) -> np.ndarray:
        """Use core per-group variability on the same globally eligible genes."""
        if column not in self.batch_columns:
            raise ValueError("Batch-aware ranking needs an approved technical column")
        from ..experimental_context.characterization import _SelectionBoundCells

        cells = _SelectionBoundCells(self.store.zw, self.store.cells, self.cells)
        values, counts = np.unique(cells.fetch(column), return_counts=True)
        groups = []
        for value, n_cells in zip(values, counts, strict=True):
            if n_cells < 20:
                continue
            selection = self.store.filter_cells(
                [column],
                [value],
                [value],
                cell_selection=self.cells,
                keep_bounds=True,
                invalidate_cache=False,
            )
            reference = self.store.select_hvgs(
                selection,
                from_assay=self.handoff.assay,
                top_n=self.store.get_assay(self.handoff.assay).feats.N,
                min_cells=1,
                max_cells=np.inf,
                blacklist="",
                show_plot=False,
                invalidate_cache=False,
            )
            status = self.store.inspect_artifact(reference)
            summary = ArtifactRef.from_dict(dict(status.inputs["feature_summary"]))
            group = self.store.load_artifact(reference)
            detected = (
                np.asarray(self.store.load_artifact(summary)["normed_n"][:]) >= 20
            )
            groups.append(
                HvgGroupVariability(
                    group_id=str(value),
                    cell_count=int(n_cells),
                    corrected_variance=np.asarray(
                        group["corrected_variance"][:], dtype=np.float64
                    ),
                    detected_features=detected,
                )
            )
        if len(groups) < 2:
            raise ValueError(
                "Batch-aware ranking lacks two groups with sufficient cells"
            )
        mask = np.asarray(self.store.load_artifact(eligible)["values"][:], dtype=bool)
        statistics = artifact_model_to_ref(
            self.handoff.graphFeatureCandidates["eligibleAll"]
        )
        variance = np.asarray(
            self.store.load_artifact(statistics)["corrected_variance"][:],
            dtype=np.float64,
        )
        ranking = aggregate_hvg_rankings(
            variance,
            mask,
            groups,
            valid_group_count=len(groups),
            candidate_targets=(count,),
        )
        return ranking.ranking

    def apply_experiment(
        self, selected: ParameterCandidateEvaluation, experiment: dict[str, Any]
    ) -> RnaSetting:
        setting = self.settings[selected.candidateId]
        field, value = experiment["parameter"], experiment["value"]
        if field in {"dimensions", "neighborsK", "leidenResolution", "useHarmony"}:
            return setting.model_copy(
                update={
                    "parameters": setting.parameters.model_copy(update={field: value})
                }
            )
        eligible = artifact_model_to_ref(setting.eligibleFeatures)
        count = int(value) if field == "hvgCount" else setting.hvgCount
        ranking_mode = value if field == "hvgRanking" else setting.ranking
        ranking_column = (
            experiment.get("column") if field == "hvgRanking" else setting.rankingColumn
        )
        if field in {
            "includeFamily",
            "excludeFamily",
            "includeFeature",
            "excludeFeature",
        }:
            mask = np.asarray(
                self.store.load_artifact(eligible)["values"][:], dtype=bool
            )
            names = np.asarray(
                self.store.get_assay(self.handoff.assay).feats.fetch_all("names")
            ).astype(str)
            feature_ids = np.asarray(
                self.store.get_assay(self.handoff.assay).feats.fetch_all("ids")
            ).astype(str)
            if field.endswith("Family"):
                pattern = re.compile(self.family_patterns[value], flags=re.IGNORECASE)
                family_mask = np.asarray(
                    [pattern.search(name) is not None for name in names]
                )
            else:
                family_mask = (names == value) | (feature_ids == value)
                if not family_mask.any():
                    raise ValueError(
                        "The nominated exact feature is absent from the assay"
                    )
            if field.startswith("include"):
                all_eligible = artifact_model_to_ref(
                    self.handoff.graphFeatureCandidates["eligibleAll"]
                )
                allowed = np.asarray(
                    self.store.load_artifact(all_eligible)["values"][:], dtype=bool
                )
                mask |= allowed & family_mask
            else:
                policy = self.plan.assays[0].featureParameters
                protected_mask = np.isin(
                    names, policy.get("protectFeatures", [])
                ) | np.isin(feature_ids, policy.get("protectFeatures", []))
                for family in policy.get("protectFamilies", []):
                    family_protection = _family_mask(names, family)
                    if family_protection is not None:
                        protected_mask |= family_protection
                    elif family in self.family_patterns:
                        pattern = re.compile(
                            self.family_patterns[family], flags=re.IGNORECASE
                        )
                        protected_mask |= np.asarray(
                            [pattern.search(name) is not None for name in names]
                        )
                if np.any(family_mask & protected_mask):
                    raise ValueError(
                        "An objective-protected feature or family cannot be excluded"
                    )
                mask &= ~family_mask
            eligible = self.store.set_feature_selection(
                from_assay=self.handoff.assay, mask=mask, invalidate_cache=False
            )
        if ranking_mode == "batchAware" and ranking_column is None:
            raise ValueError(
                "Batch-aware ranking requires an explicit technical column"
            )
        indices = (
            self.batch_ranking(eligible, count, ranking_column)
            if ranking_mode == "batchAware" and ranking_column is not None
            else None
        )
        features = rank_core_hvgs(
            self.store,
            eligible=eligible,
            statistics=artifact_model_to_ref(
                self.handoff.graphFeatureCandidates["eligibleAll"]
            ),
            top_n=count,
            ranking=indices,
        )
        n_features = int(
            np.asarray(
                self.store.load_artifact(features)["values"][:], dtype=bool
            ).sum()
        )
        if n_features <= setting.parameters.dimensions:
            raise ValueError(
                "This feature experiment cannot retain the fixed PCA dimension"
            )
        return setting.model_copy(
            update={
                "features": ArtifactReferenceModel.from_artifact_ref(features),
                "eligibleFeatures": ArtifactReferenceModel.from_artifact_ref(eligible),
                "hvgCount": n_features,
                "ranking": ranking_mode,
                "rankingColumn": ranking_column,
            }
        )

    def feature_evidence(
        self, selected: ParameterCandidateEvaluation
    ) -> dict[str, Any]:
        """Summarize frozen core statistics without rerunning feature variability."""
        setting = self.settings[selected.candidateId]
        key = setting.model_dump_json(exclude={"parameters"})
        if key in self.feature_evidence_cache:
            return self.feature_evidence_cache[key]
        mask = np.asarray(
            self.store.load_artifact(artifact_model_to_ref(setting.features))["values"][
                :
            ],
            dtype=bool,
        )
        eligible = np.asarray(
            self.store.load_artifact(artifact_model_to_ref(setting.eligibleFeatures))[
                "values"
            ][:],
            dtype=bool,
        )
        reference = artifact_model_to_ref(
            self.handoff.graphFeatureCandidates["eligibleAll"]
        )
        variance = np.asarray(
            self.store.load_artifact(reference)["corrected_variance"][:],
            dtype=np.float64,
        )
        names = np.asarray(
            self.store.get_assay(self.handoff.assay).feats.fetch_all("names")
        ).astype(str)
        indices = np.flatnonzero(eligible)
        ranked = indices[np.lexsort((indices, -variance[indices]))]
        families = {}
        for family, pattern in self.family_patterns.items():
            expression = re.compile(pattern, flags=re.IGNORECASE)
            membership = np.asarray(
                [expression.search(name) is not None for name in names]
            )
            families[family] = {
                "eligibleGenes": int((eligible & membership).sum()),
                "selectedGenes": int((mask & membership).sum()),
                "selectedExamples": names[
                    np.flatnonzero(mask & membership)[:8]
                ].tolist(),
                "excludedExamples": names[
                    np.flatnonzero(~eligible & membership)[:8]
                ].tolist(),
            }
        evidence = {
            "statistics": reference.to_dict(),
            "statisticsCells": self.cells.to_dict(),
            "basis": "Core Scarf corrected variance on the full QC-retained cells; feature-axis summaries are descriptive, not a substitute for downstream comparisons.",
            "selectedGenes": int(mask.sum()),
            "eligibleGenes": int(eligible.sum()),
            "ranking": setting.ranking,
            "rankingColumn": setting.rankingColumn,
            "globalRankLandmarks": [
                {
                    "rank": rank,
                    "gene": str(names[ranked[rank - 1]]),
                    "correctedVariance": float(variance[ranked[rank - 1]]),
                }
                for rank in (1, 500, 1000, 2000, 4000)
                if rank <= len(ranked)
            ],
            "topSelectedGenes": names[
                [index for index in ranked if mask[index]][:20]
            ].tolist(),
            "families": families,
        }
        self.feature_evidence_cache[key] = evidence
        return evidence

    def review(
        self,
        scope: str,
        review_index: int,
        selected: ParameterCandidateEvaluation,
        coverage: dict[str, Any],
    ) -> TuningAction:
        from .tuning import _analysis_visual_content

        candidates = self.evaluations[scope]
        key = f"parameter_tuning/{scope}/review{review_index}"
        previous_review = journal.read_checkpoint(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            key,
        )
        candidate_evidence = [item.model_dump(mode="json") for item in candidates]
        setting_evidence = {
            item.candidateId: self.settings[item.candidateId].model_dump(mode="json")
            for item in candidates
        }
        if previous_review is not None and (
            previous_review["inputs"].get("candidates") != candidate_evidence
            or previous_review["inputs"].get("settings") != setting_evidence
        ):
            raise ValueError(
                "Saved review has different candidate evidence or settings"
            )
        experiments = (
            previous_review["inputs"]["experiments"]
            if previous_review is not None
            else self.experiments(selected)
        )
        completed_experiments: dict[str, str] = {}
        matched_comparisons = []
        current_setting = self.settings[selected.candidateId]
        for candidate in candidates:
            other = self.settings[candidate.candidateId]
            if (
                candidate.candidateId == selected.candidateId
                or candidate.cellSelection != selected.cellSelection
                or other.features != current_setting.features
                or other.parameters.reductionMethod
                != current_setting.parameters.reductionMethod
            ):
                continue
            changes = {
                field: {
                    "current": getattr(current_setting.parameters, field),
                    "alternative": getattr(other.parameters, field),
                }
                for field in (
                    "dimensions",
                    "neighborsK",
                    "leidenResolution",
                    "useHarmony",
                )
                if getattr(current_setting.parameters, field)
                != getattr(other.parameters, field)
            }
            if len(changes) != 1:
                continue
            matched_comparisons.append(
                {
                    "currentCandidateId": selected.candidateId,
                    "alternativeCandidateId": candidate.candidateId,
                    "changedParameter": changes,
                    "basis": "Same frozen cells and graph features; all other analysis parameters match. Compare these exact candidates rather than mixing dimensions and resolution effects.",
                }
            )
            if previous_review is None and candidate.status == "done":
                field, values = next(iter(changes.items()))
                for experiment_id, experiment in experiments.items():
                    if (
                        experiment["parameter"] == field
                        and experiment["value"] == values["alternative"]
                    ):
                        completed_experiments[experiment_id] = candidate.candidateId
        if previous_review is None:
            experiments = {
                key: value
                for key, value in experiments.items()
                if key not in completed_experiments
            }
        comparisons = (
            previous_review["inputs"].get("neighborComparisons", [])
            if previous_review is not None
            else []
        )
        selected_neighbors = selected.artifacts.get("neighbors")
        for alternative in [] if previous_review is not None else candidates:
            other_neighbors = alternative.artifacts.get("neighbors")
            if (
                selected_neighbors is None
                or other_neighbors is None
                or selected_neighbors == other_neighbors
                or selected.parameters.neighborsK != alternative.parameters.neighborsK
            ):
                continue
            left_id, right_id = sorted(
                (selected_neighbors.artifactId, other_neighbors.artifactId)
            )
            pair = (left_id, right_id)
            if pair not in self.neighbor_comparisons:
                self.neighbor_comparisons[pair] = _neighbor_overlap(
                    self.store,
                    artifact_model_to_ref(selected_neighbors),
                    artifact_model_to_ref(other_neighbors),
                )
            comparisons.append(
                {
                    "leftCandidateId": selected.candidateId,
                    "rightCandidateId": alternative.candidateId,
                    "meanNeighborJaccard": self.neighbor_comparisons[pair],
                    "basis": "Same frozen cells and k; descriptive response to settings, not independent stability or evidence that correction is beneficial.",
                }
            )
        declared_image_input = _configured_image_input(self.owner.model)
        capability_key = "parameter_tuning/structured_evidence"
        capability_inputs = {
            **self.provenance,
            "configuredImageInput": declared_image_input,
        }
        capability = journal.load_checkpoint(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            capability_key,
            inputs=capability_inputs,
        )
        if capability is None and declared_image_input is False:
            capability = journal.save_checkpoint(
                self.store,
                self.prefix,
                self.workflow.workflowRunId,
                capability_key,
                inputs=capability_inputs,
                outputs={
                    "evidenceMode": "structured",
                    "reason": "configuredImageInputUnsupported",
                },
            )
        if capability is not None and capability.get("evidenceMode") != "structured":
            raise ValueError("The recorded model capability is invalid")
        mode = (
            previous_review["inputs"].get("evidenceMode")
            if previous_review is not None
            else "structured"
            if capability is not None
            else "visual"
        )
        if mode not in {"visual", "structured"}:
            raise ValueError(
                "Saved review lacks an exact evidence mode; start a new workflow"
            )
        visual_inspection = "available" if mode == "visual" else "unavailable"
        images = []
        if previous_review is not None:
            image_hashes = previous_review["inputs"].get("imageHashes", {})
        elif mode == "visual":
            images = _analysis_visual_content(
                self.store,
                selected,
                candidates,
                qc_columns=self.plan.cellQc.attributes,
                qc_artifact_metrics=[
                    (item.name, item.artifact)
                    for item in self.plan.cellQc.artifactMetrics
                ],
            )
            image_hashes = {
                image.identifier: hashlib.sha256(image.data).hexdigest()
                for image in images
            }
        else:
            image_hashes = {}
        if not isinstance(image_hashes, dict) or bool(image_hashes) != (
            mode == "visual"
        ):
            raise ValueError("Review images do not match its evidence mode")
        evidence_ids = (
            previous_review["inputs"]["availableEvidenceIds"]
            if previous_review is not None
            else list(
                dict.fromkeys(
                    [
                        *(f"candidate:{item.candidateId}" for item in candidates),
                        *(key for item in candidates for key in item.evidenceIds),
                        *self.study.evidenceIds,
                        *self.plan.cellQc.evidenceIds,
                        *(item.evidenceId for item in self.design_comparisons),
                        *image_hashes,
                        "studyContract",
                        "qcPolicy",
                        "samplingCoverage",
                        "featureEvidence",
                        "neighborComparisons",
                        "assessmentContext",
                    ]
                )
            )
        )
        if not isinstance(evidence_ids, list) or any(
            not isinstance(value, str) for value in evidence_ids
        ):
            raise ValueError("Review evidence IDs must be a list of strings")
        evidence = {
            "studyContract": self.study.model_dump(mode="json"),
            "qcPolicy": self.plan.cellQc.model_dump(mode="json"),
            "scope": scope,
            "evidenceMode": mode,
            "visualInspection": visual_inspection,
            "configuredImageInput": declared_image_input,
            "coverage": coverage,
            "currentCandidateId": selected.candidateId,
            "candidates": candidate_evidence,
            "settings": setting_evidence,
            "featureEvidence": {
                item.candidateId: self.feature_evidence(item) for item in candidates
            },
            "neighborComparisons": comparisons,
            "harmonyGates": {
                item.candidateId: self.harmony_gate(scope, item)
                for item in candidates
                if item.parameters.useHarmony
            },
            "experiments": experiments,
            "availableEvidenceIds": evidence_ids,
            "imageHashes": image_hashes,
            "assessedDomains": sorted(_DOMAINS),
            "budget": {
                "visibleEvaluations": {
                    name: len(rows) for name, rows in self.evaluations.items()
                },
                "limits": self.budget.summary()["limits"],
            },
            "fullRepairsUsed": self.full_repairs,
            "pilotPopulationWarnings": {
                item.candidateId: "This sampled partition includes fewer than 20 cells in a population. Assess its relevance and support; it is not evidence of an invalid biological group. Accepting this partition requires a larger sample or full-cohort assessment."
                for item in candidates
                if scope != "full"
                and item.metrics.minClusterCells is not None
                and item.metrics.minClusterCells < 20
            },
            "smallPopulationPolicy": "Small full-cohort groups require explicit marker, stability, graph and applicable independent-unit support assessment against the objective. Their size alone is neither proof of biology nor grounds for rejection. Defer when essential evidence is insufficient.",
        }
        if previous_review is None:
            support_columns = list(
                dict.fromkeys(
                    column
                    for column in (
                        self.study.physicalCaptureColumn,
                        *self.study.independentUnitColumns,
                    )
                    if column is not None
                )
            )
            evidence["assessmentContext"] = {
                "correctionPolicy": {
                    "license": self.study.correctionLicense,
                    "harmonyPermitted": self.study.correctionLicense == "safe"
                    and bool(self.batch_columns),
                    "nativeAcceptance": "A native descriptive analysis may be accepted when its required evidence supports the objective, while explicitly retaining confounding limitations. A Harmony gate is required only for accepting Harmony. Defer if the objective requires effects that the design cannot separate.",
                },
                "matchedComparisons": matched_comparisons,
                "alreadyEvaluatedExperiments": completed_experiments,
                "designComparisons": [
                    item.model_dump(mode="json") for item in self.design_comparisons
                ],
                "populationSupport": {
                    selected.candidateId: population_support_evidence(
                        self.store, selected, support_columns
                    )
                }
                if support_columns
                else {},
                "previousActions": [
                    {
                        "scope": row["scope"],
                        **{
                            name: row["review"][name]
                            for name in (
                                "action",
                                "selectedCandidateId",
                                "experimentId",
                                "correctionNeed",
                            )
                        },
                    }
                    for row in self.history
                    if "review" in row
                ],
            }
        elif "assessmentContext" in previous_review["inputs"]:
            evidence["assessmentContext"] = previous_review["inputs"][
                "assessmentContext"
            ]

        def validate(action: TuningAction, *, replay: bool = False) -> TuningAction:
            by_id = {item.candidateId: item for item in candidates}
            if action.selectedCandidateId not in by_id:
                raise ValueError("Choose an observed candidate from the current cells")
            unknown_ids = sorted(set(action.evidenceIds).difference(evidence_ids))
            if unknown_ids:
                raise ValueError(
                    f"Assessment cited unknown evidence: {unknown_ids[:8]!r}. "
                    "Copy exact IDs from availableEvidenceIds; do not invent "
                    "suffixes or use artifact IDs as evidence IDs. The selected "
                    f"candidate anchor is 'candidate:{action.selectedCandidateId}'."
                )
            if mode == "visual" and not set(action.evidenceIds).intersection(
                image_hashes
            ):
                raise ValueError("Assessment must cite its actual visual evidence")
            chosen = by_id[action.selectedCandidateId]
            if not set(action.evidenceIds).intersection(
                {f"candidate:{chosen.candidateId}", *chosen.evidenceIds}
            ):
                raise ValueError(
                    "Assessment must cite the selected numerical evidence: "
                    f"'candidate:{chosen.candidateId}' or one of that candidate's "
                    "supplied evidenceIds in availableEvidenceIds."
                )
            if (
                action.experimentId is not None
                and action.experimentId not in experiments
            ):
                raise ValueError(
                    f"Unknown experiment ID {action.experimentId!r}. "
                    f"For current candidate {selected.candidateId!r}, copy one exact "
                    f"key from experiments: {list(experiments)!r}. "
                    "Only action='experiment' may name a next operation."
                )
            if (
                action.action == "experiment"
                and action.selectedCandidateId != selected.candidateId
            ):
                raise ValueError(
                    "An offered experiment must use the current candidate as its fixed baseline"
                )
            if (
                self.study.correctionLicense == "unsafeConfounded"
                and action.correctionNeed in {"needed", "notNeeded"}
            ):
                message = (
                    "The design confounds the batch columns with protected biology; their association cannot establish a removable technical effect. "
                    "Harmony is not permitted, and a PCA or feature experiment is not a substitute for Harmony. "
                    "Use notApplicable for the prohibited correction with an explicit confounding limitation, or uncertain and defer if an essential question cannot be resolved."
                )
                if replay and action.action == "experiment":
                    self.history.append(
                        {
                            "scope": scope,
                            "reason": "A saved screening rationale claimed identifiable correction necessity despite the confounded design. Its numerical experiment remains in the audit history; that scientific claim must be reassessed from the supplied evidence.",
                        }
                    )
                else:
                    raise ValueError(message)
            if (
                self.study.correctionLicense == "safe"
                and action.correctionNeed == "notApplicable"
            ):
                raise ValueError(
                    "A safe correction design still requires an observed necessity assessment"
                )
            if action.action == "accept":
                if action.correctionNeed == "uncertain":
                    raise ValueError(
                        "Uncertain correction necessity requires further evidence or deferral before acceptance"
                    )
                if not chosen.eligible:
                    raise ValueError(
                        "The selected candidate failed required full-cell checks"
                    )
                accepted, reasons = self.harmony_gate(scope, chosen)
                if not accepted:
                    raise ValueError("Harmony acceptance failed: " + "; ".join(reasons))
                required = (
                    "seedStability",
                    "subsampleStability",
                    "markerCoherence",
                    "membershipStrengthMean",
                    "clusterConnectivity",
                )
                if any(getattr(chosen.metrics, field) is None for field in required):
                    raise ValueError(
                        "Required stability, marker or graph evidence is missing"
                    )
                if (
                    self.study.independentUnitColumns
                    and chosen.metrics.crossUnitSupport is None
                ):
                    raise ValueError(
                        "Required independent-unit support evidence is missing"
                    )
                if chosen.parameters.useHarmony and action.correctionNeed != "needed":
                    raise ValueError(
                        "Accepting Harmony requires observed correction necessity"
                    )
                if self.study.unsupportedProtection and action.correctionNeed in {
                    "needed",
                    "uncertain",
                }:
                    raise ValueError(
                        "Correction necessity remains unresolved because required matched biological protection is unsupported"
                    )
                if (
                    not chosen.parameters.useHarmony
                    and action.correctionNeed == "needed"
                ):
                    raise ValueError(
                        "Native acceptance leaves required correction unresolved; provide more evidence or defer"
                    )
                if self.study.correctionLicense == "safe" and action.correctionNeed in {
                    "needed",
                    "uncertain",
                }:
                    chosen_setting = self.settings[chosen.candidateId]
                    has_comparison = any(
                        item.parameters.useHarmony
                        and item.status == "done"
                        and self.settings[item.candidateId].features
                        == chosen_setting.features
                        and item.parameters.model_dump(
                            exclude={"candidateId", "useHarmony"}
                        )
                        == chosen.parameters.model_dump(
                            exclude={"candidateId", "useHarmony"}
                        )
                        for item in candidates
                    )
                    if not has_comparison:
                        raise ValueError(
                            "Safe but uncertain/needed correction requires a matched Harmony experiment"
                        )
                if self.study.correctionLicense == "indeterminate":
                    raise ValueError(
                        "Correction design authorization remains indeterminate"
                    )
            return action

        saved = journal.load_checkpoint(
            self.store, self.prefix, self.workflow.workflowRunId, key, inputs=evidence
        )
        pending_saved = saved is not None and saved["action"]["action"] == "defer"
        answer = self.answers.get(key)
        if (
            answer is None
            and not self.answer_consumed
            and (saved is None or pending_saved)
        ):
            answer = self.answers.get("parameter_tuning")
        if pending_saved:
            assert saved is not None
            answer_inputs = {**evidence, "deferredAction": saved["action"]}
            prior_answer = journal.load_checkpoint(
                self.store,
                self.prefix,
                self.workflow.workflowRunId,
                key + "/answer",
                inputs=answer_inputs,
            )
            if prior_answer is not None:
                if (
                    answer is not None
                    and TuningAction.model_validate(answer).model_dump(mode="json")
                    == prior_answer["action"]
                ):
                    self.answer_consumed = True
                key += "/answer"
                evidence = answer_inputs
                saved = prior_answer
                answer = None
            elif answer is not None:
                key += "/answer"
                evidence = answer_inputs
                saved = None
        if saved is not None:
            action = validate(TuningAction.model_validate(saved["action"]), replay=True)
        elif answer is not None:
            action = validate(TuningAction.model_validate(answer))
            self.answer_consumed = True
        else:
            prompt = (
                "Assess this RNA analysis as a computational biologist against the exact study objective. "
                "Start from Scarf defaults; keep them when observed quantitative evidence and biological interpretation support them. "
                "Do not execute a search grid or favor a default solely because it is a default. "
                "Assess every named scientific domain before acceptance. Explain observed marker programs and relevant PCA loading genes, "
                "QC/capture retention, batch associations per PC and protected biological structure. "
                "Family dominance alone never proves nuisance; inclusion, exclusion and HVG bans need evidence and objective justification. "
                "If a specific concern warrants testing, choose exactly one offered experiment and state its expected improvement and "
                "what objective-relevant biology must be preserved. Family policy remains revisable when later evidence warrants it. "
                "PCA, HVG, k and resolution changes are separate comparisons. Do not equate a small sampled cluster with an artifact. "
                "For a safe design license, needed or uncertain correction requires a matched Harmony experiment. "
                "A notNeeded choice needs observed native batch/PC and biological evidence. Unsafe or unknown design is never authorization. "
                "Only accepting a Harmony representation requires a matched Harmony gate; a native representation does not require one. "
                "When correctionLicense is unsafeConfounded, do not infer correction necessity from batch mixing or PCA association. "
                "Use notApplicable for the prohibited correction, retain the confounding limitation, and assess whether native descriptive population discovery satisfies the objective. "
                "If essential effects remain inseparable, defer. Never request a different parameter change as a proxy for unavailable Harmony. "
                "Screening estimates do not prove full-cohort transfer. "
                "Findings describe observed results. The experimentId names the exact next operation and expectedImprovement predicts only that operation's effect. "
                "Already evaluated settings are existing alternatives, not new experiments; use the supplied matchedComparisons to avoid mixing PCA effects with resolution effects. "
                "For example, compare 10 versus 21 PCs at the same resolution rather than quoting the stability of another resolution. "
                "A high scaled cLISI is local purity of the supplied label, not proof that a clinical phenotype or all cell types are preserved. "
                "Graph connectivity is within-label connectivity. Low library mixing and high PCA association can reflect donor biology or cell composition; neither proves technical causality. "
                "A QC association is correlation. Check featureEvidence for actual selected genes: marker-family enrichment cannot show that an excluded family drives PCA. "
                "More retained cells, balanced group counts, or cross-unit support alone do not prove healthy cells or biological preservation. "
                "Check proposed cell identities against the tissue context. Unexpected marker programs require capture/donor and provenance investigation; do not declare them ordinary tissue populations or assert contamination without evidence. "
                "Detailed populationSupport is supplied for currentCandidateId only; do not claim to have compared unprovided distributions for alternatives. Inspect its capture/donor distribution and missing metadata. Broad support does not prove a biological identity; concentration alone does not prove contamination. "
                "Unsupported design comparisons establish neither association nor absence; keep their unresolved requirements visible. "
                "Previous actions are history, not scientific authority. Reassess their claims against the exact current evidence. "
                "Request enlarge when sample evidence is insufficient; on full cells there is one targeted repair, then defer. "
                "Copy evidence IDs exactly from availableEvidenceIds, including the selected candidate's anchor "
                "or one of its supplied diagnostic evidence IDs. Do not invent IDs, suffixes or substitute artifact IDs. "
                "Never accept merely because work limits are exhausted. "
            )
            prompt += (
                "The supplied images are available for visual assessment. Cite actual image evidence IDs and connect the observed plots to the numerical evidence."
                if mode == "visual"
                else "Visual inspection is unavailable: no images were supplied. Do not claim to have seen, inspected or compared plots or images, and do not cite image IDs. Use qualitativeFindings to interpret the reported marker identities, PCA loading genes, feature families and structured diagnostic tables. State limitations and defer if the supplied evidence cannot resolve an essential question."
            )
            try:
                result = run_agent_sync(
                    model=self.owner.model,
                    output_type=_assessment_output_type(
                        [item.candidateId for item in candidates],
                        list(experiments),
                        scope=scope,
                    ),
                    system_prompt=prompt,
                    user_prompt=build_visual_evidence_prompt(
                        json.dumps(evidence, sort_keys=True), images
                    )
                    if mode == "visual"
                    else json.dumps(evidence, sort_keys=True),
                    config=self.request.config.agentRunConfig,
                    name=f"rna_{scope}_assessment",
                    output_validator=validate,
                )
            except ImageInputUnsupportedError:
                if mode != "visual":
                    raise
                journal.save_checkpoint(
                    self.store,
                    self.prefix,
                    self.workflow.workflowRunId,
                    capability_key,
                    inputs=capability_inputs,
                    outputs={
                        "evidenceMode": "structured",
                        "reason": "providerRejectedImageInput",
                    },
                )
                logger.info(
                    "Analysis assessment: model rejected images; continuing with structured marker, PCA and diagnostic evidence."
                )
                return self.review(scope, review_index, selected, coverage)
            action = validate(result.output)
        journal.save_checkpoint(
            self.store,
            self.prefix,
            self.workflow.workflowRunId,
            key,
            inputs=evidence,
            outputs={"action": action.model_dump(mode="json")},
        )
        self.history.append(
            {
                "scope": scope,
                "evidenceMode": mode,
                "visualInspection": visual_inspection,
                "review": action.model_dump(mode="json"),
                "imageHashes": image_hashes,
                "checkpointKey": key,
                "checkpointSha256": hashlib.sha256(
                    record_io.canonical_json_bytes(
                        {
                            "inputs": evidence,
                            "outputs": {"action": action.model_dump(mode="json")},
                        }
                    )
                ).hexdigest(),
            }
        )
        self.last_action = action
        operation = (
            f"experiment {action.experimentId} from {action.selectedCandidateId}"
            if action.action == "experiment"
            else f"{action.action} {action.selectedCandidateId}"
        )
        logger.info(f"Analysis assessment ({operation}): {action.rationale}")
        return action

    def assess_scope(
        self,
        scope: str,
        cells: ArtifactRef,
        initial: RnaSetting | None,
    ) -> tuple[
        Literal["accept", "enlarge", "defer"], ParameterCandidateEvaluation | None
    ]:
        coverage, insufficient = screening_coverage(
            self.store,
            self.cells,
            cells,
            self.coverage_columns,
            self.study.protectedCombinations,
        )
        self.scope_sizes[scope] = coverage["screeningCells"]
        self.history.append(
            {"scope": scope, "coverage": coverage, "coverageConcerns": insufficient}
        )
        if scope != "full" and insufficient:
            return "enlarge", None
        if initial is None:
            settings = [
                self.baseline(resolution) for resolution in (0.5, 0.75, 1.0, 1.25)
            ]
            settings = [
                value.model_copy(
                    update={
                        "parameters": value.parameters.model_copy(
                            update={
                                "dimensions": min(
                                    value.parameters.dimensions,
                                    coverage["screeningCells"] - 1,
                                ),
                                "neighborsK": min(
                                    value.parameters.neighborsK,
                                    coverage["screeningCells"] - 1,
                                ),
                            }
                        )
                    }
                )
                for value in settings
            ]
            self.budget.admit_many(
                scope, [self.execution_inputs(cells, value) for value in settings]
            )
            baseline = [self.execute(scope, cells, value) for value in settings]
            selected = next(
                (item for item in baseline if item.parameters.leidenResolution == 1.0),
                baseline[0],
            )
        else:
            selected = self.execute_matched(scope, cells, initial)
        limit = (
            self.request.config.maxFullPartitions
            if scope == "full"
            else self.request.config.maxScreeningEvaluations
        )
        for review_index in range(limit + 1):
            action = self.review(scope, review_index, selected, coverage)
            selected = next(
                item
                for item in self.evaluations[scope]
                if item.candidateId == action.selectedCandidateId
            )
            if action.action in {"accept", "enlarge", "defer"}:
                if (
                    action.action == "accept"
                    and scope != "full"
                    and selected.metrics.minClusterCells is not None
                    and selected.metrics.minClusterCells < 20
                ):
                    self.history.append(
                        {
                            "scope": scope,
                            "reason": "The selected sampled partition contains a small population requiring more cells for assessment; all its cells are retained.",
                        }
                    )
                    return "enlarge", selected
                return action.action, selected
            assert action.experimentId is not None
            experiment = self.experiments(selected)[action.experimentId]
            if scope == "full" and experiment["parameter"] != "useHarmony":
                if self.full_repairs >= self.request.config.maxFullRepairs:
                    raise CandidateBudgetExceeded(
                        "The allowed full-cohort repair has been used; scientific acceptance remains unresolved"
                    )
                self.full_repairs += 1
            if experiment["parameter"] in {
                "hvgRanking",
                "hvgCount",
                "includeFamily",
                "excludeFamily",
                "includeFeature",
                "excludeFeature",
            }:
                feature_key = (
                    f"parameter_tuning/{scope}/review{review_index}/feature_experiment"
                )
                feature_inputs = {
                    "baseline": self.settings[selected.candidateId].model_dump(
                        mode="json"
                    ),
                    "experiment": experiment,
                    "cells": self.cells.to_dict(),
                }
                saved_feature = journal.load_checkpoint(
                    self.store,
                    self.prefix,
                    self.workflow.workflowRunId,
                    feature_key,
                    inputs=feature_inputs,
                )
                if saved_feature is None:
                    proposed = self.execution_inputs(
                        cells, self.settings[selected.candidateId]
                    )
                    proposed["features"] = {
                        "requestedFeatureExperiment": feature_inputs
                    }
                    proposals = [proposed]
                    if selected.parameters.useHarmony:
                        proposals.append(
                            {
                                **proposed,
                                "parameters": {
                                    **proposed["parameters"],
                                    "useHarmony": False,
                                },
                            }
                        )
                    self.budget.check_many(scope, proposals)
                    setting = self.apply_experiment(selected, experiment)
                    journal.save_checkpoint(
                        self.store,
                        self.prefix,
                        self.workflow.workflowRunId,
                        feature_key,
                        inputs=feature_inputs,
                        outputs={"setting": setting.model_dump(mode="json")},
                    )
                else:
                    setting = RnaSetting.model_validate(saved_feature["setting"])
            else:
                setting = self.apply_experiment(selected, experiment)
            next_selected = self.execute_matched(scope, cells, setting)
            if next_selected.candidateId == selected.candidateId:
                self.history.append(
                    {
                        "scope": scope,
                        "experiment": action.experimentId,
                        "result": "The intervention did not change the selected genes or numerical representation; exact artifacts were reused.",
                    }
                )
            selected = next_selected
        return "defer", selected

    def run(self) -> tuple[ParameterTuningReport, dict[str, Any]]:
        selected: ParameterCandidateEvaluation | None = None
        final_status = "defer"
        reason = "Required scientific evidence remains unresolved."
        try:
            initial: RnaSetting | None = None
            if self.handoff.nCells > self.request.config.screeningCells:
                for index, size in enumerate(
                    (
                        self.request.config.screeningCells,
                        self.request.config.maxScreeningCells,
                    )
                ):
                    sample = uniform_screening_selection(
                        self.store,
                        self.cells,
                        size=size,
                        seed=self.request.config.randomSeed,
                    )
                    if sample == self.cells:
                        break
                    status, screened = self.assess_scope(f"sample{index}", sample, None)
                    if status == "accept" and screened is not None:
                        initial = self.settings[screened.candidateId]
                        break
                    if status == "defer":
                        return self.report(
                            None,
                            self.last_action.rationale
                            if self.last_action is not None
                            else reason,
                        ), self.summary()
                else:
                    logger.info(
                        "Screening evidence remains insufficient; assessing the bounded full Scarf baseline."
                    )
            final_status, selected = self.assess_scope("full", self.cells, initial)
            if final_status != "accept" and self.last_action is not None:
                reason = self.last_action.rationale
        except CandidateBudgetExceeded as exc:
            reason = f"Scientific assessment paused: {exc}"
        if final_status != "accept":
            selected = None
        return self.report(selected, reason), self.summary()

    def summary(self) -> dict[str, Any]:
        budget = self.budget.summary()
        diagnostic_artifacts: dict[str, set[ArtifactRef]] = {
            "pcaDiagnostics": set(),
            "stabilityClusters": set(),
            "markerTable": set(),
            "doubletScore": set(),
        }
        subsample_evaluations = {}
        for scope, evaluations in self.evaluations.items():
            subsample_evaluations[scope] = len(
                {
                    evaluation.candidateId
                    for evaluation in evaluations
                    if evaluation.status == "done"
                    and evaluation.metrics.subsampleStability is not None
                }
            )
            for evaluation in evaluations:
                if evaluation.status != "done":
                    continue
                for name, artifact in evaluation.artifacts.items():
                    diagnostic = (
                        "pcaDiagnostics"
                        if name == "representationDiagnostic"
                        else name.split(":", 1)[0]
                    )
                    if diagnostic in diagnostic_artifacts:
                        diagnostic_artifacts[diagnostic].add(
                            artifact_model_to_ref(artifact)
                        )
        diagnostic_counts = {
            name: len(refs) for name, refs in diagnostic_artifacts.items()
        }
        for scope, counts in budget["scopes"].items():
            if counts["reserved"]["partitions"]:
                label = {
                    "sample0": "screening sample 1",
                    "sample1": "screening sample 2",
                    "full": "full cohort",
                }[scope]
                completed, reserved = counts["completed"], counts["reserved"]
                logger.info(
                    f"Tuning {label}: {completed['graphs']}/{reserved['graphs']} graphs "
                    f"and {completed['partitions']}/{reserved['partitions']} partitions "
                    "completed/reserved."
                )
        limits = budget["limits"]
        logger.info(
            f"Tuning limits: {limits['perScreen']} partitions per screen, "
            f"{limits['totalScreens']} across screens; full cohort "
            f"{limits['fullGraphs']} graphs and {limits['fullPartitions']} partitions."
        )
        logger.info(
            f"Diagnostic evidence: {diagnostic_counts['pcaDiagnostics']} PCA summaries, "
            f"{diagnostic_counts['stabilityClusters']} alternate-seed partitions, "
            f"{diagnostic_counts['markerTable']} marker tables, "
            f"{diagnostic_counts['doubletScore']} doublet scores; "
            f"{sum(subsample_evaluations.values())} subsample-stability evaluations. "
            "Saved evidence may be reused; these are not computation counts."
        )
        return {
            "history": self.history,
            "budget": budget,
            "diagnosticEvidence": {
                "uniqueArtifacts": diagnostic_counts,
                "subsampleStabilityEvaluations": subsample_evaluations,
                "interpretation": (
                    "Counts describe saved diagnostic evidence used by these evaluations "
                    "and may include reused artifacts or metrics, not new computations."
                ),
            },
            "fullRepairs": self.full_repairs,
        }

    def report(
        self, selected: ParameterCandidateEvaluation | None, reason: str
    ) -> ParameterTuningReport:
        evaluations = self.evaluations["full"]
        common: dict[str, Any] = {
            "fromAssay": self.handoff.assay,
            "cellSelection": self.handoff.cellSelection,
            "evaluations": evaluations,
            "totalCandidates": len(evaluations),
            "markerAssay": self.handoff.assay,
            "limitations": [
                "Screening comparisons describe their exact sampled cells; final artifacts and validation use the full QC-retained cohort.",
                *self.study.limitations,
                *unsupported_comparison_limitations(self.design_comparisons),
                *(
                    [_STRUCTURED_VISUAL_LIMITATION]
                    if any(
                        row.get("evidenceMode") == "structured" for row in self.history
                    )
                    else []
                ),
                *(
                    f"Unsupported matched protection: {column}."
                    for column in self.study.unsupportedProtection
                ),
                *(str(row["reason"]) for row in self.history if "reason" in row),
                *(
                    "Screening coverage concern: " + concern
                    for row in self.history
                    for concern in row.get("coverageConcerns", [])
                ),
            ],
        }
        common["limitations"] = list(dict.fromkeys(common["limitations"]))
        if selected is None:
            return ParameterTuningReport(
                **common,
                status="needsInput",
                rationale=reason,
                stopReason=reason,
                needsInput=ParameterTuningNeedsInput(question=reason),
            )
        assert self.last_action is not None
        report = ParameterTuningReport(
            **common,
            status="done",
            recommendedCandidateId=selected.candidateId,
            selectedArtifacts=selected.artifacts,
            confidence="medium",
            rationale=self.last_action.rationale,
            evidenceIds=self.last_action.evidenceIds,
            stopReason="The objective-driven assessment accepted the full-cohort evidence.",
            recommendedByAssay={self.handoff.assay: selected.candidateId},
        )
        return finalize_parameter_tuning_selection(
            report, marker_assay=self.handoff.assay, native_assay=self.handoff.assay
        )
