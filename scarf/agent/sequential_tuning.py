"""Causal phase planning for RNA parameter adjudication.

This module constructs executor-compatible candidate sets one scientific choice
at a time. It does not call a model and it never chooses a fallback candidate.
The orchestration layer can persist ``ParameterPhaseEvidence`` after requesting
an exact candidate ID from an agent or human.
"""

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from .parameter_tuning import (
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterSearchPlan,
    ParameterTuningDependencies,
    ParameterTuningNeedsInput,
    ParameterTuningReport,
    annotate_candidate_dominance,
    execute_parameter_candidate,
    execute_parameter_search_plan,
    finalize_parameter_tuning_selection,
    prepare_parameter_tuning_dependencies,
    require_dominated_candidate_evidence,
    validate_parameter_search_plan,
)
from .tools import core_artifact_reference
from .types import AgentDataModel, ExperimentalTuningHandoff


type ParameterPhase = Literal[
    "pcaPrefix",
    "batchCorrection",
    "graphK",
    "clusteringResolution",
]
type ParameterPhaseStatus = Literal["selected", "needsInput", "abstained"]
type ParameterDecisionSource = Literal["rule", "agent", "human"]
type VariedParameter = Literal[
    "dimensions",
    "useHarmony",
    "neighborsK",
    "leidenResolution",
]

_PHASE_ORDER: tuple[ParameterPhase, ...] = (
    "pcaPrefix",
    "batchCorrection",
    "graphK",
    "clusteringResolution",
)
_VARIED_PARAMETER: dict[ParameterPhase, VariedParameter] = {
    "pcaPrefix": "dimensions",
    "batchCorrection": "useHarmony",
    "graphK": "neighborsK",
    "clusteringResolution": "leidenResolution",
}
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{0,63}$")


class SequentialTuningModel(AgentDataModel):
    """Strict immutable base for sequential tuning records."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ParameterPhasePlan(SequentialTuningModel):
    """One exact candidate set varying a single parameter."""

    phase: ParameterPhase
    assay: str = Field(min_length=1, max_length=256)
    variedParameter: VariedParameter
    basedOnCandidateId: str | None = None
    candidates: list[ParameterCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_causal_candidate_set(self) -> "ParameterPhasePlan":
        if self.variedParameter != _VARIED_PARAMETER[self.phase]:
            raise ValueError("variedParameter does not match the tuning phase")
        if (self.phase == "pcaPrefix") != (self.basedOnCandidateId is None):
            raise ValueError("Only the PCA-prefix phase may omit basedOnCandidateId")
        if self.basedOnCandidateId is not None and (
            _CANDIDATE_ID.fullmatch(self.basedOnCandidateId) is None
        ):
            raise ValueError("basedOnCandidateId is not a stable candidate ID")
        candidate_ids = [candidate.candidateId for candidate in self.candidates]
        if any(_CANDIDATE_ID.fullmatch(value) is None for value in candidate_ids):
            raise ValueError("Candidate IDs must be stable non-empty identifiers")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("A parameter phase cannot contain duplicate candidate IDs")
        if any(candidate.reductionMethod != "pca" for candidate in self.candidates):
            raise ValueError("Sequential v1 tuning accepts RNA PCA candidates only")

        varied_values = [
            getattr(candidate, self.variedParameter) for candidate in self.candidates
        ]
        if len(varied_values) != len(set(varied_values)):
            raise ValueError("A phase must vary its target parameter exactly once")
        fixed_names = {
            "dimensions",
            "useHarmony",
            "neighborsK",
            "leidenResolution",
        } - {self.variedParameter}
        for field_name in fixed_names:
            values = {getattr(candidate, field_name) for candidate in self.candidates}
            if len(values) != 1:
                raise ValueError(
                    f"Phase {self.phase!r} changes non-target parameter {field_name!r}"
                )
        if self.phase == "pcaPrefix" and any(
            candidate.useHarmony for candidate in self.candidates
        ):
            raise ValueError("PCA-prefix candidates must use the native representation")
        if self.phase == "batchCorrection":
            harmony_values = {candidate.useHarmony for candidate in self.candidates}
            if False not in harmony_values or not harmony_values.issubset(
                {False, True}
            ):
                raise ValueError(
                    "Batch-correction candidates must include the native baseline"
                )
        return self

    def candidate_by_id(self) -> dict[str, ParameterCandidate]:
        """Return this phase's exact executor candidates by ID."""
        return {candidate.candidateId: candidate for candidate in self.candidates}


class ParameterPhaseSelection(SequentialTuningModel):
    """The only model-authored output for one tuning phase."""

    phase: ParameterPhase
    status: ParameterPhaseStatus
    selectedCandidateId: str | None = None
    evidenceIds: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_selection_shape(self) -> "ParameterPhaseSelection":
        if (self.status == "selected") != (self.selectedCandidateId is not None):
            raise ValueError("Only a selected phase may contain selectedCandidateId")
        if self.selectedCandidateId is not None and (
            _CANDIDATE_ID.fullmatch(self.selectedCandidateId) is None
        ):
            raise ValueError("selectedCandidateId is not a stable candidate ID")
        if len(self.evidenceIds) != len(set(self.evidenceIds)):
            raise ValueError("evidenceIds must not contain duplicates")
        if any(not value for value in self.evidenceIds):
            raise ValueError("evidenceIds must contain non-empty values")
        if self.status == "selected" and not self.evidenceIds:
            raise ValueError("A selected phase must cite executor evidence")
        if self.rationale != self.rationale.strip():
            raise ValueError("rationale must not contain surrounding whitespace")
        return self


class ParameterPhaseEvidence(SequentialTuningModel):
    """Executed evidence and one validated selection for a causal phase."""

    plan: ParameterPhasePlan
    evaluations: list[ParameterCandidateEvaluation] = Field(min_length=1)
    selection: ParameterPhaseSelection

    @model_validator(mode="after")
    def validate_execution_and_selection(self) -> "ParameterPhaseEvidence":
        if self.selection.phase != self.plan.phase:
            raise ValueError("Phase selection does not match its candidate plan")
        candidates = self.plan.candidate_by_id()
        evaluations = {value.candidateId: value for value in self.evaluations}
        if len(evaluations) != len(self.evaluations):
            raise ValueError("Phase evaluations contain duplicate candidate IDs")
        if set(evaluations) != set(candidates):
            raise ValueError("Phase evaluations must cover the exact candidate set")
        cell_selections = [
            value.cellSelection
            for value in self.evaluations
            if value.cellSelection is not None
        ]
        if cell_selections and any(
            value != cell_selections[0] for value in cell_selections[1:]
        ):
            raise ValueError("Phase evaluations must use one exact cell selection")
        for candidate_id, evaluation in evaluations.items():
            if evaluation.parameters != candidates[candidate_id]:
                raise ValueError(
                    f"Evaluation {candidate_id!r} changed its registered parameters"
                )
        available_evidence = {
            evidence_id
            for evaluation in self.evaluations
            for evidence_id in evaluation.evidenceIds
        }
        if not set(self.selection.evidenceIds).issubset(available_evidence):
            raise ValueError("Selection cites evidence outside its phase evaluations")
        if self.selection.status == "selected":
            selected = evaluations.get(self.selection.selectedCandidateId or "")
            if selected is None or selected.status != "done" or not selected.eligible:
                raise ValueError(
                    "Selected candidate must be an eligible completed execution"
                )
            if selected.cellSelection is None:
                raise ValueError("Selected candidate lacks an exact cell selection")
            selected_evidence = set(selected.evidenceIds)
            if not selected_evidence.intersection(self.selection.evidenceIds):
                raise ValueError(
                    "A selected phase must cite evidence from its selected candidate"
                )
            if self.plan.phase in {"graphK", "clusteringResolution"}:
                dominance_evaluations = {
                    evaluation.candidateId: evaluation
                    for evaluation in annotate_candidate_dominance(self.evaluations)
                }
                require_dominated_candidate_evidence(
                    dominance_evaluations[selected.candidateId],
                    self.selection.evidenceIds,
                    context=f"The {self.plan.phase} selection",
                )
        if self.selection.status == "abstained" and self.plan.phase != (
            "clusteringResolution"
        ):
            raise ValueError("Only clustering may produce scientific abstention")
        return self

    def selected_evaluation(self) -> ParameterCandidateEvaluation | None:
        """Return the selected eligible evaluation, or None after a pause."""
        if self.selection.selectedCandidateId is None:
            return None
        return next(
            value
            for value in self.evaluations
            if value.candidateId == self.selection.selectedCandidateId
        )


class CorrectionNeedSelection(SequentialTuningModel):
    """Separate semantic decision about whether correction is needed."""

    status: Literal["selected", "needsInput"]
    selectedOptionId: Literal[
        "correctionNeed:needed",
        "correctionNeed:notNeeded",
        "correctionNeed:indeterminate",
    ]
    evidenceIds: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_need(self) -> "CorrectionNeedSelection":
        is_indeterminate = self.selectedOptionId == "correctionNeed:indeterminate"
        if (self.status == "needsInput") != is_indeterminate:
            raise ValueError(
                "Only correctionNeed:indeterminate may have needsInput status"
            )
        if len(self.evidenceIds) != len(set(self.evidenceIds)):
            raise ValueError("evidenceIds must not contain duplicates")
        if self.status == "selected" and not self.evidenceIds:
            raise ValueError("A correction-need decision must cite evidence")
        if self.rationale != self.rationale.strip():
            raise ValueError("rationale must not contain surrounding whitespace")
        return self


class SequentialAssayTuningEvidence(SequentialTuningModel):
    """Ordered, persistable evidence for one RNA assay's four decisions."""

    assay: str = Field(min_length=1, max_length=256)
    phases: list[ParameterPhaseEvidence] = Field(min_length=1, max_length=4)
    correctionLicense: Literal[
        "safe", "unsafeConfounded", "indeterminate", "notApplicable"
    ] = "notApplicable"
    correctionNeed: CorrectionNeedSelection | None = None
    decisionSources: dict[str, ParameterDecisionSource] = Field(default_factory=dict)
    pendingDecisionId: (
        Literal[
            "pcaPrefix",
            "correctionLicense",
            "correctionNeed",
            "correctionOutcome",
            "graphK",
            "clusterPartition",
        ]
        | None
    ) = None
    pendingOptionIds: list[str] = Field(default_factory=list)
    pendingEvidenceIds: list[str] = Field(default_factory=list)
    finalCandidateId: str | None = None

    @model_validator(mode="after")
    def validate_phase_lineage(self) -> "SequentialAssayTuningEvidence":
        observed_order = tuple(item.plan.phase for item in self.phases)
        if observed_order != _PHASE_ORDER[: len(observed_order)]:
            raise ValueError("Sequential tuning phases are missing or out of order")
        if any(item.plan.assay != self.assay for item in self.phases):
            raise ValueError("Sequential tuning phases must use one assay")
        valid_decision_ids = {
            "pcaPrefix",
            "correctionLicense",
            "correctionNeed",
            "correctionOutcome",
            "graphK",
            "clusterPartition",
        }
        if not set(self.decisionSources).issubset(valid_decision_ids):
            raise ValueError("decisionSources contains an unknown RNA decision")
        for field_name, values in (
            ("pendingOptionIds", self.pendingOptionIds),
            ("pendingEvidenceIds", self.pendingEvidenceIds),
        ):
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(
                    f"{field_name} must contain unique non-empty identifiers"
                )
        if self.pendingDecisionId is None:
            if self.pendingOptionIds or self.pendingEvidenceIds:
                raise ValueError(
                    "Pending option and evidence IDs require pendingDecisionId"
                )
        elif not self.pendingOptionIds:
            raise ValueError("A pending decision requires its exact offered options")
        has_batch_phase = any(
            item.plan.phase == "batchCorrection" for item in self.phases
        )
        if self.correctionLicense == "safe" and (
            has_batch_phase or self.pendingDecisionId == "correctionNeed"
        ):
            if self.correctionNeed is None:
                raise ValueError("A safe correction branch requires correctionNeed")
        elif self.correctionNeed is not None:
            raise ValueError(
                "Correction need must be absent without a safe correction license"
            )
        if self.pendingDecisionId == "correctionNeed" and (
            self.correctionNeed is None or self.correctionNeed.status != "needsInput"
        ):
            raise ValueError("pending correctionNeed requires an indeterminate need")
        if self.pendingDecisionId == "correctionOutcome":
            batch_phases = [
                item for item in self.phases if item.plan.phase == "batchCorrection"
            ]
            if (
                not batch_phases
                or batch_phases[-1].selection.status != "needsInput"
                or self.correctionNeed is None
                or self.correctionNeed.selectedOptionId != "correctionNeed:needed"
            ):
                raise ValueError(
                    "pending correctionOutcome requires a needed correction and pause"
                )
        for index, phase in enumerate(self.phases[1:], start=1):
            previous = self.phases[index - 1]
            previous_evaluation = previous.selected_evaluation()
            if previous_evaluation is None:
                raise ValueError("No phase may follow needsInput or abstained")
            if phase.plan.basedOnCandidateId != previous_evaluation.candidateId:
                raise ValueError("Phase basedOnCandidateId breaks selection lineage")
            target = phase.plan.variedParameter
            for candidate in phase.plan.candidates:
                for field_name in (
                    "dimensions",
                    "useHarmony",
                    "neighborsK",
                    "leidenResolution",
                ):
                    if field_name == target:
                        continue
                    if getattr(candidate, field_name) != getattr(
                        previous_evaluation.parameters,
                        field_name,
                    ):
                        raise ValueError(
                            f"Phase {phase.plan.phase!r} does not preserve "
                            f"selected {field_name!r}"
                        )
        completed = (
            len(self.phases) == len(_PHASE_ORDER)
            and self.phases[-1].selection.status == "selected"
        )
        expected_final = (
            self.phases[-1].selection.selectedCandidateId if completed else None
        )
        if self.finalCandidateId != expected_final:
            raise ValueError("finalCandidateId requires four selected causal phases")
        return self


class SequentialRefinementResult(SequentialTuningModel):
    """One validated post-grid refinement disposition and optional execution."""

    plan: ParameterSearchPlan
    evaluation: ParameterCandidateEvaluation | None = None

    @model_validator(mode="after")
    def validate_refinement_result(self) -> "SequentialRefinementResult":
        if (
            not self.plan.basedOnCandidateIds
            or not self.plan.evidenceIds
            or not self.plan.rationale.strip()
            or not self.plan.stoppingCriteria
        ):
            raise ValueError(
                "Sequential refinement results require parents, evidence, "
                "rationale, and stopping criteria"
            )
        if self.plan.status == "complete":
            if self.plan.candidates or self.evaluation is not None:
                raise ValueError(
                    "A complete refinement review cannot contain an execution"
                )
            return self
        if len(self.plan.candidates) != 1 or self.evaluation is None:
            raise ValueError(
                "A refinement review must contain exactly one candidate execution"
            )
        candidate = self.plan.candidates[0]
        if (
            self.evaluation.candidateId != candidate.candidateId
            or self.evaluation.parameters != candidate
            or self.evaluation.phase != "refined"
        ):
            raise ValueError("Refinement execution does not match its validated plan")
        return self


class SequentialRefinementSelection(SequentialTuningModel):
    """Explicit final choice after an optional refinement execution."""

    selectedCandidateId: str
    evidenceIds: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_selection(self) -> "SequentialRefinementSelection":
        if _CANDIDATE_ID.fullmatch(self.selectedCandidateId) is None:
            raise ValueError("selectedCandidateId is not a stable candidate ID")
        if len(self.evidenceIds) != len(set(self.evidenceIds)):
            raise ValueError("evidenceIds must not contain duplicates")
        if any(not value for value in self.evidenceIds):
            raise ValueError("evidenceIds must contain non-empty values")
        if self.rationale != self.rationale.strip():
            raise ValueError("rationale must not contain surrounding whitespace")
        return self


class SequentialRnaTuningPlanner:
    """Construct fixed, rank-capped candidates for four causal RNA phases."""

    def __init__(
        self,
        *,
        workflow_run_id: str,
        assay: str,
        n_cells: int,
        n_features: int,
        harmony_authorized: bool,
        matrix_rank: int | None = None,
        dimension_candidates: Sequence[int] = (10, 20, 30, 50),
        neighbor_candidates: Sequence[int] = (11, 21, 41),
        resolution_candidates: Sequence[float] = (
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
        ),
    ) -> None:
        if not workflow_run_id or not assay:
            raise ValueError("workflow_run_id and assay must be non-empty")
        if isinstance(n_cells, bool) or not isinstance(n_cells, int) or n_cells < 3:
            raise ValueError("Sequential tuning requires at least three cells")
        if (
            isinstance(n_features, bool)
            or not isinstance(n_features, int)
            or n_features < 3
        ):
            raise ValueError("Sequential tuning requires at least three features")
        if not isinstance(harmony_authorized, bool):
            raise TypeError("harmony_authorized must be a boolean")
        maximum_rank = min(n_cells, n_features) - 1
        if matrix_rank is not None:
            if (
                isinstance(matrix_rank, bool)
                or not isinstance(matrix_rank, int)
                or not 2 <= matrix_rank <= maximum_rank
            ):
                raise ValueError(
                    "matrix_rank must be between two and the shape-derived rank cap"
                )
            maximum_rank = matrix_rank
        self.workflow_run_id = workflow_run_id
        self.assay = assay
        self.n_cells = n_cells
        self.n_features = n_features
        self.harmony_authorized = harmony_authorized
        self.dimensions = self._capped_integers(
            dimension_candidates,
            maximum=maximum_rank,
            name="dimension_candidates",
        )
        self.neighbors = self._capped_integers(
            neighbor_candidates,
            maximum=n_cells - 1,
            name="neighbor_candidates",
        )
        resolutions = tuple(float(value) for value in resolution_candidates)
        if (
            not resolutions
            or any(not 0 < value < float("inf") for value in resolutions)
            or len(resolutions) != len(set(resolutions))
        ):
            raise ValueError(
                "resolution_candidates must be unique, finite, and positive"
            )
        self.resolutions = resolutions
        token = re.sub(r"[^A-Za-z0-9]+", "_", workflow_run_id).strip("_")[:12]
        assay_token = re.sub(r"[^A-Za-z0-9]+", "_", assay).strip("_")[:12]
        digest = hashlib.blake2b(
            f"{workflow_run_id}\0{assay}".encode(),
            digest_size=5,
        ).hexdigest()
        self.prefix = f"seq_{token or 'run'}_{assay_token or 'assay'}_{digest}"

    @staticmethod
    def _capped_integers(
        values: Sequence[int],
        *,
        maximum: int,
        name: str,
    ) -> tuple[int, ...]:
        raw = tuple(values)
        if not raw or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in raw
        ):
            raise ValueError(f"{name} must contain integers of at least two")
        capped = tuple(dict.fromkeys(min(value, maximum) for value in raw))
        if not capped or any(value < 2 for value in capped):
            raise ValueError(f"{name} has no rank-valid values")
        return capped

    def pca_prefix_phase(self) -> ParameterPhasePlan:
        """Vary only the bounded PCA prefix on the native representation."""
        audit_k = min(21, self.n_cells - 1)
        return ParameterPhasePlan(
            phase="pcaPrefix",
            assay=self.assay,
            variedParameter="dimensions",
            candidates=[
                ParameterCandidate(
                    candidateId=f"{self.prefix}_pca_{dimensions}",
                    reductionMethod="pca",
                    dimensions=dimensions,
                    leidenResolution=1.0,
                    neighborsK=audit_k,
                    useHarmony=False,
                )
                for dimensions in self.dimensions
            ],
        )

    def batch_correction_phase(
        self,
        selected: ParameterCandidate,
    ) -> ParameterPhasePlan:
        """Compare matched native and Harmony representations when licensed."""
        self._require_selected_pca(selected)
        methods = (False, True) if self.harmony_authorized else (False,)
        return ParameterPhasePlan(
            phase="batchCorrection",
            assay=self.assay,
            variedParameter="useHarmony",
            basedOnCandidateId=selected.candidateId,
            candidates=[
                selected.model_copy(
                    update={
                        "candidateId": (
                            f"{self.prefix}_correction_"
                            f"{'harmony' if use_harmony else 'native'}"
                        ),
                        "useHarmony": use_harmony,
                    }
                )
                for use_harmony in methods
            ],
        )

    def graph_phase(self, selected: ParameterCandidate) -> ParameterPhasePlan:
        """Vary only graph neighbourhood size after representation selection."""
        self._require_selected_pca(selected)
        return ParameterPhasePlan(
            phase="graphK",
            assay=self.assay,
            variedParameter="neighborsK",
            basedOnCandidateId=selected.candidateId,
            candidates=[
                selected.model_copy(
                    update={
                        "candidateId": f"{self.prefix}_graph_k{k}",
                        "neighborsK": k,
                    }
                )
                for k in self.neighbors
            ],
        )

    def clustering_phase(
        self,
        selected: ParameterCandidate,
    ) -> ParameterPhasePlan:
        """Vary only Leiden resolution on the selected graph configuration."""
        self._require_selected_pca(selected)
        return ParameterPhasePlan(
            phase="clusteringResolution",
            assay=self.assay,
            variedParameter="leidenResolution",
            basedOnCandidateId=selected.candidateId,
            candidates=[
                selected.model_copy(
                    update={
                        "candidateId": (
                            f"{self.prefix}_resolution_"
                            f"{str(resolution).replace('.', 'p')}"
                        ),
                        "leidenResolution": resolution,
                    }
                )
                for resolution in self.resolutions
            ],
        )

    @staticmethod
    def _require_selected_pca(candidate: ParameterCandidate) -> None:
        if not isinstance(candidate, ParameterCandidate):
            raise TypeError("selected must be a ParameterCandidate")
        if candidate.reductionMethod != "pca" or not candidate.candidateId:
            raise ValueError("selected must be an exact RNA PCA candidate")


def validate_parameter_phase_selection(
    plan: ParameterPhasePlan,
    evaluations: Sequence[ParameterCandidateEvaluation],
    selection: ParameterPhaseSelection,
) -> ParameterPhaseEvidence:
    """Validate an ID-only selection against complete executor evidence."""
    return ParameterPhaseEvidence(
        plan=plan,
        evaluations=list(annotate_candidate_dominance(evaluations)),
        selection=selection,
    )


def execute_parameter_phase(
    store: Any,
    *,
    normalized: Any,
    plan: ParameterPhasePlan,
    batch_columns: Sequence[str] = (),
    preservation_columns: Sequence[str] = (),
    experimental_handoff: ExperimentalTuningHandoff | None = None,
    min_cluster_cells: int = 20,
    identity_feature_limit: int = 64,
) -> tuple[ParameterCandidateEvaluation, ...]:
    """Execute one phase through the existing deterministic candidate executor."""
    deps, candidate_ids = prepare_parameter_tuning_dependencies(
        store,
        normalized=normalized,
        candidates=plan.candidates,
        batch_columns=batch_columns,
        preservation_columns=preservation_columns,
        experimental_handoff=experimental_handoff,
        max_candidates=len(plan.candidates),
        max_refined_candidates=0,
        min_cluster_cells=min_cluster_cells,
        identity_feature_limit=identity_feature_limit,
    )
    expected_ids = tuple(candidate.candidateId for candidate in plan.candidates)
    if tuple(candidate_ids) != expected_ids:
        raise ValueError("Prepared executor candidate inventory changed the phase plan")
    return annotate_candidate_dominance(
        tuple(execute_parameter_candidate(deps, value) for value in candidate_ids)
    )


def _sequential_candidate_evaluations(
    evidence: SequentialAssayTuningEvidence,
) -> tuple[ParameterCandidateEvaluation, ...]:
    if evidence.finalCandidateId is None:
        raise ValueError("Sequential refinement requires complete grid evidence")
    evaluations = tuple(
        evaluation for phase in evidence.phases for evaluation in phase.evaluations
    )
    if not evaluations:
        raise ValueError("Sequential refinement requires executed grid candidates")
    candidate_ids = [evaluation.candidateId for evaluation in evaluations]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Sequential grid candidate IDs must be unique")
    if evidence.finalCandidateId not in set(candidate_ids):
        raise ValueError("Sequential final candidate is not present in grid evidence")
    return evaluations


def prepare_sequential_refinement_dependencies(
    store: Any,
    *,
    normalized: Any,
    evidence: SequentialAssayTuningEvidence,
    batch_columns: Sequence[str] = (),
    preservation_columns: Sequence[str] = (),
    experimental_handoff: ExperimentalTuningHandoff | None = None,
    min_cluster_cells: int = 20,
    identity_feature_limit: int = 64,
) -> tuple[ParameterTuningDependencies, list[str]]:
    """Prepare one refinement executor from already executed sequential candidates."""

    evaluations = _sequential_candidate_evaluations(evidence)
    candidates = [evaluation.parameters for evaluation in evaluations]
    deps, candidate_ids = prepare_parameter_tuning_dependencies(
        store,
        normalized=normalized,
        candidates=candidates,
        batch_columns=batch_columns,
        preservation_columns=preservation_columns,
        experimental_handoff=experimental_handoff,
        max_candidates=len(candidates),
        max_refined_candidates=1,
        min_cluster_cells=min_cluster_cells,
        identity_feature_limit=identity_feature_limit,
        pair_harmony_candidates=False,
    )
    expected_ids = [candidate.candidateId for candidate in candidates]
    if candidate_ids != expected_ids:
        raise ValueError("Prepared refinement inventory changed the grid candidates")
    for evaluation in evaluations:
        if (
            evaluation.status == "done"
            and core_artifact_reference(evaluation.cellSelection) != deps.cellSelection
        ):
            raise ValueError(
                "Sequential grid evaluation does not match the normalized cell axis"
            )
    deps.evaluations = {
        evaluation.candidateId: evaluation for evaluation in evaluations
    }
    deps.executionOrder = list(candidate_ids)
    return deps, candidate_ids


def _changed_refinement_parameters(
    candidate: ParameterCandidate,
    parent: ParameterCandidate,
) -> tuple[str, ...]:
    tunable_fields = (
        "reductionMethod",
        "dimensions",
        "neighborsK",
        "leidenResolution",
        "useHarmony",
    )
    return tuple(
        field_name
        for field_name in tunable_fields
        if getattr(candidate, field_name) != getattr(parent, field_name)
    )


def validate_sequential_refinement_plan(
    plan: ParameterSearchPlan,
    deps: ParameterTuningDependencies,
    initial_candidate_ids: Sequence[str],
) -> ParameterSearchPlan:
    """Validate a no-refinement decision or one bounded sequential candidate."""

    initial_ids = tuple(initial_candidate_ids)
    if not initial_ids or len(initial_ids) != len(set(initial_ids)):
        raise ValueError("Sequential refinement requires unique grid candidate IDs")
    if set(deps.evaluations) != set(initial_ids):
        raise ValueError("Refinement dependencies do not match the executed grid")
    eligible_ids = {
        candidate_id
        for candidate_id in initial_ids
        if deps.evaluations[candidate_id].status == "done"
        and deps.evaluations[candidate_id].eligible
    }
    if not eligible_ids:
        raise ValueError("Sequential refinement requires an eligible successful parent")

    validated = validate_parameter_search_plan(
        plan,
        deps,
        initial_candidate_ids=initial_ids,
        max_refined_candidates=1,
    )
    parent_ids = tuple(validated.basedOnCandidateIds)
    if not parent_ids:
        raise ValueError("Sequential refinement must name its successful parent")
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Sequential refinement parent IDs must be unique")
    if any(parent_id not in eligible_ids for parent_id in parent_ids):
        raise ValueError(
            "Sequential refinement parents must be eligible successful grid candidates"
        )
    if any(
        not any(
            evidence_id.startswith(f"candidate:{parent_id}:")
            for evidence_id in validated.evidenceIds
        )
        for parent_id in parent_ids
    ):
        raise ValueError("Sequential refinement must cite each parent candidate")

    if validated.status == "complete":
        if not validated.evidenceIds:
            raise ValueError("A no-refinement decision requires observed evidence")
        if not validated.rationale.strip():
            raise ValueError("A no-refinement decision requires a rationale")
        if not validated.stoppingCriteria:
            raise ValueError("A no-refinement decision requires a stopping criterion")
        return validated

    candidate = validated.candidates[0]
    if candidate.useHarmony:
        raise ValueError(
            "One-candidate sequential refinement cannot evaluate Harmony because "
            "acceptance requires a newly parameter-matched native control"
        )
    parent_changes = {
        parent_id: _changed_refinement_parameters(
            candidate,
            deps.evaluations[parent_id].parameters,
        )
        for parent_id in parent_ids
    }
    if not any(
        len(changes) == 1
        and changes[0] in {"dimensions", "neighborsK", "leidenResolution"}
        for changes in parent_changes.values()
    ):
        raise ValueError(
            "The refinement candidate must vary one numeric parameter from a "
            "cited parent"
        )
    matched_mode = [
        deps.evaluations[candidate_id].parameters
        for candidate_id in initial_ids
        if (
            deps.evaluations[candidate_id].parameters.reductionMethod
            == candidate.reductionMethod
            and deps.evaluations[candidate_id].parameters.useHarmony
            == candidate.useHarmony
        )
    ]
    for field_name in ("dimensions", "neighborsK", "leidenResolution"):
        observed = [getattr(value, field_name) for value in matched_mode]
        if not min(observed) <= getattr(candidate, field_name) <= max(observed):
            raise ValueError(
                f"Refined {field_name} must remain inside its observed "
                "correction-mode envelope"
            )
    if not validated.objectives:
        raise ValueError("A refinement candidate requires an evidence-based objective")
    return validated


def execute_sequential_refinement(
    deps: ParameterTuningDependencies,
    plan: ParameterSearchPlan,
    initial_candidate_ids: Sequence[str],
) -> SequentialRefinementResult:
    """Execute at most one validated post-grid sequential candidate."""

    validated = validate_sequential_refinement_plan(
        plan,
        deps,
        initial_candidate_ids,
    )
    validated, evaluations = execute_parameter_search_plan(
        deps,
        validated,
        initial_candidate_ids=initial_candidate_ids,
        max_refined_candidates=1,
    )
    return SequentialRefinementResult(
        plan=validated,
        evaluation=evaluations[0] if evaluations else None,
    )


def sequential_refinement_selection_candidates(
    evidence: SequentialAssayTuningEvidence,
    refinement: SequentialRefinementResult,
) -> tuple[ParameterCandidateEvaluation, ...]:
    """Return the grid and optional refinement as one final-selection inventory."""

    evaluations = list(_sequential_candidate_evaluations(evidence))
    if refinement.evaluation is not None:
        if refinement.evaluation.candidateId in {
            evaluation.candidateId for evaluation in evaluations
        }:
            raise ValueError("Refinement candidate duplicates a grid candidate ID")
        evaluations.append(refinement.evaluation)
    return annotate_candidate_dominance(evaluations)


def sequential_evidence_to_report(
    evidence: SequentialAssayTuningEvidence,
    *,
    marker_assay: str | None = None,
    refinement: SequentialRefinementResult | None = None,
    refinement_selection: SequentialRefinementSelection | None = None,
) -> ParameterTuningReport:
    """Adapt four selected phases to the report consumed by finalization."""
    if evidence.finalCandidateId is None:
        if refinement is not None or refinement_selection is not None:
            raise ValueError("Post-grid refinement requires four selected phases")
        final_phase = evidence.phases[-1]
        cell_selection = next(
            (
                evaluation.cellSelection
                for evaluation in final_phase.evaluations
                if evaluation.cellSelection is not None
            ),
            None,
        )
        if cell_selection is None:
            raise ValueError(
                "Incomplete sequential evidence lacks an exact cell selection"
            )
        selection_status = final_phase.selection.status
        if selection_status == "selected" and evidence.pendingDecisionId is None:
            raise ValueError(
                "A selected intermediate phase must be followed before report adaptation"
            )
        if evidence.pendingDecisionId is not None or selection_status == "needsInput":
            report_status: Literal["needsInput", "abstained"] = "needsInput"
        else:
            report_status = "abstained"
        needs_input = (
            ParameterTuningNeedsInput(
                question=(
                    f"Resolve the registered {evidence.pendingDecisionId} decision."
                ),
                options=list(evidence.pendingOptionIds),
                evidenceIds=list(evidence.pendingEvidenceIds),
            )
            if report_status == "needsInput"
            else None
        )
        return ParameterTuningReport(
            status=report_status,
            fromAssay=evidence.assay,
            cellSelection=cell_selection,
            evaluations=list(final_phase.evaluations),
            rationale=final_phase.selection.rationale,
            evidenceIds=list(final_phase.selection.evidenceIds),
            limitations=[
                "Sequential parameter adjudication did not select all four phases."
            ],
            stopReason=report_status,
            needsInput=needs_input,
            totalCandidates=sum(len(value.evaluations) for value in evidence.phases),
        )
    final_phase = evidence.phases[-1]
    selected = final_phase.selected_evaluation()
    assert selected is not None
    phase_selected = selected
    phase_evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for value in evidence.phases
            for evidence_id in value.selection.evidenceIds
        )
    )
    evaluations = list(_sequential_candidate_evaluations(evidence))
    search_plan: ParameterSearchPlan | None = None
    rationale = " ".join(value.selection.rationale for value in evidence.phases)
    stop_reason = "Four causal RNA parameter phases were selected."
    evidence_ids = phase_evidence_ids
    if refinement is not None:
        search_plan = refinement.plan
        evaluations = list(
            sequential_refinement_selection_candidates(evidence, refinement)
        )
        evidence_ids = list(
            dict.fromkeys([*phase_evidence_ids, *refinement.plan.evidenceIds])
        )
        rationale = f"{rationale} {refinement.plan.rationale}"
        if refinement.evaluation is None:
            if refinement_selection is not None:
                raise ValueError(
                    "A no-refinement result cannot have a refinement selection"
                )
            stop_reason = "The bounded post-grid review found no justified refinement."
        else:
            if refinement_selection is None:
                raise ValueError(
                    "Executed refinement requires an explicit final selection"
                )
            by_id = {evaluation.candidateId: evaluation for evaluation in evaluations}
            selected = by_id.get(refinement_selection.selectedCandidateId)
            if selected is None:
                raise ValueError("Refinement selection references an unknown candidate")
            if selected.status != "done" or not selected.eligible:
                raise ValueError(
                    "Refinement selection must choose an eligible execution"
                )
            if core_artifact_reference(
                selected.cellSelection
            ) != core_artifact_reference(phase_selected.cellSelection):
                raise ValueError("Refinement selection changed the exact cell axis")
            known_evidence = {
                evidence_id
                for evaluation in evaluations
                for evidence_id in evaluation.evidenceIds
            }
            unknown_evidence = sorted(
                set(refinement_selection.evidenceIds) - known_evidence
            )
            if unknown_evidence:
                raise ValueError(
                    f"Refinement selection cites unknown evidence {unknown_evidence}"
                )
            prefix = f"candidate:{selected.candidateId}:"
            if not any(
                evidence_id.startswith(prefix)
                for evidence_id in refinement_selection.evidenceIds
            ):
                raise ValueError(
                    "Refinement selection must cite its selected candidate"
                )
            graph_partition_dominators = [
                candidate_id
                for candidate_id in selected.metrics.dominatedByCandidateIds
                if candidate_id in by_id
                and _changed_refinement_parameters(
                    selected.parameters,
                    by_id[candidate_id].parameters,
                )
                in {("neighborsK",), ("leidenResolution",)}
            ]
            if graph_partition_dominators:
                require_dominated_candidate_evidence(
                    selected,
                    refinement_selection.evidenceIds,
                    context="The post-grid selection",
                )
            evidence_ids = list(
                dict.fromkeys([*evidence_ids, *refinement_selection.evidenceIds])
            )
            rationale = f"{rationale} {refinement_selection.rationale}"
            stop_reason = "One bounded post-grid refinement was adjudicated."
    elif refinement_selection is not None:
        raise ValueError("Refinement selection requires a refinement result")
    assay_report = ParameterTuningReport(
        status="done",
        fromAssay=evidence.assay,
        cellSelection=selected.cellSelection,
        evaluations=evaluations,
        recommendedCandidateId=selected.candidateId,
        selectedArtifacts=dict(selected.artifacts),
        confidence="medium",
        rationale=rationale,
        evidenceIds=evidence_ids,
        limitations=[],
        stopReason=stop_reason,
        searchPlan=search_plan,
        recommendedByAssay={evidence.assay: selected.candidateId},
        totalCandidates=len(evaluations),
    )
    report = assay_report.model_copy(
        update={"assayReports": {evidence.assay: assay_report}}
    )
    return finalize_parameter_tuning_selection(
        report,
        marker_assay=marker_assay or evidence.assay,
        native_assay=evidence.assay,
    )


__all__ = [
    "CorrectionNeedSelection",
    "execute_parameter_phase",
    "execute_sequential_refinement",
    "ParameterPhaseEvidence",
    "ParameterPhasePlan",
    "ParameterPhaseSelection",
    "prepare_sequential_refinement_dependencies",
    "SequentialAssayTuningEvidence",
    "SequentialRefinementResult",
    "SequentialRefinementSelection",
    "SequentialRnaTuningPlanner",
    "sequential_evidence_to_report",
    "sequential_refinement_selection_candidates",
    "validate_parameter_phase_selection",
    "validate_sequential_refinement_plan",
]
