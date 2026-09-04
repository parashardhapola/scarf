"""Parameter tuning and multimodal integration workflow stages."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ...datastore.datastore import DataStore
from ...utils.logging import logger
from .. import record_io
from ..decision_kernel import DecisionEvidence, DecisionSelection, EvidenceBundle
from ..experimental_context import ExperimentalContextResult
from ..parameter_tuning import (
    ArtifactRecord,
    FinalGraphComparison,
    FinalGraphSelection,
    IntegrationCandidateEvaluation,
    IntegrationMetrics,
    ParameterCandidate,
    ParameterCandidateEvaluation,
    ParameterTuningAgent,
    ParameterTuningAssayInput,
    ParameterTuningReport,
    final_graph_options,
    finalize_parameter_tuning_selection,
    validate_final_graph_selection,
)
from ..persistence import (
    AgentInvocation,
    AgentReportLink,
    AgentReportReference,
    AgentWorkflowRun,
    list_agent_reports,
    load_agent_record,
    load_agent_report,
    save_agent_report,
)
from ..sequential_tuning import (
    CorrectionNeedSelection,
    ParameterPhaseEvidence,
    ParameterPhasePlan,
    ParameterPhaseSelection,
    SequentialAssayTuningEvidence,
    SequentialRnaTuningPlanner,
    execute_parameter_phase,
    sequential_evidence_to_report,
    validate_parameter_phase_selection,
)
from ..rna_decisions import (
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
from ..study_contract import StudyContract
from ..tuning_diagnostics import (
    augment_cluster_evaluations,
    augment_pca_evaluations,
    score_advisory_doublets,
)
from ..types import ArtifactReferenceModel, ExperimentalTuningHandoff
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


def harmony_acceptance_gate(
    native: ParameterCandidateEvaluation | None,
    harmony: ParameterCandidateEvaluation | None,
    *,
    batch_columns: Sequence[str],
    protected_columns: Sequence[str],
    independent_unit_columns: Sequence[str],
    tolerance: float = 0.05,
) -> tuple[bool, list[str]]:
    """Require measured batch improvement without material biological loss."""
    reasons: list[str] = []
    if native is None or harmony is None:
        return False, ["Matched native and Harmony candidates are unavailable."]
    native_parameters = native.parameters.model_dump(
        mode="json",
        exclude={"candidateId", "useHarmony"},
    )
    harmony_parameters = harmony.parameters.model_dump(
        mode="json",
        exclude={"candidateId", "useHarmony"},
    )
    if native_parameters != harmony_parameters:
        reasons.append("Native and Harmony candidate parameters are not matched.")
    batch_deltas: dict[str, float] = {}
    for column in batch_columns:
        native_score = native.metrics.batchMixing.get(column)
        harmony_score = harmony.metrics.batchMixing.get(column)
        if native_score is None or harmony_score is None:
            reasons.append(f"Batch comparison is missing for {column!r}.")
            continue
        batch_deltas[column] = harmony_score - native_score
    if len(batch_deltas) != len(batch_columns):
        reasons.append("Not every approved batch metric was compared.")
    elif not any(delta > tolerance for delta in batch_deltas.values()):
        reasons.append(
            "Harmony did not improve an approved batch metric beyond tolerance."
        )
    if any(delta < -tolerance for delta in batch_deltas.values()):
        reasons.append("Harmony materially worsened an approved batch metric.")

    for column in protected_columns:
        native_scores = native.metrics.biologicalPreservation.get(column)
        harmony_scores = harmony.metrics.biologicalPreservation.get(column)
        if not native_scores or not harmony_scores:
            reasons.append(f"Protected comparison is missing for {column!r}.")
            continue
        missing_metrics = set(native_scores).difference(harmony_scores)
        if missing_metrics:
            reasons.append(
                f"Harmony is missing protected metrics for {column!r}: "
                f"{sorted(missing_metrics)}."
            )
            continue
        shared = set(native_scores).intersection(harmony_scores)
        if not shared:
            reasons.append(f"Protected metrics do not align for {column!r}.")
            continue
        if any(
            harmony_scores[name] < native_scores[name] - tolerance for name in shared
        ):
            reasons.append(
                f"Harmony materially degraded protected evidence for {column!r}."
            )
    if independent_unit_columns:
        if (
            native.metrics.crossUnitSupport is None
            or harmony.metrics.crossUnitSupport is None
        ):
            reasons.append("Cross-unit support comparison is missing.")
        elif (
            harmony.metrics.crossUnitSupport
            < native.metrics.crossUnitSupport - tolerance
        ):
            reasons.append("Harmony materially degraded cross-unit support.")
    if (
        native.metrics.markerCoherence is None
        or harmony.metrics.markerCoherence is None
    ):
        reasons.append("Marker-coherence comparison is missing.")
    elif harmony.metrics.markerCoherence < native.metrics.markerCoherence - tolerance:
        reasons.append("Harmony materially degraded marker coherence.")
    return not reasons, reasons


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
            or handoff.normalized is None
            or handoff.graphFeatures is None
            or handoff.markerFeatures is None
        ):
            raise ValueError("Decision-driven v1 tuning requires normalized RNA")
        if prior is not None and prior.assay != handoff.assay:
            raise ValueError("Persisted sequential evidence belongs to another assay")
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
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing persisted "
                f"{phase_plan.phase} executor evidence"
            )
            return tuple(persisted.evaluations)

        harmony_authorized = (
            study_contract.correctionLicense == "safe"
            and experimental_handoff.batchAction == "evaluateHarmony"
            and bool(experimental_handoff.batchColumns)
        )
        planner = SequentialRnaTuningPlanner(
            workflow_run_id=workflow.workflowRunId,
            assay=handoff.assay,
            n_cells=handoff.nCells,
            n_features=handoff.nFeatures,
            harmony_authorized=harmony_authorized,
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
        pca_plan = planner.pca_prefix_phase()
        raw_pca = phase_evaluations(
            pca_plan,
            lambda: execute_parameter_phase(
                store,
                normalized=normalized,
                plan=pca_plan,
                batch_columns=(
                    experimental_handoff.batchColumns if harmony_authorized else []
                ),
                preservation_columns=experimental_handoff.preservationColumns,
                experimental_handoff=experimental_handoff,
                min_cluster_cells=request_record.config.minClusterCells,
                identity_feature_limit=request_record.config.maxIdentityFeatures,
            ),
        )
        raw_pca = augment_pca_evaluations(
            store,
            raw_pca,
            feature_selection=artifact_model_to_ref(handoff.graphFeatures),
            nominated_families=nominated_families,
            protected_families=protected_families,
            technical_columns=study_contract.technicalBatchColumns,
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
        for evaluation in raw_pca:
            evidence_ids: list[str] = []
            if evaluation.status == "done" and evaluation.eligible:
                eligible_pca_dimensions.append(evaluation.parameters.dimensions)
                technical_id = f"evidence:pca:{evaluation.candidateId}:technical"
                pca_items.append(
                    DecisionEvidence(
                        evidenceId=technical_id,
                        evidenceClass="technical",
                        summary=(
                            f"The exact PCA candidate used "
                            f"{evaluation.effectiveDimensions} dimensions; "
                            f"component variance={evaluation.metrics.componentVariance}; "
                            "maximum nominated-family loading enrichment="
                            f"{evaluation.metrics.loadingFamilyEnrichment}; "
                            "technical PC association="
                            f"{evaluation.metrics.technicalPcaAssociation}; "
                            "protected PC association="
                            f"{evaluation.metrics.protectedPcaAssociation}; "
                            f"QC PC association={evaluation.metrics.qcPcaAssociation}; "
                            f"warnings={evaluation.warnings}."
                        ),
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
                        summary=(
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
        if not (license_payload.license == "safe" and correction_need == "needed"):
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
                    batch_columns=(
                        experimental_handoff.batchColumns
                        if any(
                            candidate.useHarmony
                            for candidate in correction_plan.candidates
                        )
                        else []
                    ),
                    preservation_columns=experimental_handoff.preservationColumns,
                    experimental_handoff=experimental_handoff,
                    min_cluster_cells=request_record.config.minClusterCells,
                    identity_feature_limit=request_record.config.maxIdentityFeatures,
                ),
            )
        )
        correction_evaluations = list(
            augment_cluster_evaluations(
                store,
                correction_evaluations,
                marker_assay=plan.markerAssay,
                marker_features=artifact_model_to_ref(handoff.markerFeatures),
                independent_unit_columns=study_contract.independentUnitColumns,
                technical_columns=study_contract.technicalBatchColumns,
                nominated_families=nominated_families,
                protected_families=protected_families,
            )
        )
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
            batch_columns=study_contract.technicalBatchColumns,
            protected_columns=study_contract.protectedColumns,
            independent_unit_columns=study_contract.independentUnitColumns,
        )
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
        graph_items: list[DecisionEvidence] = []
        graph_evaluations: list[ParameterCandidateEvaluation] = []
        eligible_graph_values: list[int] = []
        graph_evidence_by_k: dict[int, list[str]] = {}
        for evaluation in raw_graph:
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
                graph_evidence_by_k[evaluation.parameters.neighborsK] = list(
                    graph_extra
                )
            else:
                evidence_id = f"evidence:graph:{evaluation.candidateId}:failure"
                graph_items.append(
                    DecisionEvidence(
                        evidenceId=evidence_id,
                        evidenceClass="other",
                        summary=(
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

        doublet_evidence = score_advisory_doublets(
            store,
            selected_graph,
            graph_evaluations,
            assay=handoff.assay,
            feature_selection=artifact_model_to_ref(handoff.graphFeatures),
            capture_column=study_contract.physicalCaptureColumn,
        )
        cluster_plan = planner.clustering_phase(selected_graph.parameters)
        persisted_cluster = prior_phases.get(cluster_plan.phase)
        if persisted_cluster is not None:
            if persisted_cluster.plan != cluster_plan:
                raise ValueError(
                    "Persisted clusteringResolution plan differs from the "
                    "current registered plan"
                )
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing persisted "
                "clusteringResolution executor evidence"
            )
            raw_clusters: Sequence[ParameterCandidateEvaluation] = (
                persisted_cluster.evaluations
            )
        else:
            raw_clusters = execute_parameter_phase(
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
            )
        cluster_evaluations = list(
            augment_cluster_evaluations(
                store,
                raw_clusters,
                marker_assay=plan.markerAssay,
                marker_features=artifact_model_to_ref(handoff.markerFeatures),
                independent_unit_columns=study_contract.independentUnitColumns,
                technical_columns=study_contract.technicalBatchColumns,
                nominated_families=nominated_families,
                protected_families=protected_families,
                doublet_evidence=doublet_evidence,
            )
        )
        cluster_items: list[DecisionEvidence] = []
        scored: list[tuple[float, float, ParameterCandidateEvaluation]] = []
        eligible_cluster_values: list[float] = []
        augmented_clusters: list[ParameterCandidateEvaluation] = []
        cluster_evidence_by_resolution: dict[float, list[str]] = {}
        for evaluation in cluster_evaluations:
            cluster_extra: list[str] = []
            if evaluation.status == "done" and evaluation.eligible:
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
                                f"{evaluation.metrics.minClusterCells}."
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
                            summary=(
                                "The fraction of clusters with marker programs is "
                                f"{evaluation.metrics.markerCoherence}; nominated "
                                "family marker enrichment is "
                                f"{evaluation.metrics.markerFamilyEnrichment}; "
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
                unit_support = evaluation.metrics.crossUnitSupport or 0.0
                technical = max(
                    evaluation.metrics.technicalAssociation.values(),
                    default=0.0,
                )
                score = (
                    geometry
                    + 0.25 * stability
                    + 0.2 * marker
                    + 0.1 * unit_support
                    - 0.1 * technical
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
                        summary=(
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
        return (
            sequential_evidence_to_report(
                state,
                marker_assay=plan.markerAssay,
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
            aliases = {"sex": "sexLinked"}
            allowed_families = {
                "mitochondrial",
                "ribosomal",
                "histone",
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
            eligible = [
                cast(ConditionalGeneFamily, family)
                for family in nominated
                if family not in protected
                and (
                    loading_enrichment.get(family, 0.0) >= 2.0
                    or marker_enrichment.get(family, 0.0) >= 2.0
                )
            ]
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
            definition = require_option_evidence(definition, requirements)
            if eligible:
                review = self._reconsider_rna_decision(
                    store,
                    request_record,
                    definition,
                    bundle,
                    answers,
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
                    else FeaturePolicyExecutorPayload(
                        policy="keepAll",
                        excludedFamilies=[],
                    )
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
                    policy="keepAll",
                    excludedFamilies=[],
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
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            stage_name,
            request_record,
            parents,
        )
        if existing is not None:
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
        resumable_report: ParameterTuningReport | None = None
        prior_sequential: SequentialAssayTuningEvidence | None = None
        if paused is not None and paused.reportReferences:
            loaded = journal.load_stage_report(store, paused, ParameterTuningReport)
            candidate_report = cast(ParameterTuningReport, loaded)
            if paused.outputs.get("sequentialEvidence") is not None:
                prior_sequential = SequentialAssayTuningEvidence.model_validate(
                    paused.outputs["sequentialEvidence"]
                )
            if (
                candidate_report.finalSelection is not None
                and candidate_report.finalSelection.status == "needsInput"
                and candidate_report.assayReports
            ):
                resumable_report = candidate_report
        cell_selection = preprocessed[0].cellSelection if preprocessed else None
        if cell_selection is None or any(
            value.cellSelection != cell_selection for value in preprocessed
        ):
            raise ValueError("Preprocessed assays must share one exact cell selection")
        experimental_handoff = experimental.to_parameter_tuning_handoff().model_copy(
            update={"cellSelection": cell_selection}
        )
        tuning_answer = answers.get("parameter_tuning")
        if isinstance(tuning_answer, Mapping):
            tuning_directions = json.dumps(
                dict(tuning_answer),
                sort_keys=True,
            )
        elif isinstance(tuning_answer, str):
            tuning_directions = tuning_answer.strip()
        else:
            tuning_directions = ""
        objective_direction = (
            "Authoritative study objective: "
            f"{request_record.request.studyObjective.strip()}"
        )
        tuning_directions = "\n".join(
            value for value in (objective_direction, tuning_directions) if value
        )
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
                "resumeFromAttempt": (paused.attemptId if paused is not None else None),
            },
            resume_record=resume_record,
        )
        report = ParameterTuningReport.get_blank()
        actions: list[str] = []
        integration_evaluations: list[IntegrationCandidateEvaluation] = []
        candidate_payload: dict[str, list[dict[str, Any]]] = {}
        paired = list(plan.pairedAssays)
        logger.info(
            f"Workflow {workflow.workflowRunId}: Parameter Tuning started for "
            f"{len(preprocessed)} assay(s), paired={len(paired)}"
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
                integration_evaluations = list(report.integrationEvaluations)
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
                    integration_evaluations,
                    candidate_payload,
                    paired,
                    enrichment_reference,
                    experimental_reference,
                    experimental_handoff,
                    agent,
                    actions,
                    persisted_reference=recovered_reference,
                )
            if len(preprocessed) == 1 and not plan.pairedAssays:
                if study_contract is None:
                    raise ValueError(
                        "Decision-driven RNA tuning requires a StudyContract"
                    )
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
                        candidate.model_dump(mode="json")
                        for phase in sequential_evidence.phases
                        for candidate in phase.plan.candidates
                    ]
                }
                actions.extend(
                    f"adjudicate_{phase.plan.phase}"
                    for phase in sequential_evidence.phases
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
                    [],
                    candidate_payload,
                    [],
                    enrichment_reference,
                    experimental_reference,
                    experimental_handoff,
                    agent,
                    actions,
                    sequential_evidence=sequential_evidence,
                )
            if resumable_report is not None:
                assert paused is not None
                resumed_integration_evaluations = list(
                    resumable_report.integrationEvaluations
                )
                report = resumable_report.model_copy(
                    update={
                        "status": "done",
                        "needsInput": None,
                        "finalSelection": None,
                        "recommendedIntegrationId": None,
                        "finalClusterColumn": None,
                        "finalClusterArtifact": None,
                        "graphAssay": None,
                    }
                )
                report = self.select_final_graph(
                    agent,
                    report,
                    resumed_integration_evaluations,
                    marker_assay=plan.markerAssay,
                    answers=answers,
                )
                logger.info(
                    f"Workflow {workflow.workflowRunId}: resumed final graph "
                    "selection without rerunning candidate evaluation"
                )
                resumed_candidate_payload = {
                    assay: [
                        evaluation.parameters.model_dump(mode="json")
                        for evaluation in assay_report.evaluations
                    ]
                    for assay, assay_report in report.assayReports.items()
                }
                return self.save_parameter_tuning_outcome(
                    store,
                    prefix,
                    workflow,
                    request_record,
                    started,
                    report,
                    plan,
                    preprocessed,
                    resumed_integration_evaluations,
                    resumed_candidate_payload,
                    list(plan.pairedAssays),
                    enrichment_reference,
                    experimental_reference,
                    experimental_handoff,
                    agent,
                    ["reuse_parameter_screen_and_integrations"],
                    prior_tuning_reference=paused.reportReferences[0],
                )
            handoff_by_assay = {value.assay: value for value in preprocessed}
            common_k = None
            if paired:
                common_k = min(
                    21,
                    min(handoff_by_assay[assay].nCells - 1 for assay in paired),
                )
                if common_k < 2:
                    raise ValueError("Paired integration requires at least three cells")
            integration_budget = (
                2 * request_record.config.integrationResolutionCandidates
                if len(paired) >= 2
                else 0
            )
            assay_inputs: list[ParameterTuningAssayInput] = []
            for handoff in preprocessed:
                if handoff.normalized is None:
                    raise ValueError(f"Assay {handoff.assay!r} lacks normalization")
                initial_count = (
                    request_record.config.primaryInitialCandidates
                    if handoff.assay == plan.primaryAssay
                    else request_record.config.secondaryInitialCandidates
                )
                neighbors_k = common_k or min(21, handoff.nCells - 1)
                candidates = self.initial_parameter_candidates(
                    workflow.workflowRunId,
                    handoff,
                    count=initial_count,
                    neighbors_k=neighbors_k,
                    dimension_candidates=(request_record.config.pcaCandidateDimensions),
                    neighbor_candidates=(request_record.config.graphNeighborCandidates),
                    resolution_candidates=(
                        request_record.config.leidenResolutionCandidates
                    ),
                )
                if (
                    experimental_handoff.batchAction == "evaluateHarmony"
                    and request_record.config.maxHarmonyCandidatesPerAssay == 1
                ):
                    baseline = candidates[0]
                    candidates.append(
                        baseline.model_copy(
                            update={
                                "candidateId": f"{baseline.candidateId}_harmony",
                                "useHarmony": True,
                            }
                        )
                    )
                candidate_payload[handoff.assay] = [
                    value.model_dump(mode="json") for value in candidates
                ]
                logger.info(
                    f"Workflow {workflow.workflowRunId}: planned "
                    f"{len(candidates)} native candidate(s) for assay "
                    f"{handoff.assay!r} (harmony="
                    f"{sum(value.useHarmony for value in candidates)})"
                )
                assay_inputs.append(
                    ParameterTuningAssayInput(
                        normalized=artifact_model_to_ref(handoff.normalized),
                        candidates=candidates,
                        batchColumns=(
                            list(experimental_handoff.batchColumns)
                            if experimental_handoff.batchAction == "evaluateHarmony"
                            else []
                        ),
                        preservationColumns=list(
                            experimental_handoff.preservationColumns
                        ),
                        experimentalHandoff=(
                            None
                            if experimental_handoff.batchAction == "evaluateHarmony"
                            else experimental_handoff
                        ),
                        maxCandidates=(
                            len(candidates)
                            + request_record.config.maxRefinedCandidatesPerAssay
                        ),
                        maxRefinedCandidates=(
                            request_record.config.maxRefinedCandidatesPerAssay
                        ),
                        allowHarmonyRefinement=False,
                        minClusterCells=request_record.config.minClusterCells,
                        identityFeatureLimit=request_record.config.maxIdentityFeatures,
                    )
                )
            planned_native = sum(value.maxCandidates for value in assay_inputs)
            if (
                planned_native + integration_budget
                > request_record.config.maxCandidateBranches
            ):
                raise ValueError(
                    "The native and integrated candidate plan exceeds the global "
                    f"branch limit {request_record.config.maxCandidateBranches}"
                )
            logger.info(
                f"Workflow {workflow.workflowRunId}: executing "
                f"{planned_native} native candidate branch(es) with "
                f"{integration_budget} reserved integration branch(es)"
            )
            report = agent.run_batch(
                store,
                assays=assay_inputs,
                primary_assay=plan.primaryAssay,
                max_total_candidates=(
                    request_record.config.maxCandidateBranches - integration_budget
                ),
                selection_directions=tuning_directions,
            )
            logger.info(
                f"Workflow {workflow.workflowRunId}: Parameter Tuning returned "
                f"status={report.status!r}, evaluated={report.totalCandidates}"
            )
            if report.status == "done":
                for assay, assay_report in report.assayReports.items():
                    normalized = handoff_by_assay[assay].normalized
                    assert normalized is not None
                    agent.promote(
                        store,
                        report=assay_report,
                        normalized=artifact_model_to_ref(normalized),
                        identity_feature_limit=request_record.config.maxIdentityFeatures,
                    )
                    actions.append(f"promote_native:{assay}")
                    logger.info(
                        f"Workflow {workflow.workflowRunId}: promoted native "
                        f"candidate {assay_report.recommendedCandidateId!r} for "
                        f"assay {assay!r}"
                    )
                integration_evaluations = self.evaluate_integrations(
                    store,
                    workflow.workflowRunId,
                    plan,
                    report,
                    experimental_handoff,
                    request_record.config,
                    started=started,
                    parent_reports=[
                        journal._report_link(enrichment_reference),
                        journal._report_link(experimental_reference),
                    ],
                    actions=actions,
                )
                logger.info(
                    f"Workflow {workflow.workflowRunId}: evaluated "
                    f"{len(integration_evaluations)} integration candidate(s)"
                )
                report = self.select_final_graph(
                    agent,
                    report,
                    integration_evaluations,
                    marker_assay=plan.markerAssay,
                    answers=answers,
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
                integration_evaluations,
                candidate_payload,
                paired,
                enrichment_reference,
                experimental_reference,
                experimental_handoff,
                agent,
                actions,
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
            for integration_evaluation in integration_evaluations:
                if integration_evaluation.graphArtifact is not None:
                    failure_artifacts[
                        f"{integration_evaluation.integrationId}_graph"
                    ] = ArtifactReferenceModel.model_validate(
                        integration_evaluation.graphArtifact.model_dump()
                    )
                if integration_evaluation.clusterArtifact is not None:
                    failure_artifacts[
                        f"{integration_evaluation.integrationId}_clusters"
                    ] = ArtifactReferenceModel.model_validate(
                        integration_evaluation.clusterArtifact.model_dump()
                    )
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
                    "integrationEvaluations": [
                        value.model_dump(mode="json")
                        for value in integration_evaluations
                    ],
                },
            )
            return outcome, ParameterTuningReport.get_blank()

    def select_final_graph(
        self,
        agent: ParameterTuningAgent,
        report: ParameterTuningReport,
        integration_evaluations: Sequence[IntegrationCandidateEvaluation],
        *,
        marker_assay: str,
        answers: Mapping[str, Any],
    ) -> ParameterTuningReport:
        directed_option = answers.get("finalGraphOptionId")
        if not isinstance(directed_option, str) or not directed_option:
            logger.info(
                f"Selecting final graph from native and "
                f"{len(integration_evaluations)} integration evaluation(s)"
            )
            return agent.select_final(
                report=report,
                integration_evaluations=integration_evaluations,
                marker_assay=marker_assay,
            )
        options = final_graph_options(report, integration_evaluations)
        if directed_option not in options:
            raise ValueError("finalGraphOptionId is not an eligible option")
        logger.info(f"Applying caller-selected final graph {directed_option!r}")
        selected_evidence = list(options[directed_option]["evidenceIds"])
        selection = FinalGraphSelection(
            status="done",
            selectedOptionId=directed_option,
            markerAssay=marker_assay,
            confidence="high",
            rationale="The caller selected this persisted eligible option.",
            evidenceIds=selected_evidence,
            comparisons=[
                FinalGraphComparison(
                    optionId=option_id,
                    summary="The caller preferred the selected eligible option.",
                    evidenceIds=[
                        *selected_evidence,
                        *cast(list[str], option["evidenceIds"]),
                    ],
                )
                for option_id, option in options.items()
                if option_id != directed_option
            ],
        )
        selection = validate_final_graph_selection(
            selection,
            report,
            integration_evaluations=integration_evaluations,
            marker_assay=marker_assay,
        )
        return finalize_parameter_tuning_selection(
            report,
            marker_assay=marker_assay,
            integration_evaluations=integration_evaluations,
            recommended_integration_id=selection.integrationId,
            native_assay=selection.nativeAssay,
            final_selection=selection,
        )

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
        integration_evaluations: Sequence[IntegrationCandidateEvaluation],
        candidate_payload: Mapping[str, list[dict[str, Any]]],
        paired: Sequence[str],
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
        for integration_evaluation in integration_evaluations:
            if integration_evaluation.graphArtifact is not None:
                invocation_artifacts[
                    f"{integration_evaluation.integrationId}_graph"
                ] = ArtifactReferenceModel.model_validate(
                    integration_evaluation.graphArtifact.model_dump()
                )
            if integration_evaluation.clusterArtifact is not None:
                invocation_artifacts[
                    f"{integration_evaluation.integrationId}_clusters"
                ] = ArtifactReferenceModel.model_validate(
                    integration_evaluation.clusterArtifact.model_dump()
                )
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
        if report.recommendedIntegrationId is not None:
            selected_integration = next(
                value
                for value in integration_evaluations
                if value.integrationId == report.recommendedIntegrationId
            )
            if selected_integration.graphArtifact is not None:
                stage_artifacts["final_graph"] = ArtifactReferenceModel.model_validate(
                    selected_integration.graphArtifact.model_dump()
                )
        elif report.graphAssay is not None:
            assay_reports = report.assayReports or {report.fromAssay: report}
            graph_artifact = assay_reports[report.graphAssay].selectedArtifacts[
                "connectivityMap"
            ]
            stage_artifacts["final_graph"] = ArtifactReferenceModel.model_validate(
                graph_artifact.model_dump()
            )
        checkpoint_ids = {
            f"{journal._stage_execution_id(started)}_integration_{method}"
            for method in {evaluation.method for evaluation in integration_evaluations}
        }
        checkpoint_references = sorted(
            (
                reference
                for reference in list_agent_reports(
                    store,
                    started.workflowRunId,
                    agent_name="parameter_tuning",
                )
                if reference.agentRunId in checkpoint_ids
            ),
            key=lambda value: value.agentRunId,
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
                        *[
                            journal._report_link(value)
                            for value in checkpoint_references
                        ],
                    ],
                    inputs={
                        "assays": dict(candidate_payload),
                        "primaryAssay": plan.primaryAssay,
                        "markerAssay": plan.markerAssay,
                        "pairedAssays": list(paired),
                        "cellSelection": (
                            experimental_handoff.cellSelection.model_dump(mode="json")
                            if experimental_handoff.cellSelection is not None
                            else None
                        ),
                        "maxCandidateBranches": request_record.config.maxCandidateBranches,
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
        stage_report_references = [reference, *checkpoint_references]
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
        seen_integrated_graphs: set[tuple[str, str]] = set()
        for integration_evaluation in integration_evaluations:
            integration_graph = integration_evaluation.graphArtifact
            graph_id = (
                integration_graph.artifactId if integration_graph is not None else ""
            )
            graph_key = (integration_evaluation.method, graph_id)
            if graph_id and graph_key not in seen_integrated_graphs:
                assert integration_graph is not None
                seen_integrated_graphs.add(graph_key)
                source_key = (
                    "connectivityMap"
                    if integration_evaluation.method == "snn"
                    else "neighbors"
                )
                source_artifacts = []
                for assay in integration_evaluation.assays:
                    assay_report = report.assayReports[assay]
                    selected = next(
                        value
                        for value in assay_report.evaluations
                        if value.candidateId == assay_report.recommendedCandidateId
                    )
                    source_artifacts.append(
                        selected.artifacts[source_key].model_dump(mode="json")
                    )
                operations.append(
                    {
                        "operation": "integrate_assays",
                        "method": integration_evaluation.method,
                        "sources": source_artifacts,
                        "invalidateCache": True,
                        "l2Normalize": True,
                        "artifact": integration_graph.model_dump(mode="json"),
                    }
                )
            if integration_graph is None:
                continue
            operations.append(
                {
                    "operation": "run_leiden_clustering",
                    "integrationId": integration_evaluation.integrationId,
                    "status": integration_evaluation.status,
                    "resolution": integration_evaluation.resolution,
                    "graph": integration_graph.model_dump(mode="json"),
                    "cellSelection": (
                        integration_evaluation.cellSelection.model_dump(mode="json")
                        if integration_evaluation.cellSelection is not None
                        else None
                    ),
                    "backend": "igraph",
                    "symmetricGraph": False,
                    "graphUpperOnly": False,
                    "randomSeed": 4444,
                    "invalidateCache": False,
                    "artifact": (
                        integration_evaluation.clusterArtifact.model_dump(mode="json")
                        if integration_evaluation.clusterArtifact is not None
                        else None
                    ),
                }
            )
        if report.status == "needsInput":
            needs_input = report.needsInput
            assert needs_input is not None
            if (
                sequential_evidence is not None
                and sequential_evidence.pendingDecisionId is None
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
            f"status={outcome.status!r}, candidates={report.totalCandidates}, "
            f"integrations={len(integration_evaluations)}"
        )
        if outcome.status == "failed":
            journal.finalize_failed(store, workflow, outcome.error or "tuning failed")
        return outcome, report

    def initial_parameter_candidates(
        self,
        workflow_run_id: str,
        handoff: PreprocessedAssayHandoff,
        *,
        count: int,
        neighbors_k: int,
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
    ) -> list[ParameterCandidate]:
        max_dimensions = min(handoff.nCells, handoff.nFeatures) - 1
        if neighbors_k < 2 or neighbors_k >= handoff.nCells:
            raise ValueError(
                f"Assay {handoff.assay!r} has no rank-valid graph candidate"
            )
        if handoff.reductionMethod == "identity":
            if handoff.nFeatures < 2:
                raise ValueError(
                    f"Assay {handoff.assay!r} has no rank-valid graph candidate"
                )
            dimensions = handoff.nFeatures
            dimension_values = [dimensions]
        elif max_dimensions < 2:
            raise ValueError(
                f"Assay {handoff.assay!r} has no rank-valid graph candidate"
            )
        elif handoff.reductionMethod == "lsi":
            dimensions = min(50, max_dimensions)
            dimension_values = [
                dimensions,
                min(30, max_dimensions),
                min(70, max_dimensions),
            ]
        else:
            dimension_values = [
                min(value, max_dimensions)
                for value in dimension_candidates
                if value >= 2
            ]
            if not dimension_values:
                dimension_values = [min(20, max_dimensions)]
            dimensions = min(20, max_dimensions)
            if dimensions not in dimension_values:
                dimension_values.append(dimensions)
        unique_dimensions = list(
            dict.fromkeys(value for value in dimension_values if value >= 2)
        )
        baseline_dimensions = (
            min(20, max_dimensions)
            if handoff.reductionMethod == "pca"
            else unique_dimensions[0]
        )
        unique_dimensions = [
            baseline_dimensions,
            *(value for value in unique_dimensions if value != baseline_dimensions),
        ]
        specifications: list[tuple[int, float, int]] = [
            (value, 1.0, neighbors_k) for value in unique_dimensions
        ]
        for candidate_k in neighbor_candidates:
            effective_k = min(candidate_k, handoff.nCells - 1)
            specification = (baseline_dimensions, 1.0, effective_k)
            if effective_k >= 2 and specification not in specifications:
                specifications.append(specification)
        for resolution in resolution_candidates:
            if len(specifications) >= count:
                break
            specification = (baseline_dimensions, float(resolution), neighbors_k)
            if specification not in specifications:
                specifications.append(specification)
        token = workflow_run_id[:10]
        assay_token = journal._safe_label(handoff.assay).lower()
        if len(assay_token) > 32:
            digest = hashlib.blake2b(
                handoff.assay.encode("utf-8"), digest_size=4
            ).hexdigest()
            assay_token = f"{assay_token[:23]}_{digest}"
        if handoff.reductionMethod == "identity":
            candidates = [
                ParameterCandidate(
                    candidateId=f"w_{token}_{assay_token}_0",
                    reductionMethod="identity",
                    dimensions=handoff.nFeatures,
                    leidenResolution=1.0,
                    neighborsK=neighbors_k,
                )
            ]
            if count > 1 and max_dimensions >= 2:
                candidates.append(
                    ParameterCandidate(
                        candidateId=f"w_{token}_{assay_token}_1",
                        reductionMethod="pca",
                        dimensions=min(21, max_dimensions),
                        leidenResolution=1.0,
                        neighborsK=neighbors_k,
                    )
                )
            for resolution in resolution_candidates:
                if len(candidates) >= count:
                    break
                index = len(candidates)
                candidates.append(
                    ParameterCandidate(
                        candidateId=f"w_{token}_{assay_token}_{index}",
                        reductionMethod="identity",
                        dimensions=handoff.nFeatures,
                        leidenResolution=resolution,
                        neighborsK=neighbors_k,
                    )
                )
            return candidates
        return [
            ParameterCandidate(
                candidateId=f"w_{token}_{assay_token}_{index}",
                reductionMethod=cast(Any, handoff.reductionMethod),
                dimensions=dimension,
                leidenResolution=resolution,
                neighborsK=candidate_k,
            )
            for index, (dimension, resolution, candidate_k) in enumerate(
                specifications[:count]
            )
        ]

    def load_integration_checkpoint(
        self,
        store: DataStore,
        started: WorkflowStageAttempt,
        method: Literal["snn", "wnn"],
    ) -> tuple[list[IntegrationCandidateEvaluation], AgentReportReference] | None:
        checkpoint_id = f"{journal._stage_execution_id(started)}_integration_{method}"
        matches = [
            reference
            for reference in list_agent_reports(
                store,
                started.workflowRunId,
                agent_name="parameter_tuning",
            )
            if reference.agentRunId == checkpoint_id
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("An integration checkpoint has multiple reports")
        reference = matches[0]
        record = load_agent_record(store, reference)
        if (
            record.invocation.inputs.get("orchestrationExecutionId") != checkpoint_id
            or record.invocation.inputs.get("stageExecutionId")
            != journal._stage_execution_id(started)
            or record.invocation.inputs.get("method") != method
        ):
            raise ValueError("Integration checkpoint identity is stale")
        for artifact in record.invocation.artifacts.values():
            store.load_artifact(artifact_model_to_ref(artifact))
        report = load_agent_report(store, reference)
        if not isinstance(report, ParameterTuningReport):
            raise TypeError("Integration checkpoint is not a Parameter Tuning report")
        evaluations = list(report.integrationEvaluations)
        if not evaluations or any(value.method != method for value in evaluations):
            raise ValueError("Integration checkpoint contains the wrong method")
        logger.info(
            f"Workflow {started.workflowRunId}: recovered {method.upper()} "
            f"checkpoint with {len(evaluations)} evaluation(s)"
        )
        return evaluations, reference

    def save_integration_checkpoint(
        self,
        store: DataStore,
        started: WorkflowStageAttempt,
        report: ParameterTuningReport,
        method: Literal["snn", "wnn"],
        evaluations: Sequence[IntegrationCandidateEvaluation],
        parent_reports: Sequence[AgentReportLink],
    ) -> AgentReportReference:
        checkpoint_id = f"{journal._stage_execution_id(started)}_integration_{method}"
        checkpoint_report = report.model_copy(
            update={
                "integrationEvaluations": list(evaluations),
                "recommendedIntegrationId": None,
                "finalClusterColumn": None,
                "finalClusterArtifact": None,
                "finalSelection": None,
            }
        )
        artifacts: dict[str, ArtifactReferenceModel] = {}
        cell_selection = next(
            (
                evaluation.cellSelection
                for evaluation in evaluations
                if evaluation.cellSelection is not None
            ),
            None,
        )
        if cell_selection is not None:
            artifacts["cellSelection"] = cell_selection
        for evaluation in evaluations:
            if evaluation.graphArtifact is not None:
                artifacts[f"{evaluation.integrationId}_graph"] = (
                    ArtifactReferenceModel.model_validate(
                        evaluation.graphArtifact.model_dump()
                    )
                )
            if evaluation.clusterArtifact is not None:
                artifacts[f"{evaluation.integrationId}_clusters"] = (
                    ArtifactReferenceModel.model_validate(
                        evaluation.clusterArtifact.model_dump()
                    )
                )
        invocation = AgentInvocation(
            agentName="parameter_tuning",
            parentReports=list(parent_reports),
            inputs={
                "orchestrationExecutionId": checkpoint_id,
                "stageExecutionId": journal._stage_execution_id(started),
                "method": method,
                "cellSelection": (
                    cell_selection.model_dump(mode="json")
                    if cell_selection is not None
                    else None
                ),
            },
            artifacts=artifacts,
        )
        try:
            reference = save_agent_report(
                store,
                started.workflowRunId,
                checkpoint_report,
                invocation=invocation,
                agent_run_id=checkpoint_id,
            )
            logger.info(
                f"Workflow {started.workflowRunId}: persisted {method.upper()} "
                f"checkpoint with {len(evaluations)} evaluation(s)"
            )
            return reference
        except FileExistsError:
            recovered = self.load_integration_checkpoint(store, started, method)
            if recovered is None:
                raise
            return recovered[1]

    def evaluate_integrations(
        self,
        store: DataStore,
        workflow_run_id: str,
        plan: AutomatedPreprocessingPlan,
        report: ParameterTuningReport,
        experimental_handoff: ExperimentalTuningHandoff,
        config: AutomatedWorkflowConfig,
        *,
        started: WorkflowStageAttempt | None = None,
        parent_reports: Sequence[AgentReportLink] = (),
        actions: list[str] | None = None,
    ) -> list[IntegrationCandidateEvaluation]:
        assays = list(plan.pairedAssays)
        if len(assays) < 2:
            logger.info("Skipping SNN/WNN evaluation: fewer than two paired assays")
            return []
        selected_k = {
            next(
                evaluation.parameters.neighborsK
                for evaluation in assay_report.evaluations
                if evaluation.candidateId == assay_report.recommendedCandidateId
            )
            for assay, assay_report in report.assayReports.items()
            if assay in assays
        }
        if len(selected_k) != 1:
            raise ValueError("SNN and WNN require one common selected neighborsK")
        primary_report = report.assayReports[plan.primaryAssay]
        primary_evaluation = next(
            value
            for value in primary_report.evaluations
            if value.candidateId == primary_report.recommendedCandidateId
        )
        center = primary_evaluation.parameters.leidenResolution
        count = config.integrationResolutionCandidates
        multipliers = [1.0] if count == 1 else np.linspace(0.5, 1.5, count).tolist()
        resolutions = list(
            dict.fromkeys(max(0.05, round(center * value, 6)) for value in multipliers)
        )
        logger.info(
            f"Evaluating SNN and WNN across {len(resolutions)} resolution(s) "
            f"for {len(assays)} paired assay(s)"
        )
        if report.cellSelection is None:
            raise ValueError("Parameter tuning report lacks an exact cell selection")
        cell_selection = report.cellSelection
        native_labels: dict[str, np.ndarray[Any, Any]] = {}
        for assay, assay_report in report.assayReports.items():
            if assay not in assays:
                continue
            selected = next(
                value
                for value in assay_report.evaluations
                if value.candidateId == assay_report.recommendedCandidateId
            )
            cluster_model = ArtifactReferenceModel.model_validate(
                selected.artifacts["clusters"].model_dump()
            )
            cluster_group = store.load_artifact(artifact_model_to_ref(cluster_model))
            cluster_values = cast(Any, cluster_group["values"])
            native_labels[assay] = np.asarray(cluster_values[:])
        token = workflow_run_id[:12]
        evaluations: list[IntegrationCandidateEvaluation] = []
        integration_methods: tuple[Literal["snn", "wnn"], ...] = ("snn", "wnn")
        for method in integration_methods:
            evaluations.extend(
                self.evaluate_integration_method(
                    store,
                    method,
                    token,
                    assays,
                    resolutions,
                    native_labels,
                    report,
                    experimental_handoff,
                    config,
                    cell_selection,
                    started=started,
                    parent_reports=parent_reports,
                    actions=actions,
                )
            )
        return evaluations

    def evaluate_integration_method(
        self,
        store: DataStore,
        method: Literal["snn", "wnn"],
        token: str,
        assays: list[str],
        resolutions: Sequence[float],
        native_labels: Mapping[str, np.ndarray[Any, Any]],
        report: ParameterTuningReport,
        experimental_handoff: ExperimentalTuningHandoff,
        config: AutomatedWorkflowConfig,
        cell_selection: ArtifactReferenceModel,
        *,
        started: WorkflowStageAttempt | None,
        parent_reports: Sequence[AgentReportLink],
        actions: list[str] | None,
    ) -> list[IntegrationCandidateEvaluation]:
        if started is not None:
            recovered = self.load_integration_checkpoint(store, started, method)
            if recovered is not None:
                if actions is not None:
                    actions.append(f"recover_integration_checkpoint:{method}")
                return recovered[0]
        logger.info(
            f"Evaluating {method.upper()} integration across "
            f"{len(resolutions)} resolution(s)"
        )
        evaluations: list[IntegrationCandidateEvaluation] = []
        source_key = "connectivityMap" if method == "snn" else "neighbors"
        sources = []
        for assay in assays:
            assay_report = report.assayReports[assay]
            selected = next(
                value
                for value in assay_report.evaluations
                if value.candidateId == assay_report.recommendedCandidateId
            )
            source_model = ArtifactReferenceModel.model_validate(
                selected.artifacts[source_key].model_dump()
            )
            sources.append(artifact_model_to_ref(source_model))
        try:
            graph_ref = store.integrate_assays(
                sources,
                method=method,
                invalidate_cache=True,
                l2_normalize=True,
            )
            weights_valid: bool | None = None
            if method == "wnn":
                graph_group = store.load_artifact(graph_ref)
                stored_weights = cast(Any, graph_group["modality_weights"])
                weights = np.asarray(stored_weights[:], dtype=float)
                weights_valid = bool(
                    weights.shape
                    == (len(next(iter(native_labels.values()))), len(assays))
                    and np.all(np.isfinite(weights))
                    and np.all(weights >= 0)
                    and np.allclose(weights.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)
                )
        except Exception as exc:
            logger.warning(
                f"{method.upper()} graph construction failed "
                f"({type(exc).__name__}); persisting failed evaluations"
            )
            for index, resolution in enumerate(resolutions):
                evaluations.append(
                    IntegrationCandidateEvaluation(
                        integrationId=f"{method}_{token}_{index}",
                        method=cast(Any, method),
                        assays=assays,
                        status="failed",
                        cellSelection=cell_selection,
                        resolution=resolution,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            if started is not None:
                self.save_integration_checkpoint(
                    store,
                    started,
                    report,
                    method,
                    evaluations,
                    parent_reports,
                )
                if actions is not None:
                    actions.append(f"checkpoint_integration:{method}")
            return evaluations
        for index, resolution in enumerate(resolutions):
            integration_id = f"{method}_{token}_{index}"
            warnings: list[str] = []
            evidence_ids = [f"integration:{integration_id}:clusters"]
            try:
                cluster_ref = store.run_leiden_clustering(
                    graph_ref,
                    resolution=resolution,
                    backend="igraph",
                    symmetric_graph=False,
                    graph_upper_only=False,
                    random_seed=4444,
                    invalidate_cache=False,
                )
                cluster_group = store.load_artifact(cluster_ref)
                cluster_values = cast(Any, cluster_group["values"])
                values = np.asarray(cluster_values[:])
                _labels, counts = np.unique(values, return_counts=True)
                metrics = IntegrationMetrics(
                    nClusters=int(len(counts)),
                    minClusterCells=int(counts.min()),
                    minClusterFraction=float(counts.min() / len(values)),
                    modalityWeightsValid=weights_valid,
                )
                for assay, native in native_labels.items():
                    metrics.adjustedRandByAssay[assay] = float(
                        adjusted_rand_score(native, values)
                    )
                    metrics.normalizedMutualInformationByAssay[assay] = float(
                        normalized_mutual_info_score(native, values)
                    )
                    evidence_ids.extend(
                        [
                            f"integration:{integration_id}:ari:{assay}",
                            f"integration:{integration_id}:nmi:{assay}",
                        ]
                    )
                for column in experimental_handoff.preservationColumns:
                    try:
                        value = float(
                            store.metric_graph_connectivity(
                                column,
                                graph_ref,
                            )
                        )
                        if np.isfinite(value):
                            metrics.biologicalConnectivity[column] = value
                            evidence_ids.append(
                                f"integration:{integration_id}:graphConnectivity:{column}"
                            )
                    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                        warnings.append(
                            f"Graph connectivity for {column!r} unavailable: {exc}"
                        )
                if method == "wnn":
                    evidence_ids.append(f"integration:{integration_id}:modalityWeights")
                reasons: list[str] = []
                missing_connectivity = sorted(
                    set(experimental_handoff.preservationColumns)
                    - set(metrics.biologicalConnectivity)
                )
                if missing_connectivity:
                    reasons.append(
                        "trusted-label connectivity is unavailable for "
                        + ", ".join(missing_connectivity)
                    )
                if metrics.nClusters is None or metrics.nClusters < 2:
                    reasons.append("fewer than two clusters")
                if (
                    metrics.minClusterCells is None
                    or metrics.minClusterCells < config.minClusterCells
                ):
                    reasons.append("smallest cluster is below the configured minimum")
                if method == "wnn" and weights_valid is not True:
                    reasons.append("WNN modality weights are invalid")
                evaluations.append(
                    IntegrationCandidateEvaluation(
                        integrationId=integration_id,
                        method=cast(Any, method),
                        assays=assays,
                        status="done",
                        eligible=not reasons,
                        cellSelection=cell_selection,
                        resolution=resolution,
                        graphArtifact=ArtifactRecord.from_ref(graph_ref),
                        clusterArtifact=ArtifactRecord.from_ref(cluster_ref),
                        metrics=metrics,
                        evidenceIds=evidence_ids,
                        eligibilityReasons=reasons,
                        warnings=warnings,
                    )
                )
            except Exception as exc:
                evaluations.append(
                    IntegrationCandidateEvaluation(
                        integrationId=integration_id,
                        method=cast(Any, method),
                        assays=assays,
                        status="failed",
                        cellSelection=cell_selection,
                        resolution=resolution,
                        graphArtifact=ArtifactRecord.from_ref(graph_ref),
                        evidenceIds=evidence_ids,
                        warnings=warnings,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        if started is not None:
            self.save_integration_checkpoint(
                store,
                started,
                report,
                method,
                evaluations,
                parent_reports,
            )
            if actions is not None:
                actions.append(f"checkpoint_integration:{method}")
        eligible_count = sum(
            value.status == "done" and value.eligible for value in evaluations
        )
        failed_count = sum(value.status == "failed" for value in evaluations)
        logger.info(
            f"Completed {method.upper()} integration evaluation: "
            f"eligible={eligible_count}, failed={failed_count}, "
            f"total={len(evaluations)}"
        )
        return evaluations
