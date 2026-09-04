"""Preprocessing planning and execution stages."""

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ...assay import RNAassay
from ...datastore.datastore import DataStore
from ...datastore.summary import AssaySummary
from ...metadata.selection import NamedCellArtifact
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_mask
from ...storage.types import as_zarr_array
from ...utils.logging import logger
from .. import record_io
from ..data_enrichment import (
    AssayFeatureInspection,
    DataEnrichmentReport,
    FeatureSelectionPolicy,
)
from ..decision_kernel import DecisionEvidence, DecisionSelection, EvidenceBundle
from ..experimental_context import (
    CellQcPlan,
    CellQcProfileEvidence,
    ExperimentalContextResult,
)
from ..hvg_diagnostics import run_hvg_diagnostic_artifacts
from ..persistence import AgentWorkflowRun
from ..qc_execution import execute_registered_cell_qc
from ..rna_decisions import (
    CellQualityExecutorPayload,
    CellQualityProfile,
    FeaturePolicyExecutorPayload,
    HvgExecutorPayload,
    HvgRankingExecutorPayload,
    QcGroupingExecutorPayload,
    build_cell_quality_decision,
    build_feature_policy_decision,
    build_hvg_count_decision,
    build_hvg_ranking_decision,
    build_qc_grouping_decision,
    require_option_evidence,
)
from ..study_contract import StudyContract
from ..types import ArtifactReferenceModel
from . import journal
from .decisions import DecisionStagesMixin
from .models import (
    AssayPreprocessingPlan,
    AutomatedPreprocessingPlan,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    PreprocessedAssayHandoff,
    ReductionMethod,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    WorkflowStageName,
    artifact_model_to_ref,
)


class _DecisionNeedsInput(RuntimeError):
    def __init__(
        self,
        question: WorkflowQuestion,
        snapshot_sha256: str,
    ) -> None:
        super().__init__("A registered RNA decision requires human input")
        self.question = question
        self.snapshotSha256 = snapshot_sha256


def apply_feature_policy_to_plan(
    plan: AutomatedPreprocessingPlan,
    payload: FeaturePolicyExecutorPayload,
) -> AutomatedPreprocessingPlan:
    assays: list[AssayPreprocessingPlan] = []
    for assay in plan.assays:
        if assay.assay != plan.primaryAssay:
            assays.append(assay)
            continue
        parameters = {
            **assay.featureParameters,
            "excludeFamilies": list(payload.excludedFamilies),
        }
        assays.append(assay.model_copy(update={"featureParameters": parameters}))
    updated = plan.model_copy(update={"assays": assays, "planChecksum": ""})
    checksum = hashlib.sha256(
        record_io.canonical_json_bytes(
            updated.model_dump(mode="json", exclude={"planChecksum"})
        )
    ).hexdigest()
    return updated.model_copy(update={"planChecksum": checksum})


class PreprocessingStagesMixin(DecisionStagesMixin):
    """Stages that plan and execute modality-specific preprocessing."""

    @staticmethod
    def _cell_qc_stage_artifacts(
        plan: CellQcPlan,
    ) -> dict[str, ArtifactReferenceModel]:
        artifacts = {
            f"cellQcMetric:{source.name}": source.artifact
            for source in plan.artifactMetrics
        }
        if len(artifacts) != len(plan.artifactMetrics):
            raise ValueError("Cell-QC artifact metric names must be unique")
        if plan.sampleArtifact is not None:
            if plan.sampleArtifact.name in {
                source.name for source in plan.artifactMetrics
            }:
                raise ValueError("Cell-QC sample artifact name collides with a metric")
            key = f"cellQcSample:{plan.sampleArtifact.name}"
            artifacts[key] = plan.sampleArtifact.artifact
        return artifacts

    @classmethod
    def _cell_qc_candidate_artifacts(
        cls,
        profiles: Sequence[CellQcProfileEvidence],
    ) -> dict[str, ArtifactReferenceModel]:
        artifacts: dict[str, ArtifactReferenceModel] = {}
        for profile in profiles:
            plan = CellQcPlan(
                action=profile.action,
                registeredProfile=profile.registeredProfile,
                profileId=profile.profileId,
                driverAssay=profile.driverAssay,
                driverAssayType=profile.driverAssayType,
                sampleColumn=profile.sampleColumn,
                sampleArtifact=profile.sampleArtifact,
                attributes=profile.attributes,
                artifactMetrics=profile.artifactMetrics,
                evidenceIds=[profile.evidenceId],
            )
            for key, artifact in cls._cell_qc_stage_artifacts(plan).items():
                existing = artifacts.get(key)
                if existing is not None and existing != artifact:
                    raise ValueError(
                        f"Cell-QC candidate artifact key {key!r} is ambiguous"
                    )
                artifacts[key] = artifact
        return artifacts

    @staticmethod
    def _decision_evidence_bundle(
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
    def _profile_is_safe(profile: CellQcProfileEvidence) -> bool:
        if profile.unsafeRetentionGroups:
            return False
        if (
            profile.registeredProfile
            in {
                "captureMad5",
                "captureMad3Sensitivity",
            }
            and profile.failedCaptureCandidates
        ):
            return False
        if profile.registeredProfile == "pooledReferenceMad5" and set(
            profile.failedCaptureCandidates
        ).intersection(profile.parameters.get("pooledReferenceCaptures", [])):
            return False
        return True

    @staticmethod
    def _profile_evidence(profile: CellQcProfileEvidence) -> DecisionEvidence:
        return DecisionEvidence(
            evidenceId=profile.evidenceId,
            evidenceClass="qualityControl",
            summary=(
                f"{profile.registeredProfile} retains "
                f"{profile.retainedCells}/{profile.activeCells} active cells; "
                f"retention by capture={profile.sampleRetainedCells}; "
                f"retention by design column={profile.retainedCellsByColumn}; "
                f"failed capture candidates={profile.failedCaptureCandidates}; "
                f"unsafe retention groups={profile.unsafeRetentionGroups}."
            ),
            artifactReferences=[
                *[source.artifact for source in profile.artifactMetrics],
                *(
                    [profile.sampleArtifact.artifact]
                    if profile.sampleArtifact is not None
                    else []
                ),
            ],
        )

    def _resolve_qc_grouping_decision(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        experimental: ExperimentalContextResult,
        study_contract: StudyContract,
        answers: Mapping[str, Any],
    ) -> tuple[QcGroupingExecutorPayload, str]:
        profiles = [
            profile
            for profile in experimental.qcProfiles
            if profile.registeredProfile is not None
        ]
        if not profiles:
            raise ValueError("RNA decision workflow requires registered QC evidence")
        safe_profiles = {
            profile.registeredProfile: profile
            for profile in profiles
            if profile.registeredProfile is not None and self._profile_is_safe(profile)
        }
        capture_eligible = bool(
            study_contract.physicalCaptureColumn is not None
            and "captureMad5" in safe_profiles
        )
        pooled_eligible = bool(
            capture_eligible and "pooledReferenceMad5" in safe_profiles
        )
        design_id = "evidence:qcGrouping:studyContract"
        evidence = [
            DecisionEvidence(
                evidenceId=design_id,
                evidenceClass="design",
                summary=(
                    "The validated physical capture is "
                    f"{study_contract.physicalCaptureColumn!r}; independent units="
                    f"{study_contract.independentUnitColumns}; conditions="
                    f"{study_contract.conditionColumns}."
                ),
            )
        ]
        mode_profile: dict[str, CellQcProfileEvidence] = {}
        global_profile = safe_profiles.get("globalMad5") or safe_profiles.get(
            "retainWithFlags"
        )
        if global_profile is None:
            raise ValueError("No safe global or retain-only QC profile is available")
        mode_profile["qcGrouping:global"] = global_profile
        evidence.append(self._profile_evidence(global_profile))
        if capture_eligible:
            capture_profile = safe_profiles["captureMad5"]
            mode_profile["qcGrouping:physicalCapture"] = capture_profile
            evidence.append(self._profile_evidence(capture_profile))
        if pooled_eligible:
            pooled_profile = safe_profiles["pooledReferenceMad5"]
            mode_profile["qcGrouping:pooledReference"] = pooled_profile
            evidence.append(self._profile_evidence(pooled_profile))
        bundle = self._decision_evidence_bundle("qcGrouping", evidence)
        definition = build_qc_grouping_decision(
            evidence_bundle_id=bundle.bundleId,
            physical_capture_eligible=capture_eligible,
            pooled_reference_eligible=pooled_eligible,
        )
        definition = require_option_evidence(
            definition,
            {
                option_id: [design_id, profile.evidenceId]
                for option_id, profile in mode_profile.items()
            },
        )
        selectable_ids = [
            option.optionId
            for option in definition.spec.options
            if option.status != "defer"
        ]
        rule_selection = (
            DecisionSelection(
                selectedOptionId=selectable_ids[0],
                evidenceIds=list(
                    definition.spec.option_by_id()[
                        selectable_ids[0]
                    ].requiredEvidenceIds
                ),
                rationale=(
                    "Use the only grouping mode licensed by the validated design "
                    "and retention evidence."
                ),
            )
            if len(selectable_ids) == 1
            else None
        )
        resolution = self._resolve_rna_decision(
            store,
            request_record,
            definition,
            bundle,
            answers,
            rule_selection=rule_selection,
        )
        if resolution.compiled is None:
            raise _DecisionNeedsInput(
                self._pending_decision_question(resolution, definition),
                resolution.snapshotSha256,
            )
        payload = resolution.compiled.executorPayload
        if not isinstance(payload, QcGroupingExecutorPayload):
            raise TypeError("QC-grouping decision compiled an unexpected payload")
        return payload, resolution.snapshotSha256

    def _resolve_cell_quality_decision(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        experimental: ExperimentalContextResult,
        grouping: QcGroupingExecutorPayload,
        answers: Mapping[str, Any],
    ) -> tuple[CellQualityExecutorPayload, CellQcPlan, str]:
        all_profiles = [
            profile
            for profile in experimental.qcProfiles
            if profile.registeredProfile is not None
        ]
        allowed_by_grouping: dict[str, set[CellQualityProfile]] = {
            "global": {"retainWithFlags", "globalMad5"},
            "physicalCapture": {"retainWithFlags", "captureMad5"},
            "pooledReference": {"retainWithFlags", "pooledReferenceMad5"},
        }
        allowed = allowed_by_grouping[grouping.groupingMode]
        profiles = [
            profile
            for profile in all_profiles
            if profile.registeredProfile in allowed and self._profile_is_safe(profile)
        ]
        if not profiles:
            raise ValueError("QC grouping has no safe registered profile")
        evidence = [self._profile_evidence(profile) for profile in profiles]
        if grouping.groupingMode == "physicalCapture":
            sensitivity = next(
                (
                    profile
                    for profile in all_profiles
                    if profile.registeredProfile == "captureMad3Sensitivity"
                ),
                None,
            )
            if sensitivity is not None:
                evidence.append(self._profile_evidence(sensitivity))
        bundle = self._decision_evidence_bundle("cellQuality", evidence)
        available_profiles = [
            profile.registeredProfile
            for profile in profiles
            if profile.registeredProfile is not None
        ]
        definition = build_cell_quality_decision(
            evidence_bundle_id=bundle.bundleId,
            available_profiles=available_profiles,
        )
        definition = require_option_evidence(
            definition,
            {
                f"cellQuality:{profile.registeredProfile}": [profile.evidenceId]
                for profile in profiles
            },
        )
        resolution = self._resolve_rna_decision(
            store,
            request_record,
            definition,
            bundle,
            answers,
        )
        if resolution.compiled is None:
            raise _DecisionNeedsInput(
                self._pending_decision_question(resolution, definition),
                resolution.snapshotSha256,
            )
        payload = resolution.compiled.executorPayload
        if not isinstance(payload, CellQualityExecutorPayload):
            raise TypeError("Cell-quality decision compiled an unexpected payload")
        selected = next(
            (
                profile
                for profile in profiles
                if profile.registeredProfile == payload.profile
            ),
            None,
        )
        if selected is None or resolution.record is None:
            raise ValueError("Audited cell-quality option lacks its exact profile")
        plan = CellQcPlan(
            action=selected.action,
            registeredProfile=selected.registeredProfile,
            profileId=selected.profileId,
            driverAssay=selected.driverAssay,
            driverAssayType=selected.driverAssayType,
            sampleColumn=selected.sampleColumn,
            sampleArtifact=selected.sampleArtifact,
            attributes=selected.attributes,
            artifactMetrics=selected.artifactMetrics,
            rationale=resolution.record.rationale,
            evidenceIds=list(resolution.record.evidenceIds),
        )
        return payload, plan, resolution.snapshotSha256

    def _resolve_feature_policy_decision(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        plan: AutomatedPreprocessingPlan,
        enrichment: DataEnrichmentReport,
        answers: Mapping[str, Any],
    ) -> tuple[FeaturePolicyExecutorPayload, str]:
        policy = next(
            (
                value
                for value in enrichment.policies
                if value.assay == plan.primaryAssay
            ),
            None,
        )
        nominations = list(policy.excludeFamilies) if policy is not None else []
        protected = list(policy.protectFamilies) if policy is not None else []
        evidence_id = f"evidence:featurePolicy:{plan.primaryAssay}"
        bundle = self._decision_evidence_bundle(
            "featurePolicy",
            [
                DecisionEvidence(
                    evidenceId=evidence_id,
                    evidenceClass="technical",
                    summary=(
                        f"Data Enrichment nominated {sorted(nominations)} and "
                        f"protected {sorted(protected)}. No representation-dominance "
                        "evidence exists before the native PCA diagnostic."
                    ),
                )
            ],
        )
        definition = build_feature_policy_decision(
            evidence_bundle_id=bundle.bundleId,
            proposed_exclusion_families=[],
            dominant_families=[],
            protected_families=[],
        )
        definition = require_option_evidence(
            definition,
            {"featurePolicy:keepAll": [evidence_id]},
        )
        resolution = self._resolve_rna_decision(
            store,
            request_record,
            definition,
            bundle,
            answers,
            rule_selection=DecisionSelection(
                selectedOptionId="featurePolicy:keepAll",
                evidenceIds=[evidence_id],
                rationale=(
                    "Keep the conditional families until native representation "
                    "evidence demonstrates technical dominance."
                ),
            ),
        )
        if resolution.compiled is None:
            raise RuntimeError("A rule-owned feature decision cannot be pending")
        payload = resolution.compiled.executorPayload
        if not isinstance(payload, FeaturePolicyExecutorPayload):
            raise TypeError("Feature-policy decision compiled an unexpected payload")
        return payload, resolution.snapshotSha256

    @staticmethod
    def _apply_feature_policy_to_plan(
        plan: AutomatedPreprocessingPlan,
        payload: FeaturePolicyExecutorPayload,
    ) -> AutomatedPreprocessingPlan:
        return apply_feature_policy_to_plan(plan, payload)

    @staticmethod
    def _plan_with_selected_hvg_counts(
        plan: AutomatedPreprocessingPlan,
        handoffs: Sequence[PreprocessedAssayHandoff],
    ) -> AutomatedPreprocessingPlan:
        selected_counts = {handoff.assay: handoff.nFeatures for handoff in handoffs}
        assays = [
            assay.model_copy(
                update={
                    "featureParameters": {
                        **assay.featureParameters,
                        "topN": selected_counts[assay.assay],
                    }
                }
            )
            if assay.featureMethod == "hvg" and assay.assay in selected_counts
            else assay
            for assay in plan.assays
        ]
        updated = plan.model_copy(update={"assays": assays, "planChecksum": ""})
        checksum = hashlib.sha256(
            record_io.canonical_json_bytes(
                updated.model_dump(mode="json", exclude={"planChecksum"})
            )
        ).hexdigest()
        return updated.model_copy(update={"planChecksum": checksum})

    def preprocessing_plan_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        enrichment: DataEnrichmentReport,
        experimental: ExperimentalContextResult,
        ingest_outcome: WorkflowStageAttempt,
        study_contract: StudyContract,
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[WorkflowStageAttempt, AutomatedPreprocessingPlan]:
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "preprocessing_plan",
            request_record,
            parents,
        )
        if existing is not None:
            logger.info(
                f"Workflow {workflow.workflowRunId}: reusing preprocessing plan"
            )
            return existing, AutomatedPreprocessingPlan.model_validate(
                existing.outputs["preprocessingPlan"]
            )
        if experimental.cellSelection is None:
            raise ValueError("Experimental Context lacks an exact cell selection")
        cell_qc_artifacts = self._cell_qc_candidate_artifacts(experimental.qcProfiles)
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "preprocessing_plan",
            request_record,
            parents,
            inputs={
                "cellSelection": experimental.cellSelection.model_dump(mode="json"),
                "decisionPolicy": "evidenceBoundedAutomaticExecution",
                "studyContract": study_contract.model_dump(mode="json"),
            },
            resume_record=resume_record,
        )
        try:
            grouping_payload, grouping_decision_snapshot = (
                self._resolve_qc_grouping_decision(
                    store,
                    request_record,
                    experimental,
                    study_contract,
                    answers,
                )
            )
            cell_payload, cell_qc, cell_decision_snapshot = (
                self._resolve_cell_quality_decision(
                    store,
                    request_record,
                    experimental,
                    grouping_payload,
                    answers,
                )
            )
            plan = self.build_preprocessing_plan(
                store,
                request_record,
                enrichment,
                experimental,
                ingest_outcome,
                cell_qc,
            )
            graph_plans = [value for value in plan.assays if value.graphEligible]
            if (
                len(graph_plans) != 1
                or graph_plans[0].assayType != "RNA"
                or plan.pairedAssays
                or plan.primaryAssay != graph_plans[0].assay
                or plan.markerAssay != graph_plans[0].assay
            ):
                raise ValueError(
                    "The automated decision workflow accepts one unpaired RNA assay"
                )
            plan = plan.model_copy(update={"cellQualityPayload": cell_payload})
            feature_payload, feature_decision_snapshot = (
                self._resolve_feature_policy_decision(
                    store,
                    request_record,
                    plan,
                    enrichment,
                    answers,
                )
            )
            plan = self._apply_feature_policy_to_plan(plan, feature_payload)
            route_summary = ", ".join(
                f"{value.assay}:{value.featureMethod}/{value.reductionMethod}"
                for value in plan.assays
            )
            logger.info(
                f"Workflow {workflow.workflowRunId}: preprocessing plan built "
                f"(primary={plan.primaryAssay!r}, marker={plan.markerAssay!r}, "
                f"routes=[{route_summary}])"
            )
        except _DecisionNeedsInput as pending:
            outcome = journal._complete_attempt(
                started,
                status="needsInput",
                artifacts={
                    "cellSelection": experimental.cellSelection,
                    **cell_qc_artifacts,
                },
                outputs={"decisionSnapshotSha256": pending.snapshotSha256},
                needs_input=WorkflowNeedsInput(questions=[pending.question]),
                notes=["A registered filtering decision requires input."],
            )
            journal._save_outcome(store.zw, prefix, outcome)
            return outcome, AutomatedPreprocessingPlan.get_blank()
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts={
                    "cellSelection": experimental.cellSelection,
                    **cell_qc_artifacts,
                },
            )
            return outcome, AutomatedPreprocessingPlan.get_blank()
        logger.info(
            f"Workflow {workflow.workflowRunId}: executing the evidence-bounded "
            "preprocessing plan"
        )
        outcome = journal._complete_attempt(
            started,
            status="done",
            artifacts={
                "cellSelection": experimental.cellSelection,
                **cell_qc_artifacts,
            },
            outputs={
                "preprocessingPlan": plan.model_dump(mode="json"),
                "qcGroupingDecisionSnapshot": grouping_decision_snapshot,
                "cellQualityDecisionSnapshot": cell_decision_snapshot,
                "featurePolicyDecisionSnapshot": feature_decision_snapshot,
            },
            actions=[
                "audit_qc_grouping_decision",
                "audit_cell_quality_decision",
                "audit_feature_policy_decision",
                "accept_evidence_bounded_preprocessing_plan",
            ],
        )
        journal._save_outcome(store.zw, prefix, outcome)
        return outcome, plan

    def build_preprocessing_plan(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        enrichment: DataEnrichmentReport,
        experimental: ExperimentalContextResult,
        ingest_outcome: WorkflowStageAttempt,
        cell_qc: CellQcPlan,
    ) -> AutomatedPreprocessingPlan:
        request = request_record.request
        store_summary = store.summary()
        summaries = {value.name: value for value in store_summary.assays}
        policies = {value.assay: value for value in enrichment.policies}
        inspections = {value.assay: value for value in enrichment.inspections}
        selected_names = request.analysisAssays or list(store.assay_names)
        assay_plans: list[AssayPreprocessingPlan] = []
        graph_assays: list[str] = []
        limitations: list[str] = list(enrichment.limitations)
        selected_qc_profile = next(
            (
                value
                for value in experimental.qcProfiles
                if value.profileId == cell_qc.profileId
            ),
            None,
        )
        projected_cells = (
            selected_qc_profile.retainedCells
            if selected_qc_profile is not None and selected_qc_profile.retainedCells > 0
            else store_summary.active_cells
        )
        effective_min_cells = min(20, max(1, projected_cells // 10))
        for assay_name in selected_names:
            summary = summaries[assay_name]
            policy = policies.get(assay_name)
            modality = (
                policy.assayModality
                if policy is not None
                else (
                    summary.assay_type
                    if summary.assay_type in {"RNA", "ATAC", "ADT", "HTO"}
                    else "unsupported"
                )
            )
            plan = self.build_assay_preprocessing_plan(
                store,
                request_record,
                assay_name,
                summary,
                policy,
                inspections.get(assay_name),
                modality,
                effective_min_cells,
            )
            if plan.graphEligible:
                graph_assays.append(assay_name)
            if modality == "unsupported":
                limitations.extend(plan.limitations)
            assay_plans.append(plan)
        if not graph_assays:
            raise ValueError("No supported graph-bearing assay remains")
        if len(graph_assays) > request_record.config.maxGraphAssays:
            raise ValueError(
                "Too many graph-bearing assays; provide analysisAssays to select at "
                f"most {request_record.config.maxGraphAssays}"
            )
        modality_counts: dict[str, int] = {}
        for name in graph_assays:
            modality_counts[summaries[name].assay_type] = (
                modality_counts.get(summaries[name].assay_type, 0) + 1
            )
        duplicate_modalities = sorted(
            name for name, count in modality_counts.items() if count > 1
        )
        if duplicate_modalities and not request.analysisAssays:
            raise ValueError(
                "Multiple same-kind biological assays require explicit "
                f"analysisAssays selection: {duplicate_modalities}"
            )
        primary = request.primaryAssay
        if primary is not None and primary not in graph_assays:
            raise ValueError("primaryAssay must name a graph-bearing selected assay")
        if primary is None:
            primary = next(
                (
                    name
                    for modality in ("RNA", "ADT", "ATAC")
                    for name in graph_assays
                    if summaries[name].assay_type == modality
                ),
                graph_assays[0],
            )
        if request.markerAssay is not None:
            if request.markerAssay not in graph_assays:
                raise ValueError("markerAssay must name a graph-bearing selected assay")
            marker_assay = request.markerAssay
        else:
            marker_assay = next(
                (
                    name
                    for modality in ("RNA", "ADT", "ATAC")
                    for name in graph_assays
                    if summaries[name].assay_type == modality
                ),
                primary,
            )
        if request.pairedAssays:
            paired = list(request.pairedAssays)
            unknown_paired = sorted(set(paired) - set(graph_assays))
            if unknown_paired:
                raise ValueError(
                    f"pairedAssays contains non-graph assays: {unknown_paired}"
                )
            if primary not in paired:
                raise ValueError("pairedAssays must include the primary assay")
        elif (
            len(graph_assays) > 1
            and ingest_outcome.outputs.get("pairingProvenance")
            == "singleSourceSharedCellAxis"
        ):
            paired = list(graph_assays)
        else:
            paired = []
            if len(graph_assays) > 1:
                limitations.append(
                    "Multimodal integration skipped because pairing provenance "
                    "was not supplied"
                )
        final_plan = AutomatedPreprocessingPlan(
            primaryAssay=primary,
            markerAssay=marker_assay,
            cellSelection=experimental.cellSelection,
            cellQc=cell_qc,
            assays=assay_plans,
            pairedAssays=paired,
            limitations=list(dict.fromkeys(limitations)),
        )
        checksum = hashlib.sha256(
            record_io.canonical_json_bytes(
                final_plan.model_dump(mode="json", exclude={"planChecksum"})
            )
        ).hexdigest()
        return final_plan.model_copy(update={"planChecksum": checksum})

    def build_assay_preprocessing_plan(
        self,
        store: DataStore,
        request_record: OrchestrationRequestRecord,
        assay_name: str,
        summary: AssaySummary,
        policy: FeatureSelectionPolicy | None,
        inspection: AssayFeatureInspection | None,
        modality: str,
        effective_min_cells: int,
    ) -> AssayPreprocessingPlan:
        excluded: list[str] = []
        evidence_ids: list[str] = []
        if policy is not None:
            excluded = list(
                dict.fromkeys(
                    [
                        *policy.excludeFeatures,
                        *policy.artificialFeatures,
                        *(
                            reference.featureId
                            for reference in policy.exactControlFeatures
                        ),
                        *(
                            reference.featureName
                            for reference in policy.exactControlFeatures
                        ),
                    ]
                )
            )
            evidence_ids = list(policy.evidenceIds)
        if modality == "RNA":
            graph_eligible = summary.total_features >= 3
            proposed_families = (
                list(policy.excludeFamilies) if policy is not None else []
            )
            return AssayPreprocessingPlan(
                assay=assay_name,
                assayType=summary.assay_type,
                role="graph" if graph_eligible else "unsupported",
                graphEligible=graph_eligible,
                markerEligible=graph_eligible,
                featureMethod="hvg" if graph_eligible else "none",
                reductionMethod="pca" if graph_eligible else "none",
                featureParameters={
                    "topN": min(2000, summary.total_features),
                    "minCells": effective_min_cells,
                    "excludeFamilies": [],
                    "proposedExcludeFamilies": proposed_families,
                    "protectFamilies": (
                        list(policy.protectFamilies) if policy is not None else []
                    ),
                },
                normalizationParameters={
                    "logTransform": True,
                    "renormalizeSubset": True,
                },
                reductionParameters={"dimensions": min(50, summary.total_features - 1)},
                exactExcludedFeatures=(
                    list(policy.artificialFeatures) if policy is not None else []
                ),
                evidenceIds=evidence_ids,
                limitations=(
                    []
                    if graph_eligible
                    else ["RNA requires at least three features for PCA"]
                ),
            )
        if modality == "ATAC":
            graph_eligible = summary.total_features >= 3
            return AssayPreprocessingPlan(
                assay=assay_name,
                assayType=summary.assay_type,
                role="graph" if graph_eligible else "unsupported",
                graphEligible=graph_eligible,
                markerEligible=graph_eligible,
                featureMethod="prevalentPeaks" if graph_eligible else "none",
                reductionMethod="lsi" if graph_eligible else "none",
                featureParameters={"topN": min(25000, summary.total_features)},
                normalizationParameters={
                    "logTransform": False,
                    "renormalizeSubset": False,
                },
                reductionParameters={"dimensions": 50, "skipFirst": True},
                evidenceIds=evidence_ids,
                limitations=list(
                    dict.fromkeys(
                        [
                            *(
                                []
                                if graph_eligible
                                else [
                                    "ATAC requires at least three peak features for LSI"
                                ]
                            ),
                            *(
                                [
                                    "ATAC feature coordinates are not uniformly "
                                    "valid chrom:start-end intervals; the genome "
                                    "build remains unknown"
                                ]
                                if policy is not None
                                and policy.peakCoordinateStatus
                                in {"partial", "invalid"}
                                else []
                            ),
                        ]
                    )
                ),
            )
        if modality == "ADT":
            assay = store.get_assay(assay_name)
            feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
            feature_names = np.asarray(assay.feats.fetch_all("names")).astype(str)
            excluded = (
                list(
                    dict.fromkeys(
                        value
                        for reference in policy.exactControlFeatures
                        for value in (reference.featureId, reference.featureName)
                        if value
                    )
                )
                if policy is not None
                else []
            )
            excluded_values = {value for value in excluded if value}
            panel_mask = ~np.isin(feature_ids, list(excluded_values))
            panel_mask &= ~np.isin(feature_names, list(excluded_values))
            selected_count = int(panel_mask.sum())
            graph_eligible = selected_count >= 2
            reduction = (
                "identity"
                if graph_eligible
                and selected_count <= request_record.config.maxIdentityFeatures
                else ("pca" if graph_eligible else "none")
            )
            return AssayPreprocessingPlan(
                assay=assay_name,
                assayType=summary.assay_type,
                role="graph" if graph_eligible else "unsupported",
                graphEligible=graph_eligible,
                markerEligible=graph_eligible,
                featureMethod="panel" if graph_eligible else "none",
                reductionMethod=cast(ReductionMethod, reduction),
                normalizationParameters={
                    "logTransform": False,
                    "renormalizeSubset": False,
                },
                reductionParameters={
                    "dimensions": (
                        selected_count
                        if reduction == "identity"
                        else min(15, max(2, selected_count - 1))
                    )
                },
                exactExcludedFeatures=excluded,
                evidenceIds=evidence_ids,
                limitations=(
                    [
                        "ADT control inventory was truncated; only exact observed "
                        "control features were excluded"
                    ]
                    if inspection is not None and inspection.modalityEvidence.truncated
                    else []
                ),
            )
        if modality == "HTO":
            return AssayPreprocessingPlan(
                assay=assay_name,
                assayType=summary.assay_type,
                role="hto",
                graphEligible=False,
                markerEligible=False,
                featureMethod="none",
                reductionMethod="none",
                evidenceIds=evidence_ids,
            )
        message = f"Unsupported assay {assay_name!r} ({summary.assay_type})"
        return AssayPreprocessingPlan(
            assay=assay_name,
            assayType=summary.assay_type,
            role="unsupported",
            limitations=[message],
        )

    def preprocessing_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        experimental: ExperimentalContextResult,
        study_contract: StudyContract,
        answers: Mapping[str, Any],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
        stage_name: WorkflowStageName = "preprocessing",
    ) -> tuple[
        WorkflowStageAttempt,
        list[PreprocessedAssayHandoff],
        AutomatedPreprocessingPlan,
    ]:
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
                f"Workflow {workflow.workflowRunId}: reusing preprocessing artifacts"
            )
            return (
                existing,
                [
                    PreprocessedAssayHandoff.model_validate(value)
                    for value in existing.outputs["assays"]
                ],
                AutomatedPreprocessingPlan.model_validate(
                    existing.outputs["resolvedPreprocessingPlan"]
                ),
            )
        if plan.cellSelection is None:
            raise ValueError("Preprocessing plan lacks an exact cell selection")
        if experimental.cellSelection != plan.cellSelection:
            raise ValueError(
                "Preprocessing plan and Experimental Context selections differ"
            )
        input_cell_selection = artifact_model_to_ref(plan.cellSelection)
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            stage_name,
            request_record,
            parents,
            inputs={
                "preprocessingPlan": plan.model_dump(mode="json"),
                "cellSelection": plan.cellSelection.model_dump(mode="json"),
            },
            resume_record=resume_record,
        )
        actions: list[str] = []
        operations: list[dict[str, Any]] = []
        artifacts: dict[str, ArtifactReferenceModel] = {
            "inputCellSelection": plan.cellSelection,
            **self._cell_qc_stage_artifacts(plan.cellQc),
        }
        try:
            if plan.cellQualityPayload is None:
                raise ValueError(
                    "Decision-driven preprocessing requires an audited "
                    "cell-quality payload"
                )
            cell_selection = self.apply_cell_qc(
                store,
                experimental,
                input_cell_selection,
                actions,
                operations,
                selected_plan=plan.cellQc,
                decision_payload=plan.cellQualityPayload,
            )
            cell_selection_model = ArtifactReferenceModel.from_artifact_ref(
                cell_selection
            )
            artifacts["cellSelection"] = cell_selection_model
            if operations:
                diagnostic_flags = operations[-1].get("diagnosticFlags")
                if diagnostic_flags is not None:
                    artifacts["cellQcDiagnosticFlags"] = (
                        ArtifactReferenceModel.model_validate(diagnostic_flags)
                    )
            active_cells = int(
                read_stored_selection_mask(
                    store.zw,
                    cell_selection,
                    kind="cell_selection",
                    scope="datastore",
                    assay=None,
                    table_path="cellData",
                ).sum()
            )
            selected_profile = next(
                (
                    profile
                    for profile in experimental.qcProfiles
                    if profile.profileId == plan.cellQc.profileId
                ),
                None,
            )
            if selected_profile is None:
                raise ValueError("Selected cell-QC profile is unavailable")
            if active_cells != selected_profile.retainedCells:
                raise ValueError(
                    "Executed cell-QC retention differs from the selected profile"
                )
            logger.info(
                f"Workflow {workflow.workflowRunId}: preprocessing retained "
                f"{active_cells} active cell(s)"
            )
            if active_cells < 3:
                raise ValueError("Preprocessing requires at least three active cells")
            handoffs: list[PreprocessedAssayHandoff] = []
            for assay_plan in plan.assays:
                if not assay_plan.graphEligible:
                    continue
                logger.info(
                    f"Workflow {workflow.workflowRunId}: preprocessing assay "
                    f"{assay_plan.assay!r} via {assay_plan.featureMethod}/"
                    f"{assay_plan.reductionMethod}"
                )
                handoffs.append(
                    self.preprocess_assay(
                        store,
                        assay_plan,
                        cell_selection=cell_selection,
                        cell_selection_model=cell_selection_model,
                        active_cells=active_cells,
                        request_record=request_record,
                        study_contract=study_contract,
                        answers=answers,
                        actions=actions,
                        operations=operations,
                        artifacts=artifacts,
                    )
                )
            resolved_plan = self._plan_with_selected_hvg_counts(
                plan,
                handoffs,
            )
            outcome = journal._complete_attempt(
                started,
                status="done",
                artifacts={
                    name: value
                    for name, value in artifacts.items()
                    if value is not None
                },
                outputs={
                    "assays": [value.model_dump(mode="json") for value in handoffs],
                    "cellSelection": cell_selection_model.model_dump(mode="json"),
                    "resolvedPreprocessingPlan": resolved_plan.model_dump(mode="json"),
                    "operations": operations,
                },
                actions=actions,
            )
            journal._save_outcome(store.zw, prefix, outcome)
            logger.info(
                f"Workflow {workflow.workflowRunId}: preprocessing produced "
                f"{len(handoffs)} graph-ready assay handoff(s)"
            )
            return outcome, handoffs, resolved_plan
        except _DecisionNeedsInput as pending:
            outcome = journal._complete_attempt(
                started,
                status="needsInput",
                artifacts={
                    name: value
                    for name, value in artifacts.items()
                    if value is not None
                },
                outputs={
                    "operations": operations,
                    "decisionSnapshotSha256": pending.snapshotSha256,
                },
                needs_input=WorkflowNeedsInput(questions=[pending.question]),
                actions=actions,
                notes=["A registered RNA preprocessing decision requires input."],
            )
            journal._save_outcome(store.zw, prefix, outcome)
            return outcome, [], plan
        except Exception as exc:
            outcome = journal.finish_exception(
                store,
                prefix,
                workflow,
                started,
                exc,
                artifacts=artifacts,
                actions=actions,
                outputs={"operations": operations},
            )
            return outcome, [], plan

    def reuse_feature_policy_preprocessing_stage(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        parents: Sequence[WorkflowStageLink],
        plan: AutomatedPreprocessingPlan,
        baseline_outcome: WorkflowStageAttempt,
        baseline_handoffs: Sequence[PreprocessedAssayHandoff],
        *,
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> tuple[
        WorkflowStageAttempt,
        list[PreprocessedAssayHandoff],
        AutomatedPreprocessingPlan,
    ]:
        """Record deterministic reuse when feature review keeps the baseline."""
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "feature_policy_preprocessing",
            request_record,
            parents,
        )
        if existing is not None:
            return (
                existing,
                [
                    PreprocessedAssayHandoff.model_validate(value)
                    for value in existing.outputs["assays"]
                ],
                AutomatedPreprocessingPlan.model_validate(
                    existing.outputs["resolvedPreprocessingPlan"]
                ),
            )
        baseline_plan = AutomatedPreprocessingPlan.model_validate(
            baseline_outcome.outputs["resolvedPreprocessingPlan"]
        )
        if baseline_plan != plan:
            raise ValueError(
                "A retained feature policy must reuse the exact baseline plan"
            )
        started = journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "feature_policy_preprocessing",
            request_record,
            parents,
            inputs={
                "baselineAttemptId": baseline_outcome.attemptId,
                "preprocessingPlan": plan.model_dump(mode="json"),
            },
            resume_record=resume_record,
        )
        outcome = journal._complete_attempt(
            started,
            status="done",
            artifacts=dict(baseline_outcome.artifacts),
            outputs={
                "assays": [
                    value.model_dump(mode="json") for value in baseline_handoffs
                ],
                "cellSelection": baseline_outcome.outputs["cellSelection"],
                "resolvedPreprocessingPlan": plan.model_dump(mode="json"),
                "operations": [
                    {
                        "operation": "reuse_baseline_preprocessing",
                        "attemptId": baseline_outcome.attemptId,
                    }
                ],
            },
            actions=["reuse_baseline_preprocessing"],
        )
        journal._save_outcome(store.zw, prefix, outcome)
        return outcome, list(baseline_handoffs), plan

    def preprocess_assay(
        self,
        store: DataStore,
        assay_plan: AssayPreprocessingPlan,
        *,
        cell_selection: ArtifactRef,
        cell_selection_model: ArtifactReferenceModel,
        active_cells: int,
        request_record: OrchestrationRequestRecord | None = None,
        study_contract: StudyContract | None = None,
        answers: Mapping[str, Any] | None = None,
        actions: list[str],
        operations: list[dict[str, Any]],
        artifacts: dict[str, ArtifactReferenceModel],
    ) -> PreprocessedAssayHandoff:
        assay = store.get_assay(assay_plan.assay)
        min_cells = int(assay_plan.featureParameters.get("minCells", 1))
        marker_features: ArtifactRef
        if assay_plan.featureMethod == "hvg":
            if request_record is None or study_contract is None:
                raise ValueError(
                    "RNA HVG preprocessing requires request and study contracts"
                )
            if not isinstance(assay, RNAassay):
                raise TypeError("The RNA HVG route requires an RNAassay")
            detected = store.select_detected_features(
                cell_selection,
                from_assay=assay_plan.assay,
                min_cells=min_cells,
                invalidate_cache=False,
            )
            eligible_features = self.exclude_exact_features(
                store,
                assay_plan,
                detected,
            )
            marker_features = self.exclude_exact_features(
                store,
                assay_plan,
                detected,
                include_families=False,
            )
            technical_columns = [
                value
                for value in (
                    study_contract.physicalCaptureColumn,
                    *study_contract.technicalBatchColumns,
                )
                if value is not None and value in store.cells.columns
            ]
            technical_column = technical_columns[0] if technical_columns else None
            diagnostics = run_hvg_diagnostic_artifacts(
                store.zw,
                assay,
                cell_selection=cell_selection,
                eligible_features=eligible_features,
                all_features=store.select_all_features(from_assay=assay_plan.assay),
                technical_group_column=technical_column,
                min_group_cells=request_record.config.minClusterCells,
                min_cells=min_cells,
                n_bins=200,
                lowess_frac=0.1,
                invalidate_cache=False,
                candidate_targets=request_record.config.hvgCandidateCounts,
            )
            if not diagnostics:
                raise ValueError("HVG diagnostics produced no registered ranking")
            ranking_evidence: list[DecisionEvidence] = []
            ranking_evidence_ids: dict[str, str] = {}
            for candidate_ranking in diagnostics:
                mode = candidate_ranking.ranking_mode
                diagnostic_model = ArtifactReferenceModel.from_artifact_ref(
                    candidate_ranking.diagnostic
                )
                artifacts[f"{assay_plan.assay}_hvg_{mode}_diagnostic"] = (
                    diagnostic_model
                )
                for candidate in candidate_ranking.candidates:
                    artifacts[
                        f"{assay_plan.assay}_hvg_{mode}_candidate_{candidate.top_n}"
                    ] = ArtifactReferenceModel.from_artifact_ref(candidate.features)
                evidence_id = (
                    f"evidence:hvgRanking:{candidate_ranking.diagnostic.artifact_id}"
                )
                ranking_evidence_ids[mode] = evidence_id
                ranking_group = store.load_artifact(candidate_ranking.diagnostic)
                ranking_values = np.asarray(
                    as_zarr_array(
                        ranking_group["ranking"],
                        name="ranking",
                    )[:],
                    dtype=np.int64,
                )
                recurrence_values = np.asarray(
                    as_zarr_array(
                        ranking_group["recurrence"],
                        name="recurrence",
                    )[:],
                    dtype=np.int32,
                )
                within_group_ranks = np.asarray(
                    as_zarr_array(
                        ranking_group["mean_within_group_rank"],
                        name="mean_within_group_rank",
                    )[:],
                    dtype=np.float64,
                )
                broad_count = max(
                    candidate.top_n for candidate in candidate_ranking.candidates
                )
                broad_indices = ranking_values[:broad_count]
                recurrence_summary = ""
                if candidate_ranking.valid_groups:
                    selected_recurrence = recurrence_values[broad_indices]
                    selected_ranks = within_group_ranks[broad_indices]
                    finite_ranks = selected_ranks[np.isfinite(selected_ranks)]
                    median_rank = (
                        f"{float(np.median(finite_ranks)):.3f}"
                        if finite_ranks.size
                        else "unavailable"
                    )
                    recurrence_summary = (
                        f" In the broad {broad_count}-gene candidate, mean technical-"
                        "group coverage is "
                        f"{float(selected_recurrence.mean()) / len(candidate_ranking.valid_groups):.1%}, "
                        f"{float((selected_recurrence >= 2).mean()):.1%} recur in at "
                        "least two groups, and the median normalized within-group "
                        f"rank is {median_rank}."
                    )
                if mode == "batchAware":
                    summary = (
                        "The technical-group ranking uses recurrence and within-group "
                        f"rank across {len(candidate_ranking.valid_groups)} valid "
                        f"groups; excluded groups={candidate_ranking.excluded_groups}."
                        f"{recurrence_summary}"
                    )
                else:
                    summary = (
                        "The global ranking orders every eligible gene by corrected "
                        "variability across the exact filtered cell selection."
                        f"{recurrence_summary}"
                    )
                ranking_evidence.append(
                    DecisionEvidence(
                        evidenceId=evidence_id,
                        evidenceClass="technical",
                        summary=summary,
                        artifactReferences=[diagnostic_model],
                    )
                )
            ranking_bundle = self._decision_evidence_bundle(
                "hvgRanking",
                ranking_evidence,
            )
            ranking_definition = build_hvg_ranking_decision(
                evidence_bundle_id=ranking_bundle.bundleId,
                batch_aware_eligible=any(
                    value.ranking_mode == "batchAware" for value in diagnostics
                ),
            )
            ranking_definition = require_option_evidence(
                ranking_definition,
                {
                    option.optionId: [ranking_evidence_ids[option.payload.rankingMode]]
                    for option in ranking_definition.executorOptions
                    if isinstance(option.payload, HvgRankingExecutorPayload)
                },
            )
            ranking_options = [
                option
                for option in ranking_definition.executorOptions
                if isinstance(option.payload, HvgRankingExecutorPayload)
            ]
            ranking_rule_selection = (
                DecisionSelection(
                    selectedOptionId=ranking_options[0].optionId,
                    evidenceIds=list(
                        ranking_definition.spec.option_by_id()[
                            ranking_options[0].optionId
                        ].requiredEvidenceIds
                    ),
                    rationale=(
                        "Use the only variability ranking licensed by the available "
                        "technical groups."
                    ),
                )
                if len(ranking_options) == 1
                else None
            )
            ranking_resolution = self._resolve_rna_decision(
                store,
                request_record,
                ranking_definition,
                ranking_bundle,
                answers or {},
                rule_selection=ranking_rule_selection,
            )
            if ranking_resolution.compiled is None:
                raise _DecisionNeedsInput(
                    self._pending_decision_question(
                        ranking_resolution,
                        ranking_definition,
                    ),
                    ranking_resolution.snapshotSha256,
                )
            ranking_payload = ranking_resolution.compiled.executorPayload
            if not isinstance(ranking_payload, HvgRankingExecutorPayload):
                raise TypeError("HVG-ranking decision compiled an unexpected payload")
            diagnostic = next(
                (
                    value
                    for value in diagnostics
                    if value.ranking_mode == ranking_payload.rankingMode
                ),
                None,
            )
            if diagnostic is None:
                raise ValueError("Selected HVG ranking has no exact diagnostic")
            diagnostic_model = ArtifactReferenceModel.from_artifact_ref(
                diagnostic.diagnostic
            )
            actions.extend(
                [
                    f"diagnose_hvg_candidates:{assay_plan.assay}",
                    f"audit_hvg_ranking:{assay_plan.assay}",
                    f"select_marker_features:{assay_plan.assay}",
                ]
            )
            artifacts[f"{assay_plan.assay}_hvg_diagnostic"] = diagnostic_model
            for candidate in diagnostic.candidates:
                artifacts[f"{assay_plan.assay}_hvg_candidate_{candidate.top_n}"] = (
                    ArtifactReferenceModel.from_artifact_ref(candidate.features)
                )
            diagnostic_group = store.load_artifact(diagnostic.diagnostic)
            ranking = np.asarray(
                as_zarr_array(diagnostic_group["ranking"], name="ranking")[:],
                dtype=np.int64,
            )
            corrected_variance = np.asarray(
                as_zarr_array(
                    diagnostic_group["global_corrected_variance"],
                    name="global_corrected_variance",
                )[:],
                dtype=np.float64,
            )
            eligible = np.asarray(
                as_zarr_array(diagnostic_group["eligible"], name="eligible")[:],
                dtype=bool,
            )
            recurrence = np.asarray(
                as_zarr_array(
                    diagnostic_group["recurrence"],
                    name="recurrence",
                )[:],
                dtype=np.int32,
            )
            eligible_variance = float(corrected_variance[eligible].sum())
            candidate_evidence: list[DecisionEvidence] = []
            evidence_ids_by_count: dict[int, str] = {}
            for candidate in diagnostic.candidates:
                selected_indices = ranking[: candidate.top_n]
                variance_fraction = (
                    float(corrected_variance[selected_indices].sum())
                    / eligible_variance
                    if eligible_variance > 0
                    else 0.0
                )
                evidence_id = (
                    f"evidence:hvg:{diagnostic.diagnostic.artifact_id}:"
                    f"top{candidate.top_n}"
                )
                evidence_ids_by_count[candidate.top_n] = evidence_id
                summary = (
                    f"The {candidate.top_n}-gene {diagnostic.ranking_mode} candidate "
                    "captures "
                    f"{variance_fraction:.1%} of corrected variance across "
                    f"{diagnostic.eligible_feature_count} eligible genes."
                )
                if diagnostic.valid_groups:
                    replicated = recurrence[selected_indices] >= max(
                        2,
                        (len(diagnostic.valid_groups) + 1) // 2,
                    )
                    summary += (
                        f" {float(replicated.mean()):.1%} of selected genes recur "
                        "across the registered technical-group rankings."
                    )
                candidate_evidence.append(
                    DecisionEvidence(
                        evidenceId=evidence_id,
                        evidenceClass="technical",
                        summary=summary,
                        artifactReferences=[
                            diagnostic_model,
                            ArtifactReferenceModel.from_artifact_ref(
                                candidate.features
                            ),
                        ],
                    )
                )
            bundle = self._decision_evidence_bundle(
                "hvgCount",
                candidate_evidence,
            )
            definition = build_hvg_count_decision(
                evidence_bundle_id=bundle.bundleId,
                eligible_feature_count=diagnostic.eligible_feature_count,
                ranking_mode=diagnostic.ranking_mode,
                valid_technical_groups=len(diagnostic.valid_groups),
                candidate_counts=[
                    candidate.top_n for candidate in diagnostic.candidates
                ],
            )
            definition = require_option_evidence(
                definition,
                {
                    option.optionId: [evidence_ids_by_count[option.payload.topN]]
                    for option in definition.executorOptions
                    if isinstance(option.payload, HvgExecutorPayload)
                },
            )
            resolution = self._resolve_rna_decision(
                store,
                request_record,
                definition,
                bundle,
                answers or {},
            )
            if resolution.compiled is None:
                raise _DecisionNeedsInput(
                    self._pending_decision_question(resolution, definition),
                    resolution.snapshotSha256,
                )
            hvg_payload = resolution.compiled.executorPayload
            if not isinstance(hvg_payload, HvgExecutorPayload):
                raise TypeError("HVG decision compiled an unexpected payload")
            selected_candidate = next(
                (
                    candidate
                    for candidate in diagnostic.candidates
                    if candidate.top_n == hvg_payload.topN
                ),
                None,
            )
            if selected_candidate is None:
                raise ValueError(
                    "Selected HVG count has no exact persisted candidate artifact"
                )
            graph_features = selected_candidate.features
            actions.append(f"audit_hvg_count:{assay_plan.assay}")
            operations.append(
                {
                    "operation": "diagnose_hvg_candidates",
                    "assay": assay_plan.assay,
                    "cellSelection": cell_selection_model.model_dump(mode="json"),
                    "minCells": min_cells,
                    "technicalGroupColumn": technical_column,
                    "rankingMode": diagnostic.ranking_mode,
                    "validTechnicalGroups": list(diagnostic.valid_groups),
                    "excludedTechnicalGroups": list(diagnostic.excluded_groups),
                    "candidateCounts": [
                        candidate.top_n for candidate in diagnostic.candidates
                    ],
                    "selectedTopN": hvg_payload.topN,
                    "rankingDecisionSnapshotSha256": (
                        ranking_resolution.snapshotSha256
                    ),
                    "countDecisionSnapshotSha256": resolution.snapshotSha256,
                    "invalidateCache": False,
                    "artifact": diagnostic_model.model_dump(mode="json"),
                }
            )
            operations.extend(
                [
                    {
                        "operation": "set_feature_selection",
                        "assay": assay_plan.assay,
                        "source": ArtifactReferenceModel.from_artifact_ref(
                            detected
                        ).model_dump(mode="json"),
                        "exactExcludedFeatures": list(assay_plan.exactExcludedFeatures),
                        "excludeFamilies": list(
                            assay_plan.featureParameters.get("excludeFamilies", [])
                        ),
                        "artifact": ArtifactReferenceModel.from_artifact_ref(
                            eligible_features
                        ).model_dump(mode="json"),
                    },
                    {
                        "operation": "select_detected_features",
                        "assay": assay_plan.assay,
                        "cellSelection": cell_selection_model.model_dump(mode="json"),
                        "minCells": min_cells,
                        "artifact": ArtifactReferenceModel.from_artifact_ref(
                            detected
                        ).model_dump(mode="json"),
                    },
                    {
                        "operation": "set_feature_selection",
                        "assay": assay_plan.assay,
                        "source": ArtifactReferenceModel.from_artifact_ref(
                            detected
                        ).model_dump(mode="json"),
                        "exactExcludedFeatures": list(assay_plan.exactExcludedFeatures),
                        "excludeFamilies": [],
                        "artifact": ArtifactReferenceModel.from_artifact_ref(
                            marker_features
                        ).model_dump(mode="json"),
                    },
                ]
            )
            artifacts.update(
                {
                    f"{assay_plan.assay}_eligible_features": (
                        ArtifactReferenceModel.from_artifact_ref(eligible_features)
                    ),
                    f"{assay_plan.assay}_detected_features": (
                        ArtifactReferenceModel.from_artifact_ref(detected)
                    ),
                }
            )
        elif assay_plan.featureMethod == "prevalentPeaks":
            actual_top_n = min(
                int(assay_plan.featureParameters["topN"]),
                assay.feats.N - 1,
            )
            graph_features = store.select_prevalent_peaks(
                cell_selection,
                from_assay=assay_plan.assay,
                top_n=actual_top_n,
                invalidate_cache=False,
            )
            marker_features = graph_features
            actions.append(f"select_prevalent_peaks:{assay_plan.assay}")
            operations.append(
                {
                    "operation": "select_prevalent_peaks",
                    "assay": assay_plan.assay,
                    "cellSelection": cell_selection_model.model_dump(mode="json"),
                    "topN": actual_top_n,
                    "invalidateCache": False,
                    "artifact": ArtifactReferenceModel.from_artifact_ref(
                        graph_features
                    ).model_dump(mode="json"),
                }
            )
        elif assay_plan.featureMethod == "panel":
            mask = np.ones(assay.feats.N, dtype=bool)
            ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
            names = np.asarray(assay.feats.fetch_all("names")).astype(str)
            excluded = set(assay_plan.exactExcludedFeatures)
            if excluded:
                mask &= ~np.isin(ids, list(excluded))
                mask &= ~np.isin(names, list(excluded))
            if int(mask.sum()) < 2:
                raise ValueError(
                    f"ADT assay {assay_plan.assay!r} has fewer than two non-control features"
                )
            graph_features = store.set_feature_selection(
                from_assay=assay_plan.assay,
                mask=mask,
                invalidate_cache=False,
            )
            marker_features = graph_features
            actions.append(f"select_adt_panel:{assay_plan.assay}")
            operations.append(
                {
                    "operation": "set_feature_selection",
                    "assay": assay_plan.assay,
                    "selectedFeatures": int(mask.sum()),
                    "exactExcludedFeatures": sorted(excluded),
                    "artifact": ArtifactReferenceModel.from_artifact_ref(
                        graph_features
                    ).model_dump(mode="json"),
                }
            )
        else:
            raise ValueError(f"Unsupported feature route {assay_plan.featureMethod!r}")
        normalized = store.run_normalization(
            cell_selection,
            features=graph_features,
            log_transform=cast(
                bool,
                assay_plan.normalizationParameters.get("logTransform"),
            ),
            renormalize_subset=cast(
                bool,
                assay_plan.normalizationParameters.get("renormalizeSubset"),
            ),
            invalidate_cache=False,
        )
        graph_feature_group = store.load_artifact(graph_features)
        graph_feature_values = cast(Any, graph_feature_group["values"])
        selected_values = np.asarray(graph_feature_values[:], dtype=bool)
        graph_features_model = ArtifactReferenceModel.from_artifact_ref(graph_features)
        marker_features_model = ArtifactReferenceModel.from_artifact_ref(
            marker_features
        )
        normalized_model = ArtifactReferenceModel.from_artifact_ref(normalized)
        handoff = PreprocessedAssayHandoff(
            assay=assay_plan.assay,
            assayType=assay_plan.assayType,
            cellSelection=cell_selection_model,
            reductionMethod=assay_plan.reductionMethod,
            graphFeatures=graph_features_model,
            markerFeatures=marker_features_model,
            normalized=normalized_model,
            nCells=active_cells,
            nFeatures=int(selected_values.sum()),
        )
        artifacts.update(
            {
                f"{assay_plan.assay}_graph_features": graph_features_model,
                f"{assay_plan.assay}_marker_features": marker_features_model,
                f"{assay_plan.assay}_normalized": normalized_model,
            }
        )
        actions.append(f"normalize:{assay_plan.assay}")
        operations.append(
            {
                "operation": "run_normalization",
                "assay": assay_plan.assay,
                "cellSelection": cell_selection_model.model_dump(mode="json"),
                "features": graph_features_model.model_dump(mode="json"),
                "logTransform": assay_plan.normalizationParameters.get("logTransform"),
                "renormalizeSubset": assay_plan.normalizationParameters.get(
                    "renormalizeSubset"
                ),
                "invalidateCache": False,
                "artifact": normalized_model.model_dump(mode="json"),
            }
        )
        logger.info(
            f"Preprocessed assay {assay_plan.assay!r}: "
            f"cells={handoff.nCells}, features={handoff.nFeatures}, "
            f"reduction={handoff.reductionMethod!r}"
        )
        return handoff

    def apply_cell_qc(
        self,
        store: DataStore,
        experimental: ExperimentalContextResult,
        cell_selection: ArtifactRef,
        actions: list[str],
        operations: list[dict[str, Any]],
        *,
        selected_plan: CellQcPlan | None = None,
        decision_payload: CellQualityExecutorPayload | None = None,
    ) -> ArtifactRef:
        plan = selected_plan or experimental.cellQc
        if decision_payload is not None:
            if plan.registeredProfile != decision_payload.profile:
                raise ValueError(
                    "Cell-QC execution plan differs from its audited payload"
                )
            expected_capture = decision_payload.profile in {
                "captureMad5",
                "captureMad3Sensitivity",
                "pooledReferenceMad5",
            }
            if expected_capture != (
                plan.sampleColumn is not None or plan.sampleArtifact is not None
            ):
                raise ValueError(
                    "Cell-QC capture source differs from its audited payload"
                )
        input_model = ArtifactReferenceModel.from_artifact_ref(cell_selection)
        logger.info(
            f"Applying cell QC action={plan.action!r}, profile={plan.profileId!r}"
        )
        profile = next(
            (
                value
                for value in experimental.qcProfiles
                if value.profileId == plan.profileId
            ),
            None,
        )
        if profile is None:
            raise ValueError("Experimental Context selected an unknown QC profile")
        for field_name in (
            "action",
            "registeredProfile",
            "driverAssay",
            "driverAssayType",
            "sampleColumn",
            "sampleArtifact",
            "attributes",
            "artifactMetrics",
        ):
            if getattr(plan, field_name) != getattr(profile, field_name):
                raise ValueError(
                    "Selected cell-QC plan does not match its exact offered profile"
                )
        if any(
            source not in experimental.qualityMetricArtifacts
            for source in plan.artifactMetrics
        ):
            raise ValueError(
                "Selected cell-QC metrics are absent from Experimental Context"
            )
        if (
            plan.sampleArtifact is not None
            and plan.sampleArtifact not in experimental.htoIdentityArtifacts
        ):
            raise ValueError(
                "Selected cell-QC sample artifact is absent from Experimental Context"
            )
        artifact_metrics = [
            NamedCellArtifact(
                name=source.name,
                artifact=artifact_model_to_ref(source.artifact),
            )
            for source in plan.artifactMetrics
        ]
        sample_artifact = (
            None
            if plan.sampleArtifact is None
            else NamedCellArtifact(
                name=plan.sampleArtifact.name,
                artifact=artifact_model_to_ref(plan.sampleArtifact.artifact),
            )
        )
        if plan.registeredProfile is not None:
            result, diagnostic_flags = execute_registered_cell_qc(
                store,
                plan.registeredProfile,
                profile_parameters=profile.parameters,
                expected_active_cells=profile.activeCells,
                expected_retained_cells=profile.retainedCells,
                expected_flag_counts=profile.flaggedCells,
                attrs=plan.attributes,
                artifact_metrics=artifact_metrics,
                cell_selection=cell_selection,
                sample_column=plan.sampleColumn,
                sample_artifact=sample_artifact,
                invalidate_cache=False,
            )
            result_model = ArtifactReferenceModel.from_artifact_ref(result)
            flags_model = (
                None
                if diagnostic_flags is None
                else ArtifactReferenceModel.from_artifact_ref(diagnostic_flags)
            )
            actions.append(f"cell_qc_registered:{profile.registeredProfile}")
            operations.append(
                {
                    "operation": "run_registered_cell_qc",
                    "profileId": profile.profileId,
                    "registeredProfile": profile.registeredProfile,
                    "cellSelection": input_model.model_dump(mode="json"),
                    "attrs": list(plan.attributes),
                    "artifactMetrics": [
                        source.model_dump(mode="json")
                        for source in plan.artifactMetrics
                    ],
                    "sampleColumn": plan.sampleColumn,
                    "sampleArtifact": (
                        None
                        if plan.sampleArtifact is None
                        else plan.sampleArtifact.model_dump(mode="json")
                    ),
                    "profileParameters": profile.parameters,
                    "expectedActiveCells": profile.activeCells,
                    "expectedRetainedCells": profile.retainedCells,
                    "expectedFlagCounts": profile.flaggedCells,
                    "invalidateCache": False,
                    "diagnosticFlags": (
                        None
                        if flags_model is None
                        else flags_model.model_dump(mode="json")
                    ),
                    "artifact": result_model.model_dump(mode="json"),
                }
            )
            return result
        if plan.action == "skip":
            actions.append("skip_cell_qc")
            operations.append(
                {
                    "operation": "skip_cell_qc",
                    "profileId": plan.profileId,
                    "cellSelection": input_model.model_dump(mode="json"),
                    "artifact": input_model.model_dump(mode="json"),
                }
            )
            return cell_selection
        if plan.action == "globalGaussian":
            if plan.sampleColumn is not None or sample_artifact is not None:
                raise ValueError("globalGaussian QC cannot include a sample source")
            result = store.auto_filter_cells(
                attrs=plan.attributes,
                artifact_metrics=artifact_metrics,
                min_p=float(profile.parameters.get("minP", 0.01)),
                max_p=float(profile.parameters.get("maxP", 0.99)),
                cell_selection=cell_selection,
                invalidate_cache=False,
            )
            result_model = ArtifactReferenceModel.from_artifact_ref(result)
            actions.append(f"cell_qc_global:{profile.profileId}")
            operations.append(
                {
                    "operation": "auto_filter_cells",
                    "profileId": profile.profileId,
                    "cellSelection": input_model.model_dump(mode="json"),
                    "attrs": list(plan.attributes),
                    "artifactMetrics": [
                        source.model_dump(mode="json")
                        for source in plan.artifactMetrics
                    ],
                    "minP": float(profile.parameters.get("minP", 0.01)),
                    "maxP": float(profile.parameters.get("maxP", 0.99)),
                    "sampleColumn": None,
                    "invalidateCache": False,
                    "artifact": result_model.model_dump(mode="json"),
                }
            )
            return result
        if plan.action == "sampleMad":
            if (plan.sampleColumn is None) == (sample_artifact is None):
                raise ValueError(
                    "sampleMad QC requires exactly one metadata or artifact "
                    "sample source"
                )
            result = store.auto_filter_cells(
                attrs=plan.attributes,
                artifact_metrics=artifact_metrics,
                cell_selection=cell_selection,
                sample_column=plan.sampleColumn,
                sample_artifact=sample_artifact,
                n_mads=float(profile.parameters.get("nMads", 3.0)),
                min_cells_per_sample=int(
                    profile.parameters.get("minCellsPerSample", 20)
                ),
                invalidate_cache=False,
            )
            result_model = ArtifactReferenceModel.from_artifact_ref(result)
            actions.append(f"cell_qc_sample_mad:{profile.profileId}")
            operations.append(
                {
                    "operation": "auto_filter_cells",
                    "profileId": profile.profileId,
                    "cellSelection": input_model.model_dump(mode="json"),
                    "attrs": list(plan.attributes),
                    "artifactMetrics": [
                        source.model_dump(mode="json")
                        for source in plan.artifactMetrics
                    ],
                    "minP": 0.01,
                    "maxP": 0.99,
                    "sampleColumn": plan.sampleColumn,
                    "sampleArtifact": (
                        None
                        if plan.sampleArtifact is None
                        else plan.sampleArtifact.model_dump(mode="json")
                    ),
                    "nMads": float(profile.parameters.get("nMads", 3.0)),
                    "minCellsPerSample": int(
                        profile.parameters.get("minCellsPerSample", 20)
                    ),
                    "invalidateCache": False,
                    "artifact": result_model.model_dump(mode="json"),
                }
            )
            return result
        raise ValueError(f"Unsupported cell QC action {plan.action!r}")

    def rna_blacklist(self, plan: AssayPreprocessingPlan) -> str:
        patterns: list[str] = []
        families = set(
            cast(list[str], plan.featureParameters.get("excludeFamilies", []))
        )
        if "mitochondrial" in families:
            patterns.append(r"^(MT-|mt-)")
        if "ribosomal" in families:
            patterns.append(r"^(RPS|RPL|MRPS|MRPL|Rps|Rpl|Mrps|Mrpl)")
        if "histone" in families:
            patterns.append(r"^(HIST|Hist)")
        patterns.extend(
            rf"^{re.escape(value)}$" for value in plan.exactExcludedFeatures if value
        )
        return "|".join(patterns) if patterns else r"(?!)"

    def exclude_exact_features(
        self,
        store: DataStore,
        plan: AssayPreprocessingPlan,
        source: ArtifactRef,
        *,
        include_families: bool = True,
    ) -> ArtifactRef:
        families = (
            set(cast(list[str], plan.featureParameters.get("excludeFamilies", [])))
            if include_families
            else set()
        )
        if not plan.exactExcludedFeatures and not families:
            return source
        assay = store.get_assay(plan.assay)
        source_group = store.load_artifact(source)
        source_values = cast(Any, source_group["values"])
        mask = np.asarray(source_values[:], dtype=bool)
        ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
        names = np.asarray(assay.feats.fetch_all("names")).astype(str)
        excluded = set(plan.exactExcludedFeatures)
        mask &= ~np.isin(ids, list(excluded))
        mask &= ~np.isin(names, list(excluded))
        family_patterns: list[str] = []
        if "mitochondrial" in families:
            family_patterns.append(r"^(MT-|mt-)")
        if "ribosomal" in families:
            family_patterns.append(r"^(RPS|RPL|MRPS|MRPL|Rps|Rpl|Mrps|Mrpl)")
        if "histone" in families:
            family_patterns.append(r"^(HIST|Hist)")
        if family_patterns:
            technical = np.zeros(len(mask), dtype=bool)
            combined = re.compile("|".join(family_patterns))
            technical |= np.fromiter(
                (combined.search(value) is not None for value in ids),
                dtype=bool,
                count=len(ids),
            )
            technical |= np.fromiter(
                (combined.search(value) is not None for value in names),
                dtype=bool,
                count=len(names),
            )
            mask &= ~technical
        if not mask.any():
            raise ValueError("Exact marker exclusions removed every feature")
        return store.set_feature_selection(
            from_assay=plan.assay,
            mask=mask,
            invalidate_cache=False,
        )
