"""Shared resolution and persistence for RNA workflow decisions."""

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai.exceptions import AgentRunError

from .. import record_io
from ..config.agent_exec import run_agent_sync
from ..decisions.kernel import (
    DecisionRecord,
    DecisionSelection,
    DecisionSource,
    DecisionWorkflowRun,
    EvidenceBundle,
    PendingDecision,
    RevisionRequest,
)
from ..decisions.rna import (
    CompiledRnaDecision,
    RnaDecisionDefinition,
    compile_rna_decision,
)
from ..persistence.decisions import (
    attach_audited_rna_decision,
    load_latest_decision_workflow_snapshot,
    pause_decision_workflow,
    save_decision_workflow_snapshot,
)
from .models import OrchestrationRequestRecord, WorkflowQuestion


@dataclass(frozen=True, slots=True)
class DecisionResolution:
    """Runtime result of resolving or pausing one exact checkpoint."""

    workflow: DecisionWorkflowRun
    record: DecisionRecord | None
    compiled: CompiledRnaDecision | None
    snapshotSha256: str

    @property
    def pending(self) -> PendingDecision | None:
        return self.workflow.pendingDecision


@dataclass(frozen=True, slots=True)
class DecisionReconsideration:
    """Result of one downstream-evidence review of an active decision."""

    selection: DecisionSelection | None
    resolution: DecisionResolution | None
    revised: bool
    snapshotSha256: str
    question: WorkflowQuestion | None = None


def _sha256(value: object) -> str:
    return hashlib.sha256(record_io.canonical_json_bytes(value)).hexdigest()


def _software_sha256(definition: RnaDecisionDefinition) -> str:
    return _sha256(
        {
            "controller": "rnaDecisionResolver",
            "definition": definition.model_dump(mode="json"),
        }
    )


def _selection_evidence_for_human(
    definition: RnaDecisionDefinition,
    evidence: EvidenceBundle,
    option_id: str,
) -> list[str]:
    option = definition.spec.option_by_id()[option_id]
    required = set(option.requiredEvidenceClasses)
    evidence_ids = [
        item.evidenceId
        for item in evidence.evidence
        if not required or item.evidenceClass in required
    ]
    return list(dict.fromkeys([*option.requiredEvidenceIds, *evidence_ids]))


def _selection_for_option(
    definition: RnaDecisionDefinition,
    evidence: EvidenceBundle,
    option_id: str,
    *,
    rationale: str,
) -> DecisionSelection:
    evidence_ids = _selection_evidence_for_human(definition, evidence, option_id)
    override_of: str | None = None
    override_evidence_ids: list[str] = []
    selected = definition.spec.option_by_id()[option_id]
    if (
        definition.spec.requireIndependentOverrideEvidence
        and definition.spec.metricPreferredOptionId is not None
        and option_id != definition.spec.metricPreferredOptionId
        and selected.status in {"apply", "skip"}
    ):
        override_of = definition.spec.metricPreferredOptionId
        required_ids = set(selected.requiredEvidenceIds)
        override_evidence_ids = [
            item.evidenceId
            for item in evidence.evidence
            if item.evidenceClass
            in {
                "markerCoherence",
                "resamplingStability",
                "crossUnitSupport",
                "protectedVariablePreservation",
            }
            and (not required_ids or item.evidenceId in required_ids)
        ]
        evidence_ids = list(dict.fromkeys([*evidence_ids, *override_evidence_ids]))
    return _validate_selection(
        definition,
        evidence,
        DecisionSelection(
            selectedOptionId=option_id,
            evidenceIds=evidence_ids,
            rationale=rationale,
            confidence="notApplicable",
            overrideOfOptionId=override_of,
            overrideEvidenceIds=override_evidence_ids,
        ),
    )


def _unattended_option_id(definition: RnaDecisionDefinition) -> str:
    options = definition.spec.option_by_id()
    ordered = [
        definition.spec.metricPreferredOptionId,
        definition.spec.baselineOptionId,
        *(option.optionId for option in definition.spec.options),
    ]
    for option_id in ordered:
        if option_id is not None and options[option_id].status != "defer":
            return option_id
    raise ValueError("The registered decision has no non-deferred option")


def _unattended_selection(
    definition: RnaDecisionDefinition,
    evidence: EvidenceBundle,
    *,
    option_id: str | None = None,
    reason: str,
) -> DecisionSelection:
    if "rule" not in definition.spec.allowedSources:
        raise ValueError(
            "The registered decision does not allow deterministic resolution"
        )
    selected_option_id = option_id or _unattended_option_id(definition)
    return _selection_for_option(
        definition,
        evidence,
        selected_option_id,
        rationale=reason,
    )


def _validate_selection(
    definition: RnaDecisionDefinition,
    evidence: EvidenceBundle,
    selection: DecisionSelection,
) -> DecisionSelection:
    options = definition.spec.option_by_id()
    if selection.selectedOptionId not in options:
        raise ValueError("selectedOptionId is not an offered decision option")
    available = evidence.evidence_by_id()
    if not set(selection.evidenceIds).issubset(available):
        raise ValueError("Decision selection cites unavailable evidence")
    selected = options[selection.selectedOptionId]
    cited_classes = {
        available[evidence_id].evidenceClass for evidence_id in selection.evidenceIds
    }
    if not set(selected.requiredEvidenceClasses).issubset(cited_classes):
        raise ValueError("Decision selection omits a required evidence class")
    if not set(selected.requiredEvidenceIds).issubset(selection.evidenceIds):
        raise ValueError("Decision selection omits option-specific evidence")
    if selection.overrideOfOptionId is not None and (
        selection.overrideOfOptionId not in options
    ):
        raise ValueError("overrideOfOptionId is not an offered decision option")
    if (
        definition.spec.requireIndependentOverrideEvidence
        and definition.spec.metricPreferredOptionId is not None
        and selection.selectedOptionId != definition.spec.metricPreferredOptionId
        and selected.status in {"apply", "skip"}
    ):
        evidence_classes = {
            available[evidence_id].evidenceClass
            for evidence_id in selection.overrideEvidenceIds
            if evidence_id in available
        }
        independent_classes = evidence_classes.intersection(
            {
                "markerCoherence",
                "resamplingStability",
                "crossUnitSupport",
                "protectedVariablePreservation",
            }
        )
        if (
            selection.overrideOfOptionId != definition.spec.metricPreferredOptionId
            or not set(selection.overrideEvidenceIds).issubset(selection.evidenceIds)
            or (
                selected.requiredEvidenceIds
                and not set(selection.overrideEvidenceIds).issubset(
                    selected.requiredEvidenceIds
                )
            )
            or len(independent_classes) < 2
        ):
            raise ValueError(
                "A metric override requires two independent non-geometric "
                "evidence classes"
            )
    return selection


def _record_from_selection(
    *,
    workflow_run_id: str,
    definition: RnaDecisionDefinition,
    evidence: EvidenceBundle,
    selection: DecisionSelection,
    source: DecisionSource,
    model_name: str | None,
    prompt_sha256: str | None,
    supersedes: str | None,
    created_at_ns: int,
) -> DecisionRecord:
    if evidence.contentSha256 is None:
        raise ValueError("Decision evidence bundle requires a content checksum")
    option = definition.spec.option_by_id()[selection.selectedOptionId]
    identity = {
        "workflowRunId": workflow_run_id,
        "decisionId": definition.spec.decisionId,
        "definitionVersion": definition.spec.definitionVersion,
        "evidenceBundleId": evidence.bundleId,
        "evidenceBundleSha256": evidence.contentSha256,
        "selection": selection.model_dump(mode="json"),
        "source": source,
        "supersedes": supersedes,
    }
    record_id = f"decision:{definition.spec.decisionId}:{_sha256(identity)[:24]}"
    return DecisionRecord(
        recordId=record_id,
        decisionId=definition.spec.decisionId,
        definitionVersion=definition.spec.definitionVersion,
        evidenceBundleId=evidence.bundleId,
        evidenceBundleSha256=evidence.contentSha256,
        offeredOptionIds=[offered.optionId for offered in definition.spec.options],
        availableEvidenceIds=[item.evidenceId for item in evidence.evidence],
        selectedOptionId=selection.selectedOptionId,
        status=option.status,
        source=source,
        evidenceIds=list(selection.evidenceIds),
        rationale=selection.rationale,
        confidence=selection.confidence,
        protectedVariableEffects=list(selection.protectedVariableEffects),
        overrideOfOptionId=selection.overrideOfOptionId,
        overrideEvidenceIds=list(selection.overrideEvidenceIds),
        promptSha256=prompt_sha256,
        modelName=model_name,
        softwareSha256=_software_sha256(definition),
        verificationId=f"verification:{record_id}",
        supersedes=supersedes,
        createdAtNs=created_at_ns,
    )


def _active_record(
    workflow: DecisionWorkflowRun,
    decision_id: str,
) -> DecisionRecord | None:
    matches = [
        record
        for record in workflow.active_decision_records()
        if record.decisionId == decision_id
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Decision workflow has multiple active {decision_id!r} records"
        )
    return matches[0] if matches else None


class DecisionStagesMixin:
    """Resolve every rule, agent, and human choice through one ledger path."""

    model: Any

    def _load_or_create_decision_workflow(
        self,
        store: Any,
        request_record: OrchestrationRequestRecord,
    ) -> tuple[DecisionWorkflowRun, str]:
        try:
            snapshot = load_latest_decision_workflow_snapshot(
                store,
                request_record.workflowRunId,
                workspace=request_record.request.workspace,
            )
        except KeyError:
            workflow = DecisionWorkflowRun(
                workflowRunId=request_record.workflowRunId,
                maxRevisions=request_record.config.maxRevisions,
            )
            snapshot = save_decision_workflow_snapshot(
                store,
                workflow,
                workspace=request_record.request.workspace,
            )
        if snapshot.workflow.maxRevisions != request_record.config.maxRevisions:
            raise ValueError(
                "Persisted decision revision limit differs from the request"
            )
        return snapshot.workflow, snapshot.contentSha256

    def _reconsider_rna_decision(
        self,
        store: Any,
        request_record: OrchestrationRequestRecord,
        definition: RnaDecisionDefinition,
        evidence: EvidenceBundle,
        answers: Mapping[str, Any],
        *,
        review_instructions: str | None = None,
        visual_content: Sequence[Any] = (),
        agent_selection: DecisionSelection | None = None,
        agent_model_name: str | None = None,
    ) -> DecisionReconsideration:
        """Select from new evidence, then create a revision only when it changes."""
        evidence = (
            evidence
            if evidence.contentSha256 is not None
            else evidence.with_content_sha256()
        )
        if evidence.contentSha256 is None:
            raise RuntimeError("Reconsideration evidence checksum was not created")
        workflow, snapshot_sha256 = self._load_or_create_decision_workflow(
            store,
            request_record,
        )
        target = _active_record(workflow, definition.spec.decisionId)
        if target is None:
            raise ValueError("Reconsideration requires an active target decision")
        question_id = f"decision:{definition.spec.decisionId}Review"
        raw_answer = answers.get(question_id)
        if raw_answer is not None and agent_selection is not None:
            raise ValueError(
                "Decision reconsideration cannot combine human and agent selections"
            )
        source: DecisionSource
        model_name: str | None = None
        if raw_answer is not None:
            if not isinstance(raw_answer, Mapping):
                raise ValueError("Human reconsideration answer must be a mapping")
            if set(raw_answer) != {"decisionId", "optionId", "rationale"}:
                raise ValueError(
                    "Human reconsideration answer requires decisionId, optionId, "
                    "and rationale"
                )
            if raw_answer.get("decisionId") != definition.spec.decisionId:
                raise ValueError("Human reconsideration answer has a stale decisionId")
            option_id = raw_answer.get("optionId")
            rationale = raw_answer.get("rationale")
            if not isinstance(option_id, str) or not isinstance(rationale, str):
                raise ValueError("Human reconsideration answer has invalid values")
            selection = _validate_selection(
                definition,
                evidence,
                DecisionSelection(
                    selectedOptionId=option_id,
                    evidenceIds=_selection_evidence_for_human(
                        definition,
                        evidence,
                        option_id,
                    ),
                    rationale=rationale.strip(),
                    confidence="notApplicable",
                ),
            )
            source = "human"
        elif agent_selection is not None:
            selection = _validate_selection(
                definition,
                evidence,
                agent_selection,
            )
            source = "agent"
            model_name = agent_model_name
        else:
            payload = {
                "decisionId": definition.spec.decisionId,
                "studyObjective": request_record.request.studyObjective,
                "question": definition.spec.question,
                "baselineOptionId": definition.spec.baselineOptionId,
                "options": [
                    option.model_dump(mode="json") for option in definition.spec.options
                ],
                "evidence": [
                    item.model_dump(mode="json") for item in evidence.evidence
                ],
            }
            text_prompt = json.dumps(payload, indent=2, sort_keys=True)
            user_prompt: Any = (
                [text_prompt, *visual_content] if visual_content else text_prompt
            )
            try:
                execution = run_agent_sync(
                    model=self.model,
                    output_type=DecisionSelection,
                    system_prompt=(
                        review_instructions
                        or (
                            "Reconsider the active decision using only the "
                            "new evidence and registered options. Cite every "
                            "required evidence ID and class. Keep the active option "
                            "unless the evidence supports a specific replacement. "
                            "Do not invent operations, parameters, artifacts, or "
                            "evidence."
                        )
                    ),
                    user_prompt=user_prompt,
                    config=request_record.config.agentRunConfig,
                    name=f"decision_{definition.spec.decisionId}_review",
                    output_validator=lambda value: _validate_selection(
                        definition,
                        evidence,
                        value,
                    ),
                )
            except AgentRunError:
                if request_record.config.inputPolicy == "unattended":
                    selection = _unattended_selection(
                        definition,
                        evidence,
                        option_id=target.selectedOptionId,
                        reason=(
                            "The reconsideration model failed, so the unattended "
                            "workflow retained the active registered option."
                        ),
                    )
                    source = "rule"
                else:
                    return DecisionReconsideration(
                        selection=None,
                        resolution=None,
                        revised=False,
                        snapshotSha256=snapshot_sha256,
                        question=WorkflowQuestion(
                            questionId=question_id,
                            decisionId=definition.spec.decisionId,
                            question=definition.spec.question,
                            options=[
                                option.optionId for option in definition.spec.options
                            ],
                            evidenceIds=[item.evidenceId for item in evidence.evidence],
                        ),
                    )
            else:
                if not isinstance(execution.output, DecisionSelection):
                    raise TypeError(
                        "Decision reconsideration returned an unexpected output type"
                    )
                selection = _validate_selection(definition, evidence, execution.output)
                source = "agent"
                model_name = execution.runInfo.modelName

        selected_status = definition.spec.option_by_id()[
            selection.selectedOptionId
        ].status
        if selected_status == "defer":
            if request_record.config.inputPolicy == "unattended":
                selection = _unattended_selection(
                    definition,
                    evidence,
                    option_id=target.selectedOptionId,
                    reason=(
                        "The reconsideration model deferred, so the unattended "
                        "workflow retained the active registered option."
                    ),
                )
                source = "rule"
                selected_status = definition.spec.option_by_id()[
                    selection.selectedOptionId
                ].status
            else:
                return DecisionReconsideration(
                    selection=selection,
                    resolution=None,
                    revised=False,
                    snapshotSha256=snapshot_sha256,
                    question=WorkflowQuestion(
                        questionId=question_id,
                        decisionId=definition.spec.decisionId,
                        question=definition.spec.question,
                        options=[option.optionId for option in definition.spec.options],
                        evidenceIds=[item.evidenceId for item in evidence.evidence],
                    ),
                )
        if selection.selectedOptionId == target.selectedOptionId:
            return DecisionReconsideration(
                selection=selection,
                resolution=None,
                revised=False,
                snapshotSha256=snapshot_sha256,
            )
        if len(workflow.revisionRequests) >= workflow.maxRevisions:
            if request_record.config.inputPolicy == "unattended":
                retained = _unattended_selection(
                    definition,
                    evidence,
                    option_id=target.selectedOptionId,
                    reason=(
                        "The revision budget is exhausted, so the unattended "
                        "workflow retained the active registered option."
                    ),
                )
                return DecisionReconsideration(
                    selection=retained,
                    resolution=None,
                    revised=False,
                    snapshotSha256=snapshot_sha256,
                )
            return DecisionReconsideration(
                selection=selection,
                resolution=None,
                revised=False,
                snapshotSha256=snapshot_sha256,
                question=WorkflowQuestion(
                    questionId=question_id,
                    decisionId=definition.spec.decisionId,
                    question=(
                        "The observed evidence supports changing this decision, "
                        "but the bounded revision budget is exhausted. Retain the "
                        "active option explicitly or stop the workflow."
                    ),
                    options=[target.selectedOptionId],
                    evidenceIds=[item.evidenceId for item in evidence.evidence],
                ),
            )

        target_position = workflow.decisionRecords.index(target)
        active_ids = {record.recordId for record in workflow.active_decision_records()}
        invalidated = [
            record.recordId
            for record in workflow.decisionRecords[target_position + 1 :]
            if record.recordId in active_ids
        ]
        revision_id = (
            "revision:"
            f"{_sha256({'target': target.recordId, 'bundle': evidence.bundleId, 'option': selection.selectedOptionId})[:24]}"
        )
        if target.verificationId is None:
            raise ValueError("Reconsideration target lacks deterministic verification")
        revision = RevisionRequest(
            revisionId=revision_id,
            targetDecisionRecordId=target.recordId,
            verificationId=target.verificationId,
            replacementOptionId=selection.selectedOptionId,
            reason=selection.rationale,
            evidenceBundleId=evidence.bundleId,
            evidenceBundleSha256=evidence.contentSha256,
            availableEvidenceIds=[item.evidenceId for item in evidence.evidence],
            evidenceIds=list(selection.evidenceIds),
            invalidatesDecisionRecordIds=invalidated,
            createdAtNs=time.time_ns(),
        )
        if source == "agent":
            resolution = self._resolve_rna_decision(
                store,
                request_record,
                definition,
                evidence,
                {},
                agent_selection=selection,
                agent_model_name=model_name,
                revision=revision,
            )
        else:
            resolution = self._resolve_rna_decision(
                store,
                request_record,
                definition,
                evidence,
                {
                    f"decision:{definition.spec.decisionId}": {
                        "decisionId": definition.spec.decisionId,
                        "optionId": selection.selectedOptionId,
                        "rationale": selection.rationale,
                    }
                },
                revision=revision,
            )
        return DecisionReconsideration(
            selection=selection,
            resolution=resolution,
            revised=True,
            snapshotSha256=resolution.snapshotSha256,
        )

    def _resolve_rna_decision(
        self,
        store: Any,
        request_record: OrchestrationRequestRecord,
        definition: RnaDecisionDefinition,
        evidence: EvidenceBundle,
        answers: Mapping[str, Any],
        *,
        rule_selection: DecisionSelection | None = None,
        agent_selection: DecisionSelection | None = None,
        agent_model_name: str | None = None,
        revision: RevisionRequest | None = None,
    ) -> DecisionResolution:
        evidence = (
            evidence
            if evidence.contentSha256 is not None
            else evidence.with_content_sha256()
        )
        evidence_sha256 = evidence.contentSha256
        if evidence_sha256 is None:
            raise RuntimeError("Decision evidence checksum was not created")
        if evidence.decisionId != definition.spec.decisionId:
            raise ValueError("Evidence does not match the decision definition")
        if evidence.bundleId != definition.spec.evidenceBundleId:
            raise ValueError("Evidence bundle identity does not match the definition")
        if revision is not None and (
            revision.evidenceBundleId != evidence.bundleId
            or revision.evidenceBundleSha256 != evidence_sha256
            or revision.availableEvidenceIds
            != [item.evidenceId for item in evidence.evidence]
        ):
            raise ValueError(
                "Revision does not reference the exact replacement evidence bundle"
            )

        workflow, snapshot_sha256 = self._load_or_create_decision_workflow(
            store,
            request_record,
        )
        supersedes = revision.targetDecisionRecordId if revision is not None else None
        existing = _active_record(workflow, definition.spec.decisionId)
        if existing is not None and revision is None:
            if (
                existing.definitionVersion != definition.spec.definitionVersion
                or existing.evidenceBundleId != evidence.bundleId
                or existing.offeredOptionIds
                != [option.optionId for option in definition.spec.options]
                or existing.availableEvidenceIds
                != [item.evidenceId for item in evidence.evidence]
            ):
                raise ValueError(
                    "Persisted decision does not match the current definition and evidence"
                )
            persisted_verifications = [
                verification
                for verification in workflow.verificationRecords
                if verification.decisionRecordId == existing.recordId
            ]
            if len(persisted_verifications) != 1:
                raise ValueError(
                    "Persisted decision lacks one exact verification record"
                )
            persisted_verification = persisted_verifications[0]
            compiled = compile_rna_decision(
                definition,
                evidence,
                existing,
                created_at_ns=persisted_verification.createdAtNs,
            )
            if compiled.verification != persisted_verification:
                raise ValueError(
                    "Persisted verification does not match deterministic replay"
                )
            return DecisionResolution(
                workflow=workflow,
                record=existing,
                compiled=compiled,
                snapshotSha256=snapshot_sha256,
            )
        if existing is None and revision is None:
            latest_matching = next(
                (
                    record
                    for record in reversed(workflow.decisionRecords)
                    if record.decisionId == definition.spec.decisionId
                ),
                None,
            )
            if (
                latest_matching is not None
                and latest_matching.recordId
                in workflow.invalidated_decision_record_ids()
            ):
                supersedes = latest_matching.recordId

        question_id = f"decision:{definition.spec.decisionId}"
        raw_answer = answers.get(question_id)
        source: DecisionSource
        prompt_sha256: str | None = None
        model_name: str | None = None
        selection: DecisionSelection

        if rule_selection is not None and agent_selection is not None:
            raise ValueError("A decision cannot have both rule and agent selections")
        if rule_selection is not None:
            if raw_answer is not None:
                raise ValueError("A rule-owned decision cannot accept a human answer")
            source = "rule"
            selection = _validate_selection(definition, evidence, rule_selection)
        elif agent_selection is not None:
            if raw_answer is not None:
                raise ValueError(
                    "A preselected agent decision cannot accept a human answer"
                )
            source = "agent"
            selection = _validate_selection(definition, evidence, agent_selection)
            model_name = agent_model_name
        elif raw_answer is not None:
            if not isinstance(raw_answer, Mapping):
                raise ValueError("Human decision answer must be a mapping")
            if set(raw_answer) != {"decisionId", "optionId", "rationale"}:
                raise ValueError(
                    "Human decision answer requires decisionId, optionId, and rationale"
                )
            if raw_answer.get("decisionId") != definition.spec.decisionId:
                raise ValueError("Human decision answer has a stale decisionId")
            option_id = raw_answer.get("optionId")
            rationale = raw_answer.get("rationale")
            if not isinstance(option_id, str) or not isinstance(rationale, str):
                raise ValueError("Human decision answer has invalid values")
            evidence_ids = _selection_evidence_for_human(
                definition, evidence, option_id
            )
            override_of: str | None = None
            override_evidence_ids: list[str] = []
            if (
                definition.spec.requireIndependentOverrideEvidence
                and definition.spec.metricPreferredOptionId is not None
                and option_id != definition.spec.metricPreferredOptionId
                and definition.spec.option_by_id()[option_id].status
                in {"apply", "skip"}
            ):
                override_of = definition.spec.metricPreferredOptionId
                selected_option = definition.spec.option_by_id()[option_id]
                option_evidence_ids = set(selected_option.requiredEvidenceIds)
                override_evidence_ids = [
                    item.evidenceId
                    for item in evidence.evidence
                    if item.evidenceClass
                    in {
                        "markerCoherence",
                        "resamplingStability",
                        "crossUnitSupport",
                        "protectedVariablePreservation",
                    }
                    and (
                        not option_evidence_ids
                        or item.evidenceId in option_evidence_ids
                    )
                ]
                evidence_ids = list(
                    dict.fromkeys([*evidence_ids, *override_evidence_ids])
                )
            selection = _validate_selection(
                definition,
                evidence,
                DecisionSelection(
                    selectedOptionId=option_id,
                    evidenceIds=evidence_ids,
                    rationale=rationale.strip(),
                    confidence="notApplicable",
                    overrideOfOptionId=override_of,
                    overrideEvidenceIds=override_evidence_ids,
                ),
            )
            source = "human"
        elif workflow.status == "needsInput" and workflow.pendingDecision is not None:
            if workflow.pendingDecision.decisionId != definition.spec.decisionId:
                raise ValueError(
                    "Decision workflow is paused at another exact checkpoint"
                )
            if request_record.config.inputPolicy == "unattended":
                selection = _unattended_selection(
                    definition,
                    evidence,
                    reason=(
                        "The unattended workflow resolved the persisted checkpoint "
                        "with the registered metric-preferred or baseline option."
                    ),
                )
                source = "rule"
            else:
                return DecisionResolution(
                    workflow=workflow,
                    record=None,
                    compiled=None,
                    snapshotSha256=snapshot_sha256,
                )
        else:
            payload = {
                "decisionId": definition.spec.decisionId,
                "studyObjective": request_record.request.studyObjective,
                "question": definition.spec.question,
                "baselineOptionId": definition.spec.baselineOptionId,
                "metricPreferredOptionId": definition.spec.metricPreferredOptionId,
                "requireIndependentOverrideEvidence": (
                    definition.spec.requireIndependentOverrideEvidence
                ),
                "options": [
                    option.model_dump(mode="json") for option in definition.spec.options
                ],
                "evidence": [
                    item.model_dump(mode="json") for item in evidence.evidence
                ],
            }
            user_prompt = json.dumps(payload, indent=2, sort_keys=True)
            prompt_sha256 = hashlib.sha256(user_prompt.encode()).hexdigest()
            try:
                execution = run_agent_sync(
                    model=self.model,
                    output_type=DecisionSelection,
                    system_prompt=(
                        "The task is to select one offered option from the supplied "
                        "evidence. A valid selection includes the option ID, every "
                        "option-specific evidence ID, the required evidence classes, "
                        "and a concise rationale. Numeric parameters and operations "
                        "are fixed by the executor. When independent override "
                        "evidence is required, a non-preferred option also identifies "
                        "the preferred option it overrides and cites two independent "
                        "non-geometric evidence classes."
                    ),
                    user_prompt=user_prompt,
                    config=request_record.config.agentRunConfig,
                    name=f"rna_{definition.spec.decisionId}_decision",
                    output_validator=lambda value: _validate_selection(
                        definition, evidence, value
                    ),
                )
            except AgentRunError as exc:
                if request_record.config.inputPolicy == "unattended":
                    selection = _unattended_selection(
                        definition,
                        evidence,
                        reason=(
                            "The bounded model run failed, so the unattended "
                            "workflow selected the registered metric-preferred or "
                            "baseline option."
                        ),
                    )
                    source = "rule"
                    prompt_sha256 = None
                    model_name = None
                else:
                    pending = PendingDecision(
                        questionId=question_id,
                        decisionId=definition.spec.decisionId,
                        definitionVersion=definition.spec.definitionVersion,
                        evidenceBundleId=evidence.bundleId,
                        evidenceBundleSha256=evidence_sha256,
                        offeredOptionIds=[
                            option.optionId for option in definition.spec.options
                        ],
                        availableEvidenceIds=[
                            item.evidenceId for item in evidence.evidence
                        ],
                        reason=(
                            "The bounded model run did not return a valid registered "
                            f"selection ({type(exc).__name__})."
                        ),
                        createdAtNs=time.time_ns(),
                    )
                    paused = pause_decision_workflow(workflow, pending)
                    snapshot = save_decision_workflow_snapshot(
                        store,
                        paused,
                        workspace=request_record.request.workspace,
                    )
                    return DecisionResolution(
                        workflow=snapshot.workflow,
                        record=None,
                        compiled=None,
                        snapshotSha256=snapshot.contentSha256,
                    )
            else:
                if not isinstance(execution.output, DecisionSelection):
                    raise TypeError(
                        "RNA decision model returned an unexpected output type"
                    )
                selection = _validate_selection(definition, evidence, execution.output)
                source = "agent"
                model_name = execution.runInfo.modelName

        selected_option = definition.spec.option_by_id()[selection.selectedOptionId]
        if selected_option.status == "defer":
            if request_record.config.inputPolicy == "unattended":
                selection = _unattended_selection(
                    definition,
                    evidence,
                    reason=(
                        "The model deferred, so the unattended workflow selected "
                        "the registered metric-preferred or baseline option."
                    ),
                )
                selected_option = definition.spec.option_by_id()[
                    selection.selectedOptionId
                ]
                source = "rule"
                prompt_sha256 = None
                model_name = None
            else:
                if workflow.status == "needsInput":
                    active_pending = workflow.pendingDecision
                    if active_pending is None or (
                        active_pending.decisionId != definition.spec.decisionId
                        or active_pending.definitionVersion
                        != definition.spec.definitionVersion
                        or active_pending.evidenceBundleId != evidence.bundleId
                        or active_pending.evidenceBundleSha256 != evidence_sha256
                        or active_pending.offeredOptionIds
                        != [option.optionId for option in definition.spec.options]
                        or active_pending.availableEvidenceIds
                        != [item.evidenceId for item in evidence.evidence]
                    ):
                        raise ValueError(
                            "Deferred answer does not match the exact pending checkpoint"
                        )
                    return DecisionResolution(
                        workflow=workflow,
                        record=None,
                        compiled=None,
                        snapshotSha256=snapshot_sha256,
                    )
                pending = PendingDecision(
                    questionId=question_id,
                    decisionId=definition.spec.decisionId,
                    definitionVersion=definition.spec.definitionVersion,
                    evidenceBundleId=evidence.bundleId,
                    evidenceBundleSha256=evidence_sha256,
                    offeredOptionIds=[
                        option.optionId for option in definition.spec.options
                    ],
                    availableEvidenceIds=[
                        item.evidenceId for item in evidence.evidence
                    ],
                    reason=selection.rationale,
                    createdAtNs=time.time_ns(),
                )
                paused = pause_decision_workflow(workflow, pending)
                snapshot = save_decision_workflow_snapshot(
                    store,
                    paused,
                    workspace=request_record.request.workspace,
                )
                return DecisionResolution(
                    workflow=snapshot.workflow,
                    record=None,
                    compiled=None,
                    snapshotSha256=snapshot.contentSha256,
                )

        created_at_ns = time.time_ns()
        record = _record_from_selection(
            workflow_run_id=request_record.workflowRunId,
            definition=definition,
            evidence=evidence,
            selection=selection,
            source=source,
            model_name=model_name,
            prompt_sha256=prompt_sha256,
            supersedes=supersedes,
            created_at_ns=created_at_ns,
        )
        compiled = compile_rna_decision(
            definition,
            evidence,
            record,
            created_at_ns=created_at_ns,
        )
        updated = attach_audited_rna_decision(
            workflow,
            record,
            compiled,
            revision=revision,
        )
        snapshot = save_decision_workflow_snapshot(
            store,
            updated,
            workspace=request_record.request.workspace,
        )
        return DecisionResolution(
            workflow=snapshot.workflow,
            record=record,
            compiled=compiled,
            snapshotSha256=snapshot.contentSha256,
        )

    @staticmethod
    def _pending_decision_question(
        resolution: DecisionResolution,
        definition: RnaDecisionDefinition,
    ) -> WorkflowQuestion:
        pending = resolution.pending
        if pending is None:
            raise ValueError("Decision resolution has no pending checkpoint")
        return WorkflowQuestion(
            questionId=pending.questionId,
            decisionId=pending.decisionId,
            question=definition.spec.question,
            options=list(pending.offeredOptionIds),
            evidenceIds=list(pending.availableEvidenceIds),
        )


__all__ = ["DecisionResolution", "DecisionStagesMixin"]
