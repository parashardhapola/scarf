"""Contract tests for the decision kernel and deterministic auditor."""

import pytest
from pydantic import ValidationError

from scarf.agent.decisions.kernel import (
    DecisionEvidence,
    DecisionOption,
    DecisionRecord,
    DecisionSpec,
    DecisionWorkflowRun,
    DeterministicDecisionAuditor,
    EvidenceBundle,
    RevisionRequest,
    VerificationCheck,
    VerificationRecord,
)


def _evidence_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        bundleId="bundle:clusterPartition",
        decisionId="clusterPartition",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:silhouette",
                evidenceClass="geometric",
                summary="The coarse partition has the largest silhouette.",
            ),
            DecisionEvidence(
                evidenceId="evidence:markers",
                evidenceClass="markerCoherence",
                summary="The finer partition has distinct marker programs.",
            ),
            DecisionEvidence(
                evidenceId="evidence:stability",
                evidenceClass="resamplingStability",
                summary="The finer partition is stable under subsampling.",
            ),
            DecisionEvidence(
                evidenceId="evidence:replicates",
                evidenceClass="crossUnitSupport",
                summary="The finer populations occur in independent units.",
            ),
        ],
    )


def _decision_spec() -> DecisionSpec:
    return DecisionSpec(
        decisionId="clusterPartition",
        definitionVersion=1,
        checkpoint="clustering",
        question="Which registered partition is defensible?",
        evidenceBundleId="bundle:clusterPartition",
        options=[
            DecisionOption(
                optionId="partition:coarse",
                status="apply",
                label="Coarse partition",
                description="Use the metric-preferred coarse partition.",
            ),
            DecisionOption(
                optionId="partition:fine",
                status="apply",
                label="Fine partition",
                description="Use the finer marker-supported partition.",
                requiredEvidenceClasses=["markerCoherence"],
            ),
            DecisionOption(
                optionId="partition:abstain",
                status="abstain",
                label="No discrete partition",
                description="Do not claim that a discrete partition is supported.",
            ),
        ],
        baselineOptionId="partition:coarse",
        metricPreferredOptionId="partition:coarse",
        requireIndependentOverrideEvidence=True,
    )


def _decision_record(
    *,
    record_id: str = "decision:cluster:1",
    selected_option_id: str = "partition:coarse",
    status: str = "apply",
    source: str = "agent",
    evidence_ids: list[str] | None = None,
    override_of: str | None = None,
    override_evidence_ids: list[str] | None = None,
    available_evidence_ids: list[str] | None = None,
    supersedes: str | None = None,
) -> DecisionRecord:
    bundle = _evidence_bundle().with_content_sha256()
    assert bundle.contentSha256 is not None
    return DecisionRecord(
        recordId=record_id,
        decisionId="clusterPartition",
        definitionVersion=1,
        evidenceBundleId=bundle.bundleId,
        evidenceBundleSha256=bundle.contentSha256,
        offeredOptionIds=[
            "partition:coarse",
            "partition:fine",
            "partition:abstain",
        ],
        availableEvidenceIds=available_evidence_ids
        if available_evidence_ids is not None
        else [item.evidenceId for item in bundle.evidence],
        selectedOptionId=selected_option_id,
        status=status,
        source=source,
        evidenceIds=evidence_ids or ["evidence:silhouette"],
        rationale="The cited evidence supports this registered option.",
        confidence="medium",
        overrideOfOptionId=override_of,
        overrideEvidenceIds=override_evidence_ids or [],
        verificationId=f"verification:{record_id}",
        supersedes=supersedes,
    )


def test_option_contract_rejects_freeform_execution_parameters() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DecisionOption.model_validate(
            {
                "optionId": "pca:30",
                "status": "apply",
                "label": "Thirty PCs",
                "description": "Use the registered thirty-PC prefix.",
                "dimensions": 30,
            }
        )


def test_evidence_bundle_rejects_duplicate_evidence_ids() -> None:
    item = DecisionEvidence(
        evidenceId="evidence:qc",
        evidenceClass="qualityControl",
        summary="Observed cell quality evidence.",
    )
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        EvidenceBundle(
            bundleId="bundle:qc",
            decisionId="cellQuality",
            evidence=[item, item],
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"selectedOptionId": "partition:invented"},
            "selectedOptionId must reference an offered option",
        ),
        (
            {"evidenceIds": ["evidence:invented"]},
            "evidenceIds must reference only available evidence",
        ),
        (
            {
                "overrideOfOptionId": None,
                "evidenceIds": ["evidence:markers"],
                "overrideEvidenceIds": ["evidence:markers"],
            },
            "overrideEvidenceIds require overrideOfOptionId",
        ),
    ],
)
def test_decision_record_rejects_non_exact_references(
    changes: dict[str, object], message: str
) -> None:
    values = _decision_record().model_dump()
    values.update(changes)
    with pytest.raises(ValidationError, match=message):
        DecisionRecord.model_validate(values)


def test_auditor_accepts_an_exact_metric_preferred_decision() -> None:
    record = _decision_record()

    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(), _evidence_bundle(), record, created_at_ns=10
    )

    assert verification.status == "passed"
    assert verification.verificationId == record.verificationId
    assert {check.status for check in verification.checks} == {"passed"}


def test_auditor_rejects_a_tampered_evidence_bundle_checksum() -> None:
    values = _decision_record().model_dump()
    values["evidenceBundleSha256"] = "f" * 64
    record = DecisionRecord.model_validate(values)

    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(),
        _evidence_bundle(),
        record,
    )

    assert verification.status == "failed"
    assert [
        check.checkId for check in verification.checks if check.status == "failed"
    ] == ["decisionIdentity"]


def test_auditor_requires_evidence_bound_to_the_selected_option() -> None:
    spec_values = _decision_spec().model_dump()
    spec_values["options"][0]["requiredEvidenceIds"] = ["evidence:markers"]
    spec = DecisionSpec.model_validate(spec_values)

    verification = DeterministicDecisionAuditor.audit(
        spec,
        _evidence_bundle(),
        _decision_record(),
    )

    assert verification.status == "failed"
    assert [
        check.checkId for check in verification.checks if check.status == "failed"
    ] == ["requiredEvidence"]


def test_auditor_rejects_a_tampered_option_or_evidence_inventory() -> None:
    record = _decision_record(
        available_evidence_ids=["evidence:silhouette"],
    )
    values = record.model_dump()
    values["offeredOptionIds"] = ["partition:coarse", "partition:abstain"]
    record = DecisionRecord.model_validate(values)

    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(), _evidence_bundle(), record
    )

    assert verification.status == "failed"
    failed = {
        check.checkId for check in verification.checks if check.status == "failed"
    }
    assert failed == {"exactOptionInventory", "exactEvidenceInventory"}


def test_auditor_requires_two_independent_non_geometric_override_classes() -> None:
    record = _decision_record(
        selected_option_id="partition:fine",
        evidence_ids=["evidence:silhouette", "evidence:markers"],
        override_of="partition:coarse",
        override_evidence_ids=["evidence:silhouette", "evidence:markers"],
    )

    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(), _evidence_bundle(), record
    )

    assert verification.status == "failed"
    failed = [check for check in verification.checks if check.status == "failed"]
    assert [check.checkId for check in failed] == ["independentOverrideEvidence"]


def test_auditor_accepts_two_independent_non_geometric_override_classes() -> None:
    record = _decision_record(
        selected_option_id="partition:fine",
        evidence_ids=["evidence:markers", "evidence:stability"],
        override_of="partition:coarse",
        override_evidence_ids=["evidence:markers", "evidence:stability"],
    )

    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(), _evidence_bundle(), record
    )

    assert verification.status == "passed"


def test_auditor_rejects_override_evidence_from_another_option() -> None:
    spec_values = _decision_spec().model_dump()
    spec_values["options"][1]["requiredEvidenceIds"] = [
        "evidence:markers",
        "evidence:stability",
    ]
    spec = DecisionSpec.model_validate(spec_values)
    record = _decision_record(
        selected_option_id="partition:fine",
        evidence_ids=[
            "evidence:markers",
            "evidence:stability",
            "evidence:replicates",
        ],
        override_of="partition:coarse",
        override_evidence_ids=["evidence:markers", "evidence:replicates"],
    )

    verification = DeterministicDecisionAuditor.audit(
        spec,
        _evidence_bundle(),
        record,
    )

    assert verification.status == "failed"
    assert [
        check.checkId for check in verification.checks if check.status == "failed"
    ] == ["independentOverrideEvidence"]


def test_human_choices_obey_the_same_source_and_status_contracts() -> None:
    spec_values = _decision_spec().model_dump()
    spec_values["allowedSources"] = ["rule", "agent"]
    spec = DecisionSpec.model_validate(spec_values)
    record = _decision_record(source="human", status="skip")

    verification = DeterministicDecisionAuditor.audit(spec, _evidence_bundle(), record)

    assert verification.status == "failed"
    failed = {
        check.checkId for check in verification.checks if check.status == "failed"
    }
    assert failed == {"selectedOption", "decisionSource"}


def test_workflow_ledger_accepts_one_verified_revision_chain() -> None:
    original = _decision_record(record_id="decision:cluster:1")
    original_verification = VerificationRecord(
        verificationId="verification:decision:cluster:1",
        decisionRecordId=original.recordId,
        status="failed",
        checks=[
            VerificationCheck(
                checkId="clusterAudit",
                status="failed",
                summary="The coarse partition merges marker-supported populations.",
                evidenceIds=["evidence:markers", "evidence:stability"],
            )
        ],
    )
    revision = RevisionRequest(
        revisionId="revision:cluster:1",
        targetDecisionRecordId=original.recordId,
        verificationId=original_verification.verificationId,
        replacementOptionId="partition:fine",
        reason="Independent evidence supports the finer registered partition.",
        evidenceBundleId="bundle:cluster-revision",
        evidenceBundleSha256="1" * 64,
        availableEvidenceIds=["evidence:markers", "evidence:stability"],
        evidenceIds=["evidence:markers", "evidence:stability"],
    )
    replacement = _decision_record(
        record_id="decision:cluster:2",
        selected_option_id="partition:fine",
        evidence_ids=["evidence:markers", "evidence:stability"],
        override_of="partition:coarse",
        override_evidence_ids=["evidence:markers", "evidence:stability"],
        supersedes=original.recordId,
    )
    replacement_verification = DeterministicDecisionAuditor.audit(
        _decision_spec(), _evidence_bundle(), replacement
    )

    run = DecisionWorkflowRun(
        workflowRunId="workflow:1",
        decisionRecords=[original, replacement],
        verificationRecords=[original_verification, replacement_verification],
        revisionRequests=[revision],
    )

    assert run.formatVersion == 2
    assert run.maxRevisions == 2


def test_workflow_ledger_rejects_more_than_two_revisions() -> None:
    values = {
        "workflowRunId": "workflow:1",
        "revisionRequests": [
            {
                "revisionId": f"revision:{index}",
                "targetDecisionRecordId": "decision:target",
                "verificationId": "verification:target",
                "replacementOptionId": "option:replacement",
                "reason": "Retry a registered alternative.",
            }
            for index in range(3)
        ],
    }

    with pytest.raises(ValidationError, match="configured revision limit"):
        DecisionWorkflowRun.model_validate(values)


def test_workflow_ledger_honors_a_disabled_revision_budget() -> None:
    with pytest.raises(ValidationError, match="configured revision limit"):
        DecisionWorkflowRun(
            workflowRunId="workflow:no-revisions",
            maxRevisions=0,
            revisionRequests=[
                RevisionRequest(
                    revisionId="revision:disabled",
                    targetDecisionRecordId="decision:target",
                    verificationId="verification:target",
                    replacementOptionId="option:replacement",
                    reason="This revision should be rejected before execution.",
                )
            ],
        )


def test_workflow_ledger_rejects_upstream_revision_invalidation() -> None:
    original = _decision_record(record_id="decision:cluster:1")
    verification = VerificationRecord(
        verificationId="verification:decision:cluster:1",
        decisionRecordId=original.recordId,
        status="failed",
        checks=[
            VerificationCheck(
                checkId="clusterAudit",
                status="failed",
                summary="The partition failed its deterministic audit.",
            )
        ],
    )
    revision = RevisionRequest(
        revisionId="revision:cluster:1",
        targetDecisionRecordId=original.recordId,
        verificationId=verification.verificationId,
        replacementOptionId="partition:fine",
        reason="Use the registered alternative.",
        invalidatesDecisionRecordIds=[original.recordId],
    )

    with pytest.raises(ValidationError, match="only downstream decisions"):
        DecisionWorkflowRun(
            workflowRunId="workflow:1",
            decisionRecords=[original],
            verificationRecords=[verification],
            revisionRequests=[revision],
        )


def test_revision_replaces_target_and_recomputed_downstream_records() -> None:
    def record(
        record_id: str,
        decision_id: str,
        selected_option_id: str,
        offered_option_ids: list[str],
        *,
        supersedes: str | None = None,
    ) -> DecisionRecord:
        return DecisionRecord(
            recordId=record_id,
            decisionId=decision_id,
            definitionVersion=1,
            evidenceBundleId=f"bundle:{record_id}",
            evidenceBundleSha256="0" * 64,
            offeredOptionIds=offered_option_ids,
            availableEvidenceIds=[],
            selectedOptionId=selected_option_id,
            status="apply",
            source="agent",
            rationale="The exact registered option is supported.",
            verificationId=f"verification:{record_id}",
            supersedes=supersedes,
        )

    cell = record(
        "decision:cell:1",
        "cellQuality",
        "cell:global",
        ["cell:global"],
    )
    feature = record(
        "decision:feature:1",
        "featurePolicy",
        "feature:keep",
        ["feature:keep", "feature:exclude"],
    )
    old_hvg = record(
        "decision:hvg:1",
        "hvgCount",
        "hvg:standard",
        ["hvg:standard"],
    )
    revised_feature = record(
        "decision:feature:2",
        "featurePolicy",
        "feature:exclude",
        ["feature:keep", "feature:exclude"],
        supersedes=feature.recordId,
    )
    recomputed_hvg = record(
        "decision:hvg:2",
        "hvgCount",
        "hvg:standard",
        ["hvg:standard"],
        supersedes=old_hvg.recordId,
    )
    records = [cell, feature, old_hvg, revised_feature, recomputed_hvg]
    verifications = [
        VerificationRecord(
            verificationId=f"verification:{value.recordId}",
            decisionRecordId=value.recordId,
            status="passed",
            checks=[
                VerificationCheck(
                    checkId="exactContract",
                    status="passed",
                    summary="The exact decision contract passed.",
                )
            ],
        )
        for value in records
    ]
    revision = RevisionRequest(
        revisionId="revision:feature:1",
        targetDecisionRecordId=feature.recordId,
        verificationId=f"verification:{feature.recordId}",
        replacementOptionId="feature:exclude",
        reason="Downstream representation evidence supports the registered exclusion.",
        evidenceBundleId="bundle:feature-dominance",
        evidenceBundleSha256="1" * 64,
        availableEvidenceIds=["evidence:feature-dominance"],
        evidenceIds=["evidence:feature-dominance"],
        invalidatesDecisionRecordIds=[old_hvg.recordId],
    )

    workflow = DecisionWorkflowRun(
        workflowRunId="workflow:revision",
        decisionRecords=records,
        verificationRecords=verifications,
        revisionRequests=[revision],
    )

    assert [value.recordId for value in workflow.active_decision_records()] == [
        cell.recordId,
        revised_feature.recordId,
        recomputed_hvg.recordId,
    ]


def test_completed_workflow_requires_verified_active_decisions() -> None:
    record = _decision_record()
    verification = DeterministicDecisionAuditor.audit(
        _decision_spec(), _evidence_bundle(), record
    )

    run = DecisionWorkflowRun(
        workflowRunId="workflow:complete",
        status="completed",
        decisionRecords=[record],
        verificationRecords=[verification],
        finalHandoffId="handoff:1",
    )

    assert run.status == "completed"
    assert run.finalHandoffId == "handoff:1"


@pytest.mark.parametrize(
    ("status", "decision_status"),
    [("needsInput", "defer"), ("abstained", "abstain")],
)
def test_non_success_terminal_status_requires_matching_active_decision(
    status: str, decision_status: str
) -> None:
    values = _decision_record().model_dump()
    values["status"] = decision_status
    if decision_status == "defer":
        values["selectedOptionId"] = "partition:abstain"
    record = DecisionRecord.model_validate(values)

    run = DecisionWorkflowRun(
        workflowRunId=f"workflow:{status}",
        status=status,
        decisionRecords=[record],
    )

    assert run.status == status
