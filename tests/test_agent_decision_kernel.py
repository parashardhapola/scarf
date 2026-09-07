"""Contract tests for the decision kernel and deterministic auditor."""

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from scarf.agent.decisions.kernel import (
    DecisionEvidence,
    DecisionOption,
    DecisionRecord,
    DecisionSelection,
    DecisionSpec,
    DecisionWorkflowRun,
    DeterministicDecisionAuditor,
    EvidenceBundle,
    PendingDecision,
    ProtectedVariableEffect,
    RevisionRequest,
    VerificationCheck,
    VerificationRecord,
)
from scarf.agent.types import ArtifactReferenceModel


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


def _verification_record(
    record: DecisionRecord,
    *,
    verification_id: str | None = None,
    status: str = "passed",
) -> VerificationRecord:
    return VerificationRecord(
        verificationId=verification_id or f"verification:{record.recordId}",
        decisionRecordId=record.recordId,
        status=status,
        checks=[
            VerificationCheck(
                checkId=f"check:{record.recordId}",
                status=status,
                summary=f"The decision {status} its deterministic check.",
            )
        ],
    )


def _pending_decision() -> PendingDecision:
    return PendingDecision(
        questionId="question:cluster",
        decisionId="clusterPartition",
        definitionVersion=1,
        evidenceBundleId="bundle:clusterPartition",
        evidenceBundleSha256="0" * 64,
        offeredOptionIds=["partition:coarse", "partition:fine"],
        reason="More evidence is required.",
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


def test_evidence_contract_rejects_duplicate_artifacts_and_bad_checksums() -> None:
    reference = ArtifactReferenceModel(
        scope="assay",
        assay="RNA",
        kind="pca",
        artifactId="a" * 64,
    )
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        DecisionEvidence(
            evidenceId="evidence:pca",
            evidenceClass="geometric",
            summary="Observed PCA evidence.",
            artifactReferences=[reference, reference],
        )

    values = _evidence_bundle().model_dump()
    values["contentSha256"] = "0" * 64
    with pytest.raises(ValidationError, match="does not match"):
        EvidenceBundle.model_validate(values)
    values["contentSha256"] = "INVALID"
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        EvidenceBundle.model_validate(values)


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


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "overrideOfOptionId": "partition:fine",
                "overrideEvidenceIds": ["evidence:stability"],
            },
            "overrideEvidenceIds must be included",
        ),
        (
            {"overrideOfOptionId": "partition:invented"},
            "overrideOfOptionId must reference",
        ),
        (
            {"overrideOfOptionId": "partition:coarse"},
            "overrideOfOptionId must differ",
        ),
        (
            {"supersedes": "decision:cluster:1"},
            "cannot supersede itself",
        ),
        (
            {
                "protectedVariableEffects": [
                    ProtectedVariableEffect(
                        variable="disease",
                        status="preserved",
                        evidenceIds=["evidence:invented"],
                        summary="Disease structure is preserved.",
                    ).model_dump()
                ]
            },
            "protectedVariableEffects must reference",
        ),
    ],
)
def test_decision_record_rejects_invalid_override_and_lineage_references(
    changes: dict[str, object],
    message: str,
) -> None:
    values = _decision_record().model_dump()
    values.update(changes)
    with pytest.raises(ValidationError, match=message):
        DecisionRecord.model_validate(values)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: DecisionEvidence(
                evidenceId="invalid evidence",
                evidenceClass="technical",
                summary="Observed evidence.",
            ),
            "stable identifier",
        ),
        (
            lambda: DecisionEvidence(
                evidenceId="evidence:trim",
                evidenceClass="technical",
                summary=" surrounding whitespace ",
            ),
            "surrounding whitespace",
        ),
        (
            lambda: DecisionEvidence(
                evidenceId="evidence:artifact",
                evidenceClass="technical",
                summary="Observed evidence.",
                artifactReferences=[
                    ArtifactReferenceModel(
                        scope="assay",
                        assay="RNA",
                        kind="",
                        artifactId="a" * 64,
                    )
                ],
            ),
            "require kind and artifactId",
        ),
        (
            lambda: DecisionOption(
                optionId="option:trim",
                status="apply",
                label=" Label ",
                description="Description.",
            ),
            "surrounding whitespace",
        ),
        (
            lambda: DecisionOption(
                optionId="option:classes",
                status="apply",
                label="Label",
                description="Description.",
                requiredEvidenceClasses=["technical", "technical"],
            ),
            "must not contain duplicates",
        ),
        (
            lambda: DecisionSpec.model_validate(
                {**_decision_spec().model_dump(), "question": " Question "}
            ),
            "surrounding whitespace",
        ),
        (
            lambda: DecisionSpec.model_validate(
                {**_decision_spec().model_dump(), "allowedSources": []}
            ),
            "must not be empty",
        ),
        (
            lambda: DecisionSpec.model_validate(
                {
                    **_decision_spec().model_dump(),
                    "allowedSources": ["agent", "agent"],
                }
            ),
            "must not contain duplicates",
        ),
        (
            lambda: DecisionSpec.model_validate(
                {
                    **_decision_spec().model_dump(),
                    "baselineOptionId": "partition:unknown",
                }
            ),
            "baselineOptionId must reference",
        ),
        (
            lambda: DecisionSpec.model_validate(
                {
                    **_decision_spec().model_dump(),
                    "metricPreferredOptionId": "partition:unknown",
                }
            ),
            "metricPreferredOptionId must reference",
        ),
        (
            lambda: DecisionSpec.model_validate(
                {
                    **_decision_spec().model_dump(),
                    "metricPreferredOptionId": None,
                }
            ),
            "requires metricPreferredOptionId",
        ),
        (
            lambda: ProtectedVariableEffect(
                variable="condition",
                status="preserved",
                summary=" Whitespace ",
            ),
            "surrounding whitespace",
        ),
        (
            lambda: DecisionSelection(
                selectedOptionId="partition:coarse",
                rationale=" Whitespace ",
            ),
            "surrounding whitespace",
        ),
        (
            lambda: DecisionSelection(
                selectedOptionId="partition:coarse",
                evidenceIds=[],
                rationale="Reason.",
                overrideOfOptionId="partition:fine",
                overrideEvidenceIds=["evidence:markers"],
            ),
            "must be included in evidenceIds",
        ),
        (
            lambda: DecisionSelection(
                selectedOptionId="partition:coarse",
                evidenceIds=["evidence:markers"],
                rationale="Reason.",
                overrideEvidenceIds=["evidence:markers"],
            ),
            "require overrideOfOptionId",
        ),
        (
            lambda: DecisionSelection(
                selectedOptionId="partition:coarse",
                rationale="Reason.",
                overrideOfOptionId="partition:coarse",
            ),
            "must differ from selectedOptionId",
        ),
        (
            lambda: PendingDecision.model_validate(
                {**_pending_decision().model_dump(), "reason": " Reason "}
            ),
            "surrounding whitespace",
        ),
        (
            lambda: PendingDecision.model_validate(
                {
                    **_pending_decision().model_dump(),
                    "evidenceBundleSha256": "invalid",
                }
            ),
            "lowercase SHA-256",
        ),
        (
            lambda: DecisionRecord.model_validate(
                {**_decision_record().model_dump(), "rationale": " Reason "}
            ),
            "surrounding whitespace",
        ),
        (
            lambda: DecisionRecord.model_validate(
                {
                    **_decision_record().model_dump(),
                    "evidenceBundleSha256": "invalid",
                }
            ),
            "lowercase SHA-256",
        ),
        (
            lambda: DecisionRecord.model_validate(
                {**_decision_record().model_dump(), "modelName": " model "}
            ),
            "without surrounding whitespace",
        ),
        (
            lambda: VerificationCheck(
                checkId="check:trim",
                status="passed",
                summary=" Summary ",
            ),
            "surrounding whitespace",
        ),
        (
            lambda: RevisionRequest(
                revisionId="revision:checksum",
                targetDecisionRecordId="decision:cluster:1",
                verificationId="verification:decision:cluster:1",
                replacementOptionId="partition:fine",
                reason="Reason.",
                evidenceBundleSha256="invalid",
            ),
            "lowercase SHA-256",
        ),
        (
            lambda: RevisionRequest(
                revisionId="revision:reason",
                targetDecisionRecordId="decision:cluster:1",
                verificationId="verification:decision:cluster:1",
                replacementOptionId="partition:fine",
                reason=" Reason ",
            ),
            "surrounding whitespace",
        ),
    ],
)
def test_kernel_rejects_invalid_scalar_and_collection_contracts(
    factory: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        factory()


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


@pytest.mark.parametrize(
    ("status", "check_statuses", "message"),
    [
        ("passed", ["passed", "failed"], "every check to pass"),
        ("failed", ["passed"], "requires a failed check"),
        (
            "inconclusive",
            ["inconclusive", "failed"],
            "inconclusive check and no failures",
        ),
        (
            "inconclusive",
            ["passed"],
            "inconclusive check and no failures",
        ),
    ],
)
def test_verification_record_enforces_aggregate_status(
    status: str,
    check_statuses: list[str],
    message: str,
) -> None:
    checks = [
        VerificationCheck(
            checkId=f"check:{index}",
            status=check_status,
            summary="The check has an explicit result.",
        )
        for index, check_status in enumerate(check_statuses)
    ]
    with pytest.raises(ValidationError, match=message):
        VerificationRecord(
            verificationId="verification:aggregate",
            decisionRecordId="decision:aggregate",
            status=status,
            checks=checks,
        )


def test_verification_record_rejects_duplicate_check_ids() -> None:
    check = VerificationCheck(
        checkId="check:duplicate",
        status="passed",
        summary="The check passed.",
    )
    with pytest.raises(ValidationError, match="check IDs"):
        VerificationRecord(
            verificationId="verification:duplicate",
            decisionRecordId="decision:duplicate",
            status="passed",
            checks=[check, check],
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"evidenceBundleId": "bundle:revision"},
            "ID and checksum must be provided together",
        ),
        (
            {"availableEvidenceIds": ["evidence:new"]},
            "requires an exact evidence bundle",
        ),
        (
            {
                "evidenceBundleId": "bundle:revision",
                "evidenceBundleSha256": "1" * 64,
                "availableEvidenceIds": ["evidence:new"],
                "evidenceIds": ["evidence:missing"],
            },
            "reference its exact available inventory",
        ),
    ],
)
def test_revision_request_rejects_incomplete_evidence_references(
    changes: dict[str, object],
    message: str,
) -> None:
    values = {
        "revisionId": "revision:cluster",
        "targetDecisionRecordId": "decision:cluster:1",
        "verificationId": "verification:decision:cluster:1",
        "replacementOptionId": "partition:fine",
        "reason": "Use the registered alternative.",
        **changes,
    }
    with pytest.raises(ValidationError, match=message):
        RevisionRequest.model_validate(values)


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


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicateRecord", "unique recordId"),
        ("missingSupersedes", "must supersede the current active"),
        ("unexpectedSupersedes", "must reference an earlier matching decision"),
        ("duplicateVerification", "unique verificationId"),
        ("unknownVerificationRecord", "exact decision record"),
        ("secondVerification", "only one verification"),
        ("mismatchedVerification", "references must agree exactly"),
    ],
)
def test_workflow_ledger_rejects_invalid_record_and_verification_topology(
    case: str,
    message: str,
) -> None:
    first = _decision_record(record_id="decision:cluster:1")
    second = _decision_record(record_id="decision:cluster:2")
    records = [first]
    verifications: list[VerificationRecord] = []

    if case == "duplicateRecord":
        records.append(first)
    elif case == "missingSupersedes":
        records.append(second)
    elif case == "unexpectedSupersedes":
        records = [
            _decision_record(
                record_id="decision:cluster:2",
                supersedes="decision:cluster:missing",
            )
        ]
    elif case == "duplicateVerification":
        other = second.model_copy(
            update={
                "decisionId": "featurePolicy",
                "verificationId": first.verificationId,
            }
        )
        records.append(other)
        verifications = [
            _verification_record(first),
            _verification_record(
                other,
                verification_id=first.verificationId,
            ),
        ]
    elif case == "unknownVerificationRecord":
        verification = _verification_record(first).model_copy(
            update={"decisionRecordId": "decision:missing"}
        )
        verifications = [verification]
    elif case == "secondVerification":
        verifications = [
            _verification_record(first),
            _verification_record(
                first,
                verification_id="verification:decision:cluster:other",
            ),
        ]
    elif case == "mismatchedVerification":
        verifications = [
            _verification_record(
                first,
                verification_id="verification:decision:cluster:other",
            )
        ]

    with pytest.raises(ValidationError, match=message):
        DecisionWorkflowRun(
            workflowRunId="workflow:invalid-topology",
            decisionRecords=records,
            verificationRecords=verifications,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicateRevision", "unique revisionId"),
        ("duplicateTarget", "may be revised only once"),
        ("unknownTarget", "exact decision record"),
        ("wrongVerification", "target decision's verification"),
        ("passedWithoutEvidence", "requires exact downstream evidence"),
        ("unchangedOption", "must change the selected option"),
        ("bundleDrift", "match the target bundle exactly"),
        ("unknownInvalidation", "exact decision record"),
        ("supersedingWithoutRevision", "requires a revision request"),
        ("replacementMismatch", "select the requested replacement"),
    ],
)
def test_workflow_ledger_rejects_invalid_revision_references(
    case: str,
    message: str,
) -> None:
    original = _decision_record(record_id="decision:cluster:1")
    failed_verification = _verification_record(original, status="failed")
    revision = RevisionRequest(
        revisionId="revision:cluster:1",
        targetDecisionRecordId=original.recordId,
        verificationId=failed_verification.verificationId,
        replacementOptionId="partition:fine",
        reason="Use the registered alternative.",
    )
    replacement = _decision_record(
        record_id="decision:cluster:2",
        selected_option_id="partition:fine",
        supersedes=original.recordId,
    )
    records = [original]
    verifications = [failed_verification]
    revisions = [revision]

    if case == "duplicateRevision":
        revisions.append(revision)
    elif case == "duplicateTarget":
        revisions.append(
            revision.model_copy(update={"revisionId": "revision:cluster:2"})
        )
    elif case == "unknownTarget":
        revisions = [
            revision.model_copy(update={"targetDecisionRecordId": "decision:missing"})
        ]
    elif case == "wrongVerification":
        revisions = [
            revision.model_copy(update={"verificationId": "verification:missing"})
        ]
    elif case == "passedWithoutEvidence":
        verifications = [_verification_record(original)]
    elif case == "unchangedOption":
        revisions = [
            revision.model_copy(
                update={"replacementOptionId": original.selectedOptionId}
            )
        ]
    elif case == "bundleDrift":
        revisions = [
            revision.model_copy(
                update={
                    "evidenceBundleId": original.evidenceBundleId,
                    "evidenceBundleSha256": "1" * 64,
                }
            )
        ]
    elif case == "unknownInvalidation":
        revisions = [
            revision.model_copy(
                update={"invalidatesDecisionRecordIds": ["decision:missing"]}
            )
        ]
    elif case == "supersedingWithoutRevision":
        records.append(replacement)
        revisions = []
    elif case == "replacementMismatch":
        records.append(
            replacement.model_copy(
                update={
                    "selectedOptionId": "partition:abstain",
                    "status": "abstain",
                }
            )
        )

    with pytest.raises(ValidationError, match=message):
        DecisionWorkflowRun(
            workflowRunId="workflow:invalid-revision",
            decisionRecords=records,
            verificationRecords=verifications,
            revisionRequests=revisions,
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
    ("case", "message"),
    [
        ("completedWithoutHandoff", "require finalHandoffId"),
        ("completedWithPending", "cannot contain a pending decision"),
        ("completedUnverified", "every active decision to pass"),
        ("runningWithHandoff", "Only completed workflows"),
        ("needsInputWithoutPause", "require a pending or active defer"),
        ("runningWithPending", "Only needsInput workflows"),
        ("abstainedWithoutDecision", "require an active abstain"),
    ],
)
def test_workflow_terminal_status_requires_matching_ledger_state(
    case: str,
    message: str,
) -> None:
    record = _decision_record()
    values: dict[str, object] = {
        "workflowRunId": "workflow:terminal",
        "status": "running",
        "decisionRecords": [record],
        "verificationRecords": [_verification_record(record)],
    }
    if case == "completedWithoutHandoff":
        values["status"] = "completed"
    elif case == "completedWithPending":
        values.update(
            status="completed",
            finalHandoffId="handoff:1",
            pendingDecision=_pending_decision(),
        )
    elif case == "completedUnverified":
        values.update(
            status="completed",
            finalHandoffId="handoff:1",
            verificationRecords=[],
        )
    elif case == "runningWithHandoff":
        values["finalHandoffId"] = "handoff:1"
    elif case == "needsInputWithoutPause":
        values["status"] = "needsInput"
    elif case == "runningWithPending":
        values["pendingDecision"] = _pending_decision()
    elif case == "abstainedWithoutDecision":
        values["status"] = "abstained"

    with pytest.raises(ValidationError, match=message):
        DecisionWorkflowRun.model_validate(values)


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
