"""Persistence tests for immutable decision-workflow snapshots."""

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
import zarr
from pydantic_ai.exceptions import AgentRunError
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

import scarf.agent.persistence.decisions as persistence_module
import scarf.agent.orchestrator.decisions as decisions_module
from scarf.agent import record_io
from scarf.agent.decisions.kernel import (
    DecisionEvidence,
    DecisionRecord,
    DecisionSelection,
    DecisionWorkflowRun,
    EvidenceBundle,
    VerificationCheck,
    VerificationRecord,
)
from scarf.agent.persistence.decisions import (
    DecisionPersistenceFormatError,
    DecisionWorkflowSnapshot,
    attach_audited_rna_decision,
    decision_record_checksum,
    list_decision_workflow_snapshots,
    load_decision_workflow_for_replay,
    load_decision_workflow_snapshot,
    load_latest_decision_workflow_snapshot,
    save_decision_workflow_snapshot,
)
from scarf.agent.orchestrator.models import (
    _ORCHESTRATION_FORMAT,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    OrchestrationRequestRecord,
)
from scarf.agent.orchestrator.decisions import DecisionStagesMixin
from scarf.agent.decisions.rna import (
    build_cell_quality_decision,
    build_feature_policy_decision,
    build_pca_prefix_decision,
    build_qc_grouping_decision,
    compile_rna_decision,
)
from tests.agent_orchestrator_store import create_store


def _set_raw(group: zarr.Group, key: str, payload: bytes) -> None:
    buffer = default_buffer_prototype().buffer.from_bytes(payload)
    sync(group.store.set(key, buffer))


def _orchestration_request_record(
    path: Any,
    workflow_run_id: str,
) -> OrchestrationRequestRecord:
    request = AutomatedWorkflowRequest(
        sourcePath=str(path),
        zarrPath=str(path),
        studyContext="A test single-cell study.",
        studyObjective="Discover stable RNA populations.",
    )
    config = AutomatedWorkflowConfig()
    record = OrchestrationRequestRecord(
        workflowRunId=workflow_run_id,
        createdAtNs=1,
        request=request,
        config=config,
        requestSha256=persistence_module._model_checksum(request),
        configSha256=persistence_module._model_checksum(config),
    )
    return record.model_copy(
        update={"contentSha256": persistence_module._record_checksum(record)}
    )


def _seed_orchestration(
    path: Any,
    workflow_run_id: str,
    *,
    format_version: int = 2,
) -> zarr.Group:
    create_store(path)
    group = zarr.open_group(str(path), mode="r+")
    agents = group.create_group(
        "agents",
        attributes={"format": "scarf_agent_reports", "format_version": 2},
    )
    agents.create_group(
        "orchestrations",
        attributes={
            "format": _ORCHESTRATION_FORMAT,
            "format_version": format_version,
        },
    )
    record = _orchestration_request_record(path, workflow_run_id)
    _set_raw(
        group,
        f"agents/orchestrations/{workflow_run_id}/request.json",
        record_io.display_json_bytes(record.model_dump(mode="json")),
    )
    return group


def _decision_record(
    workflow_run_id: str,
    *,
    status: str,
    verification: bool,
) -> tuple[DecisionRecord, VerificationRecord | None]:
    option_id = f"option:{status}"
    verification_id = f"verification:{workflow_run_id}:1" if verification else None
    record = DecisionRecord(
        recordId=f"decision:{workflow_run_id}:1",
        decisionId="cellQuality",
        definitionVersion=1,
        evidenceBundleId="bundle:test",
        evidenceBundleSha256="0" * 64,
        offeredOptionIds=[option_id],
        availableEvidenceIds=[],
        selectedOptionId=option_id,
        status=status,
        source="rule",
        rationale="Use the exact registered test outcome.",
        verificationId=verification_id,
    )
    if not verification:
        return record, None
    return record, VerificationRecord(
        verificationId=verification_id,
        decisionRecordId=record.recordId,
        status="passed",
        checks=[
            VerificationCheck(
                checkId="exactContract",
                status="passed",
                summary="The registered test contract is exact.",
            )
        ],
    )


def _workflow_for_status(workflow_run_id: str, status: str) -> DecisionWorkflowRun:
    if status == "running":
        return DecisionWorkflowRun(workflowRunId=workflow_run_id)
    decision_status = "defer" if status == "needsInput" else "abstain"
    if status == "completed":
        decision_status = "apply"
    record, verification = _decision_record(
        workflow_run_id,
        status=decision_status,
        verification=status == "completed",
    )
    return DecisionWorkflowRun(
        workflowRunId=workflow_run_id,
        status=status,
        decisionRecords=[record],
        verificationRecords=[verification] if verification is not None else [],
        finalHandoffId=f"handoff-{workflow_run_id}" if status == "completed" else None,
    )


def _compiled_cell_quality(
    workflow_run_id: str,
) -> tuple[DecisionRecord, Any]:
    definition = build_cell_quality_decision(
        evidence_bundle_id="bundle:cellQuality",
        available_profiles=["retainWithFlags", "globalMad5"],
    )
    bundle = EvidenceBundle(
        bundleId="bundle:cellQuality",
        decisionId="cellQuality",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:quality",
                evidenceClass="qualityControl",
                summary="The published cell set passes lenient quality review.",
            )
        ],
    ).with_content_sha256()
    assert bundle.contentSha256 is not None
    record = DecisionRecord(
        recordId=f"decision:{workflow_run_id}:cellQuality",
        decisionId="cellQuality",
        definitionVersion=1,
        evidenceBundleId=bundle.bundleId,
        evidenceBundleSha256=bundle.contentSha256,
        offeredOptionIds=[option.optionId for option in definition.spec.options],
        availableEvidenceIds=["evidence:quality"],
        selectedOptionId="cellQuality:retainWithFlags",
        status="skip",
        source="agent",
        evidenceIds=["evidence:quality"],
        rationale="The published cells do not need another destructive filter.",
        verificationId=f"verification:decision:{workflow_run_id}:cellQuality",
    )
    return record, compile_rna_decision(definition, bundle, record)


def _compiled_qc_grouping(
    workflow_run_id: str,
) -> tuple[DecisionRecord, Any]:
    definition = build_qc_grouping_decision(
        evidence_bundle_id="bundle:qcGrouping",
        physical_capture_eligible=False,
        pooled_reference_eligible=False,
    )
    bundle = EvidenceBundle(
        bundleId="bundle:qcGrouping",
        decisionId="qcGrouping",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:quality",
                evidenceClass="qualityControl",
                summary="Global quality-reference evidence is available.",
            ),
            DecisionEvidence(
                evidenceId="evidence:design",
                evidenceClass="design",
                summary="No physical capture is registered.",
            ),
        ],
    ).with_content_sha256()
    assert bundle.contentSha256 is not None
    record = DecisionRecord(
        recordId=f"decision:{workflow_run_id}:qcGrouping",
        decisionId="qcGrouping",
        definitionVersion=1,
        evidenceBundleId=bundle.bundleId,
        evidenceBundleSha256=bundle.contentSha256,
        offeredOptionIds=[option.optionId for option in definition.spec.options],
        availableEvidenceIds=[item.evidenceId for item in bundle.evidence],
        selectedOptionId="qcGrouping:global",
        status="apply",
        source="agent",
        evidenceIds=[item.evidenceId for item in bundle.evidence],
        rationale="Use a global quality reference without proven captures.",
        verificationId=f"verification:decision:{workflow_run_id}:qcGrouping",
    )
    return record, compile_rna_decision(definition, bundle, record)


@pytest.mark.parametrize("status", ["running", "needsInput", "abstained", "completed"])
def test_snapshot_round_trip_supports_workflow_statuses(
    tmp_path: Any, status: str
) -> None:
    workflow_run_id = f"workflow-{status.lower()}"
    path = tmp_path / f"{status}.zarr"
    _seed_orchestration(path, workflow_run_id)
    workflow = _workflow_for_status(workflow_run_id, status)

    saved = save_decision_workflow_snapshot(path, workflow, created_at_ns=10)
    loaded = load_decision_workflow_snapshot(
        path,
        workflow_run_id,
        saved.contentSha256,
    )

    assert loaded == saved
    assert loaded.workflow.status == status
    assert loaded.orchestrationRun.workflowRunId == workflow_run_id
    assert loaded.orchestrationRun.requestContentSha256


def test_snapshots_are_content_addressed_append_only_and_idempotent(
    tmp_path: Any,
) -> None:
    workflow_run_id = "workflow-chain"
    path = tmp_path / "chain.zarr"
    _seed_orchestration(path, workflow_run_id)
    initial = DecisionWorkflowRun(workflowRunId=workflow_run_id)
    first = save_decision_workflow_snapshot(path, initial, created_at_ns=10)
    record, compiled = _compiled_qc_grouping(workflow_run_id)
    advanced = attach_audited_rna_decision(initial, record, compiled)
    second = save_decision_workflow_snapshot(path, advanced, created_at_ns=20)

    snapshots = list_decision_workflow_snapshots(path, workflow_run_id)
    assert [snapshot.sequence for snapshot in snapshots] == [0, 1]
    assert second.parentContentSha256 == first.contentSha256
    assert load_latest_decision_workflow_snapshot(path, workflow_run_id) == second
    assert (
        load_decision_workflow_snapshot(path, workflow_run_id, first.contentSha256)
        == first
    )

    retried = save_decision_workflow_snapshot(path, advanced, created_at_ns=30)
    assert retried == second
    assert len(list_decision_workflow_snapshots(path, workflow_run_id)) == 2


def test_decision_record_checksum_is_canonical_and_content_sensitive() -> None:
    record, _verification = _decision_record(
        "workflow-checksum", status="apply", verification=True
    )
    expected = hashlib.sha256(
        record_io.canonical_json_bytes(record.model_dump(mode="json"))
    ).hexdigest()

    assert decision_record_checksum(record) == expected
    changed = record.model_copy(update={"rationale": "A different rationale."})
    assert decision_record_checksum(changed) != expected


def test_exact_load_rejects_tampered_snapshot_content(tmp_path: Any) -> None:
    workflow_run_id = "workflow-tampered"
    path = tmp_path / "tampered.zarr"
    group = _seed_orchestration(path, workflow_run_id)
    snapshot = save_decision_workflow_snapshot(
        path,
        DecisionWorkflowRun(workflowRunId=workflow_run_id),
        created_at_ns=10,
    )
    key = persistence_module._snapshot_key(
        "agents/orchestrations",
        workflow_run_id,
        snapshot.contentSha256,
    )
    raw = record_io.read_key(group, key)
    assert raw is not None
    payload = json.loads(raw)
    payload["createdAtNs"] = 11
    _set_raw(group, key, record_io.display_json_bytes(payload))

    with pytest.raises(ValueError, match="does not match its content"):
        load_decision_workflow_snapshot(path, workflow_run_id, snapshot.contentSha256)


def test_latest_load_rejects_a_gap_or_fork_in_the_chain(tmp_path: Any) -> None:
    workflow_run_id = "workflow-gap"
    path = tmp_path / "gap.zarr"
    group = _seed_orchestration(path, workflow_run_id)
    first = save_decision_workflow_snapshot(
        path,
        DecisionWorkflowRun(workflowRunId=workflow_run_id),
        created_at_ns=10,
    )
    values = {
        "sequence": 2,
        "createdAtNs": 20,
        "parentContentSha256": first.contentSha256,
        "orchestrationRun": first.orchestrationRun,
        "workflow": first.workflow,
        "contentSha256": "0" * 64,
    }
    unhashed = DecisionWorkflowSnapshot.model_validate(values)
    values["contentSha256"] = persistence_module._snapshot_checksum(unhashed)
    orphan = DecisionWorkflowSnapshot.model_validate(values)
    key = persistence_module._snapshot_key(
        "agents/orchestrations", workflow_run_id, orphan.contentSha256
    )
    _set_raw(
        group,
        key,
        record_io.display_json_bytes(orphan.model_dump(mode="json")),
    )

    with pytest.raises(ValueError, match="gap or fork"):
        load_latest_decision_workflow_snapshot(path, workflow_run_id)


def test_unknown_or_old_formats_fail_with_actionable_rerun_message(
    tmp_path: Any,
) -> None:
    workflow_run_id = "workflow-old"
    old_path = tmp_path / "old.zarr"
    _seed_orchestration(old_path, workflow_run_id, format_version=1)

    with pytest.raises(
        DecisionPersistenceFormatError, match="Start a new orchestration run"
    ):
        save_decision_workflow_snapshot(
            old_path,
            DecisionWorkflowRun(workflowRunId=workflow_run_id),
            created_at_ns=10,
        )

    current_path = tmp_path / "unknown-snapshot.zarr"
    group = _seed_orchestration(current_path, workflow_run_id)
    snapshot = save_decision_workflow_snapshot(
        current_path,
        DecisionWorkflowRun(workflowRunId=workflow_run_id),
        created_at_ns=10,
    )
    key = persistence_module._snapshot_key(
        "agents/orchestrations", workflow_run_id, snapshot.contentSha256
    )
    raw = record_io.read_key(group, key)
    assert raw is not None
    payload = json.loads(raw)
    payload["formatVersion"] = 1
    _set_raw(group, key, record_io.display_json_bytes(payload))

    with pytest.raises(
        DecisionPersistenceFormatError, match="Start a new orchestration run"
    ):
        load_decision_workflow_snapshot(
            current_path, workflow_run_id, snapshot.contentSha256
        )


def test_snapshot_requires_exact_orchestration_run_identity(tmp_path: Any) -> None:
    path = tmp_path / "identity.zarr"
    _seed_orchestration(path, "workflow-identity")

    with pytest.raises(KeyError, match="Unknown orchestration run"):
        save_decision_workflow_snapshot(
            path,
            DecisionWorkflowRun(workflowRunId="workflow-other"),
            created_at_ns=10,
        )


def test_replay_requires_exact_completed_snapshot_and_handoff(tmp_path: Any) -> None:
    workflow_run_id = "workflow-replay"
    path = tmp_path / "replay.zarr"
    _seed_orchestration(path, workflow_run_id)
    completed = _workflow_for_status(workflow_run_id, "completed")
    snapshot = save_decision_workflow_snapshot(path, completed, created_at_ns=10)

    replay = load_decision_workflow_for_replay(
        path,
        workflow_run_id,
        snapshot.contentSha256,
        expected_handoff_id=completed.finalHandoffId,
    )
    assert replay == completed
    with pytest.raises(ValueError, match="final handoff identity"):
        load_decision_workflow_for_replay(
            path,
            workflow_run_id,
            snapshot.contentSha256,
            expected_handoff_id="handoff-other",
        )

    running_id = "workflow-running-replay"
    running_path = tmp_path / "running-replay.zarr"
    _seed_orchestration(running_path, running_id)
    running = save_decision_workflow_snapshot(
        running_path,
        DecisionWorkflowRun(workflowRunId=running_id),
        created_at_ns=10,
    )
    with pytest.raises(RuntimeError, match="completed decision workflow"):
        load_decision_workflow_for_replay(
            running_path, running_id, running.contentSha256
        )


def test_builder_attaches_only_exact_audited_transition_order() -> None:
    workflow_run_id = "workflow-builder"
    workflow = DecisionWorkflowRun(workflowRunId=workflow_run_id)
    grouping_record, grouping_compiled = _compiled_qc_grouping(workflow_run_id)
    after_grouping = attach_audited_rna_decision(
        workflow,
        grouping_record,
        grouping_compiled,
    )
    cell_record, cell_compiled = _compiled_cell_quality(workflow_run_id)

    after_cell = attach_audited_rna_decision(
        after_grouping,
        cell_record,
        cell_compiled,
    )
    assert after_cell.decisionRecords == [grouping_record, cell_record]
    assert after_cell.verificationRecords == [
        grouping_compiled.verification,
        cell_compiled.verification,
    ]

    pca = build_pca_prefix_decision(evidence_bundle_id="bundle:pca", matrix_rank=50)
    pca_bundle = EvidenceBundle(
        bundleId="bundle:pca",
        decisionId="pcaPrefix",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:geometry",
                evidenceClass="geometric",
                summary="The standard prefix has stable neighbors.",
            ),
            DecisionEvidence(
                evidenceId="evidence:technical",
                evidenceClass="technical",
                summary="The standard prefix is not dominated by technical loadings.",
            ),
        ],
    ).with_content_sha256()
    assert pca_bundle.contentSha256 is not None
    pca_record = DecisionRecord(
        recordId="decision:workflow-builder:pca",
        decisionId="pcaPrefix",
        definitionVersion=1,
        evidenceBundleId=pca_bundle.bundleId,
        evidenceBundleSha256=pca_bundle.contentSha256,
        offeredOptionIds=[option.optionId for option in pca.spec.options],
        availableEvidenceIds=[item.evidenceId for item in pca_bundle.evidence],
        selectedOptionId="pcaPrefix:standard",
        status="apply",
        source="agent",
        evidenceIds=[item.evidenceId for item in pca_bundle.evidence],
        rationale="The standard prefix is the smallest stable registered option.",
        verificationId="verification:decision:workflow-builder:pca",
    )
    pca_compiled = compile_rna_decision(pca, pca_bundle, pca_record)
    with pytest.raises(ValueError, match="transition order"):
        attach_audited_rna_decision(after_cell, pca_record, pca_compiled)

    features = build_feature_policy_decision(
        evidence_bundle_id="bundle:features",
        proposed_exclusion_families=[],
        dominant_families=[],
        protected_families=[],
    )
    feature_bundle = EvidenceBundle(
        bundleId="bundle:features",
        decisionId="featurePolicy",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:technical",
                evidenceClass="technical",
                summary="No conditional family dominates the representation.",
            )
        ],
    ).with_content_sha256()
    assert feature_bundle.contentSha256 is not None
    feature_record = DecisionRecord(
        recordId="decision:workflow-builder:features",
        decisionId="featurePolicy",
        definitionVersion=1,
        evidenceBundleId=feature_bundle.bundleId,
        evidenceBundleSha256=feature_bundle.contentSha256,
        offeredOptionIds=[option.optionId for option in features.spec.options],
        availableEvidenceIds=["evidence:technical"],
        selectedOptionId="featurePolicy:keepAll",
        status="skip",
        source="agent",
        evidenceIds=["evidence:technical"],
        rationale="No eligible nuisance bundle is supported.",
        verificationId="verification:decision:workflow-builder:features",
    )
    feature_compiled = compile_rna_decision(features, feature_bundle, feature_record)
    after_features = attach_audited_rna_decision(
        after_cell, feature_record, feature_compiled
    )
    assert [record.decisionId for record in after_features.decisionRecords] == [
        "qcGrouping",
        "cellQuality",
        "featurePolicy",
    ]


def test_snapshot_storage_does_not_overwrite_existing_content(tmp_path: Any) -> None:
    workflow_run_id = "workflow-no-overwrite"
    path = tmp_path / "no-overwrite.zarr"
    group = _seed_orchestration(path, workflow_run_id)
    snapshot = save_decision_workflow_snapshot(
        path,
        DecisionWorkflowRun(workflowRunId=workflow_run_id),
        created_at_ns=10,
    )
    key = persistence_module._snapshot_key(
        "agents/orchestrations", workflow_run_id, snapshot.contentSha256
    )

    with pytest.raises(FileExistsError, match="already exists"):
        persistence_module._write_key_once(group, key, b"different")

    assert record_io.read_key(group, key) == record_io.display_json_bytes(
        snapshot.model_dump(mode="json")
    )


def test_selection_validator_rejects_ineligible_override_fields() -> None:
    definition = build_feature_policy_decision(
        evidence_bundle_id="bundle:features",
        proposed_exclusion_families=["ribosomal"],
        dominant_families=["ribosomal"],
        protected_families=[],
    )
    evidence = EvidenceBundle(
        bundleId="bundle:features",
        decisionId="featurePolicy",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:technical",
                evidenceClass="technical",
                summary="Ribosomal features dominate the representation.",
            )
        ],
    )
    selection = DecisionSelection(
        selectedOptionId="featurePolicy:excludeEligibleBundle",
        evidenceIds=["evidence:technical"],
        rationale="Exclude the eligible family.",
        overrideOfOptionId="featurePolicy:keepAll",
        overrideEvidenceIds=["evidence:technical"],
    )

    with pytest.raises(
        ValueError,
        match="Override fields require an eligible metric-preferred override",
    ):
        decisions_module._validate_selection(definition, evidence, selection)


def test_resolver_replays_an_exact_audited_decision_without_provider(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_run_id = "workflow-decision-replay"
    path = tmp_path / "decision-replay.zarr"
    _seed_orchestration(path, workflow_run_id)
    request_record = _orchestration_request_record(path, workflow_run_id)
    evidence = EvidenceBundle(
        bundleId="bundle:qc-grouping",
        decisionId="qcGrouping",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:quality",
                evidenceClass="qualityControl",
                summary="Global quality-reference evidence is available.",
            ),
            DecisionEvidence(
                evidenceId="evidence:design",
                evidenceClass="design",
                summary="No physical capture is registered.",
            ),
        ],
    )
    definition = build_qc_grouping_decision(
        evidence_bundle_id=evidence.bundleId,
        physical_capture_eligible=False,
        pooled_reference_eligible=False,
    )
    calls = 0

    def select_once(**_kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            output=DecisionSelection(
                selectedOptionId="qcGrouping:global",
                evidenceIds=["evidence:quality", "evidence:design"],
                rationale="The registered global reference is supported.",
            ),
            runInfo=SimpleNamespace(modelName="test-model"),
        )

    monkeypatch.setattr(decisions_module, "run_agent_sync", select_once)
    resolver = DecisionStagesMixin()
    resolver.model = object()

    first = resolver._resolve_rna_decision(
        path,
        request_record,
        definition,
        evidence,
        {},
    )
    second = resolver._resolve_rna_decision(
        path,
        request_record,
        definition,
        evidence,
        {},
    )

    assert calls == 1
    assert first.record is not None
    assert second.record == first.record
    assert second.compiled == first.compiled


def test_agent_reconsideration_revises_and_recomputes_invalidated_descendant(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_run_id = "workflow-decision-revision"
    path = tmp_path / "decision-revision.zarr"
    _seed_orchestration(path, workflow_run_id)
    request_record = _orchestration_request_record(path, workflow_run_id)
    resolver = DecisionStagesMixin()
    resolver.model = object()

    grouping_evidence = EvidenceBundle(
        bundleId="bundle:qc-grouping:initial",
        decisionId="qcGrouping",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:quality:initial",
                evidenceClass="qualityControl",
                summary="Initial quality evidence supports a global reference.",
            ),
            DecisionEvidence(
                evidenceId="evidence:design:initial",
                evidenceClass="design",
                summary="No physical capture was initially licensed.",
            ),
        ],
    )
    grouping_definition = build_qc_grouping_decision(
        evidence_bundle_id=grouping_evidence.bundleId,
        physical_capture_eligible=False,
        pooled_reference_eligible=False,
    )
    initial_grouping = resolver._resolve_rna_decision(
        path,
        request_record,
        grouping_definition,
        grouping_evidence,
        {},
        rule_selection=DecisionSelection(
            selectedOptionId="qcGrouping:global",
            evidenceIds=[
                "evidence:quality:initial",
                "evidence:design:initial",
            ],
            rationale="Use the only licensed global reference.",
        ),
    )
    assert initial_grouping.record is not None

    cell_evidence = EvidenceBundle(
        bundleId="bundle:cell-quality",
        decisionId="cellQuality",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:cell-quality",
                evidenceClass="qualityControl",
                summary="The published cells can be retained with diagnostic flags.",
            )
        ],
    )
    cell_definition = build_cell_quality_decision(
        evidence_bundle_id=cell_evidence.bundleId,
        available_profiles=["retainWithFlags", "globalMad5"],
    )
    initial_cell = resolver._resolve_rna_decision(
        path,
        request_record,
        cell_definition,
        cell_evidence,
        {},
        rule_selection=DecisionSelection(
            selectedOptionId="cellQuality:retainWithFlags",
            evidenceIds=["evidence:cell-quality"],
            rationale="Retain the initial cells with diagnostic flags.",
        ),
    )
    assert initial_cell.record is not None

    revision_evidence = EvidenceBundle(
        bundleId="bundle:qc-grouping:revision",
        decisionId="qcGrouping",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:quality:revision",
                evidenceClass="qualityControl",
                summary="Capture-level projections are now available.",
            ),
            DecisionEvidence(
                evidenceId="evidence:design:revision",
                evidenceClass="design",
                summary="Physical captures are now explicitly registered.",
            ),
        ],
    ).with_content_sha256()
    assert revision_evidence.contentSha256 is not None
    revision_definition = build_qc_grouping_decision(
        evidence_bundle_id=revision_evidence.bundleId,
        physical_capture_eligible=True,
        pooled_reference_eligible=False,
    )
    monkeypatch.setattr(
        decisions_module,
        "run_agent_sync",
        lambda **_kwargs: SimpleNamespace(
            output=DecisionSelection(
                selectedOptionId="qcGrouping:physicalCapture",
                evidenceIds=[
                    "evidence:quality:revision",
                    "evidence:design:revision",
                ],
                rationale="Use the newly licensed physical-capture references.",
            ),
            runInfo=SimpleNamespace(modelName="test-model"),
        ),
    )
    reconsidered = resolver._reconsider_rna_decision(
        path,
        request_record,
        revision_definition,
        revision_evidence,
        {},
    )
    assert reconsidered.revised is True
    assert reconsidered.resolution is not None
    revised_grouping = reconsidered.resolution
    revision = revised_grouping.workflow.revisionRequests[0]
    assert revised_grouping.record is not None
    assert revised_grouping.record.supersedes == initial_grouping.record.recordId
    assert revised_grouping.workflow.revisionRequests == [revision]
    assert [
        record.recordId
        for record in revised_grouping.workflow.active_decision_records()
    ] == [revised_grouping.record.recordId]

    revised_cell_definition = build_cell_quality_decision(
        evidence_bundle_id=cell_evidence.bundleId,
        available_profiles=["retainWithFlags", "captureMad5"],
    )
    recomputed_cell = resolver._resolve_rna_decision(
        path,
        request_record,
        revised_cell_definition,
        cell_evidence,
        {},
        rule_selection=DecisionSelection(
            selectedOptionId="cellQuality:captureMad5",
            evidenceIds=["evidence:cell-quality"],
            rationale="Recompute cell quality within the registered captures.",
        ),
    )
    assert recomputed_cell.record is not None
    assert recomputed_cell.record.supersedes == initial_cell.record.recordId
    assert [
        record.recordId for record in recomputed_cell.workflow.active_decision_records()
    ] == [revised_grouping.record.recordId, recomputed_cell.record.recordId]


def test_resolver_persists_pending_state_after_model_failure(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_run_id = "workflow-decision-failure"
    path = tmp_path / "decision-failure.zarr"
    _seed_orchestration(path, workflow_run_id)
    request_record = _orchestration_request_record(path, workflow_run_id)
    evidence = EvidenceBundle(
        bundleId="bundle:qc-grouping",
        decisionId="qcGrouping",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:quality",
                evidenceClass="qualityControl",
                summary="Registered cell-quality projections are available.",
            ),
            DecisionEvidence(
                evidenceId="evidence:design",
                evidenceClass="design",
                summary="No physical capture is registered.",
            ),
        ],
    )
    definition = build_qc_grouping_decision(
        evidence_bundle_id=evidence.bundleId,
        physical_capture_eligible=False,
        pooled_reference_eligible=False,
    )

    def fail_model(**_kwargs: Any) -> None:
        raise AgentRunError("bounded model failure")

    monkeypatch.setattr(decisions_module, "run_agent_sync", fail_model)
    resolver = DecisionStagesMixin()
    resolver.model = object()

    resolution = resolver._resolve_rna_decision(
        path,
        request_record,
        definition,
        evidence,
        {},
    )

    assert resolution.compiled is None
    assert resolution.record is None
    assert resolution.workflow.status == "needsInput"
    assert resolution.pending is not None
    persisted = load_latest_decision_workflow_snapshot(path, workflow_run_id)
    assert persisted.workflow.pendingDecision == resolution.pending

    resumed = resolver._resolve_rna_decision(
        path,
        request_record,
        definition,
        evidence,
        {
            "decision:qcGrouping": {
                "decisionId": "qcGrouping",
                "optionId": "qcGrouping:global",
                "rationale": "Use the completed registered grouping evidence.",
            }
        },
    )

    assert resumed.workflow.status == "running"
    assert resumed.workflow.pendingDecision is None
    assert resumed.record is not None
    assert resumed.record.source == "human"


def test_unattended_resolver_uses_registered_baseline_after_model_failure(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_run_id = "workflow-unattended-decision-failure"
    path = tmp_path / "unattended-decision-failure.zarr"
    _seed_orchestration(path, workflow_run_id)
    request_record = _orchestration_request_record(
        path,
        workflow_run_id,
    ).model_copy(
        update={
            "config": AutomatedWorkflowConfig(inputPolicy="unattended"),
        }
    )
    evidence = EvidenceBundle(
        bundleId="bundle:qc-grouping",
        decisionId="qcGrouping",
        evidence=[
            DecisionEvidence(
                evidenceId="evidence:quality",
                evidenceClass="qualityControl",
                summary="Registered cell-quality projections are available.",
            ),
            DecisionEvidence(
                evidenceId="evidence:design",
                evidenceClass="design",
                summary="No physical capture is registered.",
            ),
        ],
    )
    definition = build_qc_grouping_decision(
        evidence_bundle_id=evidence.bundleId,
        physical_capture_eligible=False,
        pooled_reference_eligible=False,
    )

    def fail_model(**_kwargs: Any) -> None:
        raise AgentRunError("bounded model failure")

    monkeypatch.setattr(decisions_module, "run_agent_sync", fail_model)
    resolver = DecisionStagesMixin()
    resolver.model = object()

    resolution = resolver._resolve_rna_decision(
        path,
        request_record,
        definition,
        evidence,
        {},
    )

    assert resolution.workflow.status == "running"
    assert resolution.pending is None
    assert resolution.record is not None
    assert resolution.record.selectedOptionId == "qcGrouping:global"
    assert resolution.record.source == "rule"
