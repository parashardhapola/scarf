"""Append-only persistence for decision-driven orchestration ledgers."""

import hashlib
import json
import re
import time
from typing import Literal, cast

import zarr
from pydantic import ConfigDict, Field, field_validator, model_validator
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from . import record_io
from .decision_kernel import (
    DecisionRecord,
    DecisionWorkflowRun,
    PendingDecision,
    RevisionRequest,
)
from .orchestrator.models import (
    _ORCHESTRATION_FORMAT,
    _ORCHESTRATION_VERSION,
    OrchestrationRequestRecord,
)
from .persistence import AgentPersistenceTarget, _resolve_target
from .rna_decisions import (
    CompiledRnaDecision,
    RNA_DECISION_TRANSITION_GRAPH,
    RnaDecisionCheckpoint,
)
from .types import AgentDataModel

_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RERUN_MESSAGE = (
    "Start a new orchestration run; decision snapshots are not migrated or resumed "
    "across persistence formats. Existing artifacts are left unchanged."
)


class DecisionPersistenceFormatError(ValueError):
    """Raised for old or unknown persistence without mutating stored data."""


class DecisionSnapshotModel(AgentDataModel):
    """Base for immutable decision-persistence records."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class OrchestrationRunIdentity(DecisionSnapshotModel):
    """Exact immutable orchestration request linked by a decision snapshot."""

    workflowRunId: str
    workspace: str | None = None
    requestSha256: str
    configSha256: str
    requestContentSha256: str

    @field_validator("workflowRunId")
    @classmethod
    def validate_workflow_run_id(cls, value: str) -> str:
        if _RUN_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("workflowRunId must be a lowercase run identifier")
        return value

    @field_validator("requestSha256", "configSha256", "requestContentSha256")
    @classmethod
    def validate_sha256(cls, value: str, info: object) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            field_name = getattr(info, "field_name", "digest")
            raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        return value


class DecisionWorkflowSnapshot(DecisionSnapshotModel):
    """One content-addressed snapshot in an immutable decision ledger chain."""

    recordType: Literal["decisionWorkflowSnapshot"] = "decisionWorkflowSnapshot"
    formatVersion: Literal[2] = 2
    sequence: int = Field(ge=0, strict=True)
    createdAtNs: int = Field(ge=1, strict=True)
    parentContentSha256: str | None = None
    orchestrationRun: OrchestrationRunIdentity
    workflow: DecisionWorkflowRun
    contentSha256: str

    @field_validator("parentContentSha256", "contentSha256")
    @classmethod
    def validate_sha256(cls, value: str | None, info: object) -> str | None:
        if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
            field_name = getattr(info, "field_name", "digest")
            raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "DecisionWorkflowSnapshot":
        if self.sequence == 0 and self.parentContentSha256 is not None:
            raise ValueError("The first snapshot cannot have a parent")
        if self.sequence > 0 and self.parentContentSha256 is None:
            raise ValueError("A later snapshot requires its exact parent checksum")
        if self.workflow.workflowRunId != self.orchestrationRun.workflowRunId:
            raise ValueError(
                "Decision workflow identity must match the orchestration run"
            )
        return self


def _validate_run_id(value: str) -> str:
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("workflow_run_id must be a lowercase run identifier")
    return value


def _validate_sha256(value: str, label: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _model_checksum(value: AgentDataModel) -> str:
    return hashlib.sha256(
        record_io.canonical_json_bytes(value.model_dump(mode="json"))
    ).hexdigest()


def _record_checksum(value: AgentDataModel) -> str:
    return hashlib.sha256(
        record_io.canonical_json_bytes(
            value.model_dump(mode="json", exclude={"contentSha256"})
        )
    ).hexdigest()


def decision_record_checksum(record: DecisionRecord) -> str:
    """Return the canonical SHA-256 identity of one immutable decision record."""
    return hashlib.sha256(
        record_io.canonical_json_bytes(record.model_dump(mode="json"))
    ).hexdigest()


def _snapshot_checksum(snapshot: DecisionWorkflowSnapshot) -> str:
    return _record_checksum(snapshot)


def _write_key_once(group: zarr.Group, key: str, payload: bytes) -> None:
    store = group.store
    if bool(getattr(store, "read_only", False)) or not bool(
        getattr(store, "supports_writes", True)
    ):
        raise PermissionError("Decision persistence target is read-only")
    if record_io.read_key(group, key) is not None:
        raise FileExistsError(f"Immutable decision snapshot {key!r} already exists")
    buffer = default_buffer_prototype().buffer.from_bytes(payload)
    sync(store.set_if_not_exists(key, buffer))
    stored = record_io.read_key(group, key)
    if stored is None:
        raise RuntimeError(f"Decision snapshot {key!r} was not stored")
    if stored != payload:
        raise FileExistsError(
            f"Immutable decision snapshot {key!r} was written by another writer"
        )


def _list_keys(group: zarr.Group, prefix: str) -> list[str]:
    if not group.store.supports_listing:
        raise NotImplementedError("Decision persistence requires a listable Zarr store")
    return record_io.list_keys(group, prefix)


def _orchestration_prefix(group: zarr.Group) -> str:
    return record_io.join_key(
        str(getattr(group, "path", "")).strip("/"),
        "agents",
        "orchestrations",
    )


def _request_key(prefix: str, workflow_run_id: str) -> str:
    return record_io.join_key(prefix, workflow_run_id, "request.json")


def _snapshot_prefix(prefix: str, workflow_run_id: str) -> str:
    return record_io.join_key(
        prefix,
        workflow_run_id,
        "decisions",
        "snapshots",
    )


def _snapshot_key(prefix: str, workflow_run_id: str, content_sha256: str) -> str:
    return record_io.join_key(
        _snapshot_prefix(prefix, workflow_run_id),
        f"{content_sha256}.json",
    )


def _format_error(detail: str) -> DecisionPersistenceFormatError:
    return DecisionPersistenceFormatError(f"{detail} {_RERUN_MESSAGE}")


def _resolve_orchestration_group(
    target: AgentPersistenceTarget,
    *,
    write: bool,
    workspace: str | None,
) -> tuple[zarr.Group, str | None, str]:
    group, _datastore, resolved_workspace, _analysis_store = _resolve_target(
        target,
        write=write,
        workspace=workspace,
    )
    if "agents" not in group:
        raise FileNotFoundError(
            "No agent namespace exists for this data group. " + _RERUN_MESSAGE
        )
    agents = group["agents"]
    if not isinstance(agents, zarr.Group):
        raise _format_error("The agents namespace is not a Zarr group.")
    if "orchestrations" not in agents:
        raise FileNotFoundError(
            "No orchestration journal exists for this data group. " + _RERUN_MESSAGE
        )
    orchestrations = agents["orchestrations"]
    if not isinstance(orchestrations, zarr.Group):
        raise _format_error("The orchestrations namespace is not a Zarr group.")
    observed_format = orchestrations.attrs.get("format")
    observed_version = orchestrations.attrs.get("format_version")
    if (
        observed_format != _ORCHESTRATION_FORMAT
        or observed_version != _ORCHESTRATION_VERSION
    ):
        raise _format_error(
            "Unsupported orchestration persistence format "
            f"{observed_format!r} version {observed_version!r}."
        )
    return group, resolved_workspace, _orchestration_prefix(group)


def _decode_json(raw: bytes, key: str) -> object:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Decision JSON record {key!r} is malformed") from exc


def _load_orchestration_identity(
    group: zarr.Group,
    prefix: str,
    workflow_run_id: str,
    workspace: str | None,
) -> OrchestrationRunIdentity:
    key = _request_key(prefix, workflow_run_id)
    raw = record_io.read_key(group, key)
    if raw is None:
        raise KeyError(
            f"Unknown orchestration run {workflow_run_id!r}; {_RERUN_MESSAGE}"
        )
    decoded = _decode_json(raw, key)
    if not isinstance(decoded, dict):
        raise ValueError(f"Orchestration request {key!r} is not a JSON object")
    if decoded.get("formatVersion") != 2:
        raise _format_error(
            f"Unsupported orchestration request version "
            f"{decoded.get('formatVersion')!r}."
        )
    if decoded.get("recordType") != "automatedWorkflowRequest":
        raise _format_error(
            f"Unsupported orchestration request type {decoded.get('recordType')!r}."
        )
    try:
        request = OrchestrationRequestRecord.model_validate(decoded)
    except ValueError as exc:
        raise ValueError(
            f"Orchestration request {key!r} does not match its schema"
        ) from exc
    if request.workflowRunId != workflow_run_id:
        raise ValueError("Orchestration request identity does not match its path")
    if request.request.workspace != workspace:
        raise ValueError("Orchestration request workspace does not match its path")
    if request.requestSha256 != _model_checksum(request.request):
        raise ValueError("Orchestration request payload checksum is invalid")
    if request.configSha256 != _model_checksum(request.config):
        raise ValueError("Orchestration configuration checksum is invalid")
    if request.contentSha256 != _record_checksum(request):
        raise ValueError("Orchestration request envelope checksum is invalid")
    return OrchestrationRunIdentity(
        workflowRunId=workflow_run_id,
        workspace=workspace,
        requestSha256=request.requestSha256,
        configSha256=request.configSha256,
        requestContentSha256=request.contentSha256,
    )


def _load_snapshot_at(
    group: zarr.Group,
    prefix: str,
    workflow_run_id: str,
    content_sha256: str,
    expected_identity: OrchestrationRunIdentity,
) -> DecisionWorkflowSnapshot:
    content_sha256 = _validate_sha256(content_sha256, "content_sha256")
    key = _snapshot_key(prefix, workflow_run_id, content_sha256)
    raw = record_io.read_key(group, key)
    if raw is None:
        raise KeyError(
            f"Unknown decision snapshot {content_sha256!r} for "
            f"workflow {workflow_run_id!r}"
        )
    decoded = _decode_json(raw, key)
    if not isinstance(decoded, dict):
        raise ValueError(f"Decision snapshot {key!r} is not a JSON object")
    if decoded.get("formatVersion") != 2:
        raise _format_error(
            f"Unsupported decision snapshot version {decoded.get('formatVersion')!r}."
        )
    if decoded.get("recordType") != "decisionWorkflowSnapshot":
        raise _format_error(
            f"Unsupported decision snapshot type {decoded.get('recordType')!r}."
        )
    try:
        snapshot = DecisionWorkflowSnapshot.model_validate(decoded)
    except ValueError as exc:
        raise ValueError(
            f"Decision snapshot {key!r} does not match its schema"
        ) from exc
    if snapshot.contentSha256 != content_sha256:
        raise ValueError("Decision snapshot checksum does not match its path")
    if snapshot.contentSha256 != _snapshot_checksum(snapshot):
        raise ValueError("Decision snapshot checksum does not match its content")
    if snapshot.orchestrationRun != expected_identity:
        raise ValueError("Decision snapshot orchestration identity is stale")
    if snapshot.workflow.workflowRunId != workflow_run_id:
        raise ValueError("Decision snapshot workflow identity does not match its path")
    return snapshot


def _validate_snapshot_evolution(
    previous: DecisionWorkflowSnapshot,
    current: DecisionWorkflowSnapshot,
) -> None:
    if previous.orchestrationRun != current.orchestrationRun:
        raise ValueError("Decision snapshot orchestration identity changed")
    if previous.workflow.workflowRunId != current.workflow.workflowRunId:
        raise ValueError("Decision snapshot workflow identity changed")
    if current.createdAtNs < previous.createdAtNs:
        raise ValueError("Decision snapshot creation times must not move backwards")
    for field_name in (
        "decisionRecords",
        "verificationRecords",
        "revisionRequests",
    ):
        old_values = getattr(previous.workflow, field_name)
        new_values = getattr(current.workflow, field_name)
        if new_values[: len(old_values)] != old_values:
            raise ValueError(
                f"Decision snapshot {field_name} must preserve its immutable prefix"
            )
    if previous.workflow.status in {"completed", "abstained", "failed"}:
        raise ValueError("Terminal decision snapshots cannot have descendants")


def _load_snapshot_chain(
    group: zarr.Group,
    prefix: str,
    workflow_run_id: str,
    identity: OrchestrationRunIdentity,
) -> list[DecisionWorkflowSnapshot]:
    snapshot_prefix = _snapshot_prefix(prefix, workflow_run_id)
    snapshots: list[DecisionWorkflowSnapshot] = []
    for key in _list_keys(group, snapshot_prefix):
        if not key.endswith(".json"):
            continue
        filename = key.rsplit("/", 1)[-1]
        content_sha256 = filename.removesuffix(".json")
        if _SHA256_PATTERN.fullmatch(content_sha256) is None:
            raise ValueError("Decision snapshot path is not content-addressed")
        snapshots.append(
            _load_snapshot_at(
                group,
                prefix,
                workflow_run_id,
                content_sha256,
                identity,
            )
        )
    snapshots.sort(key=lambda value: (value.sequence, value.contentSha256))
    for expected_sequence, snapshot in enumerate(snapshots):
        if snapshot.sequence != expected_sequence:
            raise ValueError("Decision snapshot chain has a gap or fork")
        expected_parent = (
            snapshots[expected_sequence - 1].contentSha256
            if expected_sequence > 0
            else None
        )
        if snapshot.parentContentSha256 != expected_parent:
            raise ValueError("Decision snapshot parent does not match the exact chain")
        if expected_sequence > 0:
            _validate_snapshot_evolution(
                snapshots[expected_sequence - 1],
                snapshot,
            )
    return snapshots


def save_decision_workflow_snapshot(
    target: AgentPersistenceTarget,
    workflow: DecisionWorkflowRun,
    *,
    workspace: str | None = None,
    created_at_ns: int | None = None,
) -> DecisionWorkflowSnapshot:
    """Append one content-addressed snapshot without overwriting earlier state."""
    workflow_run_id = _validate_run_id(workflow.workflowRunId)
    group, resolved_workspace, prefix = _resolve_orchestration_group(
        target,
        write=True,
        workspace=workspace,
    )
    identity = _load_orchestration_identity(
        group,
        prefix,
        workflow_run_id,
        resolved_workspace,
    )
    snapshots = _load_snapshot_chain(
        group,
        prefix,
        workflow_run_id,
        identity,
    )
    if snapshots and snapshots[-1].workflow == workflow:
        return snapshots[-1]
    if snapshots and snapshots[-1].workflow.status in {
        "completed",
        "abstained",
        "failed",
    }:
        raise RuntimeError("Cannot append after a terminal decision snapshot")
    timestamp = time.time_ns() if created_at_ns is None else created_at_ns
    if timestamp < 1:
        raise ValueError("created_at_ns must be positive")
    snapshot_values = {
        "sequence": len(snapshots),
        "createdAtNs": timestamp,
        "parentContentSha256": snapshots[-1].contentSha256 if snapshots else None,
        "orchestrationRun": identity,
        "workflow": workflow,
        "contentSha256": "0" * 64,
    }
    unhashed = DecisionWorkflowSnapshot.model_validate(snapshot_values)
    snapshot_values["contentSha256"] = _snapshot_checksum(unhashed)
    snapshot = DecisionWorkflowSnapshot.model_validate(snapshot_values)
    if snapshots:
        _validate_snapshot_evolution(snapshots[-1], snapshot)
    key = _snapshot_key(
        prefix,
        workflow_run_id,
        snapshot.contentSha256,
    )
    _write_key_once(
        group,
        key,
        record_io.display_json_bytes(snapshot.model_dump(mode="json")),
    )
    stored = _load_snapshot_at(
        group,
        prefix,
        workflow_run_id,
        snapshot.contentSha256,
        identity,
    )
    chain = _load_snapshot_chain(group, prefix, workflow_run_id, identity)
    if chain[-1].contentSha256 != stored.contentSha256:
        raise RuntimeError("Decision snapshot did not become the exact chain head")
    return stored


def load_decision_workflow_snapshot(
    target: AgentPersistenceTarget,
    workflow_run_id: str,
    content_sha256: str,
    *,
    workspace: str | None = None,
) -> DecisionWorkflowSnapshot:
    """Load one exact content-addressed snapshot and validate its live link."""
    workflow_run_id = _validate_run_id(workflow_run_id)
    group, resolved_workspace, prefix = _resolve_orchestration_group(
        target,
        write=False,
        workspace=workspace,
    )
    identity = _load_orchestration_identity(
        group,
        prefix,
        workflow_run_id,
        resolved_workspace,
    )
    return _load_snapshot_at(
        group,
        prefix,
        workflow_run_id,
        content_sha256,
        identity,
    )


def list_decision_workflow_snapshots(
    target: AgentPersistenceTarget,
    workflow_run_id: str,
    *,
    workspace: str | None = None,
) -> list[DecisionWorkflowSnapshot]:
    """Return the complete validated append-only snapshot chain."""
    workflow_run_id = _validate_run_id(workflow_run_id)
    group, resolved_workspace, prefix = _resolve_orchestration_group(
        target,
        write=False,
        workspace=workspace,
    )
    identity = _load_orchestration_identity(
        group,
        prefix,
        workflow_run_id,
        resolved_workspace,
    )
    return _load_snapshot_chain(group, prefix, workflow_run_id, identity)


def load_latest_decision_workflow_snapshot(
    target: AgentPersistenceTarget,
    workflow_run_id: str,
    *,
    workspace: str | None = None,
) -> DecisionWorkflowSnapshot:
    """Return the head of the fully validated immutable snapshot chain."""
    snapshots = list_decision_workflow_snapshots(
        target,
        workflow_run_id,
        workspace=workspace,
    )
    if not snapshots:
        raise KeyError(f"No decision snapshots for workflow {workflow_run_id!r}")
    return snapshots[-1]


def load_decision_workflow_for_replay(
    target: AgentPersistenceTarget,
    workflow_run_id: str,
    content_sha256: str,
    *,
    expected_handoff_id: str | None = None,
    workspace: str | None = None,
) -> DecisionWorkflowRun:
    """Load an exact completed ledger after validating its complete chain."""
    snapshots = list_decision_workflow_snapshots(
        target,
        workflow_run_id,
        workspace=workspace,
    )
    matches = [
        snapshot for snapshot in snapshots if snapshot.contentSha256 == content_sha256
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Snapshot {content_sha256!r} is not in the exact workflow chain"
        )
    workflow = matches[0].workflow
    if workflow.status != "completed" or workflow.finalHandoffId is None:
        raise RuntimeError("Replay requires an exact completed decision workflow")
    if (
        expected_handoff_id is not None
        and workflow.finalHandoffId != expected_handoff_id
    ):
        raise ValueError("Replay final handoff identity does not match the snapshot")
    return workflow


def pause_decision_workflow(
    workflow: DecisionWorkflowRun,
    pending: PendingDecision,
) -> DecisionWorkflowRun:
    """Persist one unresolved checkpoint without inventing a selection."""
    if workflow.status != "running":
        raise ValueError("Only a running decision workflow can pause")
    values = workflow.model_dump(mode="json")
    values["status"] = "needsInput"
    values["pendingDecision"] = pending.model_dump(mode="json")
    return DecisionWorkflowRun.model_validate(values)


def attach_audited_rna_decision(
    workflow: DecisionWorkflowRun,
    record: DecisionRecord,
    compiled: CompiledRnaDecision,
    *,
    revision: RevisionRequest | None = None,
) -> DecisionWorkflowRun:
    """Append one audited RNA decision in exact transition order."""
    if workflow.status == "needsInput" and workflow.pendingDecision is not None:
        pending = workflow.pendingDecision
        mismatches = [
            field_name
            for field_name, pending_value, record_value in (
                ("decisionId", pending.decisionId, record.decisionId),
                (
                    "definitionVersion",
                    pending.definitionVersion,
                    record.definitionVersion,
                ),
                ("evidenceBundleId", pending.evidenceBundleId, record.evidenceBundleId),
                (
                    "evidenceBundleSha256",
                    pending.evidenceBundleSha256,
                    record.evidenceBundleSha256,
                ),
                ("offeredOptionIds", pending.offeredOptionIds, record.offeredOptionIds),
                (
                    "availableEvidenceIds",
                    pending.availableEvidenceIds,
                    record.availableEvidenceIds,
                ),
            )
            if pending_value != record_value
        ]
        if mismatches:
            raise ValueError(
                "Decision does not resolve the exact pending checkpoint; "
                f"mismatched fields: {mismatches}"
            )
    elif workflow.status != "running":
        raise ValueError("Only a running decision workflow can accept a decision")
    if (
        compiled.decisionRecordId != record.recordId
        or compiled.decisionId != record.decisionId
        or compiled.selectedOptionId != record.selectedOptionId
        or compiled.status != record.status
        or compiled.verification.decisionRecordId != record.recordId
        or compiled.verification.verificationId != record.verificationId
        or compiled.verification.status != "passed"
    ):
        raise ValueError("Compiled RNA decision does not exactly match its record")
    try:
        checkpoint = cast(RnaDecisionCheckpoint, record.decisionId)
        RNA_DECISION_TRANSITION_GRAPH.resolve(checkpoint, record.status)
    except KeyError as exc:
        raise ValueError("Decision is not a registered RNA checkpoint/status") from exc

    if record.supersedes is None:
        if revision is not None:
            raise ValueError("A non-superseding decision cannot attach a revision")
        if workflow.decisionRecords:
            previous = workflow.decisionRecords[-1]
            previous_checkpoint = cast(RnaDecisionCheckpoint, previous.decisionId)
            expected_checkpoint, terminal = RNA_DECISION_TRANSITION_GRAPH.resolve(
                previous_checkpoint,
                previous.status,
            )
            if terminal is not None or expected_checkpoint != checkpoint:
                raise ValueError(
                    "Decision does not follow the exact RNA transition order"
                )
        elif checkpoint != "qcGrouping":
            raise ValueError("The first RNA decision must be qcGrouping")
    else:
        if revision is not None:
            if revision.targetDecisionRecordId != record.supersedes:
                raise ValueError(
                    "A superseding decision requires its exact revision request"
                )
        else:
            if record.supersedes not in workflow.invalidated_decision_record_ids():
                raise ValueError(
                    "A superseding decision requires a revision or invalidation"
                )
            active_records = workflow.active_decision_records()
            if not active_records:
                raise ValueError(
                    "An invalidated decision rerun requires an active predecessor"
                )
            previous = active_records[-1]
            expected_checkpoint, terminal = RNA_DECISION_TRANSITION_GRAPH.resolve(
                cast(RnaDecisionCheckpoint, previous.decisionId),
                previous.status,
            )
            if terminal is not None or expected_checkpoint != checkpoint:
                raise ValueError(
                    "Invalidated decision rerun does not follow transition order"
                )

    _next_checkpoint, terminal_status = RNA_DECISION_TRANSITION_GRAPH.resolve(
        checkpoint,
        record.status,
    )
    status = (
        "needsInput"
        if terminal_status == "needsInput"
        else "abstained"
        if terminal_status == "abstained"
        else "running"
    )
    values = workflow.model_dump(mode="json")
    values["status"] = status
    values["pendingDecision"] = None
    values["decisionRecords"] = [*workflow.decisionRecords, record]
    values["verificationRecords"] = [
        *workflow.verificationRecords,
        compiled.verification,
    ]
    if revision is not None:
        values["revisionRequests"] = [*workflow.revisionRequests, revision]
    return DecisionWorkflowRun.model_validate(values)


def complete_decision_workflow(
    workflow: DecisionWorkflowRun,
    final_handoff_id: str,
) -> DecisionWorkflowRun:
    """Finalize a fully adjudicated RNA ledger with its exact handoff ID."""
    if workflow.status != "running" or not workflow.decisionRecords:
        raise ValueError("Only a running adjudicated workflow can complete")
    active_records = {
        record.decisionId: record for record in workflow.active_decision_records()
    }
    expected: str = "qcGrouping"
    visited: set[str] = set()
    while expected != "finalize":
        record = active_records.get(expected)
        if record is None:
            raise ValueError(f"RNA decision path is missing {expected!r}")
        visited.add(expected)
        try:
            destination, terminal = RNA_DECISION_TRANSITION_GRAPH.resolve(
                cast(RnaDecisionCheckpoint, expected),
                record.status,
            )
        except KeyError as exc:
            raise ValueError(
                f"Active decision {expected!r} has no registered transition"
            ) from exc
        if terminal is not None or destination is None:
            raise ValueError("RNA decisions have not reached the finalize transition")
        expected = destination
    if visited != set(active_records):
        raise ValueError(
            "Decision workflow contains active records outside its RNA path"
        )
    values = workflow.model_dump(mode="json")
    values["status"] = "completed"
    values["finalHandoffId"] = final_handoff_id
    return DecisionWorkflowRun.model_validate(values)


__all__ = [
    "DecisionPersistenceFormatError",
    "DecisionWorkflowSnapshot",
    "OrchestrationRunIdentity",
    "attach_audited_rna_decision",
    "complete_decision_workflow",
    "decision_record_checksum",
    "list_decision_workflow_snapshots",
    "load_decision_workflow_for_replay",
    "load_decision_workflow_snapshot",
    "load_latest_decision_workflow_snapshot",
    "pause_decision_workflow",
    "save_decision_workflow_snapshot",
]
