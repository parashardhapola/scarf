"""Persisted artifact and workflow-stage collection for agent reports."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ...datastore.datastore import DataStore
from ...storage.stores import zarr_root_path
from .. import record_io
from ..orchestrator import journal
from ..orchestrator.models import (
    _STAGE_ORDER,
    AutomatedWorkflowResult,
    OrchestrationRequestRecord,
    WorkflowStageAttempt,
    artifact_model_to_ref,
)
from ..persistence.contracts import AgentWorkflowRun
from ..persistence.decisions import load_latest_decision_workflow_snapshot
from ..persistence.reports import load_agent_report
from ..types import ArtifactReferenceModel
from .contracts import _is_sequence, _mapping, _mappings, _text_values


def _local_root(target: str | Path | DataStore) -> Path:
    """Resolve a local filesystem root without accepting remote stores."""
    if isinstance(target, DataStore):
        location = zarr_root_path(target.z)
        if location is None:
            raise ValueError("Agent HTML reports require a local filesystem store")
        path = Path(location)
    elif isinstance(target, Path):
        path = target
    elif isinstance(target, str) and target.startswith("file://"):
        path = Path(target.removeprefix("file://"))
    elif isinstance(target, str):
        if "://" in target:
            raise ValueError("Agent HTML reports require a local filesystem store")
        path = Path(target)
    else:
        raise TypeError("Agent HTML reports require a local filesystem store")
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _open_datastore(
    target: str | Path | DataStore,
    root: Path,
    workflow: AgentWorkflowRun,
) -> DataStore:
    if isinstance(target, DataStore):
        if target.workspace != workflow.workspace:
            raise ValueError("Workflow workspace does not match the DataStore")
        return target
    default_assay = next(iter(workflow.datasetFingerprints))
    return DataStore(
        str(root),
        default_assay=default_assay,
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r",
        workspace=workflow.workspace,
    )


def _load_request(
    store: DataStore,
    prefix: str,
    workflow_run_id: str,
) -> OrchestrationRequestRecord:
    record = cast(
        OrchestrationRequestRecord,
        journal._read_model(
            store.zw,
            journal._request_key(prefix, workflow_run_id),
            OrchestrationRequestRecord,
        ),
    )
    if record.workflowRunId != workflow_run_id:
        raise ValueError("Stored orchestration request belongs to another workflow")
    if record.requestSha256 != journal._sha256_model(record.request):
        raise ValueError("Stored orchestration request checksum is invalid")
    if record.configSha256 != journal._sha256_model(record.config):
        raise ValueError("Stored orchestration configuration checksum is invalid")
    if record.contentSha256 != journal._record_checksum(record):
        raise ValueError("Stored orchestration request envelope is invalid")
    return record


def _load_completed_result(
    store: DataStore,
    workflow: AgentWorkflowRun,
) -> tuple[str, AutomatedWorkflowResult, OrchestrationRequestRecord]:
    if workflow.status != "completed":
        raise RuntimeError(
            "Agent HTML reports can only be generated for completed workflows"
        )
    prefix = journal._ensure_orchestration_store(store)
    result = journal._load_terminal_result(store, prefix, workflow)
    if result is None:
        raise FileNotFoundError(
            f"Completed workflow {workflow.workflowRunId!r} has no terminal result"
        )
    if result.status != "completed" or result.finalAnalysis is None:
        raise ValueError("Completed workflow result lacks its final analysis handoff")
    request = _load_request(store, prefix, workflow.workflowRunId)
    if request.request.workspace != workflow.workspace:
        raise ValueError("Stored request workspace does not match the workflow")
    return prefix, result, request


def _collect_reports(
    store: DataStore,
    result: AutomatedWorkflowResult,
) -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = {}
    for reference in result.reportReferences:
        report = load_agent_report(store, reference)
        reports.setdefault(reference.agentName, []).append(
            report.model_dump(mode="json")
        )
    return reports


def _collect_active_decisions(
    store: DataStore,
    workflow_run_id: str,
) -> dict[str, dict[str, Any]]:
    try:
        snapshot = load_latest_decision_workflow_snapshot(store, workflow_run_id)
    except KeyError:
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    for record in snapshot.workflow.active_decision_records():
        if record.decisionId in decisions:
            raise ValueError(
                f"Decision workflow has multiple active {record.decisionId!r} records"
            )
        decisions[record.decisionId] = record.model_dump(mode="json")
    return decisions


def _stage_summary(attempt: WorkflowStageAttempt) -> dict[str, Any]:
    duration = (
        (attempt.completedAtNs - attempt.startedAtNs) / 1_000_000_000
        if attempt.completedAtNs
        else None
    )
    error_type = None
    if attempt.error:
        candidate = attempt.error.partition(":")[0].strip()
        error_type = (
            candidate
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", candidate)
            else "WorkflowStageError"
        )
    return {
        "stage": attempt.stage,
        "attemptId": attempt.attemptId,
        "status": attempt.status,
        "durationSeconds": duration,
        "actions": list(attempt.actions),
        "reportCount": len(attempt.reportReferences),
        "artifactCount": len(attempt.artifacts),
        "artifacts": {
            name: artifact.model_dump(mode="json")
            for name, artifact in attempt.artifacts.items()
        },
        "parentAttempts": [
            f"{parent.stage}:{parent.attemptId}" for parent in attempt.parentAttempts
        ],
        "questionIds": (
            [question.questionId for question in attempt.needsInput.questions]
            if attempt.needsInput is not None
            else []
        ),
        "noteCount": len(attempt.notes),
        "notes": list(attempt.notes),
        "errorType": error_type,
    }


def _collect_history(
    store: DataStore,
    prefix: str,
    workflow: AgentWorkflowRun,
    request: OrchestrationRequestRecord,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempts: dict[tuple[str, str], WorkflowStageAttempt] = {}
    summaries: list[tuple[int, dict[str, Any]]] = []
    for stage in _STAGE_ORDER:
        starts = {
            item.attemptId: item
            for item in journal._stage_starts(
                store.zw, prefix, workflow.workflowRunId, stage
            )
        }
        outcomes = {
            item.attemptId: item
            for item in journal._stage_outcomes(
                store.zw, prefix, workflow.workflowRunId, stage
            )
        }
        if not set(outcomes).issubset(starts):
            raise ValueError("Workflow history contains an outcome without a start")
        for attempt_id, started in starts.items():
            attempt = outcomes.get(attempt_id, started)
            identity = (stage, attempt_id)
            if identity in attempts:
                raise ValueError("Workflow history contains duplicate stage attempts")
            attempts[identity] = attempt
            summaries.append((attempt.startedAtNs, _stage_summary(attempt)))

    for attempt in attempts.values():
        for parent in attempt.parentAttempts:
            observed = attempts.get((parent.stage, parent.attemptId))
            if (
                observed is None
                or observed.status != "done"
                or observed.contentSha256 != parent.contentSha256
            ):
                raise ValueError("Workflow parent-stage lineage does not resolve")

    terminal_candidates = [
        attempt
        for attempt in attempts.values()
        if attempt.stage == "analysis_finalization" and attempt.status == "done"
    ]
    if len(terminal_candidates) != 1:
        raise ValueError(
            "Completed workflow lacks one exact analysis finalization attempt"
        )
    current = terminal_candidates[0]
    terminal_chain: set[tuple[str, str]] = set()
    while True:
        identity = (current.stage, current.attemptId)
        if identity in terminal_chain:
            raise ValueError("Workflow stage lineage contains a cycle")
        terminal_chain.add(identity)
        if not journal._stage_outcome_resolves(
            store,
            prefix,
            workflow.workflowRunId,
            request,
            current,
        ):
            raise ValueError("Terminal workflow stage artifacts do not resolve")
        stage_index = _STAGE_ORDER.index(current.stage)
        if stage_index == 0:
            if current.parentAttempts:
                raise ValueError("The ingest stage cannot have a parent")
            break
        if len(current.parentAttempts) != 1:
            raise ValueError("Every terminal-chain stage must have one parent")
        parent = current.parentAttempts[0]
        if parent.stage != _STAGE_ORDER[stage_index - 1]:
            raise ValueError("Terminal workflow lineage skips a stage")
        current = attempts[(parent.stage, parent.attemptId)]

    resumes: list[dict[str, Any]] = []
    resume_prefix = record_io.join_key(prefix, workflow.workflowRunId, "resumes")
    for key in record_io.list_keys(store.zw, resume_prefix):
        if not key.endswith(".json"):
            continue
        resume_id = key.rsplit("/", 1)[-1].removesuffix(".json")
        resume = journal._validated_resume_record(
            store, prefix, workflow.workflowRunId, resume_id
        )
        resumes.append(
            {
                "resumeId": resume.resumeId,
                "createdAtNs": resume.createdAtNs,
                "answeredStage": (
                    resume.answeredAttempt.stage
                    if resume.answeredAttempt is not None
                    else None
                ),
                "answeredAttemptId": (
                    resume.answeredAttempt.attemptId
                    if resume.answeredAttempt is not None
                    else None
                ),
                "questionIds": list(resume.questionIds),
            }
        )
    resumes.sort(key=lambda value: (value["createdAtNs"], value["resumeId"]))
    ordered = sorted(
        summaries,
        key=lambda item: (
            item[0],
            str(item[1]["stage"]),
            str(item[1]["attemptId"]),
        ),
    )
    return [value for _, value in ordered], resumes


def _hvg_diagnostic_evidence(
    store: DataStore,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    model = ArtifactReferenceModel.model_validate(reference)
    group: Any = store.load_artifact(artifact_model_to_ref(model))
    provenance = _mapping(group.attrs.get("provenance"))
    parameters = _mapping(provenance.get("parameters"))
    ranking = np.asarray(group["ranking"][:], dtype=np.int64)
    corrected_variance = np.asarray(
        group["global_corrected_variance"][:],
        dtype=np.float64,
    )
    recurrence = np.asarray(group["recurrence"][:], dtype=np.int64)
    eligible = np.asarray(group["eligible"][:], dtype=bool)
    if (
        ranking.ndim != 1
        or corrected_variance.ndim != 1
        or recurrence.shape != corrected_variance.shape
        or eligible.shape != corrected_variance.shape
    ):
        raise ValueError("HVG diagnostic arrays are malformed")
    if ranking.size and (
        int(ranking.min()) < 0 or int(ranking.max()) >= corrected_variance.size
    ):
        raise ValueError("HVG diagnostic ranking contains out-of-range indices")
    raw_counts = parameters.get("candidate_counts")
    if not _is_sequence(raw_counts):
        raise ValueError("HVG diagnostic is missing candidate counts")
    raw_count_values = cast(Sequence[Any], raw_counts)
    candidate_counts = [
        int(value)
        for value in raw_count_values
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    if len(candidate_counts) != len(raw_count_values) or any(
        value < 1 or value > ranking.size for value in candidate_counts
    ):
        raise ValueError("HVG diagnostic candidate counts are invalid")
    valid_groups = group.attrs.get("valid_groups", [])
    if not _is_sequence(valid_groups):
        raise ValueError("HVG diagnostic valid groups are malformed")
    valid_group_count = len(cast(Sequence[Any], valid_groups))
    excluded_groups = group.attrs.get("excluded_groups", [])
    if not _is_sequence(excluded_groups):
        raise ValueError("HVG diagnostic excluded groups are malformed")
    eligible_variance = float(corrected_variance[eligible].sum())
    recurrence_threshold = max(2, (valid_group_count + 1) // 2)
    candidates: list[dict[str, Any]] = []
    for count in candidate_counts:
        selected = ranking[:count]
        variance_fraction = (
            float(corrected_variance[selected].sum()) / eligible_variance
            if eligible_variance > 0
            else 0.0
        )
        candidates.append(
            {
                "featureCount": count,
                "varianceFraction": variance_fraction,
                "recurrentFraction": (
                    float((recurrence[selected] >= recurrence_threshold).mean())
                    if valid_group_count
                    else None
                ),
            }
        )
    broad = ranking[: max(candidate_counts)]
    return {
        "rankingMode": group.attrs.get("ranking_mode"),
        "eligibleFeatureCount": int(eligible.sum()),
        "validTechnicalGroups": valid_group_count,
        "excludedTechnicalGroupCount": len(cast(Sequence[Any], excluded_groups)),
        "candidateMetrics": candidates,
        "meanTechnicalGroupCoverage": (
            float(recurrence[broad].mean()) / valid_group_count
            if valid_group_count
            else None
        ),
        "recurrentInTwoGroupsFraction": (
            float((recurrence[broad] >= 2).mean()) if valid_group_count else None
        ),
        "minimumDetectedCells": parameters.get("min_cells"),
        "minimumTechnicalGroupCells": parameters.get("min_group_cells"),
    }


def _latest_hvg_diagnostic_artifacts(
    stage_attempts: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    for attempt in reversed(stage_attempts):
        artifacts = _mapping(attempt.get("artifacts"))
        match = next(
            (
                (str(name), _mapping(reference))
                for name, reference in artifacts.items()
                if re.fullmatch(r".+_hvg_diagnostic", str(name))
            ),
            None,
        )
        if match is not None:
            return match[0], match[1], artifacts
    return "", {}, {}


def _collect_hvg_evidence(
    store: DataStore,
    stage_attempts: Sequence[Mapping[str, Any]],
    preprocessing_plan: Mapping[str, Any],
) -> dict[str, Any]:
    selected_name, selected_reference, selected_artifacts = (
        _latest_hvg_diagnostic_artifacts(stage_attempts)
    )
    if not selected_reference:
        return {}
    assay = selected_name.removesuffix("_hvg_diagnostic")
    selected = _hvg_diagnostic_evidence(store, selected_reference)
    ranking_references = (
        ("global", selected_artifacts.get(f"{assay}_hvg_global_diagnostic")),
        (
            "batchAware",
            selected_artifacts.get(f"{assay}_hvg_batchAware_diagnostic"),
        ),
    )
    rankings: list[dict[str, Any]] = []
    for mode, reference in ranking_references:
        if isinstance(reference, Mapping):
            summary = _hvg_diagnostic_evidence(store, reference)
            if summary.get("rankingMode") != mode:
                raise ValueError("HVG diagnostic ranking mode does not match its role")
            rankings.append(summary)
    if not rankings:
        rankings.append(selected)
    assay_plan = next(
        (
            value
            for value in _mappings(preprocessing_plan.get("assays"))
            if value.get("assay") == assay
        ),
        {},
    )
    selected_count = _mapping(assay_plan.get("featureParameters")).get("topN")
    default_reference_counts = sorted(
        {
            int(default_match.group(1))
            for name in selected_artifacts
            if (
                default_match := re.fullmatch(
                    rf"{re.escape(assay)}_hvg_scarf_default_([0-9]+)",
                    str(name),
                )
            )
        }
    )
    executed_branch_count = len(default_reference_counts) + sum(
        len(_mappings(ranking.get("candidateMetrics"))) for ranking in rankings
    )
    return {
        "assay": assay,
        "selectedRankingMode": selected.get("rankingMode"),
        "selectedFeatureCount": selected_count,
        "rankings": rankings,
        "candidateMetrics": selected.get("candidateMetrics"),
        "eligibleFeatureCount": selected.get("eligibleFeatureCount"),
        "validTechnicalGroups": selected.get("validTechnicalGroups"),
        "excludedTechnicalGroupCount": selected.get("excludedTechnicalGroupCount"),
        "minimumDetectedCells": selected.get("minimumDetectedCells"),
        "minimumTechnicalGroupCells": selected.get("minimumTechnicalGroupCells"),
        "scarfDefaultReferenceCounts": default_reference_counts,
        "executedBranchCount": executed_branch_count,
    }


def _collect_default_feature_inventories(
    store: DataStore,
    preprocessing_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inventories: list[dict[str, Any]] = []
    for assay_plan in _mappings(preprocessing_plan.get("assays")):
        assay_name = str(assay_plan.get("assay") or "")
        parameters = _mapping(assay_plan.get("featureParameters"))
        inventory = _mapping(parameters.get("defaultFeatureInventory"))
        if not inventory:
            continue
        feature_column = str(inventory.get("featureColumn") or "")
        blacklist = str(inventory.get("blacklist") or "")
        if not assay_name or not feature_column or not blacklist:
            raise ValueError("Scarf default feature inventory is incomplete")
        assay = store.get_assay(assay_name)
        if feature_column not in assay.feats.columns:
            raise ValueError(
                f"Scarf default feature column {feature_column!r} is unavailable "
                f"for assay {assay_name!r}"
            )
        names = [str(value) for value in assay.feats.fetch_all(feature_column)]
        try:
            compiled = re.compile(blacklist.upper())
        except re.error as exc:
            raise ValueError("Scarf default feature blacklist is invalid") from exc
        matched = sorted(
            (name for name in names if compiled.match(name.upper()) is not None),
            key=lambda value: (value.casefold(), value),
        )
        expected_total = inventory.get("totalFeatures")
        expected_matches = inventory.get("matchCount")
        if isinstance(expected_total, int) and expected_total != len(names):
            raise ValueError(
                f"Scarf default feature inventory for {assay_name!r} has stale "
                "total feature evidence"
            )
        if isinstance(expected_matches, int) and expected_matches != len(matched):
            raise ValueError(
                f"Scarf default feature inventory for {assay_name!r} has stale "
                "blacklist match evidence"
            )
        inventories.append(
            {
                **inventory,
                "assay": assay_name,
                "appliedToSelectedRepresentation": (
                    parameters.get("useScarfDefaultBlacklist") is True
                ),
                "selectedExcludeFamilies": _text_values(
                    parameters.get("excludeFamilies")
                ),
                "selectedProtectFamilies": _text_values(
                    parameters.get("protectFamilies")
                ),
                "matchedFeatures": matched,
            }
        )
    return inventories


def _default_inventory_for_assay(
    inventories: Sequence[Mapping[str, Any]],
    assay: str,
) -> dict[str, Any]:
    matches = [dict(value) for value in inventories if value.get("assay") == assay]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple Scarf default inventories found for assay {assay!r}"
        )
    return matches[0] if matches else {}
