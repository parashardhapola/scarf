"""Conservative candidate admission using the existing stage journal."""

from typing import Any

from ...datastore.datastore import DataStore
from ...utils.logging import logger
from .. import record_io
from ..persistence.contracts import AgentWorkflowRun
from . import journal
from .models import (
    AutomatedWorkflowConfig,
    OrchestrationRequestRecord,
    WorkflowStageName,
)


_PASS_STAGES: dict[WorkflowStageName, str] = {
    "preprocessing": "baseline",
    "feature_policy_preprocessing": "featureRevision",
}


def candidate_pass_breakdown(config: AutomatedWorkflowConfig) -> dict[str, int]:
    """Reserve every configured alternative, including conditional work."""
    return {
        "hvg": 3 * len(config.hvgCandidateCounts),
        "pca": len(config.pcaCandidateDimensions),
        "nativeCorrection": 1,
        "harmony": config.maxHarmonyCandidatesPerAssay,
        "neighbors": len(config.graphNeighborCandidates),
        "resolutions": len(config.leidenResolutionCandidates),
        "refinement": config.maxRefinedCandidatesPerAssay,
    }


def reserve_candidate_pass(
    store: DataStore,
    prefix: str,
    workflow: AgentWorkflowRun,
    request_record: OrchestrationRequestRecord,
    stage_name: WorkflowStageName,
) -> dict[str, Any]:
    """Admit a logical pass before its immutable started record is written.

    The caller persists the returned reservation in ``inputs.candidateBudget``.
    Repeated attempts retain the same slots; conditional work does not refund
    slots. This bounds candidate alternatives, not numerical work or wall time.
    """
    if stage_name not in _PASS_STAGES:
        raise ValueError("Candidate reservations require a preprocessing stage")
    if workflow.workflowRunId != request_record.workflowRunId:
        raise ValueError("Candidate budget belongs to a different workflow")
    breakdown = candidate_pass_breakdown(request_record.config)
    per_pass = sum(breakdown.values())
    admitted: set[str] = set()
    for stage, logical_pass in _PASS_STAGES.items():
        expected = {
            "logicalPass": logical_pass,
            "reserved": per_pass,
            "breakdown": breakdown,
        }
        for started in journal._stage_starts(
            store.zw, prefix, workflow.workflowRunId, stage
        ):
            if (
                started.requestSha256 != request_record.requestSha256
                or started.configSha256 != request_record.configSha256
            ):
                raise ValueError(
                    "Candidate reservation request/config identity differs"
                )
            reservation = started.inputs.get("candidateBudget")
            if reservation is None:
                if started.inputs.get("candidateBudgetRejected") is True:
                    continue
                if stage == "feature_policy_preprocessing" and isinstance(
                    started.inputs.get("baselineAttemptId"), str
                ):
                    continue
                raise ValueError(
                    "Preprocessing history lacks its candidate reservation; "
                    "start a new workflow"
                )
            if record_io.canonical_json_bytes(
                reservation
            ) != record_io.canonical_json_bytes(expected):
                raise ValueError(
                    f"Persisted {logical_pass} candidate reservation differs "
                    "from the immutable workflow configuration"
                )
            admitted.add(logical_pass)
    logical_pass = _PASS_STAGES[stage_name]
    if logical_pass == "featureRevision" and "baseline" not in admitted:
        raise ValueError("Feature revision requires a reserved baseline pass")
    total = per_pass * len(admitted | {logical_pass})
    limit = request_record.config.maxCandidateEvaluations
    details = ", ".join(f"{name}={count}" for name, count in breakdown.items())
    if total > limit:
        already_reserved = per_pass * len(admitted)
        raise ValueError(
            f"Candidate budget exceeded before {logical_pass}: "
            f"{already_reserved} slots already reserved, {per_pass} required "
            f"for this pass ({details}), workflow limit={limit}. "
            "Increase maxCandidateEvaluations or explicitly reduce the candidate "
            "lists in a new workflow. No candidate lists were truncated."
        )
    logger.info(
        f"Candidate work: {logical_pass} reserves {per_pass} slots ({details}); "
        f"workflow reserved {total}/{limit}. Actual evaluations may be fewer."
    )
    return {"logicalPass": logical_pass, "reserved": per_pass, "breakdown": breakdown}
