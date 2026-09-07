"""Workflow-wide reservations survive retries without charging reused passes."""

from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest
import zarr
from zarr.storage import MemoryStore

from scarf.agent.orchestrator import journal
from scarf.agent.orchestrator import AgentOrchestrator, AutomatedWorkflowRequest
from scarf.agent.orchestrator.budget import reserve_candidate_pass
from scarf.agent.orchestrator.models import (
    AutomatedWorkflowConfig,
    AutomatedPreprocessingPlan,
    OrchestrationRequestRecord,
    WorkflowStageName,
)
from scarf.agent.persistence.contracts import AgentWorkflowRun
from scarf.agent.persistence.reports import create_agent_workflow
from scarf.agent.experimental_context import ExperimentalContextResult
from scarf.agent.experimental_context.study import StudyContract
from scarf.datastore.datastore import DataStore
from tests.agent_orchestrator_store import create_store


_PREFIX = "agents/orchestrations"


def _context(
    limit: int = 50,
) -> tuple[Any, AgentWorkflowRun, OrchestrationRequestRecord]:
    store = SimpleNamespace(zw=zarr.group(store=MemoryStore()))
    workflow = AgentWorkflowRun.get_example()
    config = AutomatedWorkflowConfig(maxCandidateEvaluations=limit)
    record = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        config=config,
        configSha256=journal._sha256_model(config),
        requestSha256="a" * 64,
    )
    return store, workflow, record


def _start(
    store: Any,
    workflow: AgentWorkflowRun,
    record: OrchestrationRequestRecord,
    stage: WorkflowStageName,
    inputs: dict[str, Any],
) -> None:
    journal._start_attempt(
        store.zw, _PREFIX, workflow.workflowRunId, stage, record, [], inputs=inputs
    )


@pytest.mark.parametrize("limit", [1, 24])
def test_budget_rejects_baseline_before_any_reservation_is_written(limit: int) -> None:
    store, workflow, record = _context(limit)
    with pytest.raises(ValueError, match=r"0 slots already reserved, 25 required"):
        reserve_candidate_pass(store, _PREFIX, workflow, record, "preprocessing")
    assert (
        journal._stage_starts(
            store.zw, _PREFIX, workflow.workflowRunId, "preprocessing"
        )
        == []
    )


@pytest.mark.parametrize("limit,allow_revision", [(25, False), (49, False), (50, True)])
def test_budget_admits_whole_passes_and_counts_interrupted_work(
    limit: int, allow_revision: bool
) -> None:
    store, workflow, record = _context(limit)
    baseline = reserve_candidate_pass(store, _PREFIX, workflow, record, "preprocessing")
    assert baseline["reserved"] == 25
    assert baseline["breakdown"]["hvg"] == 9
    # An interrupted start has no outcome but retains its reservation.
    _start(store, workflow, record, "preprocessing", {"candidateBudget": baseline})
    # Starting the same logical pass again does not charge another 25 slots.
    repeated = reserve_candidate_pass(store, _PREFIX, workflow, record, "preprocessing")
    assert repeated == baseline
    _start(store, workflow, record, "preprocessing", {"candidateBudget": repeated})
    if allow_revision:
        revised = reserve_candidate_pass(
            store, _PREFIX, workflow, record, "feature_policy_preprocessing"
        )
        assert revised["logicalPass"] == "featureRevision"
        _start(
            store,
            workflow,
            record,
            "feature_policy_preprocessing",
            {"candidateBudget": revised},
        )
        assert (
            reserve_candidate_pass(
                store, _PREFIX, workflow, record, "feature_policy_preprocessing"
            )
            == revised
        )
    else:
        with pytest.raises(ValueError, match=r"25 slots already reserved, 25 required"):
            reserve_candidate_pass(
                store, _PREFIX, workflow, record, "feature_policy_preprocessing"
            )


def test_budget_does_not_charge_baseline_reuse_or_rejected_attempts() -> None:
    store, workflow, record = _context(25)
    baseline = reserve_candidate_pass(store, _PREFIX, workflow, record, "preprocessing")
    _start(store, workflow, record, "preprocessing", {"candidateBudget": baseline})
    _start(
        store,
        workflow,
        record,
        "feature_policy_preprocessing",
        {"baselineAttemptId": "reused-baseline"},
    )
    _start(
        store,
        workflow,
        record,
        "feature_policy_preprocessing",
        {"candidateBudgetRejected": True},
    )
    assert (
        reserve_candidate_pass(store, _PREFIX, workflow, record, "preprocessing")
        == baseline
    )


@pytest.mark.parametrize(
    "mutation", ["count", "breakdown", "pass", "missing", "config"]
)
def test_budget_rejects_inconsistent_persisted_reservations(mutation: str) -> None:
    store, workflow, record = _context()
    reservation = reserve_candidate_pass(
        store, _PREFIX, workflow, record, "preprocessing"
    )
    _start(store, workflow, record, "preprocessing", {"candidateBudget": reservation})
    changed = {**reservation, "breakdown": dict(reservation["breakdown"])}
    if mutation == "count":
        changed["reserved"] = 1
    elif mutation == "breakdown":
        changed["breakdown"]["hvg"] = 0
    elif mutation == "pass":
        changed["logicalPass"] = "featureRevision"
    elif mutation == "config":
        record = record.model_copy(update={"configSha256": "b" * 64})
    inputs = {} if mutation == "missing" else {"candidateBudget": changed}
    _start(store, workflow, record, "preprocessing", inputs)
    with pytest.raises(ValueError, match=r"reservation.*differs|lacks its candidate"):
        reserve_candidate_pass(store, _PREFIX, workflow, record, "preprocessing")


def test_budget_keeps_unused_reservations_and_requires_baseline_for_revision() -> None:
    store, workflow, record = _context(25)
    with pytest.raises(ValueError, match="requires a reserved baseline"):
        reserve_candidate_pass(
            store, _PREFIX, workflow, record, "feature_policy_preprocessing"
        )

    reservation = reserve_candidate_pass(
        store, _PREFIX, workflow, record, "preprocessing"
    )
    started = journal._start_attempt(
        store.zw,
        _PREFIX,
        workflow.workflowRunId,
        "preprocessing",
        record,
        [],
        inputs={"candidateBudget": reservation},
    )
    journal._save_outcome(
        store.zw,
        _PREFIX,
        journal._complete_attempt(
            started, status="done", outputs={"candidateCount": 1}
        ),
    )
    with pytest.raises(ValueError, match="25 slots already reserved"):
        reserve_candidate_pass(
            store, _PREFIX, workflow, record, "feature_policy_preprocessing"
        )


@pytest.mark.parametrize(
    "limit,stage", [(24, "preprocessing"), (25, "feature_policy_preprocessing")]
)
def test_preprocessing_budget_rejection_precedes_qc_and_candidate_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
    stage: WorkflowStageName,
) -> None:
    store = DataStore(str(create_store(tmp_path / "budget.zarr")), default_assay="RNA")
    workflow = create_agent_workflow(store)
    orchestrator = AgentOrchestrator(
        object(), config=AutomatedWorkflowConfig(maxCandidateEvaluations=limit)
    )
    request = AutomatedWorkflowRequest(
        sourcePath=str(store.zarr_loc),
        zarrPath=str(store.zarr_loc),
        studyContext="Budget admission regression.",
        studyObjective="Compare RNA populations.",
        primaryAssay="RNA",
        markerAssay="RNA",
        analysisAssays=["RNA"],
    )
    record = orchestrator.initialize_request(store, workflow, request)
    prefix = journal._ensure_orchestration_store(store)
    if stage == "feature_policy_preprocessing":
        reservation = reserve_candidate_pass(
            store, prefix, workflow, record, "preprocessing"
        )
        journal._start_attempt(
            store.zw,
            prefix,
            workflow.workflowRunId,
            "preprocessing",
            record,
            [],
            inputs={"candidateBudget": reservation},
        )

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Over-budget preprocessing must stop before numerical work")

    monkeypatch.setattr(orchestrator, "apply_cell_qc", unexpected)
    monkeypatch.setattr(orchestrator, "preprocess_assay", unexpected)
    outcome, handoffs, _ = orchestrator.preprocessing_stage(
        store,
        workflow,
        record,
        [],
        AutomatedPreprocessingPlan.get_example(),
        ExperimentalContextResult.get_example().model_copy(
            update={"htoIdentityArtifacts": []}
        ),
        StudyContract.get_blank(),
        {},
        stage_name=stage,
    )
    assert outcome.status == "failed"
    assert "Candidate budget exceeded" in (outcome.error or "")
    assert handoffs == []
    assert outcome.inputs == {"candidateBudgetRejected": True}
