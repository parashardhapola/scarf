"""Public controller for resumable automated Scarf agent workflows."""

import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import zarr

from ...datastore.datastore import DataStore
from ...datastore.summary import summarize_zarr_readonly
from ...storage.stores import zarr_root_path
from ...utils.logging import logger
from .. import record_io
from ..experimental_context.study import StudyContract
from ..ingest import IngestResult, detect_format, ingest
from ..ingest.manifest import DatasetManifest, inspect_h5ad_manifest
from ..persistence.contracts import AgentWorkflowRun
from ..persistence.decisions import load_latest_decision_workflow_snapshot
from ..persistence.reports import (
    create_agent_workflow,
    finalize_agent_workflow,
    load_agent_report,
    load_agent_workflow,
)
from . import journal
from .context import ContextStagesMixin
from .finalization import FinalizationStagesMixin
from .models import (
    _RUN_ID_PATTERN,
    _STAGE_ORDER,
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
    AutomatedWorkflowResumeRequest,
    AutomatedWorkflowStatus,
    FinalAnalysisHandoff,
    OrchestrationRequestRecord,
    OrchestrationResumeRecord,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageName,
)
from .preprocessing import PreprocessingStagesMixin
from .rna import (
    selected_rna_assay,
    validate_rna_directions,
    validate_rna_request_fields,
    validate_saved_rna_history,
)
from .tuning import TuningStagesMixin


def _generate_completed_report(
    store: DataStore,
    workflow: AgentWorkflowRun,
) -> None:
    """Generate a local report without changing the completed workflow result."""
    if workflow.status != "completed":
        return
    try:
        if zarr_root_path(store.z) is None:
            return
        from ..report.generator import generate_agent_report

        report_path = generate_agent_report(store, workflow.workflowRunId)
        relative_path = os.path.relpath(report_path, start=Path.cwd())
        logger.info(
            f"Workflow {workflow.workflowRunId}: local HTML report saved to "
            f"{relative_path}"
        )
        print(f"Agent workflow report: {relative_path}")
    except Exception as exc:
        logger.warning(
            f"Workflow {workflow.workflowRunId}: local HTML report generation "
            f"failed ({type(exc).__name__}: {exc})"
        )


class AgentOrchestrator(
    ContextStagesMixin,
    PreprocessingStagesMixin,
    TuningStagesMixin,
    FinalizationStagesMixin,
):
    """Run one bounded, persisted single-RNA analysis workflow."""

    def __init__(
        self,
        model: Any,
        *,
        config: AutomatedWorkflowConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or AutomatedWorkflowConfig()

    def run(self, request: AutomatedWorkflowRequest) -> AutomatedWorkflowResult:
        """Ingest the request and continue until completion or a persisted pause."""
        result = self._run(request)
        if result.status == "failed":
            logger.error(
                f"RNA analysis failed during {result.currentStage}: "
                + "; ".join(
                    result.notes or ["See the saved stage outcome for details."]
                )
            )
        return result

    def _run(self, request: AutomatedWorkflowRequest) -> AutomatedWorkflowResult:
        try:
            validate_rna_request_fields(request)
        except ValueError as exc:
            return AutomatedWorkflowResult(notes=[str(exc)])
        format_name = detect_format(request.sourcePath)
        dataset_manifest: DatasetManifest | None = None
        logger.info(
            f"Starting automated agent workflow from {format_name!r} input "
            f"(workspace={request.workspace is not None})"
        )
        if request.workspace is not None and format_name != "zarr":
            logger.warning(
                "Automated agent workflow rejected a workspace for a converted input"
            )
            return AutomatedWorkflowResult(
                status="failed",
                currentStage="ingest",
                notes=[
                    "workspace is supported for existing Zarr inputs; converted "
                    "inputs create their dataset at the root"
                ],
            )
        if format_name == "zarr" and request.zarrPath is not None:
            source_path = Path(request.sourcePath).resolve()
            requested_path = Path(request.zarrPath).resolve()
            if source_path != requested_path:
                logger.warning(
                    "Automated agent workflow rejected an implicit Zarr copy"
                )
                return AutomatedWorkflowResult(
                    status="failed",
                    currentStage="ingest",
                    notes=["An existing Zarr input cannot be copied implicitly"],
                )
        if format_name == "h5ad":
            matrix_key = request.ingestDirections.get("matrixKey")
            try:
                dataset_manifest = inspect_h5ad_manifest(
                    request.sourcePath,
                    source_uri=(
                        str(request.ingestDirections["sourceUri"])
                        if request.ingestDirections.get("sourceUri") is not None
                        else request.sourcePath
                    ),
                    author_label_policy=request.authorLabelPolicy,
                    matrix_key=str(matrix_key) if matrix_key is not None else None,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                return AutomatedWorkflowResult(
                    status="failed",
                    currentStage="ingest",
                    notes=[f"CELLxGENE manifest inspection failed: {exc}"],
                )
            if dataset_manifest.declaredBatchColumns:
                experimental_directions = dict(request.experimentalDirections)
                raw_batch_columns = experimental_directions.get("batchColumns")
                if raw_batch_columns is None:
                    experimental_directions["batchColumns"] = list(
                        dataset_manifest.declaredBatchColumns
                    )
                elif not isinstance(raw_batch_columns, list) or any(
                    not isinstance(value, str) or not value.strip()
                    for value in raw_batch_columns
                ):
                    return AutomatedWorkflowResult(
                        status="failed",
                        currentStage="ingest",
                        datasetManifest=dataset_manifest,
                        notes=[
                            "experimentalDirections.batchColumns must be a list "
                            "of exact observation-column names"
                        ],
                    )
                elif not set(dataset_manifest.declaredBatchColumns).issubset(
                    raw_batch_columns
                ):
                    return AutomatedWorkflowResult(
                        status="failed",
                        currentStage="ingest",
                        datasetManifest=dataset_manifest,
                        notes=[
                            "experimentalDirections.batchColumns must include the "
                            "CELLxGENE uns/batch_condition columns"
                        ],
                    )
                request = request.model_copy(
                    update={"experimentalDirections": experimental_directions}
                )
            manifest_decision = dataset_manifest.decision
            if manifest_decision.status == "needsInput":
                if self.config.inputPolicy == "unattended":
                    return AutomatedWorkflowResult(
                        status="abstained",
                        currentStage="ingest",
                        datasetManifest=dataset_manifest,
                        limitations=list(dataset_manifest.priorFiltering.limitations),
                        unresolvedClaims=[manifest_decision.summary],
                        notes=[
                            "The unattended workflow abstained because the count "
                            "matrix was ambiguous."
                        ],
                    )
                return AutomatedWorkflowResult(
                    status="needsInput",
                    currentStage="ingest",
                    datasetManifest=dataset_manifest,
                    needsInput=WorkflowNeedsInput(
                        questions=[
                            WorkflowQuestion(
                                questionId="datasetMatrixKey",
                                question=(
                                    manifest_decision.summary
                                    + ". Rerun with ingestDirections.matrixKey set "
                                    "to the selected option."
                                ),
                                options=list(manifest_decision.options),
                                evidenceIds=list(manifest_decision.evidenceIds),
                            )
                        ]
                    ),
                    limitations=list(dataset_manifest.priorFiltering.limitations),
                )
            if manifest_decision.status == "abstained":
                return AutomatedWorkflowResult(
                    status="abstained",
                    currentStage="ingest",
                    datasetManifest=dataset_manifest,
                    limitations=list(dataset_manifest.priorFiltering.limitations),
                    unresolvedClaims=[manifest_decision.summary],
                    notes=[
                        "The count-dependent RNA workflow did not run because its "
                        "input contract is not satisfied."
                    ],
                )
            selected_matrix = manifest_decision.selectedMatrixKey
            if selected_matrix is None:
                raise RuntimeError("Supported manifest lacks a selected matrix")
            ingest_directions = {
                **request.ingestDirections,
                "matrixKey": selected_matrix,
            }
            request = request.model_copy(update={"ingestDirections": ingest_directions})

        if format_name == "zarr":
            zarr_path = str(Path(request.sourcePath).resolve())
            effective_request = request.model_copy(update={"zarrPath": zarr_path})
            try:
                summary = summarize_zarr_readonly(
                    zarr_path,
                    workspace=request.workspace,
                )
            except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                return AutomatedWorkflowResult(
                    status="failed",
                    currentStage="ingest",
                    zarrPath=zarr_path,
                    notes=[f"Opening the requested RNA store failed: {exc}"],
                )
            ingest_result = IngestResult(
                status="done",
                format="zarr",
                zarrPath=zarr_path,
                assayNames=[assay.name for assay in summary.assays],
                summary=summary.to_dict(),
                actions=["summarize_zarr"],
            )
        else:
            ingest_result = ingest(
                path=request.sourcePath,
                zarrPath=request.zarrPath,
                model=self.model,
                directions=request.ingestDirections,
            )
            if ingest_result.zarrPath is not None:
                zarr_path = str(Path(ingest_result.zarrPath).resolve())
                ingest_result = ingest_result.model_copy(update={"zarrPath": zarr_path})
            effective_request = request.model_copy(
                update={"zarrPath": ingest_result.zarrPath}
            )
        logger.info(
            f"Automated workflow ingest returned status={ingest_result.status!r}, "
            f"format={ingest_result.format!r}, assays={len(ingest_result.assayNames)}"
        )
        if ingest_result.status != "done" or ingest_result.zarrPath is None:
            needs_input = None
            if ingest_result.needsInput is not None:
                needs_input = WorkflowNeedsInput(
                    questions=[
                        WorkflowQuestion(
                            questionId="ingest",
                            question=ingest_result.needsInput.question,
                            options=list(ingest_result.needsInput.options),
                            evidenceIds=list(ingest_result.needsInput.evidenceIds),
                        )
                    ]
                )
            if needs_input is not None and self.config.inputPolicy == "unattended":
                return AutomatedWorkflowResult(
                    status="abstained",
                    currentStage="ingest",
                    zarrPath=ingest_result.zarrPath,
                    unresolvedClaims=[
                        question.question for question in needs_input.questions
                    ],
                    notes=[
                        *ingest_result.notes,
                        "The unattended workflow abstained instead of waiting for "
                        "an ingest decision.",
                    ],
                )
            return AutomatedWorkflowResult(
                status=("needsInput" if needs_input is not None else "failed"),
                currentStage="ingest",
                zarrPath=ingest_result.zarrPath,
                needsInput=needs_input,
                notes=list(ingest_result.notes),
            )

        try:
            selected = selected_rna_assay(
                effective_request,
                {
                    value["name"]: value["assay_type"]
                    for value in (ingest_result.summary or {}).get("assays", [])
                },
            )
            effective_request = effective_request.model_copy(
                update={
                    "primaryAssay": selected,
                    "markerAssay": selected,
                    "analysisAssays": [selected],
                }
            )
            store = self.open_store(ingest_result.zarrPath, effective_request)
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            terminal = (
                finalize_agent_workflow(
                    ingest_result.zarrPath,
                    ingest_result.workflowRun.workflowRunId,
                    status="failed",
                    message=str(exc),
                    workspace=request.workspace,
                )
                if ingest_result.workflowRun is not None
                else None
            )
            return AutomatedWorkflowResult(
                zarrPath=ingest_result.zarrPath,
                workflowRun=terminal,
                notes=[str(exc)],
            )
        ignored = [name for name in store.assay_names if name != selected]
        logger.info(
            f"RNA analysis: selected assay {selected!r}"
            + (f"; ignored other assays {ignored}" if ignored else "")
        )
        workflow = ingest_result.workflowRun or create_agent_workflow(store)
        logger.info(
            f"Continuing automated workflow {workflow.workflowRunId} with "
            f"{len(store.assay_names)} datastore assays"
        )
        request_record = self.initialize_request(store, workflow, effective_request)
        prefix = journal._ensure_orchestration_store(store)
        self.record_ingest_stage(
            store,
            prefix,
            workflow,
            request_record,
            ingest_result,
            dataset_manifest,
        )
        return self._continue(
            store,
            workflow,
            request_record,
            answers={},
        )

    def resume(
        self,
        request: AutomatedWorkflowResumeRequest,
    ) -> AutomatedWorkflowResult:
        """Resume a running workflow after validating its immutable request."""
        result = self._resume(request)
        if result.status == "failed":
            logger.error(
                f"RNA analysis failed during {result.currentStage}: "
                + "; ".join(
                    result.notes or ["See the saved stage outcome for details."]
                )
            )
        return result

    def _resume(
        self,
        request: AutomatedWorkflowResumeRequest,
    ) -> AutomatedWorkflowResult:
        directions = request.answers.get("experimentalDirections")
        if isinstance(directions, Mapping):
            validate_rna_directions(directions)
        logger.info(
            f"Resuming automated workflow {request.workflowRunId} with "
            f"{len(request.answers)} answer field(s)"
        )
        record, store = self.load_request_for_resume(request)
        workflow = load_agent_workflow(
            store,
            request.workflowRunId,
            workspace=request.workspace,
        )
        if workflow.status != "running":
            logger.warning(
                f"Automated workflow {workflow.workflowRunId} cannot resume from "
                f"status={workflow.status!r}"
            )
            prefix = journal._ensure_orchestration_store(store)
            if journal._load_terminal_result(store, prefix, workflow) is not None:
                raise RuntimeError(
                    f"Cannot resume a workflow with status {workflow.status!r}"
                )
            return self.repair_terminal_result(store, workflow, record)
        prefix = journal._ensure_orchestration_store(store)
        outcomes = [
            outcome
            for stage in _STAGE_ORDER
            for outcome in journal._stage_outcomes(
                store.zw, prefix, workflow.workflowRunId, stage
            )
        ]
        starts = [
            started
            for stage in _STAGE_ORDER
            for started in journal._stage_starts(
                store.zw, prefix, workflow.workflowRunId, stage
            )
        ]
        completed_attempt_ids = {
            (outcome.stage, outcome.attemptId) for outcome in outcomes
        }
        interrupted_starts = [
            started
            for started in starts
            if (started.stage, started.attemptId) not in completed_attempt_ids
        ]
        logger.info(
            f"Workflow {workflow.workflowRunId} resume scan found "
            f"{len(outcomes)} outcome(s) and {len(interrupted_starts)} "
            "interrupted attempt(s)"
        )
        latest_outcome = (
            max(
                outcomes,
                key=lambda value: (
                    _STAGE_ORDER.index(value.stage),
                    value.startedAtNs,
                    value.attemptId,
                ),
            )
            if outcomes
            else None
        )
        latest_interrupted = (
            max(
                interrupted_starts,
                key=lambda value: (
                    _STAGE_ORDER.index(value.stage),
                    value.startedAtNs,
                    value.attemptId,
                ),
            )
            if interrupted_starts
            else None
        )
        interrupted_lineage = (
            latest_interrupted.inputs.get("resumeLineage")
            if latest_interrupted is not None
            else None
        )
        interrupted_answers_latest_pause = bool(
            latest_interrupted is not None
            and latest_outcome is not None
            and latest_interrupted.stage == latest_outcome.stage
            and isinstance(interrupted_lineage, Mapping)
            and interrupted_lineage.get("answeredAttempt")
            == journal._parent_link(latest_outcome).model_dump(mode="json")
        )
        active_interrupted = (
            latest_interrupted
            if latest_interrupted is not None
            and (
                latest_outcome is None
                or _STAGE_ORDER.index(latest_interrupted.stage)
                > _STAGE_ORDER.index(latest_outcome.stage)
                or interrupted_answers_latest_pause
                or (
                    latest_interrupted.stage == latest_outcome.stage
                    and (
                        latest_interrupted.startedAtNs,
                        latest_interrupted.attemptId,
                    )
                    > (latest_outcome.startedAtNs, latest_outcome.attemptId)
                )
            )
            else None
        )
        latest_paused = (
            latest_outcome
            if latest_outcome is not None
            and latest_outcome.status == "needsInput"
            and active_interrupted is None
            else None
        )
        effective_answers = dict(request.answers)
        inherited_resume: OrchestrationResumeRecord | None = None
        if active_interrupted is not None:
            lineage = active_interrupted.inputs.get("resumeLineage")
            if lineage is not None:
                if not isinstance(lineage, Mapping):
                    raise ValueError("Interrupted stage resumeLineage is malformed")
                inherited_resume_id = lineage.get("resumeId")
                if (
                    not isinstance(inherited_resume_id, str)
                    or _RUN_ID_PATTERN.fullmatch(inherited_resume_id) is None
                ):
                    raise ValueError("Interrupted stage resumeId is malformed")
                inherited_resume = journal._validated_resume_record(
                    store,
                    prefix,
                    workflow.workflowRunId,
                    inherited_resume_id,
                )
                expected_answered_attempt = (
                    inherited_resume.answeredAttempt.model_dump(mode="json")
                    if inherited_resume.answeredAttempt is not None
                    else None
                )
                if (
                    lineage.get("answeredAttempt") != expected_answered_attempt
                    or lineage.get("questionIds") != inherited_resume.questionIds
                ):
                    raise ValueError(
                        "Interrupted stage resumeLineage does not match its resume record"
                    )
                if request.answers and request.answers != inherited_resume.answers:
                    raise ValueError(
                        "Cannot change answers for an in-flight logical invocation"
                    )
                effective_answers = dict(inherited_resume.answers)
        answered_attempt = (
            journal._parent_link(latest_paused) if latest_paused is not None else None
        )
        question_ids = (
            [question.questionId for question in latest_paused.needsInput.questions]
            if latest_paused is not None and latest_paused.needsInput is not None
            else []
        )
        if inherited_resume is not None:
            answered_attempt = inherited_resume.answeredAttempt
            question_ids = list(inherited_resume.questionIds)
        elif latest_paused is None and request.answers:
            raise ValueError(
                "Cannot provide resume answers: no active persisted questions"
            )
        resume_record = OrchestrationResumeRecord(
            workflowRunId=workflow.workflowRunId,
            resumeId=uuid.uuid4().hex,
            createdAtNs=time.time_ns(),
            answeredAttempt=answered_attempt,
            questionIds=question_ids,
            answers=effective_answers,
        )
        resume_record = resume_record.model_copy(
            update={"contentSha256": journal._record_checksum(resume_record)}
        )
        journal._write_model_once(
            store.zw,
            journal._resume_key(prefix, workflow.workflowRunId, resume_record.resumeId),
            resume_record,
        )
        logger.info(
            f"Persisted resume {resume_record.resumeId} for workflow "
            f"{workflow.workflowRunId} (questions={len(question_ids)})"
        )
        if latest_paused is not None:
            answer_errors = journal._resume_answer_errors(
                latest_paused, effective_answers
            )
            if answer_errors:
                logger.warning(
                    f"Resume answers for workflow {workflow.workflowRunId} did not "
                    "satisfy the persisted questions"
                )
                result = journal.paused_or_failed_result(
                    store,
                    workflow,
                    record,
                    latest_paused,
                )
                result = result.model_copy(
                    update={"notes": [*result.notes, *answer_errors]}
                )
                return result.model_copy(
                    update={"contentSha256": journal._record_checksum(result)}
                )
        return self._continue(
            store,
            workflow,
            record,
            answers=effective_answers,
            resume_record=resume_record,
        )

    def cancel(
        self,
        request: AutomatedWorkflowResumeRequest,
        *,
        message: str = "Automated workflow cancelled by the caller",
    ) -> AutomatedWorkflowResult:
        """Finalize one running automated workflow as abandoned."""
        logger.info(f"Cancelling automated workflow {request.workflowRunId}")
        record, store = self.load_request_for_resume(request)
        workflow = load_agent_workflow(
            store,
            request.workflowRunId,
            workspace=request.workspace,
        )
        if workflow.status != "running":
            prefix = journal._ensure_orchestration_store(store)
            if (
                workflow.status == "abandoned"
                and journal._load_terminal_result(store, prefix, workflow) is None
            ):
                return self.repair_terminal_result(store, workflow, record)
            raise RuntimeError(
                f"Cannot cancel a workflow with status {workflow.status!r}"
            )
        terminal = finalize_agent_workflow(
            store,
            workflow.workflowRunId,
            status="abandoned",
            message=message,
        )
        prefix = journal._ensure_orchestration_store(store)
        observed = [
            outcome
            for stage in _STAGE_ORDER
            for outcome in journal._stage_outcomes(
                store.zw, prefix, workflow.workflowRunId, stage
            )
        ]
        current_stage: WorkflowStageName = (
            max(observed, key=lambda value: value.startedAtNs).stage
            if observed
            else "ingest"
        )
        result = AutomatedWorkflowResult(
            status="abandoned",
            currentStage=current_stage,
            zarrPath=str(store.zarr_loc),
            workflowRun=terminal,
            reportReferences=list(terminal.reports),
            notes=[message],
        )
        result = result.model_copy(
            update={"contentSha256": journal._record_checksum(result)}
        )
        logger.info(
            f"Automated workflow {workflow.workflowRunId} was abandoned at "
            f"stage={current_stage!r}"
        )
        return journal._persist_terminal_result(store, prefix, terminal, result)

    def open_store(
        self,
        zarr_path: str,
        request: AutomatedWorkflowRequest,
    ) -> DataStore:
        default_assay = cast(
            str | None,
            request.ingestDirections.get("defaultAssay") or request.primaryAssay,
        )
        return DataStore(
            zarr_path,
            default_assay=default_assay,
            min_features_per_cell=-1,
            mito_pattern="",
            ribo_pattern="",
            zarr_mode="r+",
            workspace=request.workspace,
        )

    def initialize_request(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request: AutomatedWorkflowRequest,
    ) -> OrchestrationRequestRecord:
        prefix = journal._ensure_orchestration_store(store)
        request_checksum = journal._sha256_model(request)
        config_checksum = journal._sha256_model(self.config)
        record = OrchestrationRequestRecord(
            workflowRunId=workflow.workflowRunId,
            createdAtNs=time.time_ns(),
            request=request,
            config=self.config,
            requestSha256=request_checksum,
            configSha256=config_checksum,
        )
        record = record.model_copy(
            update={"contentSha256": journal._record_checksum(record)}
        )
        journal._write_model_once(
            store.zw,
            journal._request_key(prefix, workflow.workflowRunId),
            record,
        )
        logger.debug(
            f"Persisted immutable request for workflow {workflow.workflowRunId}"
        )
        return record

    def load_request_for_resume(
        self,
        request: AutomatedWorkflowResumeRequest,
    ) -> tuple[OrchestrationRequestRecord, DataStore]:
        logger.debug(f"Loading immutable request for workflow {request.workflowRunId}")
        root = zarr.open_group(request.zarrPath, mode="r")
        active = root if request.workspace is None else root[request.workspace]
        if not isinstance(active, zarr.Group):
            raise ValueError("The requested workspace is not a Zarr group")
        root_path = str(getattr(active, "path", "")).strip("/")
        prefix = record_io.join_key(root_path, "agents", "orchestrations")
        record = cast(
            OrchestrationRequestRecord,
            journal._read_model(
                active,
                journal._request_key(prefix, request.workflowRunId),
                OrchestrationRequestRecord,
            ),
        )
        if record.workflowRunId != request.workflowRunId:
            raise ValueError("Stored orchestration request has a different workflow")
        if (
            Path(cast(str, record.request.zarrPath)).resolve()
            != Path(request.zarrPath).resolve()
        ):
            raise ValueError("Resume zarrPath does not match the stored request")
        if record.request.workspace != request.workspace:
            raise ValueError("Resume workspace does not match the stored request")
        if record.requestSha256 != journal._sha256_model(record.request):
            raise ValueError("Stored orchestration request checksum is invalid")
        if record.configSha256 != journal._sha256_model(record.config):
            raise ValueError("Stored orchestration config checksum is invalid")
        if record.contentSha256 != journal._record_checksum(record):
            raise ValueError("Stored orchestration request envelope is invalid")
        summary = summarize_zarr_readonly(request.zarrPath, workspace=request.workspace)
        selected = selected_rna_assay(
            record.request,
            {assay.name: assay.assay_type for assay in summary.assays},
        )
        validate_saved_rna_history(active, prefix, request.workflowRunId, selected)
        store = self.open_store(request.zarrPath, record.request)
        return record, store

    def repair_terminal_result(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
    ) -> AutomatedWorkflowResult:
        """Load or reconstruct the JSON result after terminal finalization."""
        prefix = journal._ensure_orchestration_store(store)
        existing = journal._load_terminal_result(store, prefix, workflow)
        if existing is not None:
            logger.debug(
                f"Loaded terminal result for workflow {workflow.workflowRunId}"
            )
            return existing

        logger.warning(
            f"Repairing missing terminal result for workflow {workflow.workflowRunId}"
        )

        observed = [
            outcome
            for stage in _STAGE_ORDER
            for outcome in journal._stage_outcomes(
                store.zw,
                prefix,
                workflow.workflowRunId,
                stage,
            )
        ]
        if not observed:
            raise RuntimeError("Terminal workflow has no persisted stage outcomes")
        by_identity = {
            (outcome.stage, outcome.attemptId): outcome for outcome in observed
        }

        def validated_chain(
            terminal: WorkflowStageAttempt,
        ) -> dict[WorkflowStageName, WorkflowStageAttempt] | None:
            chain: dict[WorkflowStageName, WorkflowStageAttempt] = {}
            current = terminal
            while True:
                stage_index = _STAGE_ORDER.index(current.stage)
                if current.stage in chain:
                    raise ValueError("Stage lineage contains a cycle")
                if not journal._stage_outcome_resolves(
                    store,
                    prefix,
                    workflow.workflowRunId,
                    request_record,
                    current,
                ):
                    return None
                chain[current.stage] = current
                if stage_index == 0:
                    if current.parentAttempts:
                        raise ValueError("The ingest stage cannot have a parent")
                    return chain
                if len(current.parentAttempts) != 1:
                    raise ValueError("Every post-ingest stage must have one parent")
                parent_link = current.parentAttempts[0]
                if (
                    workflow.status != "abandoned"
                    and parent_link.stage != _STAGE_ORDER[stage_index - 1]
                ):
                    raise ValueError("Terminal stage lineage skips a workflow stage")
                if _STAGE_ORDER.index(parent_link.stage) >= stage_index:
                    raise ValueError("Stage lineage does not move toward ingest")
                parent = by_identity.get((parent_link.stage, parent_link.attemptId))
                if (
                    parent is None
                    or parent.status != "done"
                    or parent.contentSha256 != parent_link.contentSha256
                ):
                    return None
                current = parent

        if workflow.status == "completed":
            terminal_candidates = [
                outcome
                for outcome in observed
                if outcome.stage == "analysis_finalization" and outcome.status == "done"
            ]
        elif workflow.status == "failed":
            terminal_candidates = [
                outcome for outcome in observed if outcome.status == "failed"
            ]
        else:
            terminal_candidates = list(observed)
        terminal_candidates.sort(
            key=lambda value: (value.startedAtNs, value.attemptId),
            reverse=True,
        )
        terminal_outcome: WorkflowStageAttempt | None = None
        validated_done: dict[WorkflowStageName, WorkflowStageAttempt] = {}
        for candidate in terminal_candidates:
            chain = validated_chain(candidate)
            if chain is not None:
                terminal_outcome = candidate
                validated_done = {
                    stage: outcome
                    for stage, outcome in chain.items()
                    if outcome.status == "done"
                }
                break
        if terminal_outcome is None:
            raise RuntimeError(
                "Terminal automated workflow lacks one valid persisted stage chain"
            )

        preprocessing_plan: AutomatedPreprocessingPlan | None = None
        plan_outcome = validated_done.get("preprocessing_plan")
        if plan_outcome is not None and "preprocessingPlan" in plan_outcome.outputs:
            preprocessing_plan = AutomatedPreprocessingPlan.model_validate(
                plan_outcome.outputs["preprocessingPlan"]
            )
        feature_preprocessing = validated_done.get("feature_policy_preprocessing")
        if (
            feature_preprocessing is not None
            and "resolvedPreprocessingPlan" in feature_preprocessing.outputs
        ):
            preprocessing_plan = AutomatedPreprocessingPlan.model_validate(
                feature_preprocessing.outputs["resolvedPreprocessingPlan"]
            )

        dataset_manifest: DatasetManifest | None = None
        ingest_stage = validated_done.get("ingest")
        if ingest_stage is not None and ingest_stage.outputs.get("datasetManifest"):
            dataset_manifest = DatasetManifest.model_validate(
                ingest_stage.outputs["datasetManifest"]
            )

        study_contract: StudyContract | None = None
        context_outcome = validated_done.get("experimental_context")
        if context_outcome is not None and "studyContract" in context_outcome.outputs:
            study_contract = StudyContract.model_validate(
                context_outcome.outputs["studyContract"]
            )

        final_analysis: FinalAnalysisHandoff | None = None
        finalization_outcome = validated_done.get("analysis_finalization")
        if (
            finalization_outcome is not None
            and "finalAnalysis" in finalization_outcome.outputs
        ):
            final_analysis = FinalAnalysisHandoff.model_validate(
                finalization_outcome.outputs["finalAnalysis"]
            )
            persisted_handoff = journal.load_final_analysis_handoff(
                store,
                prefix,
                workflow.workflowRunId,
                final_analysis.handoffId,
            )
            if persisted_handoff != final_analysis:
                raise ValueError("Final handoff journal content differs from outcome")

        for reference in workflow.reports:
            load_agent_report(store, reference)
        notes = [workflow.finalizationMessage] if workflow.finalizationMessage else []
        verification_summary: list[str] = []
        decision_run_id: str | None = None
        if workflow.status == "completed":
            decision_snapshot = load_latest_decision_workflow_snapshot(
                store,
                workflow.workflowRunId,
                workspace=request_record.request.workspace,
            )
            if (
                final_analysis is None
                or decision_snapshot.workflow.status != "completed"
                or decision_snapshot.workflow.finalHandoffId != final_analysis.handoffId
            ):
                raise ValueError(
                    "Terminal orchestration and decision ledger do not resolve"
                )
            decision_run_id = workflow.workflowRunId
            verification_by_record = {
                value.decisionRecordId: value
                for value in decision_snapshot.workflow.verificationRecords
            }
            verification_summary = [
                (
                    f"{record.decisionId}: "
                    f"{len(verification_by_record[record.recordId].checks)} "
                    f"deterministic checks passed ({record.source})."
                )
                for record in decision_snapshot.workflow.active_decision_records()
            ]
        result = AutomatedWorkflowResult(
            status=cast(AutomatedWorkflowStatus, workflow.status),
            currentStage=terminal_outcome.stage,
            zarrPath=str(store.zarr_loc),
            workflowRun=workflow,
            reportReferences=list(workflow.reports),
            datasetManifest=dataset_manifest,
            preprocessingPlan=preprocessing_plan,
            studyContract=study_contract,
            finalAnalysis=final_analysis,
            finalHandoffId=(
                final_analysis.handoffId if final_analysis is not None else None
            ),
            decisionRunId=decision_run_id,
            verificationSummary=verification_summary,
            notes=notes,
        )
        result = result.model_copy(
            update={"contentSha256": journal._record_checksum(result)}
        )
        persisted = journal._persist_terminal_result(store, prefix, workflow, result)
        _generate_completed_report(store, workflow)
        return persisted

    def _continue(
        self,
        store: DataStore,
        workflow: AgentWorkflowRun,
        request_record: OrchestrationRequestRecord,
        *,
        answers: Mapping[str, Any],
        resume_record: OrchestrationResumeRecord | None = None,
    ) -> AutomatedWorkflowResult:
        """Continue the stage machine from the latest validated checkpoint."""
        logger.info(f"Running stage sequence for workflow {workflow.workflowRunId}")
        prefix = journal._ensure_orchestration_store(store)
        self._load_or_create_decision_workflow(store, request_record)
        ingest_outcome = journal._validated_done_outcome(
            store,
            prefix,
            workflow.workflowRunId,
            "ingest",
            request_record,
            [],
        )
        if ingest_outcome is None:
            raise RuntimeError("The persisted ingest stage is missing")
        dataset_manifest = (
            DatasetManifest.model_validate(ingest_outcome.outputs["datasetManifest"])
            if ingest_outcome.outputs.get("datasetManifest") is not None
            else None
        )
        cell_selection = ingest_outcome.artifacts.get("cellSelection")
        if cell_selection is None or cell_selection.kind != "cell_selection":
            raise RuntimeError(
                "The persisted ingest stage lacks an exact cell selection"
            )
        parents = [journal._parent_link(ingest_outcome)]

        enrichment_outcome, enrichment = self.data_enrichment_stage(
            store,
            workflow,
            request_record,
            parents,
            cell_selection,
            answers,
            resume_record=resume_record,
        )
        if enrichment_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                enrichment_outcome,
                dataset_manifest=dataset_manifest,
            )
        parents = [journal._parent_link(enrichment_outcome)]

        hto_outcome = self._hto_stage(
            store,
            workflow,
            request_record,
            parents,
            enrichment,
            cell_selection,
            resume_record=resume_record,
        )
        if hto_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                hto_outcome,
                dataset_manifest=dataset_manifest,
            )
        quality_metric_artifacts = self._named_stage_artifacts(
            hto_outcome,
            "qualityMetricArtifacts",
            "quality_metric",
        )
        hto_identity_artifacts = self._named_stage_artifacts(
            hto_outcome,
            "htoIdentityArtifacts",
            "hto_identity",
        )
        parents = [journal._parent_link(hto_outcome)]

        context_outcome, experimental = self.experimental_context_stage(
            store,
            workflow,
            request_record,
            parents,
            cell_selection,
            enrichment_outcome.reportReferences[0],
            quality_metric_artifacts,
            hto_identity_artifacts,
            answers,
            resume_record=resume_record,
        )
        if context_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                context_outcome,
                dataset_manifest=dataset_manifest,
            )
        study_contract = StudyContract.model_validate(
            context_outcome.outputs["studyContract"]
        )
        parents = [journal._parent_link(context_outcome)]

        plan_outcome, preprocessing_plan = self.preprocessing_plan_stage(
            store,
            workflow,
            request_record,
            parents,
            enrichment,
            experimental,
            ingest_outcome,
            study_contract,
            answers,
            resume_record=resume_record,
        )
        if plan_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                plan_outcome,
                dataset_manifest=dataset_manifest,
                preprocessing_plan=preprocessing_plan,
                study_contract=study_contract,
            )
        parents = [journal._parent_link(plan_outcome)]

        (
            preprocessing_outcome,
            preprocessed,
            preprocessing_plan,
        ) = self.preprocessing_stage(
            store,
            workflow,
            request_record,
            parents,
            preprocessing_plan,
            experimental,
            study_contract,
            answers,
            resume_record=resume_record,
        )
        if preprocessing_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                preprocessing_outcome,
                dataset_manifest=dataset_manifest,
                preprocessing_plan=preprocessing_plan,
                study_contract=study_contract,
            )
        parents = [journal._parent_link(preprocessing_outcome)]

        tuning_outcome, tuning_report = self.parameter_tuning_stage(
            store,
            workflow,
            request_record,
            parents,
            preprocessing_plan,
            preprocessed,
            experimental,
            enrichment_outcome.reportReferences[0],
            context_outcome.reportReferences[0],
            answers,
            study_contract=study_contract,
            resume_record=resume_record,
        )
        if tuning_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                tuning_outcome,
                dataset_manifest=dataset_manifest,
                preprocessing_plan=preprocessing_plan,
                study_contract=study_contract,
            )
        baseline_preprocessing_outcome = preprocessing_outcome
        baseline_preprocessed = list(preprocessed)
        baseline_tuning_outcome = tuning_outcome
        baseline_tuning_report = tuning_report
        parents = [journal._parent_link(baseline_tuning_outcome)]

        (
            feature_review_outcome,
            preprocessing_plan,
            feature_policy_revised,
        ) = self.feature_policy_review_stage(
            store,
            workflow,
            request_record,
            parents,
            preprocessing_plan,
            baseline_tuning_report,
            answers,
            resume_record=resume_record,
        )
        if feature_review_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                feature_review_outcome,
                dataset_manifest=dataset_manifest,
                preprocessing_plan=preprocessing_plan,
                study_contract=study_contract,
            )
        parents = [journal._parent_link(feature_review_outcome)]

        if feature_policy_revised:
            (
                feature_preprocessing_outcome,
                preprocessed,
                preprocessing_plan,
            ) = self.preprocessing_stage(
                store,
                workflow,
                request_record,
                parents,
                preprocessing_plan,
                experimental,
                study_contract,
                answers,
                resume_record=resume_record,
                stage_name="feature_policy_preprocessing",
            )
        else:
            (
                feature_preprocessing_outcome,
                preprocessed,
                preprocessing_plan,
            ) = self.reuse_feature_policy_preprocessing_stage(
                store,
                workflow,
                request_record,
                parents,
                preprocessing_plan,
                baseline_preprocessing_outcome,
                baseline_preprocessed,
                resume_record=resume_record,
            )
        if feature_preprocessing_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                feature_preprocessing_outcome,
                dataset_manifest=dataset_manifest,
                preprocessing_plan=preprocessing_plan,
                study_contract=study_contract,
            )
        parents = [journal._parent_link(feature_preprocessing_outcome)]

        if feature_policy_revised:
            tuning_outcome, tuning_report = self.parameter_tuning_stage(
                store,
                workflow,
                request_record,
                parents,
                preprocessing_plan,
                preprocessed,
                experimental,
                enrichment_outcome.reportReferences[0],
                context_outcome.reportReferences[0],
                answers,
                study_contract=study_contract,
                resume_record=resume_record,
                stage_name="feature_policy_tuning",
            )
            tuning_reference = (
                tuning_outcome.reportReferences[0]
                if tuning_outcome.reportReferences
                else baseline_tuning_outcome.reportReferences[0]
            )
        else:
            tuning_outcome, tuning_report = self.reuse_feature_policy_tuning_stage(
                store,
                workflow,
                request_record,
                parents,
                baseline_tuning_outcome,
                baseline_tuning_report,
                resume_record=resume_record,
            )
            tuning_reference = baseline_tuning_outcome.reportReferences[0]
        if tuning_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                tuning_outcome,
                dataset_manifest=dataset_manifest,
                preprocessing_plan=preprocessing_plan,
                study_contract=study_contract,
            )
        parents = [journal._parent_link(tuning_outcome)]

        (
            analysis_review_outcome,
            tuning_report,
            tuning_reference,
        ) = self.analysis_review_stage(
            store,
            workflow,
            request_record,
            parents,
            preprocessing_plan,
            tuning_report,
            tuning_reference,
            study_contract,
            answers,
            resume_record=resume_record,
        )
        if analysis_review_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                analysis_review_outcome,
                dataset_manifest=dataset_manifest,
                preprocessing_plan=preprocessing_plan,
                study_contract=study_contract,
            )
        parents = [journal._parent_link(analysis_review_outcome)]

        finalization_outcome, final_analysis = self.analysis_finalization_stage(
            store,
            workflow,
            request_record,
            parents,
            preprocessing_plan,
            preprocessed,
            tuning_report,
            tuning_reference,
            study_contract,
            experimental=experimental,
            analysis_review_evidence=analysis_review_outcome.outputs,
            answers=answers,
            resume_record=resume_record,
        )
        if finalization_outcome.status != "done":
            return journal.paused_or_failed_result(
                store,
                workflow,
                request_record,
                finalization_outcome,
                dataset_manifest=dataset_manifest,
                preprocessing_plan=preprocessing_plan,
                study_contract=study_contract,
            )

        terminal = finalize_agent_workflow(
            store,
            workflow.workflowRunId,
            status="completed",
            message="Decision-driven Scarf analysis completed",
        )
        decision_snapshot = load_latest_decision_workflow_snapshot(
            store,
            workflow.workflowRunId,
            workspace=request_record.request.workspace,
        )
        if (
            decision_snapshot.workflow.status != "completed"
            or decision_snapshot.workflow.finalHandoffId != final_analysis.handoffId
        ):
            raise ValueError(
                "Completed orchestration and decision handoff identities differ"
            )
        verification_by_record = {
            value.decisionRecordId: value
            for value in decision_snapshot.workflow.verificationRecords
        }
        verification_summary = [
            (
                f"{record.decisionId}: "
                f"{len(verification_by_record[record.recordId].checks)} "
                f"deterministic checks passed ({record.source})."
            )
            for record in decision_snapshot.workflow.active_decision_records()
        ]
        completed = AutomatedWorkflowResult(
            status="completed",
            currentStage="analysis_finalization",
            zarrPath=str(store.zarr_loc),
            workflowRun=terminal,
            reportReferences=list(terminal.reports),
            datasetManifest=dataset_manifest,
            preprocessingPlan=preprocessing_plan,
            studyContract=study_contract,
            finalAnalysis=final_analysis,
            finalHandoffId=final_analysis.handoffId,
            decisionRunId=workflow.workflowRunId,
            verificationSummary=verification_summary,
            limitations=list(study_contract.limitations),
            unresolvedClaims=list(study_contract.unsupportedClaims),
            notes=["Decision-driven RNA analysis completed"],
        )
        completed = completed.model_copy(
            update={"contentSha256": journal._record_checksum(completed)}
        )
        logger.info(
            f"Automated workflow {workflow.workflowRunId} completed with "
            f"{len(terminal.reports)} report(s)"
        )
        persisted = journal._persist_terminal_result(
            store,
            prefix,
            terminal,
            completed,
        )
        _generate_completed_report(store, terminal)
        return persisted
