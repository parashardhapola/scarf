"""Top-level local agent report assembly."""

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...datastore.datastore import DataStore
from ...utils.logging import logger
from ..persistence.reports import load_agent_workflow
from .artifacts import (
    _collect_active_decisions,
    _collect_default_feature_inventories,
    _collect_history,
    _collect_hvg_evidence,
    _collect_reports,
    _load_completed_result,
    _local_root,
    _open_datastore,
)
from .contracts import _latest, _mapping, _selected_qc_profile
from .plots import _collect_final_artifacts, _collect_hvg_plots
from .rendering import (
    _render_analysis_document,
    _render_technical_document,
)


def _write_report_page(report_dir: Path, filename: str, document: str) -> Path:
    destination = report_dir / filename
    temporary = report_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(document, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def generate_agent_report(
    target: str | Path | DataStore,
    workflow_run_id: str,
    *,
    workspace: str | None = None,
) -> Path:
    """Generate a local HTML report for one completed automated workflow.

    The returned ``index.html`` opens the analysis and its decisions directly,
    with a secondary link to the technical details.
    Existing derived report files may be replaced; immutable agent and
    orchestration records are only read.
    """
    root = _local_root(target)
    resolved_workspace = (
        target.workspace if isinstance(target, DataStore) else workspace
    )
    if (
        isinstance(target, DataStore)
        and workspace is not None
        and workspace != target.workspace
    ):
        raise ValueError("workspace does not match the DataStore workspace")
    workflow = load_agent_workflow(
        target,
        workflow_run_id,
        workspace=resolved_workspace,
    )
    store = _open_datastore(target, root, workflow)
    prefix, result, request = _load_completed_result(store, workflow)
    reports = _collect_reports(store, result)
    stage_attempts, resumes = _collect_history(store, prefix, workflow, request)

    active_root = (
        root if workflow.workspace is None else (root / workflow.workspace).resolve()
    )
    if not active_root.is_relative_to(root):
        raise ValueError("Workflow workspace resolves outside the analysis store")
    report_dir = (
        active_root / "agents" / "runs" / workflow_run_id / "report"
    ).resolve()
    if not report_dir.is_relative_to(active_root):
        raise ValueError("Agent report path resolves outside the analysis store")
    plot_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_plan = (
        result.preprocessingPlan.model_dump(mode="json")
        if result.preprocessingPlan is not None
        else {}
    )
    experimental = _latest(reports, "experimental_context")
    selected_qc_profile = _selected_qc_profile(
        experimental,
        _mapping(preprocessing_plan.get("cellQc")),
    )
    cluster_counts, top_markers, plot_files, plot_notes = _collect_final_artifacts(
        store,
        result,
        plot_dir,
        qc_profile=selected_qc_profile,
    )
    hvg_evidence = _collect_hvg_evidence(
        store,
        stage_attempts,
        preprocessing_plan,
    )
    hvg_plots, hvg_plot_notes = _collect_hvg_plots(
        store,
        stage_attempts,
        preprocessing_plan,
        plot_dir,
    )
    plot_files.update(hvg_plots)
    plot_notes.extend(hvg_plot_notes)
    active_decisions = _collect_active_decisions(store, workflow_run_id)
    default_feature_inventories = _collect_default_feature_inventories(
        store,
        preprocessing_plan,
    )
    payload: dict[str, Any] = {
        "status": result.status,
        "currentStage": result.currentStage,
        "workflowRunId": workflow_run_id,
        "generatedAt": datetime.now(UTC).isoformat(),
        "request": request.request.model_dump(mode="json"),
        "effectiveConfig": request.config.model_dump(mode="json"),
        "workflowResult": result.model_dump(mode="json"),
        "reports": reports,
        "stageAttempts": stage_attempts,
        "workflowResumes": resumes,
        "clusterCounts": cluster_counts,
        "topMarkers": top_markers,
        "plotFiles": plot_files,
        "plotNotes": plot_notes,
        "hvgEvidence": hvg_evidence,
        "activeDecisions": active_decisions,
        "defaultFeatureInventories": default_feature_inventories,
    }
    documents = (
        ("technical.html", _render_technical_document(payload)),
        ("index.html", _render_analysis_document(payload)),
    )
    destination = report_dir / "index.html"
    for filename, document in documents:
        written = _write_report_page(report_dir, filename, document)
        if filename == "index.html":
            destination = written
    (report_dir / "analysis.html").unlink(missing_ok=True)
    logger.info(
        f"Generated HTML report for agent workflow {workflow_run_id}: {destination}"
    )
    return destination
