"""Small entry point for the supported automated RNA analysis."""

from pathlib import Path
from typing import Any

from .main import AgentOrchestrator
from .models import (
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
)


def analyze_rna(
    source: str | Path,
    *,
    model: Any,
    study_context: str,
    study_objective: str,
    assay: str | None = None,
    zarr_path: str | Path | None = None,
    max_candidates: int = 50,
) -> AutomatedWorkflowResult:
    """Choose and explain settings for one RNA assay, then execute them.

    ``source`` is a supported input file or an existing Zarr store. ``assay``
    selects the RNA assay when the input contains more than one. The workflow
    runs unattended and returns a structured outcome; check ``result.status``
    before consuming it. A completed result provides ``plot_embedding()``,
    ``get_markers()``, and ``report()``.

    ``max_candidates`` limits reserved candidate slots across the workflow.
    Each pass reserves all configured alternatives before screening, including
    conditional candidates that may not execute. Defaults reserve 25 slots for
    the baseline and another 25 if a feature-policy revision runs. A limit of
    50 admits both passes; a smaller limit never shrinks the candidate lists.
    This is admission control, not a count of actual executions or a wall-time
    or provider-token limit. Use ``AgentOrchestrator`` and
    ``AutomatedWorkflowConfig`` for explicit candidate lists, workspaces,
    provider limits, and resumable pauses.
    """
    request = AutomatedWorkflowRequest(
        sourcePath=str(source),
        zarrPath=str(zarr_path) if zarr_path is not None else None,
        studyContext=study_context,
        studyObjective=study_objective,
        primaryAssay=assay,
        markerAssay=assay,
        analysisAssays=[assay] if assay is not None else [],
    )
    config = AutomatedWorkflowConfig(
        inputPolicy="unattended",
        maxCandidateEvaluations=max_candidates,
    )
    return AgentOrchestrator(model, config=config).run(request)
