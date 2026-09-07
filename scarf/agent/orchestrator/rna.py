"""Supported assay selection for the automated RNA workflow."""

from collections.abc import Mapping, Sequence
from typing import Any


def validate_rna_directions(directions: Mapping[str, Any]) -> None:
    """Reject automated operations outside the RNA analysis contract."""
    if "hypothesisTesting" in directions:
        raise ValueError(
            "Automated hypothesis testing is not supported; run statistical "
            "analysis separately after the RNA workflow."
        )


def validate_rna_request_fields(request: Any) -> None:
    """Validate routing that does not require inspecting the input store."""
    if len(request.analysisAssays) > 1:
        raise ValueError("analysisAssays must select at most one RNA assay")
    if request.pairedAssays:
        raise ValueError("pairedAssays is unsupported by the single-RNA workflow")
    selected = request.analysisAssays[0] if request.analysisAssays else None
    if selected is not None and request.primaryAssay not in {None, selected}:
        raise ValueError("primaryAssay must match the selected analysisAssays entry")
    selected = selected or request.primaryAssay
    if selected is not None and request.markerAssay not in {None, selected}:
        raise ValueError("markerAssay must match the selected RNA assay")
    validate_rna_directions(request.experimentalDirections)


def selected_rna_assay(request: Any, assay_types: Mapping[str, str]) -> str:
    """Resolve one explicit or uniquely available persisted RNA assay."""
    validate_rna_request_fields(request)
    selected = (
        request.analysisAssays[0] if request.analysisAssays else request.primaryAssay
    )
    if selected is None:
        rna_assays = [
            name for name, assay_type in assay_types.items() if assay_type == "RNA"
        ]
        if len(rna_assays) != 1:
            raise ValueError(
                "The automated workflow requires one RNA assay; "
                f"found {len(rna_assays)}. Select one with primaryAssay or "
                "analysisAssays when multiple RNA assays are present."
            )
        selected = rna_assays[0]
    if selected not in assay_types:
        raise ValueError(f"Unknown requested RNA assay {selected!r}")
    if assay_types[selected] != "RNA":
        raise ValueError(
            f"Selected assay {selected!r} has type {assay_types[selected]!r}; "
            "the automated workflow supports RNA only."
        )
    if request.markerAssay not in {None, selected}:
        raise ValueError("markerAssay must match the selected RNA assay")
    return str(selected)


def selected_store_rna_assay(store: Any, request: Any) -> str:
    """Resolve the workflow assay from the store's persisted summary."""
    return selected_rna_assay(
        request,
        {value.name: value.assay_type for value in store.summary().assays},
    )


def validate_rna_plan(plan: Any, selected: str) -> None:
    """Reject a stale or unsupported cached preprocessing route."""
    if (
        len(plan.assays) != 1
        or plan.assays[0].assay != selected
        or plan.assays[0].assayType != "RNA"
        or not plan.assays[0].graphEligible
        or plan.primaryAssay != selected
        or plan.markerAssay != selected
        or plan.pairedAssays
    ):
        raise ValueError(
            "Preprocessing must contain only the selected RNA assay. "
            "Start a new workflow for incompatible saved analysis state."
        )


def validate_rna_context(report: Any, selected: str) -> None:
    """Keep reused QC evidence on the selected RNA assay."""
    if report.htoIdentityArtifacts:
        raise ValueError(
            "Saved automatic HTO processing is unsupported; start a new RNA workflow."
        )
    for profile in report.qcProfiles:
        if profile.driverAssay != selected or profile.driverAssayType != "RNA":
            raise ValueError(
                "QC evidence must use the selected RNA assay; start a new workflow "
                "for incompatible saved QC state."
            )
    if report.cellQc.driverAssay not in {None, selected}:
        raise ValueError("Cell QC must use the selected RNA assay")


def validate_rna_handoffs(handoffs: Sequence[Any], selected: str) -> None:
    """Keep saved preprocessing on the selected persisted RNA modality."""
    if (
        len(handoffs) != 1
        or handoffs[0].assay != selected
        or handoffs[0].assayType != "RNA"
    ):
        raise ValueError(
            "Saved preprocessing must contain only the selected RNA assay; "
            "start a new workflow for incompatible saved analysis state."
        )


def validate_saved_rna_history(
    root: Any, prefix: str, workflow_run_id: str, selected: str
) -> None:
    """Reject incompatible saved routes before resume opens the store for writes."""
    from ..persistence.reports import load_agent_report
    from . import journal
    from .models import (
        _STAGE_ORDER,
        AutomatedPreprocessingPlan,
        PreprocessedAssayHandoff,
    )

    loaded_reports: set[tuple[str, str]] = set()
    for stage in _STAGE_ORDER:
        for outcome in journal._stage_outcomes(root, prefix, workflow_run_id, stage):
            if outcome.outputs.get("htoIdentityArtifacts"):
                raise ValueError(
                    "Saved automatic HTO processing is unsupported; start a new RNA workflow."
                )
            for name in ("preprocessingPlan", "resolvedPreprocessingPlan"):
                if outcome.outputs.get(name):
                    validate_rna_plan(
                        AutomatedPreprocessingPlan.model_validate(
                            outcome.outputs[name]
                        ),
                        selected,
                    )
            if stage in {"preprocessing", "feature_policy_preprocessing"} and (
                "assays" in outcome.outputs
            ):
                validate_rna_handoffs(
                    [
                        PreprocessedAssayHandoff.model_validate(value)
                        for value in outcome.outputs["assays"]
                    ],
                    selected,
                )
            for reference in outcome.reportReferences:
                if reference.agentName not in {
                    "data_enrichment",
                    "experimental_context",
                    "parameter_tuning",
                }:
                    continue
                identity = (reference.agentName, reference.agentRunId)
                if identity in loaded_reports:
                    continue
                loaded_reports.add(identity)
                report = load_agent_report(root, reference)
                if report.status != "done":
                    continue
                if reference.agentName == "data_enrichment":
                    policies = getattr(report, "policies", [])
                    if (
                        len(policies) != 1
                        or policies[0].assay != selected
                        or policies[0].assayModality != "RNA"
                    ):
                        raise ValueError(
                            "Saved enrichment includes unsupported assays; start a new RNA workflow."
                        )
                elif reference.agentName == "experimental_context":
                    validate_rna_context(report, selected)
                elif reference.agentName == "parameter_tuning":
                    assays = getattr(report, "assayReports", {})
                    if (
                        getattr(report, "recommendedIntegrationId", None) is not None
                        or set(assays) - {selected}
                        or getattr(report, "fromAssay", selected) != selected
                    ):
                        raise ValueError(
                            "Saved tuning includes unsupported assays or integration; "
                            "start a new RNA workflow."
                        )
