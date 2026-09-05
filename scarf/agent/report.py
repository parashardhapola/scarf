"""Local HTML reports for completed automated Scarf agent workflows.

Reports are derived presentation files. They are written beside the immutable
agent records, but they are not Zarr components and never participate in
workflow checksums or artifact lineage.
"""

import html
import json
import os
import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .. import __version__
from ..datastore.datastore import DataStore
from ..storage.refs import ArtifactRef
from ..storage.stores import zarr_root_path
from ..storage.types import as_zarr_array
from ..utils.logging import logger
from . import record_io
from .decision_persistence import load_latest_decision_workflow_snapshot
from .orchestrator import journal
from .orchestrator.models import (
    _STAGE_ORDER,
    AutomatedWorkflowResult,
    OrchestrationRequestRecord,
    WorkflowStageAttempt,
    artifact_model_to_ref,
)
from .persistence import (
    AgentWorkflowRun,
    load_agent_report,
    load_agent_workflow,
)
from .types import ArtifactReferenceModel

MAX_MARKER_DOTPLOT_FEATURES = 24
CLUSTER_COUNT_BLOCK_SIZE = 100_000
MAX_EMBEDDING_PLOT_CELLS = 250_000
MAX_DOTPLOT_CELLS = 75_000
MAX_CONNECTIVITY_PLOT_CELLS = 100_000
MAX_COMPOSITION_PLOT_CELLS = 1_000_000
MAX_CHIP_LENGTH = 56
MAX_TABLE_COLUMNS = 7
MAX_INLINE_LEAVES = 12


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


def _save_plot(plot: Any, path: Path) -> None:
    """Atomically save one plot and its provenance, always closing its figure."""
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.stem}.{token}{path.suffix}")
    sidecar = path.with_suffix(path.suffix + ".json")
    temporary_sidecar = sidecar.with_name(f".{sidecar.stem}.{token}{sidecar.suffix}")
    try:
        plot.save(temporary, dpi=150)
        plot.save_provenance(temporary_sidecar, figure_path=path, dpi=150)
        os.replace(temporary, path)
        os.replace(temporary_sidecar, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        plot.close()


def _safe_assay_name(value: str, fallback: str) -> str:
    label = "_".join(part.lower() for part in re.findall(r"[A-Za-z0-9]+", value))
    return label[:64].rstrip("_") or fallback


def _annotate_qc_cutoffs(plot: Any, profile: Mapping[str, Any]) -> None:
    bounds = _qc_resolved_bounds(profile)
    if not bounds:
        return
    diagnostic_only = profile.get("action") == "skip"
    styles = {
        "lowerRemoval": (
            "lower diagnostic bound" if diagnostic_only else "lower removal cutoff",
            "#d62728",
            "--",
        ),
        "upperRemoval": (
            "upper diagnostic bound" if diagnostic_only else "upper removal cutoff",
            "#d62728",
            "--",
        ),
        "upperFlag": ("high-value diagnostic bound", "#ff7f0e", ":"),
    }
    recorded: list[dict[str, Any]] = []
    for metric, axis in plot.axes.items():
        metric_bounds = [
            bound for bound in bounds if str(bound.get("metric") or "") == str(metric)
        ]
        original_limits = axis.get_ylim()
        visible_low, visible_high = sorted(float(value) for value in original_limits)
        has_legend_entry = False
        for field, (label, color, linestyle) in styles.items():
            values = sorted(
                {
                    float(bound[field])
                    for bound in metric_bounds
                    if isinstance(bound.get(field), int | float)
                    and not isinstance(bound.get(field), bool)
                }
            )
            if not values:
                continue
            formatted_values = _analysis_number_range(values)
            if len(values) == 1:
                if visible_low <= values[0] <= visible_high:
                    axis.axhline(
                        values[0],
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.2,
                        label=f"{label}: {formatted_values}",
                    )
                else:
                    axis.plot(
                        [],
                        [],
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.2,
                        label=f"{label}: {formatted_values} (outside plot)",
                    )
            else:
                clipped_low = max(values[0], visible_low)
                clipped_high = min(values[-1], visible_high)
                if clipped_low <= clipped_high:
                    range_suffix = (
                        " (partly outside plot)"
                        if values[0] < visible_low or values[-1] > visible_high
                        else ""
                    )
                    axis.axhspan(
                        clipped_low,
                        clipped_high,
                        color=color,
                        alpha=0.1,
                        label=f"{label}: {formatted_values}{range_suffix}",
                    )
                    for value in values:
                        if visible_low <= value <= visible_high:
                            axis.axhline(
                                value,
                                color=color,
                                linestyle=linestyle,
                                linewidth=0.8,
                            )
                else:
                    axis.plot(
                        [],
                        [],
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.2,
                        label=f"{label}: {formatted_values} (outside plot)",
                    )
            has_legend_entry = True
            recorded.extend(
                {
                    "metric": str(metric),
                    "field": field,
                    "group": bound.get("group"),
                    "value": bound.get(field),
                }
                for bound in metric_bounds
                if isinstance(bound.get(field), int | float)
                and not isinstance(bound.get(field), bool)
            )
        if has_legend_entry:
            axis.legend(frameon=False, fontsize=6.5, loc="upper left")
        axis.set_ylim(original_limits)
    plot.provenance.extras["qc_cutoffs"] = recorded
    plot.provenance.extras["qc_profile"] = profile.get("registeredProfile")


def _collect_final_artifacts(
    store: DataStore,
    result: AutomatedWorkflowResult,
    plot_dir: Path,
    *,
    qc_profile: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, int],
    list[dict[str, Any]],
    dict[str, str],
    list[str],
]:
    """Validate the final handoff and derive bounded tables and plots."""
    import numpy as np

    final = result.finalAnalysis
    assert final is not None
    if final.cellSelection is None or final.clusters is None or final.umap is None:
        raise ValueError("Final handoff lacks its selection, clusters, or UMAP")

    artifact_models = [
        final.cellSelection,
        final.graph,
        final.clusters,
        final.embeddingInitialization,
        final.umap,
        final.markerFeatures,
        final.markers,
    ]
    for native in final.nativeAnalyses:
        artifact_models.extend(
            [
                native.featureSelection,
                native.markerFeatures,
                native.normalized,
                native.reduction,
                native.batchCorrection,
                native.annIndex,
                native.embeddingInitialization,
                native.neighbors,
                native.graph,
                native.clusters,
                native.umap,
            ]
        )
    for artifact in artifact_models:
        if artifact is not None:
            store.load_artifact(artifact_model_to_ref(artifact))

    cluster_ref = artifact_model_to_ref(final.clusters)
    umap_ref = artifact_model_to_ref(final.umap)
    cluster_artifact: Any = store.load_artifact(cluster_ref)
    values = cluster_artifact["values"]
    counts: Counter[str] = Counter()
    for start in range(0, int(values.shape[0]), CLUSTER_COUNT_BLOCK_SIZE):
        block = np.asarray(values[start : start + CLUSTER_COUNT_BLOCK_SIZE]).astype(str)
        block_labels, frequencies = np.unique(block, return_counts=True)
        counts.update(
            {
                str(label): int(frequency)
                for label, frequency in zip(block_labels, frequencies, strict=True)
            }
        )
    cluster_counts = dict(sorted(counts.items()))
    cluster_labels = list(cluster_counts)
    n_cells = sum(cluster_counts.values())
    plots: dict[str, str] = {}
    notes: list[str] = []
    plot_dir.mkdir(parents=True, exist_ok=True)

    def render_plot(name: str, filename: str, create: Any) -> None:
        try:
            path = plot_dir / filename
            _save_plot(create(), path)
            plots[name] = f"plots/{filename}"
        except Exception as exc:
            notes.append(f"{name}: {type(exc).__name__}: {exc}")

    if n_cells <= MAX_EMBEDDING_PLOT_CELLS:
        render_plot(
            "umapClusters",
            "final_umap.png",
            lambda: store.plots.embedding(
                layout=umap_ref,
                color_by=cluster_ref,
                show=False,
            ),
        )
    else:
        notes.append(
            "umapClusters: skipped because the final selection has "
            f"{n_cells:,} cells, above the memory-safe report limit of "
            f"{MAX_EMBEDDING_PLOT_CELLS:,}"
        )

    observed_native_names: set[str] = set()
    for index, native in enumerate(final.nativeAnalyses):
        if native.umap is None or native.clusters is None:
            continue
        native_umap = artifact_model_to_ref(native.umap)
        native_clusters = artifact_model_to_ref(native.clusters)
        if native_umap == umap_ref and native_clusters == cluster_ref:
            continue
        suffix = _safe_assay_name(native.assay, f"assay_{index + 1}")
        base_name = "nativeUmap" + "".join(
            part.capitalize() for part in suffix.split("_")
        )
        plot_name = base_name
        serial = 1
        while plot_name in observed_native_names:
            serial += 1
            plot_name = f"{base_name}{serial}"
        observed_native_names.add(plot_name)
        file_suffix = suffix if serial == 1 else f"{suffix}_{serial}"
        if n_cells <= MAX_EMBEDDING_PLOT_CELLS:
            render_plot(
                plot_name,
                f"native_umap_{file_suffix}.png",
                lambda layout=native_umap, color=native_clusters: store.plots.embedding(
                    layout=layout, color_by=color, show=False
                ),
            )
        else:
            notes.append(
                f"{plot_name}: skipped because {n_cells:,} cells exceed the "
                f"memory-safe report limit of {MAX_EMBEDDING_PLOT_CELLS:,}"
            )

    if n_cells <= MAX_COMPOSITION_PLOT_CELLS:
        render_plot(
            "clusterComposition",
            "cluster_composition.png",
            lambda: store.plots.composition(
                categories=cluster_ref,
                show_percent_labels=len(cluster_labels) <= 12,
                show=False,
            ),
        )
    else:
        notes.append(
            "clusterComposition: skipped because the final selection has "
            f"{n_cells:,} cells, above the memory-safe report limit of "
            f"{MAX_COMPOSITION_PLOT_CELLS:,}"
        )

    qc_attributes = (
        list(result.preprocessingPlan.cellQc.attributes)
        if result.preprocessingPlan is not None
        else []
    )
    available_qc_attributes = [
        value for value in qc_attributes if value in store.cells.columns
    ]
    artifact_qc_metrics = (
        [
            artifact_model_to_ref(value.artifact)
            for value in result.preprocessingPlan.cellQc.artifactMetrics
        ]
        if result.preprocessingPlan is not None
        else []
    )
    available_qc_attributes = available_qc_attributes[:4]

    def qc_distribution(selection: Any) -> Any:
        plot = store.plots.distribution(
            keys=available_qc_attributes,
            cell_selection=artifact_model_to_ref(selection),
            kind="violin",
            max_points=10_000,
            show=False,
        )
        if qc_profile:
            _annotate_qc_cutoffs(plot, qc_profile)
        return plot

    active_cells = qc_profile.get("activeCells") if qc_profile else None
    retained_cells = qc_profile.get("retainedCells") if qc_profile else None
    if (
        available_qc_attributes
        and result.preprocessingPlan is not None
        and result.preprocessingPlan.cellSelection is not None
        and isinstance(active_cells, int)
        and isinstance(retained_cells, int)
        and retained_cells != active_cells
    ):
        render_plot(
            "qcDistributionsBeforeFiltering",
            "qc_distributions_before_filtering.png",
            lambda: qc_distribution(result.preprocessingPlan.cellSelection),
        )
    if available_qc_attributes:
        render_plot(
            "qcDistributions",
            "qc_distributions.png",
            lambda: qc_distribution(final.cellSelection),
        )
    remaining_qc_plots = max(0, 4 - len(available_qc_attributes))
    for index, metric in enumerate(artifact_qc_metrics[:remaining_qc_plots]):
        render_plot(
            f"qcDistributionDerived{index + 1}",
            f"qc_distribution_derived_{index + 1}.png",
            lambda source=metric: store.plots.distribution(
                keys=source,
                kind="violin",
                max_points=10_000,
                show=False,
            ),
        )

    for index, score_model in enumerate(final.doubletScores[:4]):
        score_ref = artifact_model_to_ref(score_model)
        render_plot(
            f"doubletDistribution{index + 1}",
            f"doublet_distribution_{index + 1}.png",
            lambda score=score_ref: store.plots.distribution(
                keys=score,
                kind="hist",
                bins=40,
                show=False,
            ),
        )
        if index == 0 and n_cells <= MAX_EMBEDDING_PLOT_CELLS:
            render_plot(
                "doubletEmbedding",
                "doublet_embedding.png",
                lambda score=score_ref: store.plots.embedding(
                    layout=umap_ref,
                    color_by=score,
                    show=False,
                ),
            )

    top_markers: list[dict[str, Any]] = []
    if final.markers is not None:
        marker_ref = artifact_model_to_ref(final.markers)
        marker_parameters = store.inspect_artifact(marker_ref).parameters or {}
        raw_normalization = marker_parameters.get("normalization", {})
        marker_normalization = (
            dict(raw_normalization) if isinstance(raw_normalization, Mapping) else {}
        )
        marker_log_transform = marker_normalization.get("log_transform", False) is True
        if marker_normalization.get("renormalize_subset", False) is True:
            notes.append(
                "marker visualizations: the persisted marker search renormalized "
                "its feature subset; current plotting APIs preserve its log "
                "transform but visualize assay-wide normalized values"
            )
        for label in cluster_labels:
            try:
                table = store.get_markers(
                    marker_ref,
                    group_id=label,
                    min_score=-1,
                    min_frac_exp=-1,
                )
                if not table.empty:
                    if "score" in table:
                        table = table.sort_values(
                            "score", ascending=False, kind="stable"
                        )
                    top_markers.extend(
                        json.loads(table.head(5).to_json(orient="records"))
                    )
            except Exception as exc:
                notes.append(
                    f"marker export for cluster {label}: {type(exc).__name__}: {exc}"
                )

        render_plot(
            "markerHeatmap",
            "marker_heatmap.png",
            lambda: store.plots.marker_heatmap(
                marker=marker_ref,
                log_transform=marker_log_transform,
                show=False,
            ),
        )

        try:
            from ..plotting import FeatureRef, NormalizationSpec

            by_cluster: dict[str, list[tuple[tuple[str, str], Any]]] = {
                label: [] for label in cluster_labels
            }
            for marker in top_markers:
                group_id = str(marker.get("group_id", ""))
                if group_id not in by_cluster:
                    continue
                feature_name = marker.get("feature_name")
                feature_id = marker.get("feature_id")
                feature_index = marker.get("feature_index")
                label = str(feature_name or feature_id or feature_index or "")
                if isinstance(feature_index, (int, float)):
                    identity = ("index", str(int(feature_index)))
                    feature = FeatureRef(
                        value=int(feature_index),
                        assay=final.markerAssay,
                        by="index",
                        label=label,
                    )
                elif isinstance(feature_id, str) and feature_id:
                    identity = ("id", feature_id)
                    feature = FeatureRef(
                        value=feature_id,
                        assay=final.markerAssay,
                        by="id",
                        label=label,
                    )
                else:
                    continue
                if all(observed != identity for observed, _ in by_cluster[group_id]):
                    by_cluster[group_id].append((identity, feature))

            marker_groups: dict[str, list[Any]] = {}
            selected: set[tuple[str, str]] = set()
            max_rank = max(map(len, by_cluster.values()), default=0)
            rank = 0
            while rank < max_rank and len(selected) < MAX_MARKER_DOTPLOT_FEATURES:
                for cluster in cluster_labels:
                    features = by_cluster[cluster]
                    if rank >= len(features):
                        continue
                    identity, feature = features[rank]
                    if identity in selected:
                        continue
                    marker_groups.setdefault(f"Cluster {cluster}", []).append(feature)
                    selected.add(identity)
                    if len(selected) == MAX_MARKER_DOTPLOT_FEATURES:
                        break
                rank += 1
            if marker_groups and n_cells <= MAX_DOTPLOT_CELLS:
                render_plot(
                    "markerDotplot",
                    "marker_dotplot.png",
                    lambda: store.plots.dotplot(
                        features=marker_groups,
                        groups=cluster_ref,
                        from_assay=final.markerAssay,
                        normalization=NormalizationSpec(
                            source="assay",
                            transform=("log1p" if marker_log_transform else "none"),
                        ),
                        standardize="feature",
                        show=False,
                    ),
                )
            elif marker_groups:
                notes.append(
                    "markerDotplot: skipped because the final selection has "
                    f"{n_cells:,} cells, above the memory-safe report limit of "
                    f"{MAX_DOTPLOT_CELLS:,}"
                )
        except Exception as exc:
            notes.append(f"markerDotplot: {type(exc).__name__}: {exc}")

    if final.graph is not None:
        graph_ref = artifact_model_to_ref(final.graph)
        if n_cells <= MAX_CONNECTIVITY_PLOT_CELLS:
            render_plot(
                "clusterConnectivity",
                "cluster_connectivity.png",
                lambda: store.plots.cluster_connectivity(
                    groups=cluster_ref,
                    layout=umap_ref,
                    graph=graph_ref,
                    show=False,
                ),
            )
        else:
            notes.append(
                "clusterConnectivity: skipped because the final selection has "
                f"{n_cells:,} cells, above the memory-safe report limit of "
                f"{MAX_CONNECTIVITY_PLOT_CELLS:,}"
            )
    return cluster_counts, top_markers, plots, notes


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


def _collect_hvg_plots(
    store: DataStore,
    stage_attempts: Sequence[Mapping[str, Any]],
    preprocessing_plan: Mapping[str, Any],
    plot_dir: Path,
) -> tuple[dict[str, str], list[str]]:
    import numpy as np

    selected_name, selected_reference, artifacts = _latest_hvg_diagnostic_artifacts(
        stage_attempts
    )
    if not selected_reference:
        return {}, []
    assay_name = selected_name.removesuffix("_hvg_diagnostic")
    assay_plan = next(
        (
            value
            for value in _mappings(preprocessing_plan.get("assays"))
            if value.get("assay") == assay_name
        ),
        {},
    )
    selected_count = _mapping(assay_plan.get("featureParameters")).get("topN")
    if not isinstance(selected_count, int) or isinstance(selected_count, bool):
        return {}, ["HVG diagnostics: selected feature count is unavailable"]

    references = (
        ("global", artifacts.get(f"{assay_name}_hvg_global_diagnostic")),
        ("batchAware", artifacts.get(f"{assay_name}_hvg_batchAware_diagnostic")),
    )
    plots: dict[str, str] = {}
    notes: list[str] = []
    seen_artifact_ids: set[str] = set()
    plot_dir.mkdir(parents=True, exist_ok=True)
    for ranking_mode, raw_reference in references:
        if not isinstance(raw_reference, Mapping):
            continue
        model = ArtifactReferenceModel.model_validate(dict(raw_reference))
        if model.artifactId in seen_artifact_ids:
            continue
        seen_artifact_ids.add(model.artifactId)
        plot_name = "hvgGlobal" if ranking_mode == "global" else "hvgBatchAware"
        filename = (
            "hvg_global.png" if ranking_mode == "global" else "hvg_batch_aware.png"
        )
        try:
            diagnostic_ref = artifact_model_to_ref(model)
            diagnostic = store.load_artifact(diagnostic_ref)
            observed_mode = diagnostic.attrs.get("ranking_mode")
            if observed_mode != ranking_mode:
                raise ValueError(
                    f"HVG diagnostic expected {ranking_mode!r}, got {observed_mode!r}"
                )
            status = store.inspect_artifact(diagnostic_ref)
            raw_summary = (status.inputs or {}).get("global_feature_summary")
            if not isinstance(raw_summary, Mapping):
                raise ValueError("HVG diagnostic lacks its global feature summary")
            summary_ref = ArtifactRef.from_dict(dict(raw_summary))
            summary = store.load_artifact(summary_ref)
            corrected_variance = np.asarray(
                as_zarr_array(
                    diagnostic["global_corrected_variance"],
                    name="global_corrected_variance",
                )[:],
                dtype=np.float64,
            )
            ranking = np.asarray(
                as_zarr_array(diagnostic["ranking"], name="ranking")[:],
                dtype=np.int64,
            )
            normed_tot = np.asarray(
                as_zarr_array(summary["normed_tot"], name="normed_tot")[:],
                dtype=np.float64,
            )
            normed_n = np.asarray(
                as_zarr_array(summary["normed_n"], name="normed_n")[:],
                dtype=np.float64,
            )
            shape = corrected_variance.shape
            if (
                corrected_variance.ndim != 1
                or normed_tot.shape != shape
                or normed_n.shape != shape
                or selected_count > ranking.size
                or ranking.size
                and (int(ranking.min()) < 0 or int(ranking.max()) >= shape[0])
                or np.unique(ranking).size != ranking.size
            ):
                raise ValueError("HVG plotting arrays are malformed")
            selected = np.zeros(shape, dtype=bool)
            selected[ranking[:selected_count]] = True
            mean_nonzero = np.divide(
                normed_tot,
                normed_n,
                out=np.zeros_like(normed_tot),
                where=normed_n != 0,
            )
            from ..plotting import highly_variable_features

            plot = highly_variable_features(
                mean_nonzero=mean_nonzero,
                corrected_variance=corrected_variance,
                n_cells=normed_n,
                selected=selected,
                show=False,
            )
            plot.axes["highly_variable_features"].set_title(
                f"{_hvg_ranking_label(ranking_mode)}\n{selected_count:,} selected genes"
            )
            plot.provenance.extras.update(
                {
                    "assay": assay_name,
                    "diagnostic_artifact_id": model.artifactId,
                    "ranking_mode": ranking_mode,
                    "selected_feature_count": selected_count,
                }
            )
            _save_plot(plot, plot_dir / filename)
            plots[plot_name] = f"plots/{filename}"
        except Exception as exc:
            notes.append(f"{plot_name}: {type(exc).__name__}: {exc}")
    return plots, notes


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


REPORT_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400&display=swap');

:root {
  --blue: #0077fc;
  --black: #000000;
  --gray: #b4b4b4;
  --white: #ffffff;
}

* { box-sizing: border-box; }
html { background: var(--white); color: var(--black); font-family: Inter, sans-serif; }
body {
  margin: 0;
  overflow-x: hidden;
  background: var(--white);
  color: var(--black);
  font-family: Inter, sans-serif;
  font-weight: 300;
  letter-spacing: -0.04em;
  line-height: 1.45;
  overflow-wrap: break-word;
  word-break: normal;
}
a { color: var(--blue); }
header, main, footer {
  width: min(100%, 1240px);
  max-width: 100%;
  margin: 0 auto;
  padding-left: clamp(1.25rem, 5vw, 4.5rem);
  padding-right: clamp(1.25rem, 5vw, 4.5rem);
}
.technical-page header, .technical-page main, .technical-page footer {
  width: min(100%, 1800px);
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  border-bottom: 1px solid var(--black);
  padding-top: 1.75rem;
  padding-bottom: 1.75rem;
}
.brand {
  color: var(--black);
  font-size: 1rem;
  font-weight: 400;
  text-decoration: none;
}
main { padding-top: clamp(3rem, 8vw, 7rem); padding-bottom: 6rem; }
footer {
  border-top: 1px solid var(--black);
  padding-top: 2rem;
  padding-bottom: 2rem;
}
h1, h2, h3, p { margin-top: 0; }
h1 {
  max-width: 15ch;
  margin-bottom: 1.5rem;
  font-size: clamp(2.75rem, 7vw, 5rem);
  font-weight: 400;
  letter-spacing: 0;
  line-height: 1.2;
}
h2 {
  margin-bottom: 1.5rem;
  font-size: clamp(1.65rem, 3vw, 2.25rem);
  font-weight: 400;
  letter-spacing: -0.04em;
  line-height: 1.2;
}
h3 {
  margin-bottom: .8rem;
  font-size: 1rem;
  font-weight: 300;
  letter-spacing: -0.04em;
  line-height: 1.2;
}
p, li, td, th, summary, code, pre, a, dd, dt {
  font-family: Inter, sans-serif;
  letter-spacing: -0.04em;
  line-height: 1.45;
}
strong { font-weight: 400; }
.eyebrow {
  margin-bottom: 1rem;
  color: var(--gray);
  font-size: .75rem;
  font-weight: 400;
  text-transform: uppercase;
}
.lead {
  max-width: 48ch;
  font-size: clamp(1.2rem, 2vw, 1.7rem);
  font-weight: 300;
}
.report-nav {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: .35rem;
}
.report-nav a {
  border-radius: .35rem;
  padding: .4rem .65rem;
  color: var(--black);
  font-size: .8rem;
  font-weight: 400;
  text-decoration: none;
}
.report-nav a[aria-current="page"] {
  box-shadow: inset 0 0 0 1px var(--blue);
  color: var(--blue);
}
.pill-row, .chip-row, .metric-grid, .kv-list {
  display: flex;
  flex-wrap: wrap;
  gap: .65rem;
  min-width: 0;
  max-width: 100%;
}
.pill-row { margin-top: 1.75rem; }
.pill, .chip {
  display: inline-flex;
  max-width: 100%;
  font-size: .82rem;
  font-weight: 400;
  line-height: 1.35;
  overflow-wrap: anywhere;
  word-break: normal;
}
.pill {
  align-items: center;
  border: 1px solid var(--blue);
  border-radius: 999px;
  padding: .68rem 1.1rem;
  background: var(--blue);
  color: var(--white);
  text-decoration: none;
  white-space: nowrap;
}
.pill-outline {
  background: var(--white);
  box-shadow: inset 0 0 0 1px var(--blue);
  color: var(--blue);
}
.chip {
  display: inline-block;
  border-radius: .35rem;
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: .4rem .7rem;
  color: var(--black);
  white-space: normal;
  overflow: visible;
}
.text-item {
  display: block;
  min-width: 0;
  max-width: 100%;
  overflow-wrap: anywhere;
  word-break: normal;
}
.metric-grid { margin-top: 2rem; }
.metric {
  display: flex;
  min-width: 0;
  max-width: 100%;
  flex: 1 1 9rem;
  flex-direction: column;
  gap: .2rem;
  border-radius: 1.5rem;
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: .8rem 1.2rem;
}
.metric-label {
  color: var(--gray);
  font-size: .68rem;
  font-weight: 400;
  text-transform: uppercase;
}
.metric-value { font-size: .95rem; font-weight: 400; overflow-wrap: anywhere; }
.report-choice-grid, .summary-grid, .interpretation-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 18rem), 1fr));
  gap: 1rem;
  min-width: 0;
  max-width: 100%;
}
.report-choice {
  display: flex;
  min-width: 0;
  min-height: 14rem;
  flex-direction: column;
  border: 1px solid var(--black);
  padding: 1.5rem;
  color: var(--black);
  text-decoration: none;
}
.report-choice:hover, .report-choice:focus-visible {
  border-color: var(--blue);
}
.report-choice h2 { margin-bottom: .75rem; }
.report-choice p { max-width: 36rem; }
.report-choice-action {
  margin-top: auto;
  padding-top: 1.5rem;
  color: var(--blue);
  font-weight: 400;
}
.summary-card, .interpretation-card {
  min-width: 0;
  border: 1px solid var(--black);
  padding: 1.25rem;
}
.summary-card p:last-child, .interpretation-card p:last-child { margin-bottom: 0; }
.summary-label {
  margin-bottom: .5rem;
  color: var(--gray);
  font-size: .72rem;
  font-weight: 400;
  text-transform: uppercase;
}
.decision-tree {
  min-width: 0;
  max-width: 100%;
  margin-top: 2rem;
}
.tree-stage {
  min-width: 0;
  max-width: 100%;
  margin: 0;
  border: 0;
  padding: 0;
}
.tree-question {
  display: flex;
  width: min(100%, 19rem);
  min-height: 8rem;
  align-items: center;
  justify-content: center;
  margin: 0 auto;
  clip-path: polygon(50% 0, 100% 50%, 50% 100%, 0 50%);
  flex-direction: column;
  padding: 1.75rem 3rem;
  background: var(--blue);
  color: var(--white);
  text-align: center;
}
.tree-question span {
  margin-bottom: .35rem;
  font-size: .68rem;
  font-weight: 400;
  text-transform: uppercase;
}
.tree-question strong {
  font-size: .9rem;
  line-height: 1.25;
}
.tree-stage-description {
  max-width: 42rem;
  margin: 1rem auto 0;
  color: var(--gray);
  text-align: center;
}
.tree-branch-connectors, .tree-selection-connector {
  display: block;
  width: 100%;
  height: 5.25rem;
  color: var(--blue);
}
.tree-branch-connectors path, .tree-selection-connector path {
  fill: none;
  stroke: currentColor;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
.tree-branch-connectors marker path, .tree-selection-connector marker path {
  fill: currentColor;
  stroke: none;
}
.tree-branches {
  display: grid;
  grid-template-columns: repeat(var(--branch-count), minmax(0, 1fr));
  gap: .75rem;
  min-width: 0;
  max-width: 100%;
}
.tree-branch {
  min-width: 0;
  border: 1px solid var(--gray);
  padding: 1rem;
  background: var(--white);
}
.tree-branch-selected {
  border: 2px solid var(--blue);
  box-shadow: inset 0 .25rem 0 var(--blue);
}
.tree-branch-blocked {
  border-style: dashed;
}
.tree-branch-status {
  display: inline-block;
  margin-bottom: .65rem;
  border-radius: .3rem;
  box-shadow: inset 0 0 0 1px var(--gray);
  padding: .25rem .45rem;
  color: var(--gray);
  font-size: .68rem;
  font-weight: 400;
  text-transform: uppercase;
}
.tree-branch-selected .tree-branch-status {
  box-shadow: inset 0 0 0 1px var(--blue);
  color: var(--blue);
}
.tree-branch h3 { margin-bottom: .65rem; font-weight: 400; }
.tree-branch p { margin-bottom: 0; font-size: .82rem; }
.tree-metrics {
  margin: 0 0 .8rem;
  padding-left: 1rem;
  font-size: .76rem;
}
.tree-metrics li { margin: .25rem 0; }
.evidence-accordion {
  display: flex;
  flex-direction: column;
  gap: .8rem;
  margin-top: 1.5rem;
}
.evidence-panel {
  margin: 0;
  border: 1px solid var(--black);
  padding: 0;
}
.evidence-panel > summary {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: center;
  padding: 1.15rem 1.25rem;
  list-style: none;
}
.evidence-panel > summary::-webkit-details-marker { display: none; }
.evidence-panel > summary::after {
  color: var(--blue);
  content: "+";
  font-size: 1.4rem;
  line-height: 1;
}
.evidence-panel[open] > summary::after { content: "−"; }
.evidence-panel-title {
  display: block;
  margin-bottom: .25rem;
  color: var(--gray);
  font-size: .7rem;
  font-weight: 400;
  text-transform: uppercase;
}
.evidence-panel-outcome {
  display: block;
  font-size: .95rem;
  font-weight: 400;
}
.evidence-panel-body {
  border-top: 1px solid var(--black);
  padding: 1.25rem;
}
.evidence-panel-body > p:first-child { max-width: 55rem; }
.evidence-choice-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr));
  gap: .75rem;
  margin-top: 1rem;
}
.evidence-choice {
  min-width: 0;
  border: 1px solid var(--gray);
  padding: 1rem;
}
.evidence-choice-selected {
  border: 2px solid var(--blue);
  box-shadow: inset 0 .2rem 0 var(--blue);
}
.evidence-choice-rejected { border-style: dashed; }
.evidence-choice-status {
  display: inline-block;
  margin-bottom: .55rem;
  color: var(--gray);
  font-size: .68rem;
  font-weight: 400;
  text-transform: uppercase;
}
.evidence-choice-selected .evidence-choice-status { color: var(--blue); }
.evidence-choice h3 { margin-bottom: .55rem; font-weight: 400; }
.evidence-choice p:last-child { margin-bottom: 0; }
.evidence-choice .plain-list {
  margin-bottom: .75rem;
  font-size: .8rem;
}
.evidence-measurements {
  margin-top: 1.25rem;
  border: 0;
  border-top: 1px solid var(--gray);
  padding-top: .8rem;
}
.evidence-measurements > summary {
  color: var(--blue);
  font-size: .82rem;
}
.evidence-measurements-body { padding-top: 1rem; }
.evidence-measurement-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 14rem), 1fr));
  gap: .75rem;
}
.evidence-measurement {
  min-width: 0;
  border-bottom: 1px solid var(--gray);
  padding: .7rem 0;
}
.evidence-measurement dt {
  margin-bottom: .35rem;
  color: var(--gray);
}
.evidence-measurement dd { font-size: .86rem; }
.evidence-measurement small {
  display: block;
  margin-top: .35rem;
  color: var(--gray);
  font-size: .72rem;
}
.plain-list { margin: 0; padding-left: 1.2rem; }
.plain-list li { margin: .55rem 0; }
.column-list {
  columns: 4 12rem;
  column-gap: 2rem;
}
.column-list li {
  break-inside: avoid;
  margin: .25rem 0;
}
.section {
  margin-top: 4rem;
  min-width: 0;
  max-width: 100%;
  border-top: 1px solid var(--black);
  padding-top: 1.5rem;
}
.section:target { scroll-margin-top: 1rem; }
.section-heading {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: start;
}
.subsection { margin-top: 2rem; min-width: 0; max-width: 100%; }
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 19rem), 1fr));
  gap: 1rem;
  min-width: 0;
  max-width: 100%;
}
.card, .callout, .record {
  min-width: 0;
  max-width: 100%;
  border: 1px solid var(--black);
  padding: 1.25rem;
  background: var(--white);
  overflow: visible;
}
.callout { border-color: var(--blue); }
.record-stack {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 0;
  max-width: 100%;
}
.product-callout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: center;
  margin-top: 2.5rem;
  border-radius: 1.5rem;
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: 1.4rem;
}
.product-callout p { margin-bottom: 0; max-width: 55rem; }
.empty { color: var(--gray); font-style: italic; }
.table-wrap {
  width: 100%;
  max-width: 100%;
  overflow: visible;
}
table {
  width: 100%;
  table-layout: auto;
  border-collapse: collapse;
  font-size: .86rem;
}
th, td {
  min-width: 0;
  width: auto;
  border-bottom: 1px solid var(--black);
  padding: .8rem .7rem;
  text-align: left;
  vertical-align: top;
  overflow-wrap: break-word;
  word-break: normal;
  hyphens: auto;
}
th {
  position: sticky;
  top: 0;
  background: var(--white);
  color: var(--gray);
  font-weight: 400;
  overflow-wrap: normal;
  text-transform: uppercase;
}
td { font-weight: 300; overflow-wrap: anywhere; }
td > * { max-width: 100%; }
tr.selected { box-shadow: inset 4px 0 0 var(--blue); }
.table-records { gap: 1.25rem; }
.table-record {
  border-color: var(--gray);
  padding: 1rem;
}
.table-record-selected {
  border: 2px solid var(--blue);
  box-shadow: inset .25rem 0 0 var(--blue);
}
.record-fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 13rem), 1fr));
  gap: .9rem 1.25rem;
}
.record-field {
  min-width: 0;
  border-bottom: 1px solid var(--gray);
  padding-bottom: .65rem;
}
.record-field-wide { grid-column: 1 / -1; }
.record-field dt { margin-bottom: .3rem; }
.record-field dd {
  overflow-wrap: anywhere;
  word-break: normal;
}
dl { margin: 0; min-width: 0; max-width: 100%; }
.details > div {
  display: grid;
  grid-template-columns: minmax(0, 12rem) minmax(0, 1fr);
  gap: 1rem;
  min-width: 0;
  border-bottom: 1px solid var(--gray);
  padding: .55rem 0;
}
.details .details > div {
  grid-template-columns: minmax(0, 1fr);
  gap: .2rem;
}
dt { color: var(--gray); font-size: .78rem; font-weight: 400; text-transform: uppercase; }
dd { min-width: 0; margin: 0; overflow-wrap: anywhere; }
.kv { display: inline-flex; flex-wrap: wrap; gap: .25rem .4rem; min-width: 0; max-width: 100%; }
.kv-k {
  color: var(--gray);
  font-size: .72rem;
  font-weight: 400;
  text-transform: uppercase;
}
.kv-v { overflow-wrap: anywhere; word-break: normal; }
.nested-records {
  display: block;
  margin: .15rem 0;
  border: 0;
  padding: 0;
  min-width: 0;
  max-width: 100%;
  overflow: visible;
}
.nested-records > summary { color: var(--blue); font-size: .82rem; }
.nested-records .table-wrap, .nested-records .record-stack { margin-top: .55rem; }
.plot-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 26rem), 1fr));
  gap: 2rem;
  min-width: 0;
}
figure { margin: 0; min-width: 0; }
figure.primary { grid-column: 1 / -1; }
figure img { display: block; width: 100%; height: auto; border: 1px solid var(--black); }
figcaption { margin-top: .7rem; color: var(--black); font-size: .85rem; }
.cluster-row {
  display: grid;
  grid-template-columns: minmax(0, 8rem) minmax(0, 1fr) auto;
  gap: .7rem;
  align-items: center;
  margin: .5rem 0;
  min-width: 0;
}
.cluster-track { height: .7rem; border-radius: 999px; background: var(--gray); overflow: hidden; }
.cluster-fill { height: 100%; border-radius: 999px; background: var(--blue); }
.text-list { padding-left: 1.2rem; }
.text-list li { margin: .45rem 0; }
details { margin-top: 1rem; border-top: 1px solid var(--gray); padding-top: .8rem; }
summary { cursor: pointer; font-weight: 400; }
pre {
  max-height: 36rem;
  max-width: 100%;
  overflow: auto;
  background: var(--white);
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: 1rem;
  font-size: .76rem;
  white-space: pre-wrap;
  word-break: break-word;
}
@media (max-width: 900px) {
  .tree-branches { grid-template-columns: 1fr; }
  .tree-branch-connectors, .tree-selection-connector { display: none; }
  .tree-question { margin-bottom: 2.5rem; }
  .tree-stage:not(:last-child)::after {
    display: block;
    margin: .25rem 0 2rem;
    color: var(--blue);
    content: "↓";
    font-size: 1.5rem;
    text-align: center;
  }
  .tree-branch {
    position: relative;
    margin-bottom: 1.5rem;
  }
  .tree-branch::before {
    position: absolute;
    top: -1.65rem;
    left: 50%;
    color: var(--blue);
    content: "↓";
  }
}
@media (max-width: 680px) {
  header { align-items: flex-start; flex-direction: column; }
  .report-nav { justify-content: flex-start; }
  .section-heading, .product-callout { grid-template-columns: 1fr; }
  .details > div { grid-template-columns: 1fr; gap: .25rem; }
  .cluster-row { grid-template-columns: minmax(0, 1fr) auto; }
}
"""


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _label(value: Any) -> str:
    text = str(value).replace("_", " ").strip()
    words: list[str] = []
    for index, character in enumerate(text):
        if (
            index
            and character.isupper()
            and not text[index - 1].isupper()
            and text[index - 1] != " "
        ):
            words.append(" ")
        words.append(character)
    text = "".join(words)
    return text[:1].upper() + text[1:]


def _scalar(value: Any) -> str:
    if value is None or value == "":
        return "Not provided"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 10_000:
            return f"{value:.3g}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _is_leaf(value: Any) -> bool:
    return not isinstance(value, Mapping) and not _is_sequence(value)


def _is_simple(value: Any) -> bool:
    if _is_leaf(value):
        return True
    return _is_sequence(value) and all(_is_leaf(item) for item in value)


def _is_mapping_sequence(value: Any) -> bool:
    return (
        _is_sequence(value)
        and bool(value)
        and all(isinstance(item, Mapping) for item in value)
    )


def _chip(text: str) -> str:
    escaped = html.escape(text)
    if len(text) > MAX_CHIP_LENGTH:
        return f'<span class="text-item">{escaped}</span>'
    return f'<span class="chip">{escaped}</span>'


def _chips(value: Any, empty: str = "Not provided") -> str:
    if not _present(value):
        return f'<span class="empty">{html.escape(empty)}</span>'
    if isinstance(value, Mapping):
        items = [f"{_label(key)}: {_scalar(item)}" for key, item in value.items()]
    elif _is_sequence(value):
        items = list(value)
    else:
        items = [value]
    return '<span class="chip-row">{}</span>'.format(
        "".join(_chip(_scalar(item)) for item in items)
    )


def _kv_list(mapping: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, item in mapping.items():
        if not _present(item):
            continue
        label = html.escape(_label(key))
        if _is_leaf(item):
            text = f"{_label(key)}: {_scalar(item)}"
            if len(text) <= MAX_CHIP_LENGTH:
                parts.append(_chip(text))
            else:
                parts.append(
                    f'<span class="kv"><span class="kv-k">{label}</span>'
                    f'<span class="kv-v">{html.escape(_scalar(item))}</span></span>'
                )
        elif _is_simple(item):
            parts.append(
                f'<span class="kv"><span class="kv-k">{label}</span>{_chips(item)}</span>'
            )
        else:
            parts.append(
                '<details class="nested-records">'
                f"<summary>{label}</summary>{_value(item)}</details>"
            )
    if not parts:
        return '<span class="empty">Not provided</span>'
    return f'<div class="kv-list">{"".join(parts)}</div>'


def _cell(value: Any) -> str:
    if not _present(value):
        return '<span class="empty">Not provided</span>'
    if isinstance(value, Mapping):
        return _kv_list(value)
    if _is_mapping_sequence(value):
        count = len(value)
        return (
            '<details class="nested-records">'
            f"<summary>{count:,} records</summary>"
            f"{_mapping_list(value)}</details>"
        )
    if _is_sequence(value) and all(_is_leaf(item) for item in value):
        if len(value) > MAX_INLINE_LEAVES:
            return html.escape(", ".join(_scalar(item) for item in value))
        return _chips(value)
    if _is_sequence(value):
        return _chips(value)
    return html.escape(_scalar(value))


def _visible_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            key for row in rows for key in row if not str(key).startswith("_")
        )
    )


def _render_record_rows(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> str:
    records: list[str] = []
    for row in rows:
        fields: list[str] = []
        for key in columns:
            value = row.get(key)
            wide = not _is_simple(value) or (
                isinstance(value, str) and len(value) > MAX_CHIP_LENGTH
            )
            field_class = "record-field record-field-wide" if wide else "record-field"
            fields.append(
                f'<div class="{field_class}">'
                f"<dt>{html.escape(_label(key))}</dt>"
                f"<dd>{_cell(value)}</dd></div>"
            )
        selected_class = " table-record-selected" if row.get("_selected") else ""
        records.append(
            f'<article class="record table-record{selected_class}">'
            f'<dl class="record-fields">{"".join(fields)}</dl></article>'
        )
    return f'<div class="record-stack table-records">{"".join(records)}</div>'


def _mapping_list(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [dict(row) for row in rows]
    if not normalized:
        return '<p class="empty">No records available.</p>'
    visible = _visible_columns(normalized)
    if len(visible) <= MAX_TABLE_COLUMNS:
        return _table(normalized)
    return _render_record_rows(normalized, visible)


def _value(value: Any) -> str:
    if not _present(value):
        return '<span class="empty">Not provided</span>'
    if isinstance(value, Mapping):
        rows = "".join(
            "<div><dt>{}</dt><dd>{}</dd></div>".format(
                html.escape(_label(key)),
                (
                    _mapping_list(_mappings(item))
                    if _is_mapping_sequence(item)
                    else _value(item)
                ),
            )
            for key, item in value.items()
            if _present(item)
        )
        return f'<dl class="details">{rows}</dl>'
    if _is_mapping_sequence(value):
        return _mapping_list(value)
    if _is_sequence(value):
        if all(_is_leaf(item) for item in value):
            return _chips(value)
        return '<div class="card-grid">{}</div>'.format(
            "".join(f'<div class="card">{_value(item)}</div>' for item in value)
        )
    return html.escape(_scalar(value))


def _table(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    empty: str = "No records available.",
) -> str:
    normalized = [dict(row) for row in rows]
    if not normalized:
        return f'<p class="empty">{html.escape(empty)}</p>'
    visible = list(columns or ()) or _visible_columns(normalized)
    if len(visible) > MAX_TABLE_COLUMNS:
        return _render_record_rows(normalized, visible)
    headings = "".join(f"<th>{html.escape(_label(key))}</th>" for key in visible)
    body = "".join(
        ('<tr class="selected">' if row.get("_selected") else "<tr>")
        + "".join(f"<td>{_cell(row.get(key))}</td>" for key in visible)
        + "</tr>"
        for row in normalized
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{headings}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _latest(reports: Mapping[str, Any], agent_name: str) -> dict[str, Any]:
    values = reports.get(agent_name)
    if isinstance(values, Mapping):
        return dict(values)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        for value in reversed(values):
            if isinstance(value, Mapping):
                return dict(value)
    return {}


def _render_plots(
    plots: Mapping[str, str],
    notes: Sequence[str],
    *,
    order: Sequence[str] | None = None,
    titles: Mapping[str, tuple[str, str]] | None = None,
    show_provenance: bool = True,
    show_notes: bool = True,
    empty_message: str | None = None,
) -> str:
    plot_titles = {
        "umapClusters": (
            "Final UMAP by cluster",
            "The selected final representation, colored by final cluster.",
        ),
        "markerHeatmap": (
            "Marker heatmap",
            "Marker-feature patterns across the final clusters.",
        ),
        "markerDotplot": (
            "Marker dot plot",
            "A bounded expression summary for exact exported marker features.",
        ),
        "clusterComposition": (
            "Cluster composition",
            "The relative size of each cluster in the final cell selection.",
        ),
        "clusterConnectivity": (
            "Cluster connectivity",
            "Connectivity between final clusters in the selected graph.",
        ),
        "qcDistributions": (
            "QC distributions after the selected policy",
            "Retained-cell distributions with the selected profile's cutoff annotations.",
        ),
        "qcDistributionsBeforeFiltering": (
            "QC distributions before filtering",
            "Input-cell distributions with the selected profile's cutoff annotations.",
        ),
        "hvgGlobal": (
            "Global HVG diagnostic",
            "Mean-variance evidence with genes selected by the global ranking highlighted.",
        ),
        "hvgBatchAware": (
            "Group-aware HVG diagnostic",
            "Mean-variance evidence with genes selected for recurrence across technical groups highlighted.",
        ),
        "doubletEmbedding": (
            "Advisory doublet scores",
            "The final embedding colored by non-removing doublet evidence.",
        ),
    }
    if titles is not None:
        plot_titles.update(titles)
    plot_order = (
        list(order)
        if order is not None
        else [
            "umapClusters",
            *(name for name in plots if name.startswith("nativeUmap")),
            "markerHeatmap",
            "markerDotplot",
            "clusterComposition",
            "clusterConnectivity",
            "qcDistributionsBeforeFiltering",
            "qcDistributions",
            *(name for name in plots if name.startswith("qcDistributionDerived")),
            "hvgGlobal",
            "hvgBatchAware",
            "doubletEmbedding",
            *(name for name in plots if name.startswith("doubletDistribution")),
            *plots,
        ]
    )
    figures: list[str] = []
    for name in dict.fromkeys(plot_order):
        source = plots.get(name)
        if source is None:
            continue
        if name.startswith("nativeUmap"):
            assay = name.removeprefix("nativeUmap") or "assay"
            title = f"{assay} native UMAP"
            caption = f"The finalized native {assay} representation and clusters."
        elif name.startswith("doubletDistribution"):
            title = "Advisory doublet-score distribution"
            caption = (
                "Capture-aware doublet evidence retained as flags without removal."
            )
        elif name.startswith("qcDistributionDerived"):
            title = "Derived QC metric distribution"
            caption = "An immutable feature-family QC metric on the selected cell axis."
        else:
            title, caption = plot_titles.get(
                name, (_label(name), "A finalized Scarf analysis plot.")
            )
        escaped_source = html.escape(source, quote=True)
        plot_class = ' class="primary"' if name == "umapClusters" else ""
        provenance_markup = ""
        if show_provenance:
            provenance = html.escape(source + ".json", quote=True)
            provenance_markup = f' <a href="{provenance}">Plot provenance</a>'
        figures.append(
            f"<figure{plot_class}>"
            f'<a href="{escaped_source}"><img src="{escaped_source}" '
            f'alt="{html.escape(title, quote=True)}" loading="lazy"></a>'
            f"<figcaption><strong>{html.escape(title)}</strong><br>"
            f"{html.escape(caption)}{provenance_markup}</figcaption></figure>"
        )
    if not figures:
        if empty_message is None:
            plot_markup = (
                '<div class="callout"><p>No plots could be rendered. The structured '
                "analysis remains available below. Install Scarf with the "
                "<code>extra</code> dependency group to enable plotting.</p></div>"
            )
        else:
            plot_markup = (
                f'<div class="callout"><p>{html.escape(empty_message)}</p></div>'
            )
    else:
        plot_markup = f'<div class="plot-grid">{"".join(figures)}</div>'
    note_markup = ""
    if show_notes and notes:
        note_markup = (
            "<details><summary>Plot availability notes</summary>"
            '<ul class="text-list">'
            + "".join(f"<li>{html.escape(note)}</li>" for note in notes)
            + "</ul></details>"
        )
    return plot_markup + note_markup


def _render_clusters(cluster_counts: Mapping[str, int]) -> str:
    if not cluster_counts:
        return '<p class="empty">No final cluster counts were available.</p>'
    maximum = max(cluster_counts.values(), default=1) or 1
    return "".join(
        '<div class="cluster-row">'
        f"<span>Cluster {html.escape(str(label))}</span>"
        '<span class="cluster-track">'
        f'<span class="cluster-fill" style="width: {count / maximum * 100:.2f}%">'
        "</span></span>"
        f"<span>{count:,}</span></div>"
        for label, count in cluster_counts.items()
    )


def _parameter_rows(parameter: Mapping[str, Any]) -> list[dict[str, Any]]:
    assay_reports = _mapping(parameter.get("assayReports"))
    if not assay_reports and _present(parameter.get("evaluations")):
        assay_reports = {str(parameter.get("fromAssay") or "Primary"): dict(parameter)}
    recommended = _mapping(parameter.get("recommendedByAssay"))
    rows: list[dict[str, Any]] = []
    for assay, raw_report in assay_reports.items():
        report = _mapping(raw_report)
        selected = recommended.get(assay) or report.get("recommendedCandidateId")
        for evaluation in _mappings(report.get("evaluations")):
            parameters = _mapping(evaluation.get("parameters"))
            rows.append(
                {
                    "_selected": evaluation.get("candidateId") == selected,
                    "assay": assay,
                    "candidate": evaluation.get("candidateId"),
                    "phase": evaluation.get("phase"),
                    "status": evaluation.get("status"),
                    "eligible": evaluation.get("eligible"),
                    "selection confidence": report.get("confidence"),
                    "reduction": parameters.get("reductionMethod"),
                    "dimensions": parameters.get("dimensions"),
                    "neighbors K": parameters.get("neighborsK"),
                    "resolution": parameters.get("leidenResolution"),
                    "Harmony": parameters.get("useHarmony"),
                    "metrics": evaluation.get("metrics"),
                }
            )
    return rows


def _render_parameter_tuning(parameter: Mapping[str, Any]) -> str:
    if not parameter:
        return '<p class="empty">No Parameter Tuning report was persisted.</p>'
    candidate_rows = _parameter_rows(parameter)
    integration_rows = _mappings(parameter.get("integrationEvaluations"))
    plans: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    root_plan = _mapping(parameter.get("searchPlan"))
    if root_plan:
        plans.append({"assay": parameter.get("fromAssay"), **root_plan})
    for comparison in _mappings(parameter.get("comparisons")):
        comparisons.append(
            {"scope": parameter.get("fromAssay") or "primary assay", **comparison}
        )
    for assay, report in _mapping(parameter.get("assayReports")).items():
        assay_report = _mapping(report)
        plan = _mapping(assay_report.get("searchPlan"))
        if plan and plan not in plans:
            plans.append({"assay": assay, **plan})
        for comparison in _mappings(assay_report.get("comparisons")):
            comparisons.append({"scope": assay, **comparison})
    final_selection = _mapping(parameter.get("finalSelection"))
    for comparison in _mappings(final_selection.get("comparisons")):
        comparisons.append({"scope": "final graph", **comparison})
    narrative = {
        "status": parameter.get("status"),
        "totalCandidates": parameter.get("totalCandidates"),
        "recommendedByAssay": parameter.get("recommendedByAssay"),
        "recommendedIntegrationId": parameter.get("recommendedIntegrationId"),
        "confidence": parameter.get("confidence"),
        "rationale": parameter.get("rationale"),
        "tradeoffs": parameter.get("tradeoffs"),
        "stopReason": parameter.get("stopReason"),
        "finalSelection": final_selection,
    }
    return (
        '<div class="callout"><h3>Final graph selection</h3>'
        f"{_value(narrative)}</div>"
        '<div class="subsection"><h3>Native and Harmony candidates</h3>'
        f"{_table(candidate_rows, empty='No native candidates were recorded.')}</div>"
        '<div class="subsection"><h3>SNN and WNN integration candidates</h3>'
        f"{_table(integration_rows, empty='No integration candidates were eligible.')}</div>"
        '<div class="subsection"><h3>Model-authored comparisons</h3>'
        f"{_table(comparisons, empty='No candidate comparisons were required.')}</div>"
        '<div class="subsection"><h3>Bounded search plans</h3>'
        f"{_value(plans) if plans else '<p class="empty">No refinement plan was requested.</p>'}"
        "</div>"
    )


def _execution_rows(reports: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def visit(value: Any, stage: str, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            usage = value.get("usage")
            agent_name = value.get("agentName")
            if (
                isinstance(usage, Mapping)
                and isinstance(agent_name, str)
                and agent_name.strip()
            ):
                run_id = str(value.get("runId") or "")
                identity = (agent_name, run_id, str(value.get("modelName") or ""))
                if identity not in seen:
                    seen.add(identity)
                    rows.append(
                        {
                            "agent stage": _label(stage),
                            "execution": _label(path[-1]) if path else agent_name,
                            "agent": agent_name,
                            "run ID": run_id or "deterministic",
                            "model": value.get("modelName") or "not applicable",
                            "duration seconds": value.get("durationSeconds"),
                            "requests": usage.get("requests", 0),
                            "tool calls": usage.get("toolCalls", 0),
                            "input tokens": usage.get("inputTokens", 0),
                            "output tokens": usage.get("outputTokens", 0),
                            "total tokens": usage.get("totalTokens", 0),
                        }
                    )
            for key, item in value.items():
                visit(item, stage, (*path, str(key)))
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                visit(item, stage, (*path, str(index + 1)))

    for stage, records in reports.items():
        visit(records, str(stage), ())
    return rows


def _render_executions(reports: Mapping[str, Any]) -> str:
    rows = _execution_rows(reports)
    if not rows:
        return '<p class="empty">No provider execution metadata was recorded.</p>'
    totals = {
        "recorded executions": len(rows),
        "provider executions": sum(
            int(
                bool(row["model"] != "not applicable")
                or int(row["requests"] or 0) > 0
                or int(row["input tokens"] or 0) > 0
                or int(row["output tokens"] or 0) > 0
            )
            for row in rows
        ),
        "requests": sum(int(row["requests"] or 0) for row in rows),
        "tool calls": sum(int(row["tool calls"] or 0) for row in rows),
        "input tokens": sum(int(row["input tokens"] or 0) for row in rows),
        "output tokens": sum(int(row["output tokens"] or 0) for row in rows),
        "total tokens": sum(int(row["total tokens"] or 0) for row in rows),
    }
    return (
        '<div class="callout"><h3>Recorded totals</h3>'
        f'{_chips(totals)}</div><div class="subsection">{_table(rows)}</div>'
    )


def _render_timeline(
    attempts: Sequence[Mapping[str, Any]],
    resumes: Sequence[Mapping[str, Any]],
) -> str:
    artifacts: list[dict[str, Any]] = []
    for attempt in attempts:
        for name, reference in _mapping(attempt.get("artifacts")).items():
            artifact = _mapping(reference)
            artifacts.append(
                {
                    "stage": attempt.get("stage"),
                    "attempt": attempt.get("attemptId"),
                    "name": name,
                    "scope": artifact.get("scope"),
                    "assay": artifact.get("assay"),
                    "kind": artifact.get("kind"),
                    "artifact ID": artifact.get("artifactId"),
                }
            )
    return (
        "<h3>Stage attempts</h3>"
        + _table(
            attempts,
            columns=(
                "stage",
                "status",
                "durationSeconds",
                "actions",
                "reportCount",
                "artifactCount",
                "parentAttempts",
                "questionIds",
                "noteCount",
                "errorType",
            ),
        )
        + '<div class="subsection"><h3>Stage artifact inventory</h3>'
        + _table(artifacts, empty="No stage artifacts were recorded.")
        + "</div>"
        + '<div class="subsection"><h3>Resume lineage</h3>'
        + _table(
            resumes,
            columns=(
                "resumeId",
                "answeredStage",
                "answeredAttemptId",
                "questionIds",
            ),
            empty="No resume was required.",
        )
        + "</div>"
    )


def _text_values(value: Any) -> list[str]:
    if not _is_sequence(value):
        return []
    return [str(item).strip() for item in value if _is_leaf(item) and str(item).strip()]


def _specific_references(values: Sequence[str]) -> list[str]:
    unique = list(dict.fromkeys(values))
    return [
        value
        for value in unique
        if not any(
            value.casefold() != other.casefold()
            and value.casefold() in other.casefold()
            for other in unique
        )
    ]


def _brief_text(value: Any, *, max_length: int = 240) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    text = re.sub(
        r"\b[0-9a-f]{64}\b",
        "recorded result",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "recorded value",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b([A-Za-z][A-Za-z0-9]*)_id\b",
        lambda match: _label(match.group(1)).lower(),
        text,
    )
    if not text:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    if len(first_sentence) <= max_length:
        return first_sentence
    shortened = first_sentence[: max_length - 3].rsplit(" ", 1)[0]
    return f"{shortened or first_sentence[: max_length - 3]}..."


def _format_text_list(values: Sequence[str]) -> str:
    items = [value for value in dict.fromkeys(values) if value]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _study_overview(
    payload: Mapping[str, Any],
) -> tuple[str, list[str], list[str]]:
    reports = _mapping(payload.get("reports"))
    request = _mapping(payload.get("request"))
    enrichment = _latest(reports, "data_enrichment")
    study = _mapping(enrichment.get("studyContextSummary"))
    objective = ""
    for candidate in (
        study.get("studyObjective"),
        request.get("studyObjective"),
        study.get("studyContext"),
        request.get("studyContext"),
    ):
        objective = _brief_text(candidate)
        if objective:
            break
    return (
        objective or "The automated analysis completed successfully.",
        _specific_references(_text_values(study.get("organismReferences"))),
        _specific_references(_text_values(study.get("tissueReferences"))),
    )


def _biological_source(
    organisms: Sequence[str],
    tissues: Sequence[str],
) -> str:
    organism = _format_text_list(organisms)
    tissue = _format_text_list(tissues)
    if organism and tissue:
        return f"{organism} material from {tissue}"
    if organism:
        return f"{organism} biological material"
    if tissue:
        return f"biological material from {tissue}"
    return ""


def _assay_label(value: Any) -> str:
    labels = {
        "RNA": "RNA",
        "ATAC": "chromatin accessibility",
        "ADT": "protein abundance",
        "HTO": "sample tags",
    }
    text = str(value or "").strip()
    return labels.get(text.upper(), _label(text).lower()) if text else ""


def _report_assays(plan: Mapping[str, Any]) -> list[str]:
    assays: list[str] = []
    for assay in _mappings(plan.get("assays")):
        label = _assay_label(assay.get("assayType") or assay.get("assay"))
        if label and label not in assays:
            assays.append(label)
    return assays


def _render_metrics(metrics: Sequence[tuple[str, Any]]) -> str:
    markup = "".join(
        '<span class="metric">'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f'<span class="metric-value">{html.escape(_scalar(value))}</span></span>'
        for label, value in metrics
        if _present(value)
    )
    return f'<div class="metric-grid">{markup}</div>' if markup else ""


def _render_report_navigation(active_page: str) -> str:
    links = (
        ("index", "index.html", "Report home"),
        ("analysis", "analysis.html", "Analysis summary"),
        ("technical", "technical.html", "Technical details"),
    )
    return '<nav class="report-nav" aria-label="Report pages">{}</nav>'.format(
        "".join(
            '<a href="{}"{}>{}</a>'.format(
                html.escape(path, quote=True),
                ' aria-current="page"' if page == active_page else "",
                html.escape(label),
            )
            for page, path, label in links
        )
    )


def _render_report_shell(
    *,
    title: str,
    active_page: str,
    body: str,
) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{REPORT_STYLES}</style>
</head>
<body class="{html.escape(active_page, quote=True)}-page">
<header>
  <a class="brand" href="https://www.nygen.io/" target="_blank" rel="noopener noreferrer">Nygen Analytics</a>
  {_render_report_navigation(active_page)}
</header>
<main>
{body}
</main>
<footer>
  <p>Generated locally by Scarf agents. <a href="https://www.nygen.io/">Nygen Analytics</a></p>
</footer>
</body>
</html>
"""


def _selected_qc_profile(
    experimental: Mapping[str, Any],
    cell_qc: Mapping[str, Any],
) -> dict[str, Any]:
    profiles = _mappings(experimental.get("qcProfiles"))
    profile_id = cell_qc.get("profileId")
    if profile_id:
        for profile in profiles:
            if profile.get("profileId") == profile_id:
                return profile
    return profiles[0] if len(profiles) == 1 else {}


def _feature_family_label(value: Any) -> str:
    labels = {
        "ribosomal": "ribosomal genes",
        "ribosomalProtein": "ribosomal protein genes",
        "mitochondrial": "mitochondrial genes",
        "mitoribosomal": "mitoribosomal genes",
        "sex": "sex-linked genes",
        "sexLinked": "sex-linked genes",
        "cellCycle": "cell-cycle genes",
        "cellCycleCcn": "CCN-prefixed genes",
        "hla": "HLA genes",
        "h2": "H2 genes",
        "histone": "histone genes",
    }
    text = str(value or "").strip()
    return labels.get(text, _label(text).lower()) if text else ""


def _public_field_label(value: Any) -> str:
    labels = {
        "T2D": "T2D status",
        "donor_id": "donor",
        "library_id": "library",
        "RNA_nCounts": "RNA counts",
        "RNA_nFeatures": "detected genes",
        "RNA_percentMito": "mitochondrial percentage",
        "RNA_percentRibo": "ribosomal percentage",
        "sample_id": "sample",
        "sex": "sex",
        "tissue": "tissue",
    }
    text = str(value or "").strip()
    if not text:
        return ""
    if text in labels:
        return labels[text]
    return _label(text.removesuffix("_id")).lower()


def _tree_branch(
    *,
    label: str,
    status: str,
    state: str,
    metrics: Sequence[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "status": status,
        "state": state,
        "metrics": list(metrics),
        "reason": reason,
    }


def _qc_profile_label(profile: Mapping[str, Any]) -> str:
    labels = {
        "retainWithFlags": "Retain cells with quality flags",
        "globalMad5": "Global quality threshold",
        "captureMad5": "Per-library quality threshold",
        "captureMad3Sensitivity": "Stricter per-library sensitivity check",
    }
    registered = str(profile.get("registeredProfile") or "")
    if registered in labels:
        return labels[registered]
    profile_id = str(profile.get("profileId") or "")
    for name, label in labels.items():
        if name in profile_id:
            return label
    action = str(profile.get("action") or "")
    return {
        "skip": "Retain reviewed cells",
        "globalGaussian": "Global quality threshold",
        "sampleMad": "Per-sample quality threshold",
        "registeredMad": "Registered quality threshold",
    }.get(action, "Quality-control option")


def _qc_tree_stage(
    experimental: Mapping[str, Any],
    plan: Mapping[str, Any],
    total_cells: int,
) -> dict[str, Any] | None:
    decision = _mapping(experimental.get("decision"))
    cell_qc = _mapping(plan.get("cellQc"))
    if not cell_qc:
        cell_qc = _mapping(decision.get("cellQc"))
    if not cell_qc:
        cell_qc = _mapping(experimental.get("cellQc"))
    if not cell_qc:
        return None
    profiles = _mappings(experimental.get("qcProfiles"))
    if not profiles:
        profiles = [
            {
                **cell_qc,
                "activeCells": total_cells or None,
                "retainedCells": total_cells or None,
            }
        ]
    selected_id = cell_qc.get("profileId")
    selected_name = cell_qc.get("registeredProfile")
    branches: list[dict[str, Any]] = []
    for profile in profiles:
        selected = bool(
            (selected_id and profile.get("profileId") == selected_id)
            or (
                not selected_id
                and selected_name
                and profile.get("registeredProfile") == selected_name
            )
            or (len(profiles) == 1)
        )
        active = profile.get("activeCells")
        retained = profile.get("retainedCells")
        metrics: list[str] = []
        removed: int | None = None
        if isinstance(active, int) and isinstance(retained, int) and active:
            retained_percent = retained / active * 100
            percent_text = "100%" if retained == active else f"{retained_percent:.2f}%"
            metrics.append(
                f"Retained {retained:,} of {active:,} cells ({percent_text})"
            )
            removed = active - retained
        if selected:
            reason = (
                "Selected because it preserved the reviewed dataset without "
                "unsupported filtering."
                if removed == 0
                else "Selected as the best-supported balance of cell retention and "
                "quality control."
            )
        elif removed == 0:
            reason = (
                "Not selected because it retained the same cells while adding a "
                "filtering rule that was not needed."
            )
        elif removed is not None:
            reason = (
                f"Not selected because it removed {removed:,} additional cells "
                "without stronger support."
            )
        else:
            reason = "Evaluated but not selected for the final cell set."
        branches.append(
            _tree_branch(
                label=_qc_profile_label(profile),
                status="Selected" if selected else "Not selected",
                state="selected" if selected else "alternative",
                metrics=metrics,
                reason=reason,
            )
        )
    branches.sort(key=lambda branch: branch["state"] != "selected")
    return {
        "question": "Which cells should be retained?",
        "description": (
            "The workflow compared the registered quality-control choices before "
            "changing the cell set."
        ),
        "branches": branches,
    }


def _feature_tree_stage(
    plan: Mapping[str, Any],
    inventories: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    assay_plans = _mappings(plan.get("assays"))
    selected_assay = next(
        (assay for assay in assay_plans if assay.get("graphEligible") is True),
        assay_plans[0] if assay_plans else {},
    )
    if not selected_assay:
        return None
    feature_method = str(selected_assay.get("featureMethod") or "none")
    feature_labels = {
        "hvg": "Most variable genes",
        "prevalentPeaks": "Frequently observed chromatin regions",
        "panel": "Predefined feature panel",
        "none": "No feature subset",
    }
    parameters = _mapping(selected_assay.get("featureParameters"))
    metrics: list[str] = []
    top_n = parameters.get("topN")
    min_cells = parameters.get("minCells")
    if isinstance(top_n, int):
        metrics.append(f"Selected {top_n:,} features")
    if isinstance(min_cells, int):
        metrics.append(f"Required presence in at least {min_cells:,} cells")
    excluded = [
        _feature_family_label(item)
        for item in _text_values(parameters.get("excludeFamilies"))
    ]
    protected = [
        _feature_family_label(item)
        for item in _text_values(parameters.get("protectFamilies"))
    ]
    if excluded:
        metrics.append(f"Excluded {_format_text_list(excluded)}")
    if protected:
        metrics.append(f"Kept {_format_text_list(protected)} eligible")
    inventory = _default_inventory_for_assay(
        inventories,
        str(selected_assay.get("assay") or ""),
    )
    if inventory:
        match_count = inventory.get("matchCount")
        total_features = inventory.get("totalFeatures")
        if isinstance(match_count, int) and isinstance(total_features, int):
            metrics.append(
                f"Scarf default reference matched {match_count:,} of "
                f"{total_features:,} genes"
            )
        metrics.append(
            "Complete Scarf default blacklist applied: "
            + (
                "yes"
                if inventory.get("appliedToSelectedRepresentation") is True
                else "no"
            )
        )
    return {
        "question": "Which measurements should shape the cell map?",
        "description": (
            "The selected feature policy controls which biological variation can "
            "influence the map."
        ),
        "branches": [
            _tree_branch(
                label=feature_labels.get(
                    feature_method,
                    "Analysis-specific feature set",
                ),
                status="Selected",
                state="selected",
                metrics=metrics,
                reason=(
                    "Selected to emphasize informative variation while limiting "
                    "known unwanted signal."
                ),
            )
        ],
    }


def _batch_tree_stage(
    experimental: Mapping[str, Any],
    parameter: Mapping[str, Any],
    final: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> dict[str, Any] | None:
    decision = _mapping(experimental.get("decision"))
    batch_plan = _mapping(decision.get("batchCorrection"))
    if not batch_plan:
        return None
    native_analyses = _mappings(final.get("nativeAnalyses"))
    if final.get("graphMethod") == "native" and final.get("primaryAssay"):
        selected_native = [
            item
            for item in native_analyses
            if item.get("assay") == final.get("primaryAssay")
        ]
    else:
        selected_native = native_analyses
    adjustment_applied = any(
        _present(item.get("batchCorrection")) for item in selected_native
    )
    native_candidate, harmony_candidate = _harmony_candidate_pair(parameter, final)
    harmony_executed = _harmony_completed(native_candidate) and _harmony_completed(
        harmony_candidate
    )
    degraded = _degraded_protected_columns(native_candidate, harmony_candidate)
    safety = _mappings(experimental.get("batchSafety"))
    unsafe = [item for item in safety if item.get("status") == "unsafe"]
    coefficients = [
        _public_field_label(item.get("coefficient"))
        for item in unsafe
        if _public_field_label(item.get("coefficient"))
    ]
    coefficients = list(dict.fromkeys(coefficients))
    remaining_capacity = [
        _mapping(item.get("estimability")).get("estimableDf") for item in unsafe
    ]
    adjustment_metrics: list[str] = []
    if coefficients:
        adjustment_metrics.append(
            f"Protected comparisons at risk: {_format_text_list(coefficients)}"
        )
    if remaining_capacity and all(value == 0 for value in remaining_capacity):
        adjustment_metrics.append("Remaining comparison capacity: 0")
    if harmony_candidate:
        harmony_parameters = _mapping(harmony_candidate.get("parameters"))
        adjustment_metrics.append(
            "Matched parameters: "
            f"{_scalar(harmony_parameters.get('dimensions'))} dimensions, "
            f"{_scalar(harmony_parameters.get('neighborsK'))} neighbors, "
            f"resolution {_scalar(harmony_parameters.get('leidenResolution'))}"
        )
    if harmony_executed:
        adjustment_metrics.insert(0, "Run status: completed diagnostic")
        native_metrics = _mapping(native_candidate.get("metrics"))
        harmony_metrics = _mapping(harmony_candidate.get("metrics"))
        native_batch = _mapping(native_metrics.get("batchMixing"))
        harmony_batch = _mapping(harmony_metrics.get("batchMixing"))
        for column in dict.fromkeys([*native_batch, *harmony_batch]):
            adjustment_metrics.append(
                f"{_public_field_label(column).capitalize()} mixing: "
                f"{_score_transition(native_batch.get(column), harmony_batch.get(column))}"
            )
        if degraded:
            adjustment_metrics.append(
                "Protected evidence degraded: " + _format_text_list(degraded)
            )
    correction_license = _active_decision(decisions, "correctionLicense")
    diagnostic_only = str(correction_license.get("selectedOptionId") or "").endswith(
        "unsafeConfounded"
    )
    if diagnostic_only:
        adjustment_metrics.append("Selection license: diagnostic only")
    action = str(batch_plan.get("action") or "")
    if adjustment_applied:
        unadjusted_state = "alternative"
        adjusted_state = "selected"
        unadjusted_status = "Not selected"
        adjusted_status = "Selected"
        unadjusted_reason = (
            "The adjusted result provided stronger supported comparability."
        )
        adjusted_reason = (
            "Selected because it improved technical comparability while preserving "
            "the biological structure being studied."
        )
    else:
        unadjusted_state = "selected"
        adjusted_state = (
            "rejected"
            if harmony_executed
            else ("blocked" if action in {"unsafe", "skip"} else "alternative")
        )
        unadjusted_status = "Selected"
        adjusted_status = (
            "Run diagnostically; rejected"
            if harmony_executed
            else ("Not run" if adjusted_state == "blocked" else "Not selected")
        )
        unadjusted_reason = (
            "Selected after the matched diagnostic retained more of the protected "
            "biological structure."
            if harmony_executed
            else "Selected because adjustment was not shown to improve the data safely."
        )
        adjusted_reason = (
            "Rejected because protected evidence degraded for "
            f"{_format_text_list(degraded)}"
            + (
                " and the design allowed diagnostic use only."
                if diagnostic_only
                else "."
            )
            if harmony_executed and degraded
            else (
                "Run as a matched diagnostic but not selected."
                if harmony_executed
                else (
                    "Not run because technical and biological differences could "
                    "not be separated safely."
                    if adjusted_state == "blocked"
                    else "Tested but did not provide a safer improvement over the "
                    "unadjusted data."
                )
            )
        )
    return {
        "question": "Should technical variation be adjusted?",
        "description": (
            "Adjustment was accepted only if it improved comparability without "
            "removing protected biological differences."
        ),
        "branches": [
            _tree_branch(
                label="Use the unadjusted representation",
                status=unadjusted_status,
                state=unadjusted_state,
                metrics=[
                    "Final representation: native",
                    "Protected biological comparisons retained",
                ],
                reason=unadjusted_reason,
            ),
            _tree_branch(
                label="Apply Harmony batch adjustment",
                status=adjusted_status,
                state=adjusted_state,
                metrics=adjustment_metrics,
                reason=adjusted_reason,
            ),
        ],
    }


def _selected_parameter_context(
    parameter: Mapping[str, Any],
    final: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    assay_reports = _mapping(parameter.get("assayReports"))
    preferred_assay = str(
        parameter.get("graphAssay")
        or final.get("primaryAssay")
        or parameter.get("fromAssay")
        or ""
    )
    report = _mapping(assay_reports.get(preferred_assay))
    if not report and assay_reports:
        report = _mapping(next(iter(assay_reports.values())))
    if not report:
        report = dict(parameter)
    evaluations = _mappings(report.get("evaluations"))
    recommended = _mapping(parameter.get("recommendedByAssay"))
    selected_id = (
        recommended.get(preferred_assay)
        or report.get("recommendedCandidateId")
        or parameter.get("recommendedCandidateId")
    )
    selected = next(
        (
            evaluation
            for evaluation in evaluations
            if evaluation.get("candidateId") == selected_id
        ),
        {},
    )
    return report, evaluations, selected


def _harmony_candidate_pair(
    parameter: Mapping[str, Any],
    final: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _report, evaluations, _selected = _selected_parameter_context(parameter, final)
    for harmony in reversed(evaluations):
        harmony_parameters = _mapping(harmony.get("parameters"))
        if harmony_parameters.get("useHarmony") is not True:
            continue
        signature = {
            key: value
            for key, value in harmony_parameters.items()
            if key not in {"candidateId", "useHarmony"}
        }
        native_candidates = [
            evaluation
            for evaluation in evaluations
            if _mapping(evaluation.get("parameters")).get("useHarmony") is False
            and {
                key: value
                for key, value in _mapping(evaluation.get("parameters")).items()
                if key not in {"candidateId", "useHarmony"}
            }
            == signature
        ]
        if not native_candidates:
            continue
        expected_native_id = str(harmony.get("candidateId") or "").replace(
            "_correction_harmony",
            "_correction_native",
        )
        native = next(
            (
                evaluation
                for evaluation in native_candidates
                if evaluation.get("candidateId") == expected_native_id
            ),
            native_candidates[-1],
        )
        return native, harmony
    return {}, {}


def _harmony_completed(evaluation: Mapping[str, Any]) -> bool:
    return (
        evaluation.get("status") == "done" and evaluation.get("eligible") is not False
    )


def _score_transition(native: Any, harmony: Any) -> str:
    if not isinstance(native, (int, float)) or isinstance(native, bool):
        return "Not available"
    if not isinstance(harmony, (int, float)) or isinstance(harmony, bool):
        return "Not available"
    delta = float(harmony) - float(native)
    return f"{float(native):.3f} to {float(harmony):.3f} (change {delta:+.3f})"


def _harmony_metric_rows(
    native: Mapping[str, Any],
    harmony: Mapping[str, Any],
) -> list[dict[str, Any]]:
    native_metrics = _mapping(native.get("metrics"))
    harmony_metrics = _mapping(harmony.get("metrics"))
    rows: list[dict[str, Any]] = []

    def add(
        category: str,
        metric: str,
        native_value: Any,
        harmony_value: Any,
        interpretation: str,
    ) -> None:
        delta = (
            float(harmony_value) - float(native_value)
            if isinstance(native_value, (int, float))
            and not isinstance(native_value, bool)
            and isinstance(harmony_value, (int, float))
            and not isinstance(harmony_value, bool)
            else None
        )
        rows.append(
            {
                "category": category,
                "metric": metric,
                "native": native_value,
                "Harmony": harmony_value,
                "change": delta,
                "interpretation": interpretation,
            }
        )

    native_batch = _mapping(native_metrics.get("batchMixing"))
    harmony_batch = _mapping(harmony_metrics.get("batchMixing"))
    for column in dict.fromkeys([*native_batch, *harmony_batch]):
        add(
            "Batch removal",
            f"{_public_field_label(column)} mixing",
            native_batch.get(column),
            harmony_batch.get(column),
            "Higher values indicate stronger mixing across the technical group.",
        )

    native_association = _mapping(native_metrics.get("technicalAssociation"))
    harmony_association = _mapping(harmony_metrics.get("technicalAssociation"))
    for column in dict.fromkeys([*native_association, *harmony_association]):
        add(
            "Technical association",
            _public_field_label(column),
            native_association.get(column),
            harmony_association.get(column),
            "Lower values indicate less association with the technical group.",
        )

    native_biology = _mapping(native_metrics.get("biologicalPreservation"))
    harmony_biology = _mapping(harmony_metrics.get("biologicalPreservation"))
    for column in dict.fromkeys([*native_biology, *harmony_biology]):
        native_scores = _mapping(native_biology.get(column))
        harmony_scores = _mapping(harmony_biology.get(column))
        for name in dict.fromkeys([*native_scores, *harmony_scores]):
            add(
                "Protected biology",
                f"{_public_field_label(column)} {_label(name)}",
                native_scores.get(name),
                harmony_scores.get(name),
                "Protected evidence should not decrease materially.",
            )

    for key, label, interpretation in (
        (
            "crossUnitSupport",
            "Cross-sample support",
            "Higher values indicate broader support across study units.",
        ),
        (
            "markerCoherence",
            "Marker coherence",
            "Higher values indicate more groups with coherent markers.",
        ),
        (
            "markerSpecificityMedian",
            "Median marker specificity",
            "Higher values indicate more group-specific markers.",
        ),
        (
            "clusterConnectivity",
            "Cluster connectivity",
            "Higher values indicate better connected groups.",
        ),
        (
            "membershipStrengthMean",
            "Mean membership strength",
            "Higher values indicate more stable cluster membership.",
        ),
        (
            "doubletHighScoreConcentration",
            "Doublet-score concentration",
            "Lower values indicate less concentration of high doublet scores.",
        ),
    ):
        if key in native_metrics or key in harmony_metrics:
            add(
                "Supporting diagnostic",
                label,
                native_metrics.get(key),
                harmony_metrics.get(key),
                interpretation,
            )
    return rows


def _degraded_protected_columns(
    native: Mapping[str, Any],
    harmony: Mapping[str, Any],
    *,
    tolerance: float = 0.05,
) -> list[str]:
    native_biology = _mapping(
        _mapping(native.get("metrics")).get("biologicalPreservation")
    )
    harmony_biology = _mapping(
        _mapping(harmony.get("metrics")).get("biologicalPreservation")
    )
    degraded: list[str] = []
    for column, raw_native in native_biology.items():
        native_scores = _mapping(raw_native)
        harmony_scores = _mapping(harmony_biology.get(column))
        if any(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(harmony_scores.get(name), (int, float))
            and not isinstance(harmony_scores.get(name), bool)
            and float(harmony_scores[name]) < float(value) - tolerance
            for name, value in native_scores.items()
        ):
            degraded.append(_public_field_label(column))
    return degraded


def _active_decision(
    decisions: Mapping[str, Any],
    decision_id: str,
) -> dict[str, Any]:
    return _mapping(decisions.get(decision_id))


def _common_parameter(
    evaluations: Sequence[Mapping[str, Any]],
    key: str,
) -> Any:
    values = [
        _mapping(evaluation.get("parameters")).get(key)
        for evaluation in evaluations
        if _present(_mapping(evaluation.get("parameters")).get(key))
    ]
    return Counter(values).most_common(1)[0][0] if values else None


def _parameter_options(
    evaluations: Sequence[Mapping[str, Any]],
    key: str,
    filters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_value: dict[Any, dict[str, Any]] = {}
    for evaluation in evaluations:
        if evaluation.get("status") != "done" or evaluation.get("eligible") is False:
            continue
        parameters = _mapping(evaluation.get("parameters"))
        if any(parameters.get(name) != value for name, value in filters.items()):
            continue
        value = parameters.get(key)
        if not _present(value):
            continue
        current = by_value.get(value)
        current_metrics = _mapping(current.get("metrics")) if current else {}
        metrics = _mapping(evaluation.get("metrics"))
        if current is None or len(metrics) > len(current_metrics):
            by_value[value] = dict(evaluation)
    return [
        by_value[value]
        for value in sorted(
            by_value,
            key=lambda item: (not isinstance(item, (int, float)), item),
        )
    ]


def _candidate_metrics(
    evaluation: Mapping[str, Any],
    *,
    include_stability: bool = False,
) -> list[str]:
    metrics = _mapping(evaluation.get("metrics"))
    values: list[str] = []
    clusters = metrics.get("nClusters")
    separation = metrics.get("graphSilhouetteMedian")
    smallest = metrics.get("minClusterCells")
    if isinstance(clusters, int):
        values.append(f"Cell groups: {clusters:,}")
    if isinstance(separation, (int, float)):
        values.append(f"Separation score: {float(separation):.3f}")
    if isinstance(smallest, int):
        values.append(f"Smallest group: {smallest:,} cells")
    if include_stability:
        seed = metrics.get("seedStability")
        subsample = metrics.get("subsampleStability")
        marker = metrics.get("markerCoherence")
        support = metrics.get("crossUnitSupport")
        if isinstance(seed, (int, float)):
            values.append(f"Repeat-run stability: {float(seed):.3f}")
        if isinstance(subsample, (int, float)):
            values.append(f"Subsample stability: {float(subsample):.3f}")
        if isinstance(marker, (int, float)):
            values.append(f"Marker coherence: {float(marker):.3f}")
        if isinstance(support, (int, float)):
            values.append(f"Cross-sample support: {float(support):.3f}")
    return values


def _parameter_tree_stage(
    *,
    question: str,
    description: str,
    options: Sequence[Mapping[str, Any]],
    parameter_name: str,
    selected_value: Any,
    label: Any,
    selected_reason: str,
    alternative_reason: Any,
    include_stability: bool = False,
) -> dict[str, Any] | None:
    if not options:
        return None
    branches: list[dict[str, Any]] = []
    for evaluation in options:
        value = _mapping(evaluation.get("parameters")).get(parameter_name)
        selected = value == selected_value
        branches.append(
            _tree_branch(
                label=str(label(value)),
                status="Selected" if selected else "Not selected",
                state="selected" if selected else "alternative",
                metrics=_candidate_metrics(
                    evaluation,
                    include_stability=include_stability and selected,
                ),
                reason=(
                    selected_reason
                    if selected
                    else str(alternative_reason(value, evaluation))
                ),
            )
        )
    return {
        "question": question,
        "description": description,
        "branches": branches,
    }


def _parameter_tree_stages(
    parameter: Mapping[str, Any],
    final: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _report, evaluations, selected = _selected_parameter_context(parameter, final)
    if not evaluations or not selected:
        return [], selected
    selected_parameters = _mapping(selected.get("parameters"))
    selected_dimensions = selected_parameters.get("dimensions")
    selected_neighbors = selected_parameters.get("neighborsK")
    selected_resolution = selected_parameters.get("leidenResolution")
    selected_harmony = selected_parameters.get("useHarmony")
    common_neighbors = _common_parameter(evaluations, "neighborsK")
    common_resolution = _common_parameter(evaluations, "leidenResolution")

    dimension_options = _parameter_options(
        evaluations,
        "dimensions",
        {
            "neighborsK": common_neighbors,
            "leidenResolution": common_resolution,
            "useHarmony": selected_harmony,
        },
    )
    neighbor_options = _parameter_options(
        evaluations,
        "neighborsK",
        {
            "dimensions": selected_dimensions,
            "leidenResolution": common_resolution,
            "useHarmony": selected_harmony,
        },
    )
    resolution_options = _parameter_options(
        evaluations,
        "leidenResolution",
        {
            "dimensions": selected_dimensions,
            "neighborsK": selected_neighbors,
            "useHarmony": selected_harmony,
        },
    )

    def dimension_alternative(value: Any, _evaluation: Mapping[str, Any]) -> str:
        if isinstance(value, (int, float)) and isinstance(
            selected_dimensions, (int, float)
        ):
            if value > selected_dimensions:
                return (
                    "Not selected because the smaller representation retained "
                    "sufficient structure with less added noise."
                )
            return "Not selected because it retained too little stable structure."
        return "Evaluated but not selected."

    def neighbor_alternative(value: Any, _evaluation: Mapping[str, Any]) -> str:
        if isinstance(value, (int, float)) and isinstance(
            selected_neighbors, (int, float)
        ):
            if value < selected_neighbors:
                return (
                    "Provided finer local detail but produced smaller, less stable "
                    "groups."
                )
            return "Smoothed across more cells and reduced useful local detail."
        return "Evaluated but not selected."

    selected_metrics = _mapping(selected.get("metrics"))
    selected_separation = selected_metrics.get("graphSilhouetteMedian")

    def resolution_alternative(
        _value: Any,
        evaluation: Mapping[str, Any],
    ) -> str:
        metrics = _mapping(evaluation.get("metrics"))
        groups = metrics.get("nClusters")
        separation = metrics.get("graphSilhouetteMedian")
        if isinstance(groups, int) and isinstance(separation, (int, float)):
            return (
                f"Produced {groups:,} groups with separation "
                f"{float(separation):.3f}, weaker than the selected balance."
            )
        if isinstance(selected_separation, (int, float)):
            return (
                f"Did not match the selected separation score of "
                f"{float(selected_separation):.3f}."
            )
        return "Evaluated but not selected."

    stages = [
        stage
        for stage in (
            _parameter_tree_stage(
                question="How many variation patterns should be retained?",
                description=(
                    "Dimensions are compressed patterns of gene variation used to "
                    "build the cell map."
                ),
                options=dimension_options,
                parameter_name="dimensions",
                selected_value=selected_dimensions,
                label=lambda value: f"{int(value):,} dimensions",
                selected_reason=(
                    "Selected as the smallest representation that retained a stable "
                    "cell map."
                ),
                alternative_reason=dimension_alternative,
            ),
            _parameter_tree_stage(
                question="How local should each cell neighborhood be?",
                description=(
                    "Smaller neighborhoods emphasize local detail; larger ones "
                    "produce broader smoothing."
                ),
                options=neighbor_options,
                parameter_name="neighborsK",
                selected_value=selected_neighbors,
                label=lambda value: f"{int(value):,} nearest neighbors",
                selected_reason=(
                    "Selected to balance local detail with stable cell-group sizes."
                ),
                alternative_reason=neighbor_alternative,
            ),
            _parameter_tree_stage(
                question="How finely should cells be divided into groups?",
                description=(
                    "Resolution controls whether the final map contains broader or "
                    "more finely divided cell groups."
                ),
                options=resolution_options,
                parameter_name="leidenResolution",
                selected_value=selected_resolution,
                label=lambda value: f"Resolution {float(value):g}",
                selected_reason=(
                    "Selected for the strongest supported separation, stability, "
                    "marker coherence, and group sizes."
                ),
                alternative_reason=resolution_alternative,
                include_stability=True,
            ),
        )
        if stage is not None and len(stage["branches"]) > 1
    ]
    return stages, selected


def _analysis_tree_stages(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    plan = _mapping(workflow_result.get("preprocessingPlan"))
    final = _mapping(workflow_result.get("finalAnalysis"))
    experimental = _latest(reports, "experimental_context")
    parameter = _latest(reports, "parameter_tuning")
    biology = _latest(reports, "biological_interpretation")
    decisions = _mapping(payload.get("activeDecisions"))
    inventories = _mappings(payload.get("defaultFeatureInventories"))
    cluster_counts = _mapping(payload.get("clusterCounts"))
    total_cells = sum(int(value) for value in cluster_counts.values())
    stages: list[dict[str, Any]] = []
    for stage in (
        _qc_tree_stage(experimental, plan, total_cells),
        _feature_tree_stage(plan, inventories),
    ):
        if stage is not None:
            stages.append(stage)
    stages.extend(_hvg_tree_stages(_mapping(payload.get("hvgEvidence"))))
    batch_stage = _batch_tree_stage(experimental, parameter, final, decisions)
    if batch_stage is not None:
        stages.append(batch_stage)
    parameter_stages, selected = _parameter_tree_stages(parameter, final)
    stages.extend(parameter_stages)

    interpretations = _mappings(biology.get("clusterInterpretations"))
    final_metrics = [f"Cells analyzed: {total_cells:,}"] if total_cells else []
    final_metrics.extend(_candidate_metrics(selected, include_stability=True))
    if not selected and cluster_counts:
        final_metrics.append(f"Cell groups: {len(cluster_counts):,}")
    stages.append(
        {
            "question": "Which result became the final analysis?",
            "description": (
                "Only the selected branch was carried into visualization and marker "
                "analysis."
            ),
            "branches": [
                _tree_branch(
                    label=(
                        f"{len(cluster_counts):,} cell groups"
                        if cluster_counts
                        else "Final selected cell map"
                    ),
                    status="Final result",
                    state="selected",
                    metrics=final_metrics,
                    reason=(
                        f"{len(interpretations):,} groups also received biological "
                        "interpretations."
                        if interpretations
                        else "No biological cell-type labels were inferred."
                    ),
                )
            ],
        }
    )
    return stages


def _tree_connector_svg(
    branch_count: int,
    selected_index: int,
    stage_index: int,
    *,
    continues: bool,
) -> tuple[str, str]:
    width = 1200
    centers = [(index + 0.5) * width / branch_count for index in range(branch_count)]
    branch_marker_id = f"tree-branch-arrow-{stage_index}"
    if branch_count == 1:
        branch_paths = (
            f'<path d="M {width / 2:g} 0 V 78" marker-end="url(#{branch_marker_id})"/>'
        )
    else:
        branch_paths = (
            f'<path d="M {width / 2:g} 0 V 28"/>'
            f'<path d="M {centers[0]:g} 28 H {centers[-1]:g}"/>'
            + "".join(
                f'<path d="M {center:g} 28 V 78" '
                f'marker-end="url(#{branch_marker_id})"/>'
                for center in centers
            )
        )
    branch_definitions = (
        f'<defs><marker id="{branch_marker_id}" markerWidth="8" markerHeight="8" '
        'refX="7" refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,6 L7,3 z"/></marker></defs>'
    )
    branch_svg = (
        '<svg class="tree-branch-connectors" viewBox="0 0 1200 84" '
        'preserveAspectRatio="none" aria-hidden="true">'
        f"{branch_definitions}{branch_paths}</svg>"
    )
    if not continues:
        return branch_svg, ""
    selected_x = centers[selected_index]
    selection_marker_id = f"tree-selection-arrow-{stage_index}"
    selection_path = (
        f"M {selected_x:g} 0 V 28 H {width / 2:g} V 78"
        if selected_x != width / 2
        else f"M {width / 2:g} 0 V 78"
    )
    selection_definitions = (
        f'<defs><marker id="{selection_marker_id}" markerWidth="8" '
        'markerHeight="8" refX="7" refY="3" orient="auto" '
        'markerUnits="strokeWidth"><path d="M0,0 L0,6 L7,3 z"/>'
        "</marker></defs>"
    )
    selection_svg = (
        '<svg class="tree-selection-connector" viewBox="0 0 1200 84" '
        'preserveAspectRatio="none" aria-hidden="true">'
        f'{selection_definitions}<path d="{selection_path}" '
        f'marker-end="url(#{selection_marker_id})"/></svg>'
    )
    return branch_svg, selection_svg


def _render_decision_tree(stages: Sequence[Mapping[str, Any]]) -> str:
    if not stages:
        return '<p class="empty">No completed analysis decisions were available.</p>'
    rendered: list[str] = []
    for stage_index, stage in enumerate(stages, start=1):
        branches = _mappings(stage.get("branches"))
        if not branches:
            continue
        selected_index = next(
            (
                index
                for index, branch in enumerate(branches)
                if branch.get("state") == "selected"
            ),
            0,
        )
        branch_svg, selection_svg = _tree_connector_svg(
            len(branches),
            selected_index,
            stage_index,
            continues=stage_index < len(stages),
        )
        branch_markup = "".join(
            '<article class="tree-branch tree-branch-{}">'.format(
                html.escape(str(branch.get("state") or "alternative"), quote=True)
            )
            + '<span class="tree-branch-status">{}</span>'.format(
                html.escape(str(branch.get("status") or "Evaluated"))
            )
            + f"<h3>{html.escape(str(branch.get('label') or 'Option'))}</h3>"
            + (
                '<ul class="tree-metrics">'
                + "".join(
                    f"<li>{html.escape(metric)}</li>"
                    for metric in _text_values(branch.get("metrics"))
                )
                + "</ul>"
                if _present(branch.get("metrics"))
                else ""
            )
            + (
                f"<p>{html.escape(_brief_text(branch.get('reason')))}</p>"
                if _brief_text(branch.get("reason"))
                else ""
            )
            + "</article>"
            for branch in branches
        )
        rendered.append(
            '<section class="tree-stage">'
            '<div class="tree-question">'
            f"<span>Decision {stage_index}</span>"
            f"<strong>{html.escape(str(stage.get('question') or 'Analysis decision'))}</strong>"
            "</div>"
            + (
                f'<p class="tree-stage-description">{html.escape(_brief_text(stage.get("description")))}</p>'
                if _brief_text(stage.get("description"))
                else ""
            )
            + branch_svg
            + '<div class="tree-branches" style="--branch-count: {}">'.format(
                len(branches)
            )
            + branch_markup
            + "</div>"
            + selection_svg
            + "</section>"
        )
    return (
        '<div class="decision-tree" aria-label="Analysis decision tree">'
        + "".join(rendered)
        + "</div>"
    )


def _render_selection_evidence(payload: Mapping[str, Any]) -> str:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    final = _mapping(workflow_result.get("finalAnalysis"))
    parameter = _latest(reports, "parameter_tuning")
    _report, _evaluations, selected = _selected_parameter_context(parameter, final)
    metrics = _mapping(selected.get("metrics"))
    cards: list[tuple[str, str, str]] = []
    candidate_count = parameter.get("totalCandidates")
    if isinstance(candidate_count, int):
        cards.append(
            (
                "Settings compared",
                f"{candidate_count:,}",
                "Completed parameter combinations considered before selection.",
            )
        )
    card_specs = (
        (
            "graphSilhouetteMedian",
            "Group separation",
            "Higher values indicate clearer separation between neighboring groups.",
        ),
        (
            "minClusterCells",
            "Smallest group",
            "Number of cells in the smallest selected group.",
        ),
        (
            "seedStability",
            "Repeat-run stability",
            "Agreement when clustering is repeated with a different random seed.",
        ),
        (
            "subsampleStability",
            "Subsample stability",
            "Agreement when the analysis is repeated on a subset of cells.",
        ),
        (
            "markerCoherence",
            "Marker coherence",
            "Consistency of marker support across the selected groups.",
        ),
        (
            "crossUnitSupport",
            "Cross-sample support",
            "Support for the selected groups across the study units.",
        ),
    )
    for key, label, explanation in card_specs:
        value = metrics.get(key)
        if isinstance(value, int):
            display = f"{value:,} cells" if key == "minClusterCells" else f"{value:,}"
        elif isinstance(value, float):
            display = f"{value:.3f}"
        else:
            continue
        cards.append((label, display, explanation))
    if not cards:
        return ""
    return '<div class="summary-grid">{}</div>'.format(
        "".join(
            '<article class="summary-card">'
            f'<p class="summary-label">{html.escape(label)}</p>'
            f"<h3>{html.escape(value)}</h3>"
            f"<p>{html.escape(explanation)}</p>"
            "</article>"
            for label, value, explanation in cards
        )
    )


def _analysis_percent(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "Not available"
    return f"{float(value):.1%}"


def _analysis_number_range(values: Sequence[Any]) -> str:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numbers:
        return "Not available"
    low = min(numbers)
    high = max(numbers)

    def display(value: float) -> str:
        if abs(value) >= 100:
            return f"{value:,.0f}"
        return f"{value:,.3f}".rstrip("0").rstrip(".")

    if low == high:
        return display(low)
    return f"{display(low)} to {display(high)}"


def _render_evidence_choices(choices: Sequence[Mapping[str, Any]]) -> str:
    return '<div class="evidence-choice-grid">{}</div>'.format(
        "".join(
            '<article class="evidence-choice evidence-choice-{}">'.format(
                html.escape(str(choice.get("state") or "reviewed"), quote=True)
            )
            + '<span class="evidence-choice-status">{}</span>'.format(
                html.escape(str(choice.get("status") or "Reviewed"))
            )
            + f"<h3>{html.escape(str(choice.get('label') or 'Evidence'))}</h3>"
            + _render_plain_list(_text_values(choice.get("metrics")))
            + (
                f"<p>{html.escape(_brief_text(choice.get('reason')))}</p>"
                if _brief_text(choice.get("reason"))
                else ""
            )
            + "</article>"
            for choice in choices
        )
    )


def _render_evidence_measurements(
    measurements: Sequence[tuple[str, str, str]],
) -> str:
    if not measurements:
        return ""
    return '<dl class="evidence-measurement-grid">{}</dl>'.format(
        "".join(
            '<div class="evidence-measurement">'
            f"<dt>{html.escape(label)}</dt>"
            f"<dd>{html.escape(value)}"
            + (f"<small>{html.escape(detail)}</small>" if detail else "")
            + "</dd></div>"
            for label, value, detail in measurements
        )
    )


def _render_evidence_panel(
    *,
    title: str,
    outcome: str,
    introduction: str,
    body: str,
    measurements: str = "",
    expanded: bool = False,
) -> str:
    measurement_markup = (
        '<details class="evidence-measurements"><summary>Measurements</summary>'
        f'<div class="evidence-measurements-body">{measurements}</div></details>'
        if measurements
        else ""
    )
    open_attribute = " open" if expanded else ""
    return (
        f'<details class="evidence-panel"{open_attribute}><summary><span>'
        f'<span class="evidence-panel-title">{html.escape(title)}</span>'
        f'<span class="evidence-panel-outcome">{html.escape(outcome)}</span>'
        "</span></summary>"
        '<div class="evidence-panel-body">'
        f"<p>{html.escape(introduction)}</p>{body}{measurement_markup}</div></details>"
    )


def _qc_profile_scope(profile: Mapping[str, Any]) -> str:
    bounds = _qc_resolved_bounds(profile)
    groups = {str(item.get("group")) for item in bounds if _present(item.get("group"))}
    return "Per-library thresholds" if len(groups) > 1 else "Global thresholds"


def _qc_resolved_bounds(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct = _mappings(profile.get("resolvedBounds"))
    if direct:
        return direct
    return _mappings(_mapping(profile.get("parameters")).get("resolvedBounds"))


def _qc_flag_summary(profile: Mapping[str, Any]) -> list[str]:
    labels = (
        ("nCounts:high", "High RNA count flags"),
        ("nCounts:lowQuality", "Low RNA count flags"),
        ("nFeatures:high", "High detected-gene flags"),
        ("nFeatures:lowQuality", "Low detected-gene flags"),
        ("percentMito:highMito", "High mitochondrial-percentage flags"),
        ("percentRibo:highRibo", "High ribosomal-percentage flags"),
    )
    flags = _mapping(profile.get("flaggedCells"))
    values: list[str] = []
    for suffix, label in labels:
        count = next(
            (
                value
                for key, value in flags.items()
                if str(key).endswith(suffix) and isinstance(value, int)
            ),
            None,
        )
        if count is not None:
            values.append(f"{label}: {count:,}")
    return values


def _qc_bound_summary(profile: Mapping[str, Any]) -> str:
    bounds = _qc_resolved_bounds(profile)
    parts: list[str] = []
    for role, label in (
        ("count", "RNA counts"),
        ("feature", "Detected genes"),
        ("mitochondrial", "Mitochondrial percentage"),
        ("ribosomal", "Ribosomal percentage"),
    ):
        matching = [item for item in bounds if item.get("role") == role]
        if not matching:
            continue
        lower = _analysis_number_range([item.get("lowerRemoval") for item in matching])
        upper_removal = _analysis_number_range(
            [item.get("upperRemoval") for item in matching]
        )
        upper_flag = _analysis_number_range(
            [item.get("upperFlag") for item in matching]
        )
        cutoffs = [
            value
            for value in (
                f"lower cutoff {lower}" if lower != "Not available" else "",
                (
                    f"upper cutoff {upper_removal}"
                    if upper_removal != "Not available"
                    else ""
                ),
                (
                    f"high-value flag {upper_flag}"
                    if upper_flag != "Not available"
                    else ""
                ),
            )
            if value
        ]
        if cutoffs:
            parts.append(f"{label}: {'; '.join(cutoffs)}")
    return ". ".join(parts)


def _qc_metric_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    bounds = _qc_resolved_bounds(profile)
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for bound in bounds:
        metric = str(bound.get("metric") or bound.get("role") or "")
        if metric:
            by_metric.setdefault(metric, []).append(bound)
    capture_comparisons = _mappings(
        _mapping(profile.get("parameters")).get("captureComparisons")
    )
    if capture_comparisons:
        first_metrics = _mapping(capture_comparisons[0].get("metricComparisons"))
        for metric, raw in first_metrics.items():
            if str(metric).startswith("artifact_") or metric in by_metric:
                continue
            comparison = _mapping(raw)
            by_metric[metric] = [
                {
                    "metric": metric,
                    "group": "global diagnostic",
                    "role": comparison.get("role"),
                    "median": comparison.get("globalMedian"),
                    "diagnosticLower": comparison.get("globalLower"),
                    "diagnosticUpper": comparison.get("globalUpper"),
                }
            ]
    flags = _mapping(profile.get("metricFlaggedCells"))
    rows: list[dict[str, Any]] = []
    for metric, metric_bounds in by_metric.items():
        medians = [item.get("median") for item in metric_bounds]
        lower = [item.get("lowerRemoval") for item in metric_bounds]
        upper = [item.get("upperRemoval") for item in metric_bounds]
        high_flag = [item.get("upperFlag") for item in metric_bounds]
        diagnostic_range = [
            item.get(field)
            for item in metric_bounds
            for field in ("diagnosticLower", "diagnosticUpper")
        ]
        metric_flags = _mapping(flags.get(metric))
        rows.append(
            {
                "metric": _public_field_label(metric),
                "scope": (
                    "Global diagnostic only"
                    if any(value is not None for value in diagnostic_range)
                    else _qc_profile_scope({"resolvedBounds": metric_bounds})
                ),
                "median": _analysis_number_range(medians),
                "diagnostic reference": _analysis_number_range(diagnostic_range),
                "lower cutoff": _analysis_number_range(lower),
                "upper cutoff": _analysis_number_range(upper),
                "high flag": _analysis_number_range(high_flag),
                "flagged cells": sum(
                    int(value)
                    for value in metric_flags.values()
                    if isinstance(value, int)
                ),
            }
        )
    return rows


def _qc_profile_rows(profiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        active = profile.get("activeCells")
        retained = profile.get("retainedCells")
        removed = (
            active - retained
            if isinstance(active, int) and isinstance(retained, int)
            else None
        )
        rows.append(
            {
                "profile": _qc_profile_label(profile),
                "scope": _qc_profile_scope(profile),
                "active cells": active,
                "retained cells": retained,
                "removed cells": removed,
                "flags": _qc_flag_summary(profile),
                "failed libraries": len(
                    _text_values(profile.get("failedCaptureCandidates"))
                ),
            }
        )
    return rows


def _qc_bound_rows(profiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for bound in _qc_resolved_bounds(profile):
            rows.append(
                {
                    "profile": _qc_profile_label(profile),
                    "group": bound.get("group"),
                    "metric": _public_field_label(
                        bound.get("metric") or bound.get("role")
                    ),
                    "median": bound.get("median"),
                    "lower removal": bound.get("lowerRemoval"),
                    "upper removal": bound.get("upperRemoval"),
                    "upper flag": bound.get("upperFlag"),
                }
            )
    return rows


def _render_filtering_evidence(
    experimental: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    profiles = _mappings(experimental.get("qcProfiles"))
    if not profiles:
        return ""
    decision = _mapping(experimental.get("decision"))
    cell_qc = _mapping(plan.get("cellQc"))
    if not cell_qc:
        cell_qc = _mapping(decision.get("cellQc"))
    if not cell_qc:
        cell_qc = _mapping(experimental.get("cellQc"))
    selected = _selected_qc_profile(experimental, cell_qc)
    selected_id = selected.get("profileId")
    selected_name = selected.get("registeredProfile")
    choices: list[dict[str, Any]] = []
    for profile in profiles:
        is_selected = bool(
            (selected_id and profile.get("profileId") == selected_id)
            or (
                not selected_id
                and selected_name
                and profile.get("registeredProfile") == selected_name
            )
        )
        active = profile.get("activeCells")
        retained = profile.get("retainedCells")
        metrics: list[str] = []
        removed: int | None = None
        if isinstance(active, int) and isinstance(retained, int):
            removed = active - retained
            metrics.extend(
                (
                    f"Retained: {retained:,} of {active:,}",
                    f"Removed: {removed:,}",
                )
            )
        n_mads = _mapping(profile.get("parameters")).get("nMads")
        if isinstance(n_mads, (int, float)):
            metrics.append(
                f"Threshold distance: {float(n_mads):g} median absolute deviations"
            )
        metrics.append(_qc_profile_scope(profile))
        metrics.extend(_qc_flag_summary(profile))
        choices.append(
            {
                "label": _qc_profile_label(profile),
                "status": "Selected" if is_selected else "Not selected",
                "state": "selected" if is_selected else "rejected",
                "metrics": metrics,
                "reason": (
                    "Preserved every reviewed cell and all recorded study groups."
                    if is_selected
                    else (
                        f"Removed {removed:,} additional cells without stronger "
                        "support."
                        if removed
                        else "Produced the same retained cell set without improving "
                        "the selected rule."
                    )
                ),
            }
        )
    active = selected.get("activeCells")
    retained = selected.get("retainedCells")
    selected_label = _qc_profile_label(selected)
    selected_flags = sum(
        int(value)
        for value in _mapping(selected.get("flaggedCells")).values()
        if isinstance(value, int)
    )
    outcome = (
        f"{selected_label}; {retained:,} of {active:,} cells retained; "
        f"{selected_flags:,} diagnostic flags"
        if isinstance(active, int) and isinstance(retained, int)
        else f"{selected_label} selected"
    )
    measurements: list[tuple[str, str, str]] = []
    for profile in profiles:
        parameters = _mapping(profile.get("parameters"))
        n_mads = parameters.get("nMads")
        rule = (
            f"{float(n_mads):g} median absolute deviations (MAD), "
            f"{_qc_profile_scope(profile).lower()}"
            if isinstance(n_mads, (int, float))
            else _qc_profile_scope(profile)
        )
        measurements.append(
            (
                _qc_profile_label(profile),
                rule,
                ". ".join(
                    value
                    for value in (
                        _qc_bound_summary(profile),
                        "; ".join(_qc_flag_summary(profile)),
                    )
                    if value
                ),
            )
        )
    for column, group_counts in _mapping(selected.get("retainedCellsByColumn")).items():
        counts = list(_mapping(group_counts).values())
        numeric = [value for value in counts if isinstance(value, int)]
        if numeric:
            measurements.append(
                (
                    f"Retention across {_public_field_label(column)}",
                    f"{len(numeric):,} groups",
                    f"{min(numeric):,} to {max(numeric):,} retained cells per group.",
                )
            )
    return _render_evidence_panel(
        title="Cell filtering",
        outcome=outcome,
        introduction=(
            f"{len(profiles):,} registered filtering strategies were compared. "
            "The selected strategy retained the published cell set because stricter "
            "alternatives did not provide stronger support. Its cutoffs are "
            "diagnostic bounds and did not remove cells."
        ),
        body=(
            _render_evidence_choices(choices)
            + '<div class="subsection"><h3>Selected QC metrics and cutoffs</h3>'
            + _table(
                _qc_metric_rows(selected),
                columns=(
                    "metric",
                    "scope",
                    "median",
                    "diagnostic reference",
                    "lower cutoff",
                    "upper cutoff",
                    "high flag",
                    "flagged cells",
                ),
                empty="No selected QC metric cutoffs were recorded.",
            )
            + "</div>"
        ),
        measurements=_render_evidence_measurements(measurements),
        expanded=True,
    )


def _covariate_pair_measurements(
    characterization: Mapping[str, Any],
) -> list[tuple[str, str, str]]:
    measurements: list[tuple[str, str, str]] = []
    for item in _mappings(characterization.get("confounding")):
        coefficient = _public_field_label(item.get("coefficient"))
        for pair in _mappings(item.get("pairs")):
            technical = _public_field_label(pair.get("technical"))
            association = _mapping(pair.get("association"))
            status = str(association.get("status") or "")
            value = association.get("value")
            uncorrected = association.get("valueUncorrected")
            if status == "notComputed":
                display = "Not independently measurable"
            elif isinstance(value, (int, float)):
                display = f"Association score: {float(value):.3f}"
            else:
                display = "Association not available"
            rows_used = association.get("rowsUsed")
            details = (
                [f"{rows_used:,} study units"] if isinstance(rows_used, int) else []
            )
            if status == "notComputed" and isinstance(uncorrected, (int, float)):
                details.append(
                    f"uncorrected association score {float(uncorrected):.3f}"
                )
            elif status:
                details.append("association measured")
            measurements.append(
                (
                    f"{coefficient} and {technical}",
                    display,
                    ("; ".join(details) + ".") if details else "",
                )
            )
    return measurements


def _render_covariate_evidence(experimental: Mapping[str, Any]) -> str:
    characterization = _mapping(experimental.get("characterization"))
    columns = _mappings(characterization.get("columns"))
    if not columns:
        return ""
    domains = Counter(str(item.get("domain") or "unclassified") for item in columns)
    domain_labels = {
        "biological": "Biological variables",
        "technical": "Technical variables",
        "design": "Study-design variables",
        "ignore": "Excluded metadata",
        "unclassified": "Unclassified metadata",
    }
    role_choices = [
        {
            "label": domain_labels.get(domain, _label(domain)),
            "status": "Reviewed",
            "state": "reviewed",
            "metrics": [f"Columns: {count:,}"],
            "reason": "",
        }
        for domain, count in sorted(domains.items())
    ]
    coefficients = _mappings(characterization.get("coefficients"))
    coefficient_names = [
        _public_field_label(item.get("name"))
        for item in coefficients
        if _public_field_label(item.get("name"))
    ]
    outcome = (
        f"{len(columns):,} columns reviewed; "
        f"{_format_text_list(coefficient_names)} selected as study comparisons"
        if coefficient_names
        else f"{len(columns):,} metadata columns reviewed"
    )
    measurements: list[tuple[str, str, str]] = []
    for coefficient in coefficients:
        rows = coefficient.get("designRows")
        observation = _public_field_label(coefficient.get("observationUnit"))
        independent = _public_field_label(coefficient.get("independentUnit"))
        scope = {
            "betweenUnit": "between independent units",
            "withinUnit": "within independent units",
            "mixed": "within and between independent units",
        }.get(
            str(coefficient.get("scope") or ""),
            _label(coefficient.get("scope")).lower(),
        )
        measurements.append(
            (
                _public_field_label(coefficient.get("name")),
                (
                    f"{int(rows):,} {observation} records"
                    if isinstance(rows, int)
                    else "Selected biological comparison"
                ),
                (f"Independent unit: {independent}; comparison type: {scope}."),
            )
        )
    for nesting in _mappings(characterization.get("technicalNesting")):
        left = _public_field_label(nesting.get("left"))
        right = _public_field_label(nesting.get("right"))
        measurements.append(
            (
                "Technical nesting",
                f"{right} is nested within {left}",
                "This structure limits which technical effects can be separated.",
            )
        )
    measurements.extend(_covariate_pair_measurements(characterization))
    return _render_evidence_panel(
        title="Covariate analysis",
        outcome=outcome,
        introduction=(
            "Metadata were classified by role before correction or clustering. "
            "The review separated biological comparisons from technical structure "
            "and metadata that should not guide the analysis."
        ),
        body=_render_evidence_choices(role_choices),
        measurements=_render_evidence_measurements(measurements),
    )


def _feature_family_counts(
    enrichment: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    counts: dict[tuple[str, str], Mapping[str, Any]] = {}
    for inspection in _mappings(enrichment.get("inspections")):
        assay = str(inspection.get("assay") or "")
        for family in _mappings(inspection.get("families")):
            counts[(assay, str(family.get("family") or ""))] = family
    return counts


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


def _default_inventory_family_rows(
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "family": _feature_family_label(family.get("family")),
            "pattern": family.get("pattern"),
            "matched genes": family.get("count"),
            "examples": _text_values(family.get("examples")),
        }
        for family in _mappings(inventory.get("families"))
    ]


def _render_default_inventory_summary(
    inventory: Mapping[str, Any],
    *,
    heading: str,
) -> str:
    if not inventory:
        return ""
    match_count = inventory.get("matchCount")
    total_features = inventory.get("totalFeatures")
    applied = inventory.get("appliedToSelectedRepresentation") is True
    count_summary = (
        f"{int(match_count):,} of {int(total_features):,} genes matched."
        if isinstance(match_count, int) and isinstance(total_features, int)
        else "The exact default pattern was evaluated."
    )
    effect = (
        "The complete default blacklist was applied to the selected representation."
        if applied
        else (
            "The complete default blacklist was evaluated as a reference but was "
            "not applied wholesale to the selected representation."
        )
    )
    blacklist = str(inventory.get("blacklist") or "")
    pattern_markup = (
        "<p><strong>Exact combined pattern:</strong> "
        f"<code>{html.escape(blacklist)}</code></p>"
        if blacklist
        else ""
    )
    return (
        f'<div class="subsection"><h3>{html.escape(heading)}</h3>'
        f"<p>{html.escape(count_summary)} {html.escape(effect)}</p>"
        + _table(
            _default_inventory_family_rows(inventory),
            columns=("family", "pattern", "matched genes", "examples"),
            empty="No default blacklist families were recorded.",
        )
        + pattern_markup
        + "</div>"
    )


def _render_normalization_evidence(
    enrichment: Mapping[str, Any],
    plan: Mapping[str, Any],
    inventories: Sequence[Mapping[str, Any]],
) -> str:
    assay_plans = _mappings(plan.get("assays"))
    if not assay_plans:
        return ""
    families = _feature_family_counts(enrichment)
    choices: list[dict[str, Any]] = []
    measurements: list[tuple[str, str, str]] = []
    outcome_parts: list[str] = []
    for assay_plan in assay_plans:
        assay = str(assay_plan.get("assay") or "Assay")
        normalization = _mapping(assay_plan.get("normalizationParameters"))
        feature_parameters = _mapping(assay_plan.get("featureParameters"))
        inventory = _default_inventory_for_assay(inventories, assay)
        log_transform = normalization.get("logTransform") is True
        renormalize = normalization.get("renormalizeSubset") is True
        normalization_metrics = [
            "Log transform applied" if log_transform else "No log transform",
            (
                "Selected cells renormalized"
                if renormalize
                else "Existing normalization retained"
            ),
        ]
        choices.append(
            {
                "label": f"{_assay_label(assay)} normalization",
                "status": "Selected",
                "state": "selected",
                "metrics": normalization_metrics,
                "reason": "Used consistently for map construction.",
            }
        )
        excluded = [
            str(value)
            for value in _text_values(feature_parameters.get("excludeFamilies"))
        ]
        protected = [
            str(value)
            for value in _text_values(feature_parameters.get("protectFamilies"))
        ]
        if excluded:
            choices.append(
                {
                    "label": f"Exclude {_format_text_list([_feature_family_label(value) for value in excluded])}",
                    "status": "Excluded from map",
                    "state": "rejected",
                    "metrics": [
                        "Still available for marker testing",
                    ],
                    "reason": (
                        "Excluded only from map-building features to reduce "
                        "unwanted signal."
                    ),
                }
            )
        if protected:
            choices.append(
                {
                    "label": f"Protect {_format_text_list([_feature_family_label(value) for value in protected])}",
                    "status": "Preserved",
                    "state": "selected",
                    "metrics": ["Remained eligible for map construction"],
                    "reason": (
                        "Protected so biological structure was not removed as "
                        "technical noise."
                    ),
                }
            )
        if inventory:
            default_applied = inventory.get("appliedToSelectedRepresentation") is True
            match_count = inventory.get("matchCount")
            total_features = inventory.get("totalFeatures")
            choices.append(
                {
                    "label": "Exact Scarf default HVG blacklist",
                    "status": (
                        "Applied to selected representation"
                        if default_applied
                        else "Evaluated as reference"
                    ),
                    "state": "selected" if default_applied else "reviewed",
                    "metrics": (
                        [
                            f"Matched {int(match_count):,} of "
                            f"{int(total_features):,} genes"
                        ]
                        if isinstance(match_count, int)
                        and isinstance(total_features, int)
                        else []
                    ),
                    "reason": (
                        "Applied as the complete selected representation blacklist."
                        if default_applied
                        else (
                            "Not applied wholesale; the final policy used only the "
                            "families supported by the decision evidence."
                        )
                    ),
                }
            )
        outcome_parts.append(
            f"{_assay_label(assay)} log normalization"
            if log_transform
            else f"{_assay_label(assay)} normalization"
        )
        if excluded:
            outcome_parts.append(
                f"excluded {_format_text_list([_feature_family_label(value) for value in excluded])} from map construction"
            )
        if inventory and not inventory.get("appliedToSelectedRepresentation"):
            outcome_parts.append(
                "complete Scarf default blacklist not applied wholesale"
            )
        for family_name in dict.fromkeys([*excluded, *protected]):
            family = _mapping(families.get((assay, family_name)))
            count = family.get("count")
            skipped = family.get("skipped")
            action = (
                "Excluded from map construction"
                if family_name in excluded
                else "Protected and retained"
            )
            measurements.append(
                (
                    _feature_family_label(family_name).capitalize(),
                    (
                        "Not counted"
                        if skipped
                        else (
                            f"{int(count):,} identified features"
                            if isinstance(count, int)
                            else "Feature count unavailable"
                        )
                    ),
                    (
                        f"{action}. Inspection was skipped because "
                        f"{_label(skipped).lower()}."
                        if skipped
                        else f"{action}."
                    ),
                )
            )
        min_cells = feature_parameters.get("minCells")
        if isinstance(min_cells, int):
            measurements.append(
                (
                    f"{_assay_label(assay)} detection requirement",
                    f"Present in at least {min_cells:,} cells",
                    "Applied before variable-gene ranking.",
                )
            )
    inventory_markup = "".join(
        _render_default_inventory_summary(
            inventory,
            heading=f"{_assay_label(inventory.get('assay'))} default blacklist audit",
        )
        for inventory in inventories
    )
    return _render_evidence_panel(
        title="Normalization and feature policy",
        outcome="; ".join(outcome_parts),
        introduction=(
            "Normalization and feature-family rules were fixed before tuning. "
            "Representation exclusions changed the map-building features, not the "
            "genes available for marker analysis."
        ),
        body=_render_evidence_choices(choices) + inventory_markup,
        measurements=_render_evidence_measurements(measurements),
    )


def _render_batch_evidence(
    experimental: Mapping[str, Any],
    parameter: Mapping[str, Any],
    final: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> str:
    decision = _mapping(experimental.get("decision"))
    batch_plan = _mapping(decision.get("batchCorrection"))
    safety = _mappings(experimental.get("batchSafety"))
    if not batch_plan and not safety:
        return ""
    native_analyses = _mappings(final.get("nativeAnalyses"))
    if final.get("graphMethod") == "native" and final.get("primaryAssay"):
        native_analyses = [
            item
            for item in native_analyses
            if item.get("assay") == final.get("primaryAssay")
        ]
    adjusted = any(_present(item.get("batchCorrection")) for item in native_analyses)
    native_candidate, harmony_candidate = _harmony_candidate_pair(parameter, final)
    harmony_executed = _harmony_completed(native_candidate) and _harmony_completed(
        harmony_candidate
    )
    degraded = _degraded_protected_columns(native_candidate, harmony_candidate)
    coefficients = list(
        dict.fromkeys(
            _public_field_label(item.get("coefficient"))
            for item in safety
            if _public_field_label(item.get("coefficient"))
        )
    )
    unsafe = any(item.get("status") == "unsafe" for item in safety)
    correction_outcome = _active_decision(decisions, "correctionOutcome")
    correction_license = _active_decision(decisions, "correctionLicense")
    diagnostic_only = str(correction_license.get("selectedOptionId") or "").endswith(
        "unsafeConfounded"
    )
    native_metrics = _mapping(native_candidate.get("metrics"))
    harmony_metrics = _mapping(harmony_candidate.get("metrics"))
    native_batch = _mapping(native_metrics.get("batchMixing"))
    harmony_batch = _mapping(harmony_metrics.get("batchMixing"))
    harmony_choice_metrics: list[str] = []
    if harmony_candidate:
        parameters = _mapping(harmony_candidate.get("parameters"))
        harmony_choice_metrics.append(
            "Matched parameters: "
            f"{_scalar(parameters.get('dimensions'))} dimensions, "
            f"{_scalar(parameters.get('neighborsK'))} neighbors, "
            f"resolution {_scalar(parameters.get('leidenResolution'))}"
        )
    if harmony_executed:
        harmony_choice_metrics.insert(0, "Run status: completed")
        for column in dict.fromkeys([*native_batch, *harmony_batch]):
            harmony_choice_metrics.append(
                f"{_public_field_label(column).capitalize()} mixing: "
                f"{_score_transition(native_batch.get(column), harmony_batch.get(column))}"
            )
        if degraded:
            harmony_choice_metrics.append(
                "Protected evidence degraded: " + _format_text_list(degraded)
            )
    if coefficients:
        harmony_choice_metrics.append(
            f"Design-confounded comparisons: {_format_text_list(coefficients)}"
        )
    if diagnostic_only:
        harmony_choice_metrics.append("Selection license: diagnostic only")
    recorded_rationale = str(correction_outcome.get("rationale") or "").strip()
    if harmony_executed and degraded:
        harmony_reason = (
            "Rejected because protected evidence degraded for "
            f"{_format_text_list(degraded)}"
            + ("; the design license was diagnostic only." if diagnostic_only else ".")
        )
    elif harmony_executed:
        harmony_reason = "Executed as a matched diagnostic but not selected."
    elif unsafe:
        harmony_reason = (
            "Not run because library effects could not be separated safely from "
            "the protected study comparisons."
        )
    else:
        harmony_reason = "No completed matched Harmony diagnostic was recorded."
    choices = [
        {
            "label": "Use the unadjusted representation",
            "status": "Selected" if not adjusted else "Not selected",
            "state": "selected" if not adjusted else "rejected",
            "metrics": ["Protected biological comparisons remain intact"],
            "reason": (
                "Selected after the matched diagnostic retained more protected "
                "biological structure."
                if harmony_executed and not adjusted
                else "Selected because no safe, measurable correction was available."
                if not adjusted
                else "Not selected after the adjusted result showed a safe benefit."
            ),
        },
        {
            "label": "Apply Harmony correction",
            "status": (
                "Run and selected"
                if adjusted
                else (
                    "Run diagnostically; rejected"
                    if harmony_executed
                    else ("Not run" if unsafe else "Not selected")
                )
            ),
            "state": "rejected" if not adjusted else "selected",
            "metrics": harmony_choice_metrics,
            "reason": harmony_reason,
        },
    ]
    measurements: list[tuple[str, str, str]] = []
    for item in safety:
        estimability = _mapping(item.get("estimability"))
        coefficient = _public_field_label(item.get("coefficient"))
        estimable = estimability.get("coefficientEstimable") is True
        rows = estimability.get("rowsUsed")
        rank = estimability.get("rankTechnical")
        residual = estimability.get("residualDf")
        remaining = estimability.get("estimableDf")
        measurements.append(
            (
                f"Harmony safety for {coefficient}",
                "Estimable" if estimable else "Not estimable",
                "; ".join(
                    value
                    for value in (
                        f"Study units: {int(rows):,}" if isinstance(rows, int) else "",
                        f"Technical rank: {int(rank):,}"
                        if isinstance(rank, int)
                        else "",
                        f"Residual degrees of freedom: {int(residual):,}"
                        if isinstance(residual, int)
                        else "",
                        f"Remaining comparison capacity: {int(remaining):,}"
                        if isinstance(remaining, int)
                        else "",
                    )
                    if value
                ),
            )
        )
    measurements.extend(
        _covariate_pair_measurements(_mapping(experimental.get("characterization")))
    )
    outcome = (
        "Harmony completed and was selected"
        if adjusted
        else (
            "Diagnostic Harmony completed; rejected and native representation retained"
            if harmony_executed
            else (
                "Harmony not applied; protected comparisons were not independently estimable"
                if unsafe
                else "No batch correction was selected"
            )
        )
    )
    comparison_markup = (
        '<div class="subsection"><h3>Matched native versus Harmony metrics</h3>'
        + _table(
            _harmony_metric_rows(native_candidate, harmony_candidate),
            columns=(
                "category",
                "metric",
                "native",
                "Harmony",
                "change",
                "interpretation",
            ),
            empty="No matched Harmony measurements were recorded.",
        )
        + "</div>"
        if harmony_executed
        else ""
    )
    rationale_markup = (
        '<div class="callout subsection"><h3>Recorded correction decision</h3>'
        f"<p>{html.escape(recorded_rationale)}</p></div>"
        if recorded_rationale
        else ""
    )
    return _render_evidence_panel(
        title="Harmony and batch correction",
        outcome=outcome,
        introduction=(
            "Selection required measured technical improvement without material "
            "loss of protected tissue, T2D, donor, or sex structure. A diagnostic "
            "run could still be completed when the design was not licensed for "
            "corrected-result selection."
        ),
        body=(_render_evidence_choices(choices) + comparison_markup + rationale_markup),
        measurements=_render_evidence_measurements(measurements),
    )


def _hvg_ranking_label(value: Any) -> str:
    return {
        "global": "Global variability ranking",
        "batchAware": "Group-aware variability ranking",
    }.get(str(value or ""), "Variable-gene ranking")


def _hvg_tree_stages(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    rankings = _mappings(evidence.get("rankings"))
    candidates = _mappings(evidence.get("candidateMetrics"))
    default_counts = [
        int(value)
        for value in evidence.get("scarfDefaultReferenceCounts", [])
        if isinstance(value, int)
    ]
    selected_mode = evidence.get("selectedRankingMode")
    selected_count = evidence.get("selectedFeatureCount")
    stages: list[dict[str, Any]] = []
    if rankings:
        ranking_branches: list[dict[str, Any]] = []
        for ranking in rankings:
            selected = ranking.get("rankingMode") == selected_mode
            ranking_branches.append(
                _tree_branch(
                    label=_hvg_ranking_label(ranking.get("rankingMode")),
                    status="Selected" if selected else "Not selected",
                    state="selected" if selected else "alternative",
                    metrics=[
                        "Mean library coverage: "
                        f"{_analysis_percent(ranking.get('meanTechnicalGroupCoverage'))}",
                        "Recurring in at least two libraries: "
                        f"{_analysis_percent(ranking.get('recurrentInTwoGroupsFraction'))}",
                    ],
                    reason=(
                        "Selected after the combined recurrence, default-overlap, "
                        "technical-association, and downstream-stability comparison."
                        if selected
                        else (
                            "Not selected after the combined upstream and downstream "
                            "comparison."
                        )
                    ),
                )
            )
        if default_counts:
            ranking_branches.append(
                _tree_branch(
                    label="Exact Scarf-default blacklist reference",
                    status="Reference evaluated",
                    state="reviewed",
                    metrics=[
                        "Executed set sizes: "
                        + ", ".join(f"{value:,}" for value in default_counts)
                    ],
                    reason=(
                        "Used as an exact comparison reference; it was not a "
                        "selectable ranking mode."
                    ),
                )
            )
        stages.append(
            {
                "question": "How should highly variable genes be ranked?",
                "description": (
                    "The workflow compared a global variability ranking with a "
                    "ranking that emphasized recurrence across libraries."
                ),
                "branches": ranking_branches,
            }
        )
    if candidates:
        count_branches: list[dict[str, Any]] = []
        for candidate in candidates:
            count = candidate.get("featureCount")
            if not isinstance(count, int):
                continue
            selected = count == selected_count
            count_branches.append(
                _tree_branch(
                    label=f"{count:,} variable genes",
                    status="Selected" if selected else "Not selected",
                    state="selected" if selected else "alternative",
                    metrics=[
                        "Corrected variance captured: "
                        f"{_analysis_percent(candidate.get('varianceFraction'))}",
                        "Recurring across most libraries: "
                        f"{_analysis_percent(candidate.get('recurrentFraction'))}",
                    ],
                    reason=(
                        "Selected as the supported balance of captured variation, "
                        "reproducibility, and downstream stability."
                        if selected
                        else "Not selected after comparison with the supported set size."
                    ),
                )
            )
        if count_branches:
            stages.append(
                {
                    "question": "How many highly variable genes should be used?",
                    "description": (
                        "Registered focused, standard, and broad feature-set sizes "
                        "were all executed and compared."
                    ),
                    "branches": count_branches,
                }
            )
    return stages


def _render_hvg_evidence(evidence: Mapping[str, Any]) -> str:
    rankings = _mappings(evidence.get("rankings"))
    candidates = _mappings(evidence.get("candidateMetrics"))
    default_counts = [
        int(value)
        for value in evidence.get("scarfDefaultReferenceCounts", [])
        if isinstance(value, int)
    ]
    if not rankings and not candidates:
        return ""
    selected_mode = evidence.get("selectedRankingMode")
    selected_count = evidence.get("selectedFeatureCount")
    ranking_choices: list[dict[str, Any]] = []
    for ranking in rankings:
        selected = ranking.get("rankingMode") == selected_mode
        ranking_choices.append(
            {
                "label": _hvg_ranking_label(ranking.get("rankingMode")),
                "status": "Selected" if selected else "Not selected",
                "state": "selected" if selected else "rejected",
                "metrics": [
                    "Mean coverage across libraries: "
                    f"{_analysis_percent(ranking.get('meanTechnicalGroupCoverage'))}",
                    "Genes recurring in at least two libraries: "
                    f"{_analysis_percent(ranking.get('recurrentInTwoGroupsFraction'))}",
                ],
                "reason": (
                    "Selected after combining recurrence, exact Scarf-default "
                    "overlap, technical association, and downstream stability."
                    if selected
                    else (
                        "Not selected after the combined upstream and downstream "
                        "comparison."
                    )
                ),
            }
        )
    if default_counts:
        ranking_choices.append(
            {
                "label": "Exact Scarf-default blacklist reference",
                "status": "Reference evaluated",
                "state": "reviewed",
                "metrics": [
                    "Executed set sizes: "
                    + ", ".join(f"{value:,}" for value in default_counts)
                ],
                "reason": (
                    "Used as a fixed comparison reference, not as a selectable "
                    "ranking mode."
                ),
            }
        )
    candidate_choices: list[dict[str, Any]] = []
    for candidate in candidates:
        count = candidate.get("featureCount")
        if not isinstance(count, int):
            continue
        selected = count == selected_count
        candidate_choices.append(
            {
                "label": f"{count:,} variable genes",
                "status": "Selected" if selected else "Not selected",
                "state": "selected" if selected else "rejected",
                "metrics": [
                    "Corrected variance captured: "
                    f"{_analysis_percent(candidate.get('varianceFraction'))}",
                    "Genes recurring across most libraries: "
                    f"{_analysis_percent(candidate.get('recurrentFraction'))}",
                ],
                "reason": (
                    "Selected as the best balance of captured variation and "
                    "cross-library reproducibility."
                    if selected
                    else (
                        "Captured less variation than the selected set."
                        if count < int(selected_count or 0)
                        else "Added genes with substantially lower reproducibility."
                    )
                ),
            }
        )
    measurements = [
        (
            "Eligible genes",
            f"{int(evidence['eligibleFeatureCount']):,}",
            "Genes available after detection and feature-family rules.",
        )
        if isinstance(evidence.get("eligibleFeatureCount"), int)
        else None,
        (
            "Libraries represented",
            f"{int(evidence['validTechnicalGroups']):,}",
            "Registered technical groups used to assess recurrence.",
        )
        if isinstance(evidence.get("validTechnicalGroups"), int)
        else None,
        (
            "Minimum detection",
            f"{int(evidence['minimumDetectedCells']):,} cells",
            "Required before a gene could enter the ranking.",
        )
        if isinstance(evidence.get("minimumDetectedCells"), int)
        else None,
        (
            "Excluded libraries",
            f"{int(evidence['excludedTechnicalGroupCount']):,}",
            "Libraries omitted from the group-aware ranking.",
        )
        if isinstance(evidence.get("excludedTechnicalGroupCount"), int)
        else None,
        (
            "HVG branches executed",
            f"{int(evidence['executedBranchCount']):,}",
            "Global, group-aware, and exact Scarf-default reference branches.",
        )
        if isinstance(evidence.get("executedBranchCount"), int)
        else None,
    ]
    body = (
        '<div class="subsection"><h3>Ranking method</h3>'
        f"{_render_evidence_choices(ranking_choices)}</div>"
        '<div class="subsection"><h3>Number of variable genes</h3>'
        f"{_render_evidence_choices(candidate_choices)}</div>"
    )
    outcome = (
        f"{_hvg_ranking_label(selected_mode)}; {int(selected_count):,} genes selected"
        if isinstance(selected_count, int)
        else f"{_hvg_ranking_label(selected_mode)} selected"
    )
    return _render_evidence_panel(
        title="Highly variable genes (HVGs)",
        outcome=outcome,
        introduction=(
            "The workflow first compared how genes were ranked, then compared three "
            "registered set sizes. Selection combined recurrence, exact "
            "Scarf-default overlap, technical association, and downstream "
            "stability rather than using one metric alone."
        ),
        body=body,
        measurements=_render_evidence_measurements(
            [item for item in measurements if item is not None]
        ),
        expanded=True,
    )


def _render_analysis_evidence(payload: Mapping[str, Any]) -> str:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    plan = _mapping(workflow_result.get("preprocessingPlan"))
    final = _mapping(workflow_result.get("finalAnalysis"))
    enrichment = _latest(reports, "data_enrichment")
    experimental = _latest(reports, "experimental_context")
    parameter = _latest(reports, "parameter_tuning")
    decisions = _mapping(payload.get("activeDecisions"))
    inventories = _mappings(payload.get("defaultFeatureInventories"))
    panels = [
        _render_filtering_evidence(experimental, plan),
        _render_covariate_evidence(experimental),
        _render_normalization_evidence(enrichment, plan, inventories),
        _render_batch_evidence(experimental, parameter, final, decisions),
        _render_hvg_evidence(_mapping(payload.get("hvgEvidence"))),
    ]
    panels = [panel for panel in panels if panel]
    if not panels:
        return ""
    return (
        '<section class="section" id="decision-evidence">'
        "<h2>Evidence behind the decisions</h2>"
        "<p>Open a section to compare the selected and rejected choices. "
        "Each section keeps denser thresholds and scores under Measurements.</p>"
        f'<div class="evidence-accordion">{"".join(panels)}</div></section>'
    )


def _narrative_items(value: Any, keys: Sequence[str]) -> list[str]:
    if not _is_sequence(value):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            text = next(
                (
                    _brief_text(item.get(key))
                    for key in keys
                    if _brief_text(item.get(key))
                ),
                "",
            )
        else:
            text = _brief_text(item)
        if text:
            items.append(text)
    return items


def _render_plain_list(items: Sequence[str]) -> str:
    if not items:
        return ""
    return '<ul class="plain-list">{}</ul>'.format(
        "".join(f"<li>{html.escape(item)}</li>" for item in items)
    )


def _render_analysis_biology(biology: Mapping[str, Any]) -> str:
    interpretations = _mappings(biology.get("clusterInterpretations"))
    observations = _narrative_items(
        biology.get("treatmentObservations"),
        ("observation",),
    )
    follow_ups = _narrative_items(
        biology.get("followUps"),
        ("question", "rationale"),
    )
    if not interpretations and not observations and not follow_ups:
        return ""

    cards = "".join(
        '<article class="interpretation-card">'
        f'<p class="summary-label">Cell group {html.escape(str(item.get("clusterId") or "unresolved"))}</p>'
        f"<h3>{html.escape(str(item.get('proposedIdentity') or 'Unresolved'))}</h3>"
        + (
            f"<p>{html.escape(_brief_text(item.get('rationale')))}</p>"
            if _brief_text(item.get("rationale"))
            else ""
        )
        + (
            '<span class="chip">Tentative interpretation</span>'
            if item.get("identityIsHypothesis") is True
            else ""
        )
        + "</article>"
        for item in interpretations
    )
    interpretation_markup = (
        f'<div class="interpretation-grid">{cards}</div>' if cards else ""
    )
    observation_markup = (
        '<div class="subsection"><h3>Observed group differences</h3>'
        f"{_render_plain_list(observations)}</div>"
        if observations
        else ""
    )
    follow_up_markup = (
        '<div class="subsection"><h3>Recommended follow-up</h3>'
        f"{_render_plain_list(follow_ups)}</div>"
        if follow_ups
        else ""
    )
    return f"""
  <section class="section">
    <h2>Biological interpretation</h2>
    {interpretation_markup}
    {observation_markup}
    {follow_up_markup}
  </section>
"""


def _render_column_list(items: Sequence[str]) -> str:
    if not items:
        return '<p class="empty">No matched feature names were recorded.</p>'
    return '<ul class="plain-list column-list">{}</ul>'.format(
        "".join(f"<li>{html.escape(item)}</li>" for item in items)
    )


def _render_qc_technical_audit(
    experimental: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    profiles = _mappings(experimental.get("qcProfiles"))
    if not profiles:
        return ""
    decision = _mapping(experimental.get("decision"))
    cell_qc = _mapping(plan.get("cellQc"))
    if not cell_qc:
        cell_qc = _mapping(decision.get("cellQc"))
    if not cell_qc:
        cell_qc = _mapping(experimental.get("cellQc"))
    selected = _selected_qc_profile(experimental, cell_qc)
    selected_id = selected.get("profileId")
    selected_name = selected.get("registeredProfile")
    profile_rows = _qc_profile_rows(profiles)
    for row, profile in zip(profile_rows, profiles, strict=True):
        row["_selected"] = bool(
            (selected_id and profile.get("profileId") == selected_id)
            or (
                not selected_id
                and selected_name
                and profile.get("registeredProfile") == selected_name
            )
        )
    selected_metrics = _qc_metric_rows(selected)
    return f"""
  <section class="section" id="qc-audit">
    <h2>Cell QC audit</h2>
    <p>Selected and alternative filtering profiles, diagnostic flags, and every persisted cutoff are shown below. A cutoff in a retain-with-flags profile is diagnostic and did not remove cells.</p>
    <div class="subsection"><h3>Profile comparison</h3>{_table(profile_rows, columns=("profile", "scope", "active cells", "retained cells", "removed cells", "flags", "failed libraries"))}</div>
    <div class="subsection"><h3>Selected-profile metric summary</h3>{_table(selected_metrics, columns=("metric", "scope", "median", "diagnostic reference", "lower cutoff", "upper cutoff", "high flag", "flagged cells"))}</div>
    <details><summary>All global and per-library cutoffs</summary>{_table(_qc_bound_rows(profiles), columns=("profile", "group", "metric", "median", "lower removal", "upper removal", "upper flag"), empty="No persisted QC cutoffs were recorded.")}</details>
  </section>
"""


def _render_feature_technical_audit(
    plan: Mapping[str, Any],
    inventories: Sequence[Mapping[str, Any]],
) -> str:
    if not inventories:
        return ""
    policy_rows: list[dict[str, Any]] = []
    for assay_plan in _mappings(plan.get("assays")):
        parameters = _mapping(assay_plan.get("featureParameters"))
        if not parameters:
            continue
        policy_rows.append(
            {
                "assay": assay_plan.get("assay"),
                "selected features": parameters.get("topN"),
                "minimum detected cells": parameters.get("minCells"),
                "excluded families": _text_values(parameters.get("excludeFamilies")),
                "protected families": _text_values(parameters.get("protectFamilies")),
                "complete default blacklist applied": (
                    parameters.get("useScarfDefaultBlacklist") is True
                ),
            }
        )
    inventory_markup: list[str] = []
    for inventory in inventories:
        assay = _assay_label(inventory.get("assay")) or "Assay"
        match_count = inventory.get("matchCount")
        names = _text_values(inventory.get("matchedFeatures"))
        inventory_markup.append(
            _render_default_inventory_summary(
                inventory,
                heading=f"{assay} exact Scarf-default blacklist",
            )
            + "<details><summary>All "
            + (
                f"{int(match_count):,}"
                if isinstance(match_count, int)
                else f"{len(names):,}"
            )
            + " matched feature names</summary>"
            + _render_column_list(names)
            + "</details>"
        )
    return f"""
  <section class="section" id="feature-audit">
    <h2>Normalization and feature-selection audit</h2>
    <p>The selected representation policy is separate from the exact Scarf-default blacklist reference. Genes excluded from map construction remained available to marker testing.</p>
    {_table(policy_rows, columns=("assay", "selected features", "minimum detected cells", "excluded families", "protected families", "complete default blacklist applied"))}
    {"".join(inventory_markup)}
  </section>
"""


def _render_harmony_technical_audit(
    experimental: Mapping[str, Any],
    parameter: Mapping[str, Any],
    final: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> str:
    native, harmony = _harmony_candidate_pair(parameter, final)
    if not native and not harmony:
        return ""
    outcome = _active_decision(decisions, "correctionOutcome")
    license_record = _active_decision(decisions, "correctionLicense")
    rationale = str(outcome.get("rationale") or "").strip()
    license_option = str(license_record.get("selectedOptionId") or "not recorded")
    license_label = _label(license_option.rpartition(":")[2])
    run_status = (
        "completed" if _harmony_completed(harmony) else _scalar(harmony.get("status"))
    )
    rationale_markup = (
        '<div class="callout subsection"><h3>Recorded rejection rationale</h3>'
        f"<p>{html.escape(rationale)}</p></div>"
        if rationale
        else ""
    )
    candidate_rows = [
        {
            "candidate": "Native",
            "status": native.get("status"),
            "eligible": native.get("eligible"),
            **_mapping(native.get("parameters")),
        },
        {
            "candidate": "Harmony",
            "status": harmony.get("status"),
            "eligible": harmony.get("eligible"),
            **_mapping(harmony.get("parameters")),
        },
    ]
    safety_rows: list[dict[str, Any]] = []
    for item in _mappings(experimental.get("batchSafety")):
        estimability = _mapping(item.get("estimability"))
        safety_rows.append(
            {
                "comparison": _public_field_label(item.get("coefficient")),
                "status": item.get("status"),
                "study units": estimability.get("rowsUsed"),
                "technical rank": estimability.get("rankTechnical"),
                "residual degrees of freedom": estimability.get("residualDf"),
                "remaining capacity": estimability.get("estimableDf"),
            }
        )
    return f"""
  <section class="section" id="harmony-audit">
    <h2>Harmony diagnostic audit</h2>
    <p>Run status: <strong>{html.escape(run_status)}</strong>. Selection license: <strong>{html.escape(license_label)}</strong>.</p>
    <div class="subsection"><h3>Matched candidates</h3>{_table(candidate_rows, columns=("candidate", "status", "eligible", "dimensions", "neighborsK", "leidenResolution", "useHarmony"))}</div>
    <div class="subsection"><h3>Native versus Harmony measurements</h3>{_table(_harmony_metric_rows(native, harmony), columns=("category", "metric", "native", "Harmony", "change", "interpretation"))}</div>
    <div class="subsection"><h3>Design safety</h3>{_table(safety_rows, columns=("comparison", "status", "study units", "technical rank", "residual degrees of freedom", "remaining capacity"), empty="No design-safety rows were recorded.")}</div>
    {rationale_markup}
  </section>
"""


def _analysis_limitations(payload: Mapping[str, Any]) -> list[str]:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    final = _mapping(workflow_result.get("finalAnalysis"))
    parameter = _latest(reports, "parameter_tuning")
    biology = _latest(reports, "biological_interpretation")
    interpretations = _mappings(biology.get("clusterInterpretations"))
    limitations: list[str] = []
    if not interpretations:
        limitations.append(
            "No biological cell-type interpretation was generated, so the cell "
            "groups should not be treated as named cell types."
        )
    elif any(item.get("identityIsHypothesis") is True for item in interpretations):
        limitations.append(
            "Cell-group identities are hypotheses based on observed marker patterns "
            "and need independent validation."
        )
    if _mappings(biology.get("treatmentObservations")):
        limitations.append(
            "Reported group differences are descriptive and do not establish cause "
            "and effect."
        )
    if parameter.get("totalCandidates"):
        limitations.append(
            "The final result was selected only from the analysis settings that "
            "were explicitly evaluated."
        )
    if _present(final.get("limitations")) or _present(parameter.get("limitations")):
        limitations.append(
            "Additional technical limitations are recorded in the technical report."
        )
    if _present(payload.get("plotNotes")):
        limitations.append(
            "Some optional visualizations were unavailable; the technical report "
            "records the reason."
        )
    return limitations


def _render_index_document(payload: Mapping[str, Any]) -> str:
    workflow_result = _mapping(payload.get("workflowResult"))
    plan = _mapping(workflow_result.get("preprocessingPlan"))
    cluster_counts = _mapping(payload.get("clusterCounts"))
    total_cells = sum(int(value) for value in cluster_counts.values())
    objective, _organisms, _tissues = _study_overview(payload)
    assays = _report_assays(plan)
    metrics = _render_metrics(
        (
            ("Cells analyzed", total_cells or None),
            ("Cell groups", len(cluster_counts) or None),
            ("Data analyzed", _format_text_list(assays) or None),
        )
    )
    body = f"""  <p class="eyebrow">Completed analysis</p>
  <h1>Choose the level of detail.</h1>
  <p class="lead">{html.escape(objective)}</p>
  {metrics}

  <section class="section">
    <div class="report-choice-grid">
      <a class="report-choice" href="analysis.html">
        <p class="eyebrow">Recommended</p>
        <h2>Analysis summary</h2>
        <p>See the main findings, selected decisions, visual results, and limitations in plain language.</p>
        <span class="report-choice-action">Open analysis summary →</span>
      </a>
      <a class="report-choice" href="technical.html">
        <p class="eyebrow">Methods and provenance</p>
        <h2>Technical details</h2>
        <p>Review complete parameters, comparisons, execution records, artifact references, and structured report data.</p>
        <span class="report-choice-action">Open technical details →</span>
      </a>
    </div>
  </section>

  <aside class="product-callout">
    <p><strong>ScarfWeb</strong><br>Distributed, secure infrastructure for intuitive secondary analysis, browser-native.</p>
    <a class="pill" href="https://www.nygen.io/products/scarfweb" target="_blank" rel="noopener noreferrer">Explore ScarfWeb</a>
  </aside>
"""
    return _render_report_shell(
        title="Scarf analysis report",
        active_page="index",
        body=body,
    )


def _render_analysis_document(payload: Mapping[str, Any]) -> str:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    plan = _mapping(workflow_result.get("preprocessingPlan"))
    biology = _latest(reports, "biological_interpretation")
    cluster_counts = {
        str(key): int(value)
        for key, value in _mapping(payload.get("clusterCounts")).items()
    }
    plots = {
        str(key): str(value)
        for key, value in _mapping(payload.get("plotFiles")).items()
    }
    objective, organisms, tissues = _study_overview(payload)
    assays = _report_assays(plan)
    total_cells = sum(cluster_counts.values())
    metrics = _render_metrics(
        (
            ("Cells analyzed", total_cells or None),
            ("Cell groups", len(cluster_counts) or None),
            ("Data analyzed", _format_text_list(assays) or None),
        )
    )
    source = _biological_source(organisms, tissues) or "Not specified"
    source = source[:1].upper() + source[1:]
    tree_stages = _analysis_tree_stages(payload)
    selection_evidence = _render_selection_evidence(payload)
    decision_evidence = _render_analysis_evidence(payload)
    analysis_plots = _render_plots(
        plots,
        (),
        order=("umapClusters", "clusterComposition", "markerHeatmap"),
        titles={
            "umapClusters": (
                "Final cell map",
                "Each point is a cell, colored by its selected cell group.",
            ),
            "clusterComposition": (
                "Relative group sizes",
                "The relative size of each selected cell group.",
            ),
            "markerHeatmap": (
                "Marker patterns",
                "Features that help distinguish the selected cell groups.",
            ),
        },
        show_provenance=False,
        show_notes=False,
        empty_message=(
            "Visual results are unavailable for this report. Technical details "
            "record the reason."
        ),
    )
    diagnostic_plot_order = (
        "qcDistributionsBeforeFiltering",
        "qcDistributions",
        *(name for name in plots if name.startswith("qcDistributionDerived")),
        "hvgGlobal",
        "hvgBatchAware",
    )
    diagnostic_plots = (
        _render_plots(
            plots,
            (),
            order=diagnostic_plot_order,
            show_provenance=False,
            show_notes=False,
        )
        if any(name in plots for name in diagnostic_plot_order)
        else ""
    )
    diagnostic_section = (
        """
  <section class="section">
    <h2>Quality control and variable-gene diagnostics</h2>
    <p>QC panels show the selected cutoff annotations. HVG panels highlight the genes retained by each executed ranking at the selected feature count.</p>
    {plots}
  </section>
""".format(plots=diagnostic_plots)
        if diagnostic_plots
        else ""
    )
    limitations = _analysis_limitations(payload)
    biology_markup = _render_analysis_biology(biology)
    body = f"""  <p class="eyebrow">Analysis summary</p>
  <h1>The analysis, at a glance.</h1>
  <p class="lead">{html.escape(objective)}</p>
  {metrics}

  <section class="section">
    <h2>What was analyzed</h2>
    <div class="summary-grid">
      <article class="summary-card">
        <p class="summary-label">Biological source</p>
        <h3>{html.escape(source)}</h3>
      </article>
      <article class="summary-card">
        <p class="summary-label">Final result</p>
        <h3>{total_cells:,} cells organized into {len(cluster_counts):,} groups</h3>
      </article>
    </div>
  </section>

  <section class="section">
    <h2>Analysis decision tree</h2>
    <p>Each decision shows the selected branch, the alternatives considered, their measured values, and why the selected path continued.</p>
    {_render_decision_tree(tree_stages)}
  </section>

  {decision_evidence}

  {diagnostic_section}

  <section class="section">
    <h2>Why the final result was selected</h2>
    <p>These are the main measurements supporting the final cell map. Values closer to 1 indicate stronger agreement for the stability and coherence measures.</p>
    {selection_evidence or '<p class="empty">No final selection measurements were available.</p>'}
  </section>

  <section class="section">
    <h2>Visual results</h2>
    {analysis_plots}
  </section>

  {biology_markup}

  <section class="section">
    <h2>Limitations</h2>
    {_render_plain_list(limitations) if limitations else "<p>No additional user-facing limitations were recorded.</p>"}
  </section>

  <aside class="product-callout">
    <p><strong>Need the exact methods?</strong><br>The technical report contains parameters, comparisons, provenance, and the complete structured record.</p>
    <a class="pill" href="technical.html">Open technical details</a>
  </aside>
"""
    return _render_report_shell(
        title="Scarf analysis summary",
        active_page="analysis",
        body=body,
    )


def _render_technical_document(payload: Mapping[str, Any]) -> str:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    request = _mapping(payload.get("request"))
    final = _mapping(workflow_result.get("finalAnalysis"))
    plan = _mapping(workflow_result.get("preprocessingPlan"))
    enrichment = _latest(reports, "data_enrichment")
    experimental = _latest(reports, "experimental_context")
    parameter = _latest(reports, "parameter_tuning")
    biology = _latest(reports, "biological_interpretation")
    decisions = _mapping(payload.get("activeDecisions"))
    inventories = _mappings(payload.get("defaultFeatureInventories"))
    cluster_counts = {
        str(key): int(value)
        for key, value in _mapping(payload.get("clusterCounts")).items()
    }
    top_markers = _mappings(payload.get("topMarkers"))
    plots = {
        str(key): str(value)
        for key, value in _mapping(payload.get("plotFiles")).items()
    }
    plot_notes = [str(item) for item in payload.get("plotNotes", [])]
    attempts = _mappings(payload.get("stageAttempts"))
    resumes = _mappings(payload.get("workflowResumes"))
    workflow = _mapping(workflow_result.get("workflowRun"))
    workflow_id = str(workflow.get("workflowRunId") or "unavailable")
    total_cells = sum(cluster_counts.values())
    assay_plans = _mappings(plan.get("assays"))
    assays = [str(item.get("assay")) for item in assay_plans if item.get("assay")]
    doublet_evidence = _mapping(final.get("doubletEvidence"))
    marker_evidence = _mapping(final.get("markerEvidence"))
    metrics = [
        ("Final cells", total_cells or None),
        ("Final clusters", len(cluster_counts) or None),
        ("Assays", ", ".join(assays) or None),
        ("Candidates", parameter.get("totalCandidates")),
        ("Selected graph", final.get("graphMethod")),
        ("Marker assay", final.get("markerAssay")),
        ("Marker specificity", marker_evidence.get("specificityMedian")),
        ("Doublet capture coverage", doublet_evidence.get("captureCoverage")),
        (
            "Statistical test artifacts",
            len(_mappings(final.get("statisticalTests"))) or None,
        ),
    ]
    metric_markup = _render_metrics(metrics)
    interpretation = {
        "status": biology.get("status"),
        "clusterInterpretations": biology.get("clusterInterpretations"),
        "evidenceIds": biology.get("evidenceIds"),
        "stopReason": biology.get("stopReason"),
    }
    study = enrichment.get("studyContextSummary") or {
        "originalContext": request.get("studyContext")
    }
    enrichment_summary = {
        "status": enrichment.get("status"),
        "policies": enrichment.get("policies"),
        "inspections": enrichment.get("inspections"),
        "defaultFeatureEvidence": enrichment.get("defaultFeatureEvidence"),
        "evidenceIds": enrichment.get("evidenceIds"),
        "unresolvedQuestions": enrichment.get("unresolvedQuestions"),
    }
    experimental_summary = {
        "status": experimental.get("status"),
        "decision": experimental.get("decision"),
        "cellQc": experimental.get("cellQc"),
        "qcProfiles": experimental.get("qcProfiles"),
        "batchSafety": experimental.get("batchSafety"),
        "characterization": experimental.get("characterization"),
        "contrastPlans": experimental.get("contrastPlans"),
    }
    preprocessing_summary = {
        "primaryAssay": plan.get("primaryAssay"),
        "markerAssay": plan.get("markerAssay"),
        "pairedAssays": plan.get("pairedAssays"),
        "cellQc": plan.get("cellQc"),
        "assays": plan.get("assays"),
        "planChecksum": plan.get("planChecksum"),
    }
    limitations = {
        "Data Enrichment": enrichment.get("limitations"),
        "Experimental Context": experimental.get("notes"),
        "Parameter Tuning": parameter.get("limitations"),
        "Biological Interpretation": biology.get("limitations"),
        "Final analysis": final.get("limitations"),
        "Workflow": workflow_result.get("notes"),
        "Plots": plot_notes,
    }
    limitations = {key: value for key, value in limitations.items() if _present(value)}
    marker_columns = [
        key
        for key in (
            "group_id",
            "feature_name",
            "feature_id",
            "score",
            "frac_exp",
            "fold_change",
            "p_value",
        )
        if any(key in row for row in top_markers)
    ]
    provenance: list[dict[str, Any]] = [
        {"field": "Workflow run ID", "value": workflow_id},
        {"field": "Scarf version", "value": __version__},
        {"field": "Workspace", "value": workflow.get("workspace")},
        {"field": "Analysis store", "value": workflow.get("analysisStore")},
        {"field": "Dataset fingerprints", "value": workflow.get("datasetFingerprints")},
        {"field": "Generated at", "value": payload.get("generatedAt")},
        {"field": "Source path", "value": request.get("sourcePath")},
    ]
    raw_json = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, default=str
    )
    biology_nav = (
        '<a class="pill pill-outline" href="#biology">Biology</a>' if biology else ""
    )
    biology_markup = (
        f"""
  <section class="section" id="biology">
    <h2>Biological interpretation</h2>
    {_value(interpretation)}
    <div class="subsection"><h3>Treatment observations</h3>{_value(biology.get("treatmentObservations"))}</div>
    <div class="subsection"><h3>Follow-up recommendations</h3>{_value(biology.get("followUps"))}</div>
  </section>
"""
        if biology
        else ""
    )
    qc_audit = _render_qc_technical_audit(experimental, plan)
    feature_audit = _render_feature_technical_audit(plan, inventories)
    harmony_audit = _render_harmony_technical_audit(
        experimental,
        parameter,
        final,
        decisions,
    )
    qc_nav = '<a class="pill pill-outline" href="#qc-audit">QC</a>' if qc_audit else ""
    feature_nav = (
        '<a class="pill pill-outline" href="#feature-audit">Features</a>'
        if feature_audit
        else ""
    )
    harmony_nav = (
        '<a class="pill pill-outline" href="#harmony-audit">Harmony</a>'
        if harmony_audit
        else ""
    )
    title = f"Scarf agent report {workflow_id}"
    body = f"""  <p class="eyebrow">Technical report</p>
  <h1>Evidence from an automated analysis.</h1>
  <p class="lead">The workflow completed and its selected artifacts and decisions are summarized here.</p>
  <div class="pill-row">
    <span class="pill">Completed</span>
    <span class="pill pill-outline">{html.escape(_label(workflow_result.get("currentStage") or "completed"))}</span>
  </div>
  {metric_markup}
  <nav class="pill-row" aria-label="Report sections">
    <a class="pill pill-outline" href="#visuals">Visual results</a>
    {biology_nav}
    <a class="pill pill-outline" href="#context">Context</a>
    {qc_nav}
    {feature_nav}
    {harmony_nav}
    <a class="pill pill-outline" href="#tuning">Tuning</a>
    <a class="pill pill-outline" href="#workflow">Workflow</a>
  </nav>

  <aside class="product-callout">
    <p><strong>ScarfWeb</strong><br>Distributed, secure infrastructure for intuitive secondary analysis, browser-native.</p>
    <a class="pill" href="https://www.nygen.io/products/scarfweb" target="_blank" rel="noopener noreferrer">Explore ScarfWeb</a>
  </aside>

  <section class="section" id="visuals">
    <div class="section-heading"><h2>Visual results</h2><span class="pill pill-outline">Persisted artifacts</span></div>
    {_render_plots(plots, plot_notes)}
  </section>

  <section class="section">
    <h2>Final partition evidence</h2>
    <div class="subsection"><h3>Final cluster sizes</h3>{_render_clusters(cluster_counts)}</div>
    <div class="subsection"><h3>Top marker evidence</h3>{_table(top_markers, columns=marker_columns, empty="No marker table was available.")}</div>
    <div class="subsection"><h3>Marker-family summary</h3>{_value(marker_evidence)}</div>
    <div class="subsection"><h3>Advisory doublet summary</h3>{_value(doublet_evidence)}</div>
  </section>

  {biology_markup}

  {qc_audit}
  {feature_audit}
  {harmony_audit}

  <section class="section" id="context"><h2>Study context</h2>{_value(study)}</section>
  <section class="section"><h2>Data enrichment</h2>{_value(enrichment_summary)}</section>
  <section class="section"><h2>Experimental design</h2>{_value(experimental_summary)}</section>
  <section class="section"><h2>Preprocessing plan</h2>{_value(preprocessing_summary)}</section>

  <section class="section" id="tuning"><h2>Parameter tuning and graph selection</h2>{_render_parameter_tuning(parameter)}</section>
  <section class="section"><h2>Bounded analysis review and hypothesis tests</h2>{_value({"analysisEvidence": final.get("analysisEvidence"), "statisticalTests": final.get("statisticalTests")})}</section>
  <section class="section" id="workflow"><h2>Workflow execution</h2>{_render_timeline(attempts, resumes)}</section>
  <section class="section"><h2>Agent execution</h2>{_render_executions(reports)}</section>
  <section class="section"><h2>Limitations and workflow notes</h2>{_value(limitations) if limitations else '<p class="empty">No limitations were recorded.</p>'}</section>

  <section class="section">
    <h2>Technical provenance</h2>
    {_table(provenance)}
    <details><summary>Final immutable artifact references</summary>{_value(final)}</details>
    <details><summary>Structured report data</summary><pre>{html.escape(raw_json)}</pre></details>
  </section>
"""
    return _render_report_shell(
        title=title,
        active_page="technical",
        body=body,
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

    The report directory contains a landing page, an analysis summary, and
    technical details. The returned path points to the landing ``index.html``.
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
        ("analysis.html", _render_analysis_document(payload)),
        ("technical.html", _render_technical_document(payload)),
        ("index.html", _render_index_document(payload)),
    )
    destination = report_dir / "index.html"
    for filename, document in documents:
        written = _write_report_page(report_dir, filename, document)
        if filename == "index.html":
            destination = written
    logger.info(
        f"Generated HTML report for agent workflow {workflow_run_id}: {destination}"
    )
    return destination


__all__ = ["generate_agent_report"]
