"""Plot and bounded visual-artifact collection for agent reports."""

import html
import json
import os
import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...datastore.datastore import DataStore
from ...storage.refs import ArtifactRef
from ...storage.types import as_zarr_array
from ..orchestrator.models import AutomatedWorkflowResult, artifact_model_to_ref
from ..types import ArtifactReferenceModel
from .artifacts import _latest_hvg_diagnostic_artifacts
from .contracts import (
    _analysis_number_range,
    _label,
    _mapping,
    _mappings,
    _qc_resolved_bounds,
)

MAX_MARKER_DOTPLOT_FEATURES = 24


CLUSTER_COUNT_BLOCK_SIZE = 100_000


MAX_EMBEDDING_PLOT_CELLS = 250_000


MAX_DOTPLOT_CELLS = 75_000


MAX_CONNECTIVITY_PLOT_CELLS = 100_000


MAX_COMPOSITION_PLOT_CELLS = 1_000_000


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
            from ...plotting import FeatureRef, NormalizationSpec

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
            from ...plotting import highly_variable_features

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


def _hvg_ranking_label(value: Any) -> str:
    return {
        "global": "Global variability ranking",
        "batchAware": "Group-aware variability ranking",
    }.get(str(value or ""), "Variable-gene ranking")
