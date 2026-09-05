"""Decision-tree construction and rendering for agent reports."""

import html
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .artifacts import _default_inventory_for_assay
from .contracts import (
    _analysis_percent,
    _brief_text,
    _feature_family_label,
    _format_text_list,
    _label,
    _latest,
    _mapping,
    _mappings,
    _present,
    _public_field_label,
    _scalar,
    _text_values,
)
from .plots import _hvg_ranking_label


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
