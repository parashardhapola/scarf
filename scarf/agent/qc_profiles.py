"""Registered one-sided cell-quality profiles and bounded projections."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..quality_control.filtering import (
    _clamp_metric_bound,
    _from_work_scale,
    _mad_bounds,
    _validated_sample_labels,
    _validated_work_scale,
)

type RegisteredCellQcProfile = Literal[
    "retainWithFlags",
    "globalMad5",
    "captureMad5",
    "captureMad3Sensitivity",
    "pooledReferenceMad5",
]
type QcMetricRole = Literal["count", "feature", "mitochondrial", "diagnostic"]
type QcRemovalDirection = Literal["lower", "upper", "none"]

REGISTERED_CELL_QC_PROFILES: tuple[RegisteredCellQcProfile, ...] = (
    "retainWithFlags",
    "globalMad5",
    "captureMad5",
    "captureMad3Sensitivity",
    "pooledReferenceMad5",
)


@dataclass(frozen=True, slots=True)
class RegisteredQcThreshold:
    """One data-derived threshold for one metric and reference population."""

    metric: str
    role: QcMetricRole
    group: str
    nMads: float
    transform: Literal["identity", "log1p"]
    removalDirection: QcRemovalDirection
    median: float
    scaledMad: float
    lowerRemoval: float | None
    upperRemoval: float | None
    upperFlag: float | None
    skipReason: Literal["zeroMad"] | None = None

    def to_dict(self) -> dict[str, str | float | None]:
        """Return JSON-safe threshold evidence."""
        return {
            "metric": self.metric,
            "role": self.role,
            "group": self.group,
            "nMads": self.nMads,
            "transform": self.transform,
            "removalDirection": self.removalDirection,
            "median": self.median,
            "scaledMad": self.scaledMad,
            "lowerRemoval": self.lowerRemoval,
            "upperRemoval": self.upperRemoval,
            "upperFlag": self.upperFlag,
            "skipReason": self.skipReason,
        }


@dataclass(frozen=True, slots=True)
class CaptureQcComparison:
    """Comparison of one capture median with the complete active population."""

    capture: str
    cells: int
    adverseGlobalOutlier: bool
    reasons: tuple[str, ...] = ()
    metricComparisons: dict[str, dict[str, float | str | None]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe failed-capture evidence."""
        return {
            "capture": self.capture,
            "cells": self.cells,
            "adverseGlobalOutlier": self.adverseGlobalOutlier,
            "reasons": list(self.reasons),
            "metricComparisons": self.metricComparisons,
        }


@dataclass(frozen=True, slots=True)
class RegisteredQcProjection:
    """Exact masks and evidence for one registered cell-QC profile."""

    profile: RegisteredCellQcProfile
    keep: np.ndarray
    flags: dict[str, np.ndarray]
    thresholds: tuple[RegisteredQcThreshold, ...]
    captureSizes: dict[str, int] = field(default_factory=dict)
    retainedByCapture: dict[str, int] = field(default_factory=dict)
    captureComparisons: tuple[CaptureQcComparison, ...] = ()
    failedCaptureCandidates: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def retainedCells(self) -> int:
        """Number of active cells retained by the projection."""
        return int(self.keep.sum())

    @property
    def flagCounts(self) -> dict[str, int]:
        """Counts for each non-removal diagnostic flag."""
        return {name: int(mask.sum()) for name, mask in self.flags.items()}


def registered_qc_metric_role(metric: str) -> QcMetricRole:
    """Classify one conventional cell-quality metric without gene inspection."""
    normalized = metric.replace("_", "").lower()
    if normalized.endswith("ncounts") or normalized.endswith("totalcounts"):
        return "count"
    if normalized.endswith("nfeatures") or normalized.endswith("ngenesbycounts"):
        return "feature"
    if (
        normalized.endswith("percentmito")
        or normalized.endswith("pctcountsmt")
        or normalized.endswith("mitochondrialpercent")
    ):
        return "mitochondrial"
    return "diagnostic"


def _metric_policy(
    role: QcMetricRole,
) -> tuple[Literal["identity", "log1p"], QcRemovalDirection]:
    if role in {"count", "feature"}:
        return "log1p", "lower"
    if role == "mitochondrial":
        return "identity", "upper"
    return "identity", "none"


def _validated_inputs(
    values_by_metric: dict[str, np.ndarray],
    active: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    active_mask = np.asarray(active, dtype=bool)
    if active_mask.ndim != 1:
        raise ValueError("active must be a one-dimensional boolean vector")
    if not active_mask.any():
        raise ValueError("active must select at least one cell")
    values: dict[str, np.ndarray] = {}
    for metric, raw in values_by_metric.items():
        if (
            not isinstance(metric, str)
            or not metric.strip()
            or metric != metric.strip()
        ):
            raise ValueError("QC metric names must be non-empty trimmed strings")
        array = np.asarray(raw, dtype=float)
        if array.ndim != 1 or array.shape != active_mask.shape:
            raise ValueError(f"QC metric {metric!r} must align with active")
        if not np.isfinite(array[active_mask]).all():
            raise ValueError(f"QC metric {metric!r} contains non-finite active values")
        values[metric] = array
    return values, active_mask


def _threshold(
    metric: str,
    values: np.ndarray,
    reference: np.ndarray,
    *,
    group: str,
    n_mads: float,
) -> RegisteredQcThreshold:
    role = registered_qc_metric_role(metric)
    transform, removal_direction = _metric_policy(role)
    work = _validated_work_scale(values[reference], attr=metric, transform=transform)
    median_work = float(np.median(work))
    low_work, high_work, scaled_mad = _mad_bounds(work, n_mads)
    if scaled_mad == 0.0:
        median = _clamp_metric_bound(
            _from_work_scale(median_work, transform),
            transform=transform,
            is_percent=role == "mitochondrial",
        )
        return RegisteredQcThreshold(
            metric=metric,
            role=role,
            group=group,
            nMads=n_mads,
            transform=transform,
            removalDirection=removal_direction,
            median=median,
            scaledMad=0.0,
            lowerRemoval=None,
            upperRemoval=None,
            upperFlag=None,
            skipReason="zeroMad",
        )

    median = _clamp_metric_bound(
        _from_work_scale(median_work, transform),
        transform=transform,
        is_percent=role == "mitochondrial",
    )
    low = _clamp_metric_bound(
        _from_work_scale(low_work, transform),
        transform=transform,
        is_percent=role == "mitochondrial",
    )
    high = _clamp_metric_bound(
        _from_work_scale(high_work, transform),
        transform=transform,
        is_percent=role == "mitochondrial",
    )
    return RegisteredQcThreshold(
        metric=metric,
        role=role,
        group=group,
        nMads=n_mads,
        transform=transform,
        removalDirection=removal_direction,
        median=median,
        scaledMad=scaled_mad,
        lowerRemoval=low if removal_direction == "lower" else None,
        upperRemoval=high if removal_direction == "upper" else None,
        upperFlag=high if role in {"count", "feature"} else None,
    )


def _apply_threshold(
    values: np.ndarray,
    target: np.ndarray,
    threshold: RegisteredQcThreshold,
    keep: np.ndarray,
    flags: dict[str, np.ndarray],
    *,
    apply_removal: bool,
) -> None:
    if threshold.lowerRemoval is not None:
        removed = target & (values < threshold.lowerRemoval)
        low_flag = flags.setdefault(
            f"{threshold.metric}:lowQuality",
            np.zeros(target.shape[0], dtype=bool),
        )
        low_flag[removed] = True
        if apply_removal:
            keep[removed] = False
    if threshold.upperRemoval is not None:
        removed = target & (values > threshold.upperRemoval)
        mito_flag = flags.setdefault(
            f"{threshold.metric}:highMito",
            np.zeros(target.shape[0], dtype=bool),
        )
        mito_flag[removed] = True
        if apply_removal:
            keep[removed] = False
    if threshold.upperFlag is not None:
        flag_name = f"{threshold.metric}:high"
        flag = flags.setdefault(flag_name, np.zeros(target.shape[0], dtype=bool))
        flag[target & (values > threshold.upperFlag)] = True


def _ordered_capture_masks(
    capture_labels: np.ndarray,
    active: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    labels = _validated_sample_labels(
        capture_labels,
        active,
        label_name="physical capture labels",
    )
    captures: list[tuple[str, np.ndarray]] = []
    seen_values: list[object] = []
    seen_keys: set[str] = set()
    for raw in labels[active]:
        value = raw.item() if isinstance(raw, np.generic) else raw
        if any(value == seen for seen in seen_values):
            continue
        key = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if key in seen_keys:
            raise ValueError(
                "Physical capture labels collide after provenance encoding"
            )
        seen_values.append(value)
        seen_keys.add(key)
        captures.append((key, active & (labels == raw)))
    return captures


def _global_capture_comparisons(
    values_by_metric: dict[str, np.ndarray],
    active: np.ndarray,
    captures: list[tuple[str, np.ndarray]],
) -> tuple[CaptureQcComparison, ...]:
    global_thresholds = {
        metric: _threshold(
            metric,
            values,
            active,
            group="globalComparison",
            n_mads=5.0,
        )
        for metric, values in values_by_metric.items()
        if registered_qc_metric_role(metric) != "diagnostic"
    }
    comparisons: list[CaptureQcComparison] = []
    for capture, mask in captures:
        metric_comparisons: dict[str, dict[str, float | str | None]] = {}
        reasons: list[str] = []
        for metric, threshold in global_thresholds.items():
            role = threshold.role
            transform, _ = _metric_policy(role)
            capture_work = _validated_work_scale(
                values_by_metric[metric][mask],
                attr=metric,
                transform=transform,
            )
            capture_median_work = float(np.median(capture_work))
            capture_median = _clamp_metric_bound(
                _from_work_scale(capture_median_work, transform),
                transform=transform,
                is_percent=role == "mitochondrial",
            )
            global_mad = threshold.scaledMad
            standardized_shift = (
                None
                if global_mad == 0.0
                else (capture_median_work - np.log1p(threshold.median)) / global_mad
                if transform == "log1p"
                else (capture_median - threshold.median) / global_mad
            )
            adverse = (
                threshold.lowerRemoval is not None
                and capture_median < threshold.lowerRemoval
            ) or (
                threshold.upperRemoval is not None
                and capture_median > threshold.upperRemoval
            )
            if adverse:
                reasons.append(f"{metric}:{role}:adverseGlobalMedian")
            metric_comparisons[metric] = {
                "role": role,
                "captureMedian": capture_median,
                "globalMedian": threshold.median,
                "globalLower": threshold.lowerRemoval,
                "globalUpper": threshold.upperRemoval,
                "standardizedMedianShift": standardized_shift,
            }
        comparisons.append(
            CaptureQcComparison(
                capture=capture,
                cells=int(mask.sum()),
                adverseGlobalOutlier=bool(reasons),
                reasons=tuple(reasons),
                metricComparisons=metric_comparisons,
            )
        )
    return tuple(comparisons)


def project_registered_qc_profile(
    profile: RegisteredCellQcProfile,
    *,
    values_by_metric: dict[str, np.ndarray],
    active: np.ndarray,
    capture_labels: np.ndarray | None = None,
    grouping_proven: bool = False,
    min_cells_per_capture: int = 20,
    pooled_reference_captures: tuple[str, ...] | None = None,
) -> RegisteredQcProjection:
    """Project one registered QC profile without reading or writing a matrix."""
    if profile not in REGISTERED_CELL_QC_PROFILES:
        raise ValueError(f"Unknown registered cell-QC profile {profile!r}")
    if min_cells_per_capture < 2:
        raise ValueError("min_cells_per_capture must be at least 2")
    values, active_mask = _validated_inputs(values_by_metric, active)
    filtering_values = {
        metric: metric_values
        for metric, metric_values in values.items()
        if registered_qc_metric_role(metric) != "diagnostic"
    }
    if not filtering_values:
        if profile != "retainWithFlags":
            raise ValueError("MAD profiles require count, feature, or mito metrics")
        return RegisteredQcProjection(
            profile=profile,
            keep=active_mask.copy(),
            flags={},
            thresholds=(),
            warnings=("No registered count, feature, or mito metrics were available",),
        )

    capture_profile = profile in {
        "captureMad5",
        "captureMad3Sensitivity",
        "pooledReferenceMad5",
    }
    if capture_profile and (capture_labels is None or not grouping_proven):
        raise ValueError(
            f"{profile} requires an explicitly proven physical capture grouping"
        )
    captures = (
        _ordered_capture_masks(np.asarray(capture_labels), active_mask)
        if capture_labels is not None and grouping_proven
        else []
    )
    capture_sizes = {name: int(mask.sum()) for name, mask in captures}
    if profile in {"captureMad5", "captureMad3Sensitivity"}:
        undersized = [
            name for name, size in capture_sizes.items() if size < min_cells_per_capture
        ]
        if undersized:
            raise ValueError(
                f"{profile} requires at least {min_cells_per_capture} cells in every "
                f"capture; undersized={undersized}"
            )
    if profile == "pooledReferenceMad5":
        references = tuple(pooled_reference_captures or ())
        if len(references) < 2 or len(references) != len(set(references)):
            raise ValueError(
                "pooledReferenceMad5 requires at least two unique reference captures"
            )
        unknown = sorted(set(references).difference(capture_sizes))
        if unknown:
            raise ValueError(f"Unknown pooled reference captures: {unknown}")
        pooled_cells = sum(capture_sizes[name] for name in references)
        if pooled_cells < min_cells_per_capture:
            raise ValueError(
                "Pooled reference captures do not contain enough active cells"
            )

    n_mads = 3.0 if profile == "captureMad3Sensitivity" else 5.0
    keep = active_mask.copy()
    flags: dict[str, np.ndarray] = {}
    thresholds: list[RegisteredQcThreshold] = []
    warnings: list[str] = []

    reference_groups: list[tuple[str, np.ndarray, np.ndarray]]
    if profile in {"retainWithFlags", "globalMad5"}:
        reference_groups = [("global", active_mask, active_mask)]
    elif profile == "pooledReferenceMad5":
        reference_names = set(pooled_reference_captures or ())
        reference = np.zeros(active_mask.shape[0], dtype=bool)
        for name, mask in captures:
            if name in reference_names:
                reference |= mask
        reference_groups = [("pooledReference", active_mask, reference)]
    else:
        reference_groups = [(name, mask, mask) for name, mask in captures]

    for group, target, reference in reference_groups:
        for metric, metric_values in filtering_values.items():
            threshold = _threshold(
                metric,
                metric_values,
                reference,
                group=group,
                n_mads=n_mads,
            )
            thresholds.append(threshold)
            if threshold.skipReason is not None:
                warnings.append(
                    f"Ignored {metric!r} for {group!r} because its MAD was zero"
                )
                continue
            _apply_threshold(
                metric_values,
                target,
                threshold,
                keep,
                flags,
                apply_removal=profile != "retainWithFlags",
            )

    comparisons = (
        _global_capture_comparisons(filtering_values, active_mask, captures)
        if captures
        else ()
    )
    failed = tuple(
        comparison.capture
        for comparison in comparisons
        if comparison.adverseGlobalOutlier
    )
    if failed:
        warnings.append(
            "Capture medians outside adverse global MAD bounds require review: "
            + ", ".join(failed)
        )
    retained_by_capture = {name: int((mask & keep).sum()) for name, mask in captures}
    return RegisteredQcProjection(
        profile=profile,
        keep=keep,
        flags=flags,
        thresholds=tuple(thresholds),
        captureSizes=capture_sizes,
        retainedByCapture=retained_by_capture,
        captureComparisons=comparisons,
        failedCaptureCandidates=failed,
        warnings=tuple(warnings),
    )


def offered_registered_qc_profiles(
    *,
    values_by_metric: dict[str, np.ndarray],
    active: np.ndarray,
    capture_labels: np.ndarray | None = None,
    grouping_proven: bool = False,
    min_cells_per_capture: int = 20,
    pooled_reference_captures: tuple[str, ...] | None = None,
) -> list[RegisteredQcProjection]:
    """Build only profiles whose deterministic eligibility gates pass."""
    projections = [
        project_registered_qc_profile(
            "retainWithFlags",
            values_by_metric=values_by_metric,
            active=active,
            capture_labels=capture_labels,
            grouping_proven=grouping_proven,
        )
    ]
    try:
        global_projection = project_registered_qc_profile(
            "globalMad5",
            values_by_metric=values_by_metric,
            active=active,
            capture_labels=capture_labels,
            grouping_proven=grouping_proven,
        )
    except ValueError:
        return projections
    projections.append(global_projection)
    if capture_labels is None or not grouping_proven:
        return projections
    for profile in ("captureMad5", "captureMad3Sensitivity"):
        try:
            projection = project_registered_qc_profile(
                profile,
                values_by_metric=values_by_metric,
                active=active,
                capture_labels=capture_labels,
                grouping_proven=True,
                min_cells_per_capture=min_cells_per_capture,
            )
        except ValueError:
            continue
        projections.append(projection)
    if pooled_reference_captures:
        try:
            projection = project_registered_qc_profile(
                "pooledReferenceMad5",
                values_by_metric=values_by_metric,
                active=active,
                capture_labels=capture_labels,
                grouping_proven=True,
                min_cells_per_capture=min_cells_per_capture,
                pooled_reference_captures=pooled_reference_captures,
            )
        except ValueError:
            pass
        else:
            projections.append(projection)
    return projections


__all__ = [
    "REGISTERED_CELL_QC_PROFILES",
    "CaptureQcComparison",
    "QcMetricRole",
    "RegisteredCellQcProfile",
    "RegisteredQcProjection",
    "RegisteredQcThreshold",
    "offered_registered_qc_profiles",
    "project_registered_qc_profile",
    "registered_qc_metric_role",
]
