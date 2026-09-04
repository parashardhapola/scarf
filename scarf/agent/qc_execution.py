"""Persist exact outputs from a registered agent cell-quality decision."""

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import numpy as np

from ..metadata.artifacts import (
    plan_cell_data_artifact,
    write_cell_data_artifact,
)
from ..metadata.rows import read_metadata_rows_chunkwise
from ..metadata.selection import NamedCellArtifact, resolve_cell_aligned_artifact
from ..storage.artifacts import canonical_bytes, fingerprint_array, fingerprint_strings
from ..storage.refs import ArtifactRef
from ..storage.selections import (
    read_stored_selection_mask,
    resolve_generated_selection_artifact,
)
from ..utils.logging import logger
from .qc_profiles import (
    REGISTERED_CELL_QC_PROFILES,
    RegisteredCellQcProfile,
    project_registered_qc_profile,
)


def _validated_named_cell_artifacts(
    values: Iterable[NamedCellArtifact] | None,
    *,
    expected_kind: str,
    label: str,
) -> list[NamedCellArtifact]:
    sources = list(values or ())
    names: set[str] = set()
    for source in sources:
        if not isinstance(source, NamedCellArtifact):
            raise TypeError(f"{label} must contain NamedCellArtifact values")
        if source.artifact.kind != expected_kind:
            raise ValueError(f"{label} must reference {expected_kind!r} artifacts")
        if source.name in names:
            raise ValueError(f"{label} must use unique semantic names")
        names.add(source.name)
    return sources


def execute_registered_cell_qc(
    store: Any,
    profile: RegisteredCellQcProfile,
    *,
    profile_parameters: Mapping[str, Any],
    expected_active_cells: int,
    expected_retained_cells: int,
    expected_flag_counts: Mapping[str, int],
    attrs: Iterable[str] | None = None,
    artifact_metrics: Iterable[NamedCellArtifact] | None = None,
    cell_selection: ArtifactRef | None = None,
    sample_column: str | None = None,
    sample_artifact: NamedCellArtifact | None = None,
    invalidate_cache: bool = False,
) -> tuple[ArtifactRef, ArtifactRef | None]:
    """Recompute, verify, and persist one registered cell-QC decision."""
    if profile not in REGISTERED_CELL_QC_PROFILES:
        raise ValueError(f"Unknown registered cell-QC profile {profile!r}")
    if (
        isinstance(expected_active_cells, bool)
        or not isinstance(expected_active_cells, int)
        or expected_active_cells < 1
    ):
        raise ValueError("expected_active_cells must be a positive integer")
    if (
        isinstance(expected_retained_cells, bool)
        or not isinstance(expected_retained_cells, int)
        or expected_retained_cells < 0
    ):
        raise ValueError("expected_retained_cells must be a non-negative integer")
    flag_counts = dict(expected_flag_counts)
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in flag_counts.items()
    ):
        raise ValueError(
            "expected_flag_counts must map non-empty names to non-negative integers"
        )

    parameters = dict(profile_parameters)
    required_parameter_keys = {
        "policyVersion",
        "profile",
        "nMads",
        "boundPolicy",
        "resolvedBounds",
        "captureSizes",
        "captureComparisons",
        "captureComparisonSource",
        "pooledReferenceCaptures",
    }
    if set(parameters) != required_parameter_keys:
        raise ValueError("Registered cell-QC parameters do not match policy version 1")
    if (
        isinstance(parameters["policyVersion"], bool)
        or parameters["policyVersion"] != 1
    ):
        raise ValueError("Registered cell-QC policyVersion must be 1")
    if parameters["profile"] != profile:
        raise ValueError("Registered cell-QC profile and parameters disagree")
    expected_n_mads = 3.0 if profile == "captureMad3Sensitivity" else 5.0
    if (
        isinstance(parameters["nMads"], bool)
        or not isinstance(parameters["nMads"], Real)
        or float(parameters["nMads"]) != expected_n_mads
    ):
        raise ValueError(
            f"Registered cell-QC profile {profile!r} requires nMads={expected_n_mads}"
        )
    expected_bound_policy = {
        "count": {"remove": "lower", "flag": "upper"},
        "feature": {"remove": "lower", "flag": "upper"},
        "mitochondrial": {"remove": "upper", "fixedCutoff": None},
        "diagnostic": {"remove": "none"},
    }
    if parameters["boundPolicy"] != expected_bound_policy:
        raise ValueError("Registered cell-QC boundPolicy is not supported")
    raw_bounds = parameters["resolvedBounds"]
    if not isinstance(raw_bounds, list) or any(
        not isinstance(value, Mapping) for value in raw_bounds
    ):
        raise TypeError("Registered cell-QC resolvedBounds must be a list of maps")
    pooled_references = parameters["pooledReferenceCaptures"]
    if not isinstance(pooled_references, list) or any(
        not isinstance(value, str) for value in pooled_references
    ):
        raise TypeError(
            "Registered cell-QC pooledReferenceCaptures must be a list of strings"
        )
    canonical_bytes(parameters)

    attrs_list = list(attrs or ())
    if any(not isinstance(attr, str) for attr in attrs_list):
        raise TypeError("attrs must contain only column names")
    metric_artifacts = _validated_named_cell_artifacts(
        artifact_metrics,
        expected_kind="quality_metric",
        label="artifact_metrics",
    )
    resolved_sample_artifact: NamedCellArtifact | None = None
    if sample_artifact is not None:
        resolved_sample_artifact = _validated_named_cell_artifacts(
            [sample_artifact],
            expected_kind="hto_identity",
            label="sample_artifact",
        )[0]
    if sample_column is not None and resolved_sample_artifact is not None:
        raise ValueError("sample_column and sample_artifact are mutually exclusive")
    capture_profile = profile in {
        "captureMad5",
        "captureMad3Sensitivity",
        "pooledReferenceMad5",
    }
    has_capture_source = (sample_column is None) != (resolved_sample_artifact is None)
    if capture_profile and not has_capture_source:
        raise ValueError(
            f"Registered cell-QC profile {profile!r} requires one capture source"
        )
    if not capture_profile and has_capture_source:
        raise ValueError(
            f"Registered cell-QC profile {profile!r} cannot use a capture source"
        )
    if sample_column is not None and sample_column not in store.cells.columns:
        raise ValueError(f"sample_column '{sample_column}' not found in cell metadata")
    missing = [attr for attr in attrs_list if attr not in store.cells.columns]
    if missing:
        joined = ", ".join(repr(attr) for attr in missing)
        raise KeyError(f"Cell metadata columns not found: {joined}")
    artifact_names = {source.name for source in metric_artifacts}
    duplicate_names = sorted(set(attrs_list).intersection(artifact_names))
    if duplicate_names:
        raise ValueError(
            "Metadata and artifact QC metrics must use distinct names: "
            f"{duplicate_names}"
        )

    prior = store._filter_input_selection(cell_selection)
    active = read_stored_selection_mask(
        store.zw,
        prior,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    active_idx = np.flatnonzero(active).astype(np.int64, copy=False)
    if len(active_idx) != expected_active_cells:
        raise ValueError(
            "Registered cell-QC active-cell count differs from its evidence"
        )

    values_by_name: dict[str, np.ndarray] = {}
    metadata_fingerprints: dict[str, str] = {}
    for attr in attrs_list:
        values = np.asarray(
            read_metadata_rows_chunkwise(store.cells, attr, active_idx),
            dtype=float,
        )
        if values.shape != (len(active_idx),):
            raise ValueError(
                f"QC metadata column {attr!r} does not align with cell_selection"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"QC values in {attr!r} contain non-finite entries")
        values_by_name[attr] = values
        metadata_fingerprints[attr] = fingerprint_array(values)
    for source in metric_artifacts:
        resolved = resolve_cell_aligned_artifact(
            store.zw,
            source.artifact,
            cell_selection=prior,
            expected_kind="quality_metric",
        )
        values = np.asarray(resolved.values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"QC artifact values in {source.name!r} contain non-finite entries"
            )
        values_by_name[source.name] = values

    sample_labels: np.ndarray | None = None
    sample_inputs: dict[str, Any] = {}
    if sample_column is not None:
        sample_labels = np.asarray(
            read_metadata_rows_chunkwise(store.cells, sample_column, active_idx)
        )
        sample_inputs["capture_assignments_fingerprint"] = fingerprint_strings(
            sample_labels
        )
    elif resolved_sample_artifact is not None:
        resolved_sample = resolve_cell_aligned_artifact(
            store.zw,
            resolved_sample_artifact.artifact,
            cell_selection=prior,
            expected_kind="hto_identity",
        )
        sample_labels = np.asarray(resolved_sample.values)
        sample_inputs["capture_artifact"] = resolved_sample_artifact.artifact

    projection = project_registered_qc_profile(
        profile,
        values_by_metric=values_by_name,
        active=np.ones(len(active_idx), dtype=bool),
        capture_labels=sample_labels,
        grouping_proven=capture_profile,
        min_cells_per_capture=20,
        pooled_reference_captures=tuple(pooled_references),
    )
    recomputed_bounds = [value.to_dict() for value in projection.thresholds]
    if canonical_bytes(recomputed_bounds) != canonical_bytes(raw_bounds):
        raise ValueError("Registered cell-QC resolved bounds do not match the inputs")
    if capture_profile:
        if projection.captureSizes != parameters["captureSizes"]:
            raise ValueError(
                "Registered cell-QC capture sizes do not match the exact inputs"
            )
        comparisons = [
            comparison.to_dict() for comparison in projection.captureComparisons
        ]
        if canonical_bytes(comparisons) != canonical_bytes(
            parameters["captureComparisons"]
        ):
            raise ValueError(
                "Registered cell-QC capture comparisons do not match the inputs"
            )
    if projection.retainedCells != expected_retained_cells:
        raise ValueError(
            "Registered cell-QC retained-cell count differs from its evidence"
        )
    if projection.flagCounts != flag_counts:
        raise ValueError(
            "Registered cell-QC diagnostic-flag counts differ from their evidence"
        )

    metric_sources = [
        {"name": attr, "source": "metadataColumn", "column": attr}
        for attr in attrs_list
    ]
    metric_sources.extend(
        {"name": source.name, "source": "artifact"} for source in metric_artifacts
    )
    source_inputs: dict[str, Any] = {
        "metadata_fingerprints": metadata_fingerprints,
        "artifact_metrics": {
            source.name: source.artifact for source in metric_artifacts
        },
        **sample_inputs,
    }
    execution_parameters: dict[str, Any] = {
        "profile": profile,
        "metricSources": metric_sources,
        "profileParameters": parameters,
        "flagCounts": projection.flagCounts,
    }
    canonical_bytes(
        {
            "operation": "run_registered_cell_qc",
            "parameters": execution_parameters,
            "inputs": {"prior_cell_selection": prior, **source_inputs},
        }
    )

    flag_names = tuple(sorted(projection.flags))
    flag_ref: ArtifactRef | None = None
    if flag_names:
        flag_values = np.column_stack(
            [projection.flags[name] for name in flag_names]
        ).astype(bool, copy=False)
        planned_flags = plan_cell_data_artifact(
            store.zw,
            scope="datastore",
            kind="metadata_snapshot",
            operation="run_registered_cell_qc_flags",
            parameters={
                **execution_parameters,
                "flagNames": list(flag_names),
            },
            inputs=source_inputs,
            execution_options={},
            cell_selection=prior,
            arrays={"values": (flag_values.shape, "b")},
            invalidate_cache=invalidate_cache,
        )
        write_cell_data_artifact(
            store.zw,
            planned_flags,
            {"values": flag_values},
            fingerprint_payload=True,
        )
        flag_ref = planned_flags.ref

    keep = np.zeros(store.cells.N, dtype=bool)
    keep[active_idx] = projection.keep
    selection_inputs: dict[str, Any] = {
        "prior_cell_selection": prior,
        **source_inputs,
    }
    if flag_ref is not None:
        selection_inputs["diagnostic_flags"] = flag_ref
    ref, stored = resolve_generated_selection_artifact(
        store.zw,
        scope="datastore",
        kind="cell_selection",
        values=keep,
        row_ids=np.asarray(store.cells.fetch_all("ids")),
        operation="run_registered_cell_qc",
        parameters=execution_parameters,
        inputs=selection_inputs,
        source_column="artifact",
        invalidate_cache=invalidate_cache,
    )
    logger.info(
        f"Registered cell QC {profile!r} retained "
        f"{int(stored.sum())}/{store.cells.N} cells"
    )
    return ref, flag_ref


__all__ = ["execute_registered_cell_qc"]
