"""Experimental-context quality-control evidence assembly."""

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np

from ...metadata.selection import resolve_cell_aligned_artifact
from ...quality_control.filtering import (
    _validated_sample_labels,
    gaussian_quantile_bounds,
)
from ...storage.artifacts import (
    fingerprint_array,
    fingerprint_strings,
    inspect_artifact,
)
from ...storage.refs import ArtifactRef
from ...storage.selections import read_stored_selection_mask
from ..cell_quality.profiles import (
    AutoFilterProjection,
    QcMetricRole,
    RegisteredCellQcProfile,
    RegisteredQcProjection,
    offered_registered_qc_profiles,
    project_auto_filter_profile,
    qc_metric_execution_name,
    registered_qc_metric_role,
)
from ..tools import artifact_reference, core_artifact_reference
from ..types import ArtifactReferenceModel
from .contracts import (
    CaptureFailureEvidence,
    CellQcAction,
    CellQcDriverType,
    CellQcProfileEvidence,
    CovariateCharacterization,
    ExperimentalContextDependencies,
    LegacyCellQcAction,
    NamedArtifactSource,
    QcMetricSourceEvidence,
    QcSourceConcordance,
)

_MAX_QC_SAMPLE_PROFILES = 4


def _persisted_assay_type(store: Any, assay_name: str) -> str:
    """Read one persisted assay type without inferring modality from features."""
    root = getattr(store, "zw", None)
    attrs = getattr(root, "attrs", {})
    raw_types = attrs.get("assayTypes", {}) if isinstance(attrs, Mapping) else {}
    if isinstance(raw_types, Mapping):
        assay_type = raw_types.get(assay_name)
        if isinstance(assay_type, str):
            return assay_type
    return assay_name if assay_name in {"RNA", "ATAC", "ADT", "HTO"} else "Assay"


def _qc_driver(store: Any) -> tuple[str, CellQcDriverType] | None:
    """Choose the first RNA assay, otherwise the first ATAC assay."""
    assay_names = [str(name) for name in getattr(store, "assay_names", [])]
    for assay_type in ("RNA", "ATAC"):
        for assay_name in assay_names:
            if _persisted_assay_type(store, assay_name) == assay_type:
                return assay_name, assay_type
    return None


def _hto_identity_columns(deps: ExperimentalContextDependencies) -> list[str]:
    """Return explicitly supplied imported HTO identity metadata columns."""
    requested: list[str] = []
    directed_many = deps.directions.get("htoIdentityColumns")
    if isinstance(directed_many, list | tuple):
        requested.extend(str(value) for value in directed_many)
    directed_one = deps.directions.get("htoIdentityColumn")
    if isinstance(directed_one, str):
        requested.append(directed_one)
    available = set(deps.store.cells.columns)
    return list(dict.fromkeys(name for name in requested if name in available))


def _cell_selection_ref(deps: ExperimentalContextDependencies) -> ArtifactRef:
    selection = core_artifact_reference(deps.cellSelection)
    if not isinstance(selection, ArtifactRef):
        raise ValueError("cellSelection must identify an exact artifact")
    if selection.kind != "cell_selection" or selection.scope != "datastore":
        raise ValueError("cellSelection must identify a datastore cell selection")
    return selection


def _active_cell_count(deps: ExperimentalContextDependencies) -> int:
    selection = _cell_selection_ref(deps)
    active = read_stored_selection_mask(
        deps.store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    if active.ndim != 1 or active.shape[0] != deps.store.cells.N:
        raise ValueError(
            "cellSelection must contain an aligned boolean selection vector"
        )
    return int(active.sum())


def _source_ref(
    source: NamedArtifactSource,
    *,
    expected_kind: str,
) -> ArtifactRef:
    if not isinstance(source, NamedArtifactSource):
        raise TypeError("Artifact sources must be NamedArtifactSource values")
    if not source.name.strip():
        raise ValueError("Artifact sources require a non-empty semantic name")
    artifact = core_artifact_reference(source.artifact)
    if not isinstance(artifact, ArtifactRef) or artifact.kind != expected_kind:
        raise ValueError(
            f"Artifact source {source.name!r} must reference {expected_kind!r}"
        )
    return artifact


def _artifact_evidence_id(source: NamedArtifactSource) -> str:
    return f"htoIdentityArtifact:{source.name}:{source.artifact.artifactId}"


def _hto_artifact_map(
    deps: ExperimentalContextDependencies,
) -> dict[str, ArtifactRef]:
    artifacts: dict[str, ArtifactRef] = {}
    for source in deps.htoIdentityArtifacts:
        if source.name in artifacts:
            raise ValueError("HTO identity artifact names must be unique")
        artifacts[source.name] = _source_ref(
            source,
            expected_kind="hto_identity",
        )
    return artifacts


def _resolved_artifact_values(
    deps: ExperimentalContextDependencies,
    source: NamedArtifactSource,
    *,
    expected_kind: str,
) -> np.ndarray:
    resolved = resolve_cell_aligned_artifact(
        deps.store.zw,
        _source_ref(source, expected_kind=expected_kind),
        cell_selection=_cell_selection_ref(deps),
        expected_kind=expected_kind,
    )
    return np.asarray(resolved.values)


def _artifact_input_references(
    value: Any,
    *,
    limit: int = 16,
) -> list[ArtifactReferenceModel]:
    refs: list[ArtifactReferenceModel] = []
    seen: set[tuple[str, str | None, str, str]] = set()

    def visit(item: Any) -> None:
        if len(refs) >= limit:
            return
        if isinstance(item, ArtifactRef):
            ref = item
        elif isinstance(item, Mapping) and {
            "scope",
            "kind",
            "artifact_id",
        }.issubset(item):
            try:
                ref = ArtifactRef.from_dict(item)
            except (KeyError, TypeError, ValueError):
                ref = None
        else:
            ref = None
        if ref is not None:
            key = (ref.scope, ref.assay, ref.kind, ref.artifact_id)
            if key not in seen:
                seen.add(key)
                refs.append(artifact_reference(ref))
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list | tuple):
            for nested in item:
                visit(nested)

    visit(value)
    return refs


def _qc_metric_sources(
    deps: ExperimentalContextDependencies,
    driver: tuple[str, CellQcDriverType],
) -> tuple[
    dict[str, np.ndarray],
    list[str],
    list[NamedArtifactSource],
    list[QcMetricSourceEvidence],
    list[QcSourceConcordance],
    list[str],
    dict[str, np.ndarray],
]:
    assay_name, assay_type = driver
    del assay_type
    selection = _cell_selection_ref(deps)
    selection_model = artifact_reference(selection)
    active_cells = _active_cell_count(deps)
    metadata_names = _qc_attributes(deps.store, assay_name, driver[1])
    artifact_candidates: list[NamedArtifactSource] = []
    for source in deps.qualityMetricArtifacts:
        artifact = _source_ref(source, expected_kind="quality_metric")
        if artifact.assay == assay_name:
            artifact_candidates.append(source)
    metadata_collisions = set(metadata_names).intersection(
        source.name for source in artifact_candidates
    )

    values_by_execution_name: dict[str, np.ndarray] = {}
    values_by_source: dict[str, np.ndarray] = {}
    sources: list[QcMetricSourceEvidence] = []
    valid_metadata: list[str] = []
    valid_artifacts: list[NamedArtifactSource] = []
    notes: list[str] = []

    for name in metadata_names:
        raw = np.asarray(deps.cells.fetch(name))
        try:
            values = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            fingerprint = fingerprint_strings(raw)
            source_id = f"qcMetric:metadata:{assay_name}:{name}:{fingerprint}"
            sources.append(
                QcMetricSourceEvidence(
                    sourceId=source_id,
                    metricName=name,
                    metricRole=registered_qc_metric_role(name),
                    assay=assay_name,
                    sourceType="metadataColumn",
                    origin="ingestionMetadata",
                    executionName=name,
                    metadataColumn=name,
                    cellSelection=selection_model,
                    valuesFingerprint=fingerprint,
                    activeCells=active_cells,
                    missingCells=active_cells,
                    notes=["Metric is not numeric and cannot drive filtering"],
                )
            )
            notes.append(f"QC metadata source {name!r} is not numeric")
            continue
        if values.ndim != 1 or values.shape != (active_cells,):
            raise ValueError(
                f"QC metadata source {name!r} does not align with cellSelection"
            )
        fingerprint = fingerprint_array(values)
        missing = int((~np.isfinite(values)).sum())
        source_id = f"qcMetric:metadata:{assay_name}:{name}:{fingerprint}"
        usable = missing == 0
        source_notes = (
            [] if usable else [f"{missing} active cells have non-finite metric values"]
        )
        sources.append(
            QcMetricSourceEvidence(
                sourceId=source_id,
                metricName=name,
                metricRole=registered_qc_metric_role(name),
                assay=assay_name,
                sourceType="metadataColumn",
                origin="ingestionMetadata",
                executionName=name,
                metadataColumn=name,
                cellSelection=selection_model,
                valuesFingerprint=fingerprint,
                activeCells=active_cells,
                missingCells=missing,
                usableForFiltering=usable,
                notes=source_notes,
            )
        )
        values_by_source[source_id] = values
        if usable:
            values_by_execution_name[name] = values
            valid_metadata.append(name)
        else:
            notes.extend(source_notes)

    for source in artifact_candidates:
        artifact = _source_ref(source, expected_kind="quality_metric")
        values = np.asarray(
            _resolved_artifact_values(
                deps,
                source,
                expected_kind="quality_metric",
            ),
            dtype=float,
        )
        if values.ndim != 1 or values.shape != (active_cells,):
            raise ValueError(
                f"QC artifact {source.name!r} does not align with cellSelection"
            )
        execution_name = qc_metric_execution_name(
            source.name,
            artifact_id=artifact.artifact_id,
            collides_with_metadata=source.name in metadata_collisions,
        )
        if execution_name in values_by_execution_name:
            raise ValueError(
                f"QC execution metric name {execution_name!r} is not unique"
            )
        fingerprint = fingerprint_array(values)
        missing = int((~np.isfinite(values)).sum())
        status = inspect_artifact(deps.store.zw, artifact)
        operation = status.operation
        origin: Literal[
            "ingestionMetadata",
            "derivedArtifact",
            "externalArtifact",
        ] = (
            "derivedArtifact"
            if operation == "run_feature_percentage"
            else "externalArtifact"
        )
        source_id = (
            f"qcMetric:artifact:{artifact.assay}:{source.name}:{artifact.artifact_id}"
        )
        usable = missing == 0
        source_notes = (
            [] if usable else [f"{missing} active cells have non-finite metric values"]
        )
        sources.append(
            QcMetricSourceEvidence(
                sourceId=source_id,
                metricName=source.name,
                metricRole=registered_qc_metric_role(source.name),
                assay=assay_name,
                sourceType="artifact",
                origin=origin,
                executionName=execution_name,
                artifact=artifact_reference(artifact),
                cellSelection=selection_model,
                inputArtifacts=_artifact_input_references(status.inputs or {}),
                provenanceOperation=operation,
                valuesFingerprint=fingerprint,
                activeCells=active_cells,
                missingCells=missing,
                usableForFiltering=usable,
                notes=source_notes,
            )
        )
        values_by_source[source_id] = values
        if usable:
            values_by_execution_name[execution_name] = values
            valid_artifacts.append(source)
        else:
            notes.extend(source_notes)

    concordance: list[QcSourceConcordance] = []
    metadata_sources = [
        source for source in sources if source.sourceType == "metadataColumn"
    ]
    artifact_sources = [source for source in sources if source.sourceType == "artifact"]
    for left in metadata_sources:
        for right in artifact_sources:
            if left.metricRole != right.metricRole or left.metricRole == "diagnostic":
                continue
            if right.artifact is None:
                raise ValueError("Artifact QC source lacks its exact reference")
            left_values = values_by_source.get(left.sourceId)
            right_values = values_by_source.get(right.sourceId)
            if left_values is None or right_values is None:
                continue
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            compared = int(finite.sum())
            missing = int(len(finite) - compared)
            mean_difference: float | None = None
            maximum_difference: float | None = None
            pearson: float | None = None
            exactly_equal = False
            numerically_close = False
            if compared:
                left_finite = left_values[finite]
                right_finite = right_values[finite]
                differences = np.abs(left_finite - right_finite)
                mean_difference = float(differences.mean())
                maximum_difference = float(differences.max())
                exactly_equal = missing == 0 and bool(
                    np.array_equal(left_finite, right_finite)
                )
                numerically_close = missing == 0 and bool(
                    np.allclose(
                        left_finite,
                        right_finite,
                        rtol=1e-6,
                        atol=1e-8,
                    )
                )
                if (
                    compared >= 2
                    and float(np.std(left_finite)) > 0.0
                    and float(np.std(right_finite)) > 0.0
                ):
                    correlation = float(np.corrcoef(left_finite, right_finite)[0, 1])
                    if math.isfinite(correlation):
                        pearson = correlation
            evidence_id = (
                f"qcConcordance:{left.metricRole}:"
                f"{left.valuesFingerprint}:{right.artifact.artifactId}"
            )
            concordance.append(
                QcSourceConcordance(
                    metricRole=left.metricRole,
                    leftSourceId=left.sourceId,
                    rightSourceId=right.sourceId,
                    comparedCells=compared,
                    missingCells=missing,
                    meanAbsoluteDifference=mean_difference,
                    maximumAbsoluteDifference=maximum_difference,
                    pearsonCorrelation=pearson,
                    exactlyEqual=exactly_equal,
                    numericallyClose=numerically_close,
                    evidenceId=evidence_id,
                )
            )
    return (
        values_by_execution_name,
        valid_metadata,
        valid_artifacts,
        sources,
        concordance,
        notes,
        values_by_source,
    )


def _qc_attributes(store: Any, assay_name: str, assay_type: str) -> list[str]:
    del assay_type
    suffixes = ["nCounts", "nFeatures", "percentMito", "percentRibo"]
    available = set(store.cells.columns)
    return [
        f"{assay_name}_{suffix}"
        for suffix in suffixes
        if f"{assay_name}_{suffix}" in available
    ]


def _derive_missing_percentage_artifacts(
    store: Any,
    *,
    cell_selection: ArtifactRef,
    driver: tuple[str, CellQcDriverType] | None,
    quality_sources: Sequence[NamedArtifactSource],
) -> list[NamedArtifactSource]:
    """Derive missing RNA percentage metrics through public immutable APIs."""
    sources = list(quality_sources)
    if driver is None or driver[1] != "RNA":
        return sources
    if not callable(getattr(store, "set_feature_selection", None)) or not callable(
        getattr(store, "run_feature_percentage", None)
    ):
        return sources
    assay_name = driver[0]
    available_metadata = set(store.cells.columns)
    supplied_roles = {
        registered_qc_metric_role(source.name)
        for source in sources
        if source.artifact.assay == assay_name
    }
    assay = store.get_assay(assay_name)
    feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
    feature_names = np.asarray(assay.feats.fetch_all("names")).astype(str)
    specifications: tuple[
        tuple[QcMetricRole, str, re.Pattern[str]],
        ...,
    ] = (
        ("mitochondrial", "percentMito", re.compile(r"^(MT-|mt-)")),
        (
            "ribosomal",
            "percentRibo",
            re.compile(r"^(RPS|RPL|MRPS|MRPL|Rps|Rpl|Mrps|Mrpl)"),
        ),
    )
    existing_names = {source.name for source in sources}
    for role, suffix, pattern in specifications:
        metric_name = f"{assay_name}_{suffix}"
        if metric_name in available_metadata or role in supplied_roles:
            continue
        mask = np.fromiter(
            (
                pattern.search(feature_id) is not None
                or pattern.search(feature_name) is not None
                for feature_id, feature_name in zip(
                    feature_ids,
                    feature_names,
                    strict=True,
                )
            ),
            dtype=bool,
            count=assay.feats.N,
        )
        if not mask.any():
            continue
        if metric_name in existing_names:
            raise ValueError(f"Derived QC metric name {metric_name!r} is not unique")
        feature_selection = store.set_feature_selection(
            from_assay=assay_name,
            mask=mask,
            invalidate_cache=False,
        )
        metric = store.run_feature_percentage(
            cell_selection,
            feature_selection,
            invalidate_cache=False,
        )
        sources.append(
            NamedArtifactSource(
                name=metric_name,
                artifact=artifact_reference(metric),
            )
        )
        existing_names.add(metric_name)
        supplied_roles.add(role)
    return sources


def _qc_sample_columns(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
) -> list[str]:
    requested: list[str] = []
    directed = deps.directions.get("cellQc")
    if isinstance(directed, Mapping):
        sample_column = directed.get("sampleColumn")
        if isinstance(sample_column, str):
            requested.append(sample_column)
    if characterization is not None:
        for record in characterization.coefficients:
            observation_unit = record.get("observationUnit")
            if isinstance(observation_unit, str):
                requested.append(observation_unit)
    requested.extend(deps.htoIdentityColumns)
    available = set(deps.store.cells.columns)
    return list(
        dict.fromkeys(name for name in requested if name in available and name != "I")
    )[:_MAX_QC_SAMPLE_PROFILES]


def _qc_profile_id(
    action: LegacyCellQcAction,
    *,
    driver: tuple[str, CellQcDriverType] | None,
    sample_column: str | None = None,
    sample_artifact: NamedArtifactSource | None = None,
) -> str:
    assay_name, assay_type = driver or ("none", "none")
    suffix = {
        "skip": "skip",
        "globalGaussian": "globalGaussian:0.01:0.99",
        "sampleMad": (
            f"sampleMad:metadata:{sample_column}:3:20"
            if sample_artifact is None
            else (
                f"sampleMad:artifact:{sample_artifact.name}:"
                f"{sample_artifact.artifact.artifactId}:3:20"
            )
        ),
    }[action]
    return f"cellQc:{assay_type}:{assay_name}:{suffix}"


def _registered_qc_profile_id(
    profile: RegisteredCellQcProfile,
    *,
    driver: tuple[str, CellQcDriverType],
    sample_column: str | None,
    sample_artifact: NamedArtifactSource | None,
) -> str:
    if sample_column is not None:
        source = f"metadata:{sample_column}"
    elif sample_artifact is not None:
        source = (
            f"artifact:{sample_artifact.name}:{sample_artifact.artifact.artifactId}"
        )
    else:
        source = "global"
    return f"cellQc:{driver[1]}:{driver[0]}:registered:{profile}:{source}"


def _directed_capture_source(
    deps: ExperimentalContextDependencies,
) -> tuple[str | None, NamedArtifactSource | None, np.ndarray] | None:
    directed_qc = deps.directions.get("cellQc")
    qc_directions = dict(directed_qc) if isinstance(directed_qc, Mapping) else {}
    candidates = [
        deps.directions.get("physicalCaptureColumn"),
        qc_directions.get("physicalCaptureColumn"),
        qc_directions.get("captureColumn"),
    ]
    specified = [value for value in candidates if value is not None]
    if not specified:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in specified):
        raise ValueError("physicalCaptureColumn must be a non-empty string")
    names = list(dict.fromkeys(str(value) for value in specified))
    if len(names) != 1:
        raise ValueError("Conflicting physical capture columns were supplied")
    name = names[0]
    matching_artifacts = [
        source for source in deps.htoIdentityArtifacts if source.name == name
    ]
    if len(matching_artifacts) > 1:
        raise ValueError(f"Physical capture artifact {name!r} is not unique")
    if matching_artifacts:
        source = matching_artifacts[0]
        labels = _resolved_artifact_values(
            deps,
            source,
            expected_kind="hto_identity",
        )
        return None, source, np.asarray(labels)
    if name not in deps.cells.columns:
        raise ValueError(
            f"physicalCaptureColumn {name!r} is not observed metadata or an "
            "exact HTO identity artifact"
        )
    return name, None, np.asarray(deps.cells.fetch(name))


def _directed_pooled_reference_captures(
    deps: ExperimentalContextDependencies,
) -> tuple[str, ...] | None:
    directed_qc = deps.directions.get("cellQc")
    qc_directions = dict(directed_qc) if isinstance(directed_qc, Mapping) else {}
    raw = qc_directions.get(
        "pooledReferenceCaptures",
        deps.directions.get("pooledReferenceCaptures"),
    )
    if raw is None:
        return None
    if not isinstance(raw, list | tuple) or any(
        not isinstance(value, str) or not value.strip() for value in raw
    ):
        raise ValueError("pooledReferenceCaptures must contain non-empty strings")
    references = tuple(str(value) for value in raw)
    if len(references) < 2 or len(references) != len(set(references)):
        raise ValueError(
            "pooledReferenceCaptures must contain at least two unique captures"
        )
    return references


def _provenance_label(value: Any) -> str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str) and not value.strip():
        return None
    return str(value)


def _ordered_labels(values: np.ndarray, mask: np.ndarray) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values[mask]:
        label = _provenance_label(raw)
        if label is None or label in seen:
            continue
        seen.add(label)
        output.append(label)
    return output


def _capture_design_safety(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
    capture_labels: np.ndarray,
    capture: str,
) -> tuple[list[dict[str, Any]], bool, bool]:
    if characterization is None:
        return [], False, False
    active = np.ones(len(capture_labels), dtype=bool)
    normalized = _validated_sample_labels(
        capture_labels,
        active,
        label_name="physical capture labels",
    )
    encoded = np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in normalized
        ],
        dtype=object,
    )
    after = encoded != capture
    safety: list[dict[str, Any]] = []
    for record in characterization.coefficients:
        coefficient = record.get("name")
        observation = record.get("observationUnit")
        independent = record.get("independentUnit")
        if (
            not isinstance(coefficient, str)
            or not isinstance(observation, str)
            or record.get("scope") != "betweenUnit"
            or coefficient not in deps.cells.columns
            or observation not in deps.cells.columns
        ):
            continue
        condition_values = np.asarray(deps.cells.fetch(coefficient), dtype=object)
        observation_values = np.asarray(deps.cells.fetch(observation), dtype=object)
        if (
            condition_values.shape != after.shape
            or observation_values.shape != after.shape
        ):
            raise ValueError("Capture safety columns do not align with cellSelection")
        required_groups = _ordered_labels(condition_values, active)
        remaining_groups = _ordered_labels(condition_values, after)
        preserves_conditions = set(remaining_groups) == set(required_groups)

        observation_counts: list[dict[str, Any]] = []
        independent_counts: list[dict[str, Any]] = []
        independent_values: np.ndarray | None = None
        if isinstance(independent, str):
            if independent not in deps.cells.columns:
                continue
            independent_values = np.asarray(
                deps.cells.fetch(independent),
                dtype=object,
            )
            if independent_values.shape != after.shape:
                raise ValueError(
                    "Capture independent-unit column does not align with cellSelection"
                )
        for group in required_groups:
            group_mask = np.asarray(
                [_provenance_label(value) == group for value in condition_values],
                dtype=bool,
            )
            observation_levels = set(
                _ordered_labels(observation_values, after & group_mask)
            )
            observation_counts.append(
                {"group": group, "count": len(observation_levels)}
            )
            if independent_values is not None:
                independent_levels = set(
                    _ordered_labels(independent_values, after & group_mask)
                )
                independent_counts.append(
                    {"group": group, "count": len(independent_levels)}
                )

        replication_counts = (
            independent_counts if independent_values is not None else observation_counts
        )
        minimum_units = min(
            (int(item["count"]) for item in replication_counts),
            default=0,
        )
        complete_pairs = 0
        incomplete_pairs = 0
        duplicate_pair_groups = 0
        single_group_pairs = 0
        if independent_values is not None:
            pair_groups: dict[str, dict[str, set[str]]] = {}
            for index in np.flatnonzero(after):
                pair = _provenance_label(independent_values[index])
                pair_group = _provenance_label(condition_values[index])
                observation_value = _provenance_label(observation_values[index])
                if pair is None or pair_group is None or observation_value is None:
                    continue
                pair_groups.setdefault(pair, {}).setdefault(pair_group, set()).add(
                    observation_value
                )
            required_set = set(required_groups)
            for groups in pair_groups.values():
                if len(groups) == 1:
                    single_group_pairs += 1
                duplicate_pair_groups += sum(
                    len(observations) > 1 for observations in groups.values()
                )
                if set(groups) == required_set and all(
                    len(observations) == 1 for observations in groups.values()
                ):
                    complete_pairs += 1
                else:
                    incomplete_pairs += 1
        original_pair_design = dict(record.get("pairedCoverage") or {}).get("design")
        pair_structure_safe = (
            True
            if independent_values is None
            else (
                complete_pairs >= 2
                and incomplete_pairs == 0
                and duplicate_pair_groups == 0
            )
            if original_pair_design == "paired"
            else (len(pair_groups) >= 2 and single_group_pairs == len(pair_groups))
            if original_pair_design == "betweenIndependentUnits"
            else False
        )
        preserves_units = (
            preserves_conditions and minimum_units >= 2 and pair_structure_safe
        )
        safety.append(
            {
                "coefficient": coefficient,
                "conditionColumn": coefficient,
                "observationUnit": observation,
                "independentUnit": independent,
                "requiredGroups": required_groups,
                "remainingGroups": remaining_groups,
                "observationUnitsByGroup": observation_counts,
                "independentUnitsByGroup": independent_counts,
                "minimumIndependentUnitsAfterExclusion": minimum_units,
                "completePairsAfterExclusion": complete_pairs,
                "incompletePairsAfterExclusion": incomplete_pairs,
                "duplicatePairGroupsAfterExclusion": duplicate_pair_groups,
                "independentUnitDesign": original_pair_design,
                "preservesConditionCoverage": preserves_conditions,
                "preservesIndependentUnitCoverage": preserves_units,
            }
        )
    return (
        safety,
        bool(safety) and all(item["preservesConditionCoverage"] for item in safety),
        bool(safety)
        and all(item["preservesIndependentUnitCoverage"] for item in safety),
    )


def _capture_source_missingness(
    sources: Sequence[QcMetricSourceEvidence],
    values_by_source: Mapping[str, np.ndarray],
    capture_labels: np.ndarray | None,
) -> list[QcMetricSourceEvidence]:
    if capture_labels is None:
        return list(sources)
    active = np.ones(len(capture_labels), dtype=bool)
    normalized = _validated_sample_labels(
        capture_labels,
        active,
        label_name="physical capture labels",
    )
    captures: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for raw in normalized:
        value = raw.item() if isinstance(raw, np.generic) else raw
        key = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if key in seen:
            continue
        seen.add(key)
        captures.append((key, normalized == value))
    output: list[QcMetricSourceEvidence] = []
    for source in sources:
        values = values_by_source.get(source.sourceId)
        missing_by_capture: dict[str, int] = {}
        if values is not None:
            for capture, mask in captures:
                missing_by_capture[capture] = int((~np.isfinite(values[mask])).sum())
        elif source.missingCells == source.activeCells:
            missing_by_capture = {
                capture: int(mask.sum()) for capture, mask in captures
            }
        output.append(
            source.model_copy(update={"missingCellsByCapture": missing_by_capture})
        )
    return output


def _capture_failure_models(
    projection: RegisteredQcProjection | AutoFilterProjection,
    *,
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
    capture_labels: np.ndarray | None,
    metric_sources: Sequence[QcMetricSourceEvidence],
) -> list[CaptureFailureEvidence]:
    if capture_labels is None:
        return []
    output: list[CaptureFailureEvidence] = []
    source_by_id = {source.sourceId: source for source in metric_sources}
    for comparison in projection.captureComparisons:
        missing_fractions = {
            source_id: (
                source.missingCellsByCapture.get(comparison.capture, 0)
                / comparison.cells
                if comparison.cells
                else 0.0
            )
            for source_id, source in source_by_id.items()
        }
        safety, condition_safe, unit_safe = _capture_design_safety(
            deps,
            characterization,
            capture_labels,
            comparison.capture,
        )
        failure = CaptureFailureEvidence(
            capture=comparison.capture,
            activeCells=comparison.cells,
            retainedCells=comparison.retainedCells or 0,
            retainedFraction=comparison.retainedFraction or 0.0,
            adverseAxes=list(comparison.adverseAxes),
            independentAdverseAxes=comparison.independentAdverseAxes,
            metricMissingFractions=missing_fractions,
            reasons=list(comparison.reasons),
            wholeCaptureFailure=comparison.wholeCaptureFailure,
            conditionAndUnitSafety=safety,
            preservesConditionCoverage=condition_safe,
            preservesIndependentUnitCoverage=unit_safe,
            exclusionEligible=(
                comparison.wholeCaptureFailure and condition_safe and unit_safe
            ),
            evidenceId=(
                f"qcCapture:{comparison.capture}:"
                f"{comparison.independentAdverseAxes}axes"
            ),
        )
        output.append(failure)
    return output


def _registered_profile_evidence(
    projection: RegisteredQcProjection,
    *,
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
    driver: tuple[str, CellQcDriverType],
    active: np.ndarray,
    values_by_attr: dict[str, np.ndarray],
    metadata_attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    metric_sources: list[QcMetricSourceEvidence],
    source_concordance: list[QcSourceConcordance],
    sample_column: str | None,
    sample_artifact: NamedArtifactSource | None,
    capture_column: str | None,
    capture_artifact: NamedArtifactSource | None,
    capture_labels: np.ndarray | None,
    pooled_reference_captures: tuple[str, ...] | None,
    active_cells: int,
    comparison_source: str | None,
) -> CellQcProfileEvidence:
    attributes = list(metadata_attributes)
    metric_artifacts = list(artifact_metrics)
    profile_id = _registered_qc_profile_id(
        projection.profile,
        driver=driver,
        sample_column=sample_column,
        sample_artifact=sample_artifact,
    )
    n_mads = 3.0 if projection.profile == "captureMad3Sensitivity" else 5.0
    action: CellQcAction = (
        "skip" if projection.profile == "retainWithFlags" else "registeredMad"
    )
    parameters: dict[str, Any] = {
        "policyVersion": 1,
        "profile": projection.profile,
        "nMads": n_mads,
        "boundPolicy": {
            "count": {"remove": "lower", "flag": "upper"},
            "feature": {"remove": "lower", "flag": "upper"},
            "mitochondrial": {"remove": "upper", "fixedCutoff": None},
            "diagnostic": {"remove": "none"},
        },
        "resolvedBounds": [threshold.to_dict() for threshold in projection.thresholds],
        "captureSizes": projection.captureSizes,
        "captureComparisons": [
            comparison.to_dict() for comparison in projection.captureComparisons
        ],
        "captureComparisonSource": comparison_source,
        "pooledReferenceCaptures": list(pooled_reference_captures or ()),
    }
    failure_evidence = _capture_failure_models(
        projection,
        deps=deps,
        characterization=characterization,
        capture_labels=capture_labels,
        metric_sources=metric_sources,
    )
    cells = deps.cells if deps.cells is not None else deps.store.cells
    retention_columns: list[str] = []
    if characterization is not None:
        for coefficient in characterization.coefficients:
            for value in (
                coefficient.get("name"),
                coefficient.get("observationUnit"),
                coefficient.get("independentUnit"),
            ):
                if isinstance(value, str) and value in cells.columns:
                    retention_columns.append(value)
    retained_by_column: dict[str, dict[str, int]] = {}
    unsafe_groups: list[str] = []
    retained = np.asarray(projection.keep, dtype=bool) & np.asarray(active, dtype=bool)
    for column in dict.fromkeys(retention_columns):
        labels = np.asarray(cells.fetch(column))
        if labels.shape != retained.shape:
            raise ValueError(
                f"QC retention column {column!r} does not align with cellSelection"
            )
        counts: dict[str, int] = {}
        for raw_label in np.unique(labels[np.asarray(active, dtype=bool)]):
            label = raw_label.item() if isinstance(raw_label, np.generic) else raw_label
            key = label.decode("utf-8") if isinstance(label, bytes) else str(label)
            count = int((retained & (labels == raw_label)).sum())
            counts[key] = count
            if count == 0:
                unsafe_groups.append(f"{column}={key}")
        retained_by_column[column] = counts
    return CellQcProfileEvidence(
        profileId=profile_id,
        action=action,
        registeredProfile=projection.profile,
        driverAssay=driver[0],
        driverAssayType=driver[1],
        sampleColumn=sample_column,
        sampleArtifact=sample_artifact,
        captureColumn=capture_column,
        captureArtifact=capture_artifact,
        attributes=attributes,
        artifactMetrics=metric_artifacts,
        metricSources=metric_sources,
        sourceConcordance=source_concordance,
        parameters=parameters,
        resolvedBounds=parameters["resolvedBounds"],
        activeCells=active_cells,
        retainedCells=projection.retainedCells,
        retainedFraction=(
            projection.retainedCells / active_cells if active_cells else 0.0
        ),
        activeCellsByCapture=projection.captureSizes,
        sampleRetainedCells=projection.retainedByCapture,
        retainedCellsByColumn=retained_by_column,
        unsafeRetentionGroups=sorted(unsafe_groups),
        flaggedCells=projection.flagCounts,
        metricFlaggedCells=projection.metricFlagCounts,
        failedCaptureCandidates=list(projection.failedCaptureCandidates),
        captureFailureEvidence=failure_evidence,
        excludableCaptureCandidates=[
            item.capture for item in failure_evidence if item.exclusionEligible
        ],
        notes=list(projection.warnings),
        evidenceId=f"qcProfile:{profile_id}",
    )


def _registered_qc_profiles(
    deps: ExperimentalContextDependencies,
    *,
    characterization: CovariateCharacterization | None,
    driver: tuple[str, CellQcDriverType],
    active: np.ndarray,
    values_by_attr: dict[str, np.ndarray],
    metadata_attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    metric_sources: list[QcMetricSourceEvidence],
    source_concordance: list[QcSourceConcordance],
    capture: tuple[str | None, NamedArtifactSource | None, np.ndarray] | None = None,
) -> list[CellQcProfileEvidence]:
    if capture is None:
        capture = _directed_capture_source(deps)
    sample_column: str | None = None
    sample_artifact: NamedArtifactSource | None = None
    capture_labels: np.ndarray | None = None
    if capture is not None:
        sample_column, sample_artifact, capture_labels = capture
    if sample_column is not None:
        comparison_source = f"metadata:{sample_column}"
    elif sample_artifact is not None:
        comparison_source = (
            f"artifact:{sample_artifact.name}:{sample_artifact.artifact.artifactId}"
        )
    else:
        comparison_source = None
    pooled_references = _directed_pooled_reference_captures(deps)
    if pooled_references is not None and capture is None:
        raise ValueError(
            "pooledReferenceCaptures requires an explicit physicalCaptureColumn"
        )
    projections = offered_registered_qc_profiles(
        values_by_metric=values_by_attr,
        active=active,
        capture_labels=capture_labels,
        grouping_proven=capture is not None,
        min_cells_per_capture=20,
        pooled_reference_captures=pooled_references,
    )
    profiles: list[CellQcProfileEvidence] = []
    for projection in projections:
        uses_capture = projection.profile in {
            "captureMad5",
            "captureMad3Sensitivity",
            "pooledReferenceMad5",
        }
        profiles.append(
            _registered_profile_evidence(
                projection,
                deps=deps,
                characterization=characterization,
                driver=driver,
                active=active,
                values_by_attr=values_by_attr,
                metadata_attributes=metadata_attributes,
                artifact_metrics=artifact_metrics,
                metric_sources=metric_sources,
                source_concordance=source_concordance,
                sample_column=sample_column if uses_capture else None,
                sample_artifact=sample_artifact if uses_capture else None,
                capture_column=sample_column,
                capture_artifact=sample_artifact,
                capture_labels=capture_labels,
                pooled_reference_captures=(
                    pooled_references
                    if projection.profile == "pooledReferenceMad5"
                    else None
                ),
                active_cells=int(active.sum()),
                comparison_source=comparison_source,
            )
        )
    return profiles


def _global_qc_profile(
    deps: ExperimentalContextDependencies,
    driver: tuple[str, CellQcDriverType],
    active: np.ndarray,
    active_cells: int,
    values_by_attr: dict[str, np.ndarray],
    metadata_attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    attribute_notes: list[str],
    *,
    characterization: CovariateCharacterization | None = None,
    metric_sources: list[QcMetricSourceEvidence] | None = None,
    source_concordance: list[QcSourceConcordance] | None = None,
    capture: tuple[str | None, NamedArtifactSource | None, np.ndarray] | None = None,
) -> CellQcProfileEvidence | None:
    """Build an execution-exact projection of core global auto-filtering."""
    if not values_by_attr:
        return None
    metric_sources = list(metric_sources or [])
    source_concordance = list(source_concordance or [])
    executable_values: dict[str, np.ndarray] = {}
    for name, values in values_by_attr.items():
        selected = np.asarray(values)[active]
        if selected.size and np.all(selected == selected[0]):
            attribute_notes.append(f"Ignored constant QC metric {name!r}")
            continue
        low, high = gaussian_quantile_bounds(selected, 0.01, 0.99)
        if not np.isfinite([low, high]).all():
            attribute_notes.append(
                f"Ignored QC metric {name!r} with non-finite Gaussian bounds"
            )
            continue
        executable_values[name] = values
    if not executable_values:
        return None
    executable_names = set(executable_values)
    metadata_names = set(metadata_attributes)
    metadata_attributes = [
        name for name in metadata_attributes if name in executable_names
    ]
    artifact_metrics = [
        source
        for source in artifact_metrics
        if qc_metric_execution_name(
            source.name,
            artifact_id=source.artifact.artifactId,
            collides_with_metadata=source.name in metadata_names,
        )
        in executable_names
    ]
    metric_sources = [
        source for source in metric_sources if source.executionName in executable_names
    ]
    retained_source_ids = {source.sourceId for source in metric_sources}
    source_concordance = [
        comparison
        for comparison in source_concordance
        if comparison.leftSourceId in retained_source_ids
        and comparison.rightSourceId in retained_source_ids
    ]
    capture_column: str | None = None
    capture_artifact: NamedArtifactSource | None = None
    capture_labels: np.ndarray | None = None
    if capture is not None:
        capture_column, capture_artifact, capture_labels = capture
    try:
        projection = project_auto_filter_profile(
            "globalGaussian",
            values_by_metric=executable_values,
            active=active,
            sample_labels=capture_labels,
            grouping_proven=capture is not None,
        )
    except ValueError as exc:
        attribute_notes.append(f"Global Gaussian QC is not executable: {exc}")
        return None
    profile_id = _qc_profile_id(
        "globalGaussian",
        driver=driver,
    )
    failures = _capture_failure_models(
        projection,
        deps=deps,
        characterization=characterization,
        capture_labels=capture_labels,
        metric_sources=metric_sources,
    )
    return CellQcProfileEvidence(
        profileId=profile_id,
        action="globalGaussian",
        driverAssay=driver[0],
        driverAssayType=driver[1],
        captureColumn=capture_column,
        captureArtifact=capture_artifact,
        attributes=list(metadata_attributes),
        artifactMetrics=list(artifact_metrics),
        metricSources=metric_sources,
        sourceConcordance=source_concordance,
        parameters=projection.parameters,
        resolvedBounds=cast(dict[str, Any], projection.parameters["resolvedBounds"]),
        activeCells=active_cells,
        retainedCells=projection.retainedCells,
        retainedFraction=projection.retainedCells / active_cells,
        activeCellsByCapture=projection.captureSizes,
        sampleRetainedCells=projection.retainedByCapture,
        flaggedCells=projection.flagCounts,
        metricFlaggedCells=projection.metricFlagCounts,
        failedCaptureCandidates=list(projection.failedCaptureCandidates),
        captureFailureEvidence=failures,
        excludableCaptureCandidates=[
            item.capture for item in failures if item.exclusionEligible
        ],
        notes=[*attribute_notes, *projection.warnings],
        evidenceId=f"qcProfile:{profile_id}",
    )


def _sample_qc_profiles(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
    driver: tuple[str, CellQcDriverType],
    active: np.ndarray,
    active_cells: int,
    values_by_attr: dict[str, np.ndarray],
    metadata_attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    metric_sources: list[QcMetricSourceEvidence],
    source_concordance: list[QcSourceConcordance],
    capture: tuple[str | None, NamedArtifactSource | None, np.ndarray] | None,
) -> list[CellQcProfileEvidence]:
    """Build core-parity sample MAD profiles from exact grouping sources."""
    attributes = list(values_by_attr)
    profiles: list[CellQcProfileEvidence] = []
    sample_sources: list[
        tuple[str | None, NamedArtifactSource | None, np.ndarray | None, bool]
    ] = []
    if capture is not None:
        sample_sources.append((*capture[:2], capture[2], True))
    sample_sources.extend(
        (None, source, None, False) for source in deps.htoIdentityArtifacts
    )
    sample_sources.extend(
        (column, None, None, False)
        for column in _qc_sample_columns(deps, characterization)
    )
    seen_sources: set[str] = set()
    for (
        sample_column,
        sample_artifact,
        supplied_labels,
        is_physical_capture,
    ) in sample_sources:
        source_key = (
            f"metadata:{sample_column}"
            if sample_column is not None
            else (
                f"artifact:{sample_artifact.artifact.artifactId}"
                if sample_artifact is not None
                else ""
            )
        )
        if not source_key or source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        if len(seen_sources) > _MAX_QC_SAMPLE_PROFILES:
            break
        if not attributes:
            break
        artifact_labels = (
            supplied_labels
            if supplied_labels is not None
            else None
            if sample_artifact is None
            else _resolved_artifact_values(
                deps,
                sample_artifact,
                expected_kind="hto_identity",
            )
        )
        try:
            sample_labels = (
                np.asarray(supplied_labels)
                if supplied_labels is not None
                else np.asarray(deps.cells.fetch(sample_column))
                if sample_column is not None
                else np.asarray(artifact_labels)
            )
            projection = project_auto_filter_profile(
                "sampleMad",
                values_by_metric=values_by_attr,
                sample_labels=sample_labels,
                active=active,
                grouping_proven=True,
                n_mads=3.0,
                min_cells_per_sample=20,
            )
        except (TypeError, ValueError):
            continue
        profile_id = _qc_profile_id(
            "sampleMad",
            driver=driver,
            sample_column=sample_column,
            sample_artifact=sample_artifact,
        )
        failures = (
            _capture_failure_models(
                projection,
                deps=deps,
                characterization=characterization,
                capture_labels=sample_labels,
                metric_sources=metric_sources,
            )
            if is_physical_capture
            else []
        )
        skip_reasons = cast(
            dict[str, object],
            projection.parameters["skipReasons"],
        )
        profiles.append(
            CellQcProfileEvidence(
                profileId=profile_id,
                action="sampleMad",
                driverAssay=driver[0],
                driverAssayType=driver[1],
                sampleColumn=sample_column,
                sampleArtifact=sample_artifact,
                captureColumn=sample_column if is_physical_capture else None,
                captureArtifact=sample_artifact if is_physical_capture else None,
                attributes=list(metadata_attributes),
                artifactMetrics=list(artifact_metrics),
                metricSources=metric_sources,
                sourceConcordance=source_concordance,
                parameters={
                    "nMads": 3.0,
                    "minCellsPerSample": 20,
                    "nSamples": len(projection.captureSizes),
                    "nSkippedSamples": len(skip_reasons),
                },
                resolvedBounds=cast(
                    dict[str, Any],
                    projection.parameters["resolvedBounds"],
                ),
                activeCells=active_cells,
                retainedCells=projection.retainedCells,
                retainedFraction=projection.retainedCells / active_cells,
                activeCellsByCapture=projection.captureSizes,
                sampleRetainedCells=projection.retainedByCapture,
                flaggedCells=projection.flagCounts,
                metricFlaggedCells=projection.metricFlagCounts,
                failedCaptureCandidates=(
                    list(projection.failedCaptureCandidates)
                    if is_physical_capture
                    else []
                ),
                captureFailureEvidence=failures,
                excludableCaptureCandidates=[
                    item.capture for item in failures if item.exclusionEligible
                ],
                notes=list(projection.warnings),
                evidenceId=f"qcProfile:{profile_id}",
            )
        )
    return profiles


def _offered_qc_profiles(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None = None,
) -> list[CellQcProfileEvidence]:
    """Project bounded QC profiles against the exact shared cell selection."""
    active_cells = _active_cell_count(deps)
    active = np.ones(active_cells, dtype=bool)
    driver = _qc_driver(deps.store)
    driver_assay = driver[0] if driver is not None else None
    driver_type = driver[1] if driver is not None else None
    skip_id = _qc_profile_id(
        "skip",
        driver=driver,
    )
    skip_notes = (
        []
        if driver is not None
        else ["No RNA or ATAC assay is eligible to drive automatic cell QC"]
    )
    registered_only = deps.directions.get("registeredQcOnly") is True
    profiles = (
        []
        if registered_only
        else [
            CellQcProfileEvidence(
                profileId=skip_id,
                action="skip",
                driverAssay=driver_assay,
                driverAssayType=driver_type,
                activeCells=active_cells,
                retainedCells=active_cells,
                retainedFraction=1.0 if active_cells else 0.0,
                notes=skip_notes,
                evidenceId=f"qcProfile:{skip_id}",
            )
        ]
    )
    if driver is None or active_cells == 0:
        if registered_only:
            profiles.append(
                CellQcProfileEvidence(
                    profileId=skip_id,
                    action="skip",
                    driverAssay=driver_assay,
                    driverAssayType=driver_type,
                    activeCells=active_cells,
                    retainedCells=active_cells,
                    retainedFraction=1.0 if active_cells else 0.0,
                    notes=skip_notes,
                    evidenceId=f"qcProfile:{skip_id}",
                )
            )
        deps.qcProfiles = {profile.profileId: profile for profile in profiles}
        return profiles

    (
        values_by_attr,
        valid_metadata_attributes,
        artifact_metrics,
        metric_sources,
        source_concordance,
        attribute_notes,
        values_by_source,
    ) = _qc_metric_sources(deps, driver)
    capture = _directed_capture_source(deps)
    capture_column: str | None = None
    capture_artifact: NamedArtifactSource | None = None
    capture_labels: np.ndarray | None = None
    capture_sizes: dict[str, int] = {}
    if capture is not None:
        capture_column, capture_artifact, capture_labels = capture
        normalized = _validated_sample_labels(
            capture_labels,
            active,
            label_name="physical capture labels",
        )
        for raw in normalized:
            value = raw.item() if isinstance(raw, np.generic) else raw
            key = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            capture_sizes[key] = capture_sizes.get(key, 0) + 1
    metric_sources = _capture_source_missingness(
        metric_sources,
        values_by_source,
        capture_labels,
    )
    deps.qcMetricSources = metric_sources
    deps.qcSourceConcordance = source_concordance
    if not registered_only:
        profiles = [
            CellQcProfileEvidence(
                profileId=skip_id,
                action="skip",
                driverAssay=driver_assay,
                driverAssayType=driver_type,
                captureColumn=capture_column,
                captureArtifact=capture_artifact,
                metricSources=metric_sources,
                sourceConcordance=source_concordance,
                activeCells=active_cells,
                retainedCells=active_cells,
                retainedFraction=1.0,
                activeCellsByCapture=capture_sizes,
                sampleRetainedCells=capture_sizes,
                notes=[*skip_notes, *attribute_notes],
                evidenceId=f"qcProfile:{skip_id}",
            )
        ]

    if not registered_only:
        global_profile = _global_qc_profile(
            deps,
            driver,
            active,
            active_cells,
            values_by_attr,
            valid_metadata_attributes,
            artifact_metrics,
            attribute_notes,
            characterization=characterization,
            metric_sources=metric_sources,
            source_concordance=source_concordance,
            capture=capture,
        )
        if global_profile is not None:
            profiles.append(global_profile)
        profiles.extend(
            _sample_qc_profiles(
                deps,
                characterization,
                driver,
                active,
                active_cells,
                values_by_attr,
                valid_metadata_attributes,
                artifact_metrics,
                metric_sources,
                source_concordance,
                capture,
            )
        )
    profiles.extend(
        _registered_qc_profiles(
            deps,
            characterization=characterization,
            driver=driver,
            active=active,
            values_by_attr=values_by_attr,
            metadata_attributes=valid_metadata_attributes,
            artifact_metrics=artifact_metrics,
            metric_sources=metric_sources,
            source_concordance=source_concordance,
            capture=capture,
        )
    )

    deps.qcProfiles = {profile.profileId: profile for profile in profiles}
    return profiles
