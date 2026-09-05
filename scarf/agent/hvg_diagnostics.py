import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import zarr

from ..assay import RNAassay
from ..features.variability import DEFAULT_HVG_BLACKLIST, fit_lowess
from ..storage.arrays import create_zarr_dataset
from ..storage.artifact_writer import (
    ArrayRequirement,
    AttributeRequirement,
    finish_artifact,
    plan_artifact,
    start_artifact,
)
from ..storage.artifacts import (
    ArtifactRef,
    artifact_group,
    fingerprint_array,
    fingerprint_stored_arrays,
)
from ..storage.feature_selection import (
    _feature_selection_plan,
    _feature_selection_values,
    _ordered_feature_ids_fingerprint,
    _write_feature_selection,
    read_feature_selection_indices,
)
from ..storage.selections import (
    read_stored_selection_indices,
    snapshot_run_metadata,
    validate_run_metadata_snapshot,
)
from ..storage.types import as_zarr_array

HVG_CANDIDATE_TARGETS = (1000, 2000, 4000)
_HVG_COMPARISON_EXAMPLE_LIMIT = 8
_HVG_DIAGNOSTIC_ARRAYS = (
    "eligible",
    "global_corrected_variance",
    "recurrence",
    "mean_within_group_rank",
    "ranking",
)
_HVG_DIAGNOSTIC_VERSION = 1


@dataclass(frozen=True, slots=True)
class HvgGroupVariability:
    """One technical group's feature variability, streamed into aggregation."""

    group_id: str
    cell_count: int
    corrected_variance: np.ndarray
    detected_features: np.ndarray


@dataclass(frozen=True, slots=True)
class HvgRanking:
    """Bounded feature-axis output of global or batch-aware HVG ranking."""

    ranking_mode: Literal["global", "batchAware"]
    eligible: np.ndarray
    global_corrected_variance: np.ndarray
    recurrence: np.ndarray
    mean_within_group_rank: np.ndarray
    ranking: np.ndarray
    valid_group_count: int
    candidate_counts: tuple[int, ...]

    @property
    def eligible_feature_count(self) -> int:
        return int(self.eligible.sum())

    def candidate_mask(self, top_n: int) -> np.ndarray:
        """Return one registered nested candidate, never an arbitrary count."""
        if top_n not in self.candidate_counts:
            raise ValueError(
                f"top_n must be one of the registered counts {self.candidate_counts}"
            )
        values = np.zeros(self.eligible.shape, dtype=bool)
        values[self.ranking[:top_n]] = True
        return values


@dataclass(frozen=True, slots=True)
class HvgDefaultFamilyLeakage:
    """Default-family representation within one agent-ranked HVG candidate."""

    family: str
    pattern: str
    inventory_count: int
    scarf_default_selected_count: int
    agent_selected_count: int
    agent_only_count: int
    agent_selected_fraction: float
    examples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HvgSelectionComparison:
    """Overlap between one Scarf-default selection and one agent candidate."""

    ranking_mode: Literal["global", "batchAware"]
    top_n: int
    scarf_default_blacklist: str
    scarf_default_count: int
    agent_count: int
    overlap_count: int
    union_count: int
    scarf_default_only_count: int
    agent_only_count: int
    scarf_default_overlap_fraction: float
    agent_overlap_fraction: float
    jaccard: float
    default_family_leakage: tuple[HvgDefaultFamilyLeakage, ...]


@dataclass(frozen=True, slots=True)
class HvgCandidateArtifact:
    """One persisted candidate with its effective capped feature count."""

    top_n: int
    features: ArtifactRef


@dataclass(frozen=True, slots=True)
class HvgDiagnosticArtifacts:
    """Persisted HVG diagnostic and its registered feature selections."""

    diagnostic: ArtifactRef
    ranking_mode: Literal["global", "batchAware"]
    technical_group_column: str | None
    valid_groups: tuple[str, ...]
    excluded_groups: tuple[str, ...]
    eligible_feature_count: int
    candidates: tuple[HvgCandidateArtifact, ...]


def effective_hvg_candidate_counts(
    eligible_feature_count: int,
    targets: Sequence[int] = HVG_CANDIDATE_TARGETS,
) -> tuple[int, ...]:
    """Cap registered HVG counts by eligibility and remove capped duplicates."""
    if isinstance(eligible_feature_count, bool) or not isinstance(
        eligible_feature_count, int
    ):
        raise TypeError("eligible_feature_count must be an integer")
    if eligible_feature_count < 1:
        raise ValueError("eligible_feature_count must be greater than 0")
    if isinstance(targets, str | bytes):
        raise TypeError("targets must be a sequence of positive integers")
    resolved: list[int] = []
    for target in targets:
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("HVG candidate targets must be integers")
        if target < 1:
            raise ValueError("HVG candidate targets must be greater than 0")
        effective = min(target, eligible_feature_count)
        if effective not in resolved:
            resolved.append(effective)
    if not resolved:
        raise ValueError("At least one HVG candidate target is required")
    return tuple(resolved)


def compare_hvg_ranking_to_default(
    scarf_default_selection: np.ndarray,
    ranking: HvgRanking,
    *,
    feature_names: Sequence[Any],
    default_family_patterns: Mapping[str, str],
    max_examples: int = _HVG_COMPARISON_EXAMPLE_LIMIT,
) -> tuple[HvgSelectionComparison, ...]:
    """Compare registered agent candidates with an exact Scarf-default selection."""
    scarf_default = np.asarray(scarf_default_selection, dtype=bool)
    if scarf_default.ndim != 1:
        raise ValueError("scarf_default_selection must be a one-dimensional mask")
    if ranking.eligible.shape != scarf_default.shape:
        raise ValueError("Scarf-default and agent feature axes must align")
    if isinstance(feature_names, str | bytes):
        raise TypeError("feature_names must be a sequence")
    names = np.asarray(
        ["" if value is None else str(value) for value in feature_names],
        dtype=object,
    )
    if names.shape != scarf_default.shape:
        raise ValueError("feature_names must align with the feature-selection masks")
    if isinstance(max_examples, bool) or not isinstance(max_examples, int):
        raise TypeError("max_examples must be an integer")
    if not 0 <= max_examples <= _HVG_COMPARISON_EXAMPLE_LIMIT:
        raise ValueError(
            f"max_examples must be between 0 and {_HVG_COMPARISON_EXAMPLE_LIMIT}"
        )

    family_masks: list[tuple[str, str, np.ndarray]] = []
    for family, pattern in sorted(default_family_patterns.items()):
        if not isinstance(family, str) or not family:
            raise ValueError("Default-family names must be non-empty strings")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Default-family patterns must be non-empty strings")
        compiled = re.compile(pattern.upper())
        mask = np.fromiter(
            (compiled.match(name.upper()) is not None for name in names),
            dtype=bool,
            count=len(names),
        )
        family_masks.append((family, pattern, mask))

    scarf_default_count = int(scarf_default.sum())
    comparisons: list[HvgSelectionComparison] = []
    for top_n in ranking.candidate_counts:
        agent = ranking.candidate_mask(top_n)
        agent_count = int(agent.sum())
        if agent_count != top_n:
            raise ValueError(
                "Agent rankings must contain distinct indices for every candidate"
            )
        overlap = scarf_default & agent
        union = scarf_default | agent
        overlap_count = int(overlap.sum())
        union_count = int(union.sum())
        leakage: list[HvgDefaultFamilyLeakage] = []
        for family, pattern, family_mask in family_masks:
            selected = agent & family_mask
            selected_names = sorted(
                set(names[selected].tolist()),
                key=lambda value: (value.casefold(), value),
            )
            leakage.append(
                HvgDefaultFamilyLeakage(
                    family=family,
                    pattern=pattern,
                    inventory_count=int(family_mask.sum()),
                    scarf_default_selected_count=int(
                        (scarf_default & family_mask).sum()
                    ),
                    agent_selected_count=int(selected.sum()),
                    agent_only_count=int((selected & ~scarf_default).sum()),
                    agent_selected_fraction=(
                        float(selected.sum()) / agent_count if agent_count else 0.0
                    ),
                    examples=tuple(selected_names[:max_examples]),
                )
            )
        comparisons.append(
            HvgSelectionComparison(
                ranking_mode=ranking.ranking_mode,
                top_n=top_n,
                scarf_default_blacklist=DEFAULT_HVG_BLACKLIST,
                scarf_default_count=scarf_default_count,
                agent_count=agent_count,
                overlap_count=overlap_count,
                union_count=union_count,
                scarf_default_only_count=int((scarf_default & ~agent).sum()),
                agent_only_count=int((agent & ~scarf_default).sum()),
                scarf_default_overlap_fraction=(
                    overlap_count / scarf_default_count if scarf_default_count else 0.0
                ),
                agent_overlap_fraction=(
                    overlap_count / agent_count if agent_count else 0.0
                ),
                jaccard=overlap_count / union_count if union_count else 1.0,
                default_family_leakage=tuple(leakage),
            )
        )
    return tuple(comparisons)


def corrected_variance_from_summary(
    summary: Mapping[str, np.ndarray],
    *,
    n_selected: int,
    n_bins: int,
    lowess_frac: float,
) -> np.ndarray:
    """Derive LOWESS-corrected variability from feature-axis sufficient stats."""
    if isinstance(n_selected, bool) or not isinstance(n_selected, int):
        raise TypeError("n_selected must be an integer")
    if n_selected < 1:
        raise ValueError("n_selected must be greater than 0")
    required = ("normed_tot", "normed_n", "sigmas")
    try:
        normed_tot, normed_n, sigmas = (
            np.asarray(summary[name], dtype=np.float64) for name in required
        )
    except KeyError as exc:
        raise ValueError(f"RNA feature summary is missing {exc.args[0]!r}") from exc
    shape = normed_tot.shape
    if normed_tot.ndim != 1 or normed_n.shape != shape or sigmas.shape != shape:
        raise ValueError("RNA feature-summary arrays must be aligned vectors")
    if not all(np.isfinite(values).all() for values in (normed_tot, normed_n, sigmas)):
        raise ValueError("RNA feature-summary arrays must contain only finite values")

    average = normed_tot / n_selected
    corrected = np.zeros(shape, dtype=np.float64)
    positive = (average > 0) & (sigmas > 0)
    if positive.any():
        corrected[positive] = fit_lowess(
            average[positive],
            sigmas[positive],
            n_bins,
            lowess_frac,
            bin_strategy="adaptive",
        )
    if not np.isfinite(corrected).all() or (corrected < 0).any():
        raise ValueError("Corrected feature variability is invalid")
    return corrected


def aggregate_hvg_rankings(
    global_corrected_variance: np.ndarray,
    eligible_features: np.ndarray,
    group_variability: Iterable[HvgGroupVariability],
    *,
    valid_group_count: int,
    candidate_targets: Sequence[int] = HVG_CANDIDATE_TARGETS,
) -> HvgRanking:
    """Aggregate global and optional batch-aware feature rankings."""
    corrected = np.asarray(global_corrected_variance, dtype=np.float64)
    eligible = np.asarray(eligible_features, dtype=bool)
    if corrected.ndim != 1 or eligible.shape != corrected.shape:
        raise ValueError("Global variability and eligibility must be aligned vectors")
    if not np.isfinite(corrected).all() or (corrected < 0).any():
        raise ValueError("Global corrected variability must be finite and non-negative")
    if isinstance(valid_group_count, bool) or not isinstance(valid_group_count, int):
        raise TypeError("valid_group_count must be an integer")
    if valid_group_count < 0:
        raise ValueError("valid_group_count must be non-negative")
    eligible_count = int(eligible.sum())
    counts = effective_hvg_candidate_counts(eligible_count, candidate_targets)
    indices = np.flatnonzero(eligible)
    global_order = indices[np.lexsort((indices, -corrected[indices]))].astype(
        np.int64, copy=False
    )
    recurrence = np.zeros(corrected.shape, dtype=np.int32)
    mean_rank = np.full(corrected.shape, np.inf, dtype=np.float64)

    if valid_group_count < 2:
        denominator = max(1, len(global_order))
        mean_rank[global_order] = (
            np.arange(1, len(global_order) + 1, dtype=np.float64) / denominator
        )
        return HvgRanking(
            ranking_mode="global",
            eligible=eligible.copy(),
            global_corrected_variance=corrected.copy(),
            recurrence=recurrence,
            mean_within_group_rank=mean_rank,
            ranking=global_order.copy(),
            valid_group_count=valid_group_count,
            candidate_counts=counts,
        )

    rank_sum = np.zeros(corrected.shape, dtype=np.float64)
    broad_count = max(counts)
    received = 0
    for group in group_variability:
        received += 1
        if received > valid_group_count:
            raise ValueError("More group summaries were supplied than declared")
        if not isinstance(group.group_id, str) or not group.group_id:
            raise ValueError("Every valid technical group needs a non-empty ID")
        if isinstance(group.cell_count, bool) or not isinstance(group.cell_count, int):
            raise TypeError("Technical-group cell counts must be integers")
        if group.cell_count < 1:
            raise ValueError("Technical-group cell counts must be greater than 0")
        group_corrected = np.asarray(group.corrected_variance, dtype=np.float64)
        detected = np.asarray(group.detected_features, dtype=bool)
        if (
            group_corrected.shape != corrected.shape
            or detected.shape != corrected.shape
        ):
            raise ValueError("Technical-group feature arrays must align globally")
        if not np.isfinite(group_corrected).all() or (group_corrected < 0).any():
            raise ValueError(
                "Technical-group variability must be finite and non-negative"
            )
        group_candidates = np.flatnonzero(eligible & detected)
        if group_candidates.size == 0:
            raise ValueError(
                f"Valid technical group {group.group_id!r} has no rankable features"
            )
        order = group_candidates[
            np.lexsort((group_candidates, -group_corrected[group_candidates]))
        ]
        selected = order[: min(broad_count, len(order))]
        recurrence[selected] += 1
        rank_sum[selected] += np.arange(1, len(selected) + 1, dtype=np.float64) / len(
            order
        )
    if received != valid_group_count:
        raise ValueError(
            f"Expected {valid_group_count} group summaries but received {received}"
        )
    observed = recurrence > 0
    mean_rank[observed] = rank_sum[observed] / recurrence[observed]
    ranking = indices[
        np.lexsort(
            (
                indices,
                -corrected[indices],
                mean_rank[indices],
                -recurrence[indices],
            )
        )
    ].astype(np.int64, copy=False)
    return HvgRanking(
        ranking_mode="batchAware",
        eligible=eligible.copy(),
        global_corrected_variance=corrected.copy(),
        recurrence=recurrence,
        mean_within_group_rank=mean_rank,
        ranking=ranking.copy(),
        valid_group_count=valid_group_count,
        candidate_counts=counts,
    )


def _group_id(value: Any) -> str:
    native = value.item() if isinstance(value, np.generic) else value
    if isinstance(native, bool):
        return f"bool:{str(native).lower()}"
    if isinstance(native, int):
        return f"int:{native}"
    if isinstance(native, float):
        if not np.isfinite(native):
            raise ValueError("Non-finite technical-group values must be marked missing")
        return f"float:{native.hex()}"
    if isinstance(native, str):
        return f"str:{native}"
    raise TypeError(
        "Technical-group values must be strings, booleans, integers, or floats"
    )


def _technical_groups(
    root: zarr.Group,
    snapshot: ArtifactRef,
    column: str,
    cell_indices: np.ndarray,
    *,
    min_group_cells: int,
) -> tuple[tuple[tuple[str, np.ndarray], ...], tuple[str, ...]]:
    group = validate_run_metadata_snapshot(
        root,
        snapshot,
        axis="cell",
        assay=None,
        table_path="cellData",
        ordered_columns=(column,),
    )
    values_array = as_zarr_array(group[column], name=column)
    values = np.asarray(values_array[cell_indices])
    missing_name = values_array.attrs.get("missing_mask")
    missing = (
        np.asarray(
            as_zarr_array(group[missing_name], name=missing_name)[cell_indices],
            dtype=bool,
        )
        if isinstance(missing_name, str)
        else np.zeros(len(cell_indices), dtype=bool)
    )
    if values.dtype.kind == "f":
        missing |= ~np.isfinite(values)
    grouped: dict[str, list[int]] = {}
    for cell_index, value, is_missing in zip(
        cell_indices,
        values,
        missing,
        strict=True,
    ):
        if is_missing:
            continue
        grouped.setdefault(_group_id(value), []).append(int(cell_index))
    valid: list[tuple[str, np.ndarray]] = []
    excluded: list[str] = []
    for group_id in sorted(grouped):
        indices = grouped[group_id]
        if len(indices) >= min_group_cells:
            valid.append((group_id, np.asarray(indices, dtype=np.int64)))
        else:
            excluded.append(group_id)
    return tuple(valid), tuple(excluded)


def _diagnostic_reuse_validator(
    *,
    n_features: int,
    eligible_count: int,
    ordered_feature_ids_fingerprint: str,
) -> Any:
    def validate(_ref: ArtifactRef, group: zarr.Group) -> bool:
        try:
            if set(group.array_keys()) != set(_HVG_DIAGNOSTIC_ARRAYS):
                return False
            expected = {
                "eligible": ((n_features,), np.dtype(bool)),
                "global_corrected_variance": ((n_features,), np.dtype(np.float64)),
                "recurrence": ((n_features,), np.dtype(np.int32)),
                "mean_within_group_rank": ((n_features,), np.dtype(np.float64)),
                "ranking": ((eligible_count,), np.dtype(np.int64)),
            }
            for name, (shape, dtype) in expected.items():
                array = as_zarr_array(group[name], name=name)
                if array.shape != shape or np.dtype(array.dtype) != dtype:
                    return False
            return group.attrs.get(
                "ordered_feature_ids_fingerprint"
            ) == ordered_feature_ids_fingerprint and group.attrs.get(
                "payload_fingerprint"
            ) == fingerprint_stored_arrays(group, _HVG_DIAGNOSTIC_ARRAYS)
        except (KeyError, TypeError, ValueError):
            return False

    return validate


def _ranking_mode_predicate(
    mode: Literal["global", "batchAware"],
) -> Callable[[Any], bool]:
    return lambda value: value == mode


def _write_hvg_diagnostic(
    root: zarr.Group,
    planned: Any,
    ranking: HvgRanking,
    *,
    ordered_feature_ids_fingerprint: str,
    valid_groups: tuple[str, ...],
    excluded_groups: tuple[str, ...],
) -> None:
    group = start_artifact(root, planned)
    payload = {
        "eligible": np.asarray(ranking.eligible, dtype=bool),
        "global_corrected_variance": np.asarray(
            ranking.global_corrected_variance, dtype=np.float64
        ),
        "recurrence": np.asarray(ranking.recurrence, dtype=np.int32),
        "mean_within_group_rank": np.asarray(
            ranking.mean_within_group_rank, dtype=np.float64
        ),
        "ranking": np.asarray(ranking.ranking, dtype=np.int64),
    }
    for name in _HVG_DIAGNOSTIC_ARRAYS:
        values = payload[name]
        chunks = (min(max(len(values), 1), 100_000),)
        output = create_zarr_dataset(group, name, chunks, values.dtype, values.shape)
        output[:] = values
    group.attrs["ordered_feature_ids_fingerprint"] = ordered_feature_ids_fingerprint
    group.attrs["payload_fingerprint"] = fingerprint_stored_arrays(
        group, _HVG_DIAGNOSTIC_ARRAYS
    )
    group.attrs["ranking_mode"] = ranking.ranking_mode
    group.attrs["valid_groups"] = list(valid_groups)
    group.attrs["excluded_groups"] = list(excluded_groups)
    finish_artifact(group, planned)


def run_hvg_diagnostic_artifacts(
    root: zarr.Group,
    assay: RNAassay,
    *,
    cell_selection: ArtifactRef,
    eligible_features: ArtifactRef,
    all_features: ArtifactRef,
    technical_group_column: str | None,
    min_group_cells: int,
    min_cells: int,
    n_bins: int,
    lowess_frac: float,
    invalidate_cache: bool,
    candidate_targets: Sequence[int] = HVG_CANDIDATE_TARGETS,
) -> tuple[HvgDiagnosticArtifacts, ...]:
    """Run and persist global and eligible technical-group HVG rankings."""
    cell_indices = read_stored_selection_indices(
        root,
        cell_selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    ).astype(np.int64, copy=False)
    if cell_indices.size == 0:
        raise ValueError("cell_selection must select at least one cell")
    n_features = int(assay.feats.N)
    selected_feature_indices = read_feature_selection_indices(
        root,
        assay.name,
        eligible_features,
    )
    eligible_input = np.zeros(n_features, dtype=bool)
    eligible_input[selected_feature_indices] = True
    if not eligible_input.any():
        raise ValueError("eligible_features must select at least one feature")
    if (
        len(
            read_feature_selection_indices(
                root,
                assay.name,
                all_features,
            )
        )
        != n_features
    ):
        raise ValueError("all_features must select the complete feature universe")

    from ..assay.feature_summary import ensure_feature_summary, feature_summary_values

    global_summary_ref = ensure_feature_summary(
        root,
        assay,
        cell_selection,
        invalidate_cache=invalidate_cache,
    )
    global_summary = feature_summary_values(
        root,
        global_summary_ref,
        n_selected=len(cell_indices),
    )
    global_corrected = corrected_variance_from_summary(
        global_summary,
        n_selected=len(cell_indices),
        n_bins=n_bins,
        lowess_frac=lowess_frac,
    )
    detected_global = np.asarray(global_summary["normed_n"], dtype=np.float64)
    eligible = eligible_input & (detected_global >= min_cells)
    eligible_count = int(eligible.sum())
    candidate_counts = effective_hvg_candidate_counts(
        eligible_count,
        candidate_targets,
    )

    technical_snapshot: ArtifactRef | None = None
    valid_group_rows: tuple[tuple[str, np.ndarray], ...] = ()
    excluded_groups: tuple[str, ...] = ()
    if technical_group_column is not None:
        technical_snapshot = snapshot_run_metadata(
            root,
            table_path="cellData",
            id_column="ids",
            columns=(technical_group_column,),
            axis="cell",
            invalidate_cache=invalidate_cache,
        )
        valid_group_rows, excluded_groups = _technical_groups(
            root,
            technical_snapshot,
            technical_group_column,
            cell_indices,
            min_group_cells=min_group_cells,
        )
    valid_groups = tuple(group_id for group_id, _indices in valid_group_rows)
    ordered_feature_ids_fingerprint = _ordered_feature_ids_fingerprint(assay)
    diagnostic_inputs: dict[str, Any] = {
        "cell_selection": cell_selection,
        "eligible_features": eligible_features,
        "global_feature_summary": global_summary_ref,
    }
    if technical_snapshot is not None:
        diagnostic_inputs["technical_group_snapshot"] = technical_snapshot
    feature_indices = np.arange(n_features, dtype=np.int64)
    group_variability: list[HvgGroupVariability] = []
    for group_id, group_cells in valid_group_rows:
        summary = assay._compute_feature_summary(group_cells, feature_indices)
        corrected = corrected_variance_from_summary(
            summary,
            n_selected=len(group_cells),
            n_bins=n_bins,
            lowess_frac=lowess_frac,
        )
        group_variability.append(
            HvgGroupVariability(
                group_id=group_id,
                cell_count=len(group_cells),
                corrected_variance=corrected,
                detected_features=(
                    np.asarray(summary["normed_n"], dtype=np.float64) >= min_cells
                ),
            )
        )
    sensitivity_ranking = aggregate_hvg_rankings(
        global_corrected,
        eligible,
        group_variability,
        valid_group_count=len(valid_group_rows),
        candidate_targets=candidate_targets,
    )
    eligible_indices = np.flatnonzero(eligible)
    global_order = eligible_indices[
        np.lexsort((eligible_indices, -global_corrected[eligible_indices]))
    ].astype(np.int64, copy=False)
    global_ranking = HvgRanking(
        ranking_mode="global",
        eligible=sensitivity_ranking.eligible,
        global_corrected_variance=sensitivity_ranking.global_corrected_variance,
        recurrence=sensitivity_ranking.recurrence,
        mean_within_group_rank=sensitivity_ranking.mean_within_group_rank,
        ranking=global_order,
        valid_group_count=sensitivity_ranking.valid_group_count,
        candidate_counts=sensitivity_ranking.candidate_counts,
    )
    rankings = [global_ranking]
    if sensitivity_ranking.ranking_mode == "batchAware":
        rankings.append(sensitivity_ranking)

    results: list[HvgDiagnosticArtifacts] = []
    for ranking in rankings:
        parameters = {
            "algorithm_version": _HVG_DIAGNOSTIC_VERSION,
            "candidate_counts": list(candidate_counts),
            "min_cells": min_cells,
            "min_group_cells": min_group_cells,
            "n_bins": n_bins,
            "lowess_frac": lowess_frac,
            "ranking_mode": ranking.ranking_mode,
            "technical_group_column": technical_group_column,
        }
        planned = plan_artifact(
            root,
            scope="assay",
            assay=assay.name,
            kind="feature_summary",
            operation="diagnose_hvg_candidates",
            parameters=parameters,
            inputs=diagnostic_inputs,
            execution_options={"nthreads": assay.nthreads},
            invalidate_cache=invalidate_cache,
            required_arrays=(
                ArrayRequirement("eligible", shape=(n_features,), dtype=bool),
                ArrayRequirement(
                    "global_corrected_variance",
                    shape=(n_features,),
                    dtype=np.float64,
                ),
                ArrayRequirement("recurrence", shape=(n_features,), dtype=np.int32),
                ArrayRequirement(
                    "mean_within_group_rank",
                    shape=(n_features,),
                    dtype=np.float64,
                ),
                ArrayRequirement("ranking", shape=(eligible_count,), dtype=np.int64),
            ),
            required_attributes=(
                AttributeRequirement(
                    "ordered_feature_ids_fingerprint", expected_types=(str,)
                ),
                AttributeRequirement("payload_fingerprint", expected_types=(str,)),
                AttributeRequirement(
                    "ranking_mode",
                    expected_types=(str,),
                    predicate=_ranking_mode_predicate(ranking.ranking_mode),
                ),
                AttributeRequirement("valid_groups", expected_types=(list,)),
                AttributeRequirement("excluded_groups", expected_types=(list,)),
            ),
            reuse_validator=_diagnostic_reuse_validator(
                n_features=n_features,
                eligible_count=eligible_count,
                ordered_feature_ids_fingerprint=ordered_feature_ids_fingerprint,
            ),
        )
        if not planned.reused:
            _write_hvg_diagnostic(
                root,
                planned,
                ranking,
                ordered_feature_ids_fingerprint=ordered_feature_ids_fingerprint,
                valid_groups=valid_groups,
                excluded_groups=excluded_groups,
            )
        else:
            diagnostic_group = artifact_group(root, planned.ref)
            ranking = HvgRanking(
                ranking_mode=ranking.ranking_mode,
                eligible=np.asarray(
                    as_zarr_array(diagnostic_group["eligible"], name="eligible")[:],
                    dtype=bool,
                ),
                global_corrected_variance=np.asarray(
                    as_zarr_array(
                        diagnostic_group["global_corrected_variance"],
                        name="global_corrected_variance",
                    )[:],
                    dtype=np.float64,
                ),
                recurrence=np.asarray(
                    as_zarr_array(
                        diagnostic_group["recurrence"],
                        name="recurrence",
                    )[:],
                    dtype=np.int32,
                ),
                mean_within_group_rank=np.asarray(
                    as_zarr_array(
                        diagnostic_group["mean_within_group_rank"],
                        name="mean_within_group_rank",
                    )[:],
                    dtype=np.float64,
                ),
                ranking=np.asarray(
                    as_zarr_array(diagnostic_group["ranking"], name="ranking")[:],
                    dtype=np.int64,
                ),
                valid_group_count=len(valid_groups),
                candidate_counts=candidate_counts,
            )

        candidates: list[HvgCandidateArtifact] = []
        for top_n in candidate_counts:
            values = ranking.candidate_mask(top_n)
            values_fingerprint = fingerprint_array(values)
            selection_plan = _feature_selection_plan(
                root,
                assay=assay.name,
                n_features=n_features,
                ordered_feature_ids_fingerprint=ordered_feature_ids_fingerprint,
                operation="set_feature_selection",
                parameters={"values_fingerprint": values_fingerprint},
                inputs={
                    "all_features": all_features,
                },
                execution_options={"invalidate_cache": invalidate_cache},
                expected_payload_fingerprint=values_fingerprint,
                invalidate_cache=invalidate_cache,
            )
            if selection_plan.reused:
                stored = np.asarray(
                    _feature_selection_values(root, selection_plan.ref), dtype=bool
                )
                if not np.array_equal(stored, values):
                    selection_plan = selection_plan.invalidated(root)
            _write_feature_selection(
                root,
                selection_plan,
                ordered_feature_ids_fingerprint=ordered_feature_ids_fingerprint,
                payload={"values": values},
            )
            candidates.append(HvgCandidateArtifact(top_n, selection_plan.ref))

        results.append(
            HvgDiagnosticArtifacts(
                diagnostic=planned.ref,
                ranking_mode=ranking.ranking_mode,
                technical_group_column=technical_group_column,
                valid_groups=valid_groups,
                excluded_groups=excluded_groups,
                eligible_feature_count=eligible_count,
                candidates=tuple(candidates),
            )
        )
    return tuple(results)


__all__ = [
    "HVG_CANDIDATE_TARGETS",
    "HvgCandidateArtifact",
    "HvgDefaultFamilyLeakage",
    "HvgDiagnosticArtifacts",
    "HvgGroupVariability",
    "HvgRanking",
    "HvgSelectionComparison",
    "aggregate_hvg_rankings",
    "compare_hvg_ranking_to_default",
    "corrected_variance_from_summary",
    "effective_hvg_candidate_counts",
    "run_hvg_diagnostic_artifacts",
]
