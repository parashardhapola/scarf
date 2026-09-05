"""Tests for registered one-sided cell-quality profiles."""

import asyncio
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import zarr
from pydantic import ValidationError
from zarr.storage import MemoryStore

import scarf.agent.experimental_context as experimental_context_module
import scarf.agent.orchestrator.preprocessing as preprocessing_module
from scarf.agent.experimental_context import (
    CellQcPlan,
    CellQcProfileEvidence,
    ExperimentalContextResult,
    inspect_cell_covariates,
)
from scarf.agent.orchestrator.main import AgentOrchestrator
from scarf.agent.qc_execution import execute_registered_cell_qc
from scarf.agent.qc_profiles import (
    RegisteredQcProjection,
    offered_registered_qc_profiles,
    project_registered_qc_profile,
)
from scarf.agent.types import ArtifactReferenceModel
from scarf.datastore._operations.quality_control import _QualityControlOperationsMixin
from scarf.storage.artifacts import (
    ArtifactRef,
    artifact_group,
    inspect_artifact,
)
from scarf.storage.selections import (
    read_stored_selection_mask,
    resolve_selection_artifact,
)
from tests.test_agent_experimental_context import _Cells, _Store, _context


def _quality_values() -> dict[str, np.ndarray]:
    return {
        "RNA_nCounts": np.concatenate(
            [np.linspace(80, 120, 39), np.asarray([1.0, 1_000.0, 100.0])]
        ),
        "RNA_nFeatures": np.concatenate(
            [np.linspace(40, 60, 39), np.asarray([1.0, 500.0, 50.0])]
        ),
        "RNA_percentMito": np.concatenate([np.linspace(1, 5, 41), np.asarray([40.0])]),
        "RNA_percentRibo": np.linspace(5, 25, 42),
    }


def _profile_parameters(
    projection: RegisteredQcProjection,
) -> dict[str, Any]:
    return {
        "policyVersion": 1,
        "profile": projection.profile,
        "nMads": 3.0 if projection.profile == "captureMad3Sensitivity" else 5.0,
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
        "captureComparisonSource": None,
        "pooledReferenceCaptures": [],
    }


class _MemoryQcCells:
    def __init__(self, group: zarr.Group) -> None:
        self._group = group
        self.N = int(group["ids"].shape[0])

    @property
    def columns(self) -> list[str]:
        return list(self._group.array_keys())

    def _get_array(self, column: str):
        return self._group[column]

    def fetch_all(self, column: str) -> np.ndarray:
        return np.asarray(self._group[column][:])


class _MemoryQcStore(_QualityControlOperationsMixin):
    def __init__(self, root: zarr.Group) -> None:
        self.zw = root
        self.cells = _MemoryQcCells(root["cellData"])

    def snapshot_cell_selection(self, column: str = "I") -> ArtifactRef:
        values = np.asarray(self.cells.fetch_all(column), dtype=bool)
        return resolve_selection_artifact(
            self.zw,
            scope="datastore",
            kind="cell_selection",
            values=values,
            row_ids=self.cells.fetch_all("ids"),
            operation="snapshot_cell_selection",
            parameters={"column": column},
            inputs={},
            source_column=column,
        )

    def inspect_artifact(self, ref: ArtifactRef):
        return inspect_artifact(self.zw, ref)


def _memory_qc_store(
    values_by_metric: dict[str, np.ndarray],
) -> tuple[_MemoryQcStore, ArtifactRef]:
    first = next(iter(values_by_metric.values()))
    n_cells = len(first)
    if any(len(values) != n_cells for values in values_by_metric.values()):
        raise ValueError("Memory QC metrics must have equal lengths")
    root = zarr.open_group(store=MemoryStore(), mode="w")
    cell_data = root.create_group("cellData")
    cell_data.create_array(
        "ids",
        data=np.asarray([f"cell-{index}" for index in range(n_cells)]),
    )
    cell_data.create_array("I", data=np.ones(n_cells, dtype=bool))
    for name, values in values_by_metric.items():
        cell_data.create_array(name, data=np.asarray(values))
    store = _MemoryQcStore(root)
    return store, store.snapshot_cell_selection("I")


def test_global_registered_profile_uses_one_sided_data_derived_bounds() -> None:
    values = _quality_values()
    projection = project_registered_qc_profile(
        "globalMad5",
        values_by_metric=values,
        active=np.ones(42, dtype=bool),
    )

    assert projection.keep[39] == np.False_
    assert projection.keep[40] == np.True_
    assert projection.keep[41] == np.False_
    assert projection.flags["RNA_nCounts:high"][40] == np.True_
    assert projection.flags["RNA_nFeatures:high"][40] == np.True_
    assert "RNA_percentRibo:highMito" not in projection.flags
    assert not any(
        threshold.metric == "RNA_percentRibo" for threshold in projection.thresholds
    )

    count_threshold = next(
        threshold
        for threshold in projection.thresholds
        if threshold.metric == "RNA_nCounts"
    )
    mito_threshold = next(
        threshold
        for threshold in projection.thresholds
        if threshold.metric == "RNA_percentMito"
    )
    assert count_threshold.lowerRemoval is not None
    assert count_threshold.upperRemoval is None
    assert count_threshold.upperFlag is not None
    assert mito_threshold.lowerRemoval is None
    assert mito_threshold.upperRemoval is not None
    assert mito_threshold.upperRemoval != pytest.approx(8.0)


def test_retain_with_flags_never_removes_flagged_cells() -> None:
    projection = project_registered_qc_profile(
        "retainWithFlags",
        values_by_metric=_quality_values(),
        active=np.ones(42, dtype=bool),
    )

    assert projection.retainedCells == 42
    assert projection.keep.all()
    assert projection.flags["RNA_nCounts:lowQuality"][39] == np.True_
    assert projection.flags["RNA_nCounts:high"][40] == np.True_
    assert projection.flags["RNA_percentMito:highMito"][41] == np.True_


def test_capture_profiles_require_proof_and_minimum_capture_size() -> None:
    values = _quality_values()
    active = np.ones(42, dtype=bool)
    labels = np.asarray(["a"] * 21 + ["b"] * 21)

    unproven = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=labels,
        grouping_proven=False,
    )
    proven = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=labels,
        grouping_proven=True,
        min_cells_per_capture=20,
    )
    undersized = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=np.asarray(["a"] * 19 + ["b"] * 23),
        grouping_proven=True,
        min_cells_per_capture=20,
    )

    assert [projection.profile for projection in unproven] == [
        "retainWithFlags",
        "globalMad5",
    ]
    assert [projection.profile for projection in proven] == [
        "retainWithFlags",
        "globalMad5",
        "captureMad5",
        "captureMad3Sensitivity",
    ]
    global_projection = next(
        projection for projection in proven if projection.profile == "globalMad5"
    )
    assert {item.capture for item in global_projection.captureComparisons} == {
        "a",
        "b",
    }
    assert [projection.profile for projection in undersized] == [
        "retainWithFlags",
        "globalMad5",
    ]


def test_capture_profiles_surface_adverse_global_capture_comparison() -> None:
    labels = np.asarray(["good-a"] * 20 + ["good-b"] * 20 + ["failed"] * 20)
    values = {
        "RNA_nCounts": np.concatenate(
            [
                np.linspace(90, 110, 20),
                np.linspace(95, 115, 20),
                np.linspace(3, 7, 20),
            ]
        ),
        "RNA_nFeatures": np.concatenate(
            [
                np.linspace(45, 55, 20),
                np.linspace(48, 58, 20),
                np.linspace(2, 6, 20),
            ]
        ),
        "RNA_percentMito": np.concatenate(
            [np.linspace(1, 3, 20), np.linspace(1, 4, 20), np.linspace(25, 35, 20)]
        ),
    }

    projection = project_registered_qc_profile(
        "captureMad5",
        values_by_metric=values,
        active=np.ones(60, dtype=bool),
        capture_labels=labels,
        grouping_proven=True,
        min_cells_per_capture=20,
    )

    assert projection.captureSizes == {"good-a": 20, "good-b": 20, "failed": 20}
    assert "failed" in projection.failedCaptureCandidates
    failed = next(
        comparison
        for comparison in projection.captureComparisons
        if comparison.capture == "failed"
    )
    assert failed.adverseGlobalOutlier is True
    assert len(failed.reasons) >= 2
    assert projection.retainedByCapture["failed"] > 0


def test_pooled_reference_profile_requires_explicit_eligible_captures() -> None:
    values = _quality_values()
    active = np.ones(42, dtype=bool)
    labels = np.asarray(["small-a"] * 10 + ["small-b"] * 10 + ["large"] * 22)

    without_reference = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=labels,
        grouping_proven=True,
        min_cells_per_capture=20,
    )
    with_reference = offered_registered_qc_profiles(
        values_by_metric=values,
        active=active,
        capture_labels=labels,
        grouping_proven=True,
        min_cells_per_capture=20,
        pooled_reference_captures=("small-a", "small-b"),
    )

    assert "pooledReferenceMad5" not in {
        projection.profile for projection in without_reference
    }
    pooled = next(
        projection
        for projection in with_reference
        if projection.profile == "pooledReferenceMad5"
    )
    assert {threshold.group for threshold in pooled.thresholds} == {"pooledReference"}


def test_experimental_context_offers_and_validates_registered_global_profile() -> None:
    store = _Store()
    store.cells._values["RNA_nCounts"] = np.linspace(10, 100, 12)
    store.cells._values["RNA_nFeatures"] = np.linspace(5, 50, 12)
    context = _context(
        store,
        directions={"cellQc": {"registeredProfile": "globalMad5"}},
    )

    inspected = asyncio.run(inspect_cell_covariates(context))
    offered = {
        profile.registeredProfile: profile
        for profile in inspected.qcProfiles
        if profile.registeredProfile is not None
    }

    assert {"retainWithFlags", "globalMad5"}.issubset(offered)
    assert offered["retainWithFlags"].action == "skip"
    assert offered["globalMad5"].action == "registeredMad"
    assert offered["globalMad5"].parameters["boundPolicy"]["count"] == {
        "remove": "lower",
        "flag": "upper",
    }
    assert offered["globalMad5"].parameters["boundPolicy"]["mitochondrial"] == {
        "remove": "upper",
        "fixedCutoff": None,
    }

    selected = experimental_context_module._canonical_cell_qc_plan(
        CellQcPlan(),
        context.deps,
        inspected.characterization,
    )
    assert selected.registeredProfile == "globalMad5"
    assert selected.profileId == offered["globalMad5"].profileId
    assert selected.evidenceIds == [offered["globalMad5"].evidenceId]

    with pytest.raises(ValidationError, match="must use the registeredMad action"):
        CellQcPlan(
            action="globalGaussian",
            registeredProfile="globalMad5",
            profileId=offered["globalMad5"].profileId,
            attributes=offered["globalMad5"].attributes,
        )


def test_experimental_context_does_not_infer_capture_from_observation_unit() -> None:
    store = _Store()
    store.cells._values["RNA_nCounts"] = np.linspace(10, 100, 12)
    store.cells._values["RNA_nFeatures"] = np.linspace(5, 50, 12)
    context = _context(store)

    inspected = asyncio.run(inspect_cell_covariates(context))

    registered = {
        profile.registeredProfile
        for profile in inspected.qcProfiles
        if profile.registeredProfile is not None
    }
    assert registered == {"retainWithFlags", "globalMad5"}


def test_experimental_context_offers_capture_profiles_only_for_explicit_source() -> (
    None
):
    store = _Store()
    n_cells = 60
    store.cells = _Cells(
        {
            "I": np.ones(n_cells, dtype=bool),
            "ids": np.asarray([f"cell-{index}" for index in range(n_cells)]),
            "names": np.asarray([f"cell-{index}" for index in range(n_cells)]),
            "capture": np.asarray(["a"] * 20 + ["b"] * 20 + ["c"] * 20),
            "RNA_nCounts": np.linspace(10, 100, n_cells),
            "RNA_nFeatures": np.linspace(5, 50, n_cells),
        }
    )
    store.zw = zarr.open_group(store=MemoryStore(), mode="w")
    cell_data = store.zw.create_group("cellData")
    cell_data.create_array("ids", data=store.cells._values["ids"].astype("U16"))
    cell_data.create_array("I", data=store.cells._values["I"])
    store.refresh_cell_selection()
    context = _context(
        store,
        directions={
            "physicalCaptureColumn": "capture",
            "cellQc": {"pooledReferenceCaptures": ["a", "b"]},
        },
    )

    inspected = asyncio.run(inspect_cell_covariates(context))

    registered = {
        profile.registeredProfile: profile
        for profile in inspected.qcProfiles
        if profile.registeredProfile is not None
    }
    assert set(registered) == {
        "retainWithFlags",
        "globalMad5",
        "captureMad5",
        "captureMad3Sensitivity",
        "pooledReferenceMad5",
    }
    assert registered["captureMad5"].sampleColumn == "capture"
    assert registered["captureMad5"].parameters["captureSizes"] == {
        "a": 20,
        "b": 20,
        "c": 20,
    }
    assert registered["globalMad5"].parameters["captureComparisonSource"] == (
        "metadata:capture"
    )
    assert registered["pooledReferenceMad5"].parameters["pooledReferenceCaptures"] == [
        "a",
        "b",
    ]


def test_datastore_executes_exact_registered_bounds_and_persists_flags() -> None:
    n_cells = 64
    counts = np.linspace(80.0, 120.0, n_cells)
    counts[0] = 1.0
    counts[-1] = 10_000.0
    mito = np.linspace(1.0, 5.0, n_cells)
    mito[-2] = 80.0
    store, source = _memory_qc_store({"RNA_nCounts": counts, "RNA_percentMito": mito})
    active = np.ones(n_cells, dtype=bool)
    projection = project_registered_qc_profile(
        "globalMad5",
        values_by_metric={
            "RNA_nCounts": counts,
            "RNA_percentMito": mito,
        },
        active=active,
    )
    parameters = _profile_parameters(projection)
    live_before = np.asarray(store.cells.fetch_all("I"), dtype=bool).copy()

    selected, flags = execute_registered_cell_qc(
        store,
        "globalMad5",
        profile_parameters=parameters,
        expected_active_cells=n_cells,
        expected_retained_cells=projection.retainedCells,
        expected_flag_counts=projection.flagCounts,
        attrs=["RNA_nCounts", "RNA_percentMito"],
        cell_selection=source,
    )

    stored_selection = read_stored_selection_mask(
        store.zw,
        selected,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    np.testing.assert_array_equal(stored_selection, projection.keep)
    np.testing.assert_array_equal(store.cells.fetch_all("I"), live_before)
    assert flags is not None
    selection_status = store.inspect_artifact(selected)
    assert selection_status.operation == "run_registered_cell_qc"
    assert selection_status.parameters["profileParameters"] == parameters
    assert selection_status.inputs["diagnostic_flags"] == flags.to_dict()
    flag_status = store.inspect_artifact(flags)
    flag_names = flag_status.parameters["flagNames"]
    flag_values = np.asarray(artifact_group(store.zw, flags)["values"][:], dtype=bool)
    assert flag_values.shape == (n_cells, len(flag_names))
    assert {
        name: int(flag_values[:, index].sum()) for index, name in enumerate(flag_names)
    } == projection.flagCounts
    assert projection.keep[-1]
    assert projection.flags["RNA_nCounts:high"][-1]
    assert not projection.keep[-2]

    repeated, repeated_flags = execute_registered_cell_qc(
        store,
        "globalMad5",
        profile_parameters=parameters,
        expected_active_cells=n_cells,
        expected_retained_cells=projection.retainedCells,
        expected_flag_counts=projection.flagCounts,
        attrs=["RNA_nCounts", "RNA_percentMito"],
        cell_selection=source,
    )
    assert repeated == selected
    assert repeated_flags == flags


def test_datastore_rejects_modified_registered_bounds() -> None:
    counts = np.concatenate([np.linspace(80.0, 120.0, 39), np.asarray([1.0])])
    store, source = _memory_qc_store({"RNA_nCounts": counts})
    active = read_stored_selection_mask(
        store.zw,
        source,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    active_counts = np.asarray(store.cells.fetch_all("RNA_nCounts"), dtype=float)[
        active
    ]
    projection = project_registered_qc_profile(
        "globalMad5",
        values_by_metric={"RNA_nCounts": active_counts},
        active=np.ones(len(active_counts), dtype=bool),
    )
    parameters = deepcopy(_profile_parameters(projection))
    parameters["resolvedBounds"][0]["lowerRemoval"] += 1.0

    with pytest.raises(ValueError, match="resolved bounds do not match"):
        execute_registered_cell_qc(
            store,
            "globalMad5",
            profile_parameters=parameters,
            expected_active_cells=int(active.sum()),
            expected_retained_cells=projection.retainedCells,
            expected_flag_counts=projection.flagCounts,
            attrs=["RNA_nCounts"],
            cell_selection=source,
        )


def test_orchestrator_executes_retain_with_flags_instead_of_plain_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {"RNA_nCounts": np.asarray([1.0, 10.0, 11.0, 100.0])}
    projection = project_registered_qc_profile(
        "retainWithFlags",
        values_by_metric=values,
        active=np.ones(4, dtype=bool),
    )
    parameters = _profile_parameters(projection)
    profile = CellQcProfileEvidence(
        profileId="registered-retain",
        action="skip",
        registeredProfile="retainWithFlags",
        driverAssay="RNA",
        driverAssayType="RNA",
        attributes=["RNA_nCounts"],
        parameters=parameters,
        activeCells=4,
        retainedCells=projection.retainedCells,
        retainedFraction=1.0,
        flaggedCells=projection.flagCounts,
        evidenceId="qcProfile:registered-retain",
    )
    plan = CellQcPlan(
        action=profile.action,
        registeredProfile=profile.registeredProfile,
        profileId=profile.profileId,
        driverAssay=profile.driverAssay,
        driverAssayType=profile.driverAssayType,
        attributes=profile.attributes,
        evidenceIds=[profile.evidenceId],
    )
    experimental = ExperimentalContextResult.get_blank().model_copy(
        update={"cellQc": plan, "qcProfiles": [profile]}
    )
    source = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="a" * 64,
    )
    selected = ArtifactRef(
        scope="datastore",
        kind="cell_selection",
        artifact_id="b" * 64,
    )
    flags = ArtifactRef(
        scope="datastore",
        kind="metadata_snapshot",
        artifact_id="c" * 64,
    )
    captured: dict[str, Any] = {}

    class Store:
        pass

    def fake_execute(
        store: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[ArtifactRef, ArtifactRef]:
        captured["store"] = store
        captured["args"] = args
        captured.update(kwargs)
        return selected, flags

    monkeypatch.setattr(
        preprocessing_module,
        "execute_registered_cell_qc",
        fake_execute,
    )

    actions: list[str] = []
    operations: list[dict[str, Any]] = []
    store = Store()
    result = AgentOrchestrator(object()).apply_cell_qc(
        store, experimental, source, actions, operations
    )

    assert result == selected
    assert captured["store"] is store
    assert captured["args"] == ("retainWithFlags",)
    assert captured["profile_parameters"] == parameters
    assert actions == ["cell_qc_registered:retainWithFlags"]
    assert operations[0]["diagnosticFlags"] == (
        ArtifactReferenceModel.from_artifact_ref(flags).model_dump(mode="json")
    )
