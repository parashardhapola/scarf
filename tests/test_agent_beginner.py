"""Beginner RNA entry point and exact completed-result access."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("pydantic_ai")

from scarf.agent import analyze_rna
from scarf.agent.orchestrator import api
from scarf.agent.orchestrator.models import (
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
    OrchestrationRequestRecord,
    artifact_model_to_ref,
)
from scarf.agent.types import ArtifactReferenceModel


def _completed_result(root: Path) -> AutomatedWorkflowResult:
    result = AutomatedWorkflowResult.get_example()
    assert result.finalAnalysis is not None and result.workflowRun is not None
    final = result.finalAnalysis.model_copy(
        update={
            "handoffId": "",
            "umap": ArtifactReferenceModel(
                assay="RNA", kind="embedding", artifactId="6" * 64
            ),
            "markers": ArtifactReferenceModel(
                assay="RNA", kind="marker_table", artifactId="7" * 64
            ),
        }
    ).with_handoff_id()
    values = result.model_dump(mode="json")
    values.update(
        zarrPath=str(root),
        finalAnalysis=final.model_dump(mode="json"),
        finalHandoffId=final.handoffId,
    )
    values["workflowRun"]["workspace"] = "analysis"
    return AutomatedWorkflowResult.model_validate(values)


def test_analyze_rna_passes_one_request_and_effective_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}
    outcome = AutomatedWorkflowResult(status="abstained", notes=["Missing context"])

    class Orchestrator:
        def __init__(self, model: Any, *, config: AutomatedWorkflowConfig) -> None:
            called.update(model=model, config=config)

        def run(self, request: AutomatedWorkflowRequest) -> AutomatedWorkflowResult:
            called["request"] = request
            return outcome

    monkeypatch.setattr(api, "AgentOrchestrator", Orchestrator)
    model = object()
    result = analyze_rna(
        Path("study.h5ad"),
        model=model,
        study_context="Human blood, one donor.",
        study_objective="Identify stable populations.",
        assay="counts",
        zarr_path=Path("study.zarr"),
        max_candidates=40,
    )
    assert result is outcome
    assert called["model"] is model
    assert called["config"].inputPolicy == "unattended"
    assert called["config"].maxCandidateEvaluations == 40
    request = called["request"]
    assert request.sourcePath == "study.h5ad"
    assert request.zarrPath == "study.zarr"
    assert request.primaryAssay == request.markerAssay == "counts"
    assert request.analysisAssays == ["counts"]
    assert request.ingestDirections == {}


@pytest.mark.parametrize(
    "obsolete",
    [
        "maxCandidateBranches",
        "primaryInitialCandidates",
        "secondaryInitialCandidates",
        "integrationResolutionCandidates",
        "maxGraphAssays",
        "leidenSeeds",
        "clusterSubsamples",
        "clusterSubsampleFraction",
    ],
)
def test_legacy_config_and_saved_requests_fail_explicitly(obsolete: str) -> None:
    old_config = {obsolete: 1}
    with pytest.raises(ValueError, match="Create a new single-RNA workflow"):
        AutomatedWorkflowConfig.model_validate(old_config)
    saved = OrchestrationRequestRecord.get_example().model_dump(mode="json")
    saved["config"].update(old_config)
    with pytest.raises(ValueError, match="cannot be resumed or regenerated"):
        OrchestrationRequestRecord.model_validate(saved)


@pytest.mark.parametrize(
    "field,values",
    [
        ("hvgCandidateCounts", (1,)),
        ("hvgCandidateCounts", (2, 1000)),
        ("pcaCandidateDimensions", (1,)),
        ("graphNeighborCandidates", (1,)),
        ("leidenResolutionCandidates", (float("nan"),)),
        ("leidenResolutionCandidates", (float("inf"),)),
        ("leidenResolutionCandidates", (float("-inf"),)),
    ],
)
def test_config_rejects_impossible_candidates_before_execution(
    field: str, values: tuple[int | float, ...]
) -> None:
    with pytest.raises(ValueError, match=field):
        AutomatedWorkflowConfig.model_validate({field: values})


def test_config_minimum_candidates_meet_the_sequential_planner_contract() -> None:
    from scarf.agent.parameter_tuning.hvg import effective_hvg_candidate_counts
    from scarf.agent.parameter_tuning.sequential import SequentialRnaTuningPlanner

    config = AutomatedWorkflowConfig(
        hvgCandidateCounts=(3,),
        pcaCandidateDimensions=(2,),
        graphNeighborCandidates=(2,),
        leidenResolutionCandidates=(0.25,),
    )
    selected_features = effective_hvg_candidate_counts(3, config.hvgCandidateCounts)
    planner = SequentialRnaTuningPlanner(
        workflow_run_id="minimum-candidates",
        assay="RNA",
        n_cells=3,
        n_features=selected_features[0],
        harmony_authorized=False,
        dimension_candidates=config.pcaCandidateDimensions,
        neighbor_candidates=config.graphNeighborCandidates,
        resolution_candidates=config.leidenResolutionCandidates,
    )
    assert planner.dimensions == (2,)
    assert planner.neighbors == (2,)


@pytest.mark.parametrize(
    "routing",
    [
        {"analysisAssays": ["RNA", "ADT"]},
        {"pairedAssays": ["RNA", "ADT"]},
        {"primaryAssay": "RNA", "analysisAssays": ["counts"]},
        {"primaryAssay": "RNA", "markerAssay": "ADT"},
        {"experimentalDirections": {"hypothesisTesting": {}}},
    ],
)
def test_request_rejects_unsupported_routing(routing: dict[str, Any]) -> None:
    values = AutomatedWorkflowRequest.get_example().model_dump(mode="json")
    with pytest.raises(ValueError):
        AutomatedWorkflowRequest.model_validate({**values, **routing})


def test_result_helpers_reopen_read_only_with_exact_refs_and_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.datastore.datastore as datastore_module

    result = _completed_result(tmp_path)
    original = result.model_dump(mode="json")
    opened: list[tuple[str, dict[str, Any]]] = []
    plotted: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    plot_result, marker_table = object(), object()

    def open_store(path: str, **kwargs: Any) -> Any:
        opened.append((path, kwargs))
        return SimpleNamespace(
            plots=SimpleNamespace(
                embedding=lambda **options: plotted.append(options) or plot_result
            ),
            get_markers=lambda **options: markers.append(options) or marker_table,
        )

    monkeypatch.setattr(datastore_module, "DataStore", open_store)
    assert result.plot_embedding(frame="none") is plot_result
    assert result.get_markers(group_id="2", min_score=0.5) is marker_table
    assert all(path == str(tmp_path) for path, _ in opened)
    assert all(options["zarr_mode"] == "r" for _, options in opened)
    assert all(options["workspace"] == "analysis" for _, options in opened)
    final = result.finalAnalysis
    assert final is not None and final.umap is not None
    assert final.clusters is not None and final.markers is not None
    assert plotted == [
        {
            "layout": artifact_model_to_ref(final.umap),
            "color_by": artifact_model_to_ref(final.clusters),
            "frame": "none",
        }
    ]
    assert markers == [
        {
            "marker": artifact_model_to_ref(final.markers),
            "group_id": "2",
            "min_score": 0.5,
            "min_frac_exp": 0.2,
        }
    ]
    assert result.model_dump(mode="json") == original
    with pytest.raises(ValueError, match="completed analysis layout"):
        result.plot_embedding(layout=artifact_model_to_ref(final.umap))


@pytest.mark.parametrize("method", ["plot_embedding", "get_markers", "report"])
def test_result_helpers_explain_noncompleted_outcome(method: str) -> None:
    result = AutomatedWorkflowResult(notes=["Input file is missing"])
    with pytest.raises(RuntimeError, match="failed at ingest.*Input file is missing"):
        getattr(result, method)()


def test_result_report_reuses_existing_path_and_generates_only_if_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scarf.agent.report.generator as generator

    result = _completed_result(tmp_path)
    assert result.workflowRun is not None
    expected = (
        tmp_path
        / "analysis/agents/runs"
        / result.workflowRun.workflowRunId
        / "report/index.html"
    )
    generated: list[tuple[str, str, str | None]] = []

    def generate(target: str, run_id: str, *, workspace: str | None) -> Path:
        generated.append((target, run_id, workspace))
        expected.parent.mkdir(parents=True)
        expected.write_text("<html>Analysis</html>", encoding="utf-8")
        return expected

    monkeypatch.setattr(generator, "generate_agent_report", generate)
    assert result.report() == expected
    assert result.report() == expected
    assert generated == [(str(tmp_path), result.workflowRun.workflowRunId, "analysis")]


@pytest.mark.parametrize("workspace", [None, "analysis"])
def test_legacy_saved_config_blocks_resume_and_report_without_changing_artifacts(
    tmp_path: Path, workspace: str | None
) -> None:
    import hashlib

    import numpy as np

    from scarf.agent import generate_agent_report
    from scarf.agent import record_io
    from scarf.agent.data_enrichment.contracts import DataEnrichmentReport
    from scarf.agent.orchestrator import AgentOrchestrator, journal
    from scarf.agent.orchestrator.models import (
        AutomatedWorkflowResumeRequest,
        FinalAnalysisHandoff,
        NativeAnalysisHandoff,
    )
    from scarf.agent.persistence.reports import (
        create_agent_workflow,
        finalize_agent_workflow,
        save_agent_report,
    )
    from scarf.agent.persistence.contracts import AgentInvocation
    from scarf.datastore.datastore import DataStore
    from tests.agent_orchestrator_store import create_store

    path = create_store(tmp_path / "legacy.zarr", workspace=workspace)
    store = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        nthreads=2,
        workspace=workspace,
    )
    cells = store.snapshot_cell_selection()
    features = store.select_all_features(from_assay="RNA")
    normalized = store.run_normalization(cells, features)
    original_values = np.asarray(store.load_artifact(normalized)["data"][:])
    workflow = create_agent_workflow(store, workflow_run_id="legacy-config")
    prefix = journal._ensure_orchestration_store(store)
    request = AutomatedWorkflowRequest(
        sourcePath=str(path),
        zarrPath=str(path),
        studyContext="One human RNA sample.",
        studyObjective="Inspect stable cell populations.",
        workspace=workspace,
        primaryAssay="RNA",
    )
    old_config = AutomatedWorkflowConfig().model_dump(mode="json")
    old_config.pop("maxCandidateEvaluations")
    old_config["maxCandidateBranches"] = 24
    payload = OrchestrationRequestRecord(
        workflowRunId=workflow.workflowRunId,
        request=request,
        requestSha256=journal._sha256_model(request),
    ).model_dump(mode="json")
    payload["config"] = old_config
    payload["configSha256"] = hashlib.sha256(
        record_io.canonical_json_bytes(old_config)
    ).hexdigest()
    payload["contentSha256"] = hashlib.sha256(
        record_io.canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "contentSha256"}
        )
    ).hexdigest()
    request_key = journal._request_key(prefix, workflow.workflowRunId)
    original_request = record_io.display_json_bytes(payload)
    journal._write_key_once(store.zw, request_key, original_request)

    message = "start a new workflow.*Older saved request/config shapes"
    with pytest.raises(ValueError, match=message):
        AgentOrchestrator(object()).resume(
            AutomatedWorkflowResumeRequest(
                zarrPath=str(path),
                workflowRunId=workflow.workflowRunId,
                workspace=workspace,
            )
        )
    save_agent_report(
        store,
        workflow.workflowRunId,
        DataEnrichmentReport.get_example(),
        invocation=AgentInvocation(
            agentName="data_enrichment", inputs={"fromAssay": "RNA"}
        ),
    )
    workflow = finalize_agent_workflow(
        store, workflow.workflowRunId, status="completed"
    )
    final = FinalAnalysisHandoff(
        workflowRunId=workflow.workflowRunId,
        primaryAssay="RNA",
        markerAssay="RNA",
        cellSelection=ArtifactReferenceModel.from_artifact_ref(cells),
        nativeAnalyses=[
            NativeAnalysisHandoff(
                assay="RNA",
                normalized=ArtifactReferenceModel.from_artifact_ref(normalized),
            )
        ],
    ).with_handoff_id()
    terminal = AutomatedWorkflowResult(
        status="completed",
        currentStage="analysis_finalization",
        zarrPath=str(path),
        workflowRun=workflow,
        reportReferences=list(workflow.reports),
        finalAnalysis=final,
        finalHandoffId=final.handoffId,
        decisionRunId=workflow.workflowRunId,
    )
    terminal.contentSha256 = journal._record_checksum(terminal)
    journal._persist_terminal_result(store, prefix, workflow, terminal)

    with pytest.raises(ValueError, match=message):
        generate_agent_report(path, workflow.workflowRunId, workspace=workspace)

    reopened = DataStore(
        str(path),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        nthreads=2,
        zarr_mode="r",
        workspace=workspace,
    )
    assert record_io.read_key(reopened.zw, request_key) == original_request
    np.testing.assert_array_equal(
        reopened.load_artifact(normalized)["data"][:], original_values
    )
