"""Public facade, model, and end-to-end orchestrator contracts."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

import scarf.agent as agent_module
import scarf.agent.orchestrator as orchestrator_module
from scarf.agent.biological_interpretation import (
    BiologicalInterpretationReport,
    ClusterCompositionEvidence,
    ClusterInterpretation,
    ClusterMarkerBatchEvidence,
)
from scarf.agent.data_enrichment import (
    DataEnrichmentReport,
    FeatureSelectionPolicy,
    StudyContextSummary,
)
from scarf.agent.decisions.kernel import DecisionSelection
from scarf.agent.persistence.decisions import (
    load_latest_decision_workflow_snapshot,
)
from scarf.agent.experimental_context import (
    BatchCorrectionPlan,
    CellQcPlan,
    CovariateEvidence,
    ExperimentalContextDecision,
)
from scarf.agent.parameter_tuning import ParameterTuningReport
from scarf.agent.persistence import load_agent_record
from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AssayPreprocessingPlan,
    AutomatedPreprocessingPlan,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
    AutomatedWorkflowResumeRequest,
    FinalAnalysisHandoff,
    NativeAnalysisHandoff,
    PreprocessedAssayHandoff,
    WorkflowNeedsInput,
    WorkflowQuestion,
    WorkflowStageAttempt,
    WorkflowStageLink,
    artifact_model_to_ref,
)
from scarf.datastore.datastore import DataStore
from scarf.storage.refs import ArtifactRef
from tests.test_agent_ingest import _write_h5ad


_PLAN_CHECKSUM = "a" * 64


def _rna_workflow_model() -> tuple[FunctionModel, dict[str, int]]:
    state = {
        "enrichment": 0,
        "context": 0,
        "parameter": 0,
        "pca_pauses": 0,
        "pca_prompts": 0,
        "biology": 0,
        "requests": 0,
    }

    def prompt_text(messages: list[ModelMessage]) -> str:
        values: list[str] = []
        for message in messages:
            for part in message.parts:
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    values.append(content)
                elif isinstance(content, tuple):
                    values.extend(item for item in content if isinstance(item, str))
        return "\n".join(values)

    def tool_result(
        messages: list[ModelMessage],
        tool_name: str,
        model_type: Any,
    ) -> Any:
        for message in reversed(messages):
            for part in reversed(message.parts):
                if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                    content = part.content
                    if isinstance(content, model_type):
                        return content
                    if isinstance(content, str):
                        return model_type.model_validate_json(content)
                    return model_type.model_validate(content)
        raise AssertionError(f"Missing tool return {tool_name!r}")

    async def reply(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        state["requests"] += 1
        tools = {tool.name for tool in info.function_tools}
        if "inspect_assay_features_batch" in tools or (
            state["enrichment"] == 1 and "find_present_features_batch" in tools
        ):
            if state["enrichment"] == 0:
                state["enrichment"] = 1
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="inspect_assay_features_batch",
                            args={},
                        )
                    ]
                )
            state["enrichment"] = 2
            report = DataEnrichmentReport(
                status="done",
                studyContextSummary=StudyContextSummary(
                    organismReferences=["human"],
                    tissueReferences=["peripheral blood"],
                ),
                policies=[
                    FeatureSelectionPolicy(
                        assay="RNA",
                        species="unknown",
                        rationale="Use the observed RNA feature inventory.",
                        evidenceIds=["assay:RNA:species"],
                    )
                ],
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args=report.model_dump(),
                    )
                ]
            )

        if tools.intersection(
            {
                "inspect_cell_covariates",
                "analyze_experimental_design",
                "score_current_representation",
            }
        ) or state["context"] in {1, 2}:
            if state["context"] == 0:
                state["context"] = 1
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="inspect_cell_covariates",
                            args={},
                        )
                    ]
                )
            if state["context"] == 1:
                state["context"] = 2
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="analyze_experimental_design",
                            args={
                                "column_domains": {},
                                "coefficients_of_interest": [],
                                "units_of_inference": {},
                                "batch_columns": [],
                            },
                        )
                    ]
                )
            state["context"] = 3
            context_evidence = tool_result(
                messages,
                "analyze_experimental_design",
                CovariateEvidence,
            )
            profile = next(
                value
                for value in context_evidence.qcProfiles
                if value.registeredProfile is not None
            )
            evidence_id = profile.evidenceId
            decision = ExperimentalContextDecision(
                batchCorrection=BatchCorrectionPlan(
                    action="skip",
                    rationale="No trusted technical batch column was supplied.",
                    evidenceIds=[evidence_id],
                ),
                rationale="No experimental covariates were supplied.",
                evidenceIds=[evidence_id],
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args=decision.model_dump(),
                    )
                ]
            )

        if (
            tools.intersection(
                {"inspect_cluster_composition", "inspect_cluster_markers_batch"}
            )
            or state["biology"]
        ):
            if state["biology"] == 0:
                state["biology"] = 1
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="inspect_cluster_composition",
                            args={},
                        )
                    ]
                )
            if state["biology"] == 1:
                composition = tool_result(
                    messages,
                    "inspect_cluster_composition",
                    ClusterCompositionEvidence,
                )
                state["biology"] = 2
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="inspect_cluster_markers_batch",
                            args={"cluster_ids": list(composition.clusterCounts)},
                        )
                    ]
                )
            batch = tool_result(
                messages,
                "inspect_cluster_markers_batch",
                ClusterMarkerBatchEvidence,
            )
            interpretations = []
            for cluster in batch.clusters:
                if cluster.evidenceId and cluster.markers:
                    marker = cluster.markers[0]
                    marker_name = marker.featureName or marker.featureId
                    interpretations.append(
                        ClusterInterpretation(
                            clusterId=cluster.clusterId,
                            proposedIdentity=f"{marker_name}-high RNA state",
                            identityIsHypothesis=True,
                            confidence="low",
                            rationale=(
                                f"The observed marker panel is led by {marker_name}."
                            ),
                            evidenceIds=[cluster.evidenceId],
                        )
                    )
            state["biology"] = 3
            report = BiologicalInterpretationReport(
                status="done",
                clusterInterpretations=interpretations,
                evidenceIds=[item.evidenceIds[0] for item in interpretations],
                limitations=["Synthetic data supports marker-linked hypotheses only."],
                stopReason=(
                    "Every cluster with returned marker evidence was reviewed."
                ),
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args=report.model_dump(),
                    )
                ]
            )

        prompt = prompt_text(messages)
        if any(
            tool.parameters_json_schema.get("title") == "AnalysisVisualAdjudication"
            for tool in info.output_tools
        ):
            payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.output_tools[0].name,
                        args={
                            "status": "acceptable",
                            "selectedCandidateId": payload["selectedCandidateId"],
                            "rationale": (
                                "The bounded diagnostic board agrees with the "
                                "registered numeric evidence."
                            ),
                        },
                    )
                ]
            )
        payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])
        decision = payload
        decision_id = decision["decisionId"]
        if decision_id == "pcaPrefix":
            state["pca_prompts"] += 1
        evidence_by_class: dict[str, str] = {}
        evidence_class_by_id: dict[str, str] = {}
        for item in payload["evidence"]:
            evidence_by_class.setdefault(
                item["evidenceClass"],
                item["evidenceId"],
            )
            evidence_class_by_id[item["evidenceId"]] = item["evidenceClass"]
        preferred = decision.get("metricPreferredOptionId")
        selected = (
            next(
                option for option in decision["options"] if option["status"] == "defer"
            )
            if decision_id == "pcaPrefix" and state["pca_pauses"] == 0
            else next(
                option
                for option in decision["options"]
                if option["optionId"] == preferred
            )
            if preferred is not None
            else next(
                option
                for option in decision["options"]
                if option["status"] in {"apply", "skip"}
            )
        )
        if decision_id == "pcaPrefix" and selected["status"] == "defer":
            state["pca_pauses"] += 1
        evidence_ids = list(selected.get("requiredEvidenceIds", []))
        cited_classes = {
            evidence_class_by_id[evidence_id] for evidence_id in evidence_ids
        }
        for evidence_class in selected["requiredEvidenceClasses"]:
            if evidence_class not in cited_classes:
                evidence_ids.append(evidence_by_class[evidence_class])
        selection = DecisionSelection(
            selectedOptionId=selected["optionId"],
            evidenceIds=evidence_ids,
            rationale="Select the first eligible registered option for this test.",
            confidence="high",
        )
        state["parameter"] += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args=selection.model_dump(),
                )
            ]
        )

    return FunctionModel(reply), state


def test_public_orchestrator_models_have_factories_and_camelcase_fields() -> None:
    models = (
        AssayPreprocessingPlan,
        AutomatedPreprocessingPlan,
        AutomatedWorkflowConfig,
        AutomatedWorkflowRequest,
        AutomatedWorkflowResult,
        AutomatedWorkflowResumeRequest,
        FinalAnalysisHandoff,
        NativeAnalysisHandoff,
        PreprocessedAssayHandoff,
        StudyContextSummary,
        CellQcPlan,
        WorkflowNeedsInput,
        WorkflowQuestion,
        WorkflowStageAttempt,
        WorkflowStageLink,
    )

    for model in models:
        assert isinstance(model.get_blank(), model)
        assert isinstance(model.get_example(), model)
        assert all("_" not in field_name for field_name in model.model_fields)
    for model in (
        AutomatedPreprocessingPlan,
        PreprocessedAssayHandoff,
        FinalAnalysisHandoff,
    ):
        assert "cellSelection" in model.model_fields
        assert "cellKey" not in model.model_fields
    for model in (NativeAnalysisHandoff, FinalAnalysisHandoff):
        assert "clusterColumn" not in model.model_fields
        assert "umapColumns" not in model.model_fields


def test_orchestrator_package_preserves_the_public_facade() -> None:
    assert agent_module.AgentOrchestrator is orchestrator_module.AgentOrchestrator
    assert orchestrator_module.__all__ == [
        "AgentOrchestrator",
        "AssayPreprocessingPlan",
        "AutomatedPreprocessingPlan",
        "AutomatedWorkflowConfig",
        "AutomatedWorkflowRequest",
        "AutomatedWorkflowResult",
        "AutomatedWorkflowResumeRequest",
        "FinalAnalysisHandoff",
        "NativeAnalysisHandoff",
        "PreprocessedAssayHandoff",
        "WorkflowNeedsInput",
        "WorkflowQuestion",
        "WorkflowStageAttempt",
        "WorkflowStageLink",
        "artifact_model_to_ref",
    ]


@pytest.mark.slow
def test_rna_h5ad_completes_public_automated_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scarf.agent.orchestrator import tuning as tuning_module

    rng = np.random.default_rng(4444)
    values = rng.poisson(1.0, size=(80, 50)).astype(np.uint16)
    values[:40, :12] += rng.poisson(9.0, size=(40, 12)).astype(np.uint16)
    values[40:, 12:24] += rng.poisson(9.0, size=(40, 12)).astype(np.uint16)
    feature_names = [
        b"CD3D",
        b"CD3E",
        b"IL7R",
        b"LTB",
        b"MALAT1",
        b"CCR7",
        b"LDHB",
        b"NOSIP",
        b"TCF7",
        b"LEF1",
        b"MAL",
        b"LTST1",
        b"MS4A1",
        b"CD79A",
        b"CD37",
        b"CD74",
        b"HLA-DRA",
        b"CD22",
        b"CD83",
        b"CD19",
        b"BANK1",
        b"CD79B",
        b"CD52",
        b"CD48",
        *[f"GENE{index}".encode() for index in range(26)],
    ]
    source = tmp_path / "small_rna.h5ad"
    target = tmp_path / "small_rna.zarr"
    _write_h5ad(
        source,
        values,
        feature_types=[b"Gene Expression"] * values.shape[1],
        feature_names=feature_names,
    )
    model, state = _rna_workflow_model()
    phase_calls: list[str] = []
    execute_parameter_phase = tuning_module.execute_parameter_phase

    def track_parameter_phase(*args: Any, **kwargs: Any) -> Any:
        phase_calls.append(kwargs["plan"].phase)
        return execute_parameter_phase(*args, **kwargs)

    monkeypatch.setattr(
        tuning_module,
        "execute_parameter_phase",
        track_parameter_phase,
    )
    orchestrator = AgentOrchestrator(
        model,
        config=AutomatedWorkflowConfig(
            primaryInitialCandidates=1,
            secondaryInitialCandidates=1,
            maxRefinedCandidatesPerAssay=0,
            maxHarmonyCandidatesPerAssay=0,
            integrationResolutionCandidates=1,
            maxCandidateBranches=1,
            minClusterCells=2,
        ),
    )

    request = AutomatedWorkflowRequest(
        sourcePath=str(source),
        zarrPath=str(target),
        studyContext=(
            "A human peripheral blood RNA study for a deterministic acceptance test."
        ),
        studyObjective="Discover stable RNA populations.",
        primaryAssay="RNA",
        markerAssay="RNA",
        analysisAssays=["RNA"],
    )
    paused = orchestrator.run(request)

    assert paused.status == "needsInput"
    assert paused.currentStage == "parameter_tuning"
    assert paused.workflowRun is not None
    assert paused.needsInput is not None
    assert len(paused.needsInput.questions) == 1
    question = paused.needsInput.questions[0]
    assert question.questionId == "decision:pcaPrefix"
    assert question.decisionId == "pcaPrefix"
    assert question.options
    selected_option = next(
        option_id for option_id in question.options if option_id != "pcaPrefix:defer"
    )
    pca_calls_before_resume = phase_calls.count("pcaPrefix")
    result = orchestrator.resume(
        AutomatedWorkflowResumeRequest(
            zarrPath=str(target),
            workflowRunId=paused.workflowRun.workflowRunId,
            answers={
                question.questionId: {
                    "decisionId": question.decisionId,
                    "optionId": selected_option,
                    "rationale": "Use the completed registered PCA evidence.",
                }
            },
        )
    )

    assert result.status == "completed", result.notes
    assert phase_calls.count("pcaPrefix") == pca_calls_before_resume
    assert state["pca_prompts"] == 1
    assert result.currentStage == "analysis_finalization"
    assert result.workflowRun is not None
    assert result.workflowRun.status == "completed"
    report_path = (
        target
        / "agents"
        / "runs"
        / result.workflowRun.workflowRunId
        / "report"
        / "index.html"
    )
    assert report_path.is_file()
    assert "Nygen Analytics" in report_path.read_text(encoding="utf-8")
    assert state["requests"] >= 8
    assert state["biology"] == 0
    assert [reference.agentName for reference in result.reportReferences] == [
        "data_enrichment",
        "experimental_context",
        "parameter_tuning",
        "parameter_tuning",
    ]
    assert result.finalAnalysis is not None
    assert result.preprocessingPlan is not None
    assert result.preprocessingPlan.cellQualityPayload is not None
    assert (
        result.preprocessingPlan.cellQualityPayload.profile
        == result.preprocessingPlan.cellQc.registeredProfile
    )
    final = result.finalAnalysis
    assert result.finalHandoffId == final.handoffId
    assert result.decisionRunId == result.workflowRun.workflowRunId
    assert result.verificationSummary
    assert "pipelineRunId" not in result.model_dump()
    assert "pipelineRunId" not in final.model_dump()
    assert final.graphMethod == "native"
    assert final.primaryAssay == final.markerAssay == "RNA"
    assert final.graph is not None
    assert final.clusters is not None
    assert final.cellSelection is not None
    assert final.embeddingInitialization is not None
    assert final.umap is not None
    assert final.markers is not None
    assert len(final.doubletScores) == 1
    assert final.cellSelection.kind == "cell_selection"
    assert final.clusters.kind == "cluster_labels"
    assert final.embeddingInitialization.kind == "embedding_initialization"
    assert final.umap.kind == "embedding"
    assert final.cellSelection != result.preprocessingPlan.cellSelection
    persisted = DataStore(
        str(target),
        default_assay="RNA",
        min_features_per_cell=-1,
        mito_pattern="",
        ribo_pattern="",
        zarr_mode="r",
    )
    tuning_evaluations = [
        evaluation
        for reference in result.reportReferences
        if reference.agentName == "parameter_tuning"
        for evaluation in ParameterTuningReport.model_validate(
            load_agent_record(persisted, reference).report
        ).evaluations
    ]
    pca_evidence = next(
        evaluation
        for evaluation in tuning_evaluations
        if evaluation.metrics.componentVariance
    )
    assert "representationDiagnostic" in pca_evidence.artifacts
    assert any(
        evidence_id.endswith(":pcaLoadings") for evidence_id in pca_evidence.evidenceIds
    )
    cluster_evidence = next(
        evaluation
        for evaluation in tuning_evaluations
        if evaluation.metrics.doubletHighScoreConcentration is not None
    )
    assert cluster_evidence.metrics.markerCoherence is not None
    assert "doubletScore:0" in cluster_evidence.artifacts
    umap_inputs = persisted.inspect_artifact(artifact_model_to_ref(final.umap)).inputs
    assert umap_inputs is not None
    assert umap_inputs["graph"] == artifact_model_to_ref(final.graph).to_dict()
    assert (
        umap_inputs["initialization"]
        == artifact_model_to_ref(final.embeddingInitialization).to_dict()
    )
    marker_inputs = persisted.inspect_artifact(
        artifact_model_to_ref(final.markers)
    ).inputs
    assert marker_inputs is not None
    assert marker_inputs["clusters"] == artifact_model_to_ref(final.clusters).to_dict()
    assert (
        marker_inputs["cell_selection"]
        == artifact_model_to_ref(final.cellSelection).to_dict()
    )
    doublet_inputs = persisted.inspect_artifact(
        artifact_model_to_ref(final.doubletScores[0])
    ).inputs
    assert doublet_inputs is not None
    assert (
        doublet_inputs["connectivity_map"]
        == artifact_model_to_ref(final.graph).to_dict()
    )
    scored_partition = ArtifactRef.from_dict(doublet_inputs["clusters"])
    assert scored_partition.kind == "cluster_labels"
    assert persisted.inspect_artifact(scored_partition).complete
    decision_snapshot = load_latest_decision_workflow_snapshot(
        persisted,
        result.workflowRun.workflowRunId,
    )
    assert decision_snapshot.workflow.status == "completed"
    assert [
        record.decisionId
        for record in decision_snapshot.workflow.active_decision_records()
    ] == [
        "qcGrouping",
        "cellQuality",
        "featurePolicy",
        "hvgRanking",
        "hvgCount",
        "pcaPrefix",
        "correctionLicense",
        "correctionOutcome",
        "graphK",
        "clusterPartition",
    ]
    assert decision_snapshot.workflow.finalHandoffId == final.handoffId
    pca_record = next(
        record
        for record in decision_snapshot.workflow.active_decision_records()
        if record.decisionId == "pcaPrefix"
    )
    assert pca_record.source == "human"
    assert "pipeline" not in persisted.zw
