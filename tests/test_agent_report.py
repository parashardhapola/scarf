"""Supported local HTML report contracts for completed agent workflows."""

import asyncio
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import zarr
from pydantic_ai import ModelRetry, UnexpectedModelBehavior

import scarf.agent as agent_api
import scarf.agent.biological_interpretation.tools as biological_tools
import scarf.agent.biological_interpretation.validation as biological_validation
import scarf.agent.config.agent_exec as agent_exec_module
import scarf.agent.data_enrichment.agent as enrichment_agent
import scarf.agent.data_enrichment.tools as enrichment_tools
import scarf.agent.data_enrichment.validation as enrichment_validation
import scarf.agent.experimental_context.tools as experimental_tools
import scarf.agent.experimental_context.validation as experimental_validation
import scarf.agent.report.artifacts as report_artifacts
import scarf.agent.report.contracts as report_contracts
import scarf.agent.report.decision_tree as report_decision_tree
import scarf.agent.report.generator as report_generator
import scarf.agent.report.plots as report_plots
import scarf.agent.report.rendering as report_rendering
import scarf.agent.orchestrator.journal as journal_module
import scarf.agent.orchestrator.main as orchestrator_main
from scarf.agent import (
    AgentWorkflowRun,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
    FinalAnalysisHandoff,
    create_agent_workflow,
    generate_agent_report,
    list_agent_workflows,
    load_agent_workflow,
)
from scarf.agent.biological_interpretation import (
    BiologicalInterpretationNeedsInput,
    BiologicalInterpretationReport,
    ClusterCompositionEvidence,
    ClusterMarkerEvidence,
)
from scarf.agent.biological_interpretation.contracts import (
    BiologicalInterpretationDependencies,
)
from scarf.agent.experimental_context.contracts import CovariateCharacterization
from scarf.agent.data_enrichment import (
    AssayFeatureInspection,
    DataEnrichmentAgent,
    DataEnrichmentDependencies,
    DataEnrichmentToolCall,
)
from scarf.agent.experimental_context import (
    CellQcProfileEvidence,
    ExperimentalContextDependencies,
)
from scarf.agent.orchestrator.models import (
    AssayPreprocessingPlan,
    AutomatedPreprocessingPlan,
    NativeAnalysisHandoff,
    OrchestrationRequestRecord,
    WorkflowStageAttempt,
)
from scarf.agent.types import ArtifactReferenceModel


def _workflow(
    *,
    workspace: str | None = None,
    status: Literal["completed", "running"] = "completed",
) -> AgentWorkflowRun:
    return AgentWorkflowRun(
        workflowRunId="report-workflow",
        workspace=workspace,
        createdAtNs=1,
        finalizedAtNs=2 if status != "running" else 0,
        status=status,
        finalizationMessage="analysis completed" if status != "running" else "",
        analysisStore="data.zarr",
        datasetFingerprints={"RNA": "dataset-rna"},
    )


def _reports(study_context: str) -> dict[str, list[dict[str, Any]]]:
    run_info = {
        "agentName": "data_enrichment",
        "runId": "provider-run",
        "modelName": "test-model",
        "durationSeconds": 1.5,
        "usage": {
            "requests": 2,
            "toolCalls": 1,
            "inputTokens": 20,
            "outputTokens": 5,
            "totalTokens": 25,
        },
    }
    candidate = {
        "candidateId": "refined",
        "phase": "refined",
        "status": "done",
        "eligible": True,
        "parameters": {
            "reductionMethod": "pca",
            "dimensions": 21,
            "neighborsK": 11,
            "leidenResolution": 0.75,
            "useHarmony": False,
        },
        "metrics": {
            "nClusters": 7,
            "minClusterCells": 42,
            "graphSilhouetteMedian": 0.343,
        },
    }
    baseline_candidate = {
        "candidateId": "baseline",
        "phase": "initial",
        "status": "done",
        "eligible": True,
        "parameters": {
            "reductionMethod": "pca",
            "dimensions": 21,
            "neighborsK": 11,
            "leidenResolution": 1.0,
            "useHarmony": False,
        },
        "metrics": {
            "nClusters": 9,
            "minClusterCells": 18,
            "graphSilhouetteMedian": 0.221,
        },
    }
    return {
        "data_enrichment": [
            {
                "status": "done",
                "studyContextSummary": {
                    "studyContext": study_context,
                    "organismReferences": ["human"],
                    "tissueReferences": ["blood"],
                },
                "policies": [
                    {
                        "assay": "RNA",
                        "excludeFamilies": ["ribosomal"],
                        "protectFamilies": ["sex", "cellCycle"],
                    }
                ],
                "inspections": [
                    {
                        "assay": "RNA",
                        "families": [
                            {
                                "family": "ribosomal",
                                "count": 193,
                                "method": "symbolPrefix",
                                "skipped": None,
                            },
                            {
                                "family": "sex",
                                "count": 0,
                                "method": "chromosome",
                                "skipped": "referenceUnavailable",
                            },
                            {
                                "family": "cellCycle",
                                "count": 94,
                                "method": "staticList",
                                "skipped": None,
                            },
                        ],
                    }
                ],
                "runInfo": run_info,
            }
        ],
        "experimental_context": [
            {
                "status": "done",
                "decision": {"batchCorrection": {"action": "unsafe"}},
                "cellQc": {
                    "action": "skip",
                    "driverAssay": "RNA",
                    "profileId": "qc-selected",
                    "registeredProfile": "retainWithFlags",
                },
                "qcProfiles": [
                    {
                        "profileId": "qc-selected",
                        "registeredProfile": "retainWithFlags",
                        "activeCells": 100,
                        "retainedCells": 100,
                        "retainedFraction": 1.0,
                        "flaggedCells": {
                            "RNA_nCounts:high": 0,
                            "RNA_nCounts:lowQuality": 0,
                            "RNA_nFeatures:high": 0,
                            "RNA_nFeatures:lowQuality": 0,
                        },
                        "parameters": {
                            "nMads": 5.0,
                            "resolvedBounds": [
                                {
                                    "group": "global",
                                    "role": "count",
                                    "lowerRemoval": 50.0,
                                    "upperFlag": 200000.0,
                                },
                                {
                                    "group": "global",
                                    "role": "feature",
                                    "lowerRemoval": 125.0,
                                    "upperFlag": 28000.0,
                                },
                            ],
                        },
                        "retainedCellsByColumn": {
                            "T2D": {"no": 70, "yes": 30},
                            "donor_id": {"donor-a": 45, "donor-b": 55},
                            "sample_id": {"sample-a": 45, "sample-b": 55},
                            "tissue": {"blood": 100},
                        },
                    },
                    {
                        "profileId": "qc-alternative",
                        "registeredProfile": "captureMad5",
                        "activeCells": 100,
                        "retainedCells": 96,
                        "retainedFraction": 0.96,
                        "flaggedCells": {
                            "RNA_nCounts:high": 2,
                            "RNA_nCounts:lowQuality": 1,
                            "RNA_nFeatures:high": 1,
                            "RNA_nFeatures:lowQuality": 3,
                        },
                        "parameters": {
                            "nMads": 5.0,
                            "resolvedBounds": [
                                {
                                    "group": "library-a",
                                    "role": "count",
                                    "lowerRemoval": 40.0,
                                    "upperFlag": 180000.0,
                                },
                                {
                                    "group": "library-b",
                                    "role": "count",
                                    "lowerRemoval": 60.0,
                                    "upperFlag": 220000.0,
                                },
                                {
                                    "group": "library-a",
                                    "role": "feature",
                                    "lowerRemoval": 100.0,
                                    "upperFlag": 24000.0,
                                },
                                {
                                    "group": "library-b",
                                    "role": "feature",
                                    "lowerRemoval": 150.0,
                                    "upperFlag": 32000.0,
                                },
                            ],
                        },
                    },
                ],
                "characterization": {
                    "columns": [
                        {"name": "T2D", "domain": "biological"},
                        {"name": "tissue", "domain": "biological"},
                        {"name": "library_id", "domain": "technical"},
                        {"name": "sample_id", "domain": "design"},
                        {"name": "predicted.id", "domain": "ignore"},
                    ],
                    "coefficients": [
                        {
                            "name": "T2D",
                            "kind": "categorical",
                            "designRows": 22,
                            "observationUnit": "sample_id",
                            "independentUnit": "donor_id",
                            "scope": "betweenUnit",
                        },
                        {
                            "name": "tissue",
                            "kind": "categorical",
                            "designRows": 22,
                            "observationUnit": "sample_id",
                            "independentUnit": "donor_id",
                            "scope": "betweenUnit",
                        },
                    ],
                    "technicalNesting": [
                        {
                            "left": "origin",
                            "right": "library_id",
                            "nesting": "rightInLeft",
                        }
                    ],
                    "confounding": [
                        {
                            "coefficient": "T2D",
                            "pairs": [
                                {
                                    "technical": "library_id",
                                    "selected": True,
                                    "association": {
                                        "status": "notComputed",
                                        "rowsUsed": 22,
                                        "valueUncorrected": 1.0,
                                    },
                                }
                            ],
                        }
                    ],
                },
                "batchSafety": [
                    {
                        "coefficient": "T2D",
                        "status": "unsafe",
                        "estimability": {
                            "coefficientEstimable": False,
                            "rowsUsed": 22,
                            "rankTechnical": 22,
                            "residualDf": 0,
                            "estimableDf": 0,
                        },
                    },
                    {
                        "coefficient": "tissue",
                        "status": "unsafe",
                        "estimability": {
                            "coefficientEstimable": False,
                            "rowsUsed": 22,
                            "rankTechnical": 22,
                            "residualDf": 0,
                            "estimableDf": 0,
                        },
                    },
                ],
            }
        ],
        "parameter_tuning": [
            {
                "status": "done",
                "fromAssay": "RNA",
                "totalCandidates": 2,
                "recommendedByAssay": {"RNA": "refined"},
                "rationale": "The refined candidate balanced cluster viability.",
                "stopReason": "The bounded refinement completed.",
                "assayReports": {
                    "RNA": {
                        "recommendedCandidateId": "refined",
                        "confidence": "medium",
                        "evaluations": [baseline_candidate, candidate],
                        "comparisons": [
                            {
                                "candidateId": "baseline",
                                "summary": (
                                    "The refined candidate retained larger "
                                    "minimum clusters."
                                ),
                                "evidenceIds": ["candidate:refined:clusters"],
                            }
                        ],
                        "searchPlan": {
                            "status": "refine",
                            "objectives": ["Test an intermediate resolution."],
                        },
                    }
                },
            }
        ],
        "biological_interpretation": [
            {
                "status": "done",
                "clusterInterpretations": [
                    {
                        "clusterId": "0",
                        "proposedIdentity": "T cell",
                        "identityIsHypothesis": True,
                    }
                ],
                "treatmentObservations": [],
                "followUps": ["Validate the proposed identities."],
            }
        ],
    }


def _patch_completed_workflow(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    workspace: str | None = None,
    study_context: str = "A human blood study.",
    plots: bool = True,
) -> Path:
    group = zarr.open_group(str(root), mode="w", zarr_format=3)
    if workspace is not None:
        group.create_group(workspace)
    workflow = _workflow(workspace=workspace)
    final = (
        FinalAnalysisHandoff.get_example()
        .model_copy(
            update={
                "workflowRunId": workflow.workflowRunId,
                "handoffId": "",
            }
        )
        .with_handoff_id()
    )
    result = AutomatedWorkflowResult(
        status="completed",
        currentStage="analysis_finalization",
        zarrPath=str(root),
        workflowRun=workflow,
        preprocessingPlan=AutomatedPreprocessingPlan(
            primaryAssay="RNA",
            markerAssay="RNA",
            assays=[
                AssayPreprocessingPlan(
                    assay="RNA",
                    assayType="RNA",
                    role="graph",
                    graphEligible=True,
                    markerEligible=True,
                    featureMethod="hvg",
                    reductionMethod="pca",
                    featureParameters={
                        "topN": 2000,
                        "minCells": 20,
                        "excludeFamilies": ["ribosomal"],
                        "protectFamilies": ["sex", "cellCycle"],
                    },
                    normalizationParameters={
                        "logTransform": True,
                        "renormalizeSubset": True,
                    },
                )
            ],
        ),
        finalAnalysis=final,
        finalHandoffId=final.handoffId,
        decisionRunId=workflow.workflowRunId,
    )
    request = AutomatedWorkflowRequest(
        sourcePath="input.h5ad",
        zarrPath=str(root),
        studyContext=study_context,
        studyObjective="Discover stable RNA populations.",
        workspace=workspace,
    )
    request_record = SimpleNamespace(
        request=request,
        config=AutomatedWorkflowConfig(),
    )

    monkeypatch.setattr(
        report_generator,
        "load_agent_workflow",
        lambda *_a, **_k: workflow,
    )
    monkeypatch.setattr(report_generator, "_open_datastore", lambda *_a, **_k: object())
    monkeypatch.setattr(
        report_generator,
        "_load_completed_result",
        lambda *_a, **_k: ("agents/orchestrations", result, request_record),
    )
    monkeypatch.setattr(
        report_generator,
        "_collect_reports",
        lambda *_a, **_k: _reports(study_context),
    )
    monkeypatch.setattr(
        report_generator,
        "_collect_history",
        lambda *_a, **_k: (
            [
                {
                    "stage": "parameter_tuning",
                    "status": "done",
                    "durationSeconds": 3.5,
                    "actions": ["evaluate_refined_candidate"],
                    "reportCount": 1,
                    "artifactCount": 4,
                    "artifacts": {
                        "selectedGraph": {
                            "scope": "assay",
                            "assay": "RNA",
                            "kind": "connectivity_map",
                            "artifactId": "a" * 64,
                        }
                    },
                    "parentAttempts": ["preprocessing:attempt-1"],
                    "questionIds": [],
                    "noteCount": 0,
                    "errorType": None,
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(report_generator, "_collect_active_decisions", lambda *_a: {})
    monkeypatch.setattr(
        report_generator,
        "_collect_default_feature_inventories",
        lambda *_a: [
            {
                "assay": "RNA",
                "source": "scarfDefaultHvgBlacklist",
                "policyEffect": "evidenceOnly",
                "featureColumn": "names",
                "totalFeatures": 20_000,
                "blacklist": "^MT-|^RPS|^RPL",
                "matchCount": 2,
                "examples": ["MT-CO1", "MT-CYB"],
                "families": [
                    {
                        "family": "mitochondrial",
                        "pattern": "^MT-",
                        "count": 2,
                        "examples": ["MT-CO1", "MT-CYB"],
                    }
                ],
                "appliedToSelectedRepresentation": False,
                "selectedExcludeFamilies": ["ribosomal"],
                "selectedProtectFamilies": ["sex", "cellCycle"],
                "matchedFeatures": ["MT-CO1", "MT-CYB"],
            }
        ],
    )

    def collect_artifacts(
        _store: object,
        _result: AutomatedWorkflowResult,
        plot_dir: Path,
        *,
        qc_profile: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, str], list[str]]:
        assert qc_profile is not None
        if not plots:
            return (
                {"0": 3, "1": 2},
                [],
                {},
                ["umapClusters: ImportError: plotting dependencies are unavailable"],
            )
        plot_dir.mkdir(parents=True, exist_ok=True)
        (plot_dir / "final_umap.png").write_bytes(b"png")
        (plot_dir / "final_umap.png.json").write_text(
            '{"artifact":"umap"}\n', encoding="utf-8"
        )
        return (
            {"0": 3, "1": 2},
            [{"group_id": "0", "feature_name": "CD3D", "score": 8.5}],
            {"umapClusters": "plots/final_umap.png"},
            [],
        )

    monkeypatch.setattr(report_generator, "_collect_final_artifacts", collect_artifacts)

    def collect_hvg_plots(
        _store: object,
        _attempts: object,
        _plan: object,
        plot_dir: Path,
    ) -> tuple[dict[str, str], list[str]]:
        if not plots:
            return {}, []
        (plot_dir / "hvg_global.png").write_bytes(b"hvg")
        (plot_dir / "hvg_global.png.json").write_text(
            '{"artifact":"hvg"}\n',
            encoding="utf-8",
        )
        return {"hvgGlobal": "plots/hvg_global.png"}, []

    monkeypatch.setattr(report_generator, "_collect_hvg_plots", collect_hvg_plots)
    monkeypatch.setattr(
        report_generator,
        "_collect_hvg_evidence",
        lambda *_a, **_k: {
            "assay": "RNA",
            "selectedRankingMode": "batchAware",
            "selectedFeatureCount": 2000,
            "rankings": [
                {
                    "rankingMode": "global",
                    "eligibleFeatureCount": 29263,
                    "validTechnicalGroups": 22,
                    "excludedTechnicalGroupCount": 0,
                    "meanTechnicalGroupCoverage": 0.366,
                    "recurrentInTwoGroupsFraction": 0.630,
                },
                {
                    "rankingMode": "batchAware",
                    "eligibleFeatureCount": 29263,
                    "validTechnicalGroups": 22,
                    "excludedTechnicalGroupCount": 0,
                    "meanTechnicalGroupCoverage": 0.627,
                    "recurrentInTwoGroupsFraction": 1.0,
                },
            ],
            "candidateMetrics": [
                {
                    "featureCount": 1000,
                    "varianceFraction": 0.146,
                    "recurrentFraction": 1.0,
                },
                {
                    "featureCount": 2000,
                    "varianceFraction": 0.190,
                    "recurrentFraction": 1.0,
                },
                {
                    "featureCount": 4000,
                    "varianceFraction": 0.270,
                    "recurrentFraction": 0.655,
                },
            ],
            "eligibleFeatureCount": 29263,
            "validTechnicalGroups": 22,
            "excludedTechnicalGroupCount": 0,
            "minimumDetectedCells": 20,
            "minimumTechnicalGroupCells": 20,
            "scarfDefaultReferenceCounts": [1000, 2000, 4000],
            "executedBranchCount": 9,
        },
    )
    return root


def test_public_report_generates_branded_readable_html_and_relative_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_completed_workflow(
        monkeypatch,
        tmp_path / "data.zarr",
        study_context='Human blood <script>alert("unsafe")</script> & treatment.',
    )
    immutable_record = root / "agents/runs/report-workflow/workflow.json"
    immutable_record.parent.mkdir(parents=True)
    immutable_record.write_bytes(b'{"immutable":true}\n')

    report_path = generate_agent_report(root, "report-workflow")
    analysis_path = report_path
    technical_path = report_path.with_name("technical.html")
    analysis_markup = analysis_path.read_text(encoding="utf-8")
    technical_markup = technical_path.read_text(encoding="utf-8")

    assert agent_api.generate_agent_report is generate_agent_report
    assert report_path == root / "agents/runs/report-workflow/report/index.html"
    assert analysis_path.is_file()
    assert technical_path.is_file()
    assert immutable_record.read_bytes() == b'{"immutable":true}\n'
    for markup in (analysis_markup, technical_markup):
        assert 'href="index.html"' in markup
        assert 'href="analysis.html"' not in markup
        assert 'href="technical.html"' in markup
        assert 'href="https://www.nygen.io/"' in markup
        assert ">Nygen Analytics</a>" in markup
    assert "Choose the level of detail" not in analysis_markup
    assert not report_path.with_name("analysis.html").exists()

    assert "Analysis decision tree" in analysis_markup
    assert '<div class="decision-tree"' in analysis_markup
    assert '<svg class="tree-branch-connectors"' in analysis_markup
    assert "Resolution 0.75" in analysis_markup
    assert "Resolution 1" in analysis_markup
    assert "Separation score: 0.343" in analysis_markup
    assert "Not selected" in analysis_markup
    assert "Evidence behind the decisions" in analysis_markup
    assert "Cell filtering" in analysis_markup
    assert "Covariate analysis" in analysis_markup
    assert "Normalization and feature policy" in analysis_markup
    assert "Harmony and batch correction" in analysis_markup
    assert "Highly variable genes (HVGs)" in analysis_markup
    assert analysis_markup.count('class="evidence-panel"') == 5
    assert analysis_markup.count('<details class="evidence-panel" open>') == 2
    assert analysis_markup.count('<details class="evidence-measurements">') == 5
    assert "2,000 variable genes" in analysis_markup
    assert "Corrected variance captured: 19.0%" in analysis_markup
    assert "Genes recurring across most libraries: 65.5%" in analysis_markup
    assert "Harmony not applied" in analysis_markup
    assert "Selected QC metrics and cutoffs" in analysis_markup
    assert "Exact Scarf default HVG blacklist" in analysis_markup
    assert "Matched 2 of 20,000 genes" in analysis_markup
    assert "How should highly variable genes be ranked?" in analysis_markup
    assert "How many highly variable genes should be used?" in analysis_markup
    assert "HVG branches executed" in analysis_markup
    assert "qc-selected" not in analysis_markup
    assert "library_id" not in analysis_markup
    assert "batchAware" not in analysis_markup
    assert "Why the final result was selected" in analysis_markup
    assert "T cell" in analysis_markup
    assert "Cell QC audit" in technical_markup
    assert "All global and per-library cutoffs" in technical_markup
    assert "Normalization and feature-selection audit" in technical_markup
    assert "All 2 matched feature names" in technical_markup
    assert "MT-CO1" in technical_markup
    assert "provider-run" not in analysis_markup
    assert "a" * 64 not in analysis_markup
    assert "refined" not in analysis_markup
    assert "Plot provenance" not in analysis_markup
    assert 'src="plots/final_umap.png"' in analysis_markup
    assert 'src="plots/hvg_global.png"' in analysis_markup

    assert "Human blood &lt;script&gt;alert" in technical_markup
    assert '<script>alert("unsafe")</script>' not in technical_markup
    assert "Parameter tuning and graph selection" in technical_markup
    assert "refined" in technical_markup
    assert "0.343" in technical_markup
    assert "The refined candidate retained larger minimum clusters." in technical_markup
    assert "Stage artifact inventory" in technical_markup
    assert "connectivity_map" in technical_markup
    assert "Recorded totals" in technical_markup
    assert "evaluate_refined_candidate" in technical_markup
    assert '<body class="technical-page">' in technical_markup
    assert "table-layout: auto" in technical_markup
    assert "table-layout: fixed" not in technical_markup
    assert 'class="record table-record table-record-selected"' in technical_markup
    assert 'src="plots/final_umap.png"' in technical_markup
    assert 'href="plots/final_umap.png.json"' in technical_markup
    assert (report_path.parent / "plots/final_umap.png").read_bytes() == b"png"


def test_harmony_diagnostic_reports_execution_metrics_and_rejection_reason() -> None:
    native = {
        "candidateId": "rna_correction_native",
        "status": "done",
        "eligible": True,
        "parameters": {
            "candidateId": "rna_correction_native",
            "reductionMethod": "pca",
            "dimensions": 20,
            "neighborsK": 21,
            "leidenResolution": 1.0,
            "useHarmony": False,
        },
        "metrics": {
            "batchMixing": {"library_id": 0.05},
            "technicalAssociation": {"library_id": 0.24},
            "biologicalPreservation": {
                "tissue": {"clisi": 1.0, "graphConnectivity": 0.99}
            },
            "crossUnitSupport": 1.0,
            "markerCoherence": 0.82,
            "markerSpecificityMedian": 0.42,
            "clusterConnectivity": 1.0,
            "membershipStrengthMean": 0.96,
            "doubletHighScoreConcentration": 9.4,
        },
    }
    harmony = {
        "candidateId": "rna_correction_harmony",
        "status": "done",
        "eligible": True,
        "parameters": {
            "candidateId": "rna_correction_harmony",
            "reductionMethod": "pca",
            "dimensions": 20,
            "neighborsK": 21,
            "leidenResolution": 1.0,
            "useHarmony": True,
        },
        "metrics": {
            "batchMixing": {"library_id": 0.12},
            "technicalAssociation": {"library_id": 0.09},
            "biologicalPreservation": {
                "tissue": {"clisi": 0.62, "graphConnectivity": 0.98}
            },
            "crossUnitSupport": 1.0,
            "markerCoherence": 0.90,
            "markerSpecificityMedian": 0.51,
            "clusterConnectivity": 1.0,
            "membershipStrengthMean": 0.96,
            "doubletHighScoreConcentration": 8.8,
        },
    }
    parameter = {
        "fromAssay": "RNA",
        "recommendedByAssay": {"RNA": "rna_correction_native"},
        "assayReports": {
            "RNA": {
                "recommendedCandidateId": "rna_correction_native",
                "evaluations": [native, harmony],
            }
        },
    }
    experimental = {
        "decision": {"batchCorrection": {"action": "unsafe"}},
        "batchSafety": [
            {
                "coefficient": "tissue",
                "status": "unsafe",
                "estimability": {
                    "coefficientEstimable": False,
                    "rowsUsed": 22,
                    "rankTechnical": 22,
                    "residualDf": 0,
                    "estimableDf": 0,
                },
            }
        ],
    }
    final = {
        "graphMethod": "native",
        "primaryAssay": "RNA",
        "nativeAnalyses": [{"assay": "RNA", "batchCorrection": None}],
    }
    rejection = (
        "Retain native because Harmony materially degraded protected tissue "
        "evidence and was diagnostic-only."
    )
    decisions = {
        "correctionLicense": {"selectedOptionId": "correctionLicense:unsafeConfounded"},
        "correctionOutcome": {"rationale": rejection},
    }

    evidence_markup = report_rendering._render_batch_evidence(
        experimental,
        parameter,
        final,
        decisions,
    )
    stage = report_decision_tree._batch_tree_stage(
        experimental,
        parameter,
        final,
        decisions,
    )
    assert stage is not None
    tree_markup = report_decision_tree._render_decision_tree([stage])
    technical_markup = report_rendering._render_harmony_technical_audit(
        experimental,
        parameter,
        final,
        decisions,
    )

    assert (
        "Diagnostic Harmony completed; rejected and native representation retained"
        in evidence_markup
    )
    assert "Run status: completed" in evidence_markup
    assert "Library mixing: 0.050 to 0.120 (change +0.070)" in evidence_markup
    assert "Matched native versus Harmony metrics" in evidence_markup
    assert "Recorded correction decision" in evidence_markup
    assert rejection in evidence_markup
    assert "Run diagnostically; rejected" in tree_markup
    assert "Protected evidence degraded: tissue" in tree_markup
    assert "Harmony diagnostic audit" in technical_markup
    assert "Native versus Harmony measurements" in technical_markup


def test_report_uses_workspace_path_and_can_be_regenerated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_completed_workflow(
        monkeypatch,
        tmp_path / "data.zarr",
        workspace="analysis",
        study_context="First context",
    )

    first = generate_agent_report(root, "report-workflow", workspace="analysis")
    assert first == (root / "analysis/agents/runs/report-workflow/report/index.html")
    first_technical = first.with_name("technical.html").read_text(encoding="utf-8")
    assert "First context" in first_technical
    first.write_text("stale landing", encoding="utf-8")
    first.with_name("analysis.html").write_text("stale analysis", encoding="utf-8")
    first.with_name("technical.html").write_text("stale technical", encoding="utf-8")

    monkeypatch.setattr(
        report_generator,
        "_collect_reports",
        lambda *_a, **_k: _reports("Regenerated context"),
    )
    second = generate_agent_report(root, "report-workflow", workspace="analysis")

    assert second == first
    second_analysis = second.read_text(encoding="utf-8")
    assert not second.with_name("analysis.html").exists()
    second_technical = second.with_name("technical.html").read_text(encoding="utf-8")
    assert "Choose the level of detail" not in second_analysis
    assert "Analysis decision tree" in second_analysis
    assert "Regenerated context" in second_technical
    assert second_technical != first_technical


def test_filtering_report_explains_removed_cells_using_the_recorded_rationale() -> None:
    experimental = {
        "qcProfiles": [
            {
                "profileId": "selected-qc",
                "registeredProfile": "globalMad5",
                "activeCells": 100,
                "retainedCells": 90,
            }
        ]
    }
    plan = {
        "cellQc": {
            "profileId": "selected-qc",
            "rationale": "Remove low-quality cells while retaining the study groups.",
        }
    }
    markup = report_rendering._render_filtering_evidence(experimental, plan)
    assert "90 of 100" in markup
    assert "Removed: 10" in markup
    assert plan["cellQc"]["rationale"] in markup
    assert "Preserved every reviewed cell" not in markup


def test_report_remains_available_when_optional_plots_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_completed_workflow(
        monkeypatch,
        tmp_path / "data.zarr",
        plots=False,
    )

    report_path = generate_agent_report(root, "report-workflow")
    analysis_markup = report_path.read_text(encoding="utf-8")
    technical_markup = report_path.with_name("technical.html").read_text(
        encoding="utf-8"
    )

    assert report_path.is_file()
    assert "Visual results are unavailable for this report" in analysis_markup
    assert "No plots could be rendered" in technical_markup
    assert "plotting dependencies are unavailable" in technical_markup
    assert "Final cluster sizes" in technical_markup


def test_report_rejects_remote_and_non_completed_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="local filesystem"):
        generate_agent_report("s3://bucket/data.zarr", "report-workflow")

    root = tmp_path / "data.zarr"
    zarr.open_group(str(root), mode="w", zarr_format=3)
    running = _workflow(status="running")
    monkeypatch.setattr(
        report_generator,
        "load_agent_workflow",
        lambda *_a, **_k: running,
    )
    monkeypatch.setattr(report_generator, "_open_datastore", lambda *_a, **_k: object())

    with pytest.raises(RuntimeError, match="completed workflows"):
        generate_agent_report(root, running.workflowRunId)


def test_orchestrator_generates_only_completed_local_reports_non_fatally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    generated: list[tuple[object, str]] = []
    local_store = SimpleNamespace(z=object())
    completed = _workflow()

    monkeypatch.setattr(
        orchestrator_main,
        "zarr_root_path",
        lambda _store: tmp_path / "data.zarr",
    )
    monkeypatch.setattr(
        report_generator,
        "generate_agent_report",
        lambda target, workflow_run_id: (
            generated.append((target, workflow_run_id))
            or tmp_path / "data.zarr/agents/runs/report-workflow/report/index.html"
        ),
    )

    orchestrator_main._generate_completed_report(local_store, completed)

    assert generated == [(local_store, completed.workflowRunId)]
    assert "Agent workflow report:" in capsys.readouterr().out

    orchestrator_main._generate_completed_report(
        local_store,
        _workflow(status="running"),
    )
    monkeypatch.setattr(orchestrator_main, "zarr_root_path", lambda _store: None)
    orchestrator_main._generate_completed_report(local_store, completed)
    assert len(generated) == 1

    monkeypatch.setattr(
        orchestrator_main,
        "zarr_root_path",
        lambda _store: tmp_path / "data.zarr",
    )
    monkeypatch.setattr(
        report_generator,
        "generate_agent_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )
    orchestrator_main._generate_completed_report(local_store, completed)


def test_derived_report_files_do_not_change_workflow_record_discovery(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(str(tmp_path / "data.zarr"), mode="w", zarr_format=3)
    root.create_group("cellData")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.attrs["dataset_fingerprint"] = "dataset-rna"
    workflow = create_agent_workflow(root, workflow_run_id="report-workflow")
    report_dir = (
        tmp_path / "data.zarr" / "agents" / "runs" / workflow.workflowRunId / "report"
    )
    plot_dir = report_dir / "plots"
    plot_dir.mkdir(parents=True)
    (report_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    (plot_dir / "final_umap.png").write_bytes(b"png")
    (plot_dir / "final_umap.png.json").write_text("{}\n", encoding="utf-8")

    assert load_agent_workflow(root, workflow.workflowRunId) == workflow
    assert list_agent_workflows(root, include_incomplete=True) == [workflow]


def test_report_store_request_and_result_validation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_root = tmp_path / "data.zarr"
    local_root.mkdir()

    class LocalDataStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs
            self.workspace = kwargs.get("workspace")
            self.z = object()

    monkeypatch.setattr(report_artifacts, "DataStore", LocalDataStore)
    monkeypatch.setattr(report_generator, "DataStore", LocalDataStore)
    monkeypatch.setattr(report_artifacts, "zarr_root_path", lambda _store: None)
    with pytest.raises(ValueError, match="local filesystem"):
        report_artifacts._local_root(LocalDataStore())

    monkeypatch.setattr(
        report_artifacts,
        "zarr_root_path",
        lambda _store: local_root,
    )
    assert report_artifacts._local_root(f"file://{local_root}") == local_root.resolve()
    assert report_artifacts._local_root(str(local_root)) == local_root.resolve()
    with pytest.raises(TypeError, match="local filesystem"):
        report_artifacts._local_root(object())
    with pytest.raises(FileNotFoundError):
        report_artifacts._local_root(tmp_path / "missing.zarr")

    workflow = _workflow()
    with pytest.raises(ValueError, match="workspace"):
        report_artifacts._open_datastore(
            LocalDataStore(workspace="other"),
            local_root,
            workflow,
        )
    opened = report_artifacts._open_datastore(local_root, local_root, workflow)
    assert opened.args == (str(local_root),)
    assert opened.kwargs["default_assay"] == "RNA"
    assert opened.kwargs["zarr_mode"] == "r"

    request = AutomatedWorkflowRequest.get_example()
    config = AutomatedWorkflowConfig.get_example()
    valid_record = OrchestrationRequestRecord(
        workflowRunId="workflow-1",
        request=request,
        config=config,
        requestSha256=journal_module._sha256_model(request),
        configSha256=journal_module._sha256_model(config),
    )
    valid_record.contentSha256 = journal_module._record_checksum(valid_record)
    current_record = valid_record
    monkeypatch.setattr(
        journal_module,
        "_read_model",
        lambda *_args, **_kwargs: current_record,
    )
    store = SimpleNamespace(z=object(), zw=object())
    assert report_artifacts._load_request(store, "agents", "workflow-1") == valid_record

    invalid_records = (
        (
            valid_record.model_copy(update={"workflowRunId": "another-workflow"}),
            "another workflow",
        ),
        (
            valid_record.model_copy(update={"requestSha256": "f" * 64}),
            "request checksum",
        ),
        (
            valid_record.model_copy(update={"configSha256": "f" * 64}),
            "configuration checksum",
        ),
        (
            valid_record.model_copy(update={"contentSha256": "f" * 64}),
            "request envelope",
        ),
    )
    for current_record, message in invalid_records:
        with pytest.raises(ValueError, match=message):
            report_artifacts._load_request(store, "agents", "workflow-1")

    monkeypatch.setattr(
        journal_module,
        "_ensure_orchestration_store",
        lambda _store: "agents/orchestrations",
    )
    terminal_result: object | None = None
    monkeypatch.setattr(
        journal_module,
        "_load_terminal_result",
        lambda *_args, **_kwargs: terminal_result,
    )
    with pytest.raises(FileNotFoundError, match="no terminal result"):
        report_artifacts._load_completed_result(store, workflow)

    terminal_result = SimpleNamespace(status="failed", finalAnalysis=object())
    with pytest.raises(ValueError, match="final analysis"):
        report_artifacts._load_completed_result(store, workflow)

    terminal_result = SimpleNamespace(status="completed", finalAnalysis=object())
    monkeypatch.setattr(
        report_artifacts,
        "_load_request",
        lambda *_args, **_kwargs: SimpleNamespace(
            request=SimpleNamespace(workspace="other")
        ),
    )
    with pytest.raises(ValueError, match="request workspace"):
        report_artifacts._load_completed_result(store, workflow)

    invalid_attempt = WorkflowStageAttempt(
        status="failed",
        startedAtNs=1,
        completedAtNs=2,
        error="not a valid error type!?: details",
    )
    assert report_artifacts._stage_summary(invalid_attempt)["errorType"] == (
        "WorkflowStageError"
    )

    data_store = LocalDataStore(workspace="analysis")
    with pytest.raises(ValueError, match="workspace does not match"):
        generate_agent_report(
            data_store,
            "report-workflow",
            workspace="other",
        )


def test_hvg_report_evidence_uses_persisted_diagnostic_values(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(str(tmp_path / "hvg.zarr"), mode="w", zarr_format=3)
    groups: dict[str, Any] = {}

    def diagnostic(
        artifact_id: str,
        mode: str,
        recurrence: list[int],
    ) -> ArtifactReferenceModel:
        group = root.create_group(artifact_id)
        group.attrs["ranking_mode"] = mode
        group.attrs["valid_groups"] = ["library-a", "library-b", "library-c"]
        group.attrs["excluded_groups"] = []
        group.attrs["provenance"] = {
            "parameters": {
                "candidate_counts": [2, 4, 6],
                "min_cells": 20,
                "min_group_cells": 20,
            }
        }
        group.create_array("ranking", data=np.arange(6, dtype=np.int64))
        group.create_array(
            "global_corrected_variance",
            data=np.array([6, 5, 4, 3, 2, 1], dtype=np.float64),
        )
        group.create_array(
            "recurrence",
            data=np.asarray(recurrence, dtype=np.int32),
        )
        group.create_array(
            "eligible",
            data=np.ones(6, dtype=bool),
        )
        groups[artifact_id] = group
        return ArtifactReferenceModel(
            assay="RNA",
            kind="feature_summary",
            artifactId=artifact_id,
        )

    global_ref = diagnostic("a" * 64, "global", [3, 2, 1, 1, 0, 0])
    batch_ref = diagnostic("b" * 64, "batchAware", [3, 3, 3, 2, 1, 1])

    class HvgStore:
        def load_artifact(self, ref: Any) -> Any:
            return groups[ref.artifact_id]

    evidence = report_artifacts._collect_hvg_evidence(
        HvgStore(),
        [
            {
                "artifacts": {
                    "RNA_hvg_global_diagnostic": global_ref.model_dump(mode="json"),
                    "RNA_hvg_batchAware_diagnostic": batch_ref.model_dump(mode="json"),
                    "RNA_hvg_diagnostic": batch_ref.model_dump(mode="json"),
                }
            }
        ],
        {
            "assays": [
                {
                    "assay": "RNA",
                    "featureParameters": {"topN": 4},
                }
            ]
        },
    )

    assert evidence["selectedRankingMode"] == "batchAware"
    assert evidence["selectedFeatureCount"] == 4
    assert evidence["eligibleFeatureCount"] == 6
    assert evidence["validTechnicalGroups"] == 3
    assert evidence["candidateMetrics"] == [
        {
            "featureCount": 2,
            "varianceFraction": 11 / 21,
            "recurrentFraction": 1.0,
        },
        {
            "featureCount": 4,
            "varianceFraction": 18 / 21,
            "recurrentFraction": 1.0,
        },
        {
            "featureCount": 6,
            "varianceFraction": 1.0,
            "recurrentFraction": 4 / 6,
        },
    ]
    assert evidence["rankings"][0]["meanTechnicalGroupCoverage"] == 7 / 18
    assert evidence["rankings"][1]["meanTechnicalGroupCoverage"] == 13 / 18


def test_report_renderer_edge_branches() -> None:
    assert report_plots._safe_assay_name("RNA / strange assay", "fallback") == (
        "rna_strange_assay"
    )
    assert report_plots._safe_assay_name("***", "fallback") == "fallback"
    assert report_contracts._scalar(None) == "Not provided"
    assert "Nothing" in report_rendering._chips(None, empty="Nothing")
    assert "value" in report_rendering._chips("value")
    public_text = report_contracts._brief_text(
        f"Preserve donor_id from {'a' * 64} and 12345678-1234-1234-1234-123456789abc."
    )
    assert "donor_id" not in public_text
    assert "a" * 64 not in public_text
    assert "12345678-1234-1234-1234-123456789abc" not in public_text
    assert report_contracts._latest({"agent": {"status": "done"}}, "agent") == {
        "status": "done"
    }
    assert report_contracts._latest({}, "agent") == {}

    native_plot = report_plots._render_plots(
        {"nativeUmapRna": "plots/native.png"},
        [],
    )
    assert "Rna native UMAP" in native_plot
    assert "finalized native Rna" in native_plot
    assert "No final cluster counts" in report_rendering._render_clusters({})

    legacy_parameter = {
        "fromAssay": "RNA",
        "evaluations": [{"candidateId": "native"}],
    }
    assert report_rendering._parameter_rows(legacy_parameter)[0]["assay"] == "RNA"
    assert "No Parameter Tuning report" in report_rendering._render_parameter_tuning({})
    rendered_parameter = report_rendering._render_parameter_tuning(
        {
            "fromAssay": "RNA",
            "searchPlan": {"status": "refine"},
            "comparisons": [{"candidateId": "native"}],
            "finalSelection": {"comparisons": [{"candidateId": "integrated"}]},
        }
    )
    assert "RNA" in rendered_parameter
    assert "final graph" in rendered_parameter
    assert "No provider execution metadata" in report_rendering._render_executions({})


def test_wide_technical_records_use_readable_card_layout() -> None:
    wide_row = {f"field_{index}": f"value {index}" for index in range(8)}
    wide_row["_selected"] = True
    wide_markup = report_rendering._table([wide_row])

    assert "<table" not in wide_markup
    assert 'class="record-fields"' in wide_markup
    assert 'class="record table-record table-record-selected"' in wide_markup
    assert "Field 7" in wide_markup
    assert "value 7" in wide_markup

    compact_markup = report_rendering._table([{"name": "candidate", "score": 0.5}])
    assert "<table" in compact_markup
    assert "<th>Name</th>" in compact_markup


def test_report_collects_bounded_artifact_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def artifact(
        kind: str,
        digit: str,
        *,
        scope: Literal["assay", "datastore"] = "assay",
    ) -> ArtifactReferenceModel:
        return ArtifactReferenceModel(
            scope=scope,
            assay=None if scope == "datastore" else "RNA",
            kind=kind,
            artifactId=digit * 64,
        )

    final_clusters = artifact("cluster_labels", "3")
    final_umap = artifact("embedding", "4")
    final = FinalAnalysisHandoff(
        workflowRunId="report-workflow",
        primaryAssay="RNA",
        markerAssay="RNA",
        cellSelection=artifact("cell_selection", "c", scope="datastore"),
        graph=artifact("connectivity_map", "2"),
        clusters=final_clusters,
        umap=final_umap,
        markers=artifact("marker_table", "5"),
        nativeAnalyses=[
            NativeAnalysisHandoff.get_blank(),
            NativeAnalysisHandoff(
                assay="RNA / strange assay",
                reductionMethod="pca",
                clusters=artifact("cluster_labels", "6"),
                umap=artifact("embedding", "7"),
            ),
            NativeAnalysisHandoff(
                assay="RNA / strange assay",
                reductionMethod="pca",
                clusters=artifact("cluster_labels", "8"),
                umap=artifact("embedding", "9"),
            ),
        ],
    ).with_handoff_id()
    workflow = _workflow()
    result = AutomatedWorkflowResult(
        status="completed",
        currentStage="analysis_finalization",
        zarrPath=str(tmp_path / "data.zarr"),
        workflowRun=workflow,
        finalAnalysis=final,
        finalHandoffId=final.handoffId,
        decisionRunId=workflow.workflowRunId,
    )

    class PlotMethods:
        @staticmethod
        def marker_heatmap(**_kwargs: object) -> object:
            raise RuntimeError("heatmap unavailable")

    class ArtifactStore:
        plots = PlotMethods()

        @staticmethod
        def load_artifact(reference: object) -> dict[str, np.ndarray]:
            if getattr(reference, "artifact_id", None) == "3" * 64:
                return {"values": np.asarray(["0", "0", "1"])}
            return {}

        @staticmethod
        def inspect_artifact(_reference: object) -> SimpleNamespace:
            return SimpleNamespace(
                parameters={
                    "normalization": {
                        "log_transform": True,
                        "renormalize_subset": True,
                    }
                }
            )

        @staticmethod
        def get_markers(
            _marker: object,
            *,
            group_id: str,
            min_score: float,
            min_frac_exp: float,
        ) -> pd.DataFrame:
            assert min_score == -1
            assert min_frac_exp == -1
            if group_id == "1":
                raise RuntimeError("marker table unavailable")
            return pd.DataFrame(
                {
                    "group_id": ["unknown", "0", "0"],
                    "feature_name": ["ignored", "CD3D", "unresolved"],
                    "feature_id": ["ignored-id", "ENSG00000167286", None],
                    "feature_index": [None, None, None],
                    "score": [3.0, 2.0, 1.0],
                }
            )

    monkeypatch.setattr(report_plots, "MAX_EMBEDDING_PLOT_CELLS", 0)
    monkeypatch.setattr(report_plots, "MAX_COMPOSITION_PLOT_CELLS", 0)
    monkeypatch.setattr(report_plots, "MAX_DOTPLOT_CELLS", 0)
    monkeypatch.setattr(report_plots, "MAX_CONNECTIVITY_PLOT_CELLS", 0)

    store = ArtifactStore()
    counts, markers, plots, notes = report_plots._collect_final_artifacts(
        store,
        result,
        tmp_path / "plots",
    )
    assert counts == {"0": 2, "1": 1}
    assert len(markers) == 3
    assert plots == {}
    assert any("nativeUmapRnaStrangeAssay2" in note for note in notes)
    assert any("marker export for cluster 1" in note for note in notes)
    assert any("markerDotplot: skipped" in note for note in notes)
    assert any("clusterConnectivity: skipped" in note for note in notes)

    monkeypatch.setattr(report_plots, "MAX_MARKER_DOTPLOT_FEATURES", 1)
    _counts, _markers, _plots, one_marker_notes = report_plots._collect_final_artifacts(
        store, result, tmp_path / "plots-one"
    )
    assert any("markerDotplot: skipped" in note for note in one_marker_notes)

    monkeypatch.setattr(report_plots, "MAX_MARKER_DOTPLOT_FEATURES", object())
    _counts, _markers, _plots, invalid_limit_notes = (
        report_plots._collect_final_artifacts(store, result, tmp_path / "plots-invalid")
    )
    assert any("markerDotplot: TypeError" in note for note in invalid_limit_notes)

    incomplete = result.model_copy(
        update={"finalAnalysis": FinalAnalysisHandoff.get_blank()}
    )
    with pytest.raises(ValueError, match="lacks its selection"):
        report_plots._collect_final_artifacts(
            store,
            incomplete,
            tmp_path / "plots-incomplete",
        )


def test_data_enrichment_cache_rollback_and_pending_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection = AssayFeatureInspection.get_example()
    completed = DataEnrichmentDependencies(
        store=object(),
        assays=["RNA"],
        inspections={"RNA": inspection},
        toolCalls=[
            DataEnrichmentToolCall(
                name="inspect_assay_features_batch",
                assay="all",
            )
        ],
    )
    completed_context = SimpleNamespace(deps=completed)

    assert (
        asyncio.run(
            enrichment_tools.inspect_assay_features(
                completed_context,
                assay_name="RNA",
            )
        )
        == inspection
    )
    cached_batch = asyncio.run(
        enrichment_tools.inspect_assay_features_batch(completed_context)
    )
    assert cached_batch.inspections == [inspection]
    assert cached_batch.evidenceIds == inspection.evidenceIds

    incomplete = DataEnrichmentDependencies(
        assays=["RNA"],
        toolCalls=[DataEnrichmentToolCall(name="sentinel", assay="RNA")],
    )
    with pytest.raises(ModelRetry, match="datastore"):
        asyncio.run(
            enrichment_tools.inspect_assay_features_batch(
                SimpleNamespace(deps=incomplete)
            )
        )
    assert [call.name for call in incomplete.toolCalls] == ["sentinel"]

    provider_error = UnexpectedModelBehavior("provider output failed")
    with pytest.raises(UnexpectedModelBehavior, match="provider output failed"):
        enrichment_validation.pending_data_enrichment_report(
            DataEnrichmentDependencies(assays=["RNA"]),
            error=provider_error,
            model_name="test-model",
        )
    pending = enrichment_validation.pending_data_enrichment_report(
        DataEnrichmentDependencies(
            assays=["RNA"],
            inspections={"RNA": inspection},
            evidenceIds=set(inspection.evidenceIds),
        ),
        error=provider_error,
        model_name="test-model",
    )
    assert pending.status == "needsInput"
    assert pending.policies == []
    assert pending.inspections == [inspection]

    def fail_before_inspection(**_kwargs: object) -> object:
        raise UnexpectedModelBehavior("no inspection completed")

    monkeypatch.setattr(enrichment_agent, "run_agent_sync", fail_before_inspection)
    store = SimpleNamespace(assay_names=["RNA"])
    with pytest.raises(UnexpectedModelBehavior, match="no inspection completed"):
        DataEnrichmentAgent(object()).run(store)


def test_biological_interpretation_cache_and_fallback_branches() -> None:
    composition = ClusterCompositionEvidence.get_example()
    composition_deps = BiologicalInterpretationDependencies(
        compositionEvidence=composition
    )
    assert (
        asyncio.run(
            biological_tools.inspect_cluster_composition(
                SimpleNamespace(deps=composition_deps)
            )
        )
        == composition
    )

    marker = ClusterMarkerEvidence.get_example()
    marker_deps = BiologicalInterpretationDependencies(
        clusterValues={marker.clusterId: 0},
        markerEvidence={marker.clusterId: marker},
    )
    assert (
        asyncio.run(
            biological_tools.inspect_cluster_markers(
                SimpleNamespace(deps=marker_deps),
                cluster_id=marker.clusterId,
            )
        )
        == marker
    )

    invalid_report = BiologicalInterpretationReport(
        status="done",
        needsInput=BiologicalInterpretationNeedsInput(question="More context?"),
    )
    with pytest.raises(ModelRetry, match="Only a needsInput"):
        biological_validation.validate_biological_interpretation_report(
            invalid_report,
            BiologicalInterpretationDependencies(clusterValues={"0": 0}),
        )

    provider_error = UnexpectedModelBehavior("structured output failed")
    with pytest.raises(UnexpectedModelBehavior, match="structured output failed"):
        biological_validation.fallback_biological_interpretation_report(
            BiologicalInterpretationDependencies(),
            error=provider_error,
            model_name="test-model",
        )
    needs_markers = biological_validation.fallback_biological_interpretation_report(
        BiologicalInterpretationDependencies(
            clusterValues={"0": 0},
            evidenceIds={"composition:clusters"},
        ),
        error=provider_error,
        model_name="test-model",
    )
    assert needs_markers.status == "needsInput"
    assert needs_markers.needsInput is not None
    assert needs_markers.evidenceIds == ["composition:clusters"]


def test_experimental_context_rejects_invalid_batches_and_builds_pending_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_batches = (
        (
            [{"name": "batch", "domain": "technical", "kind": "categorical"}],
            "missing",
            "Unknown batch column",
        ),
        (
            [{"name": "condition", "domain": "biological", "kind": "categorical"}],
            "condition",
            "must be classified as technical",
        ),
        (
            [{"name": "depth", "domain": "technical", "kind": "continuous"}],
            "depth",
            "must be categorical",
        ),
    )
    for columns, batch_column, message in invalid_batches:
        deps = ExperimentalContextDependencies(
            characterization=CovariateCharacterization(
                status="done",
                columns=columns,
            )
        )
        with pytest.raises(ModelRetry, match=message):
            asyncio.run(
                experimental_tools.analyze_experimental_design(
                    SimpleNamespace(deps=deps),
                    column_domains={},
                    coefficients_of_interest=[],
                    units_of_inference={},
                    batch_columns=[batch_column],
                )
            )

    characterization = CovariateCharacterization(
        status="done",
        columns=[{"name": "condition", "domain": "biological", "kind": "categorical"}],
    )
    monkeypatch.setattr(
        experimental_validation,
        "characterize_covariates",
        lambda *_args, **_kwargs: characterization,
    )

    def offer_profile(
        deps: ExperimentalContextDependencies,
        _characterization: CovariateCharacterization,
    ) -> list[CellQcProfileEvidence]:
        profile = CellQcProfileEvidence.get_example()
        deps.qcProfiles[profile.profileId] = profile
        return [profile]

    monkeypatch.setattr(
        experimental_validation,
        "_offered_qc_profiles",
        offer_profile,
    )
    pending_deps = ExperimentalContextDependencies(
        cellSelection=ArtifactReferenceModel(
            scope="datastore",
            kind="cell_selection",
            artifactId="c" * 64,
        ),
        htoIdentityColumns=["hto_identity"],
    )
    pending = experimental_validation.pending_experimental_context_result(
        pending_deps,
        error=UnexpectedModelBehavior("design output failed"),
        model_name="test-model",
    )
    assert pending.status == "needsInput"
    assert pending_deps.characterization is characterization
    assert pending.cellQc.profileId == ""
    assert pending.qcProfiles[0].profileId == (
        CellQcProfileEvidence.get_example().profileId
    )


def test_agent_execution_logs_nested_failures_for_sync_and_async_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingAgent:
        async def __aenter__(self) -> "FailingAgent":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        async def run(self, *_args: object, **_kwargs: object) -> object:
            try:
                raise ValueError("inner failure")
            except ValueError as cause:
                raise RuntimeError("outer failure") from cause

    monkeypatch.setattr(
        agent_exec_module,
        "_build_agent",
        lambda **_kwargs: FailingAgent(),
    )
    messages: list[str] = []
    monkeypatch.setattr(agent_exec_module.logger, "error", messages.append)
    with pytest.raises(RuntimeError, match="outer failure"):
        agent_exec_module.run_agent_sync(
            model=object(),
            output_type=dict,
            system_prompt="system",
            user_prompt="user",
            name="sync-failure",
        )
    with pytest.raises(RuntimeError, match="outer failure"):
        asyncio.run(
            agent_exec_module.run_agent(
                model=object(),
                output_type=dict,
                system_prompt="system",
                user_prompt="user",
                name="async-failure",
            )
        )
    assert all("caused by ValueError: inner failure" in message for message in messages)
    assert "sync-failure" in messages[0]
    assert "async-failure" in messages[1]


def test_journal_retryable_error_handles_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    assert journal_module.is_retryable_model_error(RuntimeError("unavailable")) is False
