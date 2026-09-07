---
description: Choose, explain, and execute RNA analysis settings with Scarf agents.
jupytext:
  cell_metadata_filter: tags
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(agent_workflow)=

# Choose and explain RNA analysis settings

Scarf computes evidence about your data, the agent chooses between bounded alternatives, and
Scarf executes the selected settings. Start with a dataset, study context, and a configured
Pydantic AI model:

```python
from scarf.agent import analyze_rna

result = analyze_rna(
    "study.h5ad",
    zarr_path="study.zarr",
    model=model,
    study_context="Human blood from one healthy donor, with no treatment comparison.",
    study_objective="Identify stable major immune-cell populations.",
    max_candidates=50,
)
if result.status != "completed":
    raise RuntimeError(f"{result.status}: {'; '.join(result.notes)}")

result.plot_embedding()
markers = result.get_markers()
report_path = result.report()
```

This release analyzes one RNA assay. Stores may contain other modalities; pass `assay="counts"`
when more than one RNA assay is available. Automated multimodal integration and hypothesis testing
are outside this workflow. Markers are descriptive evidence. The optional agent dependency is
installed with `uv pip install "scarf[agent]"`.

`max_candidates` limits reserved candidate slots across the initial analysis and any feature-policy
revision. Each pass reserves all configured alternatives before screening, including conditional
candidates that may not execute. The defaults reserve 25 slots for the baseline and another 25
if a feature-policy revision runs. The default limit of 50 admits both passes; a smaller limit
never shrinks the candidate lists. An insufficient remaining budget stops admission of that pass.
The limit does not count actual executions or bound runtime or provider tokens. The result
methods use the completed analysis directly and reopen its store read-only; they do not retrain
UMAP or copy results into live metadata. `report()` returns a local path without opening a browser.

The executable example below uses the advanced `AgentOrchestrator` interface to keep a teaching
run small and reproducible. That interface also supports explicit candidate lists, workspaces,
provider limits, and resumable checkpoints.

Repository developers can also run `notebook/agent_workflow_new.ipynb` on the full abdominal
adipose cohort or `notebook/agent_workflow_new_short.ipynb` on its reproducible 2,000-cell smoke
sample. Both use unattended input policy and keep runtime files beside the notebooks.

```{mermaid}
flowchart LR
    A[Input dataset and study context] --> B[Ingest]
    B --> C[Data Enrichment]
    C --> E[Experimental Context]
    E --> F[Preprocessing plan]
    F --> G[RNA preprocessing]
    G --> H[Parameter Tuning]
    H --> I[Feature-policy review]
    I --> J[Optional revised preprocessing and tuning]
    J --> K[Analysis review]
    K --> L[UMAP, clusters, and markers]
    L --> M[Persisted reports and local HTML]
```

The committed documentation build uses one scripted Pydantic AI `FunctionModel`. It exercises the
real tools, validators, preprocessing, candidate execution, finalization, persistence, and report
generation without an API key. It does not assign biological identities; that remains a separate
`BiologicalInterpretationAgent` call after finalization. A live-provider configuration is shown at
the end.

## 1. Download the raw teaching dataset

Install the optional agent dependencies before running this workflow outside the documentation
environment:

```console
uv pip install "scarf[agent]"
```

The documentation run converts a raw H5 file into a separate teaching store. The explicit
`overwrite` direction is safe here because `agent_workflow.zarr` is a disposable derived target
owned by this tutorial. Omit it in ordinary work unless replacing that exact destination is
intentional.

```{code-cell} ipython3
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import scarf
from scarf.agent import (
    AgentOrchestrator,
    AgentRunConfig,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    DecisionSelection,
    load_agent_report,
    load_agent_workflow,
)

scarf.configure_output(level="WARNING", progress=False)

source_path = scarf.cytebase.connect("scarf_docs").download(
    "tenx_5K_pbmc_rnaseq/data.h5",
    destination="scarf_datasets",
)[0]
zarr_path = source_path.with_name("agent_workflow.zarr")

study_context = (
    "This is a human 10x Genomics 5K PBMC 3-prime gene-expression dataset "
    "from peripheral blood collected from one healthy donor. The goal is "
    "unsupervised identification and characterization of the major immune-cell "
    "populations. No treatment comparison, technical batch covariate, paired "
    "modality, or independent replication metadata is available. Do not invent "
    "absent design variables or report treatment effects."
)

{"source": source_path.name, "destination": zarr_path.name}
```

The hidden setup below routes each model request by its available tools. Every structured response
is assembled from the exact tool result, so a fabricated assay, feature family, candidate, cluster,
or evidence identifier still fails the production validator.

```{code-cell} ipython3
:tags: [remove-cell]

import json
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from scarf.agent.biological_interpretation import (
    BiologicalInterpretationReport,
    ClusterCompositionEvidence,
    ClusterInterpretation,
    ClusterMarkerBatchEvidence,
)
from scarf.agent.data_enrichment import (
    AssayFeatureInspectionBatch,
    DataEnrichmentReport,
    FeatureSelectionPolicy,
    StudyContextSummary,
)
from scarf.agent.experimental_context import (
    BatchCorrectionPlan,
    CovariateEvidence,
    ExperimentalContextDecision,
)

def _prompt_text(messages: list[ModelMessage]) -> str:
    values = []
    for message in messages:
        for part in message.parts:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                values.append(content)
            elif isinstance(content, tuple):
                values.extend(item for item in content if isinstance(item, str))
    return "\n".join(values)


def _tool_result(
    messages: list[ModelMessage],
    tool_name: str,
    model_type: Any,
) -> Any:
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, ToolReturnPart) and part.tool_name == tool_name:
                if isinstance(part.content, model_type):
                    return part.content
                if isinstance(part.content, str):
                    return model_type.model_validate_json(part.content)
                return model_type.model_validate(part.content)
    raise AssertionError(f"Missing tool return {tool_name!r}")


def _tool_call(name: str, args: dict[str, Any] | None = None) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args or {})])


def _structured_output(info: AgentInfo, value: Any) -> ModelResponse:
    payload = value.model_dump() if hasattr(value, "model_dump") else value
    return _tool_call(info.output_tools[0].name, payload)


def _scripted_workflow_model() -> tuple[FunctionModel, dict[str, int]]:
    state = {
        "enrichment": 0,
        "context": 0,
        "parameter": 0,
        "biology": 0,
        "requests": 0,
    }

    async def reply(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        state["requests"] += 1
        tools = {tool.name for tool in info.function_tools}

        if "inspect_assay_features_batch" in tools or state["enrichment"] == 1:
            if state["enrichment"] == 0:
                state["enrichment"] = 1
                return _tool_call("inspect_assay_features_batch")

            batch = _tool_result(
                messages,
                "inspect_assay_features_batch",
                AssayFeatureInspectionBatch,
            )
            policies = []
            for inspection in batch.inspections:
                species_observed = inspection.species != "unknown"
                policy_evidence = list(inspection.evidenceIds)
                if not species_observed:
                    policy_evidence.append("context:study")
                policies.append(
                    FeatureSelectionPolicy(
                        assay=inspection.assay,
                        species=(
                            inspection.species
                            if species_observed
                            else "homo_sapiens"
                        ),
                        speciesConfidence="high" if species_observed else "medium",
                        speciesRationale=(
                            inspection.speciesReason
                            or "The exact study paragraph identifies a human sample."
                        ),
                        excludeFamilies=[
                            family.family
                            for family in inspection.families
                            if family.count > 0 and family.defaultExclude is True
                        ],
                        protectFamilies=[
                            family.family
                            for family in inspection.families
                            if family.count > 0 and family.defaultExclude is False
                        ],
                        rationale=(
                            "Exclude observed technical families and preserve "
                            "observed protected families."
                        ),
                        evidenceIds=list(dict.fromkeys(policy_evidence)),
                    )
                )
            state["enrichment"] = 2
            return _structured_output(
                info,
                DataEnrichmentReport(
                    status="done",
                    studyContextSummary=StudyContextSummary(
                        organismReferences=["human"],
                        tissueReferences=["peripheral blood"],
                        experimentalReferences=[
                            "10x Genomics 5K PBMC 3-prime gene-expression dataset"
                        ],
                        analysisIntentReferences=[
                            "unsupervised identification and characterization of "
                            "the major immune-cell populations"
                        ],
                    ),
                    policies=policies,
                ),
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
                return _tool_call("inspect_cell_covariates")
            if state["context"] == 1:
                state["context"] = 2
                return _tool_call(
                    "analyze_experimental_design",
                    {
                        "column_domains": {},
                        "coefficients_of_interest": [],
                        "units_of_inference": {},
                        "batch_columns": [],
                    },
                )

            design = _tool_result(
                messages,
                "analyze_experimental_design",
                CovariateEvidence,
            )
            profile = next(
                value
                for value in design.qcProfiles
                if value.action == "skip"
            )
            evidence_id = profile.evidenceId
            state["context"] = 3
            return _structured_output(
                info,
                ExperimentalContextDecision(
                    batchCorrection=BatchCorrectionPlan(
                        action="skip",
                        rationale="No trusted technical batch column was supplied.",
                        evidenceIds=[evidence_id],
                    ),
                    rationale="No experimental covariates were supplied.",
                    evidenceIds=[evidence_id],
                ),
            )

        if tools.intersection(
            {"inspect_cluster_composition", "inspect_cluster_markers_batch"}
        ) or state["biology"]:
            if state["biology"] == 0:
                state["biology"] = 1
                return _tool_call("inspect_cluster_composition")
            if state["biology"] == 1:
                composition = _tool_result(
                    messages,
                    "inspect_cluster_composition",
                    ClusterCompositionEvidence,
                )
                state["biology"] = 2
                return _tool_call(
                    "inspect_cluster_markers_batch",
                    {"cluster_ids": list(composition.clusterCounts)},
                )

            marker_batch = _tool_result(
                messages,
                "inspect_cluster_markers_batch",
                ClusterMarkerBatchEvidence,
            )
            interpretations = []
            for cluster in marker_batch.clusters:
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
                                "The returned marker panel is led by "
                                f"{marker_name}."
                            ),
                            evidenceIds=[cluster.evidenceId],
                        )
                    )
            state["biology"] = 3
            return _structured_output(
                info,
                BiologicalInterpretationReport(
                    status="done",
                    clusterInterpretations=interpretations,
                    evidenceIds=[item.evidenceIds[0] for item in interpretations],
                    limitations=[
                        "The scripted documentation model returns marker-linked "
                        "hypotheses, not validated cell identities."
                    ],
                    stopReason=(
                        "Every cluster with returned marker evidence was reviewed."
                    ),
                ),
            )

        prompt = _prompt_text(messages)
        if any(
            tool.parameters_json_schema.get("title")
            == "AnalysisVisualAdjudication"
            for tool in info.output_tools
        ):
            payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])
            return _structured_output(
                info,
                {
                    "status": "acceptable",
                    "selectedCandidateId": payload["selectedCandidateId"],
                    "rationale": (
                        "The bounded diagnostic board agrees with the registered "
                        "numeric evidence."
                    ),
                },
            )

        decision, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])
        evidence_by_class = {}
        evidence_class_by_id = {}
        for item in decision["evidence"]:
            evidence_by_class.setdefault(
                item["evidenceClass"],
                item["evidenceId"],
            )
            evidence_class_by_id[item["evidenceId"]] = item["evidenceClass"]
        preferred = decision.get("metricPreferredOptionId")
        selected = (
            next(
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
        evidence_ids = list(selected.get("requiredEvidenceIds", []))
        cited_classes = {
            evidence_class_by_id[evidence_id] for evidence_id in evidence_ids
        }
        for evidence_class in selected["requiredEvidenceClasses"]:
            if evidence_class not in cited_classes:
                evidence_ids.append(evidence_by_class[evidence_class])
        state["parameter"] += 1
        return _structured_output(
            info,
            DecisionSelection(
                selectedOptionId=selected["optionId"],
                evidenceIds=evidence_ids,
                rationale="Select the registered metric-preferred option.",
                confidence="high",
            ),
        )

    return FunctionModel(reply), state

```

## 2. Configure a bounded teaching search

This run uses one HVG count and singleton PCA, neighbor, and clustering-resolution lists, with
no refinement or Harmony. Each tuning pass still evaluates four stage candidates: PCA, the native
correction baseline, neighbors, and clustering. Including the HVG screen and selection evaluations,
the executor reserves seven evaluations per pass. The limit of fourteen permits a second pass
after a feature-policy revision. This is a small sequential search. QC comparisons, stability
diagnostics, provider requests, and finalization have separate costs.

```{code-cell} ipython3
model, model_state = _scripted_workflow_model()
config = AutomatedWorkflowConfig(
    inputPolicy="unattended",
    maxRefinedCandidatesPerAssay=0,
    maxHarmonyCandidatesPerAssay=0,
    maxCandidateEvaluations=14,
    hvgCandidateCounts=(1000,),
    pcaCandidateDimensions=(20,),
    graphNeighborCandidates=(21,),
    leidenResolutionCandidates=(1.0,),
    minClusterCells=2,
    agentRunConfig=AgentRunConfig(
        requestLimit=5,
        toolCallLimit=5,
    ),
)
orchestrator = AgentOrchestrator(model, config=config)
request = AutomatedWorkflowRequest(
    sourcePath=str(source_path),
    zarrPath=str(zarr_path),
    studyContext=study_context,
    studyObjective="Discover stable major immune-cell populations.",
    primaryAssay="RNA",
    markerAssay="RNA",
    analysisAssays=["RNA"],
    ingestDirections={"overwrite": True, "defaultAssay": "RNA"},
)

{
    "candidate_evaluation_limit": config.maxCandidateEvaluations,
    "refinement_candidates": config.maxRefinedCandidatesPerAssay,
    "harmony_candidates": config.maxHarmonyCandidatesPerAssay,
    "input_policy": config.inputPolicy,
}
```

## 3. Run without an interactive checkpoint

The unattended policy lets registered rules resolve model deferrals. Genuine unresolved evidence
becomes an explicit abstention or failure rather than a pause. The documentation captures the
normal report-path printout so its output does not contain a random workflow identifier.

```{code-cell} ipython3
with redirect_stdout(StringIO()):
    result = orchestrator.run(request)

if (
    result.status != "completed"
    or result.finalAnalysis is None
    or result.preprocessingPlan is None
    or result.workflowRun is None
    or result.zarrPath is None
):
    raise RuntimeError(f"Unexpected workflow result: {result.status}, {result.notes}")

plan = result.preprocessingPlan
{
    "status": result.status,
    "stage": result.currentStage,
    "primary_assay": plan.primaryAssay,
    "marker_assay": plan.markerAssay,
    "cell_qc": plan.cellQc.action,
    "routes": [
        {
            "assay": assay.assay,
            "features": assay.featureMethod,
            "reduction": assay.reductionMethod,
        }
        for assay in plan.assays
    ],
}
```

The workflow persists the exact preprocessing plan, decisions, report handoffs, and final artifact
references before returning.

## 4. Inspect the persisted workflow

The returned workflow identity resolves the durable record. Reopening it does not execute an
analysis stage.

```{code-cell} ipython3
persisted_workflow = load_agent_workflow(
    result.zarrPath,
    result.workflowRun.workflowRunId,
    workspace=result.workflowRun.workspace,
)

{
    "status": persisted_workflow.status,
    "stage": result.currentStage,
    "agent_reports": [ref.agentName for ref in result.reportReferences],
    "model_requests": model_state["requests"],
    "graph_method": result.finalAnalysis.graphMethod,
    "marker_assay": result.finalAnalysis.markerAssay,
}
```

The single scripted provider handles every model-driven orchestrator stage. Deterministic
operations, such as RNA preprocessing, candidate execution, promotion, UMAP, clustering,
marker search, and persistence, do not require separate model requests.

## 5. Review parameter evidence and agent reports

The parameter agent receives executor-produced metrics for candidates that have already run. It
does not generate Scarf code. Each candidate follows the explicit reduction, optional Harmony,
ANN, neighbours, connectivity, Leiden, and metric chain. The final selected branch is replayed with
state updates and checked against the evaluated immutable references.

```{code-cell} ipython3
reports = {
    reference.agentName: load_agent_report(result.zarrPath, reference)
    for reference in result.reportReferences
}
parameter_report = reports["parameter_tuning"]

candidate_metrics = []
for assay, assay_report in parameter_report.assayReports.items():
    for index, evaluation in enumerate(assay_report.evaluations, start=1):
        candidate_metrics.append(
            {
                "assay": assay,
                "candidate": index,
                "dimensions": evaluation.parameters.dimensions,
                "resolution": evaluation.parameters.leidenResolution,
                "neighbors": evaluation.parameters.neighborsK,
                "eligible": evaluation.eligible,
                "clusters": evaluation.metrics.nClusters,
                "smallest_cluster": evaluation.metrics.minClusterCells,
                "graph_silhouette": evaluation.metrics.graphSilhouetteMedian,
            }
        )

{
    "candidates": candidate_metrics,
    "stop_reason": parameter_report.stopReason,
    "report_statuses": {
        name: report.status for name, report in reports.items()
    },
}
```

This teaching run demonstrates the successive parameter decisions with one option per stage.
The default configuration compares explicit HVG, PCA, neighbor, and resolution lists and may execute one evidence-driven
refinement. Harmony is added only when the exact Experimental Context handoff authorizes a matched
comparison.

## 6. Plot the exact final UMAP and inspect markers

The result uses its final UMAP and cluster artifacts directly. Display options are forwarded to
Scarf's plotting API. Exact artifact references remain available in `result.finalAnalysis` for
advanced workflows and Biological Interpretation.

```{code-cell} ipython3
final = result.finalAnalysis
if (
    final.cellSelection is None
    or final.clusters is None
    or final.umap is None
    or final.markers is None
):
    raise RuntimeError("The completed final handoff is missing required artifacts")

result.plot_embedding(
    legend_loc="on_data",
    frame="none",
)
```

UMAP is a presentation artifact. The tuning agent compares graph and metadata metrics, not visual
appearance, and the orchestrator does not train several UMAPs to choose the most attractive one.

```{code-cell} ipython3
marker_table = result.get_markers(
    group_id=None,
    min_score=-1,
    min_frac_exp=-1,
)
marker_table.sort_values(
    ["group_id", "score"],
    ascending=[True, False],
).groupby("group_id", sort=True).head(2)[
    ["group_id", "feature_name", "score", "frac_exp"]
].head(12)
```

Marker scores are cell-level descriptive evidence. They are not replicate-aware differential
expression, and the scripted identities remain hypotheses.

## 7. Find the local HTML report

A completed local workflow first persists its terminal result and then writes a replaceable HTML
view under `agents/runs/<workflowRunId>/report/index.html`. `result.report()` returns that path,
generating the view from saved results if it is missing. It opens directly on the analysis and
does not train another UMAP. Advanced callers can use `generate_agent_report()` to explicitly
regenerate an existing view.

```{code-cell} ipython3
report_path = result.report()
display_path = str(report_path.relative_to(Path(result.zarrPath).parent)).replace(
    result.workflowRun.workflowRunId,
    "<workflowRunId>",
)

{
    "report": display_path,
    "exists": report_path.is_file(),
    "final_artifact_kinds": {
        "selection": final.cellSelection.kind,
        "clusters": final.clusters.kind,
        "umap": final.umap.kind,
        "markers": final.markers.kind,
    },
}
```

## Pauses, failures, and other input formats

With `inputPolicy="pause"`, `needsInput` keeps the workflow running. Inspect every returned
question and supply only grounded answers through `AutomatedWorkflowResumeRequest`.
`inputPolicy="unattended"` returns an explicit abstention or failure when evidence cannot be
resolved safely. `failed` and `abandoned` are terminal. An ingest ambiguity can occur before a
persisted workflow exists; update `ingestDirections` and call `run()` again in that case. A running
workflow can also be finalized as abandoned with `orchestrator.cancel()`.

For another new local H5 or H5AD input, provide a destination that does not yet exist:

```python
request = AutomatedWorkflowRequest(
    sourcePath="study.h5ad",
    zarrPath="study.zarr",
    studyContext="One paragraph describing the study, design, and analysis intent.",
    studyObjective="Discover stable populations relevant to the study.",
)
result = AgentOrchestrator(
    model,
    config=AutomatedWorkflowConfig(inputPolicy="unattended"),
).run(request)
if result.status != "completed":
    raise RuntimeError(f"{result.status}: {'; '.join(result.notes)}")
```

For an existing Zarr input, omit `zarrPath` or set it to the same location. Its current `I`
selection is preserved and snapshotted. A workspace may be supplied only for an existing Zarr
input.

## Use a live model

Replace the scripted model with one supported Pydantic AI model. Keep credentials in environment
variables and never place them in a notebook or datastore:

```python
import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    os.environ["SCARF_AGENT_MODEL"],
    provider=OpenAIProvider(
        base_url=os.environ["SCARF_AGENT_BASE_URL"],
        api_key=os.environ["SCARF_AGENT_API_KEY"],
    ),
)

orchestrator = AgentOrchestrator(
    model,
    config=AutomatedWorkflowConfig(
        inputPolicy="unattended",
        runConfoundedHarmonyDiagnostic=True,
    ),
)
result = orchestrator.run(
    AutomatedWorkflowRequest(
        sourcePath="study.h5ad",
        zarrPath="study.zarr",
        studyContext=(
            "Human single-cell study with three biological replicates per "
            "condition; donor is the unit of inference and library is technical."
        ),
        studyObjective=(
            "Discover stable populations while preserving the condition structure."
        ),
    )
)
if result.status != "completed":
    raise RuntimeError(f"{result.status}: {'; '.join(result.notes)}")
```

Provider output remains provisional. Scarf validates evidence identifiers, operations, artifact
lineage, and resume state, but it cannot establish that a biologically plausible interpretation is
true.
