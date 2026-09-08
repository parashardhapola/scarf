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

Give Scarf a dataset, a configured Pydantic AI model, a study-context paragraph, and an objective.
Scarf computes the evidence, the agent evaluates a few consequential choices, and Scarf executes
the selected analysis.

```python
from scarf.agent import analyze_rna

result = analyze_rna(
    "study.h5ad",
    model=model,
    study_context="Human blood from one healthy donor; no treatment comparison.",
    study_objective="Identify stable major immune-cell populations.",
    zarr_path="study.zarr",
)
result.plot_embedding()
markers = result.get_markers()
report_path = result.report()
```

`analyze_rna` returns a completed result or raises `AnalysisError`. You do not need to inspect a
status field to catch an unsuccessful beginner run. Install the optional dependency with
`uv pip install "scarf[agent]"`.

This release analyzes one RNA assay. Other modalities can remain in the store; they do not enter
this workflow. Pass `assay="RNA2"` when several RNA assays exist. Automated integration, HTO
assignment, and biological significance or differential-expression hypothesis execution are
outside this workflow. Experimental Context still explores covariate patterns and possible
explanations of the study design.

The result opens its exact saved workspace and artifacts read-only. Its cluster-map helper shows
at most 50,000 cells with full population counts. Marker statistics use the complete selected
cohort. `report()` returns a local HTML path without opening a browser or rerunning an analysis.
The {doc}`../reference/api/agent` page describes the small public interface and advanced controls.

## A reproducible teaching analysis

The executable example uses a deterministic 1,000-cell teaching cohort drawn without replacement
from the public 10x Genomics 5K PBMC dataset (random seed 42). Preparation imports the public
file into a temporary Scarf store and marks those 1,000 cells as the active input; the downloaded
dataset is unchanged. Quality filtering and every required comparison still run through the
agent workflow. This small cohort demonstrates the workflow and does not represent an analysis
of all cells in the public dataset.

The example uses the real analysis operations and a local scripted `FunctionModel`.
The script chooses among observed partitions by agreement across clustering runs, then the
fraction of clusters with qualifying markers. Marker coverage alone does not establish biological
coherence. This makes
the example reproducible without an API key. It is a teaching policy, not a substitute for a model
that interprets the supplied diagnostic images and study-specific biology.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import scarf
from scarf.agent import analyze_rna

scarf.configure_output(level="WARNING", progress=False)
source_path = scarf.cytebase.connect("scarf_docs").download(
    "tenx_5K_pbmc_rnaseq/data.h5", destination="scarf_datasets",
)[0]
teaching_directory = TemporaryDirectory(prefix="scarf-agent-teaching-")
zarr_path = Path(teaching_directory.name) / "analysis.zarr"
study_context = (
    "Human 10x Genomics 5K PBMC 3-prime gene expression from peripheral blood, "
    "collected from one healthy donor. The teaching cohort is a deterministic random "
    "subset of 1,000 cells (seed 42), not the full public dataset. "
    "No treatment comparison, trusted technical "
    "batch column, paired modality, or independent replication metadata is available. "
    "Do not invent missing design variables or report treatment effects."
)
source_path.name
```

The hidden provider fixture assembles responses from actual tool results. Invented assay names,
feature families, candidates, and evidence identifiers still fail the production validators.

```{code-cell} ipython3
:tags: [remove-cell]

import json
from typing import Any

import numpy as np
from IPython import get_ipython
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

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
from scarf.agent.ingest import ingest

prepared_input = ingest(path=source_path, zarrPath=zarr_path)
if prepared_input.status != "done":
    raise RuntimeError(f"Teaching dataset import failed: {prepared_input.notes}")
teaching_store = scarf.DataStore(
    str(zarr_path), min_features_per_cell=-1, mito_pattern="", ribo_pattern="",
)
teaching_store.cells.reset_key("I")
teaching_cells = np.zeros(teaching_store.cells.N, dtype=bool)
teaching_cells[np.random.default_rng(42).choice(teaching_store.cells.N, 1000, replace=False)] = True
teaching_store.cells.update_key(teaching_cells, "I")
source_path = zarr_path
del teaching_store

notebook_shell = get_ipython()
if notebook_shell is not None:
    notebook_shell.run_line_magic("matplotlib", "inline")

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


def _scripted_workflow_model() -> tuple[FunctionModel, dict[str, Any]]:
    state = {
        "enrichment": 0,
        "context": 0,
        "parameter": 0,
        "assessments": [],
        "requests": 0,
    }

    async def reply(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        state["requests"] += 1
        tools = {tool.name for tool in info.function_tools}

        if (
            "inspect_assay_features_batch" in tools
            or state["enrichment"] == 1
            or any(
                tool.parameters_json_schema.get("title") == "DataEnrichmentReport"
                for tool in info.output_tools
            )
        ):
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
                        organismReferences=["Human"],
                        tissueReferences=["peripheral blood"],
                        experimentalReferences=[
                            "10x Genomics 5K PBMC 3-prime gene expression"
                        ],
                        analysisIntentReferences=[
                            "Discover stable major immune-cell populations."
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
        ) or state["context"] in {1, 2} or any(
            tool.parameters_json_schema.get("title") == "ExperimentalContextDecision"
            for tool in info.output_tools
        ):
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

        prompt = _prompt_text(messages)
        if any(
            {"selectedCandidateId", "comparisonConclusions"}.issubset(
                tool.parameters_json_schema.get("properties", {})
            )
            for tool in info.output_tools
        ):
            evidence, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])
            coverage = evidence["comparisonCoverage"]
            settings = coverage["candidateSettings"]
            candidates = {
                item["candidateId"]: item for item in evidence["candidates"]
                if item["status"] == "done" and item["eligible"]
            }
            if not candidates:
                raise AssertionError("The teaching run has no supported partition")

            def rank(identity):
                metrics = settings[identity]["metrics"]
                return tuple(
                    float(metrics[name]) if metrics.get(name) is not None else -1.0
                    for name in ("seedStability", "markerCoherence")
                )

            axis_names = {
                "hvgCount": "variable-gene count", "hvgRanking": "gene ranking",
                "featurePolicy": "gene-family policy", "pca": "PCA dimensions",
                "neighbors": "neighbor count", "partition": "clustering resolution",
            }
            metric_names = {
                "seedStability": "agreement across clustering runs",
                "subsampleStability": "agreement after resampling",
                "markerCoherence": "the fraction of clusters with qualifying markers",
                "markerSpecificityMedian": "median marker specificity",
                "macroF1": "classification agreement",
            }
            axis_ids = {}
            for row in coverage["comparisons"]:
                identities = axis_ids.setdefault(row["axis"], [])
                for identity in (row["baselineCandidateId"], row["alternativeCandidateId"]):
                    if identity is not None and identity not in identities:
                        identities.append(identity)
            comparison_ids = {axis: list(identities) for axis, identities in axis_ids.items()}
            if coverage["phase"] != "sensitivity":
                axis_ids["partition"] = list(dict.fromkeys(
                    [*axis_ids["partition"], *coverage["resolutionCandidateIds"]]
                ))
            preferences = {axis: max(identities, key=rank) for axis, identities in axis_ids.items()}
            pending_policy = any(row["status"] == "pending" for row in coverage["comparisons"])
            experiment_id = None
            if coverage["phase"] == "sensitivity":
                selected_id = evidence["currentCandidateId"]
                action_name = "combine"
            else:
                eligible_resolutions = [
                    identity for identity in coverage["resolutionCandidateIds"]
                    if identity in candidates
                ]
                if not eligible_resolutions:
                    raise AssertionError("The combined representation has no supported partition")
                selected_id = max(eligible_resolutions, key=rank)
                preferences["partition"] = selected_id
                action_name = "accept"
            if pending_policy:
                families = evidence["featureEvidence"][selected_id]["families"]
                supported = [
                    (key, option) for key, option in evidence["experiments"].items()
                    if option["parameter"] in {"includeFamily", "excludeFamily"}
                    and families.get(option["value"], {}).get(
                        "selectedExamples" if option["parameter"] == "excludeFamily"
                        else "excludedExamples"
                    )
                ]
                if not supported:
                    raise AssertionError("The teaching policy has no observed family program to nominate")
                experiment_id, option = max(supported, key=lambda item: (
                    families[item[1]["value"]].get("selectedGenes", 0),
                    item[1]["affectedEligibleGenes"],
                ))
                action_name = "experiment"
            selected = candidates[selected_id]
            metrics = selected["metrics"]
            genes = list(dict.fromkeys(
                gene for names in metrics.get("topMarkerGenes", {}).values()
                for gene in names
            ))[:8]
            qualitative = (
                "The saved marker preview contains " + ", ".join(genes) + "."
                if genes else "The saved marker preview is empty; cell identities remain unresolved."
            )
            conclusions = []
            for axis, identities in axis_ids.items():
                preferred = preferences[axis]
                score = settings[preferred]["metrics"]
                conclusions.append({
                    "axis": axis,
                    "candidateIds": identities,
                    "preferredCandidateId": preferred,
                    "quantitativeReason": "; ".join(
                        f"Observed alternative {index + 1}: repeat agreement "
                        f"{settings[identity]['metrics'].get('seedStability')}, "
                        f"marker coverage {settings[identity]['metrics'].get('markerCoherence')}"
                        for index, identity in enumerate(identities)
                    ),
                    "biologicalReason": (
                        "This scripted teaching policy does not establish cell identities or "
                        "infer that a gene program is a technical artifact. " + qualitative
                    ),
                    "plainLanguageSummary": (
                        f"The teaching policy compared {len(identities)} observed {axis_names[axis]} "
                        f"settings and preferred repeat agreement {score.get('seedStability')}, "
                        f"using marker coverage {score.get('markerCoherence')} to break ties."
                    ),
                    "tradeoffs": [{
                        "alternativeCandidateId": identity,
                        "metric": metric,
                        "preferredValue": score[metric],
                        "alternativeValue": settings[identity]["metrics"][metric],
                        "interpretation": (
                            f"An alternative has higher {metric_names[metric]} "
                            f"({settings[identity]['metrics'][metric]} versus {score[metric]}). "
                            "The teaching policy prioritizes repeat agreement, then marker coverage; "
                            "this loss remains an explicit limit of its choice."
                        ),
                    } for identity in comparison_ids[axis] if identity != preferred
                        for metric in ("seedStability", "subsampleStability", "markerCoherence",
                                       "markerSpecificityMedian", "macroF1")
                        if isinstance(score.get(metric), (int, float))
                        and isinstance(settings[identity]["metrics"].get(metric), (int, float))
                        and settings[identity]["metrics"][metric] > score[metric]],
                })
            quantitative = (
                f"Observed resolution {selected['parameters']['leidenResolution']} has "
                f"repeat agreement {metrics.get('seedStability')} and marker coverage "
                f"{metrics.get('markerCoherence')}."
            )
            summary = (
                "The teaching policy proposes testing a represented gene family; "
                "its contribution must be measured before retaining or changing the gene selection."
                if action_name == "experiment" else
                "The teaching policy proposes a combination of the observed settings; "
                "Scarf must execute that combination and compare its four resolutions."
                if action_name == "combine" else
                "The teaching policy selected the measured combined representation and "
                "its most repeatable eligible partition. Marker identities remain unvalidated."
            )
            action = {
                "action": action_name,
                "selectedCandidateId": selected_id,
                "experimentId": experiment_id,
                "correctionNeed": "notApplicable",
                "comparisonConclusions": conclusions,
                "plainLanguageSummary": summary,
                "evidenceIds": [
                    f"candidate:{selected_id}",
                    *list(evidence["imageHashes"])[:1],
                    "studyContract", "qcPolicy", "samplingCoverage", "featureEvidence",
                ],
                "quantitativeFindings": [quantitative],
                "qualitativeFindings": [qualitative],
                "objectivePreservation": (
                    "Preserve the single-donor population structure and retain marker "
                    "uncertainty; no batch or treatment comparison is supported."
                ),
                "rationale": summary + " " + quantitative,
                "concern": (
                    f"Observed family {option['value']} contains "
                    f"{families[option['value']].get('selectedGenes', 0)} selected genes. "
                    "Test sensitivity to this program without assuming that it is technical."
                ) if pending_policy else "",
                "expectedImprovement": (
                    "Measure whether changing this gene-family selection preserves the major "
                    "marker programs and improves repeat agreement."
                ) if pending_policy else "",
                "populationConcerns": [{
                    "candidateId": selected_id,
                    "clusterId": cluster,
                    "status": "nonEssentialLimitation",
                    "evidenceIds": [f"candidate:{selected_id}"],
                    "explanation": (
                        f"Population {cluster} has no qualifying marker genes and remains unclassified. "
                        "This tutorial demonstrates selecting analysis settings; it does not validate "
                        "cell identities or claim that every population has been biologically resolved."
                    ),
                } for cluster, names in metrics.get("topMarkerGenes", {}).items() if not names],
            }
            if action_name == "combine":
                action["combinedSettings"] = {
                    field: preferences[axis] for field, axis in (
                        ("hvgCountCandidateId", "hvgCount"),
                        ("hvgRankingCandidateId", "hvgRanking"),
                        ("featurePolicyCandidateId", "featurePolicy"),
                        ("pcaCandidateId", "pca"),
                        ("neighborsCandidateId", "neighbors"),
                    )
                }
            state["assessments"].append({
                "selection": action,
                "alternatives": [{
                    "resolution": settings[identity]["parameters"]["leidenResolution"],
                    "clusters": settings[identity]["metrics"].get("nClusters"),
                    "repeat_agreement": settings[identity]["metrics"].get("seedStability"),
                    "clusters_with_markers": settings[identity]["metrics"].get("markerCoherence"),
                    "selected": identity == selected_id,
                } for identity in (
                    coverage["resolutionCandidateIds"] if action_name == "accept"
                    else list(candidates)
                )],
            })
            return _structured_output(info, action)


        payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{") :])
        decision = payload["spec"]
        evidence_by_class = {}
        evidence_class_by_id = {}
        for item in payload["evidence"]["evidence"]:
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
            dict(
                selectedOptionId=selected["optionId"],
                evidenceIds=evidence_ids,
                rationale=f"Use the offered {selected['label']} policy with its required observed evidence.",
                confidence="high",
            ),
        )

    return FunctionModel(reply), state

```

```{code-cell} ipython3
model, model_state = _scripted_workflow_model()
result = analyze_rna(
    source_path,
    model=model,
    study_context=study_context,
    study_objective="Discover stable major immune-cell populations.",
)
{"status": result.status}
```

## See what was chosen and why

The starting graph is compared at four clustering resolutions: 0.5, 0.75, 1.0, and 1.25.
These partitions share the same cells, features, and graph. The table below contains the exact
observations offered to the scripted provider, followed by its recorded explanation.

```{code-cell} ipython3
assessment = model_state["assessments"][-1]
pd.DataFrame(assessment["alternatives"])
```

```{code-cell} ipython3
selection = assessment["selection"]
{
    "why": selection["rationale"],
    "marker_evidence": selection["qualitativeFindings"],
    "biology_to_preserve": selection["objectivePreservation"],
}
```

A live model receives required comparisons of variable-gene counts (1,000, 2,000, and 4,000),
PCA dimensions (10, 21, and 30), and neighbors (11, 21, and 41) against a shared baseline.
Supported batch-aware gene ranking and evidence-nominated gene eligibility changes also receive
matched comparisons. Infeasible values and identical gene selections are recorded explicitly.
The model must explain the observed tradeoffs and which biology should be preserved. The selected
combination is then executed and assessed at all four clustering resolutions before acceptance.
Further unresolved concerns require a targeted comparison or an incomplete outcome.
A metric rank alone does not authorize correction or deletion of a biological program. Batch
correction requires both a supported design and a matched comparison of native and corrected
representations. Confounded technical and biological variables cannot license correction.

## Inspect the analysis

The result fixes the saved layout and cluster labels. Display options include `figsize`, `show`,
`seed`, and a lower `max_points` display cap; they change only the picture.

```{code-cell} ipython3
result.plot_embedding(figsize=(9, 6))
```

```{code-cell} ipython3
marker_table = result.get_markers()
marker_table.sort_values(
    ["group_id", "score"], ascending=[True, False],
).groupby("group_id", sort=True).head(2)[
    ["group_id", "feature_name", "score", "frac_exp"]
].head(12)
```

These markers describe clusters. Replicate-aware differential expression and validated cell
identities require additional analysis.

```{code-cell} ipython3
report_path = result.report()
{"report": report_path.name, "exists": report_path.is_file()}
```

The single report page opens on the final map, population counts, and decisions. Alternatives and
recorded measurements sit beside each choice; marker findings and material limitations remain
visible. There is no separate technical-report application.

## Large datasets and saved work

Above 50,000 retained cells, candidate settings are screened on an immutable uniform sample.
Insufficient representation can trigger one enlargement to 100,000 cells. The sample is a tuning
cohort, not a new final cohort: the selected settings are executed and assessed on all QC-retained
cells before finalization. Sample measurements do not prove that rare populations or batch
correction will transfer. If sample coverage is inadequate, the workflow assesses a bounded
full-cohort baseline instead of deleting poorly represented groups.

The default advanced limits permit 24 candidate evaluations per screening population and 48
across screening populations. Additional final validation permits four full-cohort graphs,
eight partitions, and one targeted repair. When screening includes every retained cell, these
are all-cell comparisons; their exact artifacts can be reused for final validation. The
additional-validation allowance is not a cap on all graphs built during all-cell comparisons.
They count distinct admitted work, including failed attempts. Reuse of a complete exact artifact
does not spend another slot. These limits do not promise an elapsed time: ingest, QC, diagnostics,
markers, and one final UMAP also have costs.

One orchestration history owns the request, evidence, decisions, and final artifact references.
An identical call reuses a completed result or resumes matching interrupted work. Changed data,
metadata roles, model identity, or configuration cannot silently reinterpret that history. Older
agent runs without the mandatory study and comparison evidence must be restarted; they cannot
resume or regenerate a report under this contract. Their historical HTML remains available, and
their numerical artifacts remain readable through the ordinary Scarf APIs.

## Failure handling and advanced control

Use the exception's result address when an unattended analysis needs investigation:

```python
from scarf.agent import AnalysisError, analyze_rna

try:
    result = analyze_rna(
        "study.zarr", model=model,
        study_context="The observed study design and metadata roles.",
        study_objective="The biological structure that should be retained.",
    )
except AnalysisError as error:
    print(error)
    print(error.result.notes)
    raise
```

Advanced callers can import `AgentOrchestrator` and its request/configuration models from
`scarf.agent.orchestrator`, set an existing-store workspace, and use `inputPolicy="pause"` for
explicit questions. The advanced result still carries status and resume information. Supply only
grounded answers to the saved questions. A work limit pauses or fails the analysis; it does not
turn an unsupported candidate into an accepted result.

For live analysis, replace the `FunctionModel` with your configured Pydantic AI model and use the
same `analyze_rna` call. Scarf sends diagnostic images when the model supports them. Other models
assess the structured marker, loading-gene, and numerical evidence, with that limitation recorded
in the report. Credentials belong in the provider configuration, not in a study paragraph or
saved analysis record.
