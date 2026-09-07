# Agent analysis API reference

The optional `scarf[agent]` dependency provides a small interface for selecting and explaining
RNA analysis settings. The root `scarf.agent` facade exports three objects:

```{eval-rst}
.. autofunction:: scarf.agent.analyze_rna

.. autoclass:: scarf.agent.AutomatedWorkflowResult
   :members: plot_embedding, get_markers, report

.. autoexception:: scarf.agent.AnalysisError
```

## Start an analysis

```python
from scarf.agent import analyze_rna

result = analyze_rna(
    "study.zarr",
    model=model,
    study_context="Human blood from one healthy donor, with no treatment comparison.",
    study_objective="Identify stable major immune-cell populations.",
)
result.plot_embedding()
markers = result.get_markers()
report_path = result.report()
```

The four required inputs are the source, configured Pydantic AI model, study context, and study
objective. `assay` selects an RNA assay when several exist. `zarr_path` selects the destination
when converting a supported input file. Other modalities in an existing store are ignored.
Automated integration, HTO assignment, and biological significance or differential-expression
hypothesis execution are outside this workflow. Experimental Context still explores individual
and joint covariate patterns and possible explanations of the study design.

The beginner call returns only after completion. If it cannot complete, it raises `AnalysisError`,
whose `result` provides the status, notes, and exact saved address for investigation or advanced
resume. An identical repeated call reuses a completed analysis or resumes matching interrupted
work. Input and model identity checks prevent attaching changed analysis intent to saved work.

## Use the result

`plot_embedding()` displays the exact final UMAP colored by its clusters and returns Scarf's
established `PlotResult`. It accepts `figsize`, `show`, `seed`, and `max_points` from 1 to 50,000.
Sampling affects only display; cluster counts and final marker statistics describe the complete
QC-retained population. The image provenance records the workspace, artifact references,
population counts, and display sampling. No live metadata is created or overwritten.

`get_markers()` returns the saved marker table using Scarf's standard `group_id`, `min_score`,
and `min_frac_exp` filters. `report()` returns the local `index.html` path and can regenerate the
single-page summary from saved evidence. It does not open a browser or rerun the analysis.

The result stores a small address and outcome, including `zarrPath`, `workspace`, and
`workflowRunId`. Requests, scientific reports, decision histories, and final artifacts remain
owned by the orchestration journal. They are not repeated as public result fields.

## Advanced control

```python
from scarf.agent.orchestrator import (
    AgentOrchestrator,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
)

runner = AgentOrchestrator(model, config=AutomatedWorkflowConfig(inputPolicy="pause"))
result = runner.run(AutomatedWorkflowRequest(
    sourcePath="study.zarr",
    workspace="analysis",
    studyContext="The observed study design and metadata roles.",
    studyObjective="The biological structure that must be preserved.",
))
```

The advanced interface exposes numerical limits, provider limits, existing-store workspaces, and
explicit pauses. Inspect its returned status and questions before continuing. Defaults allow
50,000 screening cells, one enlargement to 100,000, 12 evaluations per screen and 24 across
screens, four full-cohort graphs, eight full-cohort partitions, and one targeted full-cohort
repair. These counts bound distinct admitted work, including failed attempts; exact reuse does
not spend another slot. They do not bound every QC, marker, I/O, or provider cost.

The previous agent workflow records, result fields, candidate-budget aliases, and root imports
are unsupported. Old agent runs must be restarted. Numerical artifacts remain accessible through
the ordinary Scarf artifact APIs; no saved records are silently migrated.

For a complete executable example, see {doc}`../../tutorials/agent_workflow`.
