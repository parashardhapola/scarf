"""Experimental-context agent prompts and execution."""

import json
from collections.abc import Mapping, Sequence
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from ...graph.feature_projection import graph_cell_selection
from ...metadata.selection import resolve_cell_aligned_artifact
from ...storage.refs import ArtifactRef
from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..config import AgentRunConfig
from ..config.agent_exec import run_agent_sync
from ..tools import artifact_reference, core_artifact_reference
from ..types import AgentRunInfo, StageStatus
from .characterization import _SelectionBoundCells, characterize_covariates
from .contracts import (
    CellQcPlan,
    ExperimentalContextDecision,
    ExperimentalContextDependencies,
    ExperimentalContextResult,
    NamedArtifactSource,
)
from .qc_evidence import (
    _derive_missing_percentage_artifacts,
    _hto_artifact_map,
    _qc_driver,
    _source_ref,
)
from .tools import (
    _prepare_experimental_context_tool,
    analyze_experimental_design,
    contrast_plans_from_characterization,
    inspect_cell_covariates,
    score_current_representation,
)
from .validation import (
    _deterministic_experimental_context_decision,
    failed_experimental_context_result,
    pending_experimental_context_result,
    validate_experimental_context,
)

if TYPE_CHECKING:
    from ...datastore.pipeline_run import PipelineRun

try:
    from pydantic_ai import ModelRetry, Tool, UnexpectedModelBehavior
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

_CONTEXT_LIMIT = 1200


class ExperimentalContextAgent:
    """A narrow agent for study design and batch-correction planning."""

    def __init__(
        self,
        model: Any,
        *,
        config: AgentRunConfig | None = None,
        unattended: bool = False,
    ) -> None:
        self.model = model
        self.unattended = unattended
        self.config = (config or AgentRunConfig()).with_limits(
            request_limit=9,
            tool_call_limit=5,
            output_token_limit=32768,
            timeout_seconds=600.0,
        )
        self.system_prompt = (
            dedent(
                """
            You are Scarf's Experimental Context Agent. Work only through the
            provided read-only tools and return the structured decision schema.

            Call inspect_cell_covariates exactly once. Then call
            analyze_experimental_design exactly once with all explicit domains,
            all biological coefficients, every unit of inference, and the complete
            exact batch-column set being considered. You may call
            score_current_representation at most once when an exact supplied graph
            can add evidence. Do not split metadata, coefficients, or batch columns across
            calls, and do not repeat a tool call. Pass batch_columns as a JSON array,
            including when the array contains exactly one column. Each tool is
            removed after it succeeds, so include the complete decision context in
            its single call.

            The tools return bounded cell-QC profiles projected against the exact
            shared cell selection. Do not choose a profile and leave cellQc blank.
            A later audited checkpoint selects one registered profile. Never author
            or alter numeric quality bounds. RNA is the preferred QC driver and
            ATAC is the fallback. ADT and HTO never drive automatic cell filtering.
            An exact HTO identity artifact may be used as grouping evidence. It is
            not a live metadata column and does not make HTO a QC driver.

            A batch column must be categorical and technical. Never use donor,
            sample, observation-unit, independent-unit, biological, cluster, or
            embedding columns as Harmony batch columns. A biological coefficient
            that is not estimable with the exact proposed batch columns makes
            correction unsafe. A sample or library identifier is not automatically
            technical. When no exact observed column is both categorical and
            technical, pass batch_columns=[] and recommend skipping Harmony. Every
            observation and independent unit must be an exact observed column name
            or null.
            LISI evaluates a representation; it does not identify which metadata
            column is a batch. Recommend evaluateHarmony, not application, because
            Parameter Tuning must compare exact uncorrected and corrected artifacts.

            Cite only evidenceIds returned by tools. Ask for input when study
            design cannot be resolved. The study objective is authoritative: use
            it to identify protected biological variables and the intended unit
            of inference, but do not broaden it or claim to test a hypothesis.
            Never propose Python, shell commands,
            direct Zarr access, or any datastore mutation. Every rationale and
            question must be plain prose. Never place serialized JSON, schema
            field names, or sibling output fields inside a narrative string.
            Return only fields defined by the structured output schema.
                """
            )
            .strip()
            .format()
        )

    def run(
        self,
        store: Any,
        *,
        study_context: str | None = None,
        study_objective: str | None = None,
        cell_selection: ArtifactRef | None = None,
        directions: Mapping[str, Any] | None = None,
        run: "PipelineRun | None" = None,
        neighbors: ArtifactRef | None = None,
        connectivity_map: ArtifactRef | None = None,
        quality_metric_artifacts: Sequence[NamedArtifactSource] = (),
        hto_identity_artifacts: Sequence[NamedArtifactSource] = (),
    ) -> ExperimentalContextResult:
        """Inspect one datastore and return a validated experimental-context report."""
        study_context = (study_context or "").strip()
        study_objective = (study_objective or "").strip()
        if len(study_context) > _CONTEXT_LIMIT:
            study_context = study_context[: _CONTEXT_LIMIT - 3] + "..."
        if len(study_objective) > _CONTEXT_LIMIT:
            study_objective = study_objective[: _CONTEXT_LIMIT - 3] + "..."
        direction_map = dict(directions or {})
        if run is not None:
            if (
                cell_selection is not None
                or neighbors is not None
                or connectivity_map is not None
            ):
                raise ValueError(
                    "run is mutually exclusive with explicit artifact inputs"
                )
            if getattr(run, "_owner", store) is not store:
                raise ValueError("run must be opened from this datastore")
            neighbors = run["neighbors"]
            cell_selection = run["analysis_cell_selection"]
            connectivity_map = (
                run["connectivity_map"] if "connectivity_map" in run else None
            )
        cell_selection = core_artifact_reference(cell_selection)
        neighbors = core_artifact_reference(neighbors)
        connectivity_map = core_artifact_reference(connectivity_map)
        if not isinstance(cell_selection, ArtifactRef) or (
            cell_selection.kind != "cell_selection"
            or cell_selection.scope != "datastore"
        ):
            raise TypeError(
                "cell_selection must be a datastore cell_selection ArtifactRef"
            )
        if neighbors is not None:
            if not isinstance(neighbors, ArtifactRef) or neighbors.kind != "neighbors":
                raise TypeError("neighbors must be a neighbors ArtifactRef")
            if graph_cell_selection(store.zw, neighbors) != cell_selection:
                raise ValueError(
                    "neighbors and metadata must use the same cell selection"
                )
        if connectivity_map is not None:
            if not isinstance(
                connectivity_map, ArtifactRef
            ) or connectivity_map.kind not in {
                "connectivity_map",
                "integrated_graph",
            }:
                raise TypeError(
                    "connectivity_map must be a connectivity graph ArtifactRef"
                )
            if graph_cell_selection(store.zw, connectivity_map) != cell_selection:
                raise ValueError(
                    "neighbors and connectivity_map must use the same cell selection"
                )
        quality_sources = _derive_missing_percentage_artifacts(
            store,
            cell_selection=cell_selection,
            driver=_qc_driver(store),
            quality_sources=quality_metric_artifacts,
        )
        hto_sources = list(hto_identity_artifacts)
        source_names: set[str] = set()
        for sources, expected_kind in (
            (quality_sources, "quality_metric"),
            (hto_sources, "hto_identity"),
        ):
            for source in sources:
                artifact = _source_ref(source, expected_kind=expected_kind)
                if source.name in source_names:
                    raise ValueError(
                        "Experimental Context artifact source names must be unique"
                    )
                source_names.add(source.name)
                resolve_cell_aligned_artifact(
                    store.zw,
                    artifact,
                    cell_selection=cell_selection,
                    expected_kind=expected_kind,
                )
        directed_qc = direction_map.get("cellQc")
        directed_qc_map = dict(directed_qc) if isinstance(directed_qc, Mapping) else {}
        if "cellKey" in directed_qc_map:
            raise ValueError(
                "cellQc.cellKey is unsupported; use the exact cell_selection input"
            )
        logger.info(
            "Experimental Context Agent started: "
            f"cellSelection={cell_selection.artifact_id}, "
            f"directions={len(direction_map)}, "
            f"qualityMetrics={len(quality_sources)}, "
            f"htoIdentities={len(hto_sources)}, "
            f"studyContextProvided={bool(study_context)}, "
            f"studyObjectiveProvided={bool(study_objective)}"
        )
        deps = ExperimentalContextDependencies(
            store=store,
            cells=_SelectionBoundCells(
                store.zw,
                store.cells,
                cell_selection,
                artifacts={
                    source.name: _source_ref(
                        source,
                        expected_kind="hto_identity",
                    )
                    for source in hto_sources
                },
            ),
            neighbors=neighbors,
            connectivityMap=connectivity_map,
            cellSelection=cell_selection,
            studyContext=study_context,
            studyObjective=study_objective,
            directions=direction_map,
            qualityMetricArtifacts=quality_sources,
            htoIdentityArtifacts=hto_sources,
        )
        user_prompt = (
            dedent(
                """
                Characterize this experiment's metadata and decide whether Harmony
                should be evaluated. Return cell-QC candidates as tool evidence;
                leave cellQc blank for the later audited filtering checkpoint.

                Study context: {study_context}
                Study objective: {study_objective}
                Exact cell-selection artifact: {cell_selection}
                Exact quality-metric artifacts: {quality_metrics}
                Exact HTO identity artifacts: {hto_identities}
                Caller directions: {directions}
                """
            )
            .strip()
            .format(
                study_context=study_context or "not provided",
                study_objective=study_objective or "not provided",
                cell_selection=cell_selection.artifact_id,
                quality_metrics=json.dumps(
                    [source.model_dump(mode="json") for source in quality_sources],
                    sort_keys=True,
                ),
                hto_identities=json.dumps(
                    [source.model_dump(mode="json") for source in hto_sources],
                    sort_keys=True,
                ),
                directions=json.dumps(direction_map, sort_keys=True, default=str),
            )
        )
        try:
            execution = run_agent_sync(
                model=self.model,
                output_type=ExperimentalContextDecision,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                tools=(
                    Tool(
                        inspect_cell_covariates,
                        prepare=_prepare_experimental_context_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                    Tool(
                        analyze_experimental_design,
                        max_retries=3,
                        prepare=_prepare_experimental_context_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                    Tool(
                        score_current_representation,
                        prepare=_prepare_experimental_context_tool,
                        sequential=self.config.sequentialTools,
                        timeout=self.config.timeoutSeconds,
                    ),
                ),
                deps_type=ExperimentalContextDependencies,
                deps=deps,
                config=self.config,
                name="experimental_context",
                output_validator=lambda decision: validate_experimental_context(
                    decision,
                    deps,
                ),
            )
        except UnexpectedModelBehavior as exc:
            model_name = getattr(self.model, "model_name", type(self.model).__name__)
            if self.unattended:
                try:
                    decision = _deterministic_experimental_context_decision(deps)
                except (
                    ModelRetry,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as fallback_exc:
                    return failed_experimental_context_result(
                        deps,
                        error=exc,
                        fallback_error=fallback_exc,
                        model_name=str(model_name),
                    )
                run_info = AgentRunInfo(
                    agentName="experimental_context_deterministic",
                    modelName=str(model_name),
                )
            else:
                return pending_experimental_context_result(
                    deps,
                    error=exc,
                    model_name=str(model_name),
                )
        else:
            decision = ExperimentalContextDecision.model_validate(execution.output)
            run_info = execution.runInfo
        if self.unattended and (
            decision.needsInput or decision.batchCorrection.action == "needsInput"
        ):
            try:
                decision = _deterministic_experimental_context_decision(deps)
            except (ModelRetry, RuntimeError, TypeError, ValueError) as fallback_exc:
                model_name = getattr(
                    self.model, "model_name", type(self.model).__name__
                )
                return failed_experimental_context_result(
                    deps,
                    error=RuntimeError(
                        "The model returned an unresolved experimental-context decision"
                    ),
                    fallback_error=fallback_exc,
                    model_name=str(model_name),
                )
            run_info = AgentRunInfo(
                agentName="experimental_context_deterministic",
                modelName=getattr(
                    self.model,
                    "model_name",
                    type(self.model).__name__,
                ),
            )
        characterization = deps.characterization
        if characterization is None:
            characterization = characterize_covariates(
                store,
                cellSelection=cell_selection,
                studyContext=(f"{study_context}\nStudy objective: {study_objective}"),
                model=None,
                directions=direction_map,
                groupingArtifacts=_hto_artifact_map(deps),
            )
        if characterization.status == "failed":
            status: StageStatus = "failed"
        elif decision.needsInput or decision.batchCorrection.action == "needsInput":
            status = "needsInput"
        else:
            status = "done"
        logger.info(
            "Experimental Context Agent completed: "
            f"status={status}, qcProfiles={len(deps.qcProfiles)}, "
            f"batchCorrection={decision.batchCorrection.action}, "
            f"coefficients={len(decision.coefficientsOfInterest)}, "
            f"toolCalls={len(deps.toolCalls)}, evidence={len(deps.evidenceIds)}"
        )
        contrast_plans = list(deps.contrastPlans.values())
        if not contrast_plans:
            contrast_plans = contrast_plans_from_characterization(characterization)
        return ExperimentalContextResult(
            status=status,
            decision=decision,
            characterization=characterization,
            cellSelection=artifact_reference(cell_selection),
            cellQc=CellQcPlan.get_blank(),
            qcProfiles=list(deps.qcProfiles.values()),
            qcMetricSources=deps.qcMetricSources,
            qcSourceConcordance=deps.qcSourceConcordance,
            contrastPlans=contrast_plans,
            qualityMetricArtifacts=deps.qualityMetricArtifacts,
            htoIdentityColumns=deps.htoIdentityColumns,
            htoIdentityArtifacts=deps.htoIdentityArtifacts,
            batchSafety=list(deps.batchSafety.values()),
            currentRepresentation=deps.currentRepresentation,
            notes=[*characterization.notes, *decision.needsInput],
            runInfo=run_info,
        )
