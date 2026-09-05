"""Experimental-context canonicalization and fallbacks."""

from collections.abc import Mapping
from typing import Any, cast

from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..tools import artifact_reference
from ..types import AgentRunInfo, BatchCorrectionAction, BatchSafetyEvidence
from .characterization import characterize_covariates
from .contracts import (
    BatchCorrectionPlan,
    CellQcPlan,
    ColumnDomain,
    CovariateCharacterization,
    ExperimentalContextDecision,
    ExperimentalContextDependencies,
    ExperimentalContextResult,
    InferenceUnit,
    IntegrationMetric,
    characterization_evidence,
)
from .qc_evidence import (
    _artifact_evidence_id,
    _hto_artifact_map,
    _hto_identity_columns,
    _offered_qc_profiles,
)
from .tools import _batch_safety_evidence, contrast_plans_from_characterization

try:
    from pydantic_ai import ModelRetry, UnexpectedModelBehavior
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


def _canonical_cell_qc_plan(
    plan: CellQcPlan,
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization,
) -> CellQcPlan:
    """Resolve one exact offered profile and reject model-authored parameters."""
    if not deps.qcProfiles:
        _offered_qc_profiles(deps, characterization)
    directed = deps.directions.get("cellQc")
    direction_map = dict(directed) if isinstance(directed, Mapping) else {}
    directed_profile_id = direction_map.get("profileId")
    if directed_profile_id is not None and not isinstance(directed_profile_id, str):
        raise ModelRetry("cellQc.profileId direction must be a string")

    has_directed_selector = any(
        key in direction_map
        for key in (
            "profileId",
            "registeredProfile",
            "action",
            "sampleColumn",
            "sampleArtifactName",
        )
    )
    selected_id = directed_profile_id or (
        "" if has_directed_selector else plan.profileId
    )
    if not selected_id:
        requested_action = direction_map.get("action")
        requested_registered_profile = direction_map.get("registeredProfile")
        requested_sample = direction_map.get("sampleColumn")
        requested_sample_artifact = direction_map.get("sampleArtifactName")
        if requested_sample is not None and requested_sample_artifact is not None:
            raise ModelRetry(
                "cellQc directions cannot select both sampleColumn and "
                "sampleArtifactName"
            )
        if requested_sample_artifact is not None and not isinstance(
            requested_sample_artifact, str
        ):
            raise ModelRetry("cellQc.sampleArtifactName must be a string")
        if requested_action is not None and requested_action not in {
            "skip",
            "globalGaussian",
            "sampleMad",
            "registeredMad",
        }:
            raise ModelRetry(f"Unsupported cellQc.action {requested_action!r}")
        if requested_registered_profile is not None and not isinstance(
            requested_registered_profile, str
        ):
            raise ModelRetry("cellQc.registeredProfile must be a string")
        if (
            requested_registered_profile is not None
            and requested_registered_profile
            not in {
                "retainWithFlags",
                "globalMad5",
                "captureMad5",
                "captureMad3Sensitivity",
                "pooledReferenceMad5",
            }
        ):
            raise ModelRetry(
                f"Unsupported cellQc.registeredProfile {requested_registered_profile!r}"
            )
        matches = [
            profile
            for profile in deps.qcProfiles.values()
            if (requested_action is None or profile.action == requested_action)
            and (
                requested_registered_profile is None
                or profile.registeredProfile == requested_registered_profile
            )
            and (requested_sample is None or profile.sampleColumn == requested_sample)
            and (
                requested_sample_artifact is None
                or (
                    profile.sampleArtifact is not None
                    and profile.sampleArtifact.name == requested_sample_artifact
                )
            )
        ]
        if requested_action is not None or requested_registered_profile is not None:
            if len(matches) != 1:
                raise ModelRetry(
                    "cellQc directions must identify exactly one offered profile"
                )
            selected_id = matches[0].profileId
        else:
            global_profiles = [
                profile
                for profile in deps.qcProfiles.values()
                if profile.action == "globalGaussian"
            ]
            if global_profiles:
                selected_id = global_profiles[0].profileId
            else:
                selected_id = next(
                    profile.profileId
                    for profile in deps.qcProfiles.values()
                    if profile.action == "skip"
                )

    profile = deps.qcProfiles.get(selected_id)
    if profile is None:
        raise ModelRetry(
            f"Cell-QC profile {selected_id!r} was not offered by the evidence tool"
        )
    model_selected = bool(plan.profileId) and not has_directed_selector
    if model_selected:
        expected_fields = {
            "action": profile.action,
            "registeredProfile": profile.registeredProfile,
            "driverAssay": profile.driverAssay,
            "driverAssayType": profile.driverAssayType,
            "sampleColumn": profile.sampleColumn,
            "sampleArtifact": profile.sampleArtifact,
            "attributes": profile.attributes,
            "artifactMetrics": profile.artifactMetrics,
        }
        mismatches = [
            name
            for name, expected in expected_fields.items()
            if getattr(plan, name) != expected
        ]
        if mismatches:
            raise ModelRetry(
                "Cell-QC plan must copy the selected offered profile exactly: "
                f"{mismatches}"
            )
        if profile.evidenceId not in plan.evidenceIds:
            raise ModelRetry(
                "Cell-QC plan must cite its exact profile retention evidence"
            )
    rationale = plan.rationale.strip()
    if not rationale:
        rationale = (
            "Selected the caller-directed bounded cell-QC profile."
            if direction_map
            else "Selected the bounded default cell-QC profile."
        )
    cited_evidence = plan.evidenceIds if model_selected else []
    return CellQcPlan(
        action=profile.action,
        registeredProfile=profile.registeredProfile,
        profileId=profile.profileId,
        driverAssay=profile.driverAssay,
        driverAssayType=profile.driverAssayType,
        sampleColumn=profile.sampleColumn,
        sampleArtifact=profile.sampleArtifact,
        attributes=profile.attributes,
        artifactMetrics=profile.artifactMetrics,
        rationale=rationale,
        evidenceIds=sorted({*cited_evidence, profile.evidenceId}),
    )


def _validate_batch_correction_plan(
    decision: ExperimentalContextDecision,
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization,
    requested_coefficients: set[str],
    units_of_inference: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    coefficient_records: dict[str, dict[str, Any]],
) -> None:
    """Validate one batch plan against exact design, safety, and metric evidence."""
    confounding_reports = {
        report.get("coefficient"): report
        for report in characterization.confounding
        if isinstance(report.get("coefficient"), str)
    }
    plan = decision.batchCorrection
    directed_batch_columns = deps.directions.get("batchColumns")
    if directed_batch_columns is not None:
        if not isinstance(directed_batch_columns, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in directed_batch_columns
        ):
            raise ModelRetry(
                "directions.batchColumns must be a list of exact metadata columns"
            )
        canonical_directed_batch = sorted(directed_batch_columns)
        directed_plan_mismatch = (
            (
                plan.action not in {"evaluateHarmony", "unsafe"}
                or sorted(plan.batchColumns) != canonical_directed_batch
            )
            if canonical_directed_batch
            else plan.action != "skip" or bool(plan.batchColumns)
        )
        if directed_plan_mismatch:
            raise ModelRetry(
                "The batch-correction plan must assess the exact directed batch "
                f"columns: {canonical_directed_batch}"
            )
    unknown_columns = sorted(set(decision.columnDomains) - set(records))
    if unknown_columns:
        raise ModelRetry(f"Unknown column domain assignments: {unknown_columns}")
    unit_columns = {
        unit_name
        for unit in units_of_inference.values()
        for unit_name in (
            unit.get("observationUnit"),
            unit.get("independentUnit"),
        )
        if isinstance(unit_name, str)
    }
    if plan.action == "evaluateHarmony" and not plan.batchColumns:
        raise ModelRetry("evaluateHarmony requires at least one batch column")
    if plan.action == "unsafe" and not plan.batchColumns:
        raise ModelRetry("unsafe requires the exact batch columns that were assessed")
    if plan.action == "skip" and plan.batchColumns:
        raise ModelRetry("skip must not include batch columns")
    if plan.action == "needsInput" and not decision.needsInput:
        raise ModelRetry("needsInput action requires at least one concrete question")
    if len(set(plan.batchColumns)) != len(plan.batchColumns):
        raise ModelRetry("Batch columns must be unique")

    for batch_column in plan.batchColumns:
        record = records.get(batch_column)
        if record is None:
            raise ModelRetry(f"Unknown batch column {batch_column!r}")
        if record.get("domain") != "technical":
            raise ModelRetry(
                f"Batch column {batch_column!r} must be classified as technical"
            )
        if record.get("kind") != "categorical":
            raise ModelRetry(
                f"Batch column {batch_column!r} must be categorical for Harmony"
            )
        if batch_column in requested_coefficients or batch_column in unit_columns:
            raise ModelRetry(
                f"Batch column {batch_column!r} cannot be a coefficient or unit of inference"
            )

    if plan.action == "evaluateHarmony":
        mixing_metrics = {"iLISI", "proportionalBatchMixing"}
        preservation_metrics = {"cLISI", "graphConnectivity"}
        if not mixing_metrics.intersection(plan.metricsRequired):
            raise ModelRetry(
                "evaluateHarmony requires iLISI or proportionalBatchMixing"
            )
        if plan.preserveColumns and not preservation_metrics.intersection(
            plan.metricsRequired
        ):
            raise ModelRetry(
                "evaluateHarmony requires cLISI or graphConnectivity for preservation"
            )
        missing_preserve = sorted(requested_coefficients - set(plan.preserveColumns))
        if missing_preserve:
            raise ModelRetry(
                "preserveColumns must include every coefficient of interest: "
                f"{missing_preserve}"
            )
        unresolved_coefficients = sorted(
            coefficient
            for coefficient in requested_coefficients
            if coefficient_records[coefficient].get("scope") != "betweenUnit"
            or coefficient not in confounding_reports
        )
        if unresolved_coefficients:
            raise ModelRetry(
                "evaluateHarmony requires a between-unit coefficient with a "
                "matching estimability report; use needsInput or unsafe for: "
                f"{unresolved_coefficients}"
            )
        for preserve_column in plan.preserveColumns:
            record = records.get(preserve_column)
            if record is None:
                raise ModelRetry(f"Unknown preservation column {preserve_column!r}")
            if record.get("domain") != "biological":
                raise ModelRetry(
                    f"Preservation column {preserve_column!r} must be biological"
                )
            if record.get("kind") != "categorical":
                raise ModelRetry(
                    f"Preservation column {preserve_column!r} must be categorical"
                )

    matched_safety: list[BatchSafetyEvidence] = []
    if plan.action in {"evaluateHarmony", "unsafe"}:
        canonical_batch_columns = sorted(plan.batchColumns)
        for coefficient in sorted(requested_coefficients):
            coefficient_record = coefficient_records[coefficient]
            report = confounding_reports.get(coefficient)
            observation_unit = (
                report.get("observationUnit")
                if report is not None
                else coefficient_record.get("observationUnit")
            )
            unit_constant = {
                pair.get("technical")
                for pair in (report.get("pairs", []) if report is not None else [])
                if isinstance(pair.get("technical"), str)
            }
            expected_effective = [
                name for name in canonical_batch_columns if name in unit_constant
            ]
            candidates = [
                item
                for item in deps.batchSafety.values()
                if item.coefficient == coefficient
                and item.coefficientKind == coefficient_record.get("kind")
                and item.observationUnit == observation_unit
                and item.batchColumns == canonical_batch_columns
                and item.unitConstantBatchColumns == expected_effective
            ]
            if len(candidates) != 1:
                raise ModelRetry(
                    "Call analyze_experimental_design with the exact proposed batch "
                    f"columns before returning a recommendation for {coefficient!r}"
                )
            matched_safety.append(candidates[0])
        missing_safety_evidence = sorted(
            item.evidenceId
            for item in matched_safety
            if item.evidenceId not in plan.evidenceIds
        )
        if missing_safety_evidence:
            raise ModelRetry(
                "Batch-correction recommendations must cite exact batch "
                f"estimability evidence: {missing_safety_evidence}"
            )
        not_computed = [
            item.coefficient for item in matched_safety if item.status == "notComputed"
        ]
        if not_computed:
            raise ModelRetry(
                "Batch estimability could not be computed; use action='needsInput' "
                f"for: {sorted(not_computed)}"
            )
        unsafe_coefficients = [
            item.coefficient for item in matched_safety if item.status == "unsafe"
        ]
        if plan.action == "evaluateHarmony" and unsafe_coefficients:
            raise ModelRetry(
                "Batch correction is unsafe because the biological coefficient is "
                "not estimable after the exact proposed batch columns; use "
                f"action='unsafe' for: {sorted(unsafe_coefficients)}"
            )
        if plan.action == "unsafe" and not unsafe_coefficients:
            raise ModelRetry(
                "The exact proposed batch columns were estimable for every "
                "coefficient; use action='evaluateHarmony' or 'skip'"
            )

    cited_ids = [
        *decision.evidenceIds,
        *plan.evidenceIds,
    ]
    unknown_evidence = sorted(set(cited_ids) - deps.evidenceIds)
    if unknown_evidence:
        raise ModelRetry(f"Unknown evidence IDs: {unknown_evidence}")
    if plan.action in {"evaluateHarmony", "skip", "unsafe"} and not plan.evidenceIds:
        raise ModelRetry("Batch-correction recommendations require evidence IDs")
    current_metric_evidence = set(deps.currentRepresentation.evidenceIds)
    stale_metric_evidence = sorted(
        evidence_id
        for evidence_id in cited_ids
        if evidence_id.startswith("metric:")
        and evidence_id not in current_metric_evidence
    )
    if stale_metric_evidence:
        raise ModelRetry(
            "Metric evidence must come from the returned exact representation: "
            f"{stale_metric_evidence}"
        )


def validate_experimental_context(
    decision: ExperimentalContextDecision,
    deps: ExperimentalContextDependencies,
) -> ExperimentalContextDecision:
    """Recompute and validate every model-authored design choice."""
    narrative_fields = {
        "rationale": decision.rationale,
        "batchCorrection.rationale": decision.batchCorrection.rationale,
        **{
            f"needsInput[{index}]": question
            for index, question in enumerate(decision.needsInput)
        },
    }
    serialized_field_markers = (
        '"evidenceIds":',
        '"needsInput":',
        '"runInfo":',
        '"batchCorrection":',
        '"cellQc":',
    )
    invalid_narratives = [
        name
        for name, value in narrative_fields.items()
        if any(
            marker in value.replace('\\"', '"') for marker in serialized_field_markers
        )
    ]
    if invalid_narratives:
        raise ModelRetry(
            "Narrative fields must contain plain prose without serialized sibling "
            f"fields: {invalid_narratives}"
        )
    directions = dict(deps.directions)
    column_domains = dict(decision.columnDomains)
    column_domains.update(dict(directions.get("columnDomains") or {}))
    directions["columnDomains"] = column_domains
    directions["coefficientsOfInterest"] = list(
        dict.fromkeys(
            [
                *decision.coefficientsOfInterest,
                *(directions.get("coefficientsOfInterest") or []),
            ]
        )
    )
    units_of_inference = {
        name: unit.model_dump(exclude_none=True)
        for name, unit in decision.unitsOfInference.items()
    }
    units_of_inference.update(dict(directions.get("unitsOfInference") or {}))
    directions["unitsOfInference"] = units_of_inference

    characterization = characterize_covariates(
        deps.store,
        cellSelection=deps.cellSelection,
        studyContext=f"{deps.studyContext}\nStudy objective: {deps.studyObjective}",
        model=None,
        directions=directions,
        groupingArtifacts=_hto_artifact_map(deps),
    )
    if characterization.status == "failed":
        raise ModelRetry("; ".join(characterization.notes))
    deps.characterization = characterization
    deps.evidenceIds.update(characterization_evidence(characterization))
    contrast_plans = contrast_plans_from_characterization(characterization)
    deps.contrastPlans = {plan.coefficient: plan for plan in contrast_plans}
    deps.evidenceIds.update(plan.evidenceId for plan in contrast_plans)

    if "inspect_cell_covariates" not in deps.toolCalls:
        raise ModelRetry("Call inspect_cell_covariates before returning a decision")
    if "analyze_experimental_design" not in deps.toolCalls:
        raise ModelRetry("Call analyze_experimental_design before returning a decision")

    if decision.cellQc != CellQcPlan.get_blank():
        raise ModelRetry(
            "Experimental Context must leave cellQc blank; the audited filtering "
            "checkpoint selects from qcProfiles"
        )
    if not deps.qcProfiles:
        _offered_qc_profiles(deps, characterization)
    deps.evidenceIds.update(profile.evidenceId for profile in deps.qcProfiles.values())
    deps.evidenceIds.update(source.sourceId for source in deps.qcMetricSources)
    deps.evidenceIds.update(item.evidenceId for item in deps.qcSourceConcordance)

    requested_coefficients = set(directions["coefficientsOfInterest"])
    characterized_coefficients = {
        record.get("name") for record in characterization.coefficients
    }
    missing_coefficients = sorted(
        name
        for name in requested_coefficients
        if name not in characterized_coefficients
    )
    if missing_coefficients:
        raise ModelRetry(
            "Coefficients of interest must be classified as biological: "
            f"{missing_coefficients}"
        )

    coefficient_records: dict[str, dict[str, Any]] = {}
    for record in characterization.coefficients:
        name = record.get("name")
        if isinstance(name, str):
            coefficient_records[name] = record
    records: dict[str, dict[str, Any]] = {}
    for record in characterization.columns:
        name = record.get("name")
        if isinstance(name, str):
            records[name] = record
    _validate_batch_correction_plan(
        decision,
        deps,
        characterization,
        requested_coefficients,
        units_of_inference,
        records,
        coefficient_records,
    )
    canonical_domains = {
        name: records[name]["domain"]
        for name in column_domains
        if name in records
        and records[name].get("domain")
        in {
            "biological",
            "technical",
            "design",
            "ignore",
            "unknown",
        }
    }
    canonical_units = {
        coefficient: InferenceUnit(
            observationUnit=coefficient_records[coefficient].get("observationUnit"),
            independentUnit=coefficient_records[coefficient].get("independentUnit"),
        )
        for coefficient in directions["coefficientsOfInterest"]
        if coefficient in coefficient_records
    }
    validated = decision.model_copy(
        update={
            "columnDomains": canonical_domains,
            "coefficientsOfInterest": list(directions["coefficientsOfInterest"]),
            "unitsOfInference": canonical_units,
            "cellQc": CellQcPlan.get_blank(),
        }
    )
    logger.debug(
        "Experimental Context decision validated: "
        f"domains={len(validated.columnDomains)}, "
        f"coefficients={len(validated.coefficientsOfInterest)}, "
        f"qcProfiles={len(deps.qcProfiles)}, "
        f"batchCorrection={validated.batchCorrection.action}, "
        f"needsInput={len(validated.needsInput)}"
    )
    return validated


def _deterministic_experimental_context_decision(
    deps: ExperimentalContextDependencies,
) -> ExperimentalContextDecision:
    characterization = deps.characterization
    if characterization is None or characterization.status == "failed":
        raise ValueError("Deterministic covariate characterization is unavailable")
    records: dict[str, dict[str, Any]] = {}
    for record in characterization.columns:
        name = record.get("name")
        if isinstance(name, str):
            records[name] = record
    coefficient_records: dict[str, dict[str, Any]] = {}
    for record in characterization.coefficients:
        name = record.get("name")
        if isinstance(name, str):
            coefficient_records[name] = record
    directions = dict(deps.directions)
    raw_batch_columns = directions.get("batchColumns")
    if raw_batch_columns is not None:
        if not isinstance(raw_batch_columns, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in raw_batch_columns
        ):
            raise ValueError(
                "directions.batchColumns must be a list of exact metadata columns"
            )
        if len(raw_batch_columns) != len(set(raw_batch_columns)):
            raise ValueError("directions.batchColumns must be unique")
        batch_columns = list(raw_batch_columns)
    else:
        candidates = sorted(
            name
            for name, record in records.items()
            if record.get("domain") == "technical"
            and record.get("kind") == "categorical"
        )
        if "batch" in candidates:
            batch_columns = ["batch"]
        elif len(candidates) <= 1:
            batch_columns = candidates
        else:
            raise ValueError(
                "Multiple categorical technical columns remain without one exact "
                "batch condition"
            )

    coefficients = [
        str(record["name"])
        for record in characterization.coefficients
        if isinstance(record.get("name"), str)
    ]
    units = {
        coefficient: InferenceUnit(
            observationUnit=coefficient_records[coefficient].get("observationUnit"),
            independentUnit=coefficient_records[coefficient].get("independentUnit"),
        )
        for coefficient in coefficients
        if coefficient in coefficient_records
    }
    batch_safety = _batch_safety_evidence(
        deps,
        characterization,
        coefficients=coefficients,
        batch_columns=batch_columns,
    )
    unresolved_safety = [
        item.coefficient for item in batch_safety if item.status == "notComputed"
    ]
    if unresolved_safety:
        raise ValueError(
            "Batch estimability is unavailable for coefficients: "
            f"{sorted(unresolved_safety)}"
        )
    if batch_columns and any(item.status == "unsafe" for item in batch_safety):
        action: BatchCorrectionAction = "unsafe"
    elif batch_columns:
        action = "evaluateHarmony"
    else:
        action = "skip"
    categorical_coefficients = [
        coefficient
        for coefficient in coefficients
        if records[coefficient].get("kind") == "categorical"
    ]
    if action == "evaluateHarmony" and set(categorical_coefficients) != set(
        coefficients
    ):
        raise ValueError(
            "Harmony preservation requires categorical coefficients of interest"
        )

    known_evidence = sorted(characterization_evidence(characterization))
    batch_evidence = [
        *(f"column:{column}" for column in batch_columns),
        *(item.evidenceId for item in batch_safety),
    ]
    if not batch_evidence:
        batch_evidence = known_evidence[:1]
    if not batch_evidence:
        raise ValueError("No deterministic evidence supports a batch decision")
    deps.evidenceIds.update(known_evidence)
    deps.evidenceIds.update(batch_evidence)
    if "analyze_experimental_design" not in deps.toolCalls:
        deps.toolCalls.append("analyze_experimental_design")
    column_domains = {
        name: cast(ColumnDomain, record["domain"])
        for name, record in records.items()
        if record.get("domain")
        in {"biological", "technical", "design", "ignore", "unknown"}
    }
    metrics_required: list[IntegrationMetric] = []
    if action == "evaluateHarmony":
        metrics_required = ["iLISI", "proportionalBatchMixing"]
        if categorical_coefficients:
            metrics_required.extend(["cLISI", "graphConnectivity"])
    plan = BatchCorrectionPlan(
        action=action,
        batchColumns=batch_columns if action != "skip" else [],
        preserveColumns=(
            categorical_coefficients if action == "evaluateHarmony" else []
        ),
        metricsRequired=metrics_required,
        rationale=(
            "Evaluate the exact declared categorical technical batch condition "
            "against the uncorrected representation."
            if action == "evaluateHarmony"
            else "The exact batch condition is confounded with the study design."
            if action == "unsafe"
            else "No exact categorical technical batch condition was available."
        ),
        evidenceIds=sorted(set(batch_evidence)),
    )
    decision = ExperimentalContextDecision(
        columnDomains=column_domains,
        coefficientsOfInterest=coefficients,
        unitsOfInference=units,
        batchCorrection=plan,
        rationale=(
            "Deterministic covariate characterization resolved the study design "
            "after the model tool call failed."
        ),
        evidenceIds=known_evidence,
    )
    return validate_experimental_context(decision, deps)


def failed_experimental_context_result(
    deps: ExperimentalContextDependencies,
    *,
    error: Exception,
    fallback_error: Exception,
    model_name: str,
) -> ExperimentalContextResult:
    """Fail unattended execution when deterministic design evidence is insufficient."""
    characterization = deps.characterization or CovariateCharacterization(
        status="failed",
        notes=["Deterministic covariate characterization is unavailable."],
    )
    model_detail = str(error).replace("\n", " ").strip()[:500]
    fallback_detail = str(fallback_error).replace("\n", " ").strip()[:500]
    return ExperimentalContextResult(
        status="failed",
        decision=ExperimentalContextDecision(
            rationale="No validated experimental-context decision was available.",
            evidenceIds=sorted(deps.evidenceIds),
        ),
        characterization=characterization,
        cellSelection=artifact_reference(deps.cellSelection),
        cellQc=CellQcPlan.get_blank(),
        qcProfiles=list(deps.qcProfiles.values()),
        qcMetricSources=deps.qcMetricSources,
        qcSourceConcordance=deps.qcSourceConcordance,
        contrastPlans=list(deps.contrastPlans.values()),
        qualityMetricArtifacts=deps.qualityMetricArtifacts,
        htoIdentityColumns=deps.htoIdentityColumns,
        htoIdentityArtifacts=deps.htoIdentityArtifacts,
        batchSafety=list(deps.batchSafety.values()),
        currentRepresentation=deps.currentRepresentation,
        notes=[
            "The model did not produce a validated experimental-context decision.",
            f"Model failure: {model_detail}",
            f"Deterministic recovery failure: {fallback_detail}",
        ],
        runInfo=AgentRunInfo(
            agentName="experimental_context_failed",
            modelName=model_name,
        ),
    )


def pending_experimental_context_result(
    deps: ExperimentalContextDependencies,
    *,
    error: UnexpectedModelBehavior,
    model_name: str,
) -> ExperimentalContextResult:
    """Pause when the model exhausts its bounded decision budget."""
    characterization = deps.characterization
    if characterization is None:
        characterization = characterize_covariates(
            deps.store,
            cellSelection=deps.cellSelection,
            studyContext=(
                f"{deps.studyContext}\nStudy objective: {deps.studyObjective}"
            ),
            model=None,
            directions=deps.directions,
            groupingArtifacts=_hto_artifact_map(deps),
        )
        deps.characterization = characterization
    if not deps.htoIdentityColumns:
        deps.htoIdentityColumns = _hto_identity_columns(deps)
    qc_profiles = list(deps.qcProfiles.values())
    if not qc_profiles:
        qc_profiles = _offered_qc_profiles(deps, characterization)
    contrast_plans = contrast_plans_from_characterization(characterization)
    deps.contrastPlans = {plan.coefficient: plan for plan in contrast_plans}
    evidence_ids = characterization_evidence(characterization)
    evidence_ids.update(profile.evidenceId for profile in qc_profiles)
    evidence_ids.update(source.sourceId for source in deps.qcMetricSources)
    evidence_ids.update(item.evidenceId for item in deps.qcSourceConcordance)
    evidence_ids.update(plan.evidenceId for plan in contrast_plans)
    evidence_ids.update(f"htoIdentity:{column}" for column in deps.htoIdentityColumns)
    evidence_ids.update(
        _artifact_evidence_id(source) for source in deps.htoIdentityArtifacts
    )
    deps.evidenceIds.update(evidence_ids)
    question = (
        "The Experimental Context agent could not produce a validated scientific "
        "decision. Provide explicit metadata roles, units of inference, cell-QC "
        "profile, and batch-correction intent before continuing."
    )
    decision = ExperimentalContextDecision(
        batchCorrection=BatchCorrectionPlan(action="needsInput"),
        cellQc=CellQcPlan.get_blank(),
        rationale="No scientific decision was selected.",
        evidenceIds=sorted(evidence_ids),
        needsInput=[question],
    )
    error_detail = str(error).replace("\n", " ").strip()[:500]
    logger.warning(
        "Experimental Context paused without a scientific decision: "
        f"reason={error_detail}"
    )
    return ExperimentalContextResult(
        status=("failed" if characterization.status == "failed" else "needsInput"),
        decision=decision,
        characterization=characterization,
        cellSelection=artifact_reference(deps.cellSelection),
        cellQc=CellQcPlan.get_blank(),
        qcProfiles=qc_profiles,
        qcMetricSources=deps.qcMetricSources,
        qcSourceConcordance=deps.qcSourceConcordance,
        contrastPlans=contrast_plans,
        qualityMetricArtifacts=deps.qualityMetricArtifacts,
        htoIdentityColumns=deps.htoIdentityColumns,
        htoIdentityArtifacts=deps.htoIdentityArtifacts,
        batchSafety=list(deps.batchSafety.values()),
        currentRepresentation=deps.currentRepresentation,
        notes=[*characterization.notes, question, error_detail],
        runInfo=AgentRunInfo(
            agentName="experimental_context_needs_input",
            modelName=model_name,
        ),
    )
