"""Ground and validate data enrichment reports."""

import re

from ...features.gene_reference import species_registry
from ...utils.logging import logger
from .._deps import AGENT_INSTALL_HINT
from ..types import AgentRunInfo
from .contracts import (
    DataEnrichmentContext,
    DataEnrichmentDependencies,
    DataEnrichmentReport,
    FeatureReference,
    FeatureSelectionPolicy,
    StudyContextSummary,
)

try:
    from pydantic_ai import UnexpectedModelBehavior, UsageLimitExceeded
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

_SUPPORTED_SPECIES = species_registry()


def _ground_study_context_summary(
    context: DataEnrichmentContext,
    proposed: StudyContextSummary,
) -> StudyContextSummary:
    """Bind structured context references to exact caller text."""
    original_context = context.studyContext
    original_objective = context.studyObjective
    grounded_text = f"{original_context}\n{original_objective}"
    organism_references = [context.organismHint] if context.organismHint else []
    for species in _SUPPORTED_SPECIES.values():
        match = re.search(
            rf"\b{re.escape(species.label)}\b",
            grounded_text,
            flags=re.IGNORECASE,
        )
        if match is not None:
            organism_references.append(match.group(0))
    field_sources = {
        "organismReferences": organism_references,
        "tissueReferences": list(context.tissueReferences),
        "cellTypeReferences": list(context.cellTypeReferences),
        "experimentalReferences": list(context.experimentalDetails),
        "hypothesisReferences": [],
        "analysisIntentReferences": [],
    }
    grounded: dict[str, list[str]] = {}
    for field_name, supplied_values in field_sources.items():
        exact_supplied = [value.strip() for value in supplied_values if value.strip()]
        proposed_values = list(getattr(proposed, field_name))
        combined = list(
            dict.fromkeys(
                value.strip()
                for value in [*exact_supplied, *proposed_values]
                if value.strip()
            )
        )
        if len(combined) > 12:
            raise ValueError(
                f"studyContextSummary.{field_name} may contain at most 12 values"
            )
        supplied = set(exact_supplied)
        invalid = [
            value
            for value in combined
            if value not in supplied and value not in grounded_text
        ]
        if invalid:
            raise ValueError(
                f"Study-context references must be verbatim caller text: {invalid}"
            )
        oversized = [value for value in combined if len(value) > 240]
        if oversized:
            raise ValueError("Study-context references may not exceed 240 characters")
        grounded[field_name] = combined

    evidence_ids: list[str] = []
    if original_context:
        evidence_ids.append("context:study")
    if original_objective:
        evidence_ids.append("context:objective")
    if context.organismHint:
        evidence_ids.append("context:organism")
    evidence_ids.extend(
        f"context:tissue:{index}"
        for index, _value in enumerate(context.tissueReferences)
    )
    evidence_ids.extend(
        f"context:cellType:{index}"
        for index, _value in enumerate(context.cellTypeReferences)
    )
    evidence_ids.extend(
        f"context:experiment:{index}"
        for index, _value in enumerate(context.experimentalDetails)
    )
    return StudyContextSummary(
        studyContext=original_context,
        studyObjective=original_objective,
        **grounded,
        evidenceIds=evidence_ids,
    )


def _validate_feature_policy(
    deps: DataEnrichmentDependencies,
    policy: FeatureSelectionPolicy,
    grounded_context: StudyContextSummary,
) -> None:
    """Ground and validate one feature policy in deterministic order."""
    supported_species = {*_SUPPORTED_SPECIES, "unknown"}
    if policy.species not in supported_species:
        raise ValueError(
            f"unsupported species {policy.species!r}; choose a supported key or unknown"
        )
    if not policy.evidenceIds:
        raise ValueError(f"policy for assay {policy.assay!r} requires evidence IDs")
    policy.organismName = (
        _SUPPORTED_SPECIES[policy.species].label
        if policy.species in _SUPPORTED_SPECIES
        else "unknown"
    )
    inspection = deps.inspections.get(policy.assay)
    if inspection is None:
        raise ValueError(f"assay {policy.assay!r} was not inspected")
    modality = inspection.modalityEvidence
    policy.assayType = modality.assayType
    policy.assayModality = modality.modality
    policy.graphEligible = modality.graphEligible
    policy.markerEligible = modality.markerEligible
    policy.demultiplexEligible = modality.demultiplexEligible
    policy.exactControlFeatures = [
        FeatureReference(
            featureId=item.featureId,
            featureName=item.featureName,
        )
        for item in modality.adtControls
    ]
    policy.exactTagFeatures = [
        FeatureReference(
            featureId=item.featureId,
            featureName=item.featureName,
        )
        for item in modality.htoTags
    ]
    policy.peakCoordinateStatus = modality.atacCoordinates.status
    policy.evidenceIds = list(
        dict.fromkeys([*policy.evidenceIds, *modality.evidenceIds])
    )
    if (
        inspection.species in _SUPPORTED_SPECIES
        and policy.species != inspection.species
    ):
        raise ValueError(
            f"policy species {policy.species!r} conflicts with inspected "
            f"species {inspection.species!r}"
        )
    if (
        inspection.species == "unknown"
        and policy.species != "unknown"
        and not any(
            evidence_id.startswith("context:") for evidence_id in policy.evidenceIds
        )
    ):
        raise ValueError(
            "A context-derived species decision must cite context evidence"
        )
    observed_families = {item.family for item in inspection.families}
    cited_families = set(policy.excludeFamilies) | set(policy.protectFamilies)
    unknown_families = cited_families - observed_families
    if unknown_families:
        raise ValueError(
            f"policy cites unobserved families: {sorted(unknown_families)}"
        )
    protected_defaults = {
        item.family for item in inspection.families if item.defaultExclude is False
    }
    excluded_protected = sorted(
        set(policy.excludeFamilies).intersection(protected_defaults)
    )
    if excluded_protected:
        raise ValueError(
            "The initial enrichment policy cannot exclude families that "
            f"deterministic evidence protects by default: {excluded_protected}"
        )
    confirmed = deps.confirmedFeatures.get(policy.assay, set())
    cited_features = {
        *policy.excludeFeatures,
        *policy.protectFeatures,
        *policy.artificialFeatures,
    }
    unknown_features = cited_features - confirmed
    if unknown_features:
        raise ValueError(
            "Call find_present_features_batch before citing individual features: "
            f"{sorted(unknown_features)}"
        )
    exogenous_evidence = {
        value: item.evidenceId
        for item in inspection.exogenous
        for value in (item.featureId, item.featureName)
    }
    unsupported_artificial: list[str] = []
    for feature in policy.artificialFeatures:
        evidence_id = exogenous_evidence.get(feature)
        if evidence_id is not None and evidence_id in policy.evidenceIds:
            continue
        matching_context_ids = {
            f"context:experiment:{index}"
            for index, detail in enumerate(deps.context.experimentalDetails)
            if feature.casefold() in detail.casefold()
        }
        if matching_context_ids.intersection(policy.evidenceIds):
            continue
        unsupported_artificial.append(feature)
    if unsupported_artificial:
        raise ValueError(
            "Artificial features require their exogenous evidence ID or a "
            "feature-specific experimental-context evidence ID: "
            f"{sorted(unsupported_artificial)}"
        )
    unknown_evidence = set(policy.evidenceIds) - deps.evidenceIds
    if unknown_evidence:
        raise ValueError(
            f"policy cites unknown evidence IDs: {sorted(unknown_evidence)}"
        )
    policy.tissueReferences = list(grounded_context.tissueReferences)
    policy.cellTypeReferences = list(grounded_context.cellTypeReferences)
    policy.experimentalReferences = list(grounded_context.experimentalReferences)


def validate_data_enrichment_report(
    deps: DataEnrichmentDependencies,
    report: DataEnrichmentReport,
) -> DataEnrichmentReport:
    """Ground an agent report in inspected assays, context, and exact lookups."""
    if not deps.inspections:
        raise ValueError("Inspect every requested assay before returning the report")

    requested = set(deps.assays)
    reported = {policy.assay for policy in report.policies}
    if len(reported) != len(report.policies):
        raise ValueError("reports may contain only one policy for each assay")
    if not reported.issubset(requested):
        raise ValueError(
            f"policies cite assays outside the requested set: {sorted(reported - requested)}"
        )
    if report.status == "done" and reported != requested:
        raise ValueError(
            f"done reports require one policy for every requested assay: {deps.assays}"
        )

    grounded_context = _ground_study_context_summary(
        deps.context,
        report.studyContextSummary,
    )
    for policy in report.policies:
        _validate_feature_policy(deps, policy, grounded_context)

    report.studyContextSummary = grounded_context
    report.inspections = [deps.inspections[name] for name in deps.assays]
    report.toolCalls = list(deps.toolCalls)
    report.evidenceIds = list(
        dict.fromkeys(
            evidence_id
            for evidence_id in [
                *report.studyContextSummary.evidenceIds,
                *(
                    evidence_id
                    for policy in report.policies
                    for evidence_id in policy.evidenceIds
                ),
            ]
        )
    )
    logger.debug(
        "Data Enrichment report validated: "
        f"status={report.status}, policies={len(report.policies)}, "
        f"inspections={len(report.inspections)}, "
        f"toolCalls={len(report.toolCalls)}, evidence={len(report.evidenceIds)}"
    )
    return report


def pending_data_enrichment_report(
    deps: DataEnrichmentDependencies,
    *,
    error: UnexpectedModelBehavior | UsageLimitExceeded,
    model_name: str,
) -> DataEnrichmentReport:
    """Pause after deterministic inspection when no valid policy was selected."""
    if set(deps.inspections) != set(deps.assays):
        raise error
    error_detail = str(error).replace("\n", " ").strip()[:500]
    report = DataEnrichmentReport(
        status="needsInput",
        studyContextSummary=StudyContextSummary.get_blank(),
        unresolvedQuestions=[
            "The Data Enrichment agent did not produce a validated feature policy. "
            "Provide explicit organism and representation-feature intent."
        ],
        limitations=[
            "No scientific feature policy was selected after model failure.",
            error_detail,
        ],
        runInfo=AgentRunInfo(
            agentName="data_enrichment_needs_input",
            modelName=model_name,
        ),
    )
    validated = validate_data_enrichment_report(deps, report)
    logger.warning(
        "Data Enrichment paused without a scientific selection: "
        f"assays={len(validated.inspections)}, evidence={len(validated.evidenceIds)}, "
        f"reason={error_detail}"
    )
    return validated


def deterministic_data_enrichment_report(
    deps: DataEnrichmentDependencies,
    *,
    error: Exception,
    model_name: str,
) -> DataEnrichmentReport:
    """Use inspected feature evidence when an unattended model run is invalid."""
    if set(deps.inspections) != set(deps.assays):
        raise error
    policies = []
    for assay in deps.assays:
        inspection = deps.inspections[assay]
        evidence_ids = list(inspection.evidenceIds)
        if not evidence_ids:
            raise ValueError(f"Assay {assay!r} has no deterministic feature evidence")
        policies.append(
            FeatureSelectionPolicy(
                assay=assay,
                species=(
                    inspection.species
                    if inspection.species in {*_SUPPORTED_SPECIES, "unknown"}
                    else "unknown"
                ),
                speciesConfidence=(
                    "high" if inspection.species in _SUPPORTED_SPECIES else "unknown"
                ),
                speciesRationale=(
                    inspection.speciesReason
                    or "Feature inspection did not resolve a supported species."
                ),
                excludeFamilies=[
                    item.family
                    for item in inspection.families
                    if item.defaultExclude is True
                ],
                protectFamilies=[
                    item.family
                    for item in inspection.families
                    if item.defaultExclude is False
                ],
                rationale=(
                    "Use the exact observed default-exclusion families as the "
                    "initial representation-sensitivity policy."
                ),
                evidenceIds=evidence_ids,
            )
        )
    summary = StudyContextSummary(
        organismReferences=(
            [deps.context.organismHint] if deps.context.organismHint else []
        ),
        tissueReferences=list(deps.context.tissueReferences),
        cellTypeReferences=list(deps.context.cellTypeReferences),
        experimentalReferences=list(deps.context.experimentalDetails),
    )
    error_detail = str(error).replace("\n", " ").strip()[:500]
    report = DataEnrichmentReport(
        status="done",
        policies=policies,
        studyContextSummary=summary,
        limitations=[
            "The model feature-policy output was invalid; the workflow used only "
            "deterministic assay inspection evidence.",
            error_detail,
        ],
        runInfo=AgentRunInfo(
            agentName="data_enrichment_deterministic",
            modelName=model_name,
        ),
    )
    return validate_data_enrichment_report(deps, report)
