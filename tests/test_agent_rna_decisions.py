"""Tests for deterministic RNA decision definitions and compilation."""

import pytest
from pydantic import ValidationError

from scarf.agent.decisions.kernel import (
    DecisionEvidence,
    DecisionRecord,
    EvidenceBundle,
)
from scarf.agent.decisions.rna import (
    ClusterExecutorPayload,
    CorrectionOutcomeExecutorPayload,
    GraphExecutorPayload,
    HvgExecutorPayload,
    PcaPrefixExecutorPayload,
    RNA_DECISION_TRANSITION_GRAPH,
    RnaDecisionCompilationError,
    RnaDecisionGateError,
    RnaDecisionRegistry,
    RnaDecisionTransition,
    RnaDecisionTransitionGraph,
    build_cell_quality_decision,
    build_cluster_partition_decision,
    build_correction_license_decision,
    build_correction_need_decision,
    build_correction_outcome_decision,
    build_feature_policy_decision,
    build_graph_k_decision,
    build_hvg_count_decision,
    build_pca_prefix_decision,
    build_qc_grouping_decision,
    compile_rna_decision,
)


def _bundle(
    decision_id: str,
    bundle_id: str,
    classes: list[str],
) -> EvidenceBundle:
    return EvidenceBundle(
        bundleId=bundle_id,
        decisionId=decision_id,
        evidence=[
            DecisionEvidence(
                evidenceId=f"evidence:{evidence_class}:{index}",
                evidenceClass=evidence_class,
                summary=f"Observed {evidence_class} evidence.",
            )
            for index, evidence_class in enumerate(classes)
        ],
    )


def _record(
    definition: object,
    bundle: EvidenceBundle,
    option_id: str,
    *,
    source: str = "agent",
    evidence_ids: list[str] | None = None,
    override_of: str | None = None,
    override_evidence_ids: list[str] | None = None,
) -> DecisionRecord:
    bundle = bundle.with_content_sha256()
    assert bundle.contentSha256 is not None
    spec = definition.spec
    option = spec.option_by_id()[option_id]
    return DecisionRecord(
        recordId=f"record:{spec.decisionId}:1",
        decisionId=spec.decisionId,
        definitionVersion=spec.definitionVersion,
        evidenceBundleId=bundle.bundleId,
        evidenceBundleSha256=bundle.contentSha256,
        offeredOptionIds=[item.optionId for item in spec.options],
        availableEvidenceIds=[item.evidenceId for item in bundle.evidence],
        selectedOptionId=option_id,
        status=option.status,
        source=source,
        evidenceIds=evidence_ids
        if evidence_ids is not None
        else [item.evidenceId for item in bundle.evidence],
        rationale="The exact cited evidence supports this registered option.",
        confidence="medium",
        overrideOfOptionId=override_of,
        overrideEvidenceIds=override_evidence_ids or [],
        verificationId=f"verification:record:{spec.decisionId}:1",
    )


def test_cell_quality_registry_gates_capture_profiles() -> None:
    global_grouping = build_qc_grouping_decision(
        evidence_bundle_id="bundle:cellQuality",
        physical_capture_eligible=False,
        pooled_reference_eligible=False,
    )

    assert [option.optionId for option in global_grouping.spec.options] == [
        "qcGrouping:global",
        "qcGrouping:defer",
    ]
    global_only = build_cell_quality_decision(
        evidence_bundle_id="bundle:cellQuality",
        available_profiles=["retainWithFlags", "globalMad5"],
    )
    assert [option.optionId for option in global_only.spec.options] == [
        "cellQuality:retainWithFlags",
        "cellQuality:globalMad5",
        "cellQuality:defer",
    ]
    global_payload = global_only.executor_option("cellQuality:globalMad5").payload
    assert global_payload.operation == "cellQualityProfile"
    assert global_payload.lowerCountMad == 5.0
    assert global_payload.flagHighCounts is True

    with pytest.raises(RnaDecisionGateError, match="physical capture"):
        build_qc_grouping_decision(
            evidence_bundle_id="bundle:cellQuality",
            physical_capture_eligible=False,
            pooled_reference_eligible=True,
        )


def test_cell_quality_registry_offers_only_eligible_pooled_reference() -> None:
    definition = build_cell_quality_decision(
        evidence_bundle_id="bundle:cellQuality",
        available_profiles=[
            "retainWithFlags",
            "captureMad5",
            "captureMad3Sensitivity",
            "pooledReferenceMad5",
        ],
    )

    assert "cellQuality:captureMad5" in definition.spec.option_by_id()
    assert "cellQuality:captureMad3Sensitivity" in definition.spec.option_by_id()
    assert "cellQuality:pooledReferenceMad5" in definition.spec.option_by_id()


def test_hvg_counts_are_capped_and_numeric_values_stay_in_payloads() -> None:
    definition = build_hvg_count_decision(
        evidence_bundle_id="bundle:hvg",
        eligible_feature_count=1500,
        ranking_mode="global",
    )

    assert [option.optionId for option in definition.spec.options] == [
        "hvgCount:focused",
        "hvgCount:allEligible",
        "hvgCount:defer",
    ]
    assert definition.spec.baselineOptionId == "hvgCount:allEligible"
    payload = definition.executor_option("hvgCount:allEligible").payload
    assert isinstance(payload, HvgExecutorPayload)
    assert payload.topN == 1500
    assert "topN" not in definition.spec.model_dump_json()


def test_batch_aware_hvgs_require_two_valid_technical_groups() -> None:
    with pytest.raises(RnaDecisionGateError, match="at least two"):
        build_hvg_count_decision(
            evidence_bundle_id="bundle:hvg",
            eligible_feature_count=4000,
            ranking_mode="batchAware",
            valid_technical_groups=1,
        )

    definition = build_hvg_count_decision(
        evidence_bundle_id="bundle:hvg",
        eligible_feature_count=4000,
        ranking_mode="batchAware",
        valid_technical_groups=2,
    )
    payload = definition.executor_option("hvgCount:standard").payload
    assert isinstance(payload, HvgExecutorPayload)
    assert payload.rankingMode == "batchAware"


def test_feature_policy_requires_dominance_and_blocks_protected_families() -> None:
    with pytest.raises(RnaDecisionGateError, match="dominance"):
        build_feature_policy_decision(
            evidence_bundle_id="bundle:features",
            proposed_exclusion_families=["ribosomal"],
            dominant_families=[],
            protected_families=[],
        )

    with pytest.raises(RnaDecisionGateError, match="protected"):
        build_feature_policy_decision(
            evidence_bundle_id="bundle:features",
            proposed_exclusion_families=["immuneReceptor"],
            dominant_families=["immuneReceptor"],
            protected_families=["immuneReceptor"],
        )

    definition = build_feature_policy_decision(
        evidence_bundle_id="bundle:features",
        proposed_exclusion_families=["ribosomal"],
        dominant_families=["ribosomal"],
        protected_families=["immuneReceptor"],
    )
    payload = definition.executor_option("featurePolicy:excludeEligibleBundle").payload
    assert payload.operation == "featurePolicy"
    assert payload.excludedFamilies == ["ribosomal"]
    assert definition.spec.baselineOptionId == "featurePolicy:keepAll"


def test_pca_prefixes_are_capped_by_rank_and_compile_to_executor_payload() -> None:
    definition = build_pca_prefix_decision(
        evidence_bundle_id="bundle:pca", matrix_rank=24
    )
    assert [option.optionId for option in definition.spec.options] == [
        "pcaPrefix:short",
        "pcaPrefix:standard",
        "pcaPrefix:maximumAvailable",
        "pcaPrefix:defer",
    ]
    assert definition.spec.baselineOptionId == "pcaPrefix:standard"
    bundle = _bundle("pcaPrefix", "bundle:pca", ["geometric", "technical"])
    record = _record(definition, bundle, "pcaPrefix:standard")

    compiled = compile_rna_decision(definition, bundle, record)

    assert isinstance(compiled.executorPayload, PcaPrefixExecutorPayload)
    assert compiled.executorPayload.dimensions == 20
    record_json = record.model_dump_json()
    assert "dimensions" not in record_json
    assert "20" not in record_json


def test_correction_license_is_rule_owned_and_need_requires_safe_license() -> None:
    unsafe = build_correction_license_decision(
        evidence_bundle_id="bundle:license", license="unsafeConfounded"
    )

    assert unsafe.spec.allowedSources == ["rule"]
    assert unsafe.spec.options[0].status == "skip"
    assert unsafe.executorOptions[0].payload.license == "unsafeConfounded"
    with pytest.raises(RnaDecisionGateError, match="safe correction license"):
        build_correction_need_decision(
            evidence_bundle_id="bundle:need", license="unsafeConfounded"
        )


def test_harmony_is_offered_only_when_safe_and_needed() -> None:
    unsafe = build_correction_outcome_decision(
        evidence_bundle_id="bundle:outcome",
        license="unsafeConfounded",
    )
    assert [option.optionId for option in unsafe.spec.options] == [
        "correctionOutcome:retainNative",
        "correctionOutcome:indeterminate",
    ]
    assert unsafe.spec.baselineOptionId == "correctionOutcome:retainNative"

    safe = build_correction_outcome_decision(
        evidence_bundle_id="bundle:outcome",
        license="safe",
        need="needed",
    )
    assert [option.optionId for option in safe.spec.options] == [
        "correctionOutcome:retainNative",
        "correctionOutcome:acceptHarmony",
        "correctionOutcome:indeterminate",
    ]
    harmony_payload = safe.executor_option("correctionOutcome:acceptHarmony").payload
    assert isinstance(harmony_payload, CorrectionOutcomeExecutorPayload)
    assert harmony_payload.useHarmony is True


@pytest.mark.parametrize(
    ("license", "need", "message"),
    [
        ("indeterminate", None, "Indeterminate correction license"),
        ("safe", None, "requires an evaluated correction need"),
        ("safe", "indeterminate", "Indeterminate correction need"),
        ("unsafeConfounded", "needed", "must not bypass"),
    ],
)
def test_correction_outcome_rejects_unsafe_or_indeterminate_bypass(
    license: str, need: str | None, message: str
) -> None:
    with pytest.raises(RnaDecisionGateError, match=message):
        build_correction_outcome_decision(
            evidence_bundle_id="bundle:outcome",
            license=license,
            need=need,
        )


def test_graph_candidates_are_capped_and_deduplicated() -> None:
    definition = build_graph_k_decision(evidence_bundle_id="bundle:graph", n_cells=15)

    assert [option.optionId for option in definition.spec.options] == [
        "graphScale:local",
        "graphScale:maximumAvailable",
        "graphScale:defer",
    ]
    assert definition.spec.baselineOptionId == "graphScale:maximumAvailable"
    payload = definition.executor_option("graphScale:maximumAvailable").payload
    assert isinstance(payload, GraphExecutorPayload)
    assert payload.neighborsK == 14


def test_clustering_uses_fixed_resolutions_and_requires_override_evidence() -> None:
    definition = build_cluster_partition_decision(
        evidence_bundle_id="bundle:cluster",
        metric_preferred_option_id="clusterResolution:balanced",
    )
    assert definition.spec.requireIndependentOverrideEvidence is True
    assert definition.spec.options[-1].optionId == "clusterPartition:abstain"
    payload = definition.executor_option("clusterResolution:detailed").payload
    assert isinstance(payload, ClusterExecutorPayload)
    assert payload.leidenResolution == 1.0

    bundle = _bundle(
        "clusterPartition",
        "bundle:cluster",
        ["geometric", "markerCoherence", "resamplingStability"],
    )
    geometric, marker, stability = [item.evidenceId for item in bundle.evidence]
    insufficient = _record(
        definition,
        bundle,
        "clusterResolution:detailed",
        evidence_ids=[geometric, marker],
        override_of="clusterResolution:balanced",
        override_evidence_ids=[marker],
    )
    with pytest.raises(
        RnaDecisionCompilationError, match="independentOverrideEvidence"
    ):
        compile_rna_decision(definition, bundle, insufficient)

    supported = _record(
        definition,
        bundle,
        "clusterResolution:detailed",
        evidence_ids=[geometric, marker, stability],
        override_of="clusterResolution:balanced",
        override_evidence_ids=[marker, stability],
    )
    compiled = compile_rna_decision(definition, bundle, supported)
    assert compiled.verification.status == "passed"


def test_clustering_can_abstain_without_inventing_a_resolution() -> None:
    definition = build_cluster_partition_decision(
        evidence_bundle_id="bundle:cluster",
        metric_preferred_option_id="clusterResolution:balanced",
    )
    bundle = _bundle("clusterPartition", "bundle:cluster", ["geometric"])
    record = _record(definition, bundle, "clusterPartition:abstain")

    compiled = compile_rna_decision(definition, bundle, record)

    assert compiled.status == "abstain"
    assert compiled.executorPayload.operation == "noExecution"
    assert compiled.executorPayload.reasonCode == "scientificAbstention"


def test_transition_graph_is_forward_only_and_routes_terminal_states() -> None:
    assert RNA_DECISION_TRANSITION_GRAPH.resolve("qcGrouping", "apply") == (
        "cellQuality",
        None,
    )
    assert RNA_DECISION_TRANSITION_GRAPH.resolve("cellQuality", "skip") == (
        "featurePolicy",
        None,
    )
    assert RNA_DECISION_TRANSITION_GRAPH.resolve("correctionLicense", "defer") == (
        None,
        "needsInput",
    )
    assert RNA_DECISION_TRANSITION_GRAPH.resolve("clusterPartition", "abstain") == (
        None,
        "abstained",
    )

    with pytest.raises(ValidationError, match="strictly forward"):
        RnaDecisionTransitionGraph(
            transitions=[
                RnaDecisionTransition(
                    fromCheckpoint="pcaPrefix",
                    onStatus="apply",
                    toCheckpoint="hvgCount",
                )
            ]
        )


def test_registry_requires_ordered_definitions_and_transition_coverage() -> None:
    cell_quality = build_cell_quality_decision(
        evidence_bundle_id="bundle:cellQuality",
        available_profiles=["retainWithFlags", "globalMad5"],
    )
    features = build_feature_policy_decision(
        evidence_bundle_id="bundle:features",
        proposed_exclusion_families=[],
        dominant_families=[],
        protected_families=[],
    )

    registry = RnaDecisionRegistry(definitions=[cell_quality, features])
    assert registry.definition("featurePolicy") == features

    with pytest.raises(ValidationError, match="checkpoint order"):
        RnaDecisionRegistry(definitions=[features, cell_quality])


def test_definition_rejects_executor_inventory_drift() -> None:
    definition = build_pca_prefix_decision(
        evidence_bundle_id="bundle:pca", matrix_rank=50
    )
    values = definition.model_dump()
    values["executorOptions"] = values["executorOptions"][:-1]

    with pytest.raises(ValidationError, match="exactly match"):
        type(definition).model_validate(values)
