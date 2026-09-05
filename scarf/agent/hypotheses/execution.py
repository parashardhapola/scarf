"""Execute licensed hypothesis contracts through Scarf statistical tests."""

from typing import Any

from ...metadata.selection import CellField
from ...storage.refs import ArtifactRef
from ..tools import artifact_reference, core_artifact_reference
from .contracts import (
    HypothesisContract,
    HypothesisExecutionStatus,
    HypothesisTestExecution,
)


def _blocked_execution(
    contract: HypothesisContract,
    *,
    reasons: list[str],
    status: HypothesisExecutionStatus,
    effective_selection: ArtifactRef | None = None,
) -> HypothesisTestExecution:
    return HypothesisTestExecution(
        contractId=contract.contractId,
        familyId=contract.familyId,
        status=status,
        contrast=contract.contrast,
        featurePanels=contract.featurePanels,
        inputCellSelection=contract.cellSelection,
        effectiveCellSelection=(
            artifact_reference(effective_selection)
            if effective_selection is not None
            else contract.cellSelection
        ),
        groupingArtifact=contract.groupingArtifact,
        clusterArtifact=(
            contract.clusterSelection.clusterArtifact
            if contract.clusterSelection is not None
            else None
        ),
        blockedReasons=list(dict.fromkeys(reasons)),
        evidenceIds=list(
            dict.fromkeys(
                [
                    *contract.evidenceIds,
                    *contract.contrast.evidenceIds,
                    *(
                        evidence_id
                        for panel in contract.featurePanels
                        for evidence_id in panel.evidenceIds
                    ),
                ]
            )
        ),
    )


def execute_hypothesis_contract(
    store: Any,
    contract: HypothesisContract,
    *,
    invalidate_cache: bool = False,
) -> HypothesisTestExecution:
    """Execute one fully licensed family through ``run_statistical_testing``."""
    if not isinstance(contract, HypothesisContract):
        raise TypeError("contract must be a HypothesisContract")
    contrast = contract.contrast
    if contrast.status != "licensed":
        status: HypothesisExecutionStatus = (
            "needsInput" if contrast.status == "needsInput" else "blocked"
        )
        return _blocked_execution(
            contract,
            reasons=contrast.blockedReasons or ["contrastIsNotLicensed"],
            status=status,
        )
    safety_failures = [
        reason
        for passed, reason in (
            (contrast.betweenUnitDesign, "coefficientIsNotBetweenUnit"),
            (contrast.replicationPassed, "insufficientIndependentReplication"),
            (contrast.estimabilityPassed, "coefficientIsNotEstimable"),
            (
                contrast.pairBy is None or contrast.pairedCoveragePassed is True,
                "pairedCoverageIsIncomplete",
            ),
        )
        if not passed
    ]
    if safety_failures:
        return _blocked_execution(
            contract,
            reasons=safety_failures,
            status="blocked",
        )
    if contract.cellSelection is None:
        return _blocked_execution(
            contract,
            reasons=["cellSelectionIsUnresolved"],
            status="needsInput",
        )
    if not contract.featurePanels:
        return _blocked_execution(
            contract,
            reasons=["featurePanelIsUnresolved"],
            status="needsInput",
        )
    tested_features = list(
        dict.fromkeys(
            feature for panel in contract.featurePanels for feature in panel.features
        )
    )
    if not tested_features:
        return _blocked_execution(
            contract,
            reasons=["featurePanelIsEmpty"],
            status="needsInput",
        )

    input_selection = core_artifact_reference(contract.cellSelection)
    if not isinstance(input_selection, ArtifactRef):
        raise TypeError("cellSelection must resolve to an ArtifactRef")
    effective_selection = input_selection
    try:
        if contract.clusterSelection is not None:
            cluster_artifact = core_artifact_reference(
                contract.clusterSelection.clusterArtifact
            )
            if not isinstance(cluster_artifact, ArtifactRef):
                raise TypeError("clusterArtifact must resolve to an ArtifactRef")
            effective_selection = store.select_cells(
                cluster_artifact,
                include=contract.clusterSelection.include,
                cell_selection=input_selection,
                invalidate_cache=invalidate_cache,
            )

        grouping = (
            core_artifact_reference(contract.groupingArtifact)
            if contract.groupingArtifact is not None
            else CellField(contrast.coefficient, kind="categorical")
        )
        if not isinstance(grouping, ArtifactRef | CellField):
            raise TypeError("grouping source could not be resolved")
        if contrast.test is None or contrast.sampleBy is None:
            return _blocked_execution(
                contract,
                reasons=["contrastExecutionFieldsAreUnresolved"],
                status="needsInput",
                effective_selection=effective_selection,
            )
        result = store.run_statistical_testing(
            tested_features,
            grouping,
            cell_selection=effective_selection,
            groups=contrast.groupOrder,
            test=contrast.test,
            adjustment=contract.adjustment,
            sample_by=contrast.sampleBy,
            pair_by=contrast.pairBy,
            sample_stat=contrast.sampleStatistic,
            expression_cutoff=contrast.expressionCutoff,
            from_assay=contract.fromAssay,
            skip_save=False,
            invalidate_cache=invalidate_cache,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked_execution(
            contract,
            reasons=[f"coreRejected:{type(exc).__name__}:{exc}"],
            status="blocked",
            effective_selection=effective_selection,
        )

    artifact = getattr(result, "artifact", None)
    if not isinstance(artifact, ArtifactRef):
        raise RuntimeError(
            "run_statistical_testing did not persist an exact result artifact"
        )
    if getattr(result, "method", None) != contrast.test:
        raise RuntimeError("Statistical test method differs from its contrast license")
    if list(getattr(result, "group_order", ())) != contrast.groupOrder:
        raise RuntimeError("Statistical group order differs from its contrast license")
    if getattr(result, "sample_by", None) != contrast.sampleBy:
        raise RuntimeError("Statistical sample unit differs from its contrast license")
    if getattr(result, "pair_by", None) != contrast.pairBy:
        raise RuntimeError("Statistical pair unit differs from its contrast license")
    return HypothesisTestExecution(
        contractId=contract.contractId,
        familyId=contract.familyId,
        status="executed",
        contrast=contrast,
        featurePanels=contract.featurePanels,
        testedFeatures=tested_features,
        inputCellSelection=contract.cellSelection,
        effectiveCellSelection=artifact_reference(effective_selection),
        groupingArtifact=contract.groupingArtifact,
        clusterArtifact=(
            contract.clusterSelection.clusterArtifact
            if contract.clusterSelection is not None
            else None
        ),
        statisticalTestArtifact=artifact_reference(artifact),
        adjustment=contract.adjustment,
        evidenceIds=list(
            dict.fromkeys(
                [
                    *contract.evidenceIds,
                    contrast.evidenceId,
                    *contrast.evidenceIds,
                    *(
                        evidence_id
                        for panel in contract.featurePanels
                        for evidence_id in panel.evidenceIds
                    ),
                ]
            )
        ),
    )


__all__ = ["execute_hypothesis_contract"]
