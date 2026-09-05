"""Evidence-gated execution of existing Scarf statistical tests."""

from typing import Any, Literal

from ..metadata.selection import CellField
from ..storage.refs import ArtifactRef
from .config._deps import AGENT_INSTALL_HINT
from .experimental_context import ContrastPlan
from .tools import artifact_reference, core_artifact_reference
from .types import AgentDataModel, ArtifactReferenceModel

try:
    from pydantic import Field, model_validator
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

type FeaturePanelPurpose = Literal["explicit", "exploratoryMarkers"]
type HypothesisExecutionStatus = Literal["executed", "blocked", "needsInput"]

_NORMALIZED_EXPRESSION_SCOPE = (
    "Sample-level normalized-expression distribution testing. This is not a "
    "raw-count pseudobulk differential-expression model."
)


class HypothesisFeaturePanel(AgentDataModel):
    """Features kept under one explicit or exploratory provenance label."""

    panelId: str = ""
    purpose: FeaturePanelPurpose = "explicit"
    features: list[str] = Field(default_factory=list)
    sourceArtifact: ArtifactReferenceModel | None = None
    evidenceIds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_panel(self) -> "HypothesisFeaturePanel":
        if self.panelId != self.panelId.strip():
            raise ValueError("Feature panel ids cannot contain surrounding whitespace")
        if any(
            not feature.strip() or feature != feature.strip()
            for feature in self.features
        ):
            raise ValueError("Feature names must be non-empty trimmed strings")
        if len(self.features) != len(set(self.features)):
            raise ValueError("Feature names must be unique within a panel")
        if self.sourceArtifact is not None and not self.sourceArtifact.artifactId:
            raise ValueError("Feature panel source artifacts must be exact")
        if self.purpose == "exploratoryMarkers" and self.sourceArtifact is None:
            raise ValueError(
                "Exploratory marker panels require their exact source artifact"
            )
        return self


class ClusterSelectionContract(AgentDataModel):
    """An exact cluster artifact and labels used for a within-cluster test."""

    clusterArtifact: ArtifactReferenceModel
    include: list[str | int | float | bool] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection(self) -> "ClusterSelectionContract":
        if not self.clusterArtifact.artifactId:
            raise ValueError("Cluster selection requires an exact artifact")
        if not self.include:
            raise ValueError("Cluster selection requires at least one label")
        keys = [(type(value).__name__, repr(value)) for value in self.include]
        if len(keys) != len(set(keys)):
            raise ValueError("Cluster labels must be unique")
        return self


class HypothesisContract(AgentDataModel):
    """One immutable-input hypothesis family licensed by a contrast plan."""

    contractId: str = ""
    familyId: str = ""
    contrast: ContrastPlan = Field(default_factory=ContrastPlan.get_blank)
    cellSelection: ArtifactReferenceModel | None = None
    groupingArtifact: ArtifactReferenceModel | None = None
    clusterSelection: ClusterSelectionContract | None = None
    featurePanels: list[HypothesisFeaturePanel] = Field(default_factory=list)
    fromAssay: str | None = None
    adjustment: Literal["fdr_bh"] = "fdr_bh"
    evidenceIds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> "HypothesisContract":
        for name, value in (
            ("contractId", self.contractId),
            ("familyId", self.familyId),
        ):
            if value != value.strip():
                raise ValueError(f"{name} cannot contain surrounding whitespace")
        panel_ids = [panel.panelId for panel in self.featurePanels]
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError("Hypothesis feature panel ids must be unique")
        if self.cellSelection is not None and (
            self.cellSelection.scope != "datastore"
            or self.cellSelection.kind != "cell_selection"
            or not self.cellSelection.artifactId
        ):
            raise ValueError(
                "Hypothesis cellSelection must be an exact datastore selection"
            )
        if self.groupingArtifact is not None and not self.groupingArtifact.artifactId:
            raise ValueError("Hypothesis groupingArtifact must be exact")
        return self


class HypothesisTestExecution(AgentDataModel):
    """Executed artifact references or explicit reasons no test was run."""

    contractId: str = ""
    familyId: str = ""
    status: HypothesisExecutionStatus = "blocked"
    contrast: ContrastPlan = Field(default_factory=ContrastPlan.get_blank)
    featurePanels: list[HypothesisFeaturePanel] = Field(default_factory=list)
    testedFeatures: list[str] = Field(default_factory=list)
    inputCellSelection: ArtifactReferenceModel | None = None
    effectiveCellSelection: ArtifactReferenceModel | None = None
    groupingArtifact: ArtifactReferenceModel | None = None
    clusterArtifact: ArtifactReferenceModel | None = None
    statisticalTestArtifact: ArtifactReferenceModel | None = None
    adjustment: Literal["fdr_bh"] = "fdr_bh"
    blockedReasons: list[str] = Field(default_factory=list)
    claimScope: str = _NORMALIZED_EXPRESSION_SCOPE
    evidenceIds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_execution(self) -> "HypothesisTestExecution":
        if self.status == "executed":
            if self.statisticalTestArtifact is None or self.blockedReasons:
                raise ValueError(
                    "Executed hypothesis tests require an artifact and no block"
                )
        elif not self.blockedReasons:
            raise ValueError("Blocked hypothesis tests require explicit reasons")
        return self

    @classmethod
    def get_blank(cls) -> "HypothesisTestExecution":
        return cls(blockedReasons=["hypothesisContractIsUnresolved"])


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


__all__ = [
    "ClusterSelectionContract",
    "FeaturePanelPurpose",
    "HypothesisContract",
    "HypothesisExecutionStatus",
    "HypothesisFeaturePanel",
    "HypothesisTestExecution",
    "execute_hypothesis_contract",
]
