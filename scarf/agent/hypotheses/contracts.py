"""Serializable contracts for evidence-gated hypothesis tests."""

from typing import Literal

from .._deps import AGENT_INSTALL_HINT
from ..experimental_context.contracts import ContrastPlan
from ..types import AgentDataModel, ArtifactReferenceModel

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
