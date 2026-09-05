"""Serializable contracts for biological interpretation."""

from typing import Any, Literal

from pydantic import Field

from ..types import (
    AgentDataModel,
    AgentRunInfo,
    ArtifactReferenceModel,
    ExperimentalBiologyHandoff,
    StageStatus,
)

type InterpretationConfidence = Literal["low", "medium", "high"]
type TreatmentDirection = Literal["higher", "lower", "equal"]

_MAX_CLUSTERS = 20
_MAX_CONDITIONS = 30
_MAX_MARKERS = 25


class BiologicalContext(AgentDataModel):
    """Caller-supplied facts that constrain biological interpretation."""

    organism: str = ""
    studyContext: str = ""
    tissue: str = ""
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalDetails: list[str] = Field(default_factory=list)
    treatmentQuestion: str = ""

    @classmethod
    def get_blank(cls) -> "BiologicalContext":
        return cls()

    @classmethod
    def get_example(cls) -> "BiologicalContext":
        return cls(
            organism="Homo sapiens",
            studyContext=(
                "Human lung samples were profiled after drug or vehicle treatment."
            ),
            tissue="lung",
            cellTypeReferences=["alveolar macrophage", "T cell"],
            experimentalDetails=["drug and vehicle groups"],
            treatmentQuestion="Which populations respond selectively to treatment?",
        )


class ConditionClusterSummary(AgentDataModel):
    """Aggregate cluster abundance for one condition without sample identifiers."""

    condition: str = ""
    clusterId: str = ""
    nSamples: int = 0
    meanFraction: float = 0.0
    minFraction: float = 0.0
    maxFraction: float = 0.0
    cellCount: int = 0
    evidenceId: str = ""

    @classmethod
    def get_example(cls) -> "ConditionClusterSummary":
        return cls(
            condition="treated",
            clusterId="3",
            nSamples=4,
            meanFraction=0.18,
            minFraction=0.12,
            maxFraction=0.25,
            cellCount=180,
            evidenceId="composition:RNA_cluster:condition:treated:cluster:3",
        )


class ClusterCompositionEvidence(AgentDataModel):
    """Bounded deterministic evidence about cluster sizes and conditions."""

    clusterArtifact: ArtifactReferenceModel | None = None
    cellSelection: ArtifactReferenceModel | None = None
    totalCells: int = 0
    clusterCounts: dict[str, int] = Field(default_factory=dict)
    sampleColumn: str | None = None
    conditionColumn: str | None = None
    conditionSummaries: list[ConditionClusterSummary] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "ClusterCompositionEvidence":
        summary = ConditionClusterSummary.get_example()
        reference_summary = ConditionClusterSummary(
            condition="control",
            clusterId=summary.clusterId,
            nSamples=4,
            meanFraction=0.11,
            minFraction=0.08,
            maxFraction=0.15,
            cellCount=110,
            evidenceId="composition:RNA_cluster:condition:control:cluster:3",
        )
        return cls(
            clusterArtifact=ArtifactReferenceModel(
                assay="RNA",
                kind="cluster_labels",
                artifactId="b" * 64,
            ),
            cellSelection=ArtifactReferenceModel(
                scope="datastore",
                assay=None,
                kind="cell_selection",
                artifactId="c" * 64,
            ),
            totalCells=1000,
            clusterCounts={"0": 520, "1": 300, "3": 180},
            sampleColumn="sample",
            conditionColumn="treatment",
            conditionSummaries=[reference_summary, summary],
            evidenceIds=[
                "composition:RNA_cluster:counts",
                reference_summary.evidenceId,
                summary.evidenceId,
            ],
        )


class MarkerFeature(AgentDataModel):
    """One observed marker feature and its available Scarf statistics."""

    featureId: str = ""
    featureName: str = ""
    featureIndex: int | None = None
    score: float | None = None
    foldChange: float | None = None
    fractionExpressed: float | None = None
    fractionExpressedRest: float | None = None
    mean: float | None = None
    meanRest: float | None = None
    auc: float | None = None
    adjustedPvalue: float | None = None

    @classmethod
    def get_example(cls) -> "MarkerFeature":
        return cls(
            featureId="ENSG00000173372",
            featureName="C1QA",
            featureIndex=123,
            score=0.83,
            foldChange=3.4,
            fractionExpressed=0.76,
            fractionExpressedRest=0.18,
            auc=0.91,
            adjustedPvalue=0.001,
        )


class ClusterMarkerEvidence(AgentDataModel):
    """Bounded markers for one exact cluster label."""

    clusterId: str = ""
    markers: list[MarkerFeature] = Field(default_factory=list)
    markerArtifact: ArtifactReferenceModel | None = None
    evidenceId: str = ""
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "ClusterMarkerEvidence":
        return cls(
            clusterId="3",
            markers=[MarkerFeature.get_example()],
            markerArtifact=ArtifactReferenceModel(
                assay="RNA",
                kind="marker_table",
                artifactId="a" * 64,
            ),
            evidenceId="markers:RNA_cluster:cluster:3",
        )


class ClusterMarkerBatchEvidence(AgentDataModel):
    """Markers for all model-selected clusters returned by one tool call."""

    clusters: list[ClusterMarkerEvidence] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ClusterMarkerBatchEvidence":
        return cls()

    @classmethod
    def get_example(cls) -> "ClusterMarkerBatchEvidence":
        cluster = ClusterMarkerEvidence.get_example()
        return cls(clusters=[cluster], evidenceIds=[cluster.evidenceId])


class ClusterInterpretation(AgentDataModel):
    """One evidence-linked cluster interpretation or hypothesis."""

    clusterId: str = ""
    proposedIdentity: str = "unresolved"
    identityIsHypothesis: bool = True
    confidence: InterpretationConfidence = "low"
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "ClusterInterpretation":
        return cls(
            clusterId="3",
            proposedIdentity="alveolar macrophage-like",
            identityIsHypothesis=True,
            confidence="medium",
            rationale="Observed marker pattern is consistent with the proposed identity.",
            evidenceIds=["markers:RNA_cluster:cluster:3"],
        )


class TreatmentObservation(AgentDataModel):
    """Descriptive treatment observation with no unsupported causal claim."""

    clusterId: str = ""
    referenceCondition: str = ""
    comparisonCondition: str = ""
    direction: TreatmentDirection = "equal"
    observation: str = ""
    isDescriptiveOnly: Literal[True] = True
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "TreatmentObservation":
        return cls(
            clusterId="3",
            referenceCondition="control",
            comparisonCondition="treated",
            direction="higher",
            observation="Cluster 3 has a higher mean fraction in treated samples.",
            evidenceIds=[
                "composition:RNA_cluster:condition:control:cluster:3",
                "composition:RNA_cluster:condition:treated:cluster:3",
            ],
        )


class FollowUpRecommendation(AgentDataModel):
    """A bounded next analysis tied to an observed uncertainty."""

    question: str = ""
    operation: str = ""
    rationale: str = ""
    requiredInputs: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "FollowUpRecommendation":
        return cls(
            question="Is the abundance difference reproducible across donors?",
            operation="sample-level differential abundance",
            rationale="Current evidence is descriptive and requires independent replicates.",
            requiredInputs=["sample", "condition", "donor"],
            evidenceIds=[
                "composition:RNA_cluster:condition:control:cluster:3",
                "composition:RNA_cluster:condition:treated:cluster:3",
            ],
        )


class BiologicalInterpretationNeedsInput(AgentDataModel):
    question: str = ""
    requiredInputs: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "BiologicalInterpretationNeedsInput":
        return cls(
            question="Provide an exact marker artifact or authorize marker search.",
            requiredInputs=["markerArtifact"],
        )


class BiologicalInterpretationReport(AgentDataModel):
    """Structured, evidence-grounded biological review."""

    status: StageStatus = "needsInput"
    clusterInterpretations: list[ClusterInterpretation] = Field(default_factory=list)
    treatmentObservations: list[TreatmentObservation] = Field(default_factory=list)
    followUps: list[FollowUpRecommendation] = Field(default_factory=list)
    clusterArtifact: ArtifactReferenceModel | None = None
    markerArtifact: ArtifactReferenceModel | None = None
    graphAssay: str | None = None
    markerAssay: str | None = None
    evidenceIds: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    stopReason: str = ""
    needsInput: BiologicalInterpretationNeedsInput | None = None
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_example(cls) -> "BiologicalInterpretationReport":
        interpretation = ClusterInterpretation.get_example()
        observation = TreatmentObservation.get_example()
        follow_up = FollowUpRecommendation.get_example()
        return cls(
            status="done",
            clusterInterpretations=[interpretation],
            treatmentObservations=[observation],
            followUps=[follow_up],
            clusterArtifact=ClusterCompositionEvidence.get_example().clusterArtifact,
            markerArtifact=ClusterMarkerEvidence.get_example().markerArtifact,
            graphAssay="RNA",
            markerAssay="RNA",
            evidenceIds=sorted(
                {
                    *interpretation.evidenceIds,
                    *observation.evidenceIds,
                    *follow_up.evidenceIds,
                }
            ),
            limitations=[
                "Cell identities remain hypotheses until independently validated."
            ],
            stopReason="The requested clusters were reviewed.",
        )


class BiologicalInterpretationDependencies(AgentDataModel):
    """Runtime state available only to biological interpretation tools."""

    store: Any = Field(default=None, exclude=True)
    cluster: Any = Field(default=None, exclude=True)
    cellSelection: Any = Field(default=None, exclude=True)
    cellIndices: Any = Field(default=None, exclude=True)
    fromAssay: str | None = None
    graphAssay: str | None = None
    markerAssay: str | None = None
    markerAssayType: str | None = None
    sampleColumn: str | None = None
    conditionColumn: str | None = None
    marker: Any = Field(default=None, exclude=True)
    markerFeatures: Any = Field(default=None, exclude=True)
    allowMarkerSearch: bool = False
    maxClusters: int = 12
    maxMarkers: int = 10
    markerMinScore: float = 0.25
    markerMinFraction: float = 0.2
    evidenceIds: set[str] = Field(default_factory=set, exclude=True)
    clusterValues: dict[str, Any] = Field(default_factory=dict, exclude=True)
    markerEvidenceIds: dict[str, str] = Field(default_factory=dict, exclude=True)
    markerEvidence: dict[str, ClusterMarkerEvidence] = Field(
        default_factory=dict,
        exclude=True,
    )
    compositionEvidence: ClusterCompositionEvidence | None = Field(
        default=None,
        exclude=True,
    )
    markerBatch: ClusterMarkerBatchEvidence | None = Field(
        default=None,
        exclude=True,
    )
    markerBatchClusterIds: list[str] = Field(default_factory=list, exclude=True)
    toolCalls: list[str] = Field(default_factory=list, exclude=True)
    conditionEvidence: dict[str, ConditionClusterSummary] = Field(
        default_factory=dict,
        exclude=True,
    )
    designHandoff: ExperimentalBiologyHandoff | None = Field(
        default=None,
        exclude=True,
    )

    @classmethod
    def get_example(cls) -> "BiologicalInterpretationDependencies":
        return cls(
            cluster=object(),
            fromAssay="RNA",
            graphAssay="RNA",
            markerAssay="RNA",
            markerAssayType="RNA",
            sampleColumn="sample",
            conditionColumn="treatment",
        )
