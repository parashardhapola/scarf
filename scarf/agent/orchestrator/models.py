"""Public data models for resumable automated agent workflows."""

import hashlib
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ...storage.refs import ArtifactRef
from .. import record_io
from ..config import AgentRunConfig
from ..decisions.rna import CellQualityExecutorPayload
from ..experimental_context.contracts import CellQcPlan
from ..experimental_context.study import AuthorLabelPolicy, StudyContract
from ..ingest.manifest import DatasetManifest
from ..persistence.contracts import AgentReportReference, AgentWorkflowRun
from ..types import AgentDataModel, ArtifactReferenceModel

type AutomatedWorkflowStatus = Literal[
    "completed",
    "needsInput",
    "abstained",
    "failed",
    "abandoned",
]
type WorkflowInputPolicy = Literal["pause", "unattended"]
type WorkflowStageStatus = Literal[
    "started", "done", "needsInput", "abstained", "failed"
]
type WorkflowStageName = Literal[
    "ingest",
    "data_enrichment",
    "hto_demultiplexing",
    "experimental_context",
    "preprocessing_plan",
    "preprocessing",
    "parameter_tuning",
    "feature_policy_review",
    "feature_policy_preprocessing",
    "feature_policy_tuning",
    "analysis_review",
    "analysis_finalization",
    "biological_interpretation",
]
type AssayRole = Literal["graph", "hto", "unsupported"]
type ReductionMethod = Literal["pca", "lsi", "identity", "none"]

_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ORCHESTRATION_FORMAT = "scarf_agent_orchestrations"
_ORCHESTRATION_VERSION = 2
_STAGE_ORDER: tuple[WorkflowStageName, ...] = (
    "ingest",
    "data_enrichment",
    "hto_demultiplexing",
    "experimental_context",
    "preprocessing_plan",
    "preprocessing",
    "parameter_tuning",
    "feature_policy_review",
    "feature_policy_preprocessing",
    "feature_policy_tuning",
    "analysis_review",
    "analysis_finalization",
)


class WorkflowQuestion(AgentDataModel):
    """One stable question that can be answered by a resume request."""

    questionId: str = ""
    decisionId: str | None = None
    question: str = ""
    options: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    planChecksum: str | None = None

    @classmethod
    def get_blank(cls) -> "WorkflowQuestion":
        return cls()

    @classmethod
    def get_example(cls) -> "WorkflowQuestion":
        return cls(
            questionId="approvePlanChecksum",
            question="Approve this preprocessing plan?",
            planChecksum="0" * 64,
        )


class WorkflowNeedsInput(AgentDataModel):
    """All questions blocking the next workflow stage."""

    questions: list[WorkflowQuestion] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "WorkflowNeedsInput":
        return cls()

    @classmethod
    def get_example(cls) -> "WorkflowNeedsInput":
        return cls(questions=[WorkflowQuestion.get_example()])


class WorkflowStageLink(AgentDataModel):
    """Immutable identity of one completed parent stage attempt."""

    stage: WorkflowStageName = "ingest"
    attemptId: str = ""
    contentSha256: str = ""

    @field_validator("attemptId")
    @classmethod
    def validate_attempt_id(cls, value: str) -> str:
        if value and _RUN_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("attemptId must be a lowercase run identifier")
        return value

    @field_validator("contentSha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if value and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("contentSha256 must be a lowercase SHA-256 digest")
        return value

    @classmethod
    def get_blank(cls) -> "WorkflowStageLink":
        return cls()

    @classmethod
    def get_example(cls) -> "WorkflowStageLink":
        return cls(stage="ingest", attemptId="attempt-1", contentSha256="0" * 64)


class WorkflowStageAttempt(AgentDataModel):
    """Append-only record for one orchestration stage attempt."""

    workflowRunId: str = ""
    stage: WorkflowStageName = "ingest"
    attemptId: str = ""
    status: WorkflowStageStatus = "started"
    startedAtNs: int = Field(default=0, ge=0)
    completedAtNs: int = Field(default=0, ge=0)
    requestSha256: str = ""
    configSha256: str = ""
    parentAttempts: list[WorkflowStageLink] = Field(default_factory=list)
    reportReferences: list[AgentReportReference] = Field(default_factory=list)
    artifacts: dict[str, ArtifactReferenceModel] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    needsInput: WorkflowNeedsInput | None = None
    error: str | None = None
    contentSha256: str = ""

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "WorkflowStageAttempt":
        if self.status == "started" and self.completedAtNs:
            raise ValueError("A started stage cannot have completedAtNs")
        if self.status != "started" and self.completedAtNs < self.startedAtNs:
            raise ValueError("A completed stage must not precede its start")
        if self.status == "needsInput" and self.needsInput is None:
            raise ValueError("needsInput stage records require questions")
        if self.status == "failed" and not self.error:
            raise ValueError("failed stage records require an error")
        return self

    @classmethod
    def get_blank(cls) -> "WorkflowStageAttempt":
        return cls()

    @classmethod
    def get_example(cls) -> "WorkflowStageAttempt":
        return cls(
            workflowRunId="workflow-1",
            stage="ingest",
            attemptId="attempt-1",
            status="done",
            startedAtNs=1,
            completedAtNs=2,
            requestSha256="0" * 64,
            configSha256="1" * 64,
            contentSha256="2" * 64,
        )


class AssayPreprocessingPlan(AgentDataModel):
    """Exact allowlisted preprocessing route for one assay."""

    assay: str = ""
    assayType: str = "Assay"
    role: AssayRole = "unsupported"
    graphEligible: bool = False
    markerEligible: bool = False
    featureMethod: Literal["hvg", "prevalentPeaks", "panel", "none"] = "none"
    reductionMethod: ReductionMethod = "none"
    featureParameters: dict[str, Any] = Field(default_factory=dict)
    normalizationParameters: dict[str, Any] = Field(default_factory=dict)
    reductionParameters: dict[str, Any] = Field(default_factory=dict)
    exactExcludedFeatures: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayPreprocessingPlan":
        return cls()

    @classmethod
    def get_example(cls) -> "AssayPreprocessingPlan":
        return cls(
            assay="RNA",
            assayType="RNA",
            role="graph",
            graphEligible=True,
            markerEligible=True,
            featureMethod="hvg",
            reductionMethod="pca",
            featureParameters={"topN": 1000, "minCells": 20},
        )


class AutomatedPreprocessingPlan(AgentDataModel):
    """Dataset-wide preprocessing plan produced before selection changes."""

    primaryAssay: str = ""
    markerAssay: str = ""
    cellSelection: ArtifactReferenceModel | None = None
    cellQc: CellQcPlan = Field(default_factory=CellQcPlan.get_blank)
    cellQualityPayload: CellQualityExecutorPayload | None = None
    assays: list[AssayPreprocessingPlan] = Field(default_factory=list)
    pairedAssays: list[str] = Field(default_factory=list)
    planChecksum: str = ""
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cell_quality_payload(self) -> "AutomatedPreprocessingPlan":
        if (
            self.cellQualityPayload is not None
            and self.cellQc.registeredProfile != self.cellQualityPayload.profile
        ):
            raise ValueError(
                "cellQualityPayload must match the selected registered QC profile"
            )
        return self

    @classmethod
    def get_blank(cls) -> "AutomatedPreprocessingPlan":
        return cls()

    @classmethod
    def get_example(cls) -> "AutomatedPreprocessingPlan":
        return cls(
            primaryAssay="RNA",
            markerAssay="RNA",
            cellSelection=ArtifactReferenceModel(
                scope="datastore",
                kind="cell_selection",
                artifactId="c" * 64,
            ),
            assays=[AssayPreprocessingPlan.get_example()],
            planChecksum="0" * 64,
        )


class PreprocessedAssayHandoff(AgentDataModel):
    """Exact normalized input and selections handed to Parameter Tuning."""

    assay: str = ""
    assayType: str = "Assay"
    cellSelection: ArtifactReferenceModel | None = None
    reductionMethod: ReductionMethod = "none"
    graphFeatures: ArtifactReferenceModel | None = None
    markerFeatures: ArtifactReferenceModel | None = None
    normalized: ArtifactReferenceModel | None = None
    graphFeatureCandidates: dict[str, ArtifactReferenceModel] = Field(
        default_factory=dict
    )
    normalizedCandidates: dict[str, ArtifactReferenceModel] = Field(
        default_factory=dict
    )
    featureCandidateEvaluations: list[dict[str, Any]] = Field(default_factory=list)
    nCells: int = 0
    nFeatures: int = 0

    @classmethod
    def get_blank(cls) -> "PreprocessedAssayHandoff":
        return cls()

    @classmethod
    def get_example(cls) -> "PreprocessedAssayHandoff":
        return cls(
            assay="RNA",
            assayType="RNA",
            cellSelection=ArtifactReferenceModel(
                scope="datastore",
                kind="cell_selection",
                artifactId="c" * 64,
            ),
            reductionMethod="pca",
            graphFeatures=ArtifactReferenceModel.get_example(),
            markerFeatures=ArtifactReferenceModel.get_example(),
            normalized=ArtifactReferenceModel(
                assay="RNA", kind="normalized", artifactId="1" * 64
            ),
            nCells=100,
            nFeatures=1000,
        )


class NativeAnalysisHandoff(AgentDataModel):
    """Selected immutable native analysis chain for one assay."""

    assay: str = ""
    reductionMethod: ReductionMethod = "none"
    featureSelection: ArtifactReferenceModel | None = None
    markerFeatures: ArtifactReferenceModel | None = None
    normalized: ArtifactReferenceModel | None = None
    reduction: ArtifactReferenceModel | None = None
    batchCorrection: ArtifactReferenceModel | None = None
    annIndex: ArtifactReferenceModel | None = None
    embeddingInitialization: ArtifactReferenceModel | None = None
    neighbors: ArtifactReferenceModel | None = None
    graph: ArtifactReferenceModel | None = None
    clusters: ArtifactReferenceModel | None = None
    umap: ArtifactReferenceModel | None = None

    @classmethod
    def get_blank(cls) -> "NativeAnalysisHandoff":
        return cls()

    @classmethod
    def get_example(cls) -> "NativeAnalysisHandoff":
        return cls(assay="RNA", reductionMethod="pca")


class FinalAnalysisHandoff(AgentDataModel):
    """Replayable final analysis used by Biological Interpretation."""

    handoffId: str = ""
    workflowRunId: str = ""
    primaryAssay: str = ""
    markerAssay: str = ""
    cellSelection: ArtifactReferenceModel | None = None
    nativeAnalyses: list[NativeAnalysisHandoff] = Field(default_factory=list)
    graph: ArtifactReferenceModel | None = None
    graphMethod: Literal["native", "snn", "wnn"] = "native"
    clusters: ArtifactReferenceModel | None = None
    embeddingInitialization: ArtifactReferenceModel | None = None
    umap: ArtifactReferenceModel | None = None
    markerFeatures: ArtifactReferenceModel | None = None
    markers: ArtifactReferenceModel | None = None
    doubletScores: list[ArtifactReferenceModel] = Field(default_factory=list)
    doubletScoreSelections: list[ArtifactReferenceModel] = Field(default_factory=list)
    doubletEvidence: dict[str, Any] = Field(default_factory=dict)
    markerEvidence: dict[str, Any] = Field(default_factory=dict)
    statisticalTests: list[ArtifactReferenceModel] = Field(default_factory=list)
    analysisEvidence: dict[str, Any] = Field(default_factory=dict)
    parameterReport: AgentReportReference | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_handoff_id(self) -> "FinalAnalysisHandoff":
        if not self.handoffId:
            return self
        expected = self._content_handoff_id()
        if self.handoffId != expected:
            raise ValueError("handoffId does not match the final artifact handoff")
        return self

    def _content_handoff_id(self) -> str:
        digest = hashlib.sha256(
            record_io.canonical_json_bytes(
                self.model_dump(mode="json", exclude={"handoffId"})
            )
        ).hexdigest()
        return f"handoff:{digest}"

    def with_handoff_id(self) -> "FinalAnalysisHandoff":
        values = self.model_dump(mode="json")
        values["handoffId"] = self._content_handoff_id()
        return FinalAnalysisHandoff.model_validate(values)

    @classmethod
    def get_blank(cls) -> "FinalAnalysisHandoff":
        return cls()

    @classmethod
    def get_example(cls) -> "FinalAnalysisHandoff":
        return cls(
            workflowRunId="workflow-1",
            primaryAssay="RNA",
            markerAssay="RNA",
            cellSelection=ArtifactReferenceModel(
                scope="datastore",
                kind="cell_selection",
                artifactId="c" * 64,
            ),
            nativeAnalyses=[NativeAnalysisHandoff.get_example()],
            graph=ArtifactReferenceModel(
                assay="RNA", kind="connectivity_map", artifactId="2" * 64
            ),
            embeddingInitialization=ArtifactReferenceModel(
                assay="RNA",
                kind="embedding_initialization",
                artifactId="5" * 64,
            ),
            clusters=ArtifactReferenceModel(
                assay="RNA", kind="cluster_labels", artifactId="3" * 64
            ),
        ).with_handoff_id()


class AutomatedWorkflowConfig(AgentDataModel):
    """Bounded execution policy for automated workflows."""

    inputPolicy: WorkflowInputPolicy = Field(
        default="pause",
        exclude_if=lambda value: value == "pause",
    )
    primaryInitialCandidates: int = Field(default=11, ge=1)
    secondaryInitialCandidates: int = Field(default=3, ge=1)
    maxRefinedCandidatesPerAssay: int = Field(default=1, ge=0, le=1)
    maxHarmonyCandidatesPerAssay: int = Field(default=1, ge=0, le=1)
    runConfoundedHarmonyDiagnostic: bool = False
    integrationResolutionCandidates: int = Field(default=3, ge=1)
    maxCandidateBranches: int = Field(default=24, ge=1)
    minClusterCells: int = Field(default=20, ge=1)
    maxIdentityFeatures: int = Field(default=64, ge=2)
    maxGraphAssays: int = Field(default=3, ge=1)
    hvgCandidateCounts: tuple[int, ...] = (1000, 2000, 4000)
    pcaCandidateDimensions: tuple[int, ...] = (10, 20, 30, 50)
    graphNeighborCandidates: tuple[int, ...] = (11, 21, 41)
    leidenResolutionCandidates: tuple[float, ...] = (
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
    )
    leidenSeeds: tuple[int, ...] = (0, 1, 2)
    clusterSubsamples: int = Field(default=2, ge=0, le=5)
    clusterSubsampleFraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    maxRevisions: int = Field(default=2, ge=0, le=2)
    allowDownloads: bool = False
    cacheDir: str | None = None
    agentRunConfig: AgentRunConfig = Field(default_factory=AgentRunConfig)

    @model_validator(mode="after")
    def validate_candidate_registry(self) -> "AutomatedWorkflowConfig":
        integer_fields = (
            "hvgCandidateCounts",
            "pcaCandidateDimensions",
            "graphNeighborCandidates",
        )
        for field_name in integer_fields:
            values = getattr(self, field_name)
            if not values or any(value < 1 for value in values):
                raise ValueError(f"{field_name} must contain positive integers")
            if len(values) != len(set(values)) or tuple(sorted(values)) != values:
                raise ValueError(f"{field_name} must be sorted and unique")
        resolutions = self.leidenResolutionCandidates
        if (
            not resolutions
            or any(value <= 0 for value in resolutions)
            or len(resolutions) != len(set(resolutions))
            or tuple(sorted(resolutions)) != resolutions
        ):
            raise ValueError(
                "leidenResolutionCandidates must be positive, sorted, and unique"
            )
        if not self.leidenSeeds or len(self.leidenSeeds) != len(set(self.leidenSeeds)):
            raise ValueError("leidenSeeds must be non-empty and unique")
        return self

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowConfig":
        return cls()

    @classmethod
    def get_example(cls) -> "AutomatedWorkflowConfig":
        return cls()


class AutomatedWorkflowRequest(AgentDataModel):
    """Immutable request for one automated analysis."""

    sourcePath: str = ""
    zarrPath: str | None = None
    studyContext: str = ""
    studyObjective: str = ""
    authorLabelPolicy: AuthorLabelPolicy = "holdout"
    workspace: str | None = None
    primaryAssay: str | None = None
    markerAssay: str | None = None
    analysisAssays: list[str] = Field(default_factory=list)
    pairedAssays: list[str] = Field(default_factory=list)
    ingestDirections: dict[str, Any] = Field(default_factory=dict)
    experimentalDirections: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> "AutomatedWorkflowRequest":
        if not self.sourcePath.strip():
            raise ValueError("sourcePath must be non-empty")
        if not self.studyContext.strip():
            raise ValueError("studyContext must be non-empty")
        if not self.studyObjective.strip():
            raise ValueError("studyObjective must be non-empty")
        if len(set(self.analysisAssays)) != len(self.analysisAssays):
            raise ValueError("analysisAssays must be unique")
        if len(set(self.pairedAssays)) != len(self.pairedAssays):
            raise ValueError("pairedAssays must be unique")
        if self.pairedAssays and len(self.pairedAssays) < 2:
            raise ValueError("pairedAssays must contain at least two assays")
        return self

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowRequest":
        return cls(
            sourcePath="dataset.h5",
            studyContext="Study context",
            studyObjective="Discover stable population structure.",
        )

    @classmethod
    def get_example(cls) -> "AutomatedWorkflowRequest":
        return cls(
            sourcePath="dataset.h5ad",
            zarrPath="dataset.zarr",
            studyContext="Single-cell profiling of treated human blood.",
            studyObjective=(
                "Discover stable populations while preserving treatment structure."
            ),
        )


class AutomatedWorkflowResumeRequest(AgentDataModel):
    """Answers used to resume one running workflow."""

    zarrPath: str = ""
    workflowRunId: str = ""
    workspace: str | None = None
    answers: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> "AutomatedWorkflowResumeRequest":
        if not self.zarrPath.strip():
            raise ValueError("zarrPath must be non-empty")
        if _RUN_ID_PATTERN.fullmatch(self.workflowRunId) is None:
            raise ValueError("workflowRunId must be a lowercase run identifier")
        return self

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowResumeRequest":
        return cls(zarrPath="dataset.zarr", workflowRunId="workflow-1")

    @classmethod
    def get_example(cls) -> "AutomatedWorkflowResumeRequest":
        return cls(
            zarrPath="dataset.zarr",
            workflowRunId="workflow-1",
            answers={"approvePlanChecksum": "0" * 64},
        )


class AutomatedWorkflowResult(AgentDataModel):
    """Bounded result of running or resuming an automated workflow."""

    status: AutomatedWorkflowStatus = "failed"
    currentStage: WorkflowStageName = "ingest"
    zarrPath: str | None = None
    workflowRun: AgentWorkflowRun | None = None
    reportReferences: list[AgentReportReference] = Field(default_factory=list)
    datasetManifest: DatasetManifest | None = None
    preprocessingPlan: AutomatedPreprocessingPlan | None = None
    studyContract: StudyContract | None = None
    finalAnalysis: FinalAnalysisHandoff | None = None
    finalHandoffId: str | None = None
    decisionRunId: str | None = None
    verificationSummary: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    unresolvedClaims: list[str] = Field(default_factory=list)
    needsInput: WorkflowNeedsInput | None = None
    notes: list[str] = Field(default_factory=list)
    contentSha256: str = ""

    @model_validator(mode="after")
    def validate_terminal_handoff(self) -> "AutomatedWorkflowResult":
        if self.status != "completed":
            return self
        if self.finalAnalysis is None or not self.finalAnalysis.handoffId:
            raise ValueError("Completed workflow results require a final handoff")
        if self.finalHandoffId != self.finalAnalysis.handoffId:
            raise ValueError("Result and final analysis handoff IDs must agree")
        if (
            self.workflowRun is None
            or self.decisionRunId != self.workflowRun.workflowRunId
        ):
            raise ValueError(
                "Completed results require the matching decision workflow ID"
            )
        return self

    @classmethod
    def get_blank(cls) -> "AutomatedWorkflowResult":
        return cls()

    @classmethod
    def get_example(cls) -> "AutomatedWorkflowResult":
        workflow = AgentWorkflowRun.get_example()
        final_analysis = FinalAnalysisHandoff.get_example()
        return cls(
            status="completed",
            currentStage="analysis_finalization",
            zarrPath="dataset.zarr",
            workflowRun=workflow,
            studyContract=StudyContract.get_example(),
            finalAnalysis=final_analysis,
            finalHandoffId=final_analysis.handoffId,
            decisionRunId=workflow.workflowRunId,
        )


class OrchestrationRequestRecord(AgentDataModel):
    """Stored immutable request and effective configuration."""

    recordType: Literal["automatedWorkflowRequest"] = "automatedWorkflowRequest"
    formatVersion: Literal[2] = 2
    workflowRunId: str = ""
    createdAtNs: int = Field(default=0, ge=0)
    request: AutomatedWorkflowRequest = Field(
        default_factory=AutomatedWorkflowRequest.get_blank
    )
    config: AutomatedWorkflowConfig = Field(
        default_factory=AutomatedWorkflowConfig.get_blank
    )
    requestSha256: str = ""
    configSha256: str = ""
    contentSha256: str = ""

    @classmethod
    def get_blank(cls) -> "OrchestrationRequestRecord":
        return cls()

    @classmethod
    def get_example(cls) -> "OrchestrationRequestRecord":
        return cls(
            workflowRunId="workflow-1",
            createdAtNs=1,
            request=AutomatedWorkflowRequest.get_example(),
            config=AutomatedWorkflowConfig.get_example(),
            requestSha256="0" * 64,
            configSha256="1" * 64,
        )


class OrchestrationResumeRecord(AgentDataModel):
    """One append-only set of answers supplied during resume."""

    recordType: Literal["automatedWorkflowResume"] = "automatedWorkflowResume"
    workflowRunId: str = ""
    resumeId: str = ""
    createdAtNs: int = Field(default=0, ge=0)
    answeredAttempt: WorkflowStageLink | None = None
    questionIds: list[str] = Field(default_factory=list)
    answers: dict[str, Any] = Field(default_factory=dict)
    contentSha256: str = ""

    @classmethod
    def get_blank(cls) -> "OrchestrationResumeRecord":
        return cls()

    @classmethod
    def get_example(cls) -> "OrchestrationResumeRecord":
        return cls(
            workflowRunId="workflow-1",
            resumeId="resume-1",
            createdAtNs=1,
            answers={"approvePlanChecksum": "0" * 64},
        )


def artifact_model_to_ref(value: ArtifactReferenceModel) -> ArtifactRef:
    """Convert an agent artifact model to a validated core artifact reference."""
    return ArtifactRef(
        scope=value.scope,
        assay=value.assay,
        kind=value.kind,
        artifact_id=value.artifactId,
    )
