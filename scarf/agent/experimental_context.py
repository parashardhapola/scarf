"""Tool-driven experimental-design and batch-correction assessment."""

import json
import math
import re
from collections.abc import Mapping, Sequence
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ..graph.feature_projection import graph_cell_selection
from ..metadata.queries import reduce_observation_units
from ..metadata.selection import resolve_cell_aligned_artifact
from ..metrics.association import coefficient_estimability
from ..quality_control.filtering import (
    _validated_sample_labels,
    gaussian_quantile_bounds,
)
from ..storage.artifacts import (
    fingerprint_array,
    fingerprint_strings,
    inspect_artifact,
)
from ..storage.refs import ArtifactRef
from ..storage.selections import read_stored_selection_mask
from ..utils.logging import logger
from .characterize_covariates import (
    CovariateCharacterization,
    _SelectionBoundCells,
    characterize_covariates,
)
from .config import AgentRunConfig
from .config._deps import AGENT_INSTALL_HINT
from .config.agent_exec import run_agent_sync
from .qc_profiles import (
    AutoFilterProjection,
    QcMetricRole,
    RegisteredCellQcProfile,
    RegisteredQcProjection,
    offered_registered_qc_profiles,
    project_auto_filter_profile,
    qc_metric_execution_name,
    registered_qc_metric_role,
)
from .tools import artifact_reference, core_artifact_reference
from .types import (
    AgentDataModel,
    AgentRunInfo,
    ArtifactReferenceModel,
    BatchCorrectionAction,
    BatchSafetyEvidence,
    BatchSafetyStatus,
    ExperimentalBiologyHandoff,
    ExperimentalTuningHandoff,
    StageStatus,
)

if TYPE_CHECKING:
    from ..datastore.pipeline_run import PipelineRun

try:
    from pydantic import ConfigDict, Field, model_validator
    from pydantic_ai import ModelRetry, RunContext, Tool, UnexpectedModelBehavior
    from pydantic_ai.tools import ToolDefinition
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc

__all__ = [
    "BatchCorrectionPlan",
    "BatchSafetyEvidence",
    "CellQcPlan",
    "CellQcProfileEvidence",
    "CaptureFailureEvidence",
    "ContrastPlan",
    "CovariateEvidence",
    "ExperimentalContextAgent",
    "ExperimentalContextDecision",
    "ExperimentalContextDependencies",
    "ExperimentalContextResult",
    "InferenceUnit",
    "NamedArtifactSource",
    "QcMetricSourceEvidence",
    "QcSourceConcordance",
    "RepresentationEvaluation",
    "RegisteredCellQcProfile",
    "analyze_experimental_design",
    "contrast_plans_from_characterization",
    "inspect_cell_covariates",
    "score_current_representation",
    "validate_experimental_context",
]

type ColumnDomain = Literal["biological", "technical", "design", "ignore", "unknown"]
type IntegrationMetric = Literal[
    "iLISI",
    "cLISI",
    "graphConnectivity",
    "proportionalBatchMixing",
]
type CellQcAction = Literal[
    "skip",
    "globalGaussian",
    "sampleMad",
    "registeredMad",
]
type LegacyCellQcAction = Literal["skip", "globalGaussian", "sampleMad"]
type CellQcDriverType = Literal["RNA", "ATAC"]

_CONTEXT_LIMIT = 1200
_MAX_QC_SAMPLE_PROFILES = 4


class InferenceUnit(AgentDataModel):
    """Observation and independent units for one biological coefficient."""

    observationUnit: str | None = None
    independentUnit: str | None = None

    @classmethod
    def get_blank(cls) -> "InferenceUnit":
        return cls()

    @classmethod
    def get_example(cls) -> "InferenceUnit":
        return cls(observationUnit="sample", independentUnit="donor")


class BatchCorrectionPlan(AgentDataModel):
    """A grounded recommendation about whether Harmony should be evaluated."""

    action: BatchCorrectionAction
    batchColumns: list[str] = Field(default_factory=list)
    preserveColumns: list[str] = Field(default_factory=list)
    metricsRequired: list[IntegrationMetric] = Field(default_factory=list)
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "BatchCorrectionPlan":
        return cls(action="needsInput")

    @classmethod
    def get_example(cls) -> "BatchCorrectionPlan":
        return cls(
            action="evaluateHarmony",
            batchColumns=["batch"],
            preserveColumns=["cell_type", "treatment"],
            metricsRequired=[
                "iLISI",
                "cLISI",
                "graphConnectivity",
            ],
            rationale=(
                "Batch is technical and crossed with treatment, so compare an exact "
                "Harmony candidate while protecting biological labels."
            ),
            evidenceIds=[
                "column:batch",
                "estimability:treatment",
                "batchEstimability:treatment:batch",
            ],
        )


class NamedArtifactSource(AgentDataModel):
    """One semantic name bound to an exact immutable artifact."""

    name: str = ""
    artifact: ArtifactReferenceModel = Field(default_factory=ArtifactReferenceModel)

    @model_validator(mode="after")
    def validate_source(self) -> "NamedArtifactSource":
        if self.name != self.name.strip():
            raise ValueError("Artifact source names cannot have surrounding whitespace")
        if bool(self.name.strip()) != bool(self.artifact.artifactId):
            raise ValueError("A named artifact source requires both name and artifact")
        return self

    @classmethod
    def get_blank(cls) -> "NamedArtifactSource":
        return cls()

    @classmethod
    def get_example(cls) -> "NamedArtifactSource":
        return cls(
            name="RNA_percentMito",
            artifact=ArtifactReferenceModel(
                assay="RNA",
                kind="quality_metric",
                artifactId="1" * 64,
            ),
        )


class QcMetricSourceEvidence(AgentDataModel):
    """One source-specific quality metric on the exact active cells."""

    sourceId: str = ""
    metricName: str = ""
    metricRole: QcMetricRole = "diagnostic"
    assay: str | None = None
    sourceType: Literal["metadataColumn", "artifact"] = "metadataColumn"
    origin: Literal[
        "ingestionMetadata",
        "derivedArtifact",
        "externalArtifact",
    ] = "ingestionMetadata"
    executionName: str = ""
    metadataColumn: str | None = None
    artifact: ArtifactReferenceModel | None = None
    cellSelection: ArtifactReferenceModel | None = None
    inputArtifacts: list[ArtifactReferenceModel] = Field(default_factory=list)
    provenanceOperation: str | None = None
    valuesFingerprint: str = ""
    activeCells: int = 0
    missingCells: int = 0
    missingCellsByCapture: dict[str, int] = Field(default_factory=dict)
    usableForFiltering: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source(self) -> "QcMetricSourceEvidence":
        if (
            not self.sourceId
            and not self.metricName
            and self.metadataColumn is None
            and self.artifact is None
        ):
            return self
        if self.sourceType == "metadataColumn":
            if self.metadataColumn is None or self.artifact is not None:
                raise ValueError(
                    "A metadata QC source requires only metadataColumn provenance"
                )
            if self.origin != "ingestionMetadata":
                raise ValueError(
                    "Metadata QC sources must use ingestionMetadata origin"
                )
        elif self.artifact is None or self.metadataColumn is not None:
            raise ValueError("An artifact QC source requires only artifact provenance")
        if self.activeCells < 0 or not 0 <= self.missingCells <= self.activeCells:
            raise ValueError("QC source active and missing counts are inconsistent")
        if self.usableForFiltering and self.missingCells:
            raise ValueError("A QC source with missing values cannot drive filtering")
        return self


class QcSourceConcordance(AgentDataModel):
    """Observed agreement between imported and derived forms of one metric."""

    metricRole: QcMetricRole = "diagnostic"
    leftSourceId: str = ""
    rightSourceId: str = ""
    comparedCells: int = 0
    missingCells: int = 0
    meanAbsoluteDifference: float | None = None
    maximumAbsoluteDifference: float | None = None
    pearsonCorrelation: float | None = None
    exactlyEqual: bool = False
    numericallyClose: bool = False
    evidenceId: str = ""


class CaptureFailureEvidence(AgentDataModel):
    """Multi-axis capture anomaly plus exclusion-safety inputs."""

    capture: str = ""
    activeCells: int = 0
    retainedCells: int = 0
    retainedFraction: float = 0.0
    adverseAxes: list[QcMetricRole] = Field(default_factory=list)
    independentAdverseAxes: int = 0
    metricMissingFractions: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    wholeCaptureFailure: bool = False
    conditionAndUnitSafety: list[dict[str, Any]] = Field(default_factory=list)
    preservesConditionCoverage: bool = False
    preservesIndependentUnitCoverage: bool = False
    exclusionEligible: bool = False
    doubletEvidenceIds: list[str] = Field(default_factory=list)
    evidenceId: str = ""

    @model_validator(mode="after")
    def validate_failure(self) -> "CaptureFailureEvidence":
        if self.independentAdverseAxes != len(set(self.adverseAxes)):
            raise ValueError("Capture failure axis count must match its unique axes")
        if self.wholeCaptureFailure != (self.independentAdverseAxes >= 2):
            raise ValueError(
                "Whole-capture failure requires at least two independent QC axes"
            )
        if self.exclusionEligible and (
            not self.wholeCaptureFailure
            or not self.preservesConditionCoverage
            or not self.preservesIndependentUnitCoverage
        ):
            raise ValueError(
                "Capture exclusion requires failure and preserved design coverage"
            )
        return self


type ContrastTest = Literal["mann_whitney", "kruskal_wallis", "wilcoxon"]
type ContrastSampleStatistic = Literal["mean", "median", "fraction"]
type ContrastStatus = Literal["licensed", "blocked", "needsInput"]


class ContrastPlan(AgentDataModel):
    """One deterministic sample-aware statistical-testing license."""

    coefficient: str = ""
    groupOrder: list[str | int | float | bool] = Field(default_factory=list)
    sampleBy: str | None = None
    pairBy: str | None = None
    test: ContrastTest | None = None
    sampleStatistic: ContrastSampleStatistic = "mean"
    expressionCutoff: float = 0.0
    status: ContrastStatus = "blocked"
    betweenUnitDesign: bool = False
    replicationPassed: bool = False
    estimabilityPassed: bool = False
    pairedCoveragePassed: bool | None = None
    replication: dict[str, Any] = Field(default_factory=dict)
    estimability: dict[str, Any] = Field(default_factory=dict)
    pairedCoverage: dict[str, Any] = Field(default_factory=dict)
    blockedReasons: list[str] = Field(default_factory=list)
    evidenceId: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contrast(self) -> "ContrastPlan":
        if self.coefficient != self.coefficient.strip():
            raise ValueError(
                "Contrast coefficient cannot contain surrounding whitespace"
            )
        if self.sampleBy is not None and (
            not self.sampleBy.strip() or self.sampleBy != self.sampleBy.strip()
        ):
            raise ValueError("Contrast sampleBy must be a non-empty trimmed name")
        if self.pairBy is not None and (
            not self.pairBy.strip() or self.pairBy != self.pairBy.strip()
        ):
            raise ValueError("Contrast pairBy must be a non-empty trimmed name")
        group_keys = [(type(value).__name__, repr(value)) for value in self.groupOrder]
        if len(group_keys) != len(set(group_keys)):
            raise ValueError("Contrast groupOrder must contain unique values")
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in self.groupOrder
        ):
            raise ValueError("Contrast groupOrder cannot contain non-finite values")
        if not math.isfinite(self.expressionCutoff):
            raise ValueError("Contrast expressionCutoff must be finite")
        if self.sampleStatistic != "fraction" and self.expressionCutoff != 0.0:
            raise ValueError(
                "Contrast expressionCutoff is only used with fraction summaries"
            )
        if self.test == "mann_whitney" and len(self.groupOrder) != 2:
            raise ValueError("mann_whitney requires exactly two ordered groups")
        if self.test == "kruskal_wallis" and len(self.groupOrder) < 3:
            raise ValueError("kruskal_wallis requires at least three ordered groups")
        if self.test == "wilcoxon":
            if len(self.groupOrder) != 2 or self.pairBy is None:
                raise ValueError(
                    "wilcoxon requires exactly two groups and an explicit pairBy"
                )
        elif self.pairBy is not None and self.test is not None:
            raise ValueError("A paired contrast must use the wilcoxon test")
        if self.status == "licensed":
            if (
                not self.coefficient
                or self.sampleBy is None
                or self.test is None
                or self.blockedReasons
                or not self.betweenUnitDesign
                or not self.replicationPassed
                or not self.estimabilityPassed
                or (self.pairBy is not None and self.pairedCoveragePassed is not True)
            ):
                raise ValueError(
                    "A licensed contrast requires resolved design, replication, "
                    "estimability, and paired coverage"
                )
        elif not self.blockedReasons:
            raise ValueError("A non-licensed contrast requires blockedReasons")
        return self

    @classmethod
    def get_blank(cls) -> "ContrastPlan":
        return cls(blockedReasons=["unresolvedContrast"])


def _validate_qc_sources(
    *,
    action: CellQcAction,
    attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    sample_column: str | None,
    sample_artifact: NamedArtifactSource | None,
    registered_profile: RegisteredCellQcProfile | None = None,
    allow_metric_name_collisions: bool = False,
) -> None:
    if len(attributes) != len(set(attributes)):
        raise ValueError("Cell-QC metadata attributes must be unique")
    if any(
        not attribute.strip() or attribute != attribute.strip()
        for attribute in attributes
    ):
        raise ValueError(
            "Cell-QC metadata attributes cannot be blank or have surrounding whitespace"
        )
    artifact_names = [source.name for source in artifact_metrics]
    if len(artifact_names) != len(set(artifact_names)):
        raise ValueError("Cell-QC artifact metric names must be unique")
    if not allow_metric_name_collisions and set(attributes) & set(artifact_names):
        raise ValueError(
            "Cell-QC metadata and artifact metric names collide; explicitly "
            "validated multi-source evidence is required"
        )
    if any(source.artifact.kind != "quality_metric" for source in artifact_metrics):
        raise ValueError(
            "Cell-QC artifactMetrics must reference quality_metric artifacts"
        )
    if sample_column is not None and sample_artifact is not None:
        raise ValueError(
            "Cell-QC sampleColumn and sampleArtifact are mutually exclusive"
        )
    if sample_column is not None and (
        not sample_column.strip() or sample_column != sample_column.strip()
    ):
        raise ValueError(
            "Cell-QC sampleColumn cannot be blank or have surrounding whitespace"
        )
    if sample_artifact is not None and sample_artifact.artifact.kind != "hto_identity":
        raise ValueError(
            "Cell-QC sampleArtifact must reference an hto_identity artifact"
        )
    if sample_artifact is not None and sample_artifact.name in artifact_names:
        raise ValueError("Cell-QC sample and metric artifact names must be distinct")
    if registered_profile is not None:
        if registered_profile == "retainWithFlags":
            if action != "skip":
                raise ValueError(
                    "retainWithFlags must use the non-filtering skip action"
                )
            if sample_column is not None or sample_artifact is not None:
                raise ValueError("retainWithFlags cannot include a capture source")
            return
        if action != "registeredMad":
            raise ValueError(f"{registered_profile} must use the registeredMad action")
        capture_profile = registered_profile in {
            "captureMad5",
            "captureMad3Sensitivity",
            "pooledReferenceMad5",
        }
        has_one_capture_source = (sample_column is None) != (sample_artifact is None)
        if capture_profile and not has_one_capture_source:
            raise ValueError(
                f"{registered_profile} requires exactly one proven capture source"
            )
        if not capture_profile and (
            sample_column is not None or sample_artifact is not None
        ):
            raise ValueError(f"{registered_profile} cannot include a capture source")
        if not attributes and not artifact_metrics:
            raise ValueError("Registered MAD filtering requires at least one metric")
        return
    if action == "registeredMad":
        raise ValueError("registeredMad requires a registeredProfile")
    if action == "skip" and (attributes or artifact_metrics):
        raise ValueError("skip cannot include Cell-QC metrics")
    if action != "skip" and not attributes and not artifact_metrics:
        raise ValueError("Cell-QC filtering requires at least one metric")
    if action == "sampleMad" and (sample_column is None) == (sample_artifact is None):
        raise ValueError(
            "sampleMad requires exactly one sampleColumn or sampleArtifact"
        )
    if action != "sampleMad" and (
        sample_column is not None or sample_artifact is not None
    ):
        raise ValueError("Only sampleMad can include a sample source")


class CellQcProfileEvidence(AgentDataModel):
    """Projected retention for one registered or legacy cell-QC profile."""

    profileId: str = ""
    action: CellQcAction = "skip"
    registeredProfile: RegisteredCellQcProfile | None = None
    driverAssay: str | None = None
    driverAssayType: CellQcDriverType | None = None
    sampleColumn: str | None = None
    sampleArtifact: NamedArtifactSource | None = None
    captureColumn: str | None = None
    captureArtifact: NamedArtifactSource | None = None
    attributes: list[str] = Field(default_factory=list)
    artifactMetrics: list[NamedArtifactSource] = Field(default_factory=list)
    metricSources: list[QcMetricSourceEvidence] = Field(default_factory=list)
    sourceConcordance: list[QcSourceConcordance] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    resolvedBounds: dict[str, Any] | list[dict[str, Any]] = Field(default_factory=dict)
    activeCells: int = 0
    retainedCells: int = 0
    retainedFraction: float = 0.0
    activeCellsByCapture: dict[str, int] = Field(default_factory=dict)
    sampleRetainedCells: dict[str, int] = Field(default_factory=dict)
    retainedCellsByColumn: dict[str, dict[str, int]] = Field(default_factory=dict)
    unsafeRetentionGroups: list[str] = Field(default_factory=list)
    flaggedCells: dict[str, int] = Field(default_factory=dict)
    metricFlaggedCells: dict[str, dict[str, int]] = Field(default_factory=dict)
    failedCaptureCandidates: list[str] = Field(default_factory=list)
    captureFailureEvidence: list[CaptureFailureEvidence] = Field(default_factory=list)
    excludableCaptureCandidates: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    evidenceId: str = ""

    @model_validator(mode="after")
    def validate_sources(self) -> "CellQcProfileEvidence":
        _validate_qc_sources(
            action=self.action,
            attributes=self.attributes,
            artifact_metrics=self.artifactMetrics,
            sample_column=self.sampleColumn,
            sample_artifact=self.sampleArtifact,
            registered_profile=self.registeredProfile,
            allow_metric_name_collisions=True,
        )
        if self.captureColumn is not None and self.captureArtifact is not None:
            raise ValueError(
                "Cell-QC captureColumn and captureArtifact are mutually exclusive"
            )
        if (
            self.captureArtifact is not None
            and self.captureArtifact.artifact.kind != "hto_identity"
        ):
            raise ValueError(
                "Cell-QC captureArtifact must reference an hto_identity artifact"
            )
        failures = {item.capture: item for item in self.captureFailureEvidence}
        if len(failures) != len(self.captureFailureEvidence):
            raise ValueError("Cell-QC capture failure evidence must be unique")
        expected_failed = sorted(
            capture for capture, item in failures.items() if item.wholeCaptureFailure
        )
        if failures and sorted(self.failedCaptureCandidates) != expected_failed:
            raise ValueError(
                "Cell-QC failed captures must match their multi-axis evidence"
            )
        expected_excludable = sorted(
            capture for capture, item in failures.items() if item.exclusionEligible
        )
        if failures and sorted(self.excludableCaptureCandidates) != expected_excludable:
            raise ValueError(
                "Cell-QC excludable captures must match design-safety evidence"
            )
        return self

    @classmethod
    def get_blank(cls) -> "CellQcProfileEvidence":
        return cls()

    @classmethod
    def get_example(cls) -> "CellQcProfileEvidence":
        return cls(
            profileId="cellQc:RNA:globalMad5",
            action="registeredMad",
            registeredProfile="globalMad5",
            driverAssay="RNA",
            driverAssayType="RNA",
            attributes=["RNA_nCounts", "RNA_nFeatures"],
            artifactMetrics=[NamedArtifactSource.get_example()],
            parameters={"nMads": 5.0},
            activeCells=100,
            retainedCells=96,
            retainedFraction=0.96,
            evidenceId="qcProfile:cellQc:RNA:globalMad5",
        )


class CellQcPlan(AgentDataModel):
    """A validated selection from the bounded cell-QC profiles."""

    action: CellQcAction = "skip"
    registeredProfile: RegisteredCellQcProfile | None = None
    profileId: str = ""
    driverAssay: str | None = None
    driverAssayType: CellQcDriverType | None = None
    sampleColumn: str | None = None
    sampleArtifact: NamedArtifactSource | None = None
    attributes: list[str] = Field(default_factory=list)
    artifactMetrics: list[NamedArtifactSource] = Field(default_factory=list)
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sources(self) -> "CellQcPlan":
        _validate_qc_sources(
            action=self.action,
            attributes=self.attributes,
            artifact_metrics=self.artifactMetrics,
            sample_column=self.sampleColumn,
            sample_artifact=self.sampleArtifact,
            registered_profile=self.registeredProfile,
            allow_metric_name_collisions=True,
        )
        return self

    @classmethod
    def get_blank(cls) -> "CellQcPlan":
        return cls()

    @classmethod
    def get_example(cls) -> "CellQcPlan":
        evidence = CellQcProfileEvidence.get_example()
        return cls(
            action=evidence.action,
            registeredProfile=evidence.registeredProfile,
            profileId=evidence.profileId,
            driverAssay=evidence.driverAssay,
            driverAssayType=evidence.driverAssayType,
            sampleColumn=evidence.sampleColumn,
            sampleArtifact=evidence.sampleArtifact,
            attributes=evidence.attributes,
            artifactMetrics=evidence.artifactMetrics,
            rationale="Use the bounded global profile for the RNA assay.",
            evidenceIds=[evidence.evidenceId],
        )


class ExperimentalContextDecision(AgentDataModel):
    """Model-authored choices that are revalidated against the datastore."""

    columnDomains: dict[str, ColumnDomain] = Field(default_factory=dict)
    coefficientsOfInterest: list[str] = Field(default_factory=list)
    unitsOfInference: dict[str, InferenceUnit] = Field(default_factory=dict)
    batchCorrection: BatchCorrectionPlan = Field(
        default_factory=BatchCorrectionPlan.get_blank
    )
    cellQc: CellQcPlan = Field(default_factory=CellQcPlan.get_blank)
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)
    needsInput: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ExperimentalContextDecision":
        return cls()

    @classmethod
    def get_example(cls) -> "ExperimentalContextDecision":
        return cls(
            columnDomains={
                "batch": "technical",
                "sample": "design",
                "donor": "design",
                "treatment": "biological",
            },
            coefficientsOfInterest=["treatment"],
            unitsOfInference={"treatment": InferenceUnit.get_example()},
            batchCorrection=BatchCorrectionPlan.get_example(),
            rationale="Treatment is the primary between-sample contrast.",
            evidenceIds=[
                "column:batch",
                "column:donor",
                "column:sample",
                "column:treatment",
            ],
        )


class RepresentationEvaluation(AgentDataModel):
    """Bounded integration metrics for one exact graph representation."""

    available: bool = False
    assay: str | None = None
    cellSelection: ArtifactReferenceModel | None = None
    neighbors: ArtifactReferenceModel | None = None
    connectivityMap: ArtifactReferenceModel | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "RepresentationEvaluation":
        return cls()

    @classmethod
    def get_example(cls) -> "RepresentationEvaluation":
        return cls(
            available=True,
            assay="RNA",
            cellSelection=ArtifactReferenceModel(
                scope="datastore",
                kind="cell_selection",
                artifactId="c" * 64,
            ),
            neighbors=ArtifactReferenceModel(
                assay="RNA",
                kind="neighbors",
                artifactId="a" * 64,
            ),
            connectivityMap=ArtifactReferenceModel(
                assay="RNA",
                kind="connectivity_map",
                artifactId="b" * 64,
            ),
            metrics={"iLISI:batch": 0.71, "cLISI:cell_type": 0.94},
            evidenceIds=[
                "metric:iLISI:batch:assay:RNA:neighbors:example-neighbors",
                "metric:cLISI:cell_type:assay:RNA:neighbors:example-neighbors",
            ],
        )


class CovariateEvidence(AgentDataModel):
    """One deterministic covariate characterization returned by a tool."""

    characterization: CovariateCharacterization = Field(
        default_factory=lambda: CovariateCharacterization(status="needsInput")
    )
    batchSafety: list[BatchSafetyEvidence] = Field(default_factory=list)
    qcProfiles: list[CellQcProfileEvidence] = Field(default_factory=list)
    qcMetricSources: list[QcMetricSourceEvidence] = Field(default_factory=list)
    qcSourceConcordance: list[QcSourceConcordance] = Field(default_factory=list)
    contrastPlans: list[ContrastPlan] = Field(default_factory=list)
    htoIdentityColumns: list[str] = Field(default_factory=list)
    htoIdentityArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_example(cls) -> "CovariateEvidence":
        return cls(
            characterization=CovariateCharacterization(
                status="done",
                notes=["Example deterministic covariate characterization"],
            ),
            qcProfiles=[CellQcProfileEvidence.get_example()],
            htoIdentityColumns=["sample_id"],
            htoIdentityArtifacts=[
                NamedArtifactSource(
                    name="HTO_htoIdentity",
                    artifact=ArtifactReferenceModel(
                        assay="HTO",
                        kind="hto_identity",
                        artifactId="2" * 64,
                    ),
                )
            ],
            evidenceIds=[
                "column:batch",
                CellQcProfileEvidence.get_example().evidenceId,
                "htoIdentity:sample_id",
                f"htoIdentityArtifact:HTO_htoIdentity:{'2' * 64}",
            ],
        )


class ExperimentalContextResult(AgentDataModel):
    """Canonical experimental-context report returned to the caller."""

    status: StageStatus
    decision: ExperimentalContextDecision
    characterization: CovariateCharacterization
    cellSelection: ArtifactReferenceModel | None = None
    cellQc: CellQcPlan = Field(default_factory=CellQcPlan.get_blank)
    qcProfiles: list[CellQcProfileEvidence] = Field(default_factory=list)
    qcMetricSources: list[QcMetricSourceEvidence] = Field(default_factory=list)
    qcSourceConcordance: list[QcSourceConcordance] = Field(default_factory=list)
    contrastPlans: list[ContrastPlan] = Field(default_factory=list)
    qualityMetricArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    htoIdentityColumns: list[str] = Field(default_factory=list)
    htoIdentityArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    batchSafety: list[BatchSafetyEvidence] = Field(default_factory=list)
    currentRepresentation: RepresentationEvaluation = Field(
        default_factory=RepresentationEvaluation.get_blank
    )
    notes: list[str] = Field(default_factory=list)
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @classmethod
    def get_blank(cls) -> "ExperimentalContextResult":
        return cls(
            status="needsInput",
            decision=ExperimentalContextDecision.get_blank(),
            characterization=CovariateCharacterization(status="needsInput"),
        )

    @classmethod
    def get_example(cls) -> "ExperimentalContextResult":
        representation = RepresentationEvaluation.get_example()
        return cls(
            status="done",
            decision=ExperimentalContextDecision.get_example(),
            characterization=CovariateCharacterization(
                status="done",
                notes=["Example deterministic design characterization"],
            ),
            cellSelection=representation.cellSelection,
            qcProfiles=[CellQcProfileEvidence.get_example()],
            qualityMetricArtifacts=[NamedArtifactSource.get_example()],
            htoIdentityColumns=["sample_id"],
            htoIdentityArtifacts=[
                NamedArtifactSource(
                    name="HTO_htoIdentity",
                    artifact=ArtifactReferenceModel(
                        assay="HTO",
                        kind="hto_identity",
                        artifactId="2" * 64,
                    ),
                )
            ],
            batchSafety=[BatchSafetyEvidence.get_example()],
            currentRepresentation=representation,
            runInfo=AgentRunInfo.get_example(),
        )

    def to_parameter_tuning_handoff(self) -> ExperimentalTuningHandoff:
        """Return validated integration inputs for Parameter Tuning."""
        if self.status != "done":
            raise ValueError(
                "Experimental Context must be done before creating a tuning handoff"
            )
        if self.cellSelection is None:
            raise ValueError("Experimental Context result lacks a cell selection")
        plan = self.decision.batchCorrection
        batch_columns = sorted(plan.batchColumns)
        safety = sorted(
            (
                item
                for item in self.batchSafety
                if item.batchColumns == batch_columns
                and item.coefficient in self.decision.coefficientsOfInterest
            ),
            key=lambda item: item.coefficient,
        )
        if plan.action in {"evaluateHarmony", "unsafe"}:
            expected = set(self.decision.coefficientsOfInterest)
            if {item.coefficient for item in safety} != expected:
                raise ValueError(
                    "Experimental Context result lacks exact batch safety evidence"
                )
            if any(item.evidenceId not in plan.evidenceIds for item in safety):
                raise ValueError(
                    "Batch-correction plan does not cite its exact safety evidence"
                )
            if plan.action == "evaluateHarmony" and any(
                item.status != "safe" for item in safety
            ):
                raise ValueError("Harmony plan contains non-safe batch evidence")
            if plan.action == "unsafe" and (
                any(item.status == "notComputed" for item in safety)
                or not any(item.status == "unsafe" for item in safety)
            ):
                raise ValueError("Unsafe plan lacks exact unsafe batch evidence")
        return ExperimentalTuningHandoff(
            cellSelection=self.cellSelection,
            batchAction=plan.action,
            batchColumns=batch_columns,
            preservationColumns=list(plan.preserveColumns),
            coefficientsOfInterest=list(self.decision.coefficientsOfInterest),
            batchSafety=safety,
            evidenceIds=sorted({*self.decision.evidenceIds, *plan.evidenceIds}),
        )

    def to_biological_handoff(
        self,
        coefficient: str | None = None,
    ) -> ExperimentalBiologyHandoff:
        """Return one explicitly resolved biological coefficient."""
        if self.status != "done":
            raise ValueError(
                "Experimental Context must be done before creating a biology handoff"
            )
        if self.cellSelection is None:
            raise ValueError("Experimental Context result lacks a cell selection")
        coefficients = list(self.decision.coefficientsOfInterest)
        if coefficient is None:
            if len(coefficients) != 1:
                raise ValueError(
                    "Select one coefficient explicitly for biological interpretation"
                )
            coefficient = coefficients[0]
        if coefficient not in coefficients:
            raise ValueError(f"Unknown coefficient of interest {coefficient!r}")
        records = {
            record.get("name"): record
            for record in self.characterization.coefficients
            if isinstance(record.get("name"), str)
        }
        record = records.get(coefficient)
        if record is None:
            raise ValueError(f"Missing characterization for {coefficient!r}")
        reports = {
            report.get("coefficient"): report
            for report in self.characterization.confounding
            if isinstance(report.get("coefficient"), str)
        }
        report = reports.get(coefficient)
        known_evidence = characterization_evidence(self.characterization)
        relevant_evidence = {
            f"column:{coefficient}",
            f"coefficient:{coefficient}",
            f"estimability:{coefficient}",
            *(
                evidence_id
                for evidence_id in known_evidence
                if evidence_id.startswith(f"confounding:{coefficient}:")
            ),
        }
        for unit_name in (
            record.get("observationUnit"),
            record.get("independentUnit"),
        ):
            if isinstance(unit_name, str):
                relevant_evidence.add(f"column:{unit_name}")
        return ExperimentalBiologyHandoff(
            cellSelection=self.cellSelection,
            conditionColumn=coefficient,
            observationUnit=record.get("observationUnit"),
            independentUnit=record.get("independentUnit"),
            coefficientScope=str(record.get("scope", "")),
            estimability=dict(report.get("estimability") or {}) if report else {},
            evidenceIds=sorted(relevant_evidence.intersection(known_evidence)),
        )


class ExperimentalContextDependencies(AgentDataModel):
    """Runtime-only state shared by the agent's read-only tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    store: Any = Field(default=None, exclude=True)
    cells: Any = Field(default=None, exclude=True)
    neighbors: Any = Field(default=None, exclude=True)
    connectivityMap: Any = Field(default=None, exclude=True)
    cellSelection: Any = Field(default=None, exclude=True)
    studyContext: str = ""
    studyObjective: str = ""
    directions: dict[str, Any] = Field(default_factory=dict)
    evidenceIds: set[str] = Field(default_factory=set)
    characterization: CovariateCharacterization | None = None
    batchSafety: dict[str, BatchSafetyEvidence] = Field(default_factory=dict)
    qcProfiles: dict[str, CellQcProfileEvidence] = Field(default_factory=dict)
    qcMetricSources: list[QcMetricSourceEvidence] = Field(default_factory=list)
    qcSourceConcordance: list[QcSourceConcordance] = Field(default_factory=list)
    contrastPlans: dict[str, ContrastPlan] = Field(default_factory=dict)
    htoIdentityColumns: list[str] = Field(default_factory=list)
    qualityMetricArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    htoIdentityArtifacts: list[NamedArtifactSource] = Field(default_factory=list)
    currentRepresentation: RepresentationEvaluation = Field(
        default_factory=RepresentationEvaluation.get_blank
    )
    toolCalls: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "ExperimentalContextDependencies":
        return cls()

    @classmethod
    def get_example(cls) -> "ExperimentalContextDependencies":
        return cls(
            studyContext="Case-control study with samples nested in donors.",
            studyObjective=(
                "Discover populations while preserving the case-control contrast."
            ),
            directions={"columnDomains": {"batch": "technical"}},
        )


def _prepare_experimental_context_tool(
    ctx: RunContext[ExperimentalContextDependencies],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    """Expose each context tool once and in its required dependency order."""
    completed_calls = set(ctx.deps.toolCalls)
    if tool_definition.name == "inspect_cell_covariates":
        return None if tool_definition.name in completed_calls else tool_definition
    if tool_definition.name == "analyze_experimental_design":
        if (
            "inspect_cell_covariates" not in completed_calls
            or tool_definition.name in completed_calls
        ):
            return None
        return tool_definition
    if tool_definition.name == "score_current_representation":
        if (
            "analyze_experimental_design" not in completed_calls
            or tool_definition.name in completed_calls
        ):
            return None
        characterization = ctx.deps.characterization
        if characterization is not None and not any(
            record.get("domain") == "technical" and record.get("kind") == "categorical"
            for record in characterization.columns
        ):
            return None
        return tool_definition
    return tool_definition


def characterization_evidence(
    characterization: CovariateCharacterization,
) -> set[str]:
    """Build stable evidence IDs from one deterministic characterization."""
    evidence_ids = {
        f"column:{record['name']}"
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    for record in characterization.coefficients:
        coefficient = record.get("name")
        if isinstance(coefficient, str):
            evidence_ids.add(f"coefficient:{coefficient}")
    for report in characterization.confounding:
        coefficient = report.get("coefficient")
        if not isinstance(coefficient, str):
            continue
        evidence_ids.add(f"estimability:{coefficient}")
        for pair in report.get("pairs", []):
            technical = pair.get("technical")
            if isinstance(technical, str):
                evidence_ids.add(f"confounding:{coefficient}:{technical}")
    return evidence_ids


def contrast_plans_from_characterization(
    characterization: CovariateCharacterization,
) -> list[ContrastPlan]:
    """Build deterministic test licenses from bounded coefficient evidence."""
    reports = {
        report.get("coefficient"): report
        for report in characterization.confounding
        if isinstance(report.get("coefficient"), str)
    }
    plans: list[ContrastPlan] = []
    for record in characterization.coefficients:
        coefficient = record.get("name")
        if not isinstance(coefficient, str):
            continue
        report = reports.get(coefficient, {})
        raw_groups = record.get("groupOrder")
        group_order = (
            list(raw_groups)
            if isinstance(raw_groups, list)
            and all(
                isinstance(value, str | int | float | bool)
                and not (isinstance(value, float) and not math.isfinite(value))
                for value in raw_groups
            )
            else []
        )
        sample_by = record.get("observationUnit")
        sample_by = sample_by if isinstance(sample_by, str) else None
        independent_unit = record.get("independentUnit")
        independent_unit = (
            independent_unit if isinstance(independent_unit, str) else None
        )
        between_unit = record.get("scope") == "betweenUnit"
        replication = dict(record.get("replication") or {})
        replication_passed = replication.get("sufficient") is True
        estimability = dict(
            record.get("estimability") or report.get("estimability") or {}
        )
        estimability_passed = (
            estimability.get("status") == "ok"
            and estimability.get("coefficientEstimable") is True
            and estimability.get("rankDeficient") is not True
        )
        paired_coverage = dict(record.get("pairedCoverage") or {})
        pair_by: str | None = None
        paired_passed: bool | None = None
        mixed_independent_design = False
        if independent_unit is not None:
            if paired_coverage.get("complete") is True:
                pair_by = independent_unit
                paired_passed = True
            elif paired_coverage.get("betweenIndependentUnits") is True:
                sample_by = independent_unit
            else:
                pair_by = independent_unit
                paired_passed = False
                mixed_independent_design = True

        reasons: list[str] = []
        needs_input = False
        if record.get("kind") != "categorical":
            reasons.append("coefficientRequiresExplicitCategoricalGroups")
            needs_input = True
        if not between_unit:
            reasons.append("coefficientIsNotBetweenUnit")
        if sample_by is None:
            reasons.append("sampleByIsUnresolved")
            needs_input = True
        if len(group_order) < 2:
            reasons.append("fewerThanTwoObservedGroups")
            needs_input = True
        if record.get("groupCountsTruncated") is True:
            reasons.append("groupOrderIsTruncated")
            needs_input = True
        if not replication_passed:
            reasons.append("insufficientIndependentReplication")
        if not estimability_passed:
            reasons.append("coefficientIsNotEstimable")

        test: ContrastTest | None = None
        if pair_by is not None:
            if len(group_order) != 2:
                reasons.append("pairedTestsRequireExactlyTwoGroups")
            else:
                test = "wilcoxon"
            if paired_passed is not True:
                reasons.append("pairedCoverageIsIncomplete")
        elif mixed_independent_design:
            reasons.append("independentUnitStructureIsMixed")
        elif len(group_order) == 2:
            test = "mann_whitney"
        elif len(group_order) >= 3:
            test = "kruskal_wallis"

        reasons = list(dict.fromkeys(reasons))
        status: ContrastStatus = (
            "licensed" if not reasons else "needsInput" if needs_input else "blocked"
        )
        evidence_ids = [
            f"column:{coefficient}",
            f"coefficient:{coefficient}",
            f"estimability:{coefficient}",
        ]
        plans.append(
            ContrastPlan(
                coefficient=coefficient,
                groupOrder=group_order,
                sampleBy=sample_by,
                pairBy=pair_by,
                test=test,
                status=status,
                betweenUnitDesign=between_unit,
                replicationPassed=replication_passed,
                estimabilityPassed=estimability_passed,
                pairedCoveragePassed=paired_passed,
                replication=replication,
                estimability=estimability,
                pairedCoverage=paired_coverage,
                blockedReasons=reasons,
                evidenceId=f"contrastPlan:{coefficient}:{status}",
                evidenceIds=evidence_ids,
            )
        )
    return plans


def _persisted_assay_type(store: Any, assay_name: str) -> str:
    """Read one persisted assay type without inferring modality from features."""
    root = getattr(store, "zw", None)
    attrs = getattr(root, "attrs", {})
    raw_types = attrs.get("assayTypes", {}) if isinstance(attrs, Mapping) else {}
    if isinstance(raw_types, Mapping):
        assay_type = raw_types.get(assay_name)
        if isinstance(assay_type, str):
            return assay_type
    return assay_name if assay_name in {"RNA", "ATAC", "ADT", "HTO"} else "Assay"


def _qc_driver(store: Any) -> tuple[str, CellQcDriverType] | None:
    """Choose the first RNA assay, otherwise the first ATAC assay."""
    assay_names = [str(name) for name in getattr(store, "assay_names", [])]
    for assay_type in ("RNA", "ATAC"):
        for assay_name in assay_names:
            if _persisted_assay_type(store, assay_name) == assay_type:
                return assay_name, assay_type
    return None


def _hto_identity_columns(deps: ExperimentalContextDependencies) -> list[str]:
    """Return explicitly supplied imported HTO identity metadata columns."""
    requested: list[str] = []
    directed_many = deps.directions.get("htoIdentityColumns")
    if isinstance(directed_many, list | tuple):
        requested.extend(str(value) for value in directed_many)
    directed_one = deps.directions.get("htoIdentityColumn")
    if isinstance(directed_one, str):
        requested.append(directed_one)
    available = set(deps.store.cells.columns)
    return list(dict.fromkeys(name for name in requested if name in available))


def _cell_selection_ref(deps: ExperimentalContextDependencies) -> ArtifactRef:
    selection = core_artifact_reference(deps.cellSelection)
    if not isinstance(selection, ArtifactRef):
        raise ValueError("cellSelection must identify an exact artifact")
    if selection.kind != "cell_selection" or selection.scope != "datastore":
        raise ValueError("cellSelection must identify a datastore cell selection")
    return selection


def _active_cell_count(deps: ExperimentalContextDependencies) -> int:
    selection = _cell_selection_ref(deps)
    active = read_stored_selection_mask(
        deps.store.zw,
        selection,
        kind="cell_selection",
        scope="datastore",
        assay=None,
        table_path="cellData",
    )
    if active.ndim != 1 or active.shape[0] != deps.store.cells.N:
        raise ValueError(
            "cellSelection must contain an aligned boolean selection vector"
        )
    return int(active.sum())


def _source_ref(
    source: NamedArtifactSource,
    *,
    expected_kind: str,
) -> ArtifactRef:
    if not isinstance(source, NamedArtifactSource):
        raise TypeError("Artifact sources must be NamedArtifactSource values")
    if not source.name.strip():
        raise ValueError("Artifact sources require a non-empty semantic name")
    artifact = core_artifact_reference(source.artifact)
    if not isinstance(artifact, ArtifactRef) or artifact.kind != expected_kind:
        raise ValueError(
            f"Artifact source {source.name!r} must reference {expected_kind!r}"
        )
    return artifact


def _artifact_evidence_id(source: NamedArtifactSource) -> str:
    return f"htoIdentityArtifact:{source.name}:{source.artifact.artifactId}"


def _hto_artifact_map(
    deps: ExperimentalContextDependencies,
) -> dict[str, ArtifactRef]:
    artifacts: dict[str, ArtifactRef] = {}
    for source in deps.htoIdentityArtifacts:
        if source.name in artifacts:
            raise ValueError("HTO identity artifact names must be unique")
        artifacts[source.name] = _source_ref(
            source,
            expected_kind="hto_identity",
        )
    return artifacts


def _resolved_artifact_values(
    deps: ExperimentalContextDependencies,
    source: NamedArtifactSource,
    *,
    expected_kind: str,
) -> np.ndarray:
    resolved = resolve_cell_aligned_artifact(
        deps.store.zw,
        _source_ref(source, expected_kind=expected_kind),
        cell_selection=_cell_selection_ref(deps),
        expected_kind=expected_kind,
    )
    return np.asarray(resolved.values)


def _artifact_input_references(
    value: Any,
    *,
    limit: int = 16,
) -> list[ArtifactReferenceModel]:
    refs: list[ArtifactReferenceModel] = []
    seen: set[tuple[str, str | None, str, str]] = set()

    def visit(item: Any) -> None:
        if len(refs) >= limit:
            return
        if isinstance(item, ArtifactRef):
            ref = item
        elif isinstance(item, Mapping) and {
            "scope",
            "kind",
            "artifact_id",
        }.issubset(item):
            try:
                ref = ArtifactRef.from_dict(item)
            except (KeyError, TypeError, ValueError):
                ref = None
        else:
            ref = None
        if ref is not None:
            key = (ref.scope, ref.assay, ref.kind, ref.artifact_id)
            if key not in seen:
                seen.add(key)
                refs.append(artifact_reference(ref))
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list | tuple):
            for nested in item:
                visit(nested)

    visit(value)
    return refs


def _qc_metric_sources(
    deps: ExperimentalContextDependencies,
    driver: tuple[str, CellQcDriverType],
) -> tuple[
    dict[str, np.ndarray],
    list[str],
    list[NamedArtifactSource],
    list[QcMetricSourceEvidence],
    list[QcSourceConcordance],
    list[str],
    dict[str, np.ndarray],
]:
    assay_name, assay_type = driver
    del assay_type
    selection = _cell_selection_ref(deps)
    selection_model = artifact_reference(selection)
    active_cells = _active_cell_count(deps)
    metadata_names = _qc_attributes(deps.store, assay_name, driver[1])
    artifact_candidates: list[NamedArtifactSource] = []
    for source in deps.qualityMetricArtifacts:
        artifact = _source_ref(source, expected_kind="quality_metric")
        if artifact.assay == assay_name:
            artifact_candidates.append(source)
    metadata_collisions = set(metadata_names).intersection(
        source.name for source in artifact_candidates
    )

    values_by_execution_name: dict[str, np.ndarray] = {}
    values_by_source: dict[str, np.ndarray] = {}
    sources: list[QcMetricSourceEvidence] = []
    valid_metadata: list[str] = []
    valid_artifacts: list[NamedArtifactSource] = []
    notes: list[str] = []

    for name in metadata_names:
        raw = np.asarray(deps.cells.fetch(name))
        try:
            values = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            fingerprint = fingerprint_strings(raw)
            source_id = f"qcMetric:metadata:{assay_name}:{name}:{fingerprint}"
            sources.append(
                QcMetricSourceEvidence(
                    sourceId=source_id,
                    metricName=name,
                    metricRole=registered_qc_metric_role(name),
                    assay=assay_name,
                    sourceType="metadataColumn",
                    origin="ingestionMetadata",
                    executionName=name,
                    metadataColumn=name,
                    cellSelection=selection_model,
                    valuesFingerprint=fingerprint,
                    activeCells=active_cells,
                    missingCells=active_cells,
                    notes=["Metric is not numeric and cannot drive filtering"],
                )
            )
            notes.append(f"QC metadata source {name!r} is not numeric")
            continue
        if values.ndim != 1 or values.shape != (active_cells,):
            raise ValueError(
                f"QC metadata source {name!r} does not align with cellSelection"
            )
        fingerprint = fingerprint_array(values)
        missing = int((~np.isfinite(values)).sum())
        source_id = f"qcMetric:metadata:{assay_name}:{name}:{fingerprint}"
        usable = missing == 0
        source_notes = (
            [] if usable else [f"{missing} active cells have non-finite metric values"]
        )
        sources.append(
            QcMetricSourceEvidence(
                sourceId=source_id,
                metricName=name,
                metricRole=registered_qc_metric_role(name),
                assay=assay_name,
                sourceType="metadataColumn",
                origin="ingestionMetadata",
                executionName=name,
                metadataColumn=name,
                cellSelection=selection_model,
                valuesFingerprint=fingerprint,
                activeCells=active_cells,
                missingCells=missing,
                usableForFiltering=usable,
                notes=source_notes,
            )
        )
        values_by_source[source_id] = values
        if usable:
            values_by_execution_name[name] = values
            valid_metadata.append(name)
        else:
            notes.extend(source_notes)

    for source in artifact_candidates:
        artifact = _source_ref(source, expected_kind="quality_metric")
        values = np.asarray(
            _resolved_artifact_values(
                deps,
                source,
                expected_kind="quality_metric",
            ),
            dtype=float,
        )
        if values.ndim != 1 or values.shape != (active_cells,):
            raise ValueError(
                f"QC artifact {source.name!r} does not align with cellSelection"
            )
        execution_name = qc_metric_execution_name(
            source.name,
            artifact_id=artifact.artifact_id,
            collides_with_metadata=source.name in metadata_collisions,
        )
        if execution_name in values_by_execution_name:
            raise ValueError(
                f"QC execution metric name {execution_name!r} is not unique"
            )
        fingerprint = fingerprint_array(values)
        missing = int((~np.isfinite(values)).sum())
        status = inspect_artifact(deps.store.zw, artifact)
        operation = status.operation
        origin: Literal[
            "ingestionMetadata",
            "derivedArtifact",
            "externalArtifact",
        ] = (
            "derivedArtifact"
            if operation == "run_feature_percentage"
            else "externalArtifact"
        )
        source_id = (
            f"qcMetric:artifact:{artifact.assay}:{source.name}:{artifact.artifact_id}"
        )
        usable = missing == 0
        source_notes = (
            [] if usable else [f"{missing} active cells have non-finite metric values"]
        )
        sources.append(
            QcMetricSourceEvidence(
                sourceId=source_id,
                metricName=source.name,
                metricRole=registered_qc_metric_role(source.name),
                assay=assay_name,
                sourceType="artifact",
                origin=origin,
                executionName=execution_name,
                artifact=artifact_reference(artifact),
                cellSelection=selection_model,
                inputArtifacts=_artifact_input_references(status.inputs or {}),
                provenanceOperation=operation,
                valuesFingerprint=fingerprint,
                activeCells=active_cells,
                missingCells=missing,
                usableForFiltering=usable,
                notes=source_notes,
            )
        )
        values_by_source[source_id] = values
        if usable:
            values_by_execution_name[execution_name] = values
            valid_artifacts.append(source)
        else:
            notes.extend(source_notes)

    concordance: list[QcSourceConcordance] = []
    metadata_sources = [
        source for source in sources if source.sourceType == "metadataColumn"
    ]
    artifact_sources = [source for source in sources if source.sourceType == "artifact"]
    for left in metadata_sources:
        for right in artifact_sources:
            if left.metricRole != right.metricRole or left.metricRole == "diagnostic":
                continue
            if right.artifact is None:
                raise ValueError("Artifact QC source lacks its exact reference")
            left_values = values_by_source.get(left.sourceId)
            right_values = values_by_source.get(right.sourceId)
            if left_values is None or right_values is None:
                continue
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            compared = int(finite.sum())
            missing = int(len(finite) - compared)
            mean_difference: float | None = None
            maximum_difference: float | None = None
            pearson: float | None = None
            exactly_equal = False
            numerically_close = False
            if compared:
                left_finite = left_values[finite]
                right_finite = right_values[finite]
                differences = np.abs(left_finite - right_finite)
                mean_difference = float(differences.mean())
                maximum_difference = float(differences.max())
                exactly_equal = missing == 0 and bool(
                    np.array_equal(left_finite, right_finite)
                )
                numerically_close = missing == 0 and bool(
                    np.allclose(
                        left_finite,
                        right_finite,
                        rtol=1e-6,
                        atol=1e-8,
                    )
                )
                if (
                    compared >= 2
                    and float(np.std(left_finite)) > 0.0
                    and float(np.std(right_finite)) > 0.0
                ):
                    correlation = float(np.corrcoef(left_finite, right_finite)[0, 1])
                    if math.isfinite(correlation):
                        pearson = correlation
            evidence_id = (
                f"qcConcordance:{left.metricRole}:"
                f"{left.valuesFingerprint}:{right.artifact.artifactId}"
            )
            concordance.append(
                QcSourceConcordance(
                    metricRole=left.metricRole,
                    leftSourceId=left.sourceId,
                    rightSourceId=right.sourceId,
                    comparedCells=compared,
                    missingCells=missing,
                    meanAbsoluteDifference=mean_difference,
                    maximumAbsoluteDifference=maximum_difference,
                    pearsonCorrelation=pearson,
                    exactlyEqual=exactly_equal,
                    numericallyClose=numerically_close,
                    evidenceId=evidence_id,
                )
            )
    return (
        values_by_execution_name,
        valid_metadata,
        valid_artifacts,
        sources,
        concordance,
        notes,
        values_by_source,
    )


def _qc_attributes(store: Any, assay_name: str, assay_type: str) -> list[str]:
    del assay_type
    suffixes = ["nCounts", "nFeatures", "percentMito", "percentRibo"]
    available = set(store.cells.columns)
    return [
        f"{assay_name}_{suffix}"
        for suffix in suffixes
        if f"{assay_name}_{suffix}" in available
    ]


def _derive_missing_percentage_artifacts(
    store: Any,
    *,
    cell_selection: ArtifactRef,
    driver: tuple[str, CellQcDriverType] | None,
    quality_sources: Sequence[NamedArtifactSource],
) -> list[NamedArtifactSource]:
    """Derive missing RNA percentage metrics through public immutable APIs."""
    sources = list(quality_sources)
    if driver is None or driver[1] != "RNA":
        return sources
    if not callable(getattr(store, "set_feature_selection", None)) or not callable(
        getattr(store, "run_feature_percentage", None)
    ):
        return sources
    assay_name = driver[0]
    available_metadata = set(store.cells.columns)
    supplied_roles = {
        registered_qc_metric_role(source.name)
        for source in sources
        if source.artifact.assay == assay_name
    }
    assay = store.get_assay(assay_name)
    feature_ids = np.asarray(assay.feats.fetch_all("ids")).astype(str)
    feature_names = np.asarray(assay.feats.fetch_all("names")).astype(str)
    specifications: tuple[
        tuple[QcMetricRole, str, re.Pattern[str]],
        ...,
    ] = (
        ("mitochondrial", "percentMito", re.compile(r"^(MT-|mt-)")),
        (
            "ribosomal",
            "percentRibo",
            re.compile(r"^(RPS|RPL|MRPS|MRPL|Rps|Rpl|Mrps|Mrpl)"),
        ),
    )
    existing_names = {source.name for source in sources}
    for role, suffix, pattern in specifications:
        metric_name = f"{assay_name}_{suffix}"
        if metric_name in available_metadata or role in supplied_roles:
            continue
        mask = np.fromiter(
            (
                pattern.search(feature_id) is not None
                or pattern.search(feature_name) is not None
                for feature_id, feature_name in zip(
                    feature_ids,
                    feature_names,
                    strict=True,
                )
            ),
            dtype=bool,
            count=assay.feats.N,
        )
        if not mask.any():
            continue
        if metric_name in existing_names:
            raise ValueError(f"Derived QC metric name {metric_name!r} is not unique")
        feature_selection = store.set_feature_selection(
            from_assay=assay_name,
            mask=mask,
            invalidate_cache=False,
        )
        metric = store.run_feature_percentage(
            cell_selection,
            feature_selection,
            invalidate_cache=False,
        )
        sources.append(
            NamedArtifactSource(
                name=metric_name,
                artifact=artifact_reference(metric),
            )
        )
        existing_names.add(metric_name)
        supplied_roles.add(role)
    return sources


def _qc_sample_columns(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
) -> list[str]:
    requested: list[str] = []
    directed = deps.directions.get("cellQc")
    if isinstance(directed, Mapping):
        sample_column = directed.get("sampleColumn")
        if isinstance(sample_column, str):
            requested.append(sample_column)
    if characterization is not None:
        for record in characterization.coefficients:
            observation_unit = record.get("observationUnit")
            if isinstance(observation_unit, str):
                requested.append(observation_unit)
    requested.extend(deps.htoIdentityColumns)
    available = set(deps.store.cells.columns)
    return list(
        dict.fromkeys(name for name in requested if name in available and name != "I")
    )[:_MAX_QC_SAMPLE_PROFILES]


def _qc_profile_id(
    action: LegacyCellQcAction,
    *,
    driver: tuple[str, CellQcDriverType] | None,
    sample_column: str | None = None,
    sample_artifact: NamedArtifactSource | None = None,
) -> str:
    assay_name, assay_type = driver or ("none", "none")
    suffix = {
        "skip": "skip",
        "globalGaussian": "globalGaussian:0.01:0.99",
        "sampleMad": (
            f"sampleMad:metadata:{sample_column}:3:20"
            if sample_artifact is None
            else (
                f"sampleMad:artifact:{sample_artifact.name}:"
                f"{sample_artifact.artifact.artifactId}:3:20"
            )
        ),
    }[action]
    return f"cellQc:{assay_type}:{assay_name}:{suffix}"


def _registered_qc_profile_id(
    profile: RegisteredCellQcProfile,
    *,
    driver: tuple[str, CellQcDriverType],
    sample_column: str | None,
    sample_artifact: NamedArtifactSource | None,
) -> str:
    if sample_column is not None:
        source = f"metadata:{sample_column}"
    elif sample_artifact is not None:
        source = (
            f"artifact:{sample_artifact.name}:{sample_artifact.artifact.artifactId}"
        )
    else:
        source = "global"
    return f"cellQc:{driver[1]}:{driver[0]}:registered:{profile}:{source}"


def _directed_capture_source(
    deps: ExperimentalContextDependencies,
) -> tuple[str | None, NamedArtifactSource | None, np.ndarray] | None:
    directed_qc = deps.directions.get("cellQc")
    qc_directions = dict(directed_qc) if isinstance(directed_qc, Mapping) else {}
    candidates = [
        deps.directions.get("physicalCaptureColumn"),
        qc_directions.get("physicalCaptureColumn"),
        qc_directions.get("captureColumn"),
    ]
    specified = [value for value in candidates if value is not None]
    if not specified:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in specified):
        raise ValueError("physicalCaptureColumn must be a non-empty string")
    names = list(dict.fromkeys(str(value) for value in specified))
    if len(names) != 1:
        raise ValueError("Conflicting physical capture columns were supplied")
    name = names[0]
    matching_artifacts = [
        source for source in deps.htoIdentityArtifacts if source.name == name
    ]
    if len(matching_artifacts) > 1:
        raise ValueError(f"Physical capture artifact {name!r} is not unique")
    if matching_artifacts:
        source = matching_artifacts[0]
        labels = _resolved_artifact_values(
            deps,
            source,
            expected_kind="hto_identity",
        )
        return None, source, np.asarray(labels)
    if name not in deps.cells.columns:
        raise ValueError(
            f"physicalCaptureColumn {name!r} is not observed metadata or an "
            "exact HTO identity artifact"
        )
    return name, None, np.asarray(deps.cells.fetch(name))


def _directed_pooled_reference_captures(
    deps: ExperimentalContextDependencies,
) -> tuple[str, ...] | None:
    directed_qc = deps.directions.get("cellQc")
    qc_directions = dict(directed_qc) if isinstance(directed_qc, Mapping) else {}
    raw = qc_directions.get(
        "pooledReferenceCaptures",
        deps.directions.get("pooledReferenceCaptures"),
    )
    if raw is None:
        return None
    if not isinstance(raw, list | tuple) or any(
        not isinstance(value, str) or not value.strip() for value in raw
    ):
        raise ValueError("pooledReferenceCaptures must contain non-empty strings")
    references = tuple(str(value) for value in raw)
    if len(references) < 2 or len(references) != len(set(references)):
        raise ValueError(
            "pooledReferenceCaptures must contain at least two unique captures"
        )
    return references


def _provenance_label(value: Any) -> str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str) and not value.strip():
        return None
    return str(value)


def _ordered_labels(values: np.ndarray, mask: np.ndarray) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values[mask]:
        label = _provenance_label(raw)
        if label is None or label in seen:
            continue
        seen.add(label)
        output.append(label)
    return output


def _capture_design_safety(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
    capture_labels: np.ndarray,
    capture: str,
) -> tuple[list[dict[str, Any]], bool, bool]:
    if characterization is None:
        return [], False, False
    active = np.ones(len(capture_labels), dtype=bool)
    normalized = _validated_sample_labels(
        capture_labels,
        active,
        label_name="physical capture labels",
    )
    encoded = np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in normalized
        ],
        dtype=object,
    )
    after = encoded != capture
    safety: list[dict[str, Any]] = []
    for record in characterization.coefficients:
        coefficient = record.get("name")
        observation = record.get("observationUnit")
        independent = record.get("independentUnit")
        if (
            not isinstance(coefficient, str)
            or not isinstance(observation, str)
            or record.get("scope") != "betweenUnit"
            or coefficient not in deps.cells.columns
            or observation not in deps.cells.columns
        ):
            continue
        condition_values = np.asarray(deps.cells.fetch(coefficient), dtype=object)
        observation_values = np.asarray(deps.cells.fetch(observation), dtype=object)
        if (
            condition_values.shape != after.shape
            or observation_values.shape != after.shape
        ):
            raise ValueError("Capture safety columns do not align with cellSelection")
        required_groups = _ordered_labels(condition_values, active)
        remaining_groups = _ordered_labels(condition_values, after)
        preserves_conditions = set(remaining_groups) == set(required_groups)

        observation_counts: list[dict[str, Any]] = []
        independent_counts: list[dict[str, Any]] = []
        independent_values: np.ndarray | None = None
        if isinstance(independent, str):
            if independent not in deps.cells.columns:
                continue
            independent_values = np.asarray(
                deps.cells.fetch(independent),
                dtype=object,
            )
            if independent_values.shape != after.shape:
                raise ValueError(
                    "Capture independent-unit column does not align with cellSelection"
                )
        for group in required_groups:
            group_mask = np.asarray(
                [_provenance_label(value) == group for value in condition_values],
                dtype=bool,
            )
            observation_levels = set(
                _ordered_labels(observation_values, after & group_mask)
            )
            observation_counts.append(
                {"group": group, "count": len(observation_levels)}
            )
            if independent_values is not None:
                independent_levels = set(
                    _ordered_labels(independent_values, after & group_mask)
                )
                independent_counts.append(
                    {"group": group, "count": len(independent_levels)}
                )

        replication_counts = (
            independent_counts if independent_values is not None else observation_counts
        )
        minimum_units = min(
            (int(item["count"]) for item in replication_counts),
            default=0,
        )
        complete_pairs = 0
        incomplete_pairs = 0
        duplicate_pair_groups = 0
        single_group_pairs = 0
        if independent_values is not None:
            pair_groups: dict[str, dict[str, set[str]]] = {}
            for index in np.flatnonzero(after):
                pair = _provenance_label(independent_values[index])
                pair_group = _provenance_label(condition_values[index])
                observation_value = _provenance_label(observation_values[index])
                if pair is None or pair_group is None or observation_value is None:
                    continue
                pair_groups.setdefault(pair, {}).setdefault(pair_group, set()).add(
                    observation_value
                )
            required_set = set(required_groups)
            for groups in pair_groups.values():
                if len(groups) == 1:
                    single_group_pairs += 1
                duplicate_pair_groups += sum(
                    len(observations) > 1 for observations in groups.values()
                )
                if set(groups) == required_set and all(
                    len(observations) == 1 for observations in groups.values()
                ):
                    complete_pairs += 1
                else:
                    incomplete_pairs += 1
        original_pair_design = dict(record.get("pairedCoverage") or {}).get("design")
        pair_structure_safe = (
            True
            if independent_values is None
            else (
                complete_pairs >= 2
                and incomplete_pairs == 0
                and duplicate_pair_groups == 0
            )
            if original_pair_design == "paired"
            else (len(pair_groups) >= 2 and single_group_pairs == len(pair_groups))
            if original_pair_design == "betweenIndependentUnits"
            else False
        )
        preserves_units = (
            preserves_conditions and minimum_units >= 2 and pair_structure_safe
        )
        safety.append(
            {
                "coefficient": coefficient,
                "conditionColumn": coefficient,
                "observationUnit": observation,
                "independentUnit": independent,
                "requiredGroups": required_groups,
                "remainingGroups": remaining_groups,
                "observationUnitsByGroup": observation_counts,
                "independentUnitsByGroup": independent_counts,
                "minimumIndependentUnitsAfterExclusion": minimum_units,
                "completePairsAfterExclusion": complete_pairs,
                "incompletePairsAfterExclusion": incomplete_pairs,
                "duplicatePairGroupsAfterExclusion": duplicate_pair_groups,
                "independentUnitDesign": original_pair_design,
                "preservesConditionCoverage": preserves_conditions,
                "preservesIndependentUnitCoverage": preserves_units,
            }
        )
    return (
        safety,
        bool(safety) and all(item["preservesConditionCoverage"] for item in safety),
        bool(safety)
        and all(item["preservesIndependentUnitCoverage"] for item in safety),
    )


def _capture_source_missingness(
    sources: Sequence[QcMetricSourceEvidence],
    values_by_source: Mapping[str, np.ndarray],
    capture_labels: np.ndarray | None,
) -> list[QcMetricSourceEvidence]:
    if capture_labels is None:
        return list(sources)
    active = np.ones(len(capture_labels), dtype=bool)
    normalized = _validated_sample_labels(
        capture_labels,
        active,
        label_name="physical capture labels",
    )
    captures: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for raw in normalized:
        value = raw.item() if isinstance(raw, np.generic) else raw
        key = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if key in seen:
            continue
        seen.add(key)
        captures.append((key, normalized == value))
    output: list[QcMetricSourceEvidence] = []
    for source in sources:
        values = values_by_source.get(source.sourceId)
        missing_by_capture: dict[str, int] = {}
        if values is not None:
            for capture, mask in captures:
                missing_by_capture[capture] = int((~np.isfinite(values[mask])).sum())
        elif source.missingCells == source.activeCells:
            missing_by_capture = {
                capture: int(mask.sum()) for capture, mask in captures
            }
        output.append(
            source.model_copy(update={"missingCellsByCapture": missing_by_capture})
        )
    return output


def _capture_failure_models(
    projection: RegisteredQcProjection | AutoFilterProjection,
    *,
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
    capture_labels: np.ndarray | None,
    metric_sources: Sequence[QcMetricSourceEvidence],
) -> list[CaptureFailureEvidence]:
    if capture_labels is None:
        return []
    output: list[CaptureFailureEvidence] = []
    source_by_id = {source.sourceId: source for source in metric_sources}
    for comparison in projection.captureComparisons:
        missing_fractions = {
            source_id: (
                source.missingCellsByCapture.get(comparison.capture, 0)
                / comparison.cells
                if comparison.cells
                else 0.0
            )
            for source_id, source in source_by_id.items()
        }
        safety, condition_safe, unit_safe = _capture_design_safety(
            deps,
            characterization,
            capture_labels,
            comparison.capture,
        )
        failure = CaptureFailureEvidence(
            capture=comparison.capture,
            activeCells=comparison.cells,
            retainedCells=comparison.retainedCells or 0,
            retainedFraction=comparison.retainedFraction or 0.0,
            adverseAxes=list(comparison.adverseAxes),
            independentAdverseAxes=comparison.independentAdverseAxes,
            metricMissingFractions=missing_fractions,
            reasons=list(comparison.reasons),
            wholeCaptureFailure=comparison.wholeCaptureFailure,
            conditionAndUnitSafety=safety,
            preservesConditionCoverage=condition_safe,
            preservesIndependentUnitCoverage=unit_safe,
            exclusionEligible=(
                comparison.wholeCaptureFailure and condition_safe and unit_safe
            ),
            evidenceId=(
                f"qcCapture:{comparison.capture}:"
                f"{comparison.independentAdverseAxes}axes"
            ),
        )
        output.append(failure)
    return output


def _registered_profile_evidence(
    projection: RegisteredQcProjection,
    *,
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
    driver: tuple[str, CellQcDriverType],
    active: np.ndarray,
    values_by_attr: dict[str, np.ndarray],
    metadata_attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    metric_sources: list[QcMetricSourceEvidence],
    source_concordance: list[QcSourceConcordance],
    sample_column: str | None,
    sample_artifact: NamedArtifactSource | None,
    capture_column: str | None,
    capture_artifact: NamedArtifactSource | None,
    capture_labels: np.ndarray | None,
    pooled_reference_captures: tuple[str, ...] | None,
    active_cells: int,
    comparison_source: str | None,
) -> CellQcProfileEvidence:
    attributes = list(metadata_attributes)
    metric_artifacts = list(artifact_metrics)
    profile_id = _registered_qc_profile_id(
        projection.profile,
        driver=driver,
        sample_column=sample_column,
        sample_artifact=sample_artifact,
    )
    n_mads = 3.0 if projection.profile == "captureMad3Sensitivity" else 5.0
    action: CellQcAction = (
        "skip" if projection.profile == "retainWithFlags" else "registeredMad"
    )
    parameters: dict[str, Any] = {
        "policyVersion": 1,
        "profile": projection.profile,
        "nMads": n_mads,
        "boundPolicy": {
            "count": {"remove": "lower", "flag": "upper"},
            "feature": {"remove": "lower", "flag": "upper"},
            "mitochondrial": {"remove": "upper", "fixedCutoff": None},
            "diagnostic": {"remove": "none"},
        },
        "resolvedBounds": [threshold.to_dict() for threshold in projection.thresholds],
        "captureSizes": projection.captureSizes,
        "captureComparisons": [
            comparison.to_dict() for comparison in projection.captureComparisons
        ],
        "captureComparisonSource": comparison_source,
        "pooledReferenceCaptures": list(pooled_reference_captures or ()),
    }
    failure_evidence = _capture_failure_models(
        projection,
        deps=deps,
        characterization=characterization,
        capture_labels=capture_labels,
        metric_sources=metric_sources,
    )
    cells = deps.cells if deps.cells is not None else deps.store.cells
    retention_columns: list[str] = []
    if characterization is not None:
        for coefficient in characterization.coefficients:
            for value in (
                coefficient.get("name"),
                coefficient.get("observationUnit"),
                coefficient.get("independentUnit"),
            ):
                if isinstance(value, str) and value in cells.columns:
                    retention_columns.append(value)
    retained_by_column: dict[str, dict[str, int]] = {}
    unsafe_groups: list[str] = []
    retained = np.asarray(projection.keep, dtype=bool) & np.asarray(active, dtype=bool)
    for column in dict.fromkeys(retention_columns):
        labels = np.asarray(cells.fetch(column))
        if labels.shape != retained.shape:
            raise ValueError(
                f"QC retention column {column!r} does not align with cellSelection"
            )
        counts: dict[str, int] = {}
        for raw_label in np.unique(labels[np.asarray(active, dtype=bool)]):
            label = raw_label.item() if isinstance(raw_label, np.generic) else raw_label
            key = label.decode("utf-8") if isinstance(label, bytes) else str(label)
            count = int((retained & (labels == raw_label)).sum())
            counts[key] = count
            if count == 0:
                unsafe_groups.append(f"{column}={key}")
        retained_by_column[column] = counts
    return CellQcProfileEvidence(
        profileId=profile_id,
        action=action,
        registeredProfile=projection.profile,
        driverAssay=driver[0],
        driverAssayType=driver[1],
        sampleColumn=sample_column,
        sampleArtifact=sample_artifact,
        captureColumn=capture_column,
        captureArtifact=capture_artifact,
        attributes=attributes,
        artifactMetrics=metric_artifacts,
        metricSources=metric_sources,
        sourceConcordance=source_concordance,
        parameters=parameters,
        resolvedBounds=parameters["resolvedBounds"],
        activeCells=active_cells,
        retainedCells=projection.retainedCells,
        retainedFraction=(
            projection.retainedCells / active_cells if active_cells else 0.0
        ),
        activeCellsByCapture=projection.captureSizes,
        sampleRetainedCells=projection.retainedByCapture,
        retainedCellsByColumn=retained_by_column,
        unsafeRetentionGroups=sorted(unsafe_groups),
        flaggedCells=projection.flagCounts,
        metricFlaggedCells=projection.metricFlagCounts,
        failedCaptureCandidates=list(projection.failedCaptureCandidates),
        captureFailureEvidence=failure_evidence,
        excludableCaptureCandidates=[
            item.capture for item in failure_evidence if item.exclusionEligible
        ],
        notes=list(projection.warnings),
        evidenceId=f"qcProfile:{profile_id}",
    )


def _registered_qc_profiles(
    deps: ExperimentalContextDependencies,
    *,
    characterization: CovariateCharacterization | None,
    driver: tuple[str, CellQcDriverType],
    active: np.ndarray,
    values_by_attr: dict[str, np.ndarray],
    metadata_attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    metric_sources: list[QcMetricSourceEvidence],
    source_concordance: list[QcSourceConcordance],
    capture: tuple[str | None, NamedArtifactSource | None, np.ndarray] | None = None,
) -> list[CellQcProfileEvidence]:
    if capture is None:
        capture = _directed_capture_source(deps)
    sample_column: str | None = None
    sample_artifact: NamedArtifactSource | None = None
    capture_labels: np.ndarray | None = None
    if capture is not None:
        sample_column, sample_artifact, capture_labels = capture
    if sample_column is not None:
        comparison_source = f"metadata:{sample_column}"
    elif sample_artifact is not None:
        comparison_source = (
            f"artifact:{sample_artifact.name}:{sample_artifact.artifact.artifactId}"
        )
    else:
        comparison_source = None
    pooled_references = _directed_pooled_reference_captures(deps)
    if pooled_references is not None and capture is None:
        raise ValueError(
            "pooledReferenceCaptures requires an explicit physicalCaptureColumn"
        )
    projections = offered_registered_qc_profiles(
        values_by_metric=values_by_attr,
        active=active,
        capture_labels=capture_labels,
        grouping_proven=capture is not None,
        min_cells_per_capture=20,
        pooled_reference_captures=pooled_references,
    )
    profiles: list[CellQcProfileEvidence] = []
    for projection in projections:
        uses_capture = projection.profile in {
            "captureMad5",
            "captureMad3Sensitivity",
            "pooledReferenceMad5",
        }
        profiles.append(
            _registered_profile_evidence(
                projection,
                deps=deps,
                characterization=characterization,
                driver=driver,
                active=active,
                values_by_attr=values_by_attr,
                metadata_attributes=metadata_attributes,
                artifact_metrics=artifact_metrics,
                metric_sources=metric_sources,
                source_concordance=source_concordance,
                sample_column=sample_column if uses_capture else None,
                sample_artifact=sample_artifact if uses_capture else None,
                capture_column=sample_column,
                capture_artifact=sample_artifact,
                capture_labels=capture_labels,
                pooled_reference_captures=(
                    pooled_references
                    if projection.profile == "pooledReferenceMad5"
                    else None
                ),
                active_cells=int(active.sum()),
                comparison_source=comparison_source,
            )
        )
    return profiles


def _global_qc_profile(
    deps: ExperimentalContextDependencies,
    driver: tuple[str, CellQcDriverType],
    active: np.ndarray,
    active_cells: int,
    values_by_attr: dict[str, np.ndarray],
    metadata_attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    attribute_notes: list[str],
    *,
    characterization: CovariateCharacterization | None = None,
    metric_sources: list[QcMetricSourceEvidence] | None = None,
    source_concordance: list[QcSourceConcordance] | None = None,
    capture: tuple[str | None, NamedArtifactSource | None, np.ndarray] | None = None,
) -> CellQcProfileEvidence | None:
    """Build an execution-exact projection of core global auto-filtering."""
    if not values_by_attr:
        return None
    metric_sources = list(metric_sources or [])
    source_concordance = list(source_concordance or [])
    executable_values: dict[str, np.ndarray] = {}
    for name, values in values_by_attr.items():
        selected = np.asarray(values)[active]
        if selected.size and np.all(selected == selected[0]):
            attribute_notes.append(f"Ignored constant QC metric {name!r}")
            continue
        low, high = gaussian_quantile_bounds(selected, 0.01, 0.99)
        if not np.isfinite([low, high]).all():
            attribute_notes.append(
                f"Ignored QC metric {name!r} with non-finite Gaussian bounds"
            )
            continue
        executable_values[name] = values
    if not executable_values:
        return None
    executable_names = set(executable_values)
    metadata_names = set(metadata_attributes)
    metadata_attributes = [
        name for name in metadata_attributes if name in executable_names
    ]
    artifact_metrics = [
        source
        for source in artifact_metrics
        if qc_metric_execution_name(
            source.name,
            artifact_id=source.artifact.artifactId,
            collides_with_metadata=source.name in metadata_names,
        )
        in executable_names
    ]
    metric_sources = [
        source for source in metric_sources if source.executionName in executable_names
    ]
    retained_source_ids = {source.sourceId for source in metric_sources}
    source_concordance = [
        comparison
        for comparison in source_concordance
        if comparison.leftSourceId in retained_source_ids
        and comparison.rightSourceId in retained_source_ids
    ]
    capture_column: str | None = None
    capture_artifact: NamedArtifactSource | None = None
    capture_labels: np.ndarray | None = None
    if capture is not None:
        capture_column, capture_artifact, capture_labels = capture
    try:
        projection = project_auto_filter_profile(
            "globalGaussian",
            values_by_metric=executable_values,
            active=active,
            sample_labels=capture_labels,
            grouping_proven=capture is not None,
        )
    except ValueError as exc:
        attribute_notes.append(f"Global Gaussian QC is not executable: {exc}")
        return None
    profile_id = _qc_profile_id(
        "globalGaussian",
        driver=driver,
    )
    failures = _capture_failure_models(
        projection,
        deps=deps,
        characterization=characterization,
        capture_labels=capture_labels,
        metric_sources=metric_sources,
    )
    return CellQcProfileEvidence(
        profileId=profile_id,
        action="globalGaussian",
        driverAssay=driver[0],
        driverAssayType=driver[1],
        captureColumn=capture_column,
        captureArtifact=capture_artifact,
        attributes=list(metadata_attributes),
        artifactMetrics=list(artifact_metrics),
        metricSources=metric_sources,
        sourceConcordance=source_concordance,
        parameters=projection.parameters,
        resolvedBounds=cast(dict[str, Any], projection.parameters["resolvedBounds"]),
        activeCells=active_cells,
        retainedCells=projection.retainedCells,
        retainedFraction=projection.retainedCells / active_cells,
        activeCellsByCapture=projection.captureSizes,
        sampleRetainedCells=projection.retainedByCapture,
        flaggedCells=projection.flagCounts,
        metricFlaggedCells=projection.metricFlagCounts,
        failedCaptureCandidates=list(projection.failedCaptureCandidates),
        captureFailureEvidence=failures,
        excludableCaptureCandidates=[
            item.capture for item in failures if item.exclusionEligible
        ],
        notes=[*attribute_notes, *projection.warnings],
        evidenceId=f"qcProfile:{profile_id}",
    )


def _sample_qc_profiles(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None,
    driver: tuple[str, CellQcDriverType],
    active: np.ndarray,
    active_cells: int,
    values_by_attr: dict[str, np.ndarray],
    metadata_attributes: list[str],
    artifact_metrics: list[NamedArtifactSource],
    metric_sources: list[QcMetricSourceEvidence],
    source_concordance: list[QcSourceConcordance],
    capture: tuple[str | None, NamedArtifactSource | None, np.ndarray] | None,
) -> list[CellQcProfileEvidence]:
    """Build core-parity sample MAD profiles from exact grouping sources."""
    attributes = list(values_by_attr)
    profiles: list[CellQcProfileEvidence] = []
    sample_sources: list[
        tuple[str | None, NamedArtifactSource | None, np.ndarray | None, bool]
    ] = []
    if capture is not None:
        sample_sources.append((*capture[:2], capture[2], True))
    sample_sources.extend(
        (None, source, None, False) for source in deps.htoIdentityArtifacts
    )
    sample_sources.extend(
        (column, None, None, False)
        for column in _qc_sample_columns(deps, characterization)
    )
    seen_sources: set[str] = set()
    for (
        sample_column,
        sample_artifact,
        supplied_labels,
        is_physical_capture,
    ) in sample_sources:
        source_key = (
            f"metadata:{sample_column}"
            if sample_column is not None
            else (
                f"artifact:{sample_artifact.artifact.artifactId}"
                if sample_artifact is not None
                else ""
            )
        )
        if not source_key or source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        if len(seen_sources) > _MAX_QC_SAMPLE_PROFILES:
            break
        if not attributes:
            break
        artifact_labels = (
            supplied_labels
            if supplied_labels is not None
            else None
            if sample_artifact is None
            else _resolved_artifact_values(
                deps,
                sample_artifact,
                expected_kind="hto_identity",
            )
        )
        try:
            sample_labels = (
                np.asarray(supplied_labels)
                if supplied_labels is not None
                else np.asarray(deps.cells.fetch(sample_column))
                if sample_column is not None
                else np.asarray(artifact_labels)
            )
            projection = project_auto_filter_profile(
                "sampleMad",
                values_by_metric=values_by_attr,
                sample_labels=sample_labels,
                active=active,
                grouping_proven=True,
                n_mads=3.0,
                min_cells_per_sample=20,
            )
        except (TypeError, ValueError):
            continue
        profile_id = _qc_profile_id(
            "sampleMad",
            driver=driver,
            sample_column=sample_column,
            sample_artifact=sample_artifact,
        )
        failures = (
            _capture_failure_models(
                projection,
                deps=deps,
                characterization=characterization,
                capture_labels=sample_labels,
                metric_sources=metric_sources,
            )
            if is_physical_capture
            else []
        )
        skip_reasons = cast(
            dict[str, object],
            projection.parameters["skipReasons"],
        )
        profiles.append(
            CellQcProfileEvidence(
                profileId=profile_id,
                action="sampleMad",
                driverAssay=driver[0],
                driverAssayType=driver[1],
                sampleColumn=sample_column,
                sampleArtifact=sample_artifact,
                captureColumn=sample_column if is_physical_capture else None,
                captureArtifact=sample_artifact if is_physical_capture else None,
                attributes=list(metadata_attributes),
                artifactMetrics=list(artifact_metrics),
                metricSources=metric_sources,
                sourceConcordance=source_concordance,
                parameters={
                    "nMads": 3.0,
                    "minCellsPerSample": 20,
                    "nSamples": len(projection.captureSizes),
                    "nSkippedSamples": len(skip_reasons),
                },
                resolvedBounds=cast(
                    dict[str, Any],
                    projection.parameters["resolvedBounds"],
                ),
                activeCells=active_cells,
                retainedCells=projection.retainedCells,
                retainedFraction=projection.retainedCells / active_cells,
                activeCellsByCapture=projection.captureSizes,
                sampleRetainedCells=projection.retainedByCapture,
                flaggedCells=projection.flagCounts,
                metricFlaggedCells=projection.metricFlagCounts,
                failedCaptureCandidates=(
                    list(projection.failedCaptureCandidates)
                    if is_physical_capture
                    else []
                ),
                captureFailureEvidence=failures,
                excludableCaptureCandidates=[
                    item.capture for item in failures if item.exclusionEligible
                ],
                notes=list(projection.warnings),
                evidenceId=f"qcProfile:{profile_id}",
            )
        )
    return profiles


def _offered_qc_profiles(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization | None = None,
) -> list[CellQcProfileEvidence]:
    """Project bounded QC profiles against the exact shared cell selection."""
    active_cells = _active_cell_count(deps)
    active = np.ones(active_cells, dtype=bool)
    driver = _qc_driver(deps.store)
    driver_assay = driver[0] if driver is not None else None
    driver_type = driver[1] if driver is not None else None
    skip_id = _qc_profile_id(
        "skip",
        driver=driver,
    )
    skip_notes = (
        []
        if driver is not None
        else ["No RNA or ATAC assay is eligible to drive automatic cell QC"]
    )
    registered_only = deps.directions.get("registeredQcOnly") is True
    profiles = (
        []
        if registered_only
        else [
            CellQcProfileEvidence(
                profileId=skip_id,
                action="skip",
                driverAssay=driver_assay,
                driverAssayType=driver_type,
                activeCells=active_cells,
                retainedCells=active_cells,
                retainedFraction=1.0 if active_cells else 0.0,
                notes=skip_notes,
                evidenceId=f"qcProfile:{skip_id}",
            )
        ]
    )
    if driver is None or active_cells == 0:
        if registered_only:
            profiles.append(
                CellQcProfileEvidence(
                    profileId=skip_id,
                    action="skip",
                    driverAssay=driver_assay,
                    driverAssayType=driver_type,
                    activeCells=active_cells,
                    retainedCells=active_cells,
                    retainedFraction=1.0 if active_cells else 0.0,
                    notes=skip_notes,
                    evidenceId=f"qcProfile:{skip_id}",
                )
            )
        deps.qcProfiles = {profile.profileId: profile for profile in profiles}
        return profiles

    (
        values_by_attr,
        valid_metadata_attributes,
        artifact_metrics,
        metric_sources,
        source_concordance,
        attribute_notes,
        values_by_source,
    ) = _qc_metric_sources(deps, driver)
    capture = _directed_capture_source(deps)
    capture_column: str | None = None
    capture_artifact: NamedArtifactSource | None = None
    capture_labels: np.ndarray | None = None
    capture_sizes: dict[str, int] = {}
    if capture is not None:
        capture_column, capture_artifact, capture_labels = capture
        normalized = _validated_sample_labels(
            capture_labels,
            active,
            label_name="physical capture labels",
        )
        for raw in normalized:
            value = raw.item() if isinstance(raw, np.generic) else raw
            key = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            capture_sizes[key] = capture_sizes.get(key, 0) + 1
    metric_sources = _capture_source_missingness(
        metric_sources,
        values_by_source,
        capture_labels,
    )
    deps.qcMetricSources = metric_sources
    deps.qcSourceConcordance = source_concordance
    if not registered_only:
        profiles = [
            CellQcProfileEvidence(
                profileId=skip_id,
                action="skip",
                driverAssay=driver_assay,
                driverAssayType=driver_type,
                captureColumn=capture_column,
                captureArtifact=capture_artifact,
                metricSources=metric_sources,
                sourceConcordance=source_concordance,
                activeCells=active_cells,
                retainedCells=active_cells,
                retainedFraction=1.0,
                activeCellsByCapture=capture_sizes,
                sampleRetainedCells=capture_sizes,
                notes=[*skip_notes, *attribute_notes],
                evidenceId=f"qcProfile:{skip_id}",
            )
        ]

    if not registered_only:
        global_profile = _global_qc_profile(
            deps,
            driver,
            active,
            active_cells,
            values_by_attr,
            valid_metadata_attributes,
            artifact_metrics,
            attribute_notes,
            characterization=characterization,
            metric_sources=metric_sources,
            source_concordance=source_concordance,
            capture=capture,
        )
        if global_profile is not None:
            profiles.append(global_profile)
        profiles.extend(
            _sample_qc_profiles(
                deps,
                characterization,
                driver,
                active,
                active_cells,
                values_by_attr,
                valid_metadata_attributes,
                artifact_metrics,
                metric_sources,
                source_concordance,
                capture,
            )
        )
    profiles.extend(
        _registered_qc_profiles(
            deps,
            characterization=characterization,
            driver=driver,
            active=active,
            values_by_attr=values_by_attr,
            metadata_attributes=valid_metadata_attributes,
            artifact_metrics=artifact_metrics,
            metric_sources=metric_sources,
            source_concordance=source_concordance,
            capture=capture,
        )
    )

    deps.qcProfiles = {profile.profileId: profile for profile in profiles}
    return profiles


async def inspect_cell_covariates(
    ctx: RunContext[ExperimentalContextDependencies],
) -> CovariateEvidence:
    """Inspect cell metadata without making model-driven choices or writing data."""
    logger.info(
        "Experimental Context covariate inspection started: "
        f"cellSelection={ctx.deps.cellSelection.artifact_id}"
    )
    ctx.deps.htoIdentityColumns = _hto_identity_columns(ctx.deps)
    characterization = characterize_covariates(
        ctx.deps.store,
        cellSelection=ctx.deps.cellSelection,
        studyContext=(
            f"{ctx.deps.studyContext}\nStudy objective: {ctx.deps.studyObjective}"
        ),
        model=None,
        directions=ctx.deps.directions,
        groupingArtifacts=_hto_artifact_map(ctx.deps),
    )
    ctx.deps.characterization = characterization
    qc_profiles = _offered_qc_profiles(ctx.deps)
    contrast_plans = contrast_plans_from_characterization(characterization)
    ctx.deps.contrastPlans = {plan.coefficient: plan for plan in contrast_plans}
    evidence_ids = characterization_evidence(characterization)
    evidence_ids.update(profile.evidenceId for profile in qc_profiles)
    evidence_ids.update(source.sourceId for source in ctx.deps.qcMetricSources)
    evidence_ids.update(item.evidenceId for item in ctx.deps.qcSourceConcordance)
    evidence_ids.update(plan.evidenceId for plan in contrast_plans)
    evidence_ids.update(
        failure.evidenceId
        for profile in qc_profiles
        for failure in profile.captureFailureEvidence
    )
    evidence_ids.update(
        f"htoIdentity:{column}" for column in ctx.deps.htoIdentityColumns
    )
    evidence_ids.update(
        _artifact_evidence_id(source) for source in ctx.deps.htoIdentityArtifacts
    )
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("inspect_cell_covariates")
    logger.info(
        "Experimental Context covariate inspection completed: "
        f"status={characterization.status}, "
        f"columns={len(characterization.columns)}, "
        f"coefficients={len(characterization.coefficients)}, "
        f"qcProfiles={len(qc_profiles)}, "
        f"htoIdentities={len(ctx.deps.htoIdentityColumns)}, "
        f"evidence={len(evidence_ids)}"
    )
    return CovariateEvidence(
        characterization=characterization,
        qcProfiles=qc_profiles,
        qcMetricSources=ctx.deps.qcMetricSources,
        qcSourceConcordance=ctx.deps.qcSourceConcordance,
        contrastPlans=contrast_plans,
        htoIdentityColumns=ctx.deps.htoIdentityColumns,
        htoIdentityArtifacts=ctx.deps.htoIdentityArtifacts,
        evidenceIds=sorted(evidence_ids),
    )


def _batch_safety_evidence(
    deps: ExperimentalContextDependencies,
    characterization: CovariateCharacterization,
    *,
    coefficients: Sequence[str],
    batch_columns: Sequence[str],
) -> list[BatchSafetyEvidence]:
    column_records = {
        record.get("name"): record
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    coefficient_records = {
        record.get("name"): record
        for record in characterization.coefficients
        if isinstance(record.get("name"), str)
    }
    confounding_reports = {
        report.get("coefficient"): report
        for report in characterization.confounding
        if isinstance(report.get("coefficient"), str)
    }
    canonical_batch_columns = sorted(batch_columns)
    batch_safety: list[BatchSafetyEvidence] = []
    for coefficient in coefficients:
        if not canonical_batch_columns:
            break
        coefficient_record = coefficient_records.get(coefficient)
        report = confounding_reports.get(coefficient)
        coefficient_kind = (
            coefficient_record.get("kind") if coefficient_record is not None else None
        )
        if coefficient_kind not in {"categorical", "continuous"}:
            coefficient_kind = None
        observation_unit = (
            report.get("observationUnit")
            if report is not None
            else (
                coefficient_record.get("observationUnit")
                if coefficient_record is not None
                else None
            )
        )
        unit_constant = {
            pair.get("technical")
            for pair in (report.get("pairs", []) if report is not None else [])
            if isinstance(pair.get("technical"), str)
        }
        effective_batch_columns = [
            name for name in canonical_batch_columns if name in unit_constant
        ]
        estimability: dict[str, Any]
        if (
            coefficient_record is None
            or coefficient_record.get("scope") != "betweenUnit"
            or report is None
            or not isinstance(observation_unit, str)
            or coefficient_kind is None
        ):
            estimability = {
                "status": "notComputed",
                "reason": "unresolvedCoefficientDesign",
            }
        else:
            try:
                design = reduce_observation_units(
                    deps.cells,
                    observation_unit,
                    [coefficient, *effective_batch_columns],
                    cell_key="I",
                )
                estimability = coefficient_estimability(
                    design[coefficient].to_numpy(),
                    coefficientKind=coefficient_kind,
                    technicals={
                        name: design[name].to_numpy()
                        for name in effective_batch_columns
                    },
                    technicalKinds={
                        name: column_records[name]["kind"]
                        for name in effective_batch_columns
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug(
                    "Experimental Context batch estimability was not computed: "
                    f"errorType={type(exc).__name__}"
                )
                estimability = {
                    "status": "notComputed",
                    "reason": type(exc).__name__,
                }
        if estimability.get("status") != "ok":
            safety_status: BatchSafetyStatus = "notComputed"
        elif estimability.get("coefficientEstimable") is True and not bool(
            estimability.get("rankDeficient")
        ):
            safety_status = "safe"
        else:
            safety_status = "unsafe"
        batch_token = ",".join(canonical_batch_columns)
        safety = BatchSafetyEvidence(
            coefficient=coefficient,
            coefficientKind=coefficient_kind,
            observationUnit=(
                observation_unit if isinstance(observation_unit, str) else None
            ),
            batchColumns=canonical_batch_columns,
            unitConstantBatchColumns=effective_batch_columns,
            status=safety_status,
            estimability=estimability,
            evidenceId=f"batchEstimability:{coefficient}:{batch_token}",
        )
        batch_safety.append(safety)
        deps.batchSafety[safety.evidenceId] = safety
    return batch_safety


async def analyze_experimental_design(
    ctx: RunContext[ExperimentalContextDependencies],
    column_domains: dict[str, ColumnDomain],
    coefficients_of_interest: list[str],
    units_of_inference: dict[str, InferenceUnit],
    batch_columns: list[str],
) -> CovariateEvidence:
    """Validate proposed domains and inference units and compute confounding.

    Args:
        ctx: Pydantic AI run context containing the existing datastore.
        column_domains: Domain assignment for each metadata column under review.
        coefficients_of_interest: Biological columns representing study contrasts.
        units_of_inference: Observation and independent units for each coefficient.
        batch_columns: Exact technical columns proposed for Harmony evaluation.
    """
    logger.info(
        "Experimental Context design analysis started: "
        f"domains={len(column_domains)}, "
        f"coefficients={len(coefficients_of_interest)}, "
        f"inferenceUnits={len(units_of_inference)}, "
        f"batchColumns={len(batch_columns)}"
    )
    directions = dict(ctx.deps.directions)
    directed_domains = dict(column_domains)
    directed_domains.update(dict(directions.get("columnDomains") or {}))
    directions["columnDomains"] = directed_domains
    directed_coefficients = list(
        dict.fromkeys(
            [
                *coefficients_of_interest,
                *(directions.get("coefficientsOfInterest") or []),
            ]
        )
    )
    directions["coefficientsOfInterest"] = directed_coefficients
    directed_units = {
        name: unit.model_dump(exclude_none=True)
        for name, unit in units_of_inference.items()
    }
    directed_units.update(dict(directions.get("unitsOfInference") or {}))
    directions["unitsOfInference"] = directed_units

    proposed_batch_columns = list(batch_columns)
    directed_batch_columns = directions.get("batchColumns")
    if directed_batch_columns is not None:
        if not isinstance(directed_batch_columns, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in directed_batch_columns
        ):
            raise ModelRetry(
                "directions.batchColumns must be a list of exact metadata columns"
            )
        if len(set(directed_batch_columns)) != len(directed_batch_columns):
            raise ModelRetry("directions.batchColumns must be unique")
        if proposed_batch_columns != directed_batch_columns:
            logger.info(
                "Experimental Context replaced model-proposed batch columns with "
                "the exact directed columns"
            )
        proposed_batch_columns = list(directed_batch_columns)
    canonical_batch_columns = sorted(set(proposed_batch_columns))
    if len(canonical_batch_columns) != len(proposed_batch_columns):
        logger.warning(
            "Experimental Context rejected duplicate proposed batch columns: "
            f"{proposed_batch_columns[:20]}"
        )
        raise ModelRetry("Proposed batch columns must be unique")
    inspected_records = {
        record.get("name"): record
        for record in (
            ctx.deps.characterization.columns
            if ctx.deps.characterization is not None
            else []
        )
        if isinstance(record.get("name"), str)
    }
    if ctx.deps.characterization is not None:
        for batch_column in canonical_batch_columns:
            inspected = inspected_records.get(batch_column)
            if inspected is None:
                logger.warning(
                    "Experimental Context rejected unknown proposed batch column "
                    f"before design recomputation: {batch_column!r}"
                )
                raise ModelRetry(f"Unknown batch column {batch_column!r}")
            proposed_domain = directed_domains.get(
                batch_column,
                inspected.get("domain"),
            )
            if proposed_domain != "technical":
                logger.warning(
                    "Experimental Context rejected proposed batch column before "
                    f"design recomputation: {batch_column!r}, "
                    f"domain={proposed_domain!r}, required='technical'"
                )
                raise ModelRetry(
                    f"Batch column {batch_column!r} must be classified as technical"
                )
            if inspected.get("kind") != "categorical":
                logger.warning(
                    "Experimental Context rejected proposed batch column before "
                    f"design recomputation: {batch_column!r}, "
                    f"kind={inspected.get('kind')!r}, required='categorical'"
                )
                raise ModelRetry(
                    f"Batch column {batch_column!r} must be categorical for Harmony"
                )

    characterization = characterize_covariates(
        ctx.deps.store,
        cellSelection=ctx.deps.cellSelection,
        studyContext=(
            f"{ctx.deps.studyContext}\nStudy objective: {ctx.deps.studyObjective}"
        ),
        model=None,
        directions=directions,
        groupingArtifacts=_hto_artifact_map(ctx.deps),
    )
    if characterization.status == "failed":
        rejection = "; ".join(characterization.notes).strip()
        logger.warning(
            "Experimental Context design characterization rejected the proposed "
            f"directions: {rejection[:1000]}; "
            f"domainColumns={sorted(column_domains)[:50]}, "
            f"coefficients={coefficients_of_interest[:50]}, "
            f"inferenceUnits={sorted(units_of_inference)[:50]}"
        )
        raise ModelRetry("; ".join(characterization.notes))

    # Retain the validated deterministic work even when the proposed Harmony
    # columns below are rejected. A bounded retry or resumed decision can reuse
    # the evidence without rescanning metadata or accepting an unsafe choice.
    ctx.deps.characterization = characterization
    if not ctx.deps.htoIdentityColumns:
        ctx.deps.htoIdentityColumns = _hto_identity_columns(ctx.deps)
    qc_profiles = _offered_qc_profiles(ctx.deps, characterization)
    contrast_plans = contrast_plans_from_characterization(characterization)
    ctx.deps.contrastPlans = {plan.coefficient: plan for plan in contrast_plans}
    evidence_ids = characterization_evidence(characterization)
    evidence_ids.update(profile.evidenceId for profile in qc_profiles)
    evidence_ids.update(source.sourceId for source in ctx.deps.qcMetricSources)
    evidence_ids.update(item.evidenceId for item in ctx.deps.qcSourceConcordance)
    evidence_ids.update(plan.evidenceId for plan in contrast_plans)
    evidence_ids.update(
        failure.evidenceId
        for profile in qc_profiles
        for failure in profile.captureFailureEvidence
    )
    evidence_ids.update(
        f"htoIdentity:{column}" for column in ctx.deps.htoIdentityColumns
    )
    evidence_ids.update(
        _artifact_evidence_id(source) for source in ctx.deps.htoIdentityArtifacts
    )
    ctx.deps.evidenceIds.update(evidence_ids)

    column_records = {
        record.get("name"): record
        for record in characterization.columns
        if isinstance(record.get("name"), str)
    }
    for batch_column in canonical_batch_columns:
        record = column_records.get(batch_column)
        if record is None:
            logger.warning(
                "Experimental Context rejected unknown proposed batch column: "
                f"{batch_column!r}"
            )
            raise ModelRetry(f"Unknown batch column {batch_column!r}")
        if record.get("domain") != "technical":
            logger.warning(
                "Experimental Context rejected proposed batch column "
                f"{batch_column!r}: domain={record.get('domain')!r}, "
                "required='technical'"
            )
            raise ModelRetry(
                f"Batch column {batch_column!r} must be classified as technical"
            )
        if record.get("kind") != "categorical":
            logger.warning(
                "Experimental Context rejected proposed batch column "
                f"{batch_column!r}: kind={record.get('kind')!r}, "
                "required='categorical'"
            )
            raise ModelRetry(
                f"Batch column {batch_column!r} must be categorical for Harmony"
            )

    batch_safety = _batch_safety_evidence(
        ctx.deps,
        characterization,
        coefficients=directed_coefficients,
        batch_columns=canonical_batch_columns,
    )

    evidence_ids.update(item.evidenceId for item in batch_safety)
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("analyze_experimental_design")
    safety_counts = {
        status: sum(item.status == status for item in batch_safety)
        for status in ("safe", "unsafe", "notComputed")
    }
    logger.info(
        "Experimental Context design analysis completed: "
        f"status={characterization.status}, "
        f"batchSafetySafe={safety_counts['safe']}, "
        f"batchSafetyUnsafe={safety_counts['unsafe']}, "
        f"batchSafetyNotComputed={safety_counts['notComputed']}, "
        f"qcProfiles={len(qc_profiles)}, evidence={len(evidence_ids)}"
    )
    return CovariateEvidence(
        characterization=characterization,
        batchSafety=batch_safety,
        qcProfiles=qc_profiles,
        qcMetricSources=ctx.deps.qcMetricSources,
        qcSourceConcordance=ctx.deps.qcSourceConcordance,
        contrastPlans=contrast_plans,
        htoIdentityColumns=ctx.deps.htoIdentityColumns,
        htoIdentityArtifacts=ctx.deps.htoIdentityArtifacts,
        evidenceIds=sorted(evidence_ids),
    )


async def score_current_representation(
    ctx: RunContext[ExperimentalContextDependencies],
    batch_column: str,
    biological_column: str | None = None,
) -> RepresentationEvaluation:
    """Score one explicitly supplied graph without changing datastore state.

    Args:
        ctx: Pydantic AI run context containing the existing datastore.
        batch_column: Categorical technical column used to assess batch mixing.
        biological_column: Optional biological label used to assess preservation.
    """
    logger.info(
        "Experimental Context representation scoring started: "
        f"graphSupplied={ctx.deps.neighbors is not None}, "
        f"biologicalLabelSpecified={biological_column is not None}"
    )
    store = ctx.deps.store
    available_columns = set(store.cells.columns)
    if batch_column not in available_columns:
        raise ModelRetry(f"Unknown batch column {batch_column!r}")
    if biological_column is not None and biological_column not in available_columns:
        raise ModelRetry(f"Unknown biological column {biological_column!r}")
    characterization = ctx.deps.characterization
    if characterization is not None:
        batch_record = next(
            (
                record
                for record in characterization.columns
                if record.get("name") == batch_column
            ),
            None,
        )
        if (
            batch_record is None
            or batch_record.get("domain") != "technical"
            or batch_record.get("kind") != "categorical"
        ):
            raise ModelRetry(
                "Representation scoring requires a characterized categorical "
                "technical batch column"
            )

    neighbors = core_artifact_reference(ctx.deps.neighbors)
    connectivity = core_artifact_reference(ctx.deps.connectivityMap)
    if neighbors is None:
        evaluation = RepresentationEvaluation(
            cellSelection=(
                artifact_reference(ctx.deps.cellSelection)
                if ctx.deps.cellSelection is not None
                else None
            ),
            notes=["No exact neighbors artifact was supplied"],
        )
        ctx.deps.currentRepresentation = evaluation
        ctx.deps.toolCalls.append("score_current_representation")
        logger.info(
            "Experimental Context representation scoring skipped: "
            "no current neighbors artifact"
        )
        return evaluation
    if not isinstance(neighbors, ArtifactRef) or neighbors.kind != "neighbors":
        raise ModelRetry("neighbors must identify an exact neighbors artifact")
    if connectivity is not None and (
        not isinstance(connectivity, ArtifactRef)
        or connectivity.kind not in {"connectivity_map", "integrated_graph"}
    ):
        raise ModelRetry(
            "connectivity_map must identify an exact connectivity graph artifact"
        )

    metrics: dict[str, float] = {}
    notes: list[str] = []
    evidence_ids: list[str] = []
    neighbor_route = f"assay:{neighbors.assay}:neighbors:{neighbors.artifact_id}"
    try:
        value = float(store.metric_ilisi(batch_column, neighbors))
        if math.isfinite(value):
            metrics[f"iLISI:{batch_column}"] = value
            evidence_ids.append(f"metric:iLISI:{batch_column}:{neighbor_route}")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.debug(
            "Experimental Context iLISI scoring was unavailable: "
            f"errorType={type(exc).__name__}"
        )
        notes.append(f"iLISI could not be scored: {exc}")
    try:
        value = float(store.metric_proportional_batch_mixing(batch_column, neighbors))
        if math.isfinite(value):
            metrics[f"proportionalBatchMixing:{batch_column}"] = value
            evidence_ids.append(
                f"metric:proportionalBatchMixing:{batch_column}:{neighbor_route}"
            )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        logger.debug(
            "Experimental Context batch-mixing scoring was unavailable: "
            f"errorType={type(exc).__name__}"
        )
        notes.append(f"Proportional batch mixing could not be scored: {exc}")
    if biological_column is not None:
        try:
            value = float(store.metric_clisi(biological_column, neighbors))
            if math.isfinite(value):
                metrics[f"cLISI:{biological_column}"] = value
                evidence_ids.append(
                    f"metric:cLISI:{biological_column}:{neighbor_route}"
                )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            logger.debug(
                "Experimental Context cLISI scoring was unavailable: "
                f"errorType={type(exc).__name__}"
            )
            notes.append(f"cLISI could not be scored: {exc}")
        if connectivity is not None:
            try:
                value = float(
                    store.metric_graph_connectivity(biological_column, connectivity)
                )
                if math.isfinite(value):
                    metrics[f"graphConnectivity:{biological_column}"] = value
                    evidence_ids.append(
                        "metric:graphConnectivity:"
                        f"{biological_column}:assay:{connectivity.assay}:connectivity:"
                        f"{connectivity.artifact_id}"
                    )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                logger.debug(
                    "Experimental Context connectivity scoring was unavailable: "
                    f"errorType={type(exc).__name__}"
                )
                notes.append(f"Graph connectivity could not be scored: {exc}")

    evaluation = RepresentationEvaluation(
        available=bool(metrics),
        assay=neighbors.assay,
        cellSelection=(
            artifact_reference(ctx.deps.cellSelection)
            if ctx.deps.cellSelection is not None
            else None
        ),
        neighbors=artifact_reference(neighbors),
        connectivityMap=(
            artifact_reference(connectivity) if connectivity is not None else None
        ),
        metrics=metrics,
        notes=notes,
        evidenceIds=evidence_ids,
    )
    ctx.deps.currentRepresentation = evaluation
    ctx.deps.evidenceIds.update(evidence_ids)
    ctx.deps.toolCalls.append("score_current_representation")
    logger.info(
        "Experimental Context representation scoring completed: "
        f"available={evaluation.available}, metrics={len(metrics)}, "
        f"notes={len(notes)}, evidence={len(evidence_ids)}"
    )
    return evaluation


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
