"""Serializable contracts for immutable Scarf agent records."""

import re
from pathlib import Path
from typing import Any, Literal

import zarr
from pydantic import Field, field_validator, model_validator

from ...datastore.datastore import DataStore
from ...storage.schema import validate_workspace_name
from ..config import AgentRunConfig
from ..types import (
    AgentDataModel,
    ArtifactReferenceModel,
    ExperimentalBiologyHandoff,
    ExperimentalTuningHandoff,
    TuningBiologyHandoff,
)

type AgentName = Literal[
    "data_enrichment",
    "experimental_context",
    "parameter_tuning",
    "biological_interpretation",
]
type AgentReportType = Literal[
    "",
    "DataEnrichmentReport",
    "ExperimentalContextResult",
    "ParameterTuningReport",
    "BiologicalInterpretationReport",
]
type AgentPersistenceTarget = str | Path | zarr.Group | DataStore
type AgentWorkflowStatus = Literal[
    "running",
    "completed",
    "abstained",
    "failed",
    "abandoned",
]
type AgentTerminalStatus = Literal["completed", "abstained", "failed", "abandoned"]

_FORMAT = "scarf_agent_reports"
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AgentReportLink(AgentDataModel):
    """Immutable identity of one report used as an invocation parent."""

    type: Literal["agentReportLink"] = "agentReportLink"
    workflowRunId: str = ""
    workspace: str | None = None
    agentName: AgentName = "data_enrichment"
    agentRunId: str = ""
    contentSha256: str = ""

    @field_validator("workflowRunId", "agentRunId")
    @classmethod
    def validate_run_ids(cls, value: str) -> str:
        if value:
            _validate_run_id(value, "run ID")
        return value

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, value: str | None) -> str | None:
        validate_workspace_name(value)
        return value

    @field_validator("contentSha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        if value and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("contentSha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_complete_identity(self) -> "AgentReportLink":
        if self.workflowRunId or self.agentRunId or self.contentSha256:
            if not self.workflowRunId or not self.agentRunId or not self.contentSha256:
                raise ValueError("A parent report link requires a complete identity")
        return self

    @classmethod
    def from_reference(cls, reference: "AgentReportReference") -> "AgentReportLink":
        return cls(
            workflowRunId=reference.workflowRunId,
            workspace=reference.workspace,
            agentName=reference.agentName,
            agentRunId=reference.agentRunId,
            contentSha256=reference.contentSha256,
        )

    @classmethod
    def get_blank(cls) -> "AgentReportLink":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentReportLink":
        return cls(
            workflowRunId="workflow-1",
            agentName="experimental_context",
            agentRunId="experimental-run-1",
            contentSha256="0" * 64,
        )


class AgentInvocation(AgentDataModel):
    """Replay-relevant inputs and typed handoffs for one agent invocation."""

    agentName: AgentName = "data_enrichment"
    parentReports: list[AgentReportLink] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactReferenceModel] = Field(default_factory=dict)
    runConfig: AgentRunConfig = Field(default_factory=AgentRunConfig)
    experimentalTuningHandoff: ExperimentalTuningHandoff | None = None
    experimentalBiologyHandoff: ExperimentalBiologyHandoff | None = None
    tuningBiologyHandoff: TuningBiologyHandoff | None = None

    @model_validator(mode="after")
    def validate_parent_reports(self) -> "AgentInvocation":
        identities = [
            (parent.workflowRunId, parent.agentName, parent.agentRunId)
            for parent in self.parentReports
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("parentReports must not contain duplicate reports")
        return self

    @classmethod
    def get_blank(cls) -> "AgentInvocation":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentInvocation":
        cell_selection = ArtifactReferenceModel(
            scope="datastore",
            kind="cell_selection",
            artifactId="c" * 64,
        )
        return cls(
            agentName="parameter_tuning",
            parentReports=[AgentReportLink.get_example()],
            inputs={
                "fromAssay": "RNA",
                "cellSelection": cell_selection.model_dump(mode="json"),
            },
            artifacts={"cellSelection": cell_selection},
            runConfig=AgentRunConfig.get_example(),
            experimentalTuningHandoff=ExperimentalTuningHandoff(
                cellSelection=cell_selection,
                batchAction="skip",
            ),
        )


class AgentReportReference(AgentDataModel):
    """Stable identity for one immutable agent report."""

    type: Literal["agentReport"] = "agentReport"
    workflowRunId: str = ""
    workspace: str | None = None
    agentName: AgentName = "data_enrichment"
    agentRunId: str = ""
    reportType: AgentReportType = ""
    executionRunId: str = ""
    createdAtNs: int = Field(default=0, ge=0, strict=True)
    complete: bool = Field(default=False, strict=True)
    parentReports: list[AgentReportLink] = Field(default_factory=list)
    contentSha256: str = ""

    @field_validator("workflowRunId", "agentRunId")
    @classmethod
    def validate_run_ids(cls, value: str) -> str:
        if value:
            _validate_run_id(value, "run ID")
        return value

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, value: str | None) -> str | None:
        validate_workspace_name(value)
        return value

    @field_validator("contentSha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        if value and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("contentSha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_complete_identity(self) -> "AgentReportReference":
        has_identity = bool(
            self.workflowRunId
            or self.agentRunId
            or self.reportType
            or self.createdAtNs
            or self.complete
            or self.contentSha256
        )
        if has_identity and (
            not self.workflowRunId
            or not self.agentRunId
            or not self.reportType
            or self.createdAtNs < 1
            or not self.complete
            or not self.contentSha256
        ):
            raise ValueError("An agent report reference requires a complete identity")
        return self

    @classmethod
    def get_blank(cls) -> "AgentReportReference":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentReportReference":
        return cls(
            workflowRunId="workflow-1",
            agentName="data_enrichment",
            agentRunId="agent-run-1",
            reportType="DataEnrichmentReport",
            executionRunId="provider-run-1",
            createdAtNs=1,
            complete=True,
            contentSha256="0" * 64,
        )


class AgentReportRecord(AgentDataModel):
    """Complete JSON envelope for one immutable report and its invocation."""

    recordType: Literal["agentReport"] = "agentReport"
    formatVersion: Literal[2] = 2
    reference: AgentReportReference = Field(default_factory=AgentReportReference)
    invocation: AgentInvocation = Field(default_factory=AgentInvocation)
    report: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self) -> "AgentReportRecord":
        if self.reference.agentName != self.invocation.agentName:
            raise ValueError("Report reference and invocation agent names differ")
        if self.reference.parentReports != self.invocation.parentReports:
            raise ValueError("Report reference and invocation parents differ")
        if any(
            parent.workflowRunId == self.reference.workflowRunId
            and parent.agentName == self.reference.agentName
            and parent.agentRunId == self.reference.agentRunId
            for parent in self.invocation.parentReports
        ):
            raise ValueError("An agent report cannot cite itself as a parent")
        return self

    @classmethod
    def get_blank(cls) -> "AgentReportRecord":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentReportRecord":
        from ..data_enrichment.contracts import DataEnrichmentReport

        report = DataEnrichmentReport.get_example()
        return cls(
            reference=AgentReportReference.get_example(),
            invocation=AgentInvocation(agentName="data_enrichment"),
            report=report.model_dump(mode="json"),
        )


class AgentWorkflowRun(AgentDataModel):
    """One dataset-bound workflow and its immutable report records."""

    type: Literal["agentWorkflowRun"] = "agentWorkflowRun"
    formatVersion: Literal[2] = 2
    workflowRunId: str = ""
    workspace: str | None = None
    createdAtNs: int = Field(default=0, ge=0, strict=True)
    finalizedAtNs: int = Field(default=0, ge=0, strict=True)
    status: AgentWorkflowStatus = "running"
    finalizationMessage: str = ""
    analysisStore: str = ""
    datasetFingerprints: dict[str, str] = Field(default_factory=dict)
    reports: list[AgentReportReference] = Field(default_factory=list)

    @field_validator("workflowRunId")
    @classmethod
    def validate_workflow_run_id(cls, value: str) -> str:
        if value:
            _validate_run_id(value, "workflowRunId")
        return value

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, value: str | None) -> str | None:
        validate_workspace_name(value)
        return value

    @field_validator("datasetFingerprints")
    @classmethod
    def validate_dataset_fingerprints(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not assay or not fingerprint for assay, fingerprint in value.items()):
            raise ValueError("Dataset fingerprint names and values must be non-empty")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "AgentWorkflowRun":
        if self.workflowRunId and self.createdAtNs < 1:
            raise ValueError("A workflow requires a positive createdAtNs")
        if self.workflowRunId and not self.datasetFingerprints:
            raise ValueError("A workflow requires exact dataset fingerprints")
        if self.status == "running" and self.finalizedAtNs != 0:
            raise ValueError("A running workflow cannot have finalizedAtNs")
        if self.status == "running" and self.finalizationMessage:
            raise ValueError("A running workflow cannot have a finalizationMessage")
        if self.status != "running" and self.finalizedAtNs < 1:
            raise ValueError("A terminal workflow requires finalizedAtNs")
        if (
            self.status != "running"
            and self.createdAtNs
            and self.finalizedAtNs < self.createdAtNs
        ):
            raise ValueError("finalizedAtNs cannot precede createdAtNs")
        return self

    @classmethod
    def get_blank(cls) -> "AgentWorkflowRun":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentWorkflowRun":
        return cls(
            workflowRunId="workflow-1",
            createdAtNs=1,
            analysisStore="analysis.zarr",
            datasetFingerprints={"RNA": "dataset-1"},
            reports=[AgentReportReference.get_example()],
        )


class AgentStoreManifest(AgentDataModel):
    """Identity document for one workspace-local agent JSON store."""

    type: Literal["agentReportStore"] = "agentReportStore"
    format: Literal["scarf_agent_reports"] = "scarf_agent_reports"
    formatVersion: Literal[2] = 2
    workspace: str | None = None

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, value: str | None) -> str | None:
        validate_workspace_name(value)
        return value

    @classmethod
    def get_blank(cls) -> "AgentStoreManifest":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentStoreManifest":
        return cls(workspace="analysis")


class AgentWorkflowFinalization(AgentDataModel):
    """Immutable terminal event for a workflow."""

    recordType: Literal["agentWorkflowFinalization"] = "agentWorkflowFinalization"
    formatVersion: Literal[2] = 2
    workflowRunId: str = ""
    workspace: str | None = None
    status: AgentTerminalStatus = "completed"
    finalizedAtNs: int = Field(default=0, ge=0, strict=True)
    message: str = ""

    @field_validator("workflowRunId")
    @classmethod
    def validate_workflow_run_id(cls, value: str) -> str:
        if value:
            _validate_run_id(value, "workflowRunId")
        return value

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, value: str | None) -> str | None:
        validate_workspace_name(value)
        return value

    @model_validator(mode="after")
    def validate_finalization(self) -> "AgentWorkflowFinalization":
        if self.workflowRunId and self.finalizedAtNs < 1:
            raise ValueError("A finalization requires a positive finalizedAtNs")
        return self

    @classmethod
    def get_blank(cls) -> "AgentWorkflowFinalization":
        return cls()

    @classmethod
    def get_example(cls) -> "AgentWorkflowFinalization":
        return cls(
            workflowRunId="workflow-1",
            status="completed",
            finalizedAtNs=2,
        )


def _validate_run_id(value: str, label: str) -> str:
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be one safe path component containing 1-128 ASCII "
            "lowercase letters, numbers, underscores, or hyphens"
        )
    return value
