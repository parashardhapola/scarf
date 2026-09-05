"""Typed decision contracts and deterministic validation for agent workflows.

The models in this module deliberately separate deterministic option construction
from agent output. An agent may select one offered option and cite observed
evidence, but it cannot add operations or numeric execution parameters.
"""

import hashlib
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .. import record_io
from ..types import AgentDataModel, ArtifactReferenceModel

type DecisionStatus = Literal["apply", "skip", "defer", "abstain"]
type DecisionSource = Literal["rule", "agent", "human"]
type DecisionConfidence = Literal["low", "medium", "high", "notApplicable"]
type EvidenceClass = Literal[
    "geometric",
    "markerCoherence",
    "resamplingStability",
    "crossUnitSupport",
    "protectedVariablePreservation",
    "qualityControl",
    "technical",
    "design",
    "provenance",
    "batchRemoval",
    "biologicalConservation",
    "other",
]
type VerificationStatus = Literal["passed", "failed", "inconclusive"]
type DecisionWorkflowStatus = Literal[
    "running",
    "completed",
    "needsInput",
    "abstained",
    "failed",
]
type ProtectedVariableEffectStatus = Literal[
    "preserved",
    "degraded",
    "improved",
    "notEvaluated",
]

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NON_GEOMETRIC_OVERRIDE_CLASSES: frozenset[EvidenceClass] = frozenset(
    {
        "markerCoherence",
        "resamplingStability",
        "crossUnitSupport",
        "protectedVariablePreservation",
    }
)


def _validate_identifier(value: str, field_name: str) -> str:
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a non-empty stable identifier containing only "
            "letters, digits, '.', '_', ':', '/', or '-'"
        )
    return value


def _validate_unique(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


def _default_decision_sources() -> list[DecisionSource]:
    return ["rule", "agent", "human"]


class DecisionKernelModel(AgentDataModel):
    """Base for immutable, closed decision-kernel contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class DecisionEvidence(DecisionKernelModel):
    """One bounded observed fact available to a decision."""

    evidenceId: str
    evidenceClass: EvidenceClass
    summary: str = Field(min_length=1, max_length=2000)
    artifactReferences: list[ArtifactReferenceModel] = Field(default_factory=list)

    @field_validator("evidenceId")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        return _validate_identifier(value, "evidenceId")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("summary must not contain surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_artifact_references(self) -> "DecisionEvidence":
        identities: list[tuple[str, str | None, str, str]] = []
        for reference in self.artifactReferences:
            if not reference.kind or not reference.artifactId:
                raise ValueError("artifactReferences require kind and artifactId")
            identities.append(
                (
                    reference.scope,
                    reference.assay,
                    reference.kind,
                    reference.artifactId,
                )
            )
        if len(identities) != len(set(identities)):
            raise ValueError("artifactReferences must not contain duplicates")
        return self


class EvidenceBundle(DecisionKernelModel):
    """The complete immutable evidence inventory offered for one decision."""

    formatVersion: Literal[1] = 1
    bundleId: str
    decisionId: str
    evidence: list[DecisionEvidence] = Field(default_factory=list)
    contentSha256: str | None = None

    @field_validator("bundleId", "decisionId")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @field_validator("contentSha256")
    @classmethod
    def validate_content_sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("contentSha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "EvidenceBundle":
        _validate_unique(
            [item.evidenceId for item in self.evidence],
            "EvidenceBundle.evidence IDs",
        )
        if self.contentSha256 is not None:
            payload = self.model_dump(mode="json", exclude={"contentSha256"})
            expected = hashlib.sha256(
                record_io.canonical_json_bytes(payload)
            ).hexdigest()
            if self.contentSha256 != expected:
                raise ValueError("contentSha256 does not match the evidence bundle")
        return self

    def evidence_by_id(self) -> dict[str, DecisionEvidence]:
        """Return evidence indexed by its stable identifier."""
        return {item.evidenceId: item for item in self.evidence}

    def with_content_sha256(self) -> "EvidenceBundle":
        """Return this bundle with its canonical content identity."""
        payload = self.model_dump(mode="json", exclude={"contentSha256"})
        digest = hashlib.sha256(record_io.canonical_json_bytes(payload)).hexdigest()
        return EvidenceBundle.model_validate({**payload, "contentSha256": digest})


class DecisionOption(DecisionKernelModel):
    """One pre-registered option whose execution details live outside the model."""

    optionId: str
    status: DecisionStatus
    label: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    requiredEvidenceClasses: list[EvidenceClass] = Field(default_factory=list)
    requiredEvidenceIds: list[str] = Field(default_factory=list)

    @field_validator("optionId")
    @classmethod
    def validate_option_id(cls, value: str) -> str:
        return _validate_identifier(value, "optionId")

    @field_validator("label", "description")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        if value != value.strip():
            field_name = getattr(info, "field_name", "text")
            raise ValueError(f"{field_name} must not contain surrounding whitespace")
        return value

    @field_validator("requiredEvidenceClasses")
    @classmethod
    def validate_required_classes(
        cls, value: list[EvidenceClass]
    ) -> list[EvidenceClass]:
        if len(value) != len(set(value)):
            raise ValueError("requiredEvidenceClasses must not contain duplicates")
        return value

    @field_validator("requiredEvidenceIds")
    @classmethod
    def validate_required_evidence_ids(cls, value: list[str]) -> list[str]:
        for evidence_id in value:
            _validate_identifier(evidence_id, "requiredEvidenceIds item")
        return _validate_unique(value, "requiredEvidenceIds")


class DecisionSpec(DecisionKernelModel):
    """Authoritative closed option set for one atomic decision."""

    decisionId: str
    definitionVersion: int = Field(ge=1, strict=True)
    checkpoint: str
    question: str = Field(min_length=1, max_length=2000)
    evidenceBundleId: str
    options: list[DecisionOption] = Field(min_length=1)
    baselineOptionId: str | None = None
    metricPreferredOptionId: str | None = None
    requireIndependentOverrideEvidence: bool = Field(default=False, strict=True)
    allowedSources: list[DecisionSource] = Field(
        default_factory=_default_decision_sources
    )

    @field_validator("decisionId", "checkpoint", "evidenceBundleId")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("question must not contain surrounding whitespace")
        return value

    @field_validator("allowedSources")
    @classmethod
    def validate_allowed_sources(
        cls, value: list[DecisionSource]
    ) -> list[DecisionSource]:
        if not value:
            raise ValueError("allowedSources must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("allowedSources must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_option_set(self) -> "DecisionSpec":
        option_ids = [option.optionId for option in self.options]
        _validate_unique(option_ids, "DecisionSpec option IDs")
        if (
            self.baselineOptionId is not None
            and self.baselineOptionId not in option_ids
        ):
            raise ValueError("baselineOptionId must reference an offered option")
        if (
            self.metricPreferredOptionId is not None
            and self.metricPreferredOptionId not in option_ids
        ):
            raise ValueError("metricPreferredOptionId must reference an offered option")
        if (
            self.requireIndependentOverrideEvidence
            and self.metricPreferredOptionId is None
        ):
            raise ValueError(
                "requireIndependentOverrideEvidence requires metricPreferredOptionId"
            )
        return self

    def option_by_id(self) -> dict[str, DecisionOption]:
        """Return offered options indexed by their stable identifier."""
        return {option.optionId: option for option in self.options}


class ProtectedVariableEffect(DecisionKernelModel):
    """Observed effect of a choice on one objective-protected variable."""

    variable: str
    status: ProtectedVariableEffectStatus
    evidenceIds: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1, max_length=1000)

    @field_validator("variable")
    @classmethod
    def validate_variable(cls, value: str) -> str:
        return _validate_identifier(value, "variable")

    @field_validator("evidenceIds")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        for evidence_id in value:
            _validate_identifier(evidence_id, "evidenceIds item")
        return _validate_unique(value, "ProtectedVariableEffect.evidenceIds")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("summary must not contain surrounding whitespace")
        return value


class DecisionSelection(DecisionKernelModel):
    """The bounded choice an agent or human may return."""

    selectedOptionId: str
    evidenceIds: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=4000)
    confidence: DecisionConfidence = "notApplicable"
    protectedVariableEffects: list[ProtectedVariableEffect] = Field(
        default_factory=list
    )
    overrideOfOptionId: str | None = None
    overrideEvidenceIds: list[str] = Field(default_factory=list)

    @field_validator("selectedOptionId", "overrideOfOptionId")
    @classmethod
    def validate_ids(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @field_validator("evidenceIds", "overrideEvidenceIds")
    @classmethod
    def validate_id_lists(cls, value: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "identifiers")
        for item in value:
            _validate_identifier(item, f"{field_name} item")
        return _validate_unique(value, field_name)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("rationale must not contain surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_override(self) -> "DecisionSelection":
        if not set(self.overrideEvidenceIds).issubset(self.evidenceIds):
            raise ValueError("overrideEvidenceIds must be included in evidenceIds")
        if self.overrideOfOptionId is None and self.overrideEvidenceIds:
            raise ValueError("overrideEvidenceIds require overrideOfOptionId")
        if self.overrideOfOptionId == self.selectedOptionId:
            raise ValueError("overrideOfOptionId must differ from selectedOptionId")
        return self


class PendingDecision(DecisionKernelModel):
    """One unresolved checkpoint persisted without fabricating a selection."""

    questionId: str
    decisionId: str
    definitionVersion: int = Field(ge=1, strict=True)
    evidenceBundleId: str
    evidenceBundleSha256: str
    offeredOptionIds: list[str] = Field(min_length=1)
    availableEvidenceIds: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=2000)
    createdAtNs: int = Field(default=0, ge=0, strict=True)

    @field_validator("questionId", "decisionId", "evidenceBundleId")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @field_validator("offeredOptionIds", "availableEvidenceIds")
    @classmethod
    def validate_id_lists(cls, value: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "identifiers")
        for item in value:
            _validate_identifier(item, f"{field_name} item")
        return _validate_unique(value, field_name)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("reason must not contain surrounding whitespace")
        return value

    @field_validator("evidenceBundleSha256")
    @classmethod
    def validate_evidence_bundle_sha256(cls, value: str) -> str:
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("evidenceBundleSha256 must be a lowercase SHA-256 digest")
        return value


class DecisionRecord(DecisionKernelModel):
    """One durable rule, agent, or human selection from an exact option set."""

    recordId: str
    decisionId: str
    definitionVersion: int = Field(ge=1, strict=True)
    evidenceBundleId: str
    evidenceBundleSha256: str
    offeredOptionIds: list[str] = Field(min_length=1)
    availableEvidenceIds: list[str] = Field(default_factory=list)
    selectedOptionId: str
    status: DecisionStatus
    source: DecisionSource
    evidenceIds: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=4000)
    confidence: DecisionConfidence = "notApplicable"
    protectedVariableEffects: list[ProtectedVariableEffect] = Field(
        default_factory=list
    )
    overrideOfOptionId: str | None = None
    overrideEvidenceIds: list[str] = Field(default_factory=list)
    promptSha256: str | None = None
    modelName: str | None = None
    softwareSha256: str | None = None
    verificationId: str | None = None
    supersedes: str | None = None
    createdAtNs: int = Field(default=0, ge=0, strict=True)

    @field_validator(
        "recordId",
        "decisionId",
        "evidenceBundleId",
        "selectedOptionId",
        "verificationId",
        "supersedes",
        "overrideOfOptionId",
    )
    @classmethod
    def validate_ids(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @field_validator(
        "offeredOptionIds",
        "availableEvidenceIds",
        "evidenceIds",
        "overrideEvidenceIds",
    )
    @classmethod
    def validate_id_lists(cls, value: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "identifiers")
        for item in value:
            _validate_identifier(item, f"{field_name} item")
        return _validate_unique(value, field_name)

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("rationale must not contain surrounding whitespace")
        return value

    @field_validator(
        "evidenceBundleSha256",
        "promptSha256",
        "softwareSha256",
    )
    @classmethod
    def validate_sha256(cls, value: str | None, info: object) -> str | None:
        if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
            field_name = getattr(info, "field_name", "digest")
            raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        return value

    @field_validator("modelName")
    @classmethod
    def validate_model_name(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError(
                "modelName must be non-empty without surrounding whitespace"
            )
        return value

    @model_validator(mode="after")
    def validate_references(self) -> "DecisionRecord":
        offered = set(self.offeredOptionIds)
        available = set(self.availableEvidenceIds)
        used = set(self.evidenceIds)
        override = set(self.overrideEvidenceIds)
        if self.selectedOptionId not in offered:
            raise ValueError("selectedOptionId must reference an offered option")
        if not used.issubset(available):
            raise ValueError("evidenceIds must reference only available evidence")
        if not override.issubset(used):
            raise ValueError("overrideEvidenceIds must be included in evidenceIds")
        protected_ids = {
            evidence_id
            for effect in self.protectedVariableEffects
            for evidence_id in effect.evidenceIds
        }
        if not protected_ids.issubset(available):
            raise ValueError(
                "protectedVariableEffects must reference only available evidence"
            )
        if self.overrideOfOptionId is None and self.overrideEvidenceIds:
            raise ValueError("overrideEvidenceIds require overrideOfOptionId")
        if self.overrideOfOptionId is not None:
            if self.overrideOfOptionId not in offered:
                raise ValueError("overrideOfOptionId must reference an offered option")
            if self.overrideOfOptionId == self.selectedOptionId:
                raise ValueError("overrideOfOptionId must differ from selectedOptionId")
        if self.supersedes == self.recordId:
            raise ValueError("A DecisionRecord cannot supersede itself")
        return self


class VerificationCheck(DecisionKernelModel):
    """One deterministic invariant evaluated by the auditor."""

    checkId: str
    status: VerificationStatus
    summary: str = Field(min_length=1, max_length=2000)
    evidenceIds: list[str] = Field(default_factory=list)

    @field_validator("checkId")
    @classmethod
    def validate_check_id(cls, value: str) -> str:
        return _validate_identifier(value, "checkId")

    @field_validator("evidenceIds")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        for evidence_id in value:
            _validate_identifier(evidence_id, "evidenceIds item")
        return _validate_unique(value, "VerificationCheck.evidenceIds")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("summary must not contain surrounding whitespace")
        return value


class VerificationRecord(DecisionKernelModel):
    """Deterministic verification result for exactly one decision record."""

    verificationId: str
    decisionRecordId: str
    status: VerificationStatus
    checks: list[VerificationCheck] = Field(min_length=1)
    createdAtNs: int = Field(default=0, ge=0, strict=True)

    @field_validator("verificationId", "decisionRecordId")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @model_validator(mode="after")
    def validate_aggregate_status(self) -> "VerificationRecord":
        check_ids = [check.checkId for check in self.checks]
        _validate_unique(check_ids, "VerificationRecord check IDs")
        statuses = {check.status for check in self.checks}
        if self.status == "passed" and statuses != {"passed"}:
            raise ValueError("passed verification requires every check to pass")
        if self.status == "failed" and "failed" not in statuses:
            raise ValueError("failed verification requires a failed check")
        if self.status == "inconclusive" and (
            "failed" in statuses or "inconclusive" not in statuses
        ):
            raise ValueError(
                "inconclusive verification requires an inconclusive check and no failures"
            )
        return self


class RevisionRequest(DecisionKernelModel):
    """A bounded request to supersede one decision using exact audit evidence."""

    revisionId: str
    targetDecisionRecordId: str
    verificationId: str
    replacementOptionId: str
    reason: str = Field(min_length=1, max_length=2000)
    evidenceBundleId: str | None = None
    evidenceBundleSha256: str | None = None
    availableEvidenceIds: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    invalidatesDecisionRecordIds: list[str] = Field(default_factory=list)
    createdAtNs: int = Field(default=0, ge=0, strict=True)

    @field_validator(
        "revisionId",
        "targetDecisionRecordId",
        "verificationId",
        "replacementOptionId",
        "evidenceBundleId",
    )
    @classmethod
    def validate_ids(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @field_validator(
        "availableEvidenceIds",
        "evidenceIds",
        "invalidatesDecisionRecordIds",
    )
    @classmethod
    def validate_id_lists(cls, value: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "identifiers")
        for item in value:
            _validate_identifier(item, f"{field_name} item")
        return _validate_unique(value, field_name)

    @field_validator("evidenceBundleSha256")
    @classmethod
    def validate_evidence_bundle_sha256(cls, value: str | None) -> str | None:
        if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("evidenceBundleSha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("reason must not contain surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_evidence_bundle(self) -> "RevisionRequest":
        if (self.evidenceBundleId is None) != (self.evidenceBundleSha256 is None):
            raise ValueError(
                "Revision evidence bundle ID and checksum must be provided together"
            )
        if self.evidenceBundleId is None and (
            self.availableEvidenceIds or self.evidenceIds
        ):
            raise ValueError(
                "Revision evidence inventory requires an exact evidence bundle"
            )
        if not set(self.evidenceIds).issubset(self.availableEvidenceIds):
            raise ValueError(
                "Revision evidence must reference its exact available inventory"
            )
        return self


class DecisionWorkflowRun(DecisionKernelModel):
    """Versioned, acyclic ledger for one bounded decision workflow."""

    recordType: Literal["decisionWorkflowRun"] = "decisionWorkflowRun"
    formatVersion: Literal[2] = 2
    workflowRunId: str
    status: DecisionWorkflowStatus = "running"
    decisionRecords: list[DecisionRecord] = Field(default_factory=list)
    verificationRecords: list[VerificationRecord] = Field(default_factory=list)
    revisionRequests: list[RevisionRequest] = Field(default_factory=list)
    maxRevisions: int = Field(default=2, ge=0, le=2, strict=True)
    pendingDecision: PendingDecision | None = None
    finalHandoffId: str | None = None
    limitations: list[str] = Field(default_factory=list)
    unresolvedClaims: list[str] = Field(default_factory=list)

    @field_validator("workflowRunId", "finalHandoffId")
    @classmethod
    def validate_ids(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "identifier")
        return _validate_identifier(value, field_name)

    @model_validator(mode="after")
    def validate_ledger(self) -> "DecisionWorkflowRun":
        if len(self.revisionRequests) > self.maxRevisions:
            raise ValueError("Decision workflow exceeds its configured revision limit")

        records: dict[str, DecisionRecord] = {}
        record_positions: dict[str, int] = {}
        active_by_decision: dict[str, str] = {}
        for position, record in enumerate(self.decisionRecords):
            if record.recordId in records:
                raise ValueError("decisionRecords must have unique recordId values")
            if record.decisionId in active_by_decision:
                expected_parent = active_by_decision[record.decisionId]
                if record.supersedes != expected_parent:
                    raise ValueError(
                        "Repeated decisions must supersede the current active record"
                    )
            elif record.supersedes is not None:
                raise ValueError(
                    "supersedes must reference an earlier matching decision"
                )
            if record.supersedes is not None:
                parent = records.get(record.supersedes)
                if parent is None or parent.decisionId != record.decisionId:
                    raise ValueError(
                        "supersedes must reference an earlier record for the same decision"
                    )
            records[record.recordId] = record
            record_positions[record.recordId] = position
            active_by_decision[record.decisionId] = record.recordId

        verifications: dict[str, VerificationRecord] = {}
        verified_records: set[str] = set()
        for verification in self.verificationRecords:
            if verification.verificationId in verifications:
                raise ValueError(
                    "verificationRecords must have unique verificationId values"
                )
            if verification.decisionRecordId not in records:
                raise ValueError("Verification must reference an exact decision record")
            if verification.decisionRecordId in verified_records:
                raise ValueError("A decision record may have only one verification")
            linked_record = records[verification.decisionRecordId]
            if linked_record.verificationId != verification.verificationId:
                raise ValueError(
                    "Decision and verification references must agree exactly"
                )
            verifications[verification.verificationId] = verification
            verified_records.add(verification.decisionRecordId)

        revision_ids: set[str] = set()
        revised_targets: set[str] = set()
        revisions_by_target: dict[str, RevisionRequest] = {}
        for revision in self.revisionRequests:
            if revision.revisionId in revision_ids:
                raise ValueError("revisionRequests must have unique revisionId values")
            if revision.targetDecisionRecordId in revised_targets:
                raise ValueError("A decision record may be revised only once")
            target = records.get(revision.targetDecisionRecordId)
            if target is None:
                raise ValueError("Revision must reference an exact decision record")
            revision_verification = verifications.get(revision.verificationId)
            if (
                revision_verification is None
                or revision_verification.decisionRecordId
                != revision.targetDecisionRecordId
            ):
                raise ValueError(
                    "Revision must reference the target decision's verification"
                )
            if revision_verification.status == "passed" and (
                revision.evidenceBundleId is None or not revision.evidenceIds
            ):
                raise ValueError(
                    "Revising a passed decision requires exact downstream evidence"
                )
            if revision.replacementOptionId == target.selectedOptionId:
                raise ValueError("Revision replacement must change the selected option")
            if revision.evidenceBundleId == target.evidenceBundleId and (
                revision.evidenceBundleSha256 != target.evidenceBundleSha256
                or revision.availableEvidenceIds != target.availableEvidenceIds
            ):
                raise ValueError(
                    "Revision evidence must match the target bundle exactly"
                )
            for invalidated_id in revision.invalidatesDecisionRecordIds:
                if invalidated_id not in records:
                    raise ValueError(
                        "Revision invalidation must reference an exact decision record"
                    )
                if (
                    record_positions[invalidated_id]
                    <= record_positions[target.recordId]
                ):
                    raise ValueError(
                        "Revision invalidation may reference only downstream decisions"
                    )
            revision_ids.add(revision.revisionId)
            revised_targets.add(revision.targetDecisionRecordId)
            revisions_by_target[revision.targetDecisionRecordId] = revision

        invalidated_record_ids = {
            record_id
            for revision in self.revisionRequests
            for record_id in revision.invalidatesDecisionRecordIds
        }
        for record in self.decisionRecords:
            if record.supersedes is None:
                continue
            matching_revision = revisions_by_target.get(record.supersedes)
            if (
                matching_revision is None
                and record.supersedes not in invalidated_record_ids
            ):
                raise ValueError("A superseding decision requires a revision request")
            if (
                matching_revision is not None
                and matching_revision.replacementOptionId != record.selectedOptionId
            ):
                raise ValueError(
                    "A superseding decision must select the requested replacement option"
                )

        active_record_ids = set(active_by_decision.values()).difference(
            invalidated_record_ids
        )
        active_records = [records[record_id] for record_id in active_record_ids]
        if self.status == "completed":
            if self.finalHandoffId is None:
                raise ValueError("completed workflows require finalHandoffId")
            if self.pendingDecision is not None:
                raise ValueError(
                    "completed workflows cannot contain a pending decision"
                )
            for record in active_records:
                verification_id = record.verificationId
                active_verification = (
                    verifications.get(verification_id)
                    if verification_id is not None
                    else None
                )
                if record.status in {"defer", "abstain"} or (
                    active_verification is None
                    or active_verification.status != "passed"
                ):
                    raise ValueError(
                        "completed workflows require every active decision to pass"
                    )
        elif self.finalHandoffId is not None:
            raise ValueError("Only completed workflows may reference a final handoff")

        if self.status == "needsInput" and (
            self.pendingDecision is None
            and not any(record.status == "defer" for record in active_records)
        ):
            raise ValueError(
                "needsInput workflows require a pending or active defer decision"
            )
        if self.status != "needsInput" and self.pendingDecision is not None:
            raise ValueError("Only needsInput workflows may contain a pending decision")
        if self.status == "abstained" and not any(
            record.status == "abstain" for record in active_records
        ):
            raise ValueError("abstained workflows require an active abstain decision")
        return self

    def invalidated_decision_record_ids(self) -> set[str]:
        """Return records invalidated by accepted revision requests."""
        return {
            record_id
            for revision in self.revisionRequests
            for record_id in revision.invalidatesDecisionRecordIds
        }

    def active_decision_records(self) -> list[DecisionRecord]:
        """Return active records in their original transition order."""
        superseded = {
            record.supersedes
            for record in self.decisionRecords
            if record.supersedes is not None
        }
        invalidated = self.invalidated_decision_record_ids()
        inactive = superseded | invalidated
        return [
            record for record in self.decisionRecords if record.recordId not in inactive
        ]


class DeterministicDecisionAuditor:
    """Cross-check a decision against its authoritative spec and evidence bundle."""

    @classmethod
    def audit(
        cls,
        spec: DecisionSpec,
        evidence: EvidenceBundle,
        record: DecisionRecord,
        *,
        created_at_ns: int = 0,
    ) -> VerificationRecord:
        """Return a deterministic verification without repairing invalid output."""
        checks: list[VerificationCheck] = []
        evidence_sha256 = (
            evidence.contentSha256
            if evidence.contentSha256 is not None
            else evidence.with_content_sha256().contentSha256
        )

        def add_check(
            check_id: str,
            passed: bool,
            pass_summary: str,
            fail_summary: str,
            evidence_ids: Iterable[str] = (),
        ) -> None:
            checks.append(
                VerificationCheck(
                    checkId=check_id,
                    status="passed" if passed else "failed",
                    summary=pass_summary if passed else fail_summary,
                    evidenceIds=list(evidence_ids),
                )
            )

        add_check(
            "decisionIdentity",
            record.decisionId == spec.decisionId
            and record.definitionVersion == spec.definitionVersion
            and evidence.decisionId == spec.decisionId
            and record.evidenceBundleId == spec.evidenceBundleId
            and evidence.bundleId == spec.evidenceBundleId
            and record.evidenceBundleSha256 == evidence_sha256,
            "Decision, definition, and evidence bundle identities agree.",
            "Decision, definition, or evidence bundle identity does not agree.",
        )

        offered_option_ids = [option.optionId for option in spec.options]
        add_check(
            "exactOptionInventory",
            record.offeredOptionIds == offered_option_ids,
            "The durable record contains the exact offered option inventory.",
            "The durable record does not contain the exact offered option inventory.",
        )

        available_evidence_ids = [item.evidenceId for item in evidence.evidence]
        add_check(
            "exactEvidenceInventory",
            record.availableEvidenceIds == available_evidence_ids,
            "The durable record contains the exact evidence inventory.",
            "The durable record does not contain the exact evidence inventory.",
        )

        option = spec.option_by_id().get(record.selectedOptionId)
        add_check(
            "selectedOption",
            option is not None and option.status == record.status,
            "The selected option and decision status agree.",
            "The selected option is unavailable or its status does not agree.",
        )

        add_check(
            "decisionSource",
            record.source in spec.allowedSources,
            "The decision source is allowed by the definition.",
            "The decision source is not allowed by the definition.",
        )

        evidence_by_id = evidence.evidence_by_id()
        cited_ids = set(record.evidenceIds)
        required_classes = set(option.requiredEvidenceClasses) if option else set()
        required_ids = set(option.requiredEvidenceIds) if option else set()
        cited_classes = {
            evidence_by_id[evidence_id].evidenceClass
            for evidence_id in cited_ids
            if evidence_id in evidence_by_id
        }
        required_evidence_ok = (
            cited_ids.issubset(evidence_by_id)
            and required_ids.issubset(cited_ids)
            and required_classes.issubset(cited_classes)
        )
        add_check(
            "requiredEvidence",
            required_evidence_ok,
            "All cited and required evidence is present.",
            "Cited evidence is unavailable or a required evidence class is missing.",
            record.evidenceIds,
        )

        protected_evidence_ids = {
            evidence_id
            for effect in record.protectedVariableEffects
            for evidence_id in effect.evidenceIds
        }
        protected_ok = not any(
            effect.status == "degraded" for effect in record.protectedVariableEffects
        ) and protected_evidence_ids.issubset(cited_ids)
        add_check(
            "protectedVariablePreservation",
            protected_ok,
            "No cited protected variable is degraded.",
            "A protected variable degraded or its evidence was not cited.",
            sorted(protected_evidence_ids),
        )

        metric_override = (
            spec.requireIndependentOverrideEvidence
            and spec.metricPreferredOptionId is not None
            and record.status in {"apply", "skip"}
            and record.selectedOptionId != spec.metricPreferredOptionId
        )
        if metric_override:
            override_classes = {
                evidence_by_id[evidence_id].evidenceClass
                for evidence_id in record.overrideEvidenceIds
                if evidence_id in evidence_by_id
            }
            qualifying_classes = override_classes & _NON_GEOMETRIC_OVERRIDE_CLASSES
            override_ok = (
                record.overrideOfOptionId == spec.metricPreferredOptionId
                and set(record.overrideEvidenceIds).issubset(record.evidenceIds)
                and (
                    not required_ids
                    or set(record.overrideEvidenceIds).issubset(required_ids)
                )
                and len(qualifying_classes) >= 2
            )
            add_check(
                "independentOverrideEvidence",
                override_ok,
                "The override cites at least two independent non-geometric evidence classes.",
                "The override requires two independent non-geometric evidence classes.",
                record.overrideEvidenceIds,
            )
        else:
            add_check(
                "independentOverrideEvidence",
                record.overrideOfOptionId is None and not record.overrideEvidenceIds,
                "No metric override evidence is required.",
                "Override fields were supplied without an eligible metric override.",
                record.overrideEvidenceIds,
            )

        verification_status: VerificationStatus = (
            "failed" if any(check.status == "failed" for check in checks) else "passed"
        )
        return VerificationRecord(
            verificationId=f"verification:{record.recordId}",
            decisionRecordId=record.recordId,
            status=verification_status,
            checks=checks,
            createdAtNs=created_at_ns,
        )


__all__ = [
    "DecisionConfidence",
    "DecisionEvidence",
    "DecisionOption",
    "DecisionRecord",
    "DecisionSelection",
    "DecisionSource",
    "DecisionSpec",
    "DecisionStatus",
    "DecisionWorkflowRun",
    "DecisionWorkflowStatus",
    "DeterministicDecisionAuditor",
    "EvidenceBundle",
    "EvidenceClass",
    "PendingDecision",
    "ProtectedVariableEffect",
    "ProtectedVariableEffectStatus",
    "RevisionRequest",
    "VerificationCheck",
    "VerificationRecord",
    "VerificationStatus",
]
