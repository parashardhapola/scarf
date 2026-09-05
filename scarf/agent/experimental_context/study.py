"""Grounded study contracts for decision-driven RNA analysis."""

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import Field, model_validator

from ..types import AgentDataModel

type AuthorLabelPolicy = Literal["holdout", "preservation"]
type ProcessingGoal = Literal[
    "populationDiscovery",
    "conditionPreservingDiscovery",
]
type CorrectionLicense = Literal[
    "safe",
    "unsafeConfounded",
    "indeterminate",
    "notApplicable",
]


class StudyContract(AgentDataModel):
    """Scientific authority and design constraints for one analysis."""

    studyContext: str = ""
    studyObjective: str = ""
    processingGoal: ProcessingGoal = "populationDiscovery"
    scientificQuestions: list[str] = Field(default_factory=list)
    targetCohort: list[str] = Field(default_factory=list)
    physicalCaptureColumn: str | None = None
    independentUnitColumns: list[str] = Field(default_factory=list)
    conditionColumns: list[str] = Field(default_factory=list)
    technicalBatchColumns: list[str] = Field(default_factory=list)
    protectedColumns: list[str] = Field(default_factory=list)
    authorLabelPolicy: AuthorLabelPolicy = "holdout"
    correctionLicense: CorrectionLicense = "notApplicable"
    allowedClaims: list[str] = Field(default_factory=list)
    unsupportedClaims: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self) -> "StudyContract":
        if not self.studyContext.strip():
            raise ValueError("studyContext must be non-empty")
        if not self.studyObjective.strip():
            raise ValueError("studyObjective must be non-empty")
        for field_name in (
            "independentUnitColumns",
            "conditionColumns",
            "technicalBatchColumns",
            "protectedColumns",
            "evidenceIds",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if self.physicalCaptureColumn in self.conditionColumns:
            raise ValueError("A condition column cannot be the physical capture")
        if set(self.technicalBatchColumns).intersection(self.conditionColumns):
            raise ValueError("Technical batch columns cannot also be condition columns")
        if self.correctionLicense == "safe" and not self.technicalBatchColumns:
            raise ValueError("A safe correction license requires batch columns")
        return self

    @classmethod
    def get_blank(cls) -> "StudyContract":
        return cls(studyContext="Study context", studyObjective="Study objective")

    @classmethod
    def get_example(cls) -> "StudyContract":
        return cls(
            studyContext="Treated and control blood samples from multiple donors.",
            studyObjective=(
                "Discover stable populations while preserving treatment-associated "
                "structure."
            ),
            processingGoal="conditionPreservingDiscovery",
            scientificQuestions=[
                "Discover stable populations while preserving treatment-associated "
                "structure."
            ],
            physicalCaptureColumn="sample",
            independentUnitColumns=["donor"],
            conditionColumns=["treatment"],
            technicalBatchColumns=["batch"],
            protectedColumns=["treatment", "donor"],
            correctionLicense="safe",
            allowedClaims=["Describe reproducible population structure."],
            unsupportedClaims=[
                "This workflow does not test differential-expression hypotheses."
            ],
            evidenceIds=["column:batch", "column:donor", "column:treatment"],
        )


def _unique(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def build_study_contract(
    *,
    study_context: str,
    study_objective: str,
    experimental_result: Any,
    author_label_policy: AuthorLabelPolicy = "holdout",
    physical_capture_column: str | None = None,
) -> StudyContract:
    """Build a strict contract from one validated Experimental Context result."""

    if getattr(experimental_result, "status", None) != "done":
        raise ValueError("Experimental Context must be done before contract creation")
    decision = experimental_result.decision
    batch_plan = decision.batchCorrection
    batch_safety = list(experimental_result.batchSafety)
    conditions = list(decision.coefficientsOfInterest)
    independent_units = _unique(
        unit.independentUnit for unit in decision.unitsOfInference.values()
    )
    protected = _unique([*conditions, *independent_units, *batch_plan.preserveColumns])
    assessed_batch_columns = _unique(
        [
            *batch_plan.batchColumns,
            *(
                column
                for assessment in batch_safety
                for column in assessment.batchColumns
            ),
        ]
    )
    correction_license: CorrectionLicense
    if any(assessment.status == "unsafe" for assessment in batch_safety):
        correction_license = "unsafeConfounded"
    elif batch_plan.action == "evaluateHarmony":
        correction_license = "safe"
    elif batch_plan.action == "unsafe":
        correction_license = "unsafeConfounded"
    elif batch_plan.action == "needsInput":
        correction_license = "indeterminate"
    elif assessed_batch_columns and all(
        assessment.status == "safe" for assessment in batch_safety
    ):
        correction_license = "safe"
    else:
        correction_license = "notApplicable"
    processing_goal: ProcessingGoal = (
        "conditionPreservingDiscovery" if conditions else "populationDiscovery"
    )
    evidence_ids = _unique(
        [
            *decision.evidenceIds,
            *batch_plan.evidenceIds,
            *(item.evidenceId for item in experimental_result.batchSafety),
        ]
    )
    limitations = list(experimental_result.notes)
    if physical_capture_column is None:
        limitations.append(
            "Physical capture identity is unresolved; capture-aware doublet removal "
            "is not authorized."
        )
    if author_label_policy == "preservation":
        limitations.append(
            "Author labels were available for preservation checks; this run is "
            "ineligible for label-based benchmark scoring."
        )
    return StudyContract(
        studyContext=study_context,
        studyObjective=study_objective,
        processingGoal=processing_goal,
        scientificQuestions=[study_objective],
        physicalCaptureColumn=physical_capture_column,
        independentUnitColumns=independent_units,
        conditionColumns=conditions,
        technicalBatchColumns=assessed_batch_columns,
        protectedColumns=protected,
        authorLabelPolicy=author_label_policy,
        correctionLicense=correction_license,
        allowedClaims=[
            "Describe population structure supported by the selected representation.",
            "Compare preprocessing alternatives against explicit evidence.",
        ],
        unsupportedClaims=[
            "This workflow does not test differential-expression hypotheses.",
            "Cells are not independent biological replicates.",
        ],
        evidenceIds=evidence_ids,
        limitations=limitations,
    )


__all__ = [
    "AuthorLabelPolicy",
    "ProcessingGoal",
    "StudyContract",
    "build_study_contract",
]
