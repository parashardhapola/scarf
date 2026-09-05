"""Serializable contracts for data enrichment."""

from pathlib import Path
from typing import Any, Literal

from ...features.variability import DEFAULT_HVG_BLACKLIST
from .._deps import AGENT_INSTALL_HINT
from ..types import AgentDataModel, AgentRunInfo, StageStatus

try:
    from pydantic import ConfigDict, Field, model_validator
except ImportError as exc:
    raise ImportError(AGENT_INSTALL_HINT) from exc


class DataEnrichmentContext(AgentDataModel):
    """Study evidence that may help resolve organism and feature policy."""

    studyContext: str = ""
    studyObjective: str = ""
    organismHint: str = ""
    tissueReferences: list[str] = Field(default_factory=list)
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalDetails: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "DataEnrichmentContext":
        return cls()

    @classmethod
    def get_example(cls) -> "DataEnrichmentContext":
        return cls(
            studyContext="Single-cell profiling of treated lung tissue",
            studyObjective=(
                "Discover stable populations while preserving treatment effects."
            ),
            organismHint="human",
            tissueReferences=["lung"],
            cellTypeReferences=["alveolar macrophage", "T cell"],
            experimentalDetails=["CRISPR perturbation", "10x 3 prime RNA-seq"],
        )


class StudyContextSummary(AgentDataModel):
    """Verbatim, evidence-backed references extracted from the study context."""

    studyContext: str = ""
    studyObjective: str = ""
    organismReferences: list[str] = Field(default_factory=list)
    tissueReferences: list[str] = Field(default_factory=list)
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalReferences: list[str] = Field(default_factory=list)
    hypothesisReferences: list[str] = Field(default_factory=list)
    analysisIntentReferences: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "StudyContextSummary":
        return cls()

    @classmethod
    def get_example(cls) -> "StudyContextSummary":
        return cls(
            studyContext=(
                "Single-cell profiling of treated human lung tests whether "
                "treatment changes alveolar macrophage states."
            ),
            studyObjective=(
                "Discover populations while preserving the treatment comparison."
            ),
            organismReferences=["human"],
            tissueReferences=["lung"],
            cellTypeReferences=["alveolar macrophage"],
            experimentalReferences=["treated"],
            hypothesisReferences=["treatment changes alveolar macrophage states"],
            analysisIntentReferences=["Single-cell profiling"],
            evidenceIds=["context:study"],
        )


class AdtControlEvidence(AgentDataModel):
    """One exact observed ADT feature carrying an explicit control token."""

    featureId: str
    featureName: str
    matchedToken: Literal["control", "isotype"]
    evidenceId: str

    @classmethod
    def get_blank(cls) -> "AdtControlEvidence":
        return cls(
            featureId="",
            featureName="",
            matchedToken="control",
            evidenceId="",
        )

    @classmethod
    def get_example(cls) -> "AdtControlEvidence":
        return cls(
            featureId="Mouse-IgG1-Control",
            featureName="Mouse IgG1 isotype control",
            matchedToken="isotype",
            evidenceId="assay:ADT:adtControl:Mouse-IgG1-Control",
        )


class HtoTagEvidence(AgentDataModel):
    """One exact feature from an assay persisted with the HTO type."""

    featureId: str
    featureName: str
    evidenceId: str

    @classmethod
    def get_blank(cls) -> "HtoTagEvidence":
        return cls(featureId="", featureName="", evidenceId="")

    @classmethod
    def get_example(cls) -> "HtoTagEvidence":
        return cls(
            featureId="HTO-1",
            featureName="Sample tag 1",
            evidenceId="assay:HTO:htoTag:HTO-1",
        )


class AtacCoordinateEvidence(AgentDataModel):
    """Validation evidence for exact ATAC feature IDs as genomic intervals."""

    status: Literal["notApplicable", "valid", "partial", "invalid"] = "notApplicable"
    coordinateColumn: Literal["ids"] = "ids"
    coordinateFormat: str = "chrom:start-end"
    totalFeatures: int = 0
    validFeatures: int = 0
    invalidExamples: list[str] = Field(default_factory=list)
    validExamples: list[str] = Field(default_factory=list)
    genomeBuild: Literal["unknown"] = "unknown"
    evidenceId: str = ""

    @classmethod
    def get_blank(cls) -> "AtacCoordinateEvidence":
        return cls()

    @classmethod
    def get_example(cls) -> "AtacCoordinateEvidence":
        return cls(
            status="valid",
            totalFeatures=2,
            validFeatures=2,
            validExamples=["chr1:100-200", "chr2:300-450"],
            evidenceId="assay:ATAC:atacCoordinates",
        )


class AssayModalityEvidence(AgentDataModel):
    """Bounded deterministic routing evidence for one persisted assay type."""

    assayType: str = "Assay"
    modality: Literal["RNA", "ATAC", "ADT", "HTO", "unsupported"] = "unsupported"
    typeSource: Literal["persisted", "assayClass", "unknown"] = "unknown"
    graphEligible: bool = False
    markerEligible: bool = False
    demultiplexEligible: bool = False
    adtControls: list[AdtControlEvidence] = Field(default_factory=list)
    htoTags: list[HtoTagEvidence] = Field(default_factory=list)
    atacCoordinates: AtacCoordinateEvidence = Field(
        default_factory=AtacCoordinateEvidence.get_blank
    )
    totalObservedFeatures: int = 0
    reportedFeatures: int = 0
    truncated: bool = False
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayModalityEvidence":
        return cls()

    @classmethod
    def get_example(cls) -> "AssayModalityEvidence":
        control = AdtControlEvidence.get_example()
        return cls(
            assayType="ADT",
            modality="ADT",
            typeSource="persisted",
            graphEligible=True,
            markerEligible=True,
            adtControls=[control],
            totalObservedFeatures=20,
            reportedFeatures=1,
            evidenceIds=["assay:ADT:modality", control.evidenceId],
        )


class FeatureFamilyEvidence(AgentDataModel):
    """One observed feature family from deterministic Scarf analysis."""

    family: str
    species: str = "unknown"
    method: str = ""
    count: int = 0
    examples: list[str] = Field(default_factory=list)
    defaultExclude: bool | None = None
    skipped: str | None = None
    catalogSuspect: str | None = None
    catalogSize: int | None = None
    catalogJoinRate: float | None = None
    catalogJoined: int | None = None
    evidenceId: str

    @classmethod
    def get_blank(cls) -> "FeatureFamilyEvidence":
        return cls(family="", evidenceId="")

    @classmethod
    def get_example(cls) -> "FeatureFamilyEvidence":
        return cls(
            family="mitochondrial",
            species="homo_sapiens",
            method="chromosome",
            count=2,
            examples=["MT-CO1", "MT-CYB"],
            defaultExclude=True,
            evidenceId="assay:RNA:family:mitochondrial",
        )


class DefaultHvgFamilyEvidence(AgentDataModel):
    """One case-insensitive family within Scarf's default HVG blacklist."""

    family: str = ""
    pattern: str = ""
    caseInsensitive: Literal[True] = True
    count: int = 0
    examples: list[str] = Field(default_factory=list)
    evidenceId: str = ""

    @classmethod
    def get_blank(cls) -> "DefaultHvgFamilyEvidence":
        return cls()

    @classmethod
    def get_example(cls) -> "DefaultHvgFamilyEvidence":
        return cls(
            family="mitochondrial",
            pattern="^MT-",
            count=2,
            examples=["MT-CO1", "MT-CYB"],
            evidenceId="assay:RNA:scarfDefaultHvg:family:mitochondrial",
        )


class RnaFeatureInventoryEvidence(AgentDataModel):
    """Exact name-column matches for Scarf's default HVG blacklist."""

    source: Literal["scarfDefaultHvgBlacklist"] = "scarfDefaultHvgBlacklist"
    policyEffect: Literal["evidenceOnly"] = "evidenceOnly"
    featureColumn: Literal["names"] = "names"
    totalFeatures: int = 0
    blacklist: str = ""
    matchCount: int = 0
    examples: list[str] = Field(default_factory=list)
    families: list[DefaultHvgFamilyEvidence] = Field(default_factory=list)
    evidenceId: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "RnaFeatureInventoryEvidence":
        return cls()

    @classmethod
    def get_example(cls) -> "RnaFeatureInventoryEvidence":
        family = DefaultHvgFamilyEvidence.get_example()
        evidence_id = "assay:RNA:scarfDefaultHvg:combined"
        return cls(
            totalFeatures=20_000,
            blacklist=DEFAULT_HVG_BLACKLIST,
            matchCount=2,
            examples=["MT-CO1", "MT-CYB"],
            families=[family],
            evidenceId=evidence_id,
            evidenceIds=[evidence_id, family.evidenceId],
        )


class ExogenousFeatureEvidence(AgentDataModel):
    """One bounded candidate for an artificial or exogenous feature."""

    featureId: str
    featureName: str
    score: int = 0
    classification: str = "unresolved"
    evidenceId: str

    @classmethod
    def get_blank(cls) -> "ExogenousFeatureEvidence":
        return cls(featureId="", featureName="", evidenceId="")

    @classmethod
    def get_example(cls) -> "ExogenousFeatureEvidence":
        return cls(
            featureId="ERCC-00002",
            featureName="ERCC-00002",
            score=4,
            classification="potentialExogenous",
            evidenceId="assay:RNA:exogenous:ERCC-00002",
        )


class AssayFeatureInspection(AgentDataModel):
    """Bounded read-only inspection returned to the model."""

    assay: str
    assayKind: str = ""
    identity: dict[str, Any] = Field(default_factory=dict)
    species: str = "unknown"
    speciesMethod: str | None = None
    speciesReason: str = ""
    families: list[FeatureFamilyEvidence] = Field(default_factory=list)
    defaultFeatureInventory: RnaFeatureInventoryEvidence | None = None
    exogenous: list[ExogenousFeatureEvidence] = Field(default_factory=list)
    modalityEvidence: AssayModalityEvidence = Field(
        default_factory=AssayModalityEvidence.get_blank
    )
    notes: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayFeatureInspection":
        return cls(assay="")

    @classmethod
    def get_example(cls) -> "AssayFeatureInspection":
        family = FeatureFamilyEvidence.get_example()
        default_inventory = RnaFeatureInventoryEvidence.get_example()
        modality = AssayModalityEvidence(
            assayType="RNA",
            modality="RNA",
            typeSource="persisted",
            graphEligible=True,
            markerEligible=True,
            totalObservedFeatures=20_000,
            evidenceIds=["assay:RNA:modality"],
        )
        return cls(
            assay="RNA",
            assayKind="RNAassay",
            identity={"nFeatures": 20_000, "nDuplicateIds": 0},
            species="homo_sapiens",
            speciesMethod="ensemblPrefix",
            speciesReason="Most feature IDs carry the ENSG prefix",
            families=[family],
            defaultFeatureInventory=default_inventory,
            modalityEvidence=modality,
            evidenceIds=[
                "assay:RNA:identity",
                "assay:RNA:species",
                family.evidenceId,
                *default_inventory.evidenceIds,
                *modality.evidenceIds,
            ],
        )


class AssayFeatureInspectionBatch(AgentDataModel):
    """All requested assay inspections returned by one model tool call."""

    inspections: list[AssayFeatureInspection] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "AssayFeatureInspectionBatch":
        return cls()

    @classmethod
    def get_example(cls) -> "AssayFeatureInspectionBatch":
        inspection = AssayFeatureInspection.get_example()
        return cls(
            inspections=[inspection],
            evidenceIds=list(inspection.evidenceIds),
        )


class FeatureReference(AgentDataModel):
    """An exact feature identifier and name observed in one assay."""

    featureId: str
    featureName: str

    @classmethod
    def get_blank(cls) -> "FeatureReference":
        return cls(featureId="", featureName="")

    @classmethod
    def get_example(cls) -> "FeatureReference":
        return cls(featureId="ENSG00000198727", featureName="MT-CYB")


class FeatureMatch(AgentDataModel):
    """Resolution of one proposed feature against an assay."""

    query: str
    status: Literal["present", "ambiguous", "absent"]
    matches: list[FeatureReference] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FeatureMatch":
        return cls(query="", status="absent")

    @classmethod
    def get_example(cls) -> "FeatureMatch":
        return cls(
            query="MT-CYB",
            status="present",
            matches=[FeatureReference.get_example()],
            evidenceIds=["assay:RNA:feature:ENSG00000198727"],
        )


class FeatureLookupResult(AgentDataModel):
    """Bounded result from exact feature lookup."""

    assay: str
    results: list[FeatureMatch] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FeatureLookupResult":
        return cls(assay="")

    @classmethod
    def get_example(cls) -> "FeatureLookupResult":
        match = FeatureMatch.get_example()
        return cls(
            assay="RNA",
            results=[match],
            evidenceIds=list(match.evidenceIds),
        )


class FeatureLookupBatch(AgentDataModel):
    """Exact feature lookups for every requested assay in one tool result."""

    lookups: list[FeatureLookupResult] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "FeatureLookupBatch":
        return cls()

    @classmethod
    def get_example(cls) -> "FeatureLookupBatch":
        lookup = FeatureLookupResult.get_example()
        return cls(lookups=[lookup], evidenceIds=list(lookup.evidenceIds))


class FeatureSelectionPolicy(AgentDataModel):
    """Grounded feature policy proposed for one assay."""

    assay: str
    species: str = "unknown"
    organismName: str = "unknown"
    speciesConfidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    speciesRationale: str = ""
    excludeFamilies: list[str] = Field(default_factory=list)
    protectFamilies: list[str] = Field(default_factory=list)
    excludeFeatures: list[str] = Field(default_factory=list)
    protectFeatures: list[str] = Field(default_factory=list)
    artificialFeatures: list[str] = Field(default_factory=list)
    tissueReferences: list[str] = Field(default_factory=list)
    cellTypeReferences: list[str] = Field(default_factory=list)
    experimentalReferences: list[str] = Field(default_factory=list)
    assayType: str = "Assay"
    assayModality: Literal["RNA", "ATAC", "ADT", "HTO", "unsupported"] = "unsupported"
    graphEligible: bool = False
    markerEligible: bool = False
    demultiplexEligible: bool = False
    exactControlFeatures: list[FeatureReference] = Field(default_factory=list)
    exactTagFeatures: list[FeatureReference] = Field(default_factory=list)
    peakCoordinateStatus: Literal["notApplicable", "valid", "partial", "invalid"] = (
        "notApplicable"
    )
    rationale: str = ""
    evidenceIds: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_non_conflicting_policy(self) -> "FeatureSelectionPolicy":
        family_overlap = set(self.excludeFamilies) & set(self.protectFamilies)
        feature_overlap = set(self.excludeFeatures) & set(self.protectFeatures)
        if family_overlap:
            raise ValueError(
                "feature families cannot be both excluded and protected: "
                f"{sorted(family_overlap)}"
            )
        if feature_overlap:
            raise ValueError(
                "features cannot be both excluded and protected: "
                f"{sorted(feature_overlap)}"
            )
        return self

    @classmethod
    def get_blank(cls) -> "FeatureSelectionPolicy":
        return cls(assay="")

    @classmethod
    def get_example(cls) -> "FeatureSelectionPolicy":
        return cls(
            assay="RNA",
            species="homo_sapiens",
            organismName="human",
            speciesConfidence="high",
            speciesRationale="Gene IDs and study context agree",
            excludeFamilies=["mitochondrial", "ribosomal"],
            protectFamilies=["cellCycle", "sex"],
            artificialFeatures=["ERCC-00002"],
            tissueReferences=["lung"],
            cellTypeReferences=["alveolar macrophage"],
            experimentalReferences=["ERCC spike-in"],
            assayType="RNA",
            assayModality="RNA",
            graphEligible=True,
            markerEligible=True,
            rationale="Use technical families for feature-selection exclusions",
            evidenceIds=["assay:RNA:species", "assay:RNA:family:mitochondrial"],
        )


class DataEnrichmentToolCall(AgentDataModel):
    """Compact audit record for one read-only model tool call."""

    name: str
    assay: str
    evidenceIds: list[str] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "DataEnrichmentToolCall":
        return cls(name="", assay="")

    @classmethod
    def get_example(cls) -> "DataEnrichmentToolCall":
        return cls(
            name="inspect_assay_features",
            assay="RNA",
            evidenceIds=["assay:RNA:identity", "assay:RNA:species"],
        )


class DataEnrichmentReport(AgentDataModel):
    """Final grounded report from :class:`DataEnrichmentAgent`."""

    status: StageStatus
    policies: list[FeatureSelectionPolicy] = Field(default_factory=list)
    inspections: list[AssayFeatureInspection] = Field(default_factory=list)
    studyContextSummary: StudyContextSummary = Field(
        default_factory=StudyContextSummary.get_blank
    )
    unresolvedQuestions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidenceIds: list[str] = Field(default_factory=list)
    toolCalls: list[DataEnrichmentToolCall] = Field(default_factory=list)
    runInfo: AgentRunInfo = Field(default_factory=AgentRunInfo)

    @model_validator(mode="after")
    def validate_status(self) -> "DataEnrichmentReport":
        if self.status == "done" and not self.policies:
            raise ValueError("done reports require at least one feature policy")
        if self.status == "needsInput" and not self.unresolvedQuestions:
            raise ValueError("needsInput reports require an unresolved question")
        if self.status == "failed" and not self.limitations:
            raise ValueError("failed reports require a limitation")
        return self

    @classmethod
    def get_blank(cls) -> "DataEnrichmentReport":
        return cls(status="failed", limitations=["No agent result was produced"])

    @classmethod
    def get_example(cls) -> "DataEnrichmentReport":
        policy = FeatureSelectionPolicy.get_example()
        inspection = AssayFeatureInspection.get_example()
        return cls(
            status="done",
            policies=[policy],
            inspections=[inspection],
            studyContextSummary=StudyContextSummary.get_example(),
            evidenceIds=list(policy.evidenceIds),
            toolCalls=[DataEnrichmentToolCall.get_example()],
            runInfo=AgentRunInfo.get_example(),
        )


class DataEnrichmentDependencies(AgentDataModel):
    """Hidden runtime state supplied to read-only enrichment tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    store: Any = Field(default=None, exclude=True)
    context: DataEnrichmentContext = Field(
        default_factory=DataEnrichmentContext.get_blank
    )
    assays: list[str] = Field(default_factory=list)
    assayTypes: dict[str, str] = Field(default_factory=dict)
    cacheDir: Path | None = None
    allowDownload: bool = False
    evidenceIds: set[str] = Field(default_factory=set)
    inspections: dict[str, AssayFeatureInspection] = Field(default_factory=dict)
    confirmedFeatures: dict[str, set[str]] = Field(default_factory=dict)
    lookupBatch: FeatureLookupBatch | None = Field(default=None, exclude=True)
    lookupQueries: dict[str, list[str]] = Field(default_factory=dict, exclude=True)
    toolCalls: list[DataEnrichmentToolCall] = Field(default_factory=list)

    @classmethod
    def get_blank(cls) -> "DataEnrichmentDependencies":
        return cls()

    @classmethod
    def get_example(cls) -> "DataEnrichmentDependencies":
        return cls(
            context=DataEnrichmentContext.get_example(),
            assays=["RNA"],
            cacheDir=Path("/tmp/scarf-gene-reference"),
            allowDownload=False,
            evidenceIds={"context:organism", "context:tissue:0"},
        )
