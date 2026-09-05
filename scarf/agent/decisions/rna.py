"""Deterministic RNA decision definitions and executor-owned option payloads."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from ..types import AgentDataModel
from .kernel import (
    DecisionOption,
    DecisionRecord,
    DecisionSpec,
    DecisionStatus,
    DeterministicDecisionAuditor,
    EvidenceBundle,
    VerificationRecord,
)

type RnaDecisionCheckpoint = Literal[
    "qcGrouping",
    "cellQuality",
    "featurePolicy",
    "hvgRanking",
    "hvgCount",
    "pcaPrefix",
    "correctionLicense",
    "correctionNeed",
    "correctionOutcome",
    "graphK",
    "clusterPartition",
]
type RnaWorkflowNode = Literal[
    "qcGrouping",
    "cellQuality",
    "featurePolicy",
    "hvgRanking",
    "hvgCount",
    "pcaPrefix",
    "correctionLicense",
    "correctionNeed",
    "correctionOutcome",
    "graphK",
    "clusterPartition",
    "finalize",
]
type DecisionTerminalStatus = Literal["needsInput", "abstained"]
type QcGroupingMode = Literal["global", "physicalCapture", "pooledReference"]
type HvgRankingMode = Literal["global", "batchAware"]
type CellQualityProfile = Literal[
    "retainWithFlags",
    "globalMad5",
    "captureMad5",
    "captureMad3Sensitivity",
    "pooledReferenceMad5",
]
type ConditionalGeneFamily = Literal[
    "mitochondrial",
    "ribosomal",
    "mitoribosomal",
    "histone",
    "hla",
    "h2",
    "hemoglobin",
    "immuneReceptor",
    "cellCycle",
    "stress",
    "dissociation",
    "sexLinked",
]
type CorrectionLicense = Literal[
    "safe",
    "unsafeConfounded",
    "indeterminate",
    "notApplicable",
]
type CorrectionNeed = Literal["needed", "notNeeded", "indeterminate"]

_CHECKPOINT_ORDER: tuple[RnaWorkflowNode, ...] = (
    "qcGrouping",
    "cellQuality",
    "featurePolicy",
    "hvgRanking",
    "hvgCount",
    "pcaPrefix",
    "correctionLicense",
    "correctionNeed",
    "correctionOutcome",
    "graphK",
    "clusterPartition",
    "finalize",
)


class RnaDecisionGateError(ValueError):
    """Raised when deterministic evidence forbids constructing an option set."""


class RnaDecisionCompilationError(ValueError):
    """Raised when an unverified decision cannot compile to an executor payload."""


class RnaRegistryModel(AgentDataModel):
    """Base for immutable, closed RNA registry contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class CellQualityExecutorPayload(RnaRegistryModel):
    """Exact cell-quality thresholds owned by the deterministic executor."""

    operation: Literal["cellQualityProfile"] = "cellQualityProfile"
    profile: CellQualityProfile
    lowerCountMad: float | None = Field(default=None, ge=0, strict=True)
    lowerFeatureMad: float | None = Field(default=None, ge=0, strict=True)
    upperMitoMad: float | None = Field(default=None, ge=0, strict=True)
    groupByCapture: bool = Field(strict=True)
    pooledReference: bool = Field(strict=True)
    sensitivityOnly: bool = Field(strict=True)
    flagHighCounts: Literal[True] = True
    flagHighFeatures: Literal[True] = True

    @model_validator(mode="after")
    def validate_profile(self) -> "CellQualityExecutorPayload":
        thresholds = (self.lowerCountMad, self.lowerFeatureMad, self.upperMitoMad)
        if self.profile == "retainWithFlags":
            if any(value is not None for value in thresholds):
                raise ValueError("retainWithFlags cannot define removal thresholds")
            if self.groupByCapture or self.pooledReference or self.sensitivityOnly:
                raise ValueError("retainWithFlags cannot enable filtering modes")
            return self
        if any(value is None for value in thresholds):
            raise ValueError("Filtering profiles require all three MAD thresholds")
        if self.profile.startswith("captureMad") != self.groupByCapture:
            raise ValueError("Capture profiles and groupByCapture must agree")
        if (self.profile == "pooledReferenceMad5") != self.pooledReference:
            raise ValueError("pooledReferenceMad5 and pooledReference must agree")
        if (self.profile == "captureMad3Sensitivity") != self.sensitivityOnly:
            raise ValueError("captureMad3Sensitivity and sensitivityOnly must agree")
        return self


class QcGroupingExecutorPayload(RnaRegistryModel):
    """Exact population used to estimate registered cell-quality thresholds."""

    operation: Literal["qcGrouping"] = "qcGrouping"
    groupingMode: QcGroupingMode


class HvgRankingExecutorPayload(RnaRegistryModel):
    """Exact variability-ranking route used to construct HVG candidates."""

    operation: Literal["hvgRanking"] = "hvgRanking"
    rankingMode: HvgRankingMode


class HvgExecutorPayload(RnaRegistryModel):
    """Exact HVG count and ranking mode owned by the executor."""

    operation: Literal["hvgSelection"] = "hvgSelection"
    topN: int = Field(ge=1, strict=True)
    rankingMode: HvgRankingMode


class FeaturePolicyExecutorPayload(RnaRegistryModel):
    """Exact conditional family policy for representation features."""

    operation: Literal["featurePolicy"] = "featurePolicy"
    policy: Literal[
        "keepAll",
        "excludeScarfDefaults",
        "excludeEligibleBundle",
    ]
    excludedFamilies: list[ConditionalGeneFamily] = Field(default_factory=list)
    useScarfDefaultBlacklist: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_policy(self) -> "FeaturePolicyExecutorPayload":
        if len(self.excludedFamilies) != len(set(self.excludedFamilies)):
            raise ValueError("excludedFamilies must not contain duplicates")
        if self.policy == "keepAll" and (
            self.excludedFamilies or self.useScarfDefaultBlacklist
        ):
            raise ValueError("keepAll cannot exclude gene families")
        if self.policy == "excludeScarfDefaults":
            if self.excludedFamilies or not self.useScarfDefaultBlacklist:
                raise ValueError(
                    "excludeScarfDefaults requires only the Scarf default blacklist"
                )
        if self.policy == "excludeEligibleBundle" and not self.excludedFamilies:
            raise ValueError("excludeEligibleBundle requires gene families")
        if self.policy == "excludeEligibleBundle" and self.useScarfDefaultBlacklist:
            raise ValueError(
                "excludeEligibleBundle cannot silently add the Scarf defaults"
            )
        return self


class PcaPrefixExecutorPayload(RnaRegistryModel):
    """Exact PCA prefix owned by the executor."""

    operation: Literal["pcaPrefix"] = "pcaPrefix"
    dimensions: int = Field(ge=2, le=50, strict=True)


class CorrectionLicensePayload(RnaRegistryModel):
    """Deterministic correction authorization derived from design evidence."""

    operation: Literal["correctionLicense"] = "correctionLicense"
    license: CorrectionLicense


class CorrectionNeedPayload(RnaRegistryModel):
    """Observed need for correction in an uncorrected representation."""

    operation: Literal["correctionNeed"] = "correctionNeed"
    need: CorrectionNeed


class CorrectionOutcomeExecutorPayload(RnaRegistryModel):
    """Exact native or Harmony representation route."""

    operation: Literal["correctionOutcome"] = "correctionOutcome"
    outcome: Literal["retainNative", "acceptHarmony"]
    useHarmony: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_outcome(self) -> "CorrectionOutcomeExecutorPayload":
        if (self.outcome == "acceptHarmony") != self.useHarmony:
            raise ValueError("acceptHarmony and useHarmony must agree")
        return self


class GraphExecutorPayload(RnaRegistryModel):
    """Exact graph neighborhood size owned by the executor."""

    operation: Literal["graphK"] = "graphK"
    neighborsK: int = Field(ge=2, le=41, strict=True)


class ClusterExecutorPayload(RnaRegistryModel):
    """Exact Leiden resolution owned by the executor."""

    operation: Literal["clusterResolution"] = "clusterResolution"
    leidenResolution: float = Field(gt=0, le=1.5, strict=True)


class NoExecutionPayload(RnaRegistryModel):
    """Typed terminal or pause outcome with no analytical operation."""

    operation: Literal["noExecution"] = "noExecution"
    reasonCode: Literal[
        "needsInput",
        "scientificAbstention",
    ]


type RnaOptionPayload = Annotated[
    QcGroupingExecutorPayload
    | CellQualityExecutorPayload
    | HvgRankingExecutorPayload
    | HvgExecutorPayload
    | FeaturePolicyExecutorPayload
    | PcaPrefixExecutorPayload
    | CorrectionLicensePayload
    | CorrectionNeedPayload
    | CorrectionOutcomeExecutorPayload
    | GraphExecutorPayload
    | ClusterExecutorPayload
    | NoExecutionPayload,
    Field(discriminator="operation"),
]


class RnaExecutorOption(RnaRegistryModel):
    """Executor-only payload keyed by the semantic option shown to an agent."""

    checkpoint: RnaDecisionCheckpoint
    optionId: str
    payload: RnaOptionPayload

    @field_validator("optionId")
    @classmethod
    def validate_option_id(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError(
                "optionId must be non-empty without surrounding whitespace"
            )
        return value


class RnaDecisionDefinition(RnaRegistryModel):
    """Agent-visible decision spec paired with executor-only payloads."""

    checkpoint: RnaDecisionCheckpoint
    spec: DecisionSpec
    executorOptions: list[RnaExecutorOption] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_exact_registry(self) -> "RnaDecisionDefinition":
        if self.spec.checkpoint != self.checkpoint:
            raise ValueError(
                "DecisionSpec checkpoint must match the registry checkpoint"
            )
        visible_ids = [option.optionId for option in self.spec.options]
        executor_ids = [option.optionId for option in self.executorOptions]
        if len(executor_ids) != len(set(executor_ids)):
            raise ValueError("executorOptions must not contain duplicate option IDs")
        if executor_ids != visible_ids:
            raise ValueError(
                "executorOptions must exactly match the ordered visible option IDs"
            )
        if any(option.checkpoint != self.checkpoint for option in self.executorOptions):
            raise ValueError("Every executor option must match the registry checkpoint")
        return self

    def executor_option(self, option_id: str) -> RnaExecutorOption:
        """Return one exact registered executor option."""
        for option in self.executorOptions:
            if option.optionId == option_id:
                return option
        raise KeyError(f"Unknown option ID for {self.spec.decisionId}: {option_id}")


class RnaDecisionTransition(RnaRegistryModel):
    """One option-status transition in the fixed acyclic RNA graph."""

    fromCheckpoint: RnaDecisionCheckpoint
    onStatus: DecisionStatus
    toCheckpoint: RnaWorkflowNode | None = None
    terminalStatus: DecisionTerminalStatus | None = None

    @model_validator(mode="after")
    def validate_destination(self) -> "RnaDecisionTransition":
        if (self.toCheckpoint is None) == (self.terminalStatus is None):
            raise ValueError(
                "A transition requires exactly one checkpoint or terminal destination"
            )
        return self


class RnaDecisionTransitionGraph(RnaRegistryModel):
    """Ordered graph that rejects cycles and ambiguous transitions."""

    orderedNodes: tuple[RnaWorkflowNode, ...] = _CHECKPOINT_ORDER
    transitions: list[RnaDecisionTransition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "RnaDecisionTransitionGraph":
        if self.orderedNodes != _CHECKPOINT_ORDER:
            raise ValueError("orderedNodes must use the v1 RNA checkpoint order")
        positions = {node: position for position, node in enumerate(self.orderedNodes)}
        triggers: set[tuple[RnaDecisionCheckpoint, DecisionStatus]] = set()
        for transition in self.transitions:
            trigger = (transition.fromCheckpoint, transition.onStatus)
            if trigger in triggers:
                raise ValueError(
                    "Transitions must have unique checkpoint/status triggers"
                )
            triggers.add(trigger)
            if transition.toCheckpoint is not None and (
                positions[transition.toCheckpoint]
                <= positions[transition.fromCheckpoint]
            ):
                raise ValueError("RNA decision transitions must point strictly forward")
        return self

    def resolve(
        self, checkpoint: RnaDecisionCheckpoint, status: DecisionStatus
    ) -> tuple[RnaWorkflowNode | None, DecisionTerminalStatus | None]:
        """Resolve one exact checkpoint/status transition."""
        for transition in self.transitions:
            if (
                transition.fromCheckpoint == checkpoint
                and transition.onStatus == status
            ):
                return transition.toCheckpoint, transition.terminalStatus
        raise KeyError(f"No RNA transition for {checkpoint}/{status}")


def _transition(
    checkpoint: RnaDecisionCheckpoint,
    status: DecisionStatus,
    *,
    to: RnaWorkflowNode | None = None,
    terminal: DecisionTerminalStatus | None = None,
) -> RnaDecisionTransition:
    return RnaDecisionTransition(
        fromCheckpoint=checkpoint,
        onStatus=status,
        toCheckpoint=to,
        terminalStatus=terminal,
    )


RNA_DECISION_TRANSITION_GRAPH = RnaDecisionTransitionGraph(
    transitions=[
        _transition("qcGrouping", "apply", to="cellQuality"),
        _transition("qcGrouping", "defer", terminal="needsInput"),
        _transition("cellQuality", "apply", to="featurePolicy"),
        _transition("cellQuality", "skip", to="featurePolicy"),
        _transition("cellQuality", "defer", terminal="needsInput"),
        _transition("featurePolicy", "apply", to="hvgRanking"),
        _transition("featurePolicy", "skip", to="hvgRanking"),
        _transition("featurePolicy", "defer", terminal="needsInput"),
        _transition("hvgRanking", "apply", to="hvgCount"),
        _transition("hvgRanking", "defer", terminal="needsInput"),
        _transition("hvgCount", "apply", to="pcaPrefix"),
        _transition("hvgCount", "defer", terminal="needsInput"),
        _transition("pcaPrefix", "apply", to="correctionLicense"),
        _transition("pcaPrefix", "defer", terminal="needsInput"),
        _transition("correctionLicense", "apply", to="correctionNeed"),
        _transition("correctionLicense", "skip", to="correctionOutcome"),
        _transition("correctionLicense", "defer", terminal="needsInput"),
        _transition("correctionNeed", "apply", to="correctionOutcome"),
        _transition("correctionNeed", "skip", to="correctionOutcome"),
        _transition("correctionNeed", "defer", terminal="needsInput"),
        _transition("correctionOutcome", "apply", to="graphK"),
        _transition("correctionOutcome", "skip", to="graphK"),
        _transition("correctionOutcome", "defer", terminal="needsInput"),
        _transition("graphK", "apply", to="clusterPartition"),
        _transition("graphK", "defer", terminal="needsInput"),
        _transition("clusterPartition", "apply", to="finalize"),
        _transition("clusterPartition", "defer", terminal="needsInput"),
        _transition("clusterPartition", "abstain", terminal="abstained"),
    ]
)


class RnaDecisionRegistry(RnaRegistryModel):
    """Ordered definitions for one concrete RNA decision run."""

    definitions: list[RnaDecisionDefinition] = Field(default_factory=list)
    transitionGraph: RnaDecisionTransitionGraph = RNA_DECISION_TRANSITION_GRAPH

    @model_validator(mode="after")
    def validate_definitions(self) -> "RnaDecisionRegistry":
        decision_ids = [definition.spec.decisionId for definition in self.definitions]
        checkpoints = [definition.checkpoint for definition in self.definitions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("Registry decision IDs must be unique")
        if len(checkpoints) != len(set(checkpoints)):
            raise ValueError("Registry checkpoints must be unique")
        positions = {
            node: position
            for position, node in enumerate(self.transitionGraph.orderedNodes)
        }
        if checkpoints != sorted(checkpoints, key=positions.__getitem__):
            raise ValueError("Registry definitions must follow RNA checkpoint order")
        for definition in self.definitions:
            for option in definition.spec.options:
                try:
                    self.transitionGraph.resolve(definition.checkpoint, option.status)
                except KeyError as exc:
                    raise ValueError(
                        f"No transition for {definition.checkpoint}/{option.status}"
                    ) from exc
        return self

    def definition(self, checkpoint: RnaDecisionCheckpoint) -> RnaDecisionDefinition:
        """Return the exact definition registered for a checkpoint."""
        for definition in self.definitions:
            if definition.checkpoint == checkpoint:
                return definition
        raise KeyError(f"No RNA decision definition for {checkpoint}")


class CompiledRnaDecision(RnaRegistryModel):
    """Verified executor handoff kept separate from agent-authored records."""

    decisionRecordId: str
    decisionId: str
    selectedOptionId: str
    status: DecisionStatus
    executorPayload: RnaOptionPayload
    verification: VerificationRecord


def compile_rna_decision(
    definition: RnaDecisionDefinition,
    evidence: EvidenceBundle,
    record: DecisionRecord,
    *,
    created_at_ns: int = 0,
) -> CompiledRnaDecision:
    """Audit an exact selection and resolve its executor-owned payload."""
    verification = DeterministicDecisionAuditor.audit(
        definition.spec,
        evidence,
        record,
        created_at_ns=created_at_ns,
    )
    failed_checks = [
        check.checkId for check in verification.checks if check.status == "failed"
    ]
    if failed_checks:
        raise RnaDecisionCompilationError(
            "Decision failed deterministic verification: " + ", ".join(failed_checks)
        )
    if record.verificationId != verification.verificationId:
        raise RnaDecisionCompilationError(
            "DecisionRecord must reference its deterministic verification ID"
        )
    executor_option = definition.executor_option(record.selectedOptionId)
    return CompiledRnaDecision(
        decisionRecordId=record.recordId,
        decisionId=record.decisionId,
        selectedOptionId=record.selectedOptionId,
        status=record.status,
        executorPayload=executor_option.payload,
        verification=verification,
    )


def _defer_option(
    checkpoint: RnaDecisionCheckpoint, option_id: str
) -> tuple[DecisionOption, RnaExecutorOption]:
    return (
        DecisionOption(
            optionId=option_id,
            status="defer",
            label="Request input",
            description="Pause because the available evidence cannot identify a choice.",
        ),
        RnaExecutorOption(
            checkpoint=checkpoint,
            optionId=option_id,
            payload=NoExecutionPayload(reasonCode="needsInput"),
        ),
    )


def _build_definition(
    *,
    checkpoint: RnaDecisionCheckpoint,
    decision_id: str,
    evidence_bundle_id: str,
    question: str,
    visible_options: list[DecisionOption],
    executor_options: list[RnaExecutorOption],
    baseline_option_id: str | None,
    metric_preferred_option_id: str | None = None,
    require_override_evidence: bool = False,
    allowed_sources: list[Literal["rule", "agent", "human"]] | None = None,
) -> RnaDecisionDefinition:
    return RnaDecisionDefinition(
        checkpoint=checkpoint,
        spec=DecisionSpec(
            decisionId=decision_id,
            definitionVersion=1,
            checkpoint=checkpoint,
            question=question,
            evidenceBundleId=evidence_bundle_id,
            options=visible_options,
            baselineOptionId=baseline_option_id,
            metricPreferredOptionId=metric_preferred_option_id,
            requireIndependentOverrideEvidence=require_override_evidence,
            allowedSources=allowed_sources or ["rule", "agent", "human"],
        ),
        executorOptions=executor_options,
    )


def require_option_evidence(
    definition: RnaDecisionDefinition,
    requirements: Mapping[str, Sequence[str]],
) -> RnaDecisionDefinition:
    """Bind exact evidence IDs to the registered options they support."""
    options = definition.spec.option_by_id()
    unknown = sorted(set(requirements) - set(options))
    if unknown:
        raise ValueError(f"Evidence requirements name unknown options: {unknown}")
    updated: list[DecisionOption] = []
    for option in definition.spec.options:
        raw_ids = requirements.get(option.optionId, ())
        if isinstance(raw_ids, str | bytes):
            raise TypeError("Option evidence requirements must be sequences of IDs")
        updated.append(
            option.model_copy(
                update={"requiredEvidenceIds": list(dict.fromkeys(raw_ids))}
            )
        )
    spec = definition.spec.model_copy(update={"options": updated})
    return definition.model_copy(update={"spec": spec})


def build_qc_grouping_decision(
    *,
    evidence_bundle_id: str,
    physical_capture_eligible: bool,
    pooled_reference_eligible: bool,
) -> RnaDecisionDefinition:
    """Build only quality-threshold grouping modes licensed by design evidence."""
    if pooled_reference_eligible and not physical_capture_eligible:
        raise RnaDecisionGateError(
            "Pooled-reference QC requires eligible physical captures"
        )
    rows: list[tuple[str, str, str, QcGroupingMode]] = [
        (
            "qcGrouping:global",
            "Global reference",
            "Estimate registered quality boundaries across the selected dataset.",
            "global",
        )
    ]
    if physical_capture_eligible:
        rows.append(
            (
                "qcGrouping:physicalCapture",
                "Physical-capture references",
                "Estimate registered boundaries separately within proven captures.",
                "physicalCapture",
            )
        )
    if pooled_reference_eligible:
        rows.append(
            (
                "qcGrouping:pooledReference",
                "Comparable pooled reference",
                "Estimate registered boundaries from the exact comparable captures.",
                "pooledReference",
            )
        )
    visible = [
        DecisionOption(
            optionId=option_id,
            status="apply",
            label=label,
            description=description,
            requiredEvidenceClasses=["qualityControl", "design"],
        )
        for option_id, label, description, _mode in rows
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="qcGrouping",
            optionId=option_id,
            payload=QcGroupingExecutorPayload(groupingMode=mode),
        )
        for option_id, _label, _description, mode in rows
    ]
    defer_visible, defer_executor = _defer_option("qcGrouping", "qcGrouping:defer")
    visible.append(defer_visible)
    executor.append(defer_executor)
    return _build_definition(
        checkpoint="qcGrouping",
        decision_id="qcGrouping",
        evidence_bundle_id=evidence_bundle_id,
        question="Which reference population should define cell-quality boundaries?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id="qcGrouping:global",
    )


def build_cell_quality_decision(
    *,
    evidence_bundle_id: str,
    available_profiles: Sequence[CellQualityProfile],
) -> RnaDecisionDefinition:
    """Build exactly the cell-quality profiles licensed by grouping evidence."""
    if len(available_profiles) != len(set(available_profiles)):
        raise RnaDecisionGateError("Cell-quality profiles must not contain duplicates")
    if not available_profiles:
        raise RnaDecisionGateError("At least one cell-quality profile is required")
    definitions: dict[
        CellQualityProfile,
        tuple[str, DecisionStatus, str, str, CellQualityExecutorPayload],
    ] = {
        "retainWithFlags": (
            "cellQuality:retainWithFlags",
            "skip",
            "Retain with flags",
            "Preserve the published cell set and retain diagnostic quality flags.",
            CellQualityExecutorPayload(
                profile="retainWithFlags",
                groupByCapture=False,
                pooledReference=False,
                sensitivityOnly=False,
            ),
        ),
        "globalMad5": (
            "cellQuality:globalMad5",
            "apply",
            "Global lenient filter",
            "Apply one-sided global five-MAD quality boundaries.",
            CellQualityExecutorPayload(
                profile="globalMad5",
                lowerCountMad=5.0,
                lowerFeatureMad=5.0,
                upperMitoMad=5.0,
                groupByCapture=False,
                pooledReference=False,
                sensitivityOnly=False,
            ),
        ),
        "captureMad5": (
            "cellQuality:captureMad5",
            "apply",
            "Capture-aware lenient filter",
            "Apply one-sided five-MAD boundaries within physical captures.",
            CellQualityExecutorPayload(
                profile="captureMad5",
                lowerCountMad=5.0,
                lowerFeatureMad=5.0,
                upperMitoMad=5.0,
                groupByCapture=True,
                pooledReference=False,
                sensitivityOnly=False,
            ),
        ),
        "captureMad3Sensitivity": (
            "cellQuality:captureMad3Sensitivity",
            "apply",
            "Capture sensitivity branch",
            "Evaluate stricter three-MAD capture boundaries as sensitivity evidence.",
            CellQualityExecutorPayload(
                profile="captureMad3Sensitivity",
                lowerCountMad=3.0,
                lowerFeatureMad=3.0,
                upperMitoMad=3.0,
                groupByCapture=True,
                pooledReference=False,
                sensitivityOnly=True,
            ),
        ),
        "pooledReferenceMad5": (
            "cellQuality:pooledReferenceMad5",
            "apply",
            "Pooled-reference filter",
            "Apply five-MAD boundaries from comparable pooled reference captures.",
            CellQualityExecutorPayload(
                profile="pooledReferenceMad5",
                lowerCountMad=5.0,
                lowerFeatureMad=5.0,
                upperMitoMad=5.0,
                groupByCapture=False,
                pooledReference=True,
                sensitivityOnly=False,
            ),
        ),
    }
    profiles = [definitions[profile] for profile in available_profiles]

    visible = [
        DecisionOption(
            optionId=option_id,
            status=status,
            label=label,
            description=description,
            requiredEvidenceClasses=["qualityControl"],
        )
        for option_id, status, label, description, _payload in profiles
    ]
    executor = [
        RnaExecutorOption(checkpoint="cellQuality", optionId=option_id, payload=payload)
        for option_id, _status, _label, _description, payload in profiles
    ]
    defer_visible, defer_executor = _defer_option("cellQuality", "cellQuality:defer")
    visible.append(defer_visible)
    executor.append(defer_executor)
    return _build_definition(
        checkpoint="cellQuality",
        decision_id="cellQuality",
        evidence_bundle_id=evidence_bundle_id,
        question="Which registered cell-quality profile preserves valid biology?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id=(
            "cellQuality:retainWithFlags"
            if "retainWithFlags" in available_profiles
            else profiles[0][0]
        ),
    )


def build_hvg_ranking_decision(
    *,
    evidence_bundle_id: str,
    batch_aware_eligible: bool,
) -> RnaDecisionDefinition:
    """Build exact global and, when licensed, technical-group HVG rankings."""
    rows: list[tuple[str, str, str, HvgRankingMode]] = [
        (
            "hvgRanking:global",
            "Global variability ranking",
            "Rank genes by corrected variability across all selected cells.",
            "global",
        )
    ]
    if batch_aware_eligible:
        rows.append(
            (
                "hvgRanking:batchAware",
                "Technical-group recurrence ranking",
                "Rank genes by recurrence and within-group rank across valid groups.",
                "batchAware",
            )
        )
    visible = [
        DecisionOption(
            optionId=option_id,
            status="apply",
            label=label,
            description=description,
            requiredEvidenceClasses=["technical"],
        )
        for option_id, label, description, _mode in rows
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="hvgRanking",
            optionId=option_id,
            payload=HvgRankingExecutorPayload(rankingMode=mode),
        )
        for option_id, _label, _description, mode in rows
    ]
    defer_visible, defer_executor = _defer_option("hvgRanking", "hvgRanking:defer")
    visible.append(defer_visible)
    executor.append(defer_executor)
    return _build_definition(
        checkpoint="hvgRanking",
        decision_id="hvgRanking",
        evidence_bundle_id=evidence_bundle_id,
        question="Which registered variability ranking is supported by the design?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id="hvgRanking:global",
    )


def _bounded_options(
    *,
    maximum: int,
    fixed: list[tuple[str, str, int]],
    maximum_id: str,
    maximum_label: str,
) -> list[tuple[str, str, int]]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    bounded = [item for item in fixed if item[2] <= maximum]
    fixed_values = {value for _option_id, _label, value in bounded}
    if maximum < fixed[-1][2] and maximum not in fixed_values:
        bounded.append((maximum_id, maximum_label, maximum))
    return bounded


def _baseline_for_value(options: list[tuple[str, str, int]], desired_value: int) -> str:
    return min(options, key=lambda item: (abs(item[2] - desired_value), item[2]))[0]


def build_hvg_count_decision(
    *,
    evidence_bundle_id: str,
    eligible_feature_count: int,
    ranking_mode: HvgRankingMode,
    valid_technical_groups: int = 0,
    candidate_counts: Sequence[int] | None = None,
) -> RnaDecisionDefinition:
    """Build capped HVG counts with the ranking route fixed by capabilities."""
    if eligible_feature_count < 2:
        raise RnaDecisionGateError("HVG selection requires at least two eligible genes")
    if ranking_mode == "batchAware" and valid_technical_groups < 2:
        raise RnaDecisionGateError(
            "Batch-aware HVGs require at least two valid technical groups"
        )
    if candidate_counts is None:
        candidates = _bounded_options(
            maximum=eligible_feature_count,
            fixed=[
                ("hvgCount:focused", "Focused HVG set", 1000),
                ("hvgCount:standard", "Standard HVG set", 2000),
                ("hvgCount:broad", "Broad HVG set", 4000),
            ],
            maximum_id="hvgCount:allEligible",
            maximum_label="All eligible genes",
        )
    else:
        effective_counts: list[int] = []
        for value in candidate_counts:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RnaDecisionGateError(
                    "HVG candidate counts must be positive integers"
                )
            effective = min(value, eligible_feature_count)
            if effective not in effective_counts:
                effective_counts.append(effective)
        if not effective_counts:
            raise RnaDecisionGateError("At least one HVG candidate is required")
        known = {
            1000: ("hvgCount:focused", "Focused HVG set"),
            2000: ("hvgCount:standard", "Standard HVG set"),
            4000: ("hvgCount:broad", "Broad HVG set"),
        }
        candidates = [
            (
                known.get(value, (f"hvgCount:n{value}", f"{value} HVGs"))[0],
                known.get(value, (f"hvgCount:n{value}", f"{value} HVGs"))[1],
                value,
            )
            for value in effective_counts
        ]
    visible = [
        DecisionOption(
            optionId=option_id,
            status="apply",
            label=label,
            description="Use this registered HVG count for representation diagnostics.",
            requiredEvidenceClasses=["technical"],
        )
        for option_id, label, _top_n in candidates
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="hvgCount",
            optionId=option_id,
            payload=HvgExecutorPayload(topN=top_n, rankingMode=ranking_mode),
        )
        for option_id, _label, top_n in candidates
    ]
    defer_visible, defer_executor = _defer_option("hvgCount", "hvgCount:defer")
    visible.append(defer_visible)
    executor.append(defer_executor)
    baseline = _baseline_for_value(candidates, min(2000, eligible_feature_count))
    return _build_definition(
        checkpoint="hvgCount",
        decision_id="hvgCount",
        evidence_bundle_id=evidence_bundle_id,
        question="Which registered HVG count retains reproducible signal?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id=baseline,
    )


def build_feature_policy_decision(
    *,
    evidence_bundle_id: str,
    proposed_exclusion_families: list[ConditionalGeneFamily],
    dominant_families: list[ConditionalGeneFamily],
    protected_families: list[ConditionalGeneFamily],
    scarf_default_eligible: bool = False,
) -> RnaDecisionDefinition:
    """Build exact representation-only feature-policy alternatives."""
    for field_name, values in (
        ("proposed_exclusion_families", proposed_exclusion_families),
        ("dominant_families", dominant_families),
        ("protected_families", protected_families),
    ):
        if len(values) != len(set(values)):
            raise RnaDecisionGateError(f"{field_name} must not contain duplicates")
    proposed = set(proposed_exclusion_families)
    if not proposed.issubset(dominant_families):
        raise RnaDecisionGateError(
            "Conditional exclusion requires observed dominance evidence"
        )
    protected_overlap = proposed.intersection(protected_families)
    if protected_overlap:
        blocked = ", ".join(sorted(protected_overlap))
        raise RnaDecisionGateError(
            f"Objective-protected gene families cannot be excluded: {blocked}"
        )

    visible = [
        DecisionOption(
            optionId="featurePolicy:keepAll",
            status="skip",
            label="Keep conditional families",
            description="Keep all conditional biological gene families in representation.",
            requiredEvidenceClasses=["technical"],
        )
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="featurePolicy",
            optionId="featurePolicy:keepAll",
            payload=FeaturePolicyExecutorPayload(policy="keepAll", excludedFamilies=[]),
        )
    ]
    if scarf_default_eligible:
        visible.append(
            DecisionOption(
                optionId="featurePolicy:excludeScarfDefaults",
                status="apply",
                label="Use the Scarf default blacklist",
                description=(
                    "Exclude the exact core Scarf default blacklist from "
                    "representation only."
                ),
                requiredEvidenceClasses=["technical"],
            )
        )
        executor.append(
            RnaExecutorOption(
                checkpoint="featurePolicy",
                optionId="featurePolicy:excludeScarfDefaults",
                payload=FeaturePolicyExecutorPayload(
                    policy="excludeScarfDefaults",
                    useScarfDefaultBlacklist=True,
                ),
            )
        )
    if proposed_exclusion_families:
        visible.append(
            DecisionOption(
                optionId="featurePolicy:excludeEligibleBundle",
                status="apply",
                label="Exclude eligible nuisance bundle",
                description=(
                    "Exclude the one deterministic nuisance-family bundle from "
                    "representation only."
                ),
                requiredEvidenceClasses=["technical"],
            )
        )
        executor.append(
            RnaExecutorOption(
                checkpoint="featurePolicy",
                optionId="featurePolicy:excludeEligibleBundle",
                payload=FeaturePolicyExecutorPayload(
                    policy="excludeEligibleBundle",
                    excludedFamilies=proposed_exclusion_families,
                ),
            )
        )
    defer_visible, defer_executor = _defer_option(
        "featurePolicy", "featurePolicy:defer"
    )
    visible.append(defer_visible)
    executor.append(defer_executor)
    return _build_definition(
        checkpoint="featurePolicy",
        decision_id="featurePolicy",
        evidence_bundle_id=evidence_bundle_id,
        question="Should the eligible nuisance-family bundle leave representation?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id="featurePolicy:keepAll",
    )


def build_pca_prefix_decision(
    *,
    evidence_bundle_id: str,
    matrix_rank: int,
    candidate_dimensions: Sequence[int] | None = None,
) -> RnaDecisionDefinition:
    """Build PCA prefixes capped by rank and by the v1 fifty-PC limit."""
    maximum = min(matrix_rank, 50)
    if maximum < 2:
        raise RnaDecisionGateError("PCA requires matrix rank of at least two")
    if candidate_dimensions is None:
        candidates = _bounded_options(
            maximum=maximum,
            fixed=[
                ("pcaPrefix:short", "Short PCA prefix", 10),
                ("pcaPrefix:standard", "Standard PCA prefix", 20),
                ("pcaPrefix:extended", "Extended PCA prefix", 30),
                ("pcaPrefix:maximum", "Maximum PCA prefix", 50),
            ],
            maximum_id="pcaPrefix:maximumAvailable",
            maximum_label="Maximum available PCA prefix",
        )
    else:
        values: list[int] = []
        for value in candidate_dimensions:
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise RnaDecisionGateError(
                    "PCA candidate dimensions must be integers of at least two"
                )
            effective = min(value, maximum)
            if effective not in values:
                values.append(effective)
        known = {
            10: ("pcaPrefix:short", "Short PCA prefix"),
            20: ("pcaPrefix:standard", "Standard PCA prefix"),
            30: ("pcaPrefix:extended", "Extended PCA prefix"),
            50: ("pcaPrefix:maximum", "Maximum PCA prefix"),
        }
        candidates = [
            (
                known.get(value, (f"pcaPrefix:n{value}", f"{value} PCs"))[0],
                known.get(value, (f"pcaPrefix:n{value}", f"{value} PCs"))[1],
                value,
            )
            for value in values
        ]
        if not candidates:
            raise RnaDecisionGateError("At least one PCA candidate is required")
    visible = [
        DecisionOption(
            optionId=option_id,
            status="apply",
            label=label,
            description="Use this registered prefix of the single computed PCA.",
            requiredEvidenceClasses=["geometric", "technical"],
        )
        for option_id, label, _dimensions in candidates
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="pcaPrefix",
            optionId=option_id,
            payload=PcaPrefixExecutorPayload(dimensions=dimensions),
        )
        for option_id, _label, dimensions in candidates
    ]
    defer_visible, defer_executor = _defer_option("pcaPrefix", "pcaPrefix:defer")
    visible.append(defer_visible)
    executor.append(defer_executor)
    baseline = _baseline_for_value(candidates, min(20, maximum))
    return _build_definition(
        checkpoint="pcaPrefix",
        decision_id="pcaPrefix",
        evidence_bundle_id=evidence_bundle_id,
        question="What is the smallest registered PCA prefix that stabilizes topology?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id=baseline,
    )


def build_correction_license_decision(
    *, evidence_bundle_id: str, license: CorrectionLicense
) -> RnaDecisionDefinition:
    """Record the one correction license authorized by deterministic design checks."""
    statuses: dict[CorrectionLicense, DecisionStatus] = {
        "safe": "apply",
        "unsafeConfounded": "skip",
        "indeterminate": "defer",
        "notApplicable": "skip",
    }
    option_id = f"correctionLicense:{license}"
    visible = [
        DecisionOption(
            optionId=option_id,
            status=statuses[license],
            label="Correction design license",
            description="Use the exact correction license produced by design validation.",
            requiredEvidenceClasses=["design"],
        )
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="correctionLicense",
            optionId=option_id,
            payload=CorrectionLicensePayload(license=license),
        )
    ]
    return _build_definition(
        checkpoint="correctionLicense",
        decision_id="correctionLicense",
        evidence_bundle_id=evidence_bundle_id,
        question="Does the experimental design authorize batch correction?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id=option_id,
        allowed_sources=["rule"],
    )


def build_correction_need_decision(
    *, evidence_bundle_id: str, license: CorrectionLicense
) -> RnaDecisionDefinition:
    """Build correction-need options only after a safe design license."""
    if license != "safe":
        raise RnaDecisionGateError(
            "Correction need is evaluated only after a safe correction license"
        )
    rows: list[tuple[str, DecisionStatus, str, CorrectionNeed]] = [
        (
            "correctionNeed:needed",
            "apply",
            "Technical separation is present within comparable populations.",
            "needed",
        ),
        (
            "correctionNeed:notNeeded",
            "skip",
            "The native representation does not show material technical separation.",
            "notNeeded",
        ),
        (
            "correctionNeed:indeterminate",
            "defer",
            "Available evidence cannot distinguish technical and protected structure.",
            "indeterminate",
        ),
    ]
    visible = [
        DecisionOption(
            optionId=option_id,
            status=status,
            label=need,
            description=description,
            requiredEvidenceClasses=["batchRemoval", "biologicalConservation"]
            if need != "indeterminate"
            else ["design"],
        )
        for option_id, status, description, need in rows
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="correctionNeed",
            optionId=option_id,
            payload=CorrectionNeedPayload(need=need),
        )
        for option_id, _status, _description, need in rows
    ]
    return _build_definition(
        checkpoint="correctionNeed",
        decision_id="correctionNeed",
        evidence_bundle_id=evidence_bundle_id,
        question="Does the native representation show a licensed need for correction?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id="correctionNeed:notNeeded",
    )


def build_correction_outcome_decision(
    *,
    evidence_bundle_id: str,
    license: CorrectionLicense,
    need: CorrectionNeed | None = None,
    harmony_eligible: bool = True,
) -> RnaDecisionDefinition:
    """Build a native baseline and offer Harmony only when licensed and needed."""
    if license == "indeterminate":
        raise RnaDecisionGateError(
            "Indeterminate correction license must resolve before outcome comparison"
        )
    if license == "safe" and need is None:
        raise RnaDecisionGateError(
            "A safe correction license requires an evaluated correction need"
        )
    if license != "safe" and need is not None:
        raise RnaDecisionGateError(
            "Correction need must not bypass an unsafe or inapplicable license"
        )
    if need == "indeterminate":
        raise RnaDecisionGateError(
            "Indeterminate correction need must resolve before outcome comparison"
        )

    visible = [
        DecisionOption(
            optionId="correctionOutcome:retainNative",
            status="skip",
            label="Retain native representation",
            description="Keep the mandatory uncorrected representation baseline.",
            requiredEvidenceClasses=["biologicalConservation"],
        )
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="correctionOutcome",
            optionId="correctionOutcome:retainNative",
            payload=CorrectionOutcomeExecutorPayload(
                outcome="retainNative", useHarmony=False
            ),
        )
    ]
    offer_harmony = license == "safe" and need == "needed" and harmony_eligible
    if offer_harmony:
        visible.append(
            DecisionOption(
                optionId="correctionOutcome:acceptHarmony",
                status="apply",
                label="Accept Harmony",
                description="Use the matched Harmony representation branch.",
                requiredEvidenceClasses=[
                    "batchRemoval",
                    "biologicalConservation",
                    "protectedVariablePreservation",
                ],
            )
        )
        executor.append(
            RnaExecutorOption(
                checkpoint="correctionOutcome",
                optionId="correctionOutcome:acceptHarmony",
                payload=CorrectionOutcomeExecutorPayload(
                    outcome="acceptHarmony", useHarmony=True
                ),
            )
        )
    defer_visible, defer_executor = _defer_option(
        "correctionOutcome", "correctionOutcome:indeterminate"
    )
    visible.append(defer_visible)
    executor.append(defer_executor)
    allowed_sources: list[Literal["rule", "agent", "human"]] = (
        ["rule", "agent", "human"] if offer_harmony else ["rule"]
    )
    return _build_definition(
        checkpoint="correctionOutcome",
        decision_id="correctionOutcome",
        evidence_bundle_id=evidence_bundle_id,
        question="Should the verified final representation remain native or use Harmony?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id="correctionOutcome:retainNative",
        allowed_sources=allowed_sources,
    )


def build_graph_k_decision(
    *,
    evidence_bundle_id: str,
    n_cells: int,
    candidate_neighbors: Sequence[int] | None = None,
) -> RnaDecisionDefinition:
    """Build graph scales capped by the selected cell count."""
    maximum = min(n_cells - 1, 41)
    if maximum < 2:
        raise RnaDecisionGateError("Graph construction requires at least three cells")
    if candidate_neighbors is None:
        candidates = _bounded_options(
            maximum=maximum,
            fixed=[
                ("graphScale:local", "Local graph", 11),
                ("graphScale:balanced", "Balanced graph", 21),
                ("graphScale:broad", "Broad graph", 41),
            ],
            maximum_id="graphScale:maximumAvailable",
            maximum_label="Maximum available graph",
        )
    else:
        values: list[int] = []
        for value in candidate_neighbors:
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise RnaDecisionGateError(
                    "Graph candidates must be integers of at least two"
                )
            effective = min(value, maximum)
            if effective not in values:
                values.append(effective)
        known = {
            11: ("graphScale:local", "Local graph"),
            21: ("graphScale:balanced", "Balanced graph"),
            41: ("graphScale:broad", "Broad graph"),
        }
        candidates = [
            (
                known.get(value, (f"graphScale:k{value}", f"{value}-neighbor graph"))[
                    0
                ],
                known.get(value, (f"graphScale:k{value}", f"{value}-neighbor graph"))[
                    1
                ],
                value,
            )
            for value in values
        ]
        if not candidates:
            raise RnaDecisionGateError("At least one graph candidate is required")
    visible = [
        DecisionOption(
            optionId=option_id,
            status="apply",
            label=label,
            description="Use this registered neighborhood scale for graph diagnostics.",
            requiredEvidenceClasses=["geometric"],
        )
        for option_id, label, _neighbors in candidates
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="graphK",
            optionId=option_id,
            payload=GraphExecutorPayload(neighborsK=neighbors),
        )
        for option_id, _label, neighbors in candidates
    ]
    defer_visible, defer_executor = _defer_option("graphK", "graphScale:defer")
    visible.append(defer_visible)
    executor.append(defer_executor)
    baseline = _baseline_for_value(candidates, min(21, maximum))
    return _build_definition(
        checkpoint="graphK",
        decision_id="graphK",
        evidence_bundle_id=evidence_bundle_id,
        question="Which registered graph scale is stable and locally informative?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id=baseline,
    )


def build_cluster_partition_decision(
    *,
    evidence_bundle_id: str,
    metric_preferred_option_id: str,
    resolution_candidates: Sequence[float] | None = None,
) -> RnaDecisionDefinition:
    """Build fixed Leiden resolutions plus explicit defer and abstain outcomes."""
    default_rows: list[tuple[str, str, float]] = [
        ("clusterResolution:veryCoarse", "Very coarse partition", 0.25),
        ("clusterResolution:coarse", "Coarse partition", 0.5),
        ("clusterResolution:balanced", "Balanced partition", 0.75),
        ("clusterResolution:detailed", "Detailed partition", 1.0),
        ("clusterResolution:fine", "Fine partition", 1.25),
        ("clusterResolution:veryFine", "Very fine partition", 1.5),
    ]
    if resolution_candidates is None:
        rows = default_rows
    else:
        known = {
            resolution: (option_id, label)
            for option_id, label, resolution in default_rows
        }
        rows = []
        seen: set[float] = set()
        for raw in resolution_candidates:
            resolution = float(raw)
            if not 0 < resolution <= 1.5 or resolution in seen:
                raise RnaDecisionGateError(
                    "Cluster resolutions must be unique values in (0, 1.5]"
                )
            seen.add(resolution)
            token = str(resolution).replace(".", "p")
            option_id, label = known.get(
                resolution,
                (
                    f"clusterResolution:r{token}",
                    f"Leiden resolution {resolution:g}",
                ),
            )
            rows.append((option_id, label, resolution))
        if not rows:
            raise RnaDecisionGateError("At least one cluster resolution is required")
    resolution_ids = [option_id for option_id, _label, _resolution in rows]
    if metric_preferred_option_id not in resolution_ids:
        raise RnaDecisionGateError(
            "metric_preferred_option_id must be a registered resolution option"
        )
    baseline_option_id = min(
        rows,
        key=lambda item: (abs(item[2] - 0.75), item[2]),
    )[0]
    visible = [
        DecisionOption(
            optionId=option_id,
            status="apply",
            label=label,
            description="Use this registered Leiden resolution.",
            requiredEvidenceClasses=["geometric"],
        )
        for option_id, label, _resolution in rows
    ]
    executor = [
        RnaExecutorOption(
            checkpoint="clusterPartition",
            optionId=option_id,
            payload=ClusterExecutorPayload(leidenResolution=resolution),
        )
        for option_id, _label, resolution in rows
    ]
    defer_visible, defer_executor = _defer_option(
        "clusterPartition", "clusterPartition:defer"
    )
    visible.append(defer_visible)
    executor.append(defer_executor)
    visible.append(
        DecisionOption(
            optionId="clusterPartition:abstain",
            status="abstain",
            label="Abstain from discrete clustering",
            description="Do not claim a defensible discrete partition.",
        )
    )
    executor.append(
        RnaExecutorOption(
            checkpoint="clusterPartition",
            optionId="clusterPartition:abstain",
            payload=NoExecutionPayload(reasonCode="scientificAbstention"),
        )
    )
    return _build_definition(
        checkpoint="clusterPartition",
        decision_id="clusterPartition",
        evidence_bundle_id=evidence_bundle_id,
        question="Which registered partition is scientifically defensible?",
        visible_options=visible,
        executor_options=executor,
        baseline_option_id=baseline_option_id,
        metric_preferred_option_id=metric_preferred_option_id,
        require_override_evidence=True,
    )


__all__ = [
    "CellQualityProfile",
    "CellQualityExecutorPayload",
    "ClusterExecutorPayload",
    "CompiledRnaDecision",
    "ConditionalGeneFamily",
    "CorrectionLicense",
    "CorrectionLicensePayload",
    "CorrectionNeed",
    "CorrectionNeedPayload",
    "CorrectionOutcomeExecutorPayload",
    "DecisionTerminalStatus",
    "FeaturePolicyExecutorPayload",
    "GraphExecutorPayload",
    "HvgExecutorPayload",
    "HvgRankingExecutorPayload",
    "HvgRankingMode",
    "NoExecutionPayload",
    "PcaPrefixExecutorPayload",
    "QcGroupingExecutorPayload",
    "QcGroupingMode",
    "RNA_DECISION_TRANSITION_GRAPH",
    "RnaDecisionCheckpoint",
    "RnaDecisionCompilationError",
    "RnaDecisionDefinition",
    "RnaDecisionGateError",
    "RnaDecisionRegistry",
    "RnaDecisionTransition",
    "RnaDecisionTransitionGraph",
    "RnaExecutorOption",
    "RnaOptionPayload",
    "RnaWorkflowNode",
    "build_cell_quality_decision",
    "build_cluster_partition_decision",
    "build_correction_license_decision",
    "build_correction_need_decision",
    "build_correction_outcome_decision",
    "build_feature_policy_decision",
    "build_graph_k_decision",
    "build_hvg_count_decision",
    "build_hvg_ranking_decision",
    "build_pca_prefix_decision",
    "build_qc_grouping_decision",
    "compile_rna_decision",
    "require_option_evidence",
]
