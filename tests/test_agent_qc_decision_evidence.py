"""QC choices distinguish reference grouping, measured retention and biology."""

from types import SimpleNamespace

import pytest

from scarf.agent.experimental_context.contracts import (
    CaptureFailureEvidence,
    CellQcProfileEvidence,
)
from scarf.agent.experimental_context.study import StudyContract
from scarf.agent.orchestrator import AgentOrchestrator
from scarf.agent.orchestrator.preprocessing import PreprocessingStagesMixin
from scarf.agent.decisions.rna import QcGroupingExecutorPayload


def profile(policy: str, retained: int) -> CellQcProfileEvidence:
    return CellQcProfileEvidence.model_validate(
        {
            "action": "skip" if policy == "retainWithFlags" else "registeredMad",
            "registeredProfile": policy,
            "attributes": [] if policy == "retainWithFlags" else ["RNA_percentMito"],
            "captureColumn": "library",
            "sampleColumn": "library" if policy == "captureMad5" else None,
            "activeCells": 100,
            "retainedCells": retained,
            "evidenceId": f"qc:{policy}",
        }
    )


def test_qc_evidence_reports_true_median_fractions_and_metric_flags() -> None:
    selected = profile("globalMad5", 40)
    selected.sampleRetainedCells = {"a": 10, "b": 30}
    selected.retainedCellsByColumn = {"condition": {"a": 10, "b": 30}}
    selected.metricFlaggedCells = {"RNA_percentMito": {"highMito": 60}}
    selected.captureFailureEvidence = [
        CaptureFailureEvidence(
            capture="a", activeCells=50, retainedCells=10, retainedFraction=0.2
        ),
        CaptureFailureEvidence(
            capture="b", activeCells=50, retainedCells=30, retainedFraction=0.6
        ),
    ]
    reference = profile("retainWithFlags", 100)
    reference.retainedCellsByColumn = {"condition": {"a": 50, "b": 50}}
    summary = PreprocessingStagesMixin._profile_evidence(selected, reference).summary
    assert "min/median/max=10/20/30" in summary
    assert "min/median/max=20.0%/40.0%/60.0%" in summary
    assert "a=10/50 (20.0%)" in summary
    assert "b=30/50 (60.0%)" in summary
    assert "'highMito': 60" in summary
    assert "flags may overlap" in summary
    assert (
        "fractions unavailable"
        in PreprocessingStagesMixin._profile_evidence(selected).summary
    )


def test_qc_grouping_compares_the_same_cutoff_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = AgentOrchestrator("test-model")
    profiles = [
        CellQcProfileEvidence(
            action="globalGaussian",
            attributes=["RNA_percentMito"],
            activeCells=100,
            retainedCells=60,
            evidenceId="qc:globalGaussian",
        ),
        CellQcProfileEvidence(
            action="sampleMad",
            sampleColumn="library",
            attributes=["RNA_percentMito"],
            activeCells=100,
            retainedCells=55,
            evidenceId="qc:sampleMad3",
        ),
        profile("globalMad5", 90),
        profile("captureMad5", 95),
        profile("retainWithFlags", 100),
    ]
    seen = {}

    def resolve(_store, _request, definition, bundle, _answers, **kwargs):
        seen["definition"] = definition
        seen["bundle"] = bundle
        return SimpleNamespace(
            compiled=SimpleNamespace(
                executorPayload=QcGroupingExecutorPayload(groupingMode="global")
            ),
            checkpointSha256="a" * 64,
        )

    monkeypatch.setattr(orchestrator, "_resolve_rna_decision", resolve)
    orchestrator._resolve_qc_grouping_decision(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(qcProfiles=profiles),
        StudyContract.get_blank().model_copy(
            update={"physicalCaptureColumn": "library"}
        ),
        {},
    )
    options = seen["definition"].spec.option_by_id()
    assert "qc:globalMad5" in options["qcGrouping:global"].requiredEvidenceIds
    assert "qc:captureMad5" in options["qcGrouping:physicalCapture"].requiredEvidenceIds
    assert (
        "qc:sampleMad3" not in options["qcGrouping:physicalCapture"].requiredEvidenceIds
    )
    design = next(
        row.summary for row in seen["bundle"].evidence if row.evidenceClass == "design"
    )
    assert "need not be independent biological units or healthy references" in design
    assert "different cutoff methods cannot isolate" in design
