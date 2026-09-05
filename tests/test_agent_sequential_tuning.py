from typing import Any

import pytest
from pydantic import ValidationError

from scarf.agent.parameter_tuning import (
    ArtifactRecord,
    ParameterCandidate,
    ParameterCandidateEvaluation,
)
from scarf.agent.sequential_tuning import (
    CorrectionNeedSelection,
    ParameterPhaseEvidence,
    ParameterPhasePlan,
    ParameterPhaseSelection,
    SequentialAssayTuningEvidence,
    SequentialRnaTuningPlanner,
    execute_parameter_phase,
    sequential_evidence_to_report,
    validate_parameter_phase_selection,
)
from scarf.agent.types import ArtifactReferenceModel


def _evaluation(candidate: ParameterCandidate) -> ParameterCandidateEvaluation:
    evidence_id = f"candidate:{candidate.candidateId}:clusters"
    return ParameterCandidateEvaluation(
        candidateId=candidate.candidateId,
        status="done",
        eligible=True,
        parameters=candidate,
        artifacts={
            "connectivityMap": ArtifactRecord(
                assay="RNA",
                kind="connectivity_map",
                artifactId="a" * 64,
            ),
            "clusters": ArtifactRecord(
                assay="RNA",
                kind="cluster_labels",
                artifactId="b" * 64,
            ),
        },
        cellSelection=ArtifactReferenceModel(
            scope="datastore",
            assay=None,
            kind="cell_selection",
            artifactId="c" * 64,
        ),
        clusterColumn=f"RNA_{candidate.candidateId}",
        clusterLabel=candidate.candidateId,
        effectiveDimensions=candidate.dimensions,
        evidenceIds=[evidence_id],
    )


def _selected_phase(
    plan: ParameterPhasePlan,
    *,
    selected_index: int = -1,
) -> ParameterPhaseEvidence:
    evaluations = [_evaluation(candidate) for candidate in plan.candidates]
    selected = evaluations[selected_index]
    return validate_parameter_phase_selection(
        plan,
        evaluations,
        ParameterPhaseSelection(
            phase=plan.phase,
            status="selected",
            selectedCandidateId=selected.candidateId,
            evidenceIds=list(selected.evidenceIds),
            rationale=f"Selected registered {plan.phase} evidence.",
        ),
    )


def test_sequential_planner_caps_registered_rna_candidates() -> None:
    planner = SequentialRnaTuningPlanner(
        workflow_run_id="workflow:with unsafe punctuation",
        assay="RNA sample",
        n_cells=35,
        n_features=100,
        matrix_rank=27,
        harmony_authorized=True,
    )

    pca = planner.pca_prefix_phase()
    assert [value.dimensions for value in pca.candidates] == [10, 20, 27]
    assert all(value.useHarmony is False for value in pca.candidates)
    assert all(
        len(value.candidateId) <= 64 and value.candidateId.replace("_", "").isalnum()
        for value in pca.candidates
    )

    correction = planner.batch_correction_phase(pca.candidates[1])
    assert [value.useHarmony for value in correction.candidates] == [False, True]
    assert {value.dimensions for value in correction.candidates} == {20}

    graph = planner.graph_phase(correction.candidates[1])
    assert [value.neighborsK for value in graph.candidates] == [11, 21, 34]
    assert all(value.useHarmony for value in graph.candidates)

    clustering = planner.clustering_phase(graph.candidates[1])
    assert [value.leidenResolution for value in clustering.candidates] == [
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
    ]
    assert {value.neighborsK for value in clustering.candidates} == {21}


def test_sequential_planner_does_not_offer_unauthorized_harmony() -> None:
    planner = SequentialRnaTuningPlanner(
        workflow_run_id="workflow",
        assay="RNA",
        n_cells=100,
        n_features=50,
        harmony_authorized=False,
    )
    selected = planner.pca_prefix_phase().candidates[0]

    correction = planner.batch_correction_phase(selected)

    assert len(correction.candidates) == 1
    assert correction.candidates[0].useHarmony is False


def test_phase_contract_rejects_noncausal_and_numeric_model_output() -> None:
    first = ParameterCandidate(
        candidateId="pca_10",
        dimensions=10,
        neighborsK=11,
    )
    second = ParameterCandidate(
        candidateId="pca_20",
        dimensions=20,
        neighborsK=21,
    )
    with pytest.raises(ValidationError, match="non-target parameter"):
        ParameterPhasePlan(
            phase="pcaPrefix",
            assay="RNA",
            variedParameter="dimensions",
            candidates=[first, second],
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ParameterPhaseSelection.model_validate(
            {
                "phase": "pcaPrefix",
                "status": "selected",
                "selectedCandidateId": "pca_10",
                "evidenceIds": ["candidate:pca_10:clusters"],
                "rationale": "Choose the exact registered option.",
                "dimensions": 10,
            }
        )


def test_phase_selection_requires_eligible_execution_and_scoped_evidence() -> None:
    plan = SequentialRnaTuningPlanner(
        workflow_run_id="workflow",
        assay="RNA",
        n_cells=100,
        n_features=50,
        harmony_authorized=False,
    ).pca_prefix_phase()
    evaluations = [_evaluation(candidate) for candidate in plan.candidates]

    with pytest.raises(ValidationError, match="outside its phase evaluations"):
        ParameterPhaseEvidence(
            plan=plan,
            evaluations=evaluations,
            selection=ParameterPhaseSelection(
                phase="pcaPrefix",
                status="selected",
                selectedCandidateId=plan.candidates[0].candidateId,
                evidenceIds=["invented:evidence"],
                rationale="This cites evidence that was not executed.",
            ),
        )
    with pytest.raises(ValidationError, match="must cite executor evidence"):
        ParameterPhaseSelection(
            phase="pcaPrefix",
            status="selected",
            selectedCandidateId=plan.candidates[0].candidateId,
            rationale="This selection omitted its evidence.",
        )


def test_completed_sequential_evidence_adapts_for_native_finalization() -> None:
    planner = SequentialRnaTuningPlanner(
        workflow_run_id="workflow",
        assay="RNA",
        n_cells=100,
        n_features=50,
        harmony_authorized=True,
    )
    pca = _selected_phase(planner.pca_prefix_phase(), selected_index=1)
    assert pca.selected_evaluation() is not None
    correction = _selected_phase(
        planner.batch_correction_phase(pca.selected_evaluation().parameters),
        selected_index=1,
    )
    assert correction.selected_evaluation() is not None
    graph = _selected_phase(
        planner.graph_phase(correction.selected_evaluation().parameters),
        selected_index=1,
    )
    assert graph.selected_evaluation() is not None
    clustering = _selected_phase(
        planner.clustering_phase(graph.selected_evaluation().parameters),
        selected_index=2,
    )
    final_id = clustering.selection.selectedCandidateId
    evidence = SequentialAssayTuningEvidence(
        assay="RNA",
        phases=[pca, correction, graph, clustering],
        finalCandidateId=final_id,
    )

    report = sequential_evidence_to_report(evidence)

    assert report.status == "done"
    assert report.recommendedCandidateId == final_id
    assert report.assayReports["RNA"].recommendedCandidateId == final_id
    assert report.graphAssay == "RNA"
    assert report.markerAssay == "RNA"
    assert report.finalClusterArtifact == report.selectedArtifacts["clusters"]
    assert report.finalClusterColumn is not None
    assert report.totalCandidates == sum(
        len(value.evaluations) for value in evidence.phases
    )


def test_pending_sequential_evidence_preserves_exact_resume_options() -> None:
    planner = SequentialRnaTuningPlanner(
        workflow_run_id="workflow",
        assay="RNA",
        n_cells=100,
        n_features=50,
        harmony_authorized=False,
    )
    plan = planner.pca_prefix_phase()
    evaluations = [_evaluation(candidate) for candidate in plan.candidates]
    option_ids = ["pcaPrefix:short", "pcaPrefix:standard", "pcaPrefix:defer"]
    evidence_ids = [value.evidenceIds[0] for value in evaluations]
    state = SequentialAssayTuningEvidence(
        assay="RNA",
        phases=[
            ParameterPhaseEvidence(
                plan=plan,
                evaluations=evaluations,
                selection=ParameterPhaseSelection(
                    phase="pcaPrefix",
                    status="needsInput",
                    rationale="The bounded decision run did not select an option.",
                ),
            )
        ],
        pendingDecisionId="pcaPrefix",
        pendingOptionIds=option_ids,
        pendingEvidenceIds=evidence_ids,
    )

    report = sequential_evidence_to_report(state)

    assert report.status == "needsInput"
    assert report.needsInput is not None
    assert report.needsInput.options == option_ids
    assert report.needsInput.evidenceIds == evidence_ids


def test_pending_correction_need_precedes_batch_phase() -> None:
    planner = SequentialRnaTuningPlanner(
        workflow_run_id="workflow",
        assay="RNA",
        n_cells=100,
        n_features=50,
        harmony_authorized=True,
    )
    pca = _selected_phase(planner.pca_prefix_phase())
    state = SequentialAssayTuningEvidence(
        assay="RNA",
        phases=[pca],
        correctionLicense="safe",
        correctionNeed=CorrectionNeedSelection(
            status="needsInput",
            selectedOptionId="correctionNeed:indeterminate",
            rationale="The native representation evidence is incomplete.",
        ),
        pendingDecisionId="correctionNeed",
        pendingOptionIds=[
            "correctionNeed:needed",
            "correctionNeed:notNeeded",
            "correctionNeed:indeterminate",
        ],
        pendingEvidenceIds=["evidence:correctionNeed:design"],
    )

    report = sequential_evidence_to_report(state)

    assert report.status == "needsInput"
    assert report.needsInput is not None
    assert report.needsInput.options[0] == "correctionNeed:needed"


def test_phase_executor_adapter_preserves_registered_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = SequentialRnaTuningPlanner(
        workflow_run_id="workflow",
        assay="RNA",
        n_cells=50,
        n_features=50,
        harmony_authorized=False,
    )
    plan = planner.pca_prefix_phase()
    expected_ids = tuple(value.candidateId for value in plan.candidates)
    calls: list[str] = []

    def prepare(*args: Any, **kwargs: Any) -> tuple[object, list[str]]:
        assert kwargs["candidates"] == plan.candidates
        return object(), list(expected_ids)

    by_id = {value.candidateId: value for value in plan.candidates}

    def execute(deps: object, candidate_id: str) -> ParameterCandidateEvaluation:
        assert deps is not None
        calls.append(candidate_id)
        return _evaluation(by_id[candidate_id])

    monkeypatch.setattr(
        "scarf.agent.sequential_tuning.prepare_parameter_tuning_dependencies",
        prepare,
    )
    monkeypatch.setattr(
        "scarf.agent.sequential_tuning.execute_parameter_candidate",
        execute,
    )

    evaluations = execute_parameter_phase(object(), normalized=object(), plan=plan)

    assert tuple(value.candidateId for value in evaluations) == expected_ids
    assert tuple(calls) == expected_ids
