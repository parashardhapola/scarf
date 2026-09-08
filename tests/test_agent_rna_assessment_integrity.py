"""Scientific choice guards preserve exact observed comparisons and prior reviews."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from scarf.agent.orchestrator import rna_tuning
from scarf.agent.experimental_context.contracts import (
    CovariateComparison,
    CovariateProposal,
)
from scarf.agent.experimental_context.study import unsupported_comparison_limitations
from scarf.agent.parameter_tuning.contracts import ParameterCandidateEvaluation
from tests.test_agent_rna_adaptive import checkpoints as memory_checkpoints  # noqa: F401
from tests.test_agent_rna_evidence_mode import assess, make_run


pytestmark = pytest.mark.usefixtures("memory_checkpoints")


def _run(
    monkeypatch: pytest.MonkeyPatch, *, confounded: bool = False
) -> tuple[rna_tuning.RnaTuningRun, ParameterCandidateEvaluation]:
    run, selected = make_run(monkeypatch, SimpleNamespace(supports_image_input=False))
    run.study = run.study.model_copy(
        update={
            "independentUnitColumns": ["donor"],
            "correctionLicense": "unsafeConfounded" if confounded else "notApplicable",
            "limitations": ["Library and protected tissue are confounded."]
            if confounded
            else [],
        }
    )
    selected.metrics.crossUnitSupport = 0.9
    return run, selected


def _experiment_response(kwargs: dict[str, Any], correction_need: str) -> Any:
    action = assess(**{**kwargs, "output_validator": lambda value: value}).output
    evidence = json.loads(kwargs["user_prompt"])
    action = action.model_copy(
        update={
            "action": "experiment",
            "experimentId": next(iter(evidence["experiments"])),
            "correctionNeed": correction_need,
            "concern": "Assess sensitivity of population support to one setting.",
            "expectedImprovement": "The comparison may improve population support.",
            "rationale": "Request one observed-evidence-driven parameter comparison.",
        }
    )
    return SimpleNamespace(output=kwargs["output_validator"](action))


@pytest.mark.parametrize("correction_need", ["needed", "notNeeded"])
def test_confounded_design_rejects_identifiable_correction_claims_before_commit(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    correction_need: str,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = _run(monkeypatch, confounded=True)
    monkeypatch.setattr(
        rna_tuning,
        "run_agent_sync",
        lambda **kwargs: _experiment_response(kwargs, correction_need),
    )
    with pytest.raises(ValueError, match="design confounds"):
        run.review("full", 0, selected, {})
    assert "parameter_tuning/full/review0" not in saved


def test_confounded_design_can_accept_supported_native_descriptive_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, selected = _run(monkeypatch, confounded=True)
    monkeypatch.setattr(rna_tuning, "run_agent_sync", assess)
    action = run.review("full", 0, selected, {})
    assert action.action == "accept"
    assert action.correctionNeed == "notApplicable"
    assert not selected.parameters.useHarmony


@pytest.mark.parametrize("correction_need", ["needed", "notNeeded"])
def test_historical_confounded_experiment_replays_with_a_scientific_warning(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    correction_need: str,
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = _run(monkeypatch, confounded=True)
    monkeypatch.setattr(
        rna_tuning,
        "run_agent_sync",
        lambda **kwargs: _experiment_response(kwargs, "notApplicable"),
    )
    run.review("full", 0, selected, {})
    key = "parameter_tuning/full/review0"
    # Represent a committed action produced before this scientific guard existed.
    saved[key]["outputs"]["action"]["correctionNeed"] = correction_need
    expected = rna_tuning.TuningAction.model_validate(saved[key]["outputs"]["action"])
    before = json.dumps(saved[key], sort_keys=True)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "A committed experiment must replay without new model or support work"
        )

    monkeypatch.setattr(rna_tuning, "run_agent_sync", unexpected)
    monkeypatch.setattr(rna_tuning, "population_support_evidence", unexpected)
    assert run.review("full", 0, selected, {}) == expected
    assert json.dumps(saved[key], sort_keys=True) == before
    assert any(
        "claimed identifiable correction necessity" in row.get("reason", "")
        for row in run.history
    )


def _alternative(
    run: rna_tuning.RnaTuningRun,
    selected: ParameterCandidateEvaluation,
    *,
    difference: str = "oneParameter",
) -> tuple[ParameterCandidateEvaluation, str]:
    offered = run.experiments(selected)
    experiment_id, experiment = next(
        (name, value)
        for name, value in offered.items()
        if value["parameter"] == "dimensions"
    )
    alternative = selected.model_copy(deep=True)
    alternative.candidateId = "alternative"
    alternative.parameters.candidateId = alternative.candidateId
    alternative.parameters.dimensions = experiment["value"]
    alternative.evidenceIds = ["candidate:alternative:seedStability"]
    if difference == "twoParameters":
        alternative.parameters.leidenResolution += 0.25
    elif difference == "selection":
        assert alternative.cellSelection is not None
        alternative.cellSelection = alternative.cellSelection.model_copy(
            update={"artifactId": "f" * 64}
        )
    elif difference == "reductionMethod":
        alternative.parameters.reductionMethod = "lsi"
    setting = run.settings[selected.candidateId].model_copy(
        update={"parameters": alternative.parameters}
    )
    if difference == "features":
        setting.features = setting.features.model_copy(update={"artifactId": "e" * 64})
    run.settings[alternative.candidateId] = setting
    run.evaluations["full"].append(alternative)
    return alternative, experiment_id


@pytest.mark.parametrize(
    "difference",
    ["oneParameter", "twoParameters", "selection", "features", "reductionMethod"],
)
def test_matched_comparisons_require_one_change_on_the_exact_representation(
    monkeypatch: pytest.MonkeyPatch, difference: str
) -> None:
    run, selected = _run(monkeypatch)
    alternative, experiment_id = _alternative(run, selected, difference=difference)
    observed: dict[str, Any] = {}

    def record(**kwargs: Any) -> Any:
        observed.update(json.loads(kwargs["user_prompt"]))
        return assess(**kwargs)

    monkeypatch.setattr(rna_tuning, "run_agent_sync", record)
    run.review("full", 0, selected, {})
    context = observed["assessmentContext"]
    if difference == "oneParameter":
        assert (
            context["alreadyEvaluatedExperiments"][experiment_id]
            == alternative.candidateId
        )
        assert context["matchedComparisons"] == [
            {
                "currentCandidateId": selected.candidateId,
                "alternativeCandidateId": alternative.candidateId,
                "changedParameter": {
                    "pca": {
                        "current": selected.parameters.dimensions,
                        "alternative": alternative.parameters.dimensions,
                    }
                },
                "partitionEvidence": context["matchedComparisons"][0][
                    "partitionEvidence"
                ],
                "basis": context["matchedComparisons"][0]["basis"],
            }
        ]
        partition = context["matchedComparisons"][0]["partitionEvidence"]
        assert partition["matchedCells"] == 100
        assert all(
            row["fractionOutsideLargestMatch"] == 0.0 for row in partition["splits"]
        )
        assert experiment_id not in observed["experiments"]
    else:
        assert not context["matchedComparisons"]
        assert not context["alreadyEvaluatedExperiments"]
        assert experiment_id in observed["experiments"]


def test_completed_numeric_comparison_remains_selectable_without_reexecution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, selected = _run(monkeypatch)
    alternative, experiment_id = _alternative(run, selected)

    def choose_alternative(**kwargs: Any) -> Any:
        evidence = json.loads(kwargs["user_prompt"])
        assert experiment_id not in evidence["experiments"]
        assert alternative.candidateId in {
            row["candidateId"] for row in evidence["candidates"]
        }
        action = assess(**{**kwargs, "output_validator": lambda value: value}).output
        action = action.model_copy(
            update={
                "selectedCandidateId": alternative.candidateId,
                "evidenceIds": [f"candidate:{alternative.candidateId}"],
            }
        )
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", choose_alternative)
    assert (
        run.review("full", 0, selected, {}).selectedCandidateId
        == alternative.candidateId
    )


def test_committed_catalogue_replays_without_new_filtering_or_population_work(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    saved = request.getfixturevalue("memory_checkpoints")
    run, selected = _run(monkeypatch)
    _alternative_candidate, experiment_id = _alternative(run, selected)
    old_experiment = run.experiments(selected)[experiment_id]
    support_calls = []

    def support(_store: Any, candidate: Any, columns: Any) -> dict[str, Any]:
        support_calls.append(candidate.candidateId)
        return {"candidateId": candidate.candidateId, "columns": list(columns)}

    monkeypatch.setattr(rna_tuning, "population_support_evidence", support)
    monkeypatch.setattr(rna_tuning, "run_agent_sync", assess)
    expected = run.review("full", 0, selected, {})
    key = "parameter_tuning/full/review0"
    assert len(run.evaluations["full"]) == 2
    assert support_calls == [item.candidateId for item in run.evaluations["full"]]
    assert set(saved[key]["inputs"]["assessmentContext"]["populationSupport"]) == {
        item.candidateId for item in run.evaluations["full"]
    }
    # Prior reviews could offer an already-completed comparison and lacked this context.
    saved[key]["inputs"]["experiments"][experiment_id] = old_experiment
    del saved[key]["inputs"]["assessmentContext"]
    saved[key]["inputs"]["availableEvidenceIds"].remove("assessmentContext")
    before = json.dumps(saved[key], sort_keys=True)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Committed catalogues and support evidence must not be rebuilt")

    monkeypatch.setattr(run, "experiments", unexpected)
    monkeypatch.setattr(rna_tuning, "run_agent_sync", unexpected)
    monkeypatch.setattr(rna_tuning, "population_support_evidence", unexpected)
    assert run.review("full", 0, selected, {}) == expected
    assert json.dumps(saved[key], sort_keys=True) == before


@pytest.mark.parametrize("already_in_study", [False, True])
def test_saved_unsupported_design_evidence_reaches_assessment_and_report_once(
    monkeypatch: pytest.MonkeyPatch, already_in_study: bool
) -> None:
    run, selected = _run(monkeypatch)
    comparison = CovariateComparison(
        proposal=CovariateProposal(
            response="mitochondrialFraction",
            explanatoryColumns=["tissue"],
            observationUnit="sample",
            independentUnit="donor",
            rationale="Assess tissue-associated quality differences across donors.",
        ),
        status="unsupported",
        evidence={"independentUnits": 3, "missingCells": 17},
        reasons=["withinIndependentUnitComparisonsAreUnsupported"],
        evidenceId="designComparison:observedUnsupported",
    )
    run.design_comparisons = (comparison,)
    limitation = unsupported_comparison_limitations([comparison])[0]
    run.study = run.study.model_copy(
        update={
            "evidenceIds": [],
            "limitations": [limitation] if already_in_study else [],
        }
    )
    original_study = run.study.model_dump_json()

    def defer_with_observed_design_evidence(**kwargs: Any) -> Any:
        evidence = json.loads(kwargs["user_prompt"])
        assert evidence["assessmentContext"]["designComparisons"] == [
            comparison.model_dump(mode="json")
        ]
        assert comparison.evidenceId in evidence["availableEvidenceIds"]
        action = assess(**{**kwargs, "output_validator": lambda value: value}).output
        action = action.model_copy(
            update={
                "action": "defer",
                "evidenceIds": [
                    f"candidate:{selected.candidateId}",
                    comparison.evidenceId,
                ],
                "rationale": "Essential design evidence remains unsupported.",
            }
        )
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(
        rna_tuning, "run_agent_sync", defer_with_observed_design_evidence
    )
    assert run.review("full", 0, selected, {}).action == "defer"
    report = run.report(None, "Essential design evidence remains unsupported.")
    assert report.limitations.count(limitation) == 1
    assert comparison.evidenceId in limitation
    assert comparison.reasons[0] in limitation
    assert "no supported association or absence finding" in limitation
    assert run.study.model_dump_json() == original_study


@pytest.mark.parametrize("license", ["safe", "unsafeConfounded"])
def test_uncertain_correction_can_request_harmony_only_with_safe_design(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    license: str,
) -> None:
    run, selected = _run(monkeypatch, confounded=license == "unsafeConfounded")
    run.study = run.study.model_copy(
        update={"correctionLicense": license, "technicalBatchColumns": ["batch"]}
    )
    run.batch_columns = ["batch"]

    def request_harmony(**kwargs: Any) -> Any:
        evidence = json.loads(kwargs["user_prompt"])
        assert ("useHarmony:true" in evidence["experiments"]) is (license == "safe")
        action = assess(**{**kwargs, "output_validator": lambda value: value}).output
        action = action.model_copy(
            update={
                "action": "experiment",
                "experimentId": "useHarmony:true",
                "correctionNeed": "uncertain",
                "concern": "Determine whether observed batch structure is removable.",
                "expectedImprovement": "Compare mixing and protected biology against matched native settings.",
                "rationale": "Request a matched native and Harmony comparison.",
            }
        )
        return SimpleNamespace(output=kwargs["output_validator"](action))

    monkeypatch.setattr(rna_tuning, "run_agent_sync", request_harmony)
    if license == "safe":
        action = run.review("full", 0, selected, {})
        assert action.action == "experiment"
        assert action.experimentId == "useHarmony:true"
        assert action.correctionNeed == "uncertain"
    else:
        with pytest.raises(ValueError, match="Unknown experiment ID"):
            run.review("full", 0, selected, {})
        saved = request.getfixturevalue("memory_checkpoints")
        assert "parameter_tuning/full/review0" not in saved
