"""Structured tradeoff repair keeps scientific counterevidence explicit."""

from copy import deepcopy
import json
from typing import Any

from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
import pytest

from scarf.agent.config import AgentRunConfig
from scarf.agent.config.agent_exec import run_agent_sync
from scarf.agent.orchestrator import rna_tuning
from scarf.agent.parameter_tuning.comparisons import (
    bind_comparison_measurements,
    comparison_advantages,
    validate_comparison_review,
)
from tests.agent_comparison_examples import comparison_review, observed_action


def _competing_review() -> tuple[dict[str, Any], dict[str, Any]]:
    review = comparison_review()
    coverage = review["comparisonCoverage"]
    settings = coverage["candidateSettings"]
    for candidate in settings.values():
        candidate["metrics"]["macroF1"] = 0.8
    for identifier in (
        "resolution-half",
        "genes-two",
        "dimensions-thirty",
        "neighbors-forty-one",
    ):
        settings[identifier]["metrics"].update(seedStability=0.97, macroF1=0.8 + 1e-10)
    review["currentCandidateId"] = review["selectedCandidateId"]
    action = observed_action(review)
    for conclusion in action["comparisonConclusions"]:
        explanations = [
            f"{row['alternativeCandidateId']} improves {row['metric']} from "
            f"{row['preferredValue']!r} to {row['alternativeValue']!r}; "
            "the reference marker program still motivates this preference."
            for row in conclusion["tradeoffs"]
        ]
        conclusion["quantitativeReason"] = " ".join(explanations) or (
            "No measured advantage requires explanation on this axis."
        )
        conclusion["biologicalReason"] = (
            "Retain the reference marker program while acknowledging every "
            "alternative's measured improvements in the quantitative explanation."
        )
        conclusion["tradeoffs"] = []
    return coverage, action


def test_provider_requires_explicit_tradeoff_field_without_numeric_authorship() -> None:
    coverage, action = _competing_review()
    output_type = rna_tuning._assessment_output_type(
        list(coverage["candidateSettings"]), [], scope="full"
    )
    schema = output_type.model_json_schema()
    conclusion_schema = schema["$defs"]["ObservedComparisonInterpretation"]
    assert "tradeoffs" in conclusion_schema["required"]
    assert (
        "comparisonAdvantages"
        in conclusion_schema["properties"]["tradeoffs"]["description"]
    )
    tradeoff_fields = schema["$defs"]["ObservedTradeoffInterpretation"]["properties"]
    assert set(tradeoff_fields) == {
        "alternativeCandidateId",
        "metric",
        "interpretation",
    }
    del action["comparisonConclusions"][0]["tradeoffs"]
    with pytest.raises(ValidationError, match="tradeoffs"):
        output_type.model_validate(action)
    # The provider contract does not rewrite the existing persisted action schema.
    saved = rna_tuning.TuningAction.model_validate(action)
    assert saved.comparisonConclusions[0].tradeoffs == []


def test_detailed_prose_cannot_replace_structured_tradeoff_explanations() -> None:
    coverage, action = _competing_review()
    original = deepcopy(action)
    with pytest.raises(ValueError) as failure:
        validate_comparison_review(coverage, action)
    feedback = str(failure.value)
    for axis in ("partition", "hvgCount", "pca", "neighbors"):
        assert axis in feedback
    for required in (
        "comparisonConclusions",
        "tradeoffs",
        "alternativeCandidateId",
        "metric",
        "interpretation",
        "quantitativeReason",
        "biologicalReason",
        "resolution-half",
        "genes-two",
        "dimensions-thirty",
        "neighbors-forty-one",
        "0.8000000001",
    ):
        assert required in feedback
    assert action == original


@pytest.mark.parametrize("repair", [True, False])
def test_model_repairs_exact_fields_or_fails_without_accepting_prose(
    repair: bool,
) -> None:
    coverage, action = _competing_review()
    inventory = comparison_advantages(coverage)
    offered = json.dumps({"comparisonAdvantages": inventory}, sort_keys=True)
    output_type = rna_tuning._assessment_output_type(
        list(coverage["candidateSettings"]), [], scope="full"
    )
    responses = []
    attempts = []

    def provider(messages: Any, info: Any) -> ModelResponse:
        proposed = deepcopy(action)
        if responses:
            feedback = " ".join(
                str(part.content)
                for message in messages
                for part in message.parts
                if isinstance(part, RetryPromptPart)
            )
            for name in (
                "comparisonConclusions",
                "tradeoffs",
                "alternativeCandidateId",
                "metric",
                "interpretation",
            ):
                assert name in feedback
            if repair:
                for conclusion in proposed["comparisonConclusions"]:
                    conclusion["tradeoffs"] = [
                        {
                            "alternativeCandidateId": row["alternativeCandidateId"],
                            "metric": row["metric"],
                            "interpretation": (
                                f"The observed {row['metric']} advantage must be "
                                "weighed against preserving the reference marker "
                                "program, rather than asserting numerical superiority."
                            ),
                        }
                        for row in inventory
                        if row["axis"] == conclusion["axis"]
                        and row["preferredCandidateId"]
                        == conclusion["preferredCandidateId"]
                    ]
        responses.append(proposed)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, proposed)])

    def validate(proposed: Any) -> rna_tuning.TuningAction:
        bound = bind_comparison_measurements(coverage, proposed.model_dump(mode="json"))
        validate_comparison_review(coverage, bound)
        return rna_tuning.TuningAction.model_validate(bound)

    def run():
        return run_agent_sync(
            model=FunctionModel(provider),
            output_type=output_type,
            system_prompt="Adjudicate the supplied measured tradeoffs.",
            user_prompt=offered,
            config=AgentRunConfig(retries=1),
            output_validator=validate,
            on_attempt=attempts.append,
        )

    if repair:
        result = run()
        completed = result.output.model_dump(mode="json")
        validate_comparison_review(coverage, completed)
        measured = {
            (
                row["axis"],
                row["preferredCandidateId"],
                row["alternativeCandidateId"],
                row["metric"],
            ): row
            for row in inventory
        }
        checked = set()
        for conclusion in completed["comparisonConclusions"]:
            for tradeoff in conclusion["tradeoffs"]:
                expected = measured[
                    (
                        conclusion["axis"],
                        conclusion["preferredCandidateId"],
                        tradeoff["alternativeCandidateId"],
                        tradeoff["metric"],
                    )
                ]
                assert tradeoff["preferredValue"] == expected["preferredValue"]
                assert tradeoff["alternativeValue"] == expected["alternativeValue"]
                checked.add(conclusion["axis"])
        assert checked == {"partition", "hvgCount", "pca", "neighbors"}
        assert attempts[0].status == "done"
        assert len(attempts[0].validationRetries) == 1
    else:
        with pytest.raises(UnexpectedModelBehavior, match="maximum output retries"):
            run()
        assert attempts[0].status == "failed"
        assert len(attempts[0].validationRetries) == 2
    assert len(responses) == 2
    assert len(attempts) == 1
    assert attempts[0].usage.requests == 2
    assert json.dumps({"comparisonAdvantages": inventory}, sort_keys=True) == offered
    assert all(
        "preferredValue" not in tradeoff and "alternativeValue" not in tradeoff
        for response in responses
        for conclusion in response["comparisonConclusions"]
        for tradeoff in conclusion["tradeoffs"]
    )
