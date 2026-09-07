"""One faithful, read-only analysis report from the authoritative journal."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from scarf.agent.orchestrator import journal
from scarf.agent.report import generator, plots
from scarf.agent.report.artifacts import (
    _local_root,
    report_directory,
    scientific_summary,
)
from scarf.agent.report.rendering import render_analysis_document
from scarf.agent.types import ArtifactReferenceModel
from tests.test_agent_analysis_plots import display_store


def snapshot() -> dict[str, Any]:
    state = {
        "runId": "exact-analysis",
        "status": "completed",
        "request": {
            "studyContext": "Human RNA cells",
            "studyObjective": "Find stable populations <without> batch artifacts.",
        },
        "finalAnalysis": {
            "primaryAssay": "RNA2",
            "limitations": ["Condition and batch are confounded."],
            "analysisEvidence": {
                "analysisReview": {
                    "tuningEvidence": {
                        "history": [
                            {
                                "scope": "full",
                                "review": {
                                    "action": "accept",
                                    "selectedCandidateId": "candidate-two",
                                    "quantitativeFindings": [
                                        "The chosen partition has seed stability 0.92."
                                    ],
                                    "qualitativeFindings": [
                                        "MS4A1 and CD79A support the same partition."
                                    ],
                                    "rationale": "The selected partition preserves a small marker-supported population.",
                                    "objectivePreservation": "Retain the rare marker program.",
                                },
                            }
                        ]
                    }
                }
            },
        },
        "stages": [
            {
                "stage": "parameter_tuning",
                "status": "done",
                "report": {
                    "recommendedCandidateId": "candidate-two",
                    "evaluations": [
                        {
                            "candidateId": "candidate-two",
                            "parameters": {
                                "dimensions": 20,
                                "neighborsK": 15,
                                "leidenResolution": 0.75,
                                "useHarmony": False,
                            },
                            "metrics": {"seedStability": 0.92, "markerCoherence": 0.84},
                        }
                    ],
                },
                "decisions": [
                    {
                        "spec": {
                            "question": "Which clustering resolution preserves supported populations?",
                            "options": [
                                {
                                    "optionId": "low",
                                    "label": "Resolution 0.5",
                                    "description": "Compare the coarser partition.",
                                },
                                {
                                    "optionId": "chosen",
                                    "label": "Resolution 0.75",
                                    "description": "Compare the marker-supported partition.",
                                },
                            ],
                        },
                        "record": {
                            "selectedOptionId": "chosen",
                            "decisionId": "clustering",
                            "rationale": "Selected 0.75 because seed stability was 0.92 and B-cell markers remained coherent.",
                            "modelName": "not-in-report",
                            "recordId": "hidden-record-id",
                        },
                        "evidence": {
                            "evidence": [
                                {
                                    "summary": "Resolution 0.5: stability 0.96; marker coherence 0.67."
                                },
                                {
                                    "summary": "Resolution 0.75: stability 0.92; marker coherence 0.84."
                                },
                            ]
                        },
                        "checks": [
                            {
                                "name": "Protected condition",
                                "status": "passed",
                                "reason": "Condition representation was retained.",
                            }
                        ],
                    }
                ],
            },
            {
                "stage": "experimental_context",
                "report": {
                    "cellQc": {"profileId": "selected"},
                    "qcProfiles": [
                        {
                            "profileId": "selected",
                            "retainedCells": 920,
                            "retainedFraction": 0.92,
                        }
                    ],
                },
            },
        ],
    }

    review = state["finalAnalysis"]["analysisEvidence"]["analysisReview"][
        "tuningEvidence"
    ]["history"][0]["review"]
    state["analysisReviews"] = [
        {
            "scope": "full",
            **review,
            "candidates": state["stages"][0]["report"]["evaluations"],
            "settings": {"candidate-two": {"hvgCount": 2000, "ranking": "global"}},
            "featureEvidence": {
                "candidate-two": {
                    "families": {"hla": {"eligibleGenes": 12, "selectedGenes": 6}}
                }
            },
        }
    ]
    return state


def display_payload() -> dict[str, Any]:
    return {
        "clusterCounts": {"0": 620_000, "1": 1_200},
        "markers": [{"cluster": "1", "feature": "MS4A1", "score": 0.84}],
        "umap": "plots/final_umap.png",
        "displayedCells": 50_000,
        "displayNotes": [],
    }


@pytest.mark.parametrize("mode", ["visual", "structured"])
def test_one_page_shows_recorded_choices_evidence_and_qualitative_findings(
    mode,
) -> None:
    state = snapshot()
    state["analysisReviews"][0]["evidenceMode"] = mode
    original = copy.deepcopy(state)
    payload = scientific_summary(state) | display_payload()
    document = render_analysis_document(payload)
    assert state == original
    assert "621,200 cells" in document and "2 clusters" in document
    assert "50,000 of 621,200" in document
    assert "QC retained 920 cells (92.0%)" in document
    assert "Selected 0.75 because seed stability was 0.92" in document
    assert "Resolution 0.5: stability 0.96; marker coherence 0.67." in document
    assert "MS4A1 and CD79A support the same partition." in document
    assert "small marker-supported population" in document
    assert "Condition and batch are confounded." in document
    assert "Condition representation was retained." in document
    assert "&lt;without&gt;" in document and "<without>" not in document
    assert "hidden-record-id" not in document and "not-in-report" not in document
    assert "technical.html" not in document and "decision-tree" not in document
    assert "added noise" not in document and "weaker" not in document
    assert ("No plots were supplied for visual inspection" in document) == (
        mode == "structured"
    )
    assert document.index("final_umap.png") < document.index("Analysis decisions")


def test_all_untrusted_scientific_text_is_escaped() -> None:
    state = snapshot()
    injection = '<img src=x onerror="alert(1)">'
    state["stages"][0]["decisions"][0]["record"]["rationale"] = injection
    state["finalAnalysis"]["limitations"] = [injection]
    payload = scientific_summary(state) | display_payload()
    payload["markers"][0]["feature"] = injection
    document = render_analysis_document(payload)
    assert injection not in document
    assert document.count("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;") == 3


def test_missing_evidence_is_reported_without_inventing_selection_reasons() -> None:
    state = snapshot()
    state["stages"] = []
    state["finalAnalysis"]["analysisEvidence"] = {}
    state["analysisReviews"] = []
    document = render_analysis_document(scientific_summary(state) | display_payload())
    assert "No consequential decisions were recorded" in document
    assert "What the evidence shows" not in document
    assert "seed stability was 0.92" not in document


def test_report_distinguishes_gene_correction_and_experiment_evidence() -> None:
    state = snapshot()
    accepted = state["analysisReviews"][0]
    accepted["correctionNeed"] = "needed"
    accepted["candidates"][0]["parameters"]["useHarmony"] = True
    accepted["settings"]["candidate-two"].update(
        ranking="batchAware", rankingColumn="library"
    )
    experiment = copy.deepcopy(accepted)
    experiment.update(
        action="experiment",
        experimentId="includeFamily:hla",
        concern="HLA markers distinguish the objective-relevant activation state.",
        expectedImprovement="Restoring HLA genes may retain that state.",
    )
    state["analysisReviews"].insert(0, experiment)
    document = render_analysis_document(scientific_summary(state) | display_payload())
    assert "includeFamily:hla" in document
    assert experiment["concern"] in document
    assert experiment["expectedImprovement"] in document
    assert "HVG count" in document and "2,000" in document
    assert "batchAware" in document and "library" in document
    assert "Feature family" in document and "Selected HVGs" in document
    assert "Correction necessity: Needed" in document


@pytest.mark.parametrize("mode", ["visual", "structured"])
@pytest.mark.parametrize("damage", [None, "digest", "scope", "action", "genes", "mode"])
def test_review_view_requires_exact_checkpoint_bindings(
    monkeypatch: pytest.MonkeyPatch, damage: str | None, mode: str
) -> None:
    import hashlib

    from scarf.agent import record_io
    from scarf.agent.orchestrator.models import AutomatedWorkflowConfig

    state = snapshot()
    view = state["analysisReviews"][0]
    candidate = copy.deepcopy(view["candidates"][0])
    features = ArtifactReferenceModel(
        assay="RNA2", kind="feature_selection", artifactId="4" * 64
    ).model_dump(mode="json")
    candidate["artifacts"] = {"graphFeatures": features}
    action = {
        key: value
        for key, value in view.items()
        if key not in {"scope", "candidates", "settings", "featureEvidence"}
    }
    payload = {
        "inputs": {
            "scope": "full",
            "imageHashes": {"observed": "image-digest"} if mode == "visual" else {},
            "evidenceMode": mode,
            "visualInspection": "available" if mode == "visual" else "unavailable",
            "candidates": [candidate],
            "settings": {
                "candidate-two": {
                    **view["settings"]["candidate-two"],
                    "parameters": candidate["parameters"],
                    "features": features,
                }
            },
            "featureEvidence": view["featureEvidence"],
        },
        "outputs": {"action": action},
    }
    digest = hashlib.sha256(record_io.canonical_json_bytes(payload)).hexdigest()
    entry = {
        "scope": "full",
        "review": action,
        "checkpointKey": "parameter_tuning/full/review0",
        "checkpointSha256": digest,
        "imageHashes": payload["inputs"]["imageHashes"],
        "evidenceMode": mode,
        "visualInspection": payload["inputs"]["visualInspection"],
    }
    if damage == "digest":
        entry["checkpointSha256"] = "bad-digest"
    elif damage == "scope":
        entry["scope"] = "sample0"
    elif damage == "action":
        entry["review"] = {**action, "rationale": "Unrecorded reasoning"}
    elif damage == "mode":
        entry["evidenceMode"] = "structured" if mode == "visual" else "visual"
    elif damage == "genes":
        payload["inputs"]["settings"]["candidate-two"]["features"] = {
            **features,
            "artifactId": "5" * 64,
        }
        digest = hashlib.sha256(record_io.canonical_json_bytes(payload)).hexdigest()
        entry["checkpointSha256"] = digest
    record = record_io.canonical_json_bytes({**payload, "contentSha256": digest})
    monkeypatch.setattr(record_io, "read_key", lambda *_args: record)
    stages = [
        {
            "stage": "parameter_tuning",
            "outputs": {"tuningEvidence": {"history": [entry]}},
        }
    ]
    if damage is not None:
        with pytest.raises(ValueError, match="Analysis review"):
            journal._analysis_review_views(
                SimpleNamespace(zw=object()),
                "agents/orchestrations",
                "workflow",
                stages,
                AutomatedWorkflowConfig(),
            )
    else:
        result = journal._analysis_review_views(
            SimpleNamespace(zw=object()),
            "agents/orchestrations",
            "workflow",
            stages,
            AutomatedWorkflowConfig(),
        )
        assert result[0]["rationale"] == action["rationale"]
        assert result[0]["evidenceMode"] == mode
        assert result[0]["settings"]["candidate-two"]["hvgCount"] == 2000
        assert "artifacts" not in result[0]["candidates"][0]


def test_report_regeneration_only_replaces_derived_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "data.zarr"
    root.mkdir()
    frozen = root / "numerical-artifact"
    frozen.write_bytes(b"immutable")
    output = report_directory(root, "exact-analysis", "workspace")
    monkeypatch.setattr(
        generator, "collect_analysis_artifacts", lambda *_: display_payload()
    )
    result = generator.render_analysis_report(SimpleNamespace(), snapshot(), output)
    assert result == output / "index.html"
    assert list(output.glob("*.html")) == [result]
    before = result.read_text()
    assert (
        generator.render_analysis_report(SimpleNamespace(), snapshot(), output)
        == result
    )
    assert result.read_text() == before
    assert frozen.read_bytes() == b"immutable"
    assert not list(output.glob(".*.tmp"))


def test_public_report_opens_exact_journal_and_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Any] = []
    store = SimpleNamespace(workspace="workspace")

    def open_store(target: Path, run_id: str, *, workspace: str | None) -> Any:
        calls.append((target, run_id, workspace))
        return store

    def load_snapshot(target: Any, run_id: str) -> dict[str, Any]:
        assert target is store and run_id == "exact-analysis"
        return snapshot()

    monkeypatch.setattr(journal, "open_analysis_store", open_store, raising=False)
    monkeypatch.setattr(journal, "analysis_snapshot", load_snapshot, raising=False)
    monkeypatch.setattr(
        generator, "collect_analysis_artifacts", lambda *_: display_payload()
    )
    path = generator.generate_agent_report(
        tmp_path, "exact-analysis", workspace="workspace"
    )
    assert calls == [(tmp_path, "exact-analysis", "workspace")]
    assert (
        path
        == tmp_path / "workspace/agents/orchestrations/exact-analysis/report/index.html"
    )


def test_report_does_not_replace_existing_page_after_render_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = tmp_path / "index.html"
    old.write_text("previous report")
    monkeypatch.setattr(
        generator, "collect_analysis_artifacts", lambda *_: display_payload()
    )

    def fail(_payload: Any) -> str:
        raise RuntimeError("render failed")

    monkeypatch.setattr(generator, "render_analysis_document", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        generator.render_analysis_report(SimpleNamespace(), snapshot(), tmp_path)
    assert old.read_text() == "previous report"
    assert not list(tmp_path.glob(".*.tmp"))


def test_report_rejects_remote_paths_escaping_workspaces_and_incomplete_runs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="local filesystem"):
        _local_root("s3://example/data.zarr")
    with pytest.raises(FileNotFoundError):
        _local_root(tmp_path / "missing")
    assert _local_root(f"file://{tmp_path}") == tmp_path
    with pytest.raises(ValueError, match="outside"):
        report_directory(tmp_path, "run", "../outside")
    with pytest.raises(ValueError, match="identifier"):
        report_directory(tmp_path, "../escape", None)
    state = snapshot()
    state["status"] = "needsInput"
    with pytest.raises(ValueError, match="completed"):
        generator.render_analysis_report(SimpleNamespace(), state, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_large_report_reads_saved_map_and_marker_table_without_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, refs, _ = display_store(monkeypatch)
    marker_calls: list[str] = []

    def markers(_ref: Any, *, group_id: str) -> pd.DataFrame:
        marker_calls.append(group_id)
        return pd.DataFrame(
            {
                "feature_name": ["CD79A", "MS4A1", "CD74", "HLA-DRA"],
                "score": [0.7, 0.9, 0.8, 0.6],
            }
        )

    store.get_markers = markers
    inspect = store.inspect_artifact

    def marker_status(ref: Any) -> Any:
        status = inspect(ref)
        if ref.kind == "marker_table":
            status.inputs["clusters"] = refs["clusters"].to_dict()
        return status

    store.inspect_artifact = marker_status
    final = {
        "umap": ArtifactReferenceModel.from_artifact_ref(refs["umap"]).model_dump(),
        "clusters": ArtifactReferenceModel.from_artifact_ref(
            refs["clusters"]
        ).model_dump(),
        "cellSelection": ArtifactReferenceModel.from_artifact_ref(
            refs["cell_selection"]
        ).model_dump(),
        "graph": ArtifactReferenceModel.from_artifact_ref(refs["graph"]).model_dump(),
        "markers": {
            "scope": "assay",
            "assay": "RNA2",
            "kind": "marker_table",
            "artifactId": "5" * 64,
        },
    }
    data = plots.collect_analysis_artifacts(store, final, tmp_path)
    assert data["displayNotes"] == []
    assert data["displayedCells"] == 50_000
    assert sum(data["clusterCounts"].values()) == 621_200
    assert len(data["markers"]) == 12 and len(marker_calls) == 4
    assert data["markers"][0]["feature"] == "MS4A1"
    assert (tmp_path / "plots/final_umap.png").is_file()
    provenance = json.loads((tmp_path / "plots/final_umap.png.json").read_text())
    assert provenance["provenance"]["extras"]["input_n_cells"] == 621_200
    assert list((tmp_path / "plots").glob("*.png")) == [
        tmp_path / "plots/final_umap.png"
    ]


def test_optional_map_failure_preserves_counts_and_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, refs, _ = display_store(monkeypatch, n=12)

    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError("matplotlib unavailable")

    monkeypatch.setattr(plots, "plot_final_umap", unavailable)
    state = snapshot()
    state["finalAnalysis"]["clusters"] = ArtifactReferenceModel.from_artifact_ref(
        refs["clusters"]
    ).model_dump()
    for name, ref_name in (
        ("umap", "umap"),
        ("cellSelection", "cell_selection"),
        ("graph", "graph"),
    ):
        state["finalAnalysis"][name] = ArtifactReferenceModel.from_artifact_ref(
            refs[ref_name]
        ).model_dump()
    path = generator.render_analysis_report(store, state, tmp_path)
    document = path.read_text()
    assert "12 cells" in document
    assert "matplotlib unavailable" in document
    assert "Selected 0.75 because seed stability was 0.92" in document
    assert 'src="plots/final_umap.png"' not in document


def test_invalid_map_lineage_is_not_hidden_as_optional_display_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, refs, _ = display_store(monkeypatch, n=12)
    final = {
        name: ArtifactReferenceModel.from_artifact_ref(refs[ref_name]).model_dump()
        for name, ref_name in (
            ("umap", "umap"),
            ("clusters", "clusters"),
            ("graph", "graph"),
            ("cellSelection", "cell_selection"),
        )
    }

    def mismatch(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("Final artifacts must share the exact frozen cell selection")

    monkeypatch.setattr(plots, "plot_final_umap", mismatch)
    with pytest.raises(ValueError, match="exact frozen cell selection"):
        plots.collect_analysis_artifacts(store, final, tmp_path)
    assert not list(tmp_path.iterdir())
