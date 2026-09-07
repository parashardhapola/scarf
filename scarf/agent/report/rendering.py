"""A single readable analysis page, using only recorded scientific evidence."""

import html
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import label, mapping, mappings, scalar, texts


_STYLES = """
:root{color-scheme:light;font:16px/1.6 system-ui,sans-serif;color:#223137;background:#f4f6f5}
*{box-sizing:border-box}body{margin:0}main{max-width:1060px;margin:auto;padding:36px 28px 64px}
header{border-bottom:2px solid #237e6a;padding-bottom:22px}h1,h2,h3{line-height:1.25;color:#164c40}
h1{font-size:2.2rem;margin:8px 0}h2{font-size:1.4rem;margin-top:36px}h3{font-size:1.05rem}
p{max-width:90ch}a{color:#17644f}small,.muted{color:#586763}.numbers{font-size:1.25rem;font-weight:600}
figure{margin:24px 0;background:white;padding:12px;border-radius:8px}figure img{width:100%;height:auto}
figcaption{font-size:.9rem;text-align:center}.decision{border-top:1px solid #ccd7d1;padding:14px 0}
.decision h3{margin:0}.decision p{margin:8px 0}details{margin:12px 0}summary{cursor:pointer;color:#17644f}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.92rem;margin:14px 0}
th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid #d9e0dc}
th{background:#e9efeb}td p{margin:0}li{margin:6px 0}
footer{margin-top:36px;border-top:1px solid #ccd7d1;padding-top:18px;font-size:.85rem}
@media(max-width:600px){main{padding:20px 14px}h1{font-size:1.7rem}th,td{padding:7px}}
@media print{body{background:white}main{padding:0}details{break-inside:avoid}}
"""


def _escape(value: Any) -> str:
    return html.escape(scalar(value))


def _list(items: Sequence[str]) -> str:
    return (
        "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"
        if items
        else ""
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _decision(decision: Mapping[str, Any]) -> str:
    spec, record = mapping(decision.get("spec")), mapping(decision.get("record"))
    options = mappings(spec.get("options"))
    selected = next(
        (
            option
            for option in options
            if option.get("optionId") == record.get("selectedOptionId")
        ),
        {},
    )
    question = str(
        spec.get("question") or label(str(record.get("decisionId", "Analysis setting")))
    )
    chosen = str(selected.get("label") or "Selection unavailable")
    rationale = str(record.get("rationale") or "No rationale was recorded.")
    source = {
        "agent": "Agent choice",
        "rule": "Scarf rule",
        "human": "User choice",
    }.get(str(record.get("source", "")), "")
    alternatives = _table(
        ("Option", "Action", "Description"),
        [
            [
                option.get("label"),
                "Selected" if option is selected else "Not selected",
                option.get("description"),
            ]
            for option in options
        ],
    )
    evidence = mappings(mapping(decision.get("evidence")).get("evidence"))
    evidence_markup = _list(
        [str(item["summary"]) for item in evidence if item.get("summary")]
    )
    checks = mappings(decision.get("checks"))
    check_markup = _table(
        ("Check", "Outcome", "Finding"),
        [
            [
                item.get("label", item.get("name")),
                item.get("status"),
                item.get("reason", item.get("summary")),
            ]
            for item in checks
        ],
    )
    return f"""<article class="decision">
<h3>{html.escape(question)}</h3>
<p><strong>{html.escape(chosen)}.</strong> {html.escape(rationale)}</p>
{"<small>" + source + "</small>" if source else ""}
<details><summary>Alternatives and supporting evidence</summary>{alternatives}{evidence_markup}{check_markup}</details>
</article>"""


def render_analysis_document(payload: Mapping[str, Any]) -> str:
    final = mapping(payload.get("finalAnalysis"))
    request = mapping(payload.get("request"))
    counts = mapping(payload.get("clusterCounts"))
    total = sum(int(value) for value in counts.values())
    context = request.get("studyObjective") or request.get("studyContext") or ""
    assay = final.get("primaryAssay") or request.get("primaryAssay") or "RNA"
    qc = mapping(payload.get("qc"))
    qc_text = ""
    if isinstance(qc.get("retainedCells"), int) and isinstance(
        qc.get("retainedFraction"), int | float
    ):
        qc_text = f"<p>QC retained {_escape(qc['retainedCells'])} cells ({float(qc['retainedFraction']):.1%}).</p>"
    map_markup = ""
    if payload.get("umap"):
        display = int(payload.get("displayedCells") or total)
        map_markup = f'<figure><img src="{html.escape(str(payload["umap"]), quote=True)}" alt="Final UMAP colored by saved cluster labels"><figcaption>{display:,} of {total:,} cells shown. Counts and marker statistics use the complete selection.</figcaption></figure>'
    decisions = "".join(_decision(item) for item in mappings(payload.get("decisions")))
    assessments = mappings(payload.get("assessments"))
    for assessment in assessments:
        scope = (
            "Full cohort" if assessment.get("scope") == "full" else "Screening sample"
        )
        alternatives = mappings(assessment.get("candidates"))
        settings = mapping(assessment.get("settings"))
        comparison = _table(
            (
                "Resolution",
                "PCA dimensions",
                "Neighbors",
                "HVGs",
                "HVG ranking",
                "Harmony",
                "Clusters",
                "Seed stability",
                "Marker coherence",
                "Selected",
            ),
            [
                [
                    mapping(item.get("parameters")).get("leidenResolution"),
                    mapping(item.get("parameters")).get("dimensions"),
                    mapping(item.get("parameters")).get("neighborsK"),
                    mapping(settings.get(str(item.get("candidateId")))).get("hvgCount"),
                    mapping(settings.get(str(item.get("candidateId")))).get("ranking"),
                    mapping(item.get("parameters")).get("useHarmony"),
                    mapping(item.get("metrics")).get("nClusters"),
                    mapping(item.get("metrics")).get("seedStability"),
                    mapping(item.get("metrics")).get("markerCoherence"),
                    item.get("candidateId") == assessment.get("selectedCandidateId"),
                ]
                for item in alternatives
            ],
        )
        assessment_findings = texts(assessment.get("quantitativeFindings")) + texts(
            assessment.get("qualitativeFindings")
        )
        details = _list(assessment_findings)
        if assessment.get("evidenceMode") == "structured":
            details = (
                "<p>The model assessed structured loading, marker and diagnostic evidence. "
                "No plots were supplied for visual inspection.</p>" + details
            )
        rationale = html.escape(str(assessment.get("rationale", "")))
        action = {
            "accept": "Accepted settings",
            "experiment": "Selected a targeted experiment",
            "enlarge": "Requested more cells",
            "defer": "Required more evidence",
        }.get(str(assessment.get("action", "")), "Analysis assessment")
        protection = html.escape(str(assessment.get("objectivePreservation", "")))
        experiment = ""
        if assessment.get("experimentId"):
            experiment = (
                f"<p><strong>Experiment: {_escape(assessment['experimentId'])}</strong></p>"
                f"<p>Observed concern: {_escape(assessment.get('concern'))}</p>"
                f"<p>Expected improvement: {_escape(assessment.get('expectedImprovement'))}</p>"
            )
        correction = assessment.get("correctionNeed")
        correction_text = (
            f"<p>Correction necessity: {_escape(label(str(correction)))}.</p>"
            if correction
            else ""
        )
        decisions += f'<article class="decision"><h3>{scope}: {action}</h3><p>{rationale}</p>{experiment}<details><summary>Compared settings and evidence</summary>{comparison}{details}{correction_text}<p>{protection}</p></details></article>'
    if not decisions:
        decisions = "<p>No consequential decisions were recorded.</p>"
    findings = _list(texts(payload.get("findings")))
    marker_rows = mappings(payload.get("markers"))
    cluster_table = _table(
        ("Cluster", "Cells", "Top marker genes"),
        [
            [
                cluster,
                count,
                ", ".join(
                    str(row["feature"])
                    for row in marker_rows
                    if row.get("cluster") == cluster
                )
                or "No markers passed the saved-table filters",
            ]
            for cluster, count in counts.items()
        ],
    )
    parameters = mapping(payload.get("selectedParameters"))
    metrics = mapping(payload.get("selectedMetrics"))
    selected_setting = mapping(payload.get("selectedSetting"))
    selected_features = mapping(payload.get("selectedFeatures"))
    methods = _table(
        ("Setting", "Selected value"),
        [
            [name, parameters[key]]
            for key, name in (
                ("dimensions", "PCA dimensions"),
                ("neighborsK", "Neighbors"),
                ("leidenResolution", "Clustering resolution"),
                ("useHarmony", "Harmony correction"),
            )
            if key in parameters
        ],
    )
    methods += _table(
        ("Gene selection", "Selected value"),
        [
            [name, selected_setting[key]]
            for key, name in (
                ("hvgCount", "HVG count"),
                ("ranking", "HVG ranking"),
                ("rankingColumn", "Ranking technical column"),
            )
            if selected_setting.get(key) is not None
        ],
    )
    methods += _table(
        ("Feature family", "Eligible genes", "Selected HVGs"),
        [
            [
                label(name),
                mapping(values).get("eligibleGenes"),
                mapping(values).get("selectedGenes"),
            ]
            for name, values in mapping(selected_features.get("families")).items()
        ],
    )
    measurements = _table(
        ("Measure", "Recorded value"),
        [
            [name, metrics[key]]
            for key, name in (
                ("seedStability", "Clustering stability across seeds"),
                ("subsampleStability", "Clustering stability across subsamples"),
                ("markerCoherence", "Marker coherence"),
                ("crossUnitSupport", "Support across study units"),
                ("minClusterCells", "Smallest cluster"),
            )
            if key in metrics and metrics[key] is not None
        ],
    )
    limitations = _list(texts(payload.get("limitations")))
    display_notes = _list(texts(payload.get("displayNotes")))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scarf analysis summary</title><style>{_STYLES}</style></head><body><main>
<header><small>Scarf analysis</small><h1>Analysis summary</h1><p>{_escape(context)}</p>
<p class="numbers">{total:,} cells · {len(counts):,} clusters · {_escape(assay)}</p>{qc_text}</header>
{map_markup}<section><h2>Analysis decisions</h2>{decisions}</section>
{"<section><h2>What the evidence shows</h2>" + findings + "</section>" if findings else ""}
<section><h2>Clusters and markers</h2>{cluster_table}<p class="muted">Marker genes describe the saved clusters; they do not establish cell identities.</p></section>
{"<section><h2>Limitations</h2>" + limitations + "</section>" if limitations else ""}
<details><summary>Selected methods and measurements</summary>{methods}{measurements}<p>All measurements and explanations are read from the completed analysis. The map uses saved coordinates and a bounded display sample. No analysis or model calls run when this report is generated.</p></details>
{"<details><summary>Unavailable displays</summary>" + display_notes + "</details>" if display_notes else ""}
<footer>Generated locally by <a href="https://scarf.readthedocs.io/">Scarf</a>.</footer>
</main></body></html>"""
