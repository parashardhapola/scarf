"""HTML sections and templates for agent reports."""

import html
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ... import __version__
from .artifacts import _default_inventory_for_assay
from .contracts import (
    _analysis_number_range,
    _analysis_percent,
    _assay_label,
    _brief_text,
    _feature_family_label,
    _format_text_list,
    _is_leaf,
    _is_mapping_sequence,
    _is_sequence,
    _is_simple,
    _label,
    _latest,
    _mapping,
    _mappings,
    _present,
    _public_field_label,
    _qc_resolved_bounds,
    _scalar,
    _selected_qc_profile,
    _specific_references,
    _text_values,
)
from .decision_tree import (
    _active_decision,
    _analysis_tree_stages,
    _degraded_protected_columns,
    _harmony_candidate_pair,
    _harmony_completed,
    _harmony_metric_rows,
    _qc_profile_label,
    _render_decision_tree,
    _score_transition,
    _selected_parameter_context,
)
from .plots import _hvg_ranking_label, _render_plots

MAX_CHIP_LENGTH = 56


MAX_TABLE_COLUMNS = 7


MAX_INLINE_LEAVES = 12


REPORT_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400&display=swap');

:root {
  --blue: #0077fc;
  --black: #000000;
  --gray: #b4b4b4;
  --white: #ffffff;
}

* { box-sizing: border-box; }
html { background: var(--white); color: var(--black); font-family: Inter, sans-serif; }
body {
  margin: 0;
  overflow-x: hidden;
  background: var(--white);
  color: var(--black);
  font-family: Inter, sans-serif;
  font-weight: 300;
  letter-spacing: -0.04em;
  line-height: 1.45;
  overflow-wrap: break-word;
  word-break: normal;
}
a { color: var(--blue); }
header, main, footer {
  width: min(100%, 1240px);
  max-width: 100%;
  margin: 0 auto;
  padding-left: clamp(1.25rem, 5vw, 4.5rem);
  padding-right: clamp(1.25rem, 5vw, 4.5rem);
}
.technical-page header, .technical-page main, .technical-page footer {
  width: min(100%, 1800px);
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  border-bottom: 1px solid var(--black);
  padding-top: 1.75rem;
  padding-bottom: 1.75rem;
}
.brand {
  color: var(--black);
  font-size: 1rem;
  font-weight: 400;
  text-decoration: none;
}
main { padding-top: clamp(3rem, 8vw, 7rem); padding-bottom: 6rem; }
footer {
  border-top: 1px solid var(--black);
  padding-top: 2rem;
  padding-bottom: 2rem;
}
h1, h2, h3, p { margin-top: 0; }
h1 {
  max-width: 15ch;
  margin-bottom: 1.5rem;
  font-size: clamp(2.75rem, 7vw, 5rem);
  font-weight: 400;
  letter-spacing: 0;
  line-height: 1.2;
}
h2 {
  margin-bottom: 1.5rem;
  font-size: clamp(1.65rem, 3vw, 2.25rem);
  font-weight: 400;
  letter-spacing: -0.04em;
  line-height: 1.2;
}
h3 {
  margin-bottom: .8rem;
  font-size: 1rem;
  font-weight: 300;
  letter-spacing: -0.04em;
  line-height: 1.2;
}
p, li, td, th, summary, code, pre, a, dd, dt {
  font-family: Inter, sans-serif;
  letter-spacing: -0.04em;
  line-height: 1.45;
}
strong { font-weight: 400; }
.eyebrow {
  margin-bottom: 1rem;
  color: var(--gray);
  font-size: .75rem;
  font-weight: 400;
  text-transform: uppercase;
}
.lead {
  max-width: 48ch;
  font-size: clamp(1.2rem, 2vw, 1.7rem);
  font-weight: 300;
}
.report-nav {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: .35rem;
}
.report-nav a {
  border-radius: .35rem;
  padding: .4rem .65rem;
  color: var(--black);
  font-size: .8rem;
  font-weight: 400;
  text-decoration: none;
}
.report-nav a[aria-current="page"] {
  box-shadow: inset 0 0 0 1px var(--blue);
  color: var(--blue);
}
.pill-row, .chip-row, .metric-grid, .kv-list {
  display: flex;
  flex-wrap: wrap;
  gap: .65rem;
  min-width: 0;
  max-width: 100%;
}
.pill-row { margin-top: 1.75rem; }
.pill, .chip {
  display: inline-flex;
  max-width: 100%;
  font-size: .82rem;
  font-weight: 400;
  line-height: 1.35;
  overflow-wrap: anywhere;
  word-break: normal;
}
.pill {
  align-items: center;
  border: 1px solid var(--blue);
  border-radius: 999px;
  padding: .68rem 1.1rem;
  background: var(--blue);
  color: var(--white);
  text-decoration: none;
  white-space: nowrap;
}
.pill-outline {
  background: var(--white);
  box-shadow: inset 0 0 0 1px var(--blue);
  color: var(--blue);
}
.chip {
  display: inline-block;
  border-radius: .35rem;
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: .4rem .7rem;
  color: var(--black);
  white-space: normal;
  overflow: visible;
}
.text-item {
  display: block;
  min-width: 0;
  max-width: 100%;
  overflow-wrap: anywhere;
  word-break: normal;
}
.metric-grid { margin-top: 2rem; }
.metric {
  display: flex;
  min-width: 0;
  max-width: 100%;
  flex: 1 1 9rem;
  flex-direction: column;
  gap: .2rem;
  border-radius: 1.5rem;
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: .8rem 1.2rem;
}
.metric-label {
  color: var(--gray);
  font-size: .68rem;
  font-weight: 400;
  text-transform: uppercase;
}
.metric-value { font-size: .95rem; font-weight: 400; overflow-wrap: anywhere; }
.summary-grid, .interpretation-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 18rem), 1fr));
  gap: 1rem;
  min-width: 0;
  max-width: 100%;
}
.summary-card, .interpretation-card {
  min-width: 0;
  border: 1px solid var(--black);
  padding: 1.25rem;
}
.summary-card p:last-child, .interpretation-card p:last-child { margin-bottom: 0; }
.summary-label {
  margin-bottom: .5rem;
  color: var(--gray);
  font-size: .72rem;
  font-weight: 400;
  text-transform: uppercase;
}
.decision-tree {
  min-width: 0;
  max-width: 100%;
  margin-top: 2rem;
}
.tree-stage {
  min-width: 0;
  max-width: 100%;
  margin: 0;
  border: 0;
  padding: 0;
}
.tree-question {
  display: flex;
  width: min(100%, 19rem);
  min-height: 8rem;
  align-items: center;
  justify-content: center;
  margin: 0 auto;
  clip-path: polygon(50% 0, 100% 50%, 50% 100%, 0 50%);
  flex-direction: column;
  padding: 1.75rem 3rem;
  background: var(--blue);
  color: var(--white);
  text-align: center;
}
.tree-question span {
  margin-bottom: .35rem;
  font-size: .68rem;
  font-weight: 400;
  text-transform: uppercase;
}
.tree-question strong {
  font-size: .9rem;
  line-height: 1.25;
}
.tree-stage-description {
  max-width: 42rem;
  margin: 1rem auto 0;
  color: var(--gray);
  text-align: center;
}
.tree-branch-connectors, .tree-selection-connector {
  display: block;
  width: 100%;
  height: 5.25rem;
  color: var(--blue);
}
.tree-branch-connectors path, .tree-selection-connector path {
  fill: none;
  stroke: currentColor;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
}
.tree-branch-connectors marker path, .tree-selection-connector marker path {
  fill: currentColor;
  stroke: none;
}
.tree-branches {
  display: grid;
  grid-template-columns: repeat(var(--branch-count), minmax(0, 1fr));
  gap: .75rem;
  min-width: 0;
  max-width: 100%;
}
.tree-branch {
  min-width: 0;
  border: 1px solid var(--gray);
  padding: 1rem;
  background: var(--white);
}
.tree-branch-selected {
  border: 2px solid var(--blue);
  box-shadow: inset 0 .25rem 0 var(--blue);
}
.tree-branch-blocked {
  border-style: dashed;
}
.tree-branch-status {
  display: inline-block;
  margin-bottom: .65rem;
  border-radius: .3rem;
  box-shadow: inset 0 0 0 1px var(--gray);
  padding: .25rem .45rem;
  color: var(--gray);
  font-size: .68rem;
  font-weight: 400;
  text-transform: uppercase;
}
.tree-branch-selected .tree-branch-status {
  box-shadow: inset 0 0 0 1px var(--blue);
  color: var(--blue);
}
.tree-branch h3 { margin-bottom: .65rem; font-weight: 400; }
.tree-branch p { margin-bottom: 0; font-size: .82rem; }
.tree-metrics {
  margin: 0 0 .8rem;
  padding-left: 1rem;
  font-size: .76rem;
}
.tree-metrics li { margin: .25rem 0; }
.evidence-accordion {
  display: flex;
  flex-direction: column;
  gap: .8rem;
  margin-top: 1.5rem;
}
.evidence-panel {
  margin: 0;
  border: 1px solid var(--black);
  padding: 0;
}
.evidence-panel > summary {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: center;
  padding: 1.15rem 1.25rem;
  list-style: none;
}
.evidence-panel > summary::-webkit-details-marker { display: none; }
.evidence-panel > summary::after {
  color: var(--blue);
  content: "+";
  font-size: 1.4rem;
  line-height: 1;
}
.evidence-panel[open] > summary::after { content: "−"; }
.evidence-panel-title {
  display: block;
  margin-bottom: .25rem;
  color: var(--gray);
  font-size: .7rem;
  font-weight: 400;
  text-transform: uppercase;
}
.evidence-panel-outcome {
  display: block;
  font-size: .95rem;
  font-weight: 400;
}
.evidence-panel-body {
  border-top: 1px solid var(--black);
  padding: 1.25rem;
}
.evidence-panel-body > p:first-child { max-width: 55rem; }
.evidence-choice-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr));
  gap: .75rem;
  margin-top: 1rem;
}
.evidence-choice {
  min-width: 0;
  border: 1px solid var(--gray);
  padding: 1rem;
}
.evidence-choice-selected {
  border: 2px solid var(--blue);
  box-shadow: inset 0 .2rem 0 var(--blue);
}
.evidence-choice-rejected { border-style: dashed; }
.evidence-choice-status {
  display: inline-block;
  margin-bottom: .55rem;
  color: var(--gray);
  font-size: .68rem;
  font-weight: 400;
  text-transform: uppercase;
}
.evidence-choice-selected .evidence-choice-status { color: var(--blue); }
.evidence-choice h3 { margin-bottom: .55rem; font-weight: 400; }
.evidence-choice p:last-child { margin-bottom: 0; }
.evidence-choice .plain-list {
  margin-bottom: .75rem;
  font-size: .8rem;
}
.evidence-measurements {
  margin-top: 1.25rem;
  border: 0;
  border-top: 1px solid var(--gray);
  padding-top: .8rem;
}
.evidence-measurements > summary {
  color: var(--blue);
  font-size: .82rem;
}
.evidence-measurements-body { padding-top: 1rem; }
.evidence-measurement-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 14rem), 1fr));
  gap: .75rem;
}
.evidence-measurement {
  min-width: 0;
  border-bottom: 1px solid var(--gray);
  padding: .7rem 0;
}
.evidence-measurement dt {
  margin-bottom: .35rem;
  color: var(--gray);
}
.evidence-measurement dd { font-size: .86rem; }
.evidence-measurement small {
  display: block;
  margin-top: .35rem;
  color: var(--gray);
  font-size: .72rem;
}
.plain-list { margin: 0; padding-left: 1.2rem; }
.plain-list li { margin: .55rem 0; }
.column-list {
  columns: 4 12rem;
  column-gap: 2rem;
}
.column-list li {
  break-inside: avoid;
  margin: .25rem 0;
}
.section {
  margin-top: 4rem;
  min-width: 0;
  max-width: 100%;
  border-top: 1px solid var(--black);
  padding-top: 1.5rem;
}
.section:target { scroll-margin-top: 1rem; }
.section-heading {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: start;
}
.subsection { margin-top: 2rem; min-width: 0; max-width: 100%; }
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 19rem), 1fr));
  gap: 1rem;
  min-width: 0;
  max-width: 100%;
}
.card, .callout, .record {
  min-width: 0;
  max-width: 100%;
  border: 1px solid var(--black);
  padding: 1.25rem;
  background: var(--white);
  overflow: visible;
}
.callout { border-color: var(--blue); }
.record-stack {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 0;
  max-width: 100%;
}
.product-callout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: center;
  margin-top: 2.5rem;
  border-radius: 1.5rem;
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: 1.4rem;
}
.product-callout p { margin-bottom: 0; max-width: 55rem; }
.empty { color: var(--gray); font-style: italic; }
.table-wrap {
  width: 100%;
  max-width: 100%;
  overflow: visible;
}
table {
  width: 100%;
  table-layout: auto;
  border-collapse: collapse;
  font-size: .86rem;
}
th, td {
  min-width: 0;
  width: auto;
  border-bottom: 1px solid var(--black);
  padding: .8rem .7rem;
  text-align: left;
  vertical-align: top;
  overflow-wrap: break-word;
  word-break: normal;
  hyphens: auto;
}
th {
  position: sticky;
  top: 0;
  background: var(--white);
  color: var(--gray);
  font-weight: 400;
  overflow-wrap: normal;
  text-transform: uppercase;
}
td { font-weight: 300; overflow-wrap: anywhere; }
td > * { max-width: 100%; }
tr.selected { box-shadow: inset 4px 0 0 var(--blue); }
.table-records { gap: 1.25rem; }
.table-record {
  border-color: var(--gray);
  padding: 1rem;
}
.table-record-selected {
  border: 2px solid var(--blue);
  box-shadow: inset .25rem 0 0 var(--blue);
}
.record-fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 13rem), 1fr));
  gap: .9rem 1.25rem;
}
.record-field {
  min-width: 0;
  border-bottom: 1px solid var(--gray);
  padding-bottom: .65rem;
}
.record-field-wide { grid-column: 1 / -1; }
.record-field dt { margin-bottom: .3rem; }
.record-field dd {
  overflow-wrap: anywhere;
  word-break: normal;
}
dl { margin: 0; min-width: 0; max-width: 100%; }
.details > div {
  display: grid;
  grid-template-columns: minmax(0, 12rem) minmax(0, 1fr);
  gap: 1rem;
  min-width: 0;
  border-bottom: 1px solid var(--gray);
  padding: .55rem 0;
}
.details .details > div {
  grid-template-columns: minmax(0, 1fr);
  gap: .2rem;
}
dt { color: var(--gray); font-size: .78rem; font-weight: 400; text-transform: uppercase; }
dd { min-width: 0; margin: 0; overflow-wrap: anywhere; }
.kv { display: inline-flex; flex-wrap: wrap; gap: .25rem .4rem; min-width: 0; max-width: 100%; }
.kv-k {
  color: var(--gray);
  font-size: .72rem;
  font-weight: 400;
  text-transform: uppercase;
}
.kv-v { overflow-wrap: anywhere; word-break: normal; }
.nested-records {
  display: block;
  margin: .15rem 0;
  border: 0;
  padding: 0;
  min-width: 0;
  max-width: 100%;
  overflow: visible;
}
.nested-records > summary { color: var(--blue); font-size: .82rem; }
.nested-records .table-wrap, .nested-records .record-stack { margin-top: .55rem; }
.plot-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 26rem), 1fr));
  gap: 2rem;
  min-width: 0;
}
figure { margin: 0; min-width: 0; }
figure.primary { grid-column: 1 / -1; }
figure img { display: block; width: 100%; height: auto; border: 1px solid var(--black); }
figcaption { margin-top: .7rem; color: var(--black); font-size: .85rem; }
.cluster-row {
  display: grid;
  grid-template-columns: minmax(0, 8rem) minmax(0, 1fr) auto;
  gap: .7rem;
  align-items: center;
  margin: .5rem 0;
  min-width: 0;
}
.cluster-track { height: .7rem; border-radius: 999px; background: var(--gray); overflow: hidden; }
.cluster-fill { height: 100%; border-radius: 999px; background: var(--blue); }
.text-list { padding-left: 1.2rem; }
.text-list li { margin: .45rem 0; }
details { margin-top: 1rem; border-top: 1px solid var(--gray); padding-top: .8rem; }
summary { cursor: pointer; font-weight: 400; }
pre {
  max-height: 36rem;
  max-width: 100%;
  overflow: auto;
  background: var(--white);
  box-shadow: inset 0 0 0 1px var(--blue);
  padding: 1rem;
  font-size: .76rem;
  white-space: pre-wrap;
  word-break: break-word;
}
@media (max-width: 900px) {
  .tree-branches { grid-template-columns: 1fr; }
  .tree-branch-connectors, .tree-selection-connector { display: none; }
  .tree-question { margin-bottom: 2.5rem; }
  .tree-stage:not(:last-child)::after {
    display: block;
    margin: .25rem 0 2rem;
    color: var(--blue);
    content: "↓";
    font-size: 1.5rem;
    text-align: center;
  }
  .tree-branch {
    position: relative;
    margin-bottom: 1.5rem;
  }
  .tree-branch::before {
    position: absolute;
    top: -1.65rem;
    left: 50%;
    color: var(--blue);
    content: "↓";
  }
}
@media (max-width: 680px) {
  header { align-items: flex-start; flex-direction: column; }
  .report-nav { justify-content: flex-start; }
  .section-heading, .product-callout { grid-template-columns: 1fr; }
  .details > div { grid-template-columns: 1fr; gap: .25rem; }
  .cluster-row { grid-template-columns: minmax(0, 1fr) auto; }
}
"""


def _chip(text: str) -> str:
    escaped = html.escape(text)
    if len(text) > MAX_CHIP_LENGTH:
        return f'<span class="text-item">{escaped}</span>'
    return f'<span class="chip">{escaped}</span>'


def _chips(value: Any, empty: str = "Not provided") -> str:
    if not _present(value):
        return f'<span class="empty">{html.escape(empty)}</span>'
    if isinstance(value, Mapping):
        items = [f"{_label(key)}: {_scalar(item)}" for key, item in value.items()]
    elif _is_sequence(value):
        items = list(value)
    else:
        items = [value]
    return '<span class="chip-row">{}</span>'.format(
        "".join(_chip(_scalar(item)) for item in items)
    )


def _kv_list(mapping: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, item in mapping.items():
        if not _present(item):
            continue
        label = html.escape(_label(key))
        if _is_leaf(item):
            text = f"{_label(key)}: {_scalar(item)}"
            if len(text) <= MAX_CHIP_LENGTH:
                parts.append(_chip(text))
            else:
                parts.append(
                    f'<span class="kv"><span class="kv-k">{label}</span>'
                    f'<span class="kv-v">{html.escape(_scalar(item))}</span></span>'
                )
        elif _is_simple(item):
            parts.append(
                f'<span class="kv"><span class="kv-k">{label}</span>{_chips(item)}</span>'
            )
        else:
            parts.append(
                '<details class="nested-records">'
                f"<summary>{label}</summary>{_value(item)}</details>"
            )
    if not parts:
        return '<span class="empty">Not provided</span>'
    return f'<div class="kv-list">{"".join(parts)}</div>'


def _cell(value: Any) -> str:
    if not _present(value):
        return '<span class="empty">Not provided</span>'
    if isinstance(value, Mapping):
        return _kv_list(value)
    if _is_mapping_sequence(value):
        count = len(value)
        return (
            '<details class="nested-records">'
            f"<summary>{count:,} records</summary>"
            f"{_mapping_list(value)}</details>"
        )
    if _is_sequence(value) and all(_is_leaf(item) for item in value):
        if len(value) > MAX_INLINE_LEAVES:
            return html.escape(", ".join(_scalar(item) for item in value))
        return _chips(value)
    if _is_sequence(value):
        return _chips(value)
    return html.escape(_scalar(value))


def _visible_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            key for row in rows for key in row if not str(key).startswith("_")
        )
    )


def _render_record_rows(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> str:
    records: list[str] = []
    for row in rows:
        fields: list[str] = []
        for key in columns:
            value = row.get(key)
            wide = not _is_simple(value) or (
                isinstance(value, str) and len(value) > MAX_CHIP_LENGTH
            )
            field_class = "record-field record-field-wide" if wide else "record-field"
            fields.append(
                f'<div class="{field_class}">'
                f"<dt>{html.escape(_label(key))}</dt>"
                f"<dd>{_cell(value)}</dd></div>"
            )
        selected_class = " table-record-selected" if row.get("_selected") else ""
        records.append(
            f'<article class="record table-record{selected_class}">'
            f'<dl class="record-fields">{"".join(fields)}</dl></article>'
        )
    return f'<div class="record-stack table-records">{"".join(records)}</div>'


def _mapping_list(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [dict(row) for row in rows]
    if not normalized:
        return '<p class="empty">No records available.</p>'
    visible = _visible_columns(normalized)
    if len(visible) <= MAX_TABLE_COLUMNS:
        return _table(normalized)
    return _render_record_rows(normalized, visible)


def _value(value: Any) -> str:
    if not _present(value):
        return '<span class="empty">Not provided</span>'
    if isinstance(value, Mapping):
        rows = "".join(
            "<div><dt>{}</dt><dd>{}</dd></div>".format(
                html.escape(_label(key)),
                (
                    _mapping_list(_mappings(item))
                    if _is_mapping_sequence(item)
                    else _value(item)
                ),
            )
            for key, item in value.items()
            if _present(item)
        )
        return f'<dl class="details">{rows}</dl>'
    if _is_mapping_sequence(value):
        return _mapping_list(value)
    if _is_sequence(value):
        if all(_is_leaf(item) for item in value):
            return _chips(value)
        return '<div class="card-grid">{}</div>'.format(
            "".join(f'<div class="card">{_value(item)}</div>' for item in value)
        )
    return html.escape(_scalar(value))


def _table(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    empty: str = "No records available.",
) -> str:
    normalized = [dict(row) for row in rows]
    if not normalized:
        return f'<p class="empty">{html.escape(empty)}</p>'
    visible = list(columns or ()) or _visible_columns(normalized)
    if len(visible) > MAX_TABLE_COLUMNS:
        return _render_record_rows(normalized, visible)
    headings = "".join(f"<th>{html.escape(_label(key))}</th>" for key in visible)
    body = "".join(
        ('<tr class="selected">' if row.get("_selected") else "<tr>")
        + "".join(f"<td>{_cell(row.get(key))}</td>" for key in visible)
        + "</tr>"
        for row in normalized
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{headings}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _render_clusters(cluster_counts: Mapping[str, int]) -> str:
    if not cluster_counts:
        return '<p class="empty">No final cluster counts were available.</p>'
    maximum = max(cluster_counts.values(), default=1) or 1
    return "".join(
        '<div class="cluster-row">'
        f"<span>Cluster {html.escape(str(label))}</span>"
        '<span class="cluster-track">'
        f'<span class="cluster-fill" style="width: {count / maximum * 100:.2f}%">'
        "</span></span>"
        f"<span>{count:,}</span></div>"
        for label, count in cluster_counts.items()
    )


def _parameter_rows(parameter: Mapping[str, Any]) -> list[dict[str, Any]]:
    assay_reports = _mapping(parameter.get("assayReports"))
    if not assay_reports and _present(parameter.get("evaluations")):
        assay_reports = {str(parameter.get("fromAssay") or "Primary"): dict(parameter)}
    recommended = _mapping(parameter.get("recommendedByAssay"))
    rows: list[dict[str, Any]] = []
    for assay, raw_report in assay_reports.items():
        report = _mapping(raw_report)
        selected = recommended.get(assay) or report.get("recommendedCandidateId")
        for evaluation in _mappings(report.get("evaluations")):
            parameters = _mapping(evaluation.get("parameters"))
            rows.append(
                {
                    "_selected": evaluation.get("candidateId") == selected,
                    "assay": assay,
                    "candidate": evaluation.get("candidateId"),
                    "phase": evaluation.get("phase"),
                    "status": evaluation.get("status"),
                    "eligible": evaluation.get("eligible"),
                    "selection confidence": report.get("confidence"),
                    "reduction": parameters.get("reductionMethod"),
                    "dimensions": parameters.get("dimensions"),
                    "neighbors K": parameters.get("neighborsK"),
                    "resolution": parameters.get("leidenResolution"),
                    "Harmony": parameters.get("useHarmony"),
                    "metrics": evaluation.get("metrics"),
                }
            )
    return rows


def _render_parameter_tuning(parameter: Mapping[str, Any]) -> str:
    if not parameter:
        return '<p class="empty">No Parameter Tuning report was persisted.</p>'
    candidate_rows = _parameter_rows(parameter)
    integration_rows = _mappings(parameter.get("integrationEvaluations"))
    plans: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    root_plan = _mapping(parameter.get("searchPlan"))
    if root_plan:
        plans.append({"assay": parameter.get("fromAssay"), **root_plan})
    for comparison in _mappings(parameter.get("comparisons")):
        comparisons.append(
            {"scope": parameter.get("fromAssay") or "primary assay", **comparison}
        )
    for assay, report in _mapping(parameter.get("assayReports")).items():
        assay_report = _mapping(report)
        plan = _mapping(assay_report.get("searchPlan"))
        if plan and plan not in plans:
            plans.append({"assay": assay, **plan})
        for comparison in _mappings(assay_report.get("comparisons")):
            comparisons.append({"scope": assay, **comparison})
    final_selection = _mapping(parameter.get("finalSelection"))
    for comparison in _mappings(final_selection.get("comparisons")):
        comparisons.append({"scope": "final graph", **comparison})
    narrative = {
        "status": parameter.get("status"),
        "totalCandidates": parameter.get("totalCandidates"),
        "recommendedByAssay": parameter.get("recommendedByAssay"),
        "recommendedIntegrationId": parameter.get("recommendedIntegrationId"),
        "confidence": parameter.get("confidence"),
        "rationale": parameter.get("rationale"),
        "tradeoffs": parameter.get("tradeoffs"),
        "stopReason": parameter.get("stopReason"),
        "finalSelection": final_selection,
    }
    return (
        '<div class="callout"><h3>Final graph selection</h3>'
        f"{_value(narrative)}</div>"
        '<div class="subsection"><h3>Native and Harmony candidates</h3>'
        f"{_table(candidate_rows, empty='No native candidates were recorded.')}</div>"
        '<div class="subsection"><h3>SNN and WNN integration candidates</h3>'
        f"{_table(integration_rows, empty='No integration candidates were eligible.')}</div>"
        '<div class="subsection"><h3>Model-authored comparisons</h3>'
        f"{_table(comparisons, empty='No candidate comparisons were required.')}</div>"
        '<div class="subsection"><h3>Bounded search plans</h3>'
        f"{_value(plans) if plans else '<p class="empty">No refinement plan was requested.</p>'}"
        "</div>"
    )


def _execution_rows(reports: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def visit(value: Any, stage: str, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            usage = value.get("usage")
            agent_name = value.get("agentName")
            if (
                isinstance(usage, Mapping)
                and isinstance(agent_name, str)
                and agent_name.strip()
            ):
                run_id = str(value.get("runId") or "")
                identity = (agent_name, run_id, str(value.get("modelName") or ""))
                if identity not in seen:
                    seen.add(identity)
                    rows.append(
                        {
                            "agent stage": _label(stage),
                            "execution": _label(path[-1]) if path else agent_name,
                            "agent": agent_name,
                            "run ID": run_id or "deterministic",
                            "model": value.get("modelName") or "not applicable",
                            "duration seconds": value.get("durationSeconds"),
                            "requests": usage.get("requests", 0),
                            "tool calls": usage.get("toolCalls", 0),
                            "input tokens": usage.get("inputTokens", 0),
                            "output tokens": usage.get("outputTokens", 0),
                            "total tokens": usage.get("totalTokens", 0),
                        }
                    )
            for key, item in value.items():
                visit(item, stage, (*path, str(key)))
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                visit(item, stage, (*path, str(index + 1)))

    for stage, records in reports.items():
        visit(records, str(stage), ())
    return rows


def _render_executions(reports: Mapping[str, Any]) -> str:
    rows = _execution_rows(reports)
    if not rows:
        return '<p class="empty">No provider execution metadata was recorded.</p>'
    totals = {
        "recorded executions": len(rows),
        "provider executions": sum(
            int(
                bool(row["model"] != "not applicable")
                or int(row["requests"] or 0) > 0
                or int(row["input tokens"] or 0) > 0
                or int(row["output tokens"] or 0) > 0
            )
            for row in rows
        ),
        "requests": sum(int(row["requests"] or 0) for row in rows),
        "tool calls": sum(int(row["tool calls"] or 0) for row in rows),
        "input tokens": sum(int(row["input tokens"] or 0) for row in rows),
        "output tokens": sum(int(row["output tokens"] or 0) for row in rows),
        "total tokens": sum(int(row["total tokens"] or 0) for row in rows),
    }
    return (
        '<div class="callout"><h3>Recorded totals</h3>'
        f'{_chips(totals)}</div><div class="subsection">{_table(rows)}</div>'
    )


def _render_timeline(
    attempts: Sequence[Mapping[str, Any]],
    resumes: Sequence[Mapping[str, Any]],
) -> str:
    artifacts: list[dict[str, Any]] = []
    for attempt in attempts:
        for name, reference in _mapping(attempt.get("artifacts")).items():
            artifact = _mapping(reference)
            artifacts.append(
                {
                    "stage": attempt.get("stage"),
                    "attempt": attempt.get("attemptId"),
                    "name": name,
                    "scope": artifact.get("scope"),
                    "assay": artifact.get("assay"),
                    "kind": artifact.get("kind"),
                    "artifact ID": artifact.get("artifactId"),
                }
            )
    return (
        "<h3>Stage attempts</h3>"
        + _table(
            attempts,
            columns=(
                "stage",
                "status",
                "durationSeconds",
                "actions",
                "reportCount",
                "artifactCount",
                "parentAttempts",
                "questionIds",
                "noteCount",
                "errorType",
            ),
        )
        + '<div class="subsection"><h3>Stage artifact inventory</h3>'
        + _table(artifacts, empty="No stage artifacts were recorded.")
        + "</div>"
        + '<div class="subsection"><h3>Resume lineage</h3>'
        + _table(
            resumes,
            columns=(
                "resumeId",
                "answeredStage",
                "answeredAttemptId",
                "questionIds",
            ),
            empty="No resume was required.",
        )
        + "</div>"
    )


def _study_overview(
    payload: Mapping[str, Any],
) -> tuple[str, list[str], list[str]]:
    reports = _mapping(payload.get("reports"))
    request = _mapping(payload.get("request"))
    enrichment = _latest(reports, "data_enrichment")
    study = _mapping(enrichment.get("studyContextSummary"))
    objective = ""
    for candidate in (
        study.get("studyObjective"),
        request.get("studyObjective"),
        study.get("studyContext"),
        request.get("studyContext"),
    ):
        objective = _brief_text(candidate)
        if objective:
            break
    return (
        objective or "The automated analysis completed successfully.",
        _specific_references(_text_values(study.get("organismReferences"))),
        _specific_references(_text_values(study.get("tissueReferences"))),
    )


def _biological_source(
    organisms: Sequence[str],
    tissues: Sequence[str],
) -> str:
    organism = _format_text_list(organisms)
    tissue = _format_text_list(tissues)
    if organism and tissue:
        return f"{organism} material from {tissue}"
    if organism:
        return f"{organism} biological material"
    if tissue:
        return f"biological material from {tissue}"
    return ""


def _report_assays(plan: Mapping[str, Any]) -> list[str]:
    assays: list[str] = []
    for assay in _mappings(plan.get("assays")):
        label = _assay_label(assay.get("assayType") or assay.get("assay"))
        if label and label not in assays:
            assays.append(label)
    return assays


def _render_metrics(metrics: Sequence[tuple[str, Any]]) -> str:
    markup = "".join(
        '<span class="metric">'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f'<span class="metric-value">{html.escape(_scalar(value))}</span></span>'
        for label, value in metrics
        if _present(value)
    )
    return f'<div class="metric-grid">{markup}</div>' if markup else ""


def _render_report_navigation(active_page: str) -> str:
    links = (
        ("analysis", "index.html", "Analysis summary"),
        ("technical", "technical.html", "Methods and evidence"),
    )
    return '<nav class="report-nav" aria-label="Report pages">{}</nav>'.format(
        "".join(
            '<a href="{}"{}>{}</a>'.format(
                html.escape(path, quote=True),
                ' aria-current="page"' if page == active_page else "",
                html.escape(label),
            )
            for page, path, label in links
        )
    )


def _render_report_shell(
    *,
    title: str,
    active_page: str,
    body: str,
) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{REPORT_STYLES}</style>
</head>
<body class="{html.escape(active_page, quote=True)}-page">
<header>
  <a class="brand" href="https://www.nygen.io/" target="_blank" rel="noopener noreferrer">Nygen Analytics</a>
  {_render_report_navigation(active_page)}
</header>
<main>
{body}
</main>
<footer>
  <p>Generated locally by Scarf agents. <a href="https://www.nygen.io/">Nygen Analytics</a></p>
</footer>
</body>
</html>
"""


def _render_selection_evidence(payload: Mapping[str, Any]) -> str:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    final = _mapping(workflow_result.get("finalAnalysis"))
    parameter = _latest(reports, "parameter_tuning")
    _report, _evaluations, selected = _selected_parameter_context(parameter, final)
    metrics = _mapping(selected.get("metrics"))
    cards: list[tuple[str, str, str]] = []
    candidate_count = parameter.get("totalCandidates")
    if isinstance(candidate_count, int):
        cards.append(
            (
                "Settings compared",
                f"{candidate_count:,}",
                "Completed parameter combinations considered before selection.",
            )
        )
    card_specs = (
        (
            "graphSilhouetteMedian",
            "Group separation",
            "Higher values indicate clearer separation between neighboring groups.",
        ),
        (
            "minClusterCells",
            "Smallest group",
            "Number of cells in the smallest selected group.",
        ),
        (
            "seedStability",
            "Repeat-run stability",
            "Agreement when clustering is repeated with a different random seed.",
        ),
        (
            "subsampleStability",
            "Subsample stability",
            "Agreement when the analysis is repeated on a subset of cells.",
        ),
        (
            "markerCoherence",
            "Marker coherence",
            "Consistency of marker support across the selected groups.",
        ),
        (
            "crossUnitSupport",
            "Cross-sample support",
            "Support for the selected groups across the study units.",
        ),
    )
    for key, label, explanation in card_specs:
        value = metrics.get(key)
        if isinstance(value, int):
            display = f"{value:,} cells" if key == "minClusterCells" else f"{value:,}"
        elif isinstance(value, float):
            display = f"{value:.3f}"
        else:
            continue
        cards.append((label, display, explanation))
    if not cards:
        return ""
    return '<div class="summary-grid">{}</div>'.format(
        "".join(
            '<article class="summary-card">'
            f'<p class="summary-label">{html.escape(label)}</p>'
            f"<h3>{html.escape(value)}</h3>"
            f"<p>{html.escape(explanation)}</p>"
            "</article>"
            for label, value, explanation in cards
        )
    )


def _render_evidence_choices(choices: Sequence[Mapping[str, Any]]) -> str:
    return '<div class="evidence-choice-grid">{}</div>'.format(
        "".join(
            '<article class="evidence-choice evidence-choice-{}">'.format(
                html.escape(str(choice.get("state") or "reviewed"), quote=True)
            )
            + '<span class="evidence-choice-status">{}</span>'.format(
                html.escape(str(choice.get("status") or "Reviewed"))
            )
            + f"<h3>{html.escape(str(choice.get('label') or 'Evidence'))}</h3>"
            + _render_plain_list(_text_values(choice.get("metrics")))
            + (
                f"<p>{html.escape(_brief_text(choice.get('reason')))}</p>"
                if _brief_text(choice.get("reason"))
                else ""
            )
            + "</article>"
            for choice in choices
        )
    )


def _render_evidence_measurements(
    measurements: Sequence[tuple[str, str, str]],
) -> str:
    if not measurements:
        return ""
    return '<dl class="evidence-measurement-grid">{}</dl>'.format(
        "".join(
            '<div class="evidence-measurement">'
            f"<dt>{html.escape(label)}</dt>"
            f"<dd>{html.escape(value)}"
            + (f"<small>{html.escape(detail)}</small>" if detail else "")
            + "</dd></div>"
            for label, value, detail in measurements
        )
    )


def _render_evidence_panel(
    *,
    title: str,
    outcome: str,
    introduction: str,
    body: str,
    measurements: str = "",
    expanded: bool = False,
) -> str:
    measurement_markup = (
        '<details class="evidence-measurements"><summary>Measurements</summary>'
        f'<div class="evidence-measurements-body">{measurements}</div></details>'
        if measurements
        else ""
    )
    open_attribute = " open" if expanded else ""
    return (
        f'<details class="evidence-panel"{open_attribute}><summary><span>'
        f'<span class="evidence-panel-title">{html.escape(title)}</span>'
        f'<span class="evidence-panel-outcome">{html.escape(outcome)}</span>'
        "</span></summary>"
        '<div class="evidence-panel-body">'
        f"<p>{html.escape(introduction)}</p>{body}{measurement_markup}</div></details>"
    )


def _qc_profile_scope(profile: Mapping[str, Any]) -> str:
    bounds = _qc_resolved_bounds(profile)
    groups = {str(item.get("group")) for item in bounds if _present(item.get("group"))}
    return "Per-library thresholds" if len(groups) > 1 else "Global thresholds"


def _qc_flag_summary(profile: Mapping[str, Any]) -> list[str]:
    labels = (
        ("nCounts:high", "High RNA count flags"),
        ("nCounts:lowQuality", "Low RNA count flags"),
        ("nFeatures:high", "High detected-gene flags"),
        ("nFeatures:lowQuality", "Low detected-gene flags"),
        ("percentMito:highMito", "High mitochondrial-percentage flags"),
        ("percentRibo:highRibo", "High ribosomal-percentage flags"),
    )
    flags = _mapping(profile.get("flaggedCells"))
    values: list[str] = []
    for suffix, label in labels:
        count = next(
            (
                value
                for key, value in flags.items()
                if str(key).endswith(suffix) and isinstance(value, int)
            ),
            None,
        )
        if count is not None:
            values.append(f"{label}: {count:,}")
    return values


def _qc_bound_summary(profile: Mapping[str, Any]) -> str:
    bounds = _qc_resolved_bounds(profile)
    parts: list[str] = []
    for role, label in (
        ("count", "RNA counts"),
        ("feature", "Detected genes"),
        ("mitochondrial", "Mitochondrial percentage"),
        ("ribosomal", "Ribosomal percentage"),
    ):
        matching = [item for item in bounds if item.get("role") == role]
        if not matching:
            continue
        lower = _analysis_number_range([item.get("lowerRemoval") for item in matching])
        upper_removal = _analysis_number_range(
            [item.get("upperRemoval") for item in matching]
        )
        upper_flag = _analysis_number_range(
            [item.get("upperFlag") for item in matching]
        )
        cutoffs = [
            value
            for value in (
                f"lower cutoff {lower}" if lower != "Not available" else "",
                (
                    f"upper cutoff {upper_removal}"
                    if upper_removal != "Not available"
                    else ""
                ),
                (
                    f"high-value flag {upper_flag}"
                    if upper_flag != "Not available"
                    else ""
                ),
            )
            if value
        ]
        if cutoffs:
            parts.append(f"{label}: {'; '.join(cutoffs)}")
    return ". ".join(parts)


def _qc_metric_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    bounds = _qc_resolved_bounds(profile)
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for bound in bounds:
        metric = str(bound.get("metric") or bound.get("role") or "")
        if metric:
            by_metric.setdefault(metric, []).append(bound)
    capture_comparisons = _mappings(
        _mapping(profile.get("parameters")).get("captureComparisons")
    )
    if capture_comparisons:
        first_metrics = _mapping(capture_comparisons[0].get("metricComparisons"))
        for metric, raw in first_metrics.items():
            if str(metric).startswith("artifact_") or metric in by_metric:
                continue
            comparison = _mapping(raw)
            by_metric[metric] = [
                {
                    "metric": metric,
                    "group": "global diagnostic",
                    "role": comparison.get("role"),
                    "median": comparison.get("globalMedian"),
                    "diagnosticLower": comparison.get("globalLower"),
                    "diagnosticUpper": comparison.get("globalUpper"),
                }
            ]
    flags = _mapping(profile.get("metricFlaggedCells"))
    rows: list[dict[str, Any]] = []
    for metric, metric_bounds in by_metric.items():
        medians = [item.get("median") for item in metric_bounds]
        lower = [item.get("lowerRemoval") for item in metric_bounds]
        upper = [item.get("upperRemoval") for item in metric_bounds]
        high_flag = [item.get("upperFlag") for item in metric_bounds]
        diagnostic_range = [
            item.get(field)
            for item in metric_bounds
            for field in ("diagnosticLower", "diagnosticUpper")
        ]
        metric_flags = _mapping(flags.get(metric))
        rows.append(
            {
                "metric": _public_field_label(metric),
                "scope": (
                    "Global diagnostic only"
                    if any(value is not None for value in diagnostic_range)
                    else _qc_profile_scope({"resolvedBounds": metric_bounds})
                ),
                "median": _analysis_number_range(medians),
                "diagnostic reference": _analysis_number_range(diagnostic_range),
                "lower cutoff": _analysis_number_range(lower),
                "upper cutoff": _analysis_number_range(upper),
                "high flag": _analysis_number_range(high_flag),
                "flagged cells": sum(
                    int(value)
                    for value in metric_flags.values()
                    if isinstance(value, int)
                ),
            }
        )
    return rows


def _qc_profile_rows(profiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        active = profile.get("activeCells")
        retained = profile.get("retainedCells")
        removed = (
            active - retained
            if isinstance(active, int) and isinstance(retained, int)
            else None
        )
        rows.append(
            {
                "profile": _qc_profile_label(profile),
                "scope": _qc_profile_scope(profile),
                "active cells": active,
                "retained cells": retained,
                "removed cells": removed,
                "flags": _qc_flag_summary(profile),
                "failed libraries": len(
                    _text_values(profile.get("failedCaptureCandidates"))
                ),
            }
        )
    return rows


def _qc_bound_rows(profiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for bound in _qc_resolved_bounds(profile):
            rows.append(
                {
                    "profile": _qc_profile_label(profile),
                    "group": bound.get("group"),
                    "metric": _public_field_label(
                        bound.get("metric") or bound.get("role")
                    ),
                    "median": bound.get("median"),
                    "lower removal": bound.get("lowerRemoval"),
                    "upper removal": bound.get("upperRemoval"),
                    "upper flag": bound.get("upperFlag"),
                }
            )
    return rows


def _render_filtering_evidence(
    experimental: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    profiles = _mappings(experimental.get("qcProfiles"))
    if not profiles:
        return ""
    decision = _mapping(experimental.get("decision"))
    cell_qc = _mapping(plan.get("cellQc"))
    if not cell_qc:
        cell_qc = _mapping(decision.get("cellQc"))
    if not cell_qc:
        cell_qc = _mapping(experimental.get("cellQc"))
    selected = _selected_qc_profile(experimental, cell_qc)
    selected_id = selected.get("profileId")
    selected_name = selected.get("registeredProfile")
    choices: list[dict[str, Any]] = []
    for profile in profiles:
        is_selected = bool(
            (selected_id and profile.get("profileId") == selected_id)
            or (
                not selected_id
                and selected_name
                and profile.get("registeredProfile") == selected_name
            )
        )
        active = profile.get("activeCells")
        retained = profile.get("retainedCells")
        metrics: list[str] = []
        removed: int | None = None
        if isinstance(active, int) and isinstance(retained, int):
            removed = active - retained
            metrics.extend(
                (
                    f"Retained: {retained:,} of {active:,}",
                    f"Removed: {removed:,}",
                )
            )
        n_mads = _mapping(profile.get("parameters")).get("nMads")
        if isinstance(n_mads, (int, float)):
            metrics.append(
                f"Threshold distance: {float(n_mads):g} median absolute deviations"
            )
        metrics.append(_qc_profile_scope(profile))
        metrics.extend(_qc_flag_summary(profile))
        choices.append(
            {
                "label": _qc_profile_label(profile),
                "status": "Selected" if is_selected else "Not selected",
                "state": "selected" if is_selected else "rejected",
                "metrics": metrics,
                "reason": (
                    str(
                        cell_qc.get("rationale")
                        or "Selected the registered QC profile shown above."
                    )
                    if is_selected
                    else "An alternative registered QC profile with the measured retention shown above."
                ),
            }
        )
    active = selected.get("activeCells")
    retained = selected.get("retainedCells")
    selected_label = _qc_profile_label(selected)
    selected_flags = sum(
        int(value)
        for value in _mapping(selected.get("flaggedCells")).values()
        if isinstance(value, int)
    )
    outcome = (
        f"{selected_label}; {retained:,} of {active:,} cells retained; "
        f"{selected_flags:,} diagnostic flags"
        if isinstance(active, int) and isinstance(retained, int)
        else f"{selected_label} selected"
    )
    measurements: list[tuple[str, str, str]] = []
    for profile in profiles:
        parameters = _mapping(profile.get("parameters"))
        n_mads = parameters.get("nMads")
        rule = (
            f"{float(n_mads):g} median absolute deviations (MAD), "
            f"{_qc_profile_scope(profile).lower()}"
            if isinstance(n_mads, (int, float))
            else _qc_profile_scope(profile)
        )
        measurements.append(
            (
                _qc_profile_label(profile),
                rule,
                ". ".join(
                    value
                    for value in (
                        _qc_bound_summary(profile),
                        "; ".join(_qc_flag_summary(profile)),
                    )
                    if value
                ),
            )
        )
    for column, group_counts in _mapping(selected.get("retainedCellsByColumn")).items():
        counts = list(_mapping(group_counts).values())
        numeric = [value for value in counts if isinstance(value, int)]
        if numeric:
            measurements.append(
                (
                    f"Retention across {_public_field_label(column)}",
                    f"{len(numeric):,} groups",
                    f"{min(numeric):,} to {max(numeric):,} retained cells per group.",
                )
            )
    return _render_evidence_panel(
        title="Cell filtering",
        outcome=outcome,
        introduction=(
            f"{len(profiles):,} registered filtering strategies were compared. "
            "The selected strategy retained the published cell set because stricter "
            "alternatives did not provide stronger support. Its cutoffs are "
            "diagnostic bounds and did not remove cells."
        ),
        body=(
            _render_evidence_choices(choices)
            + '<div class="subsection"><h3>Selected QC metrics and cutoffs</h3>'
            + _table(
                _qc_metric_rows(selected),
                columns=(
                    "metric",
                    "scope",
                    "median",
                    "diagnostic reference",
                    "lower cutoff",
                    "upper cutoff",
                    "high flag",
                    "flagged cells",
                ),
                empty="No selected QC metric cutoffs were recorded.",
            )
            + "</div>"
        ),
        measurements=_render_evidence_measurements(measurements),
        expanded=True,
    )


def _covariate_pair_measurements(
    characterization: Mapping[str, Any],
) -> list[tuple[str, str, str]]:
    measurements: list[tuple[str, str, str]] = []
    for item in _mappings(characterization.get("confounding")):
        coefficient = _public_field_label(item.get("coefficient"))
        for pair in _mappings(item.get("pairs")):
            technical = _public_field_label(pair.get("technical"))
            association = _mapping(pair.get("association"))
            status = str(association.get("status") or "")
            value = association.get("value")
            uncorrected = association.get("valueUncorrected")
            if status == "notComputed":
                display = "Not independently measurable"
            elif isinstance(value, (int, float)):
                display = f"Association score: {float(value):.3f}"
            else:
                display = "Association not available"
            rows_used = association.get("rowsUsed")
            details = (
                [f"{rows_used:,} study units"] if isinstance(rows_used, int) else []
            )
            if status == "notComputed" and isinstance(uncorrected, (int, float)):
                details.append(
                    f"uncorrected association score {float(uncorrected):.3f}"
                )
            elif status:
                details.append("association measured")
            measurements.append(
                (
                    f"{coefficient} and {technical}",
                    display,
                    ("; ".join(details) + ".") if details else "",
                )
            )
    return measurements


def _render_covariate_evidence(experimental: Mapping[str, Any]) -> str:
    characterization = _mapping(experimental.get("characterization"))
    columns = _mappings(characterization.get("columns"))
    if not columns:
        return ""
    domains = Counter(str(item.get("domain") or "unclassified") for item in columns)
    domain_labels = {
        "biological": "Biological variables",
        "technical": "Technical variables",
        "design": "Study-design variables",
        "ignore": "Excluded metadata",
        "unclassified": "Unclassified metadata",
    }
    role_choices = [
        {
            "label": domain_labels.get(domain, _label(domain)),
            "status": "Reviewed",
            "state": "reviewed",
            "metrics": [f"Columns: {count:,}"],
            "reason": "",
        }
        for domain, count in sorted(domains.items())
    ]
    coefficients = _mappings(characterization.get("coefficients"))
    coefficient_names = [
        _public_field_label(item.get("name"))
        for item in coefficients
        if _public_field_label(item.get("name"))
    ]
    outcome = (
        f"{len(columns):,} columns reviewed; "
        f"{_format_text_list(coefficient_names)} selected as study comparisons"
        if coefficient_names
        else f"{len(columns):,} metadata columns reviewed"
    )
    measurements: list[tuple[str, str, str]] = []
    for coefficient in coefficients:
        rows = coefficient.get("designRows")
        observation = _public_field_label(coefficient.get("observationUnit"))
        independent = _public_field_label(coefficient.get("independentUnit"))
        scope = {
            "betweenUnit": "between independent units",
            "withinUnit": "within independent units",
            "mixed": "within and between independent units",
        }.get(
            str(coefficient.get("scope") or ""),
            _label(coefficient.get("scope")).lower(),
        )
        measurements.append(
            (
                _public_field_label(coefficient.get("name")),
                (
                    f"{int(rows):,} {observation} records"
                    if isinstance(rows, int)
                    else "Selected biological comparison"
                ),
                (f"Independent unit: {independent}; comparison type: {scope}."),
            )
        )
    for nesting in _mappings(characterization.get("technicalNesting")):
        left = _public_field_label(nesting.get("left"))
        right = _public_field_label(nesting.get("right"))
        measurements.append(
            (
                "Technical nesting",
                f"{right} is nested within {left}",
                "This structure limits which technical effects can be separated.",
            )
        )
    measurements.extend(_covariate_pair_measurements(characterization))
    return _render_evidence_panel(
        title="Covariate analysis",
        outcome=outcome,
        introduction=(
            "Metadata were classified by role before correction or clustering. "
            "The review separated biological comparisons from technical structure "
            "and metadata that should not guide the analysis."
        ),
        body=_render_evidence_choices(role_choices),
        measurements=_render_evidence_measurements(measurements),
    )


def _feature_family_counts(
    enrichment: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    counts: dict[tuple[str, str], Mapping[str, Any]] = {}
    for inspection in _mappings(enrichment.get("inspections")):
        assay = str(inspection.get("assay") or "")
        for family in _mappings(inspection.get("families")):
            counts[(assay, str(family.get("family") or ""))] = family
    return counts


def _default_inventory_family_rows(
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "family": _feature_family_label(family.get("family")),
            "pattern": family.get("pattern"),
            "matched genes": family.get("count"),
            "examples": _text_values(family.get("examples")),
        }
        for family in _mappings(inventory.get("families"))
    ]


def _render_default_inventory_summary(
    inventory: Mapping[str, Any],
    *,
    heading: str,
) -> str:
    if not inventory:
        return ""
    match_count = inventory.get("matchCount")
    total_features = inventory.get("totalFeatures")
    applied = inventory.get("appliedToSelectedRepresentation") is True
    count_summary = (
        f"{int(match_count):,} of {int(total_features):,} genes matched."
        if isinstance(match_count, int) and isinstance(total_features, int)
        else "The exact default pattern was evaluated."
    )
    effect = (
        "The complete default blacklist was applied to the selected representation."
        if applied
        else (
            "The complete default blacklist was evaluated as a reference but was "
            "not applied wholesale to the selected representation."
        )
    )
    blacklist = str(inventory.get("blacklist") or "")
    pattern_markup = (
        "<p><strong>Exact combined pattern:</strong> "
        f"<code>{html.escape(blacklist)}</code></p>"
        if blacklist
        else ""
    )
    return (
        f'<div class="subsection"><h3>{html.escape(heading)}</h3>'
        f"<p>{html.escape(count_summary)} {html.escape(effect)}</p>"
        + _table(
            _default_inventory_family_rows(inventory),
            columns=("family", "pattern", "matched genes", "examples"),
            empty="No default blacklist families were recorded.",
        )
        + pattern_markup
        + "</div>"
    )


def _render_normalization_evidence(
    enrichment: Mapping[str, Any],
    plan: Mapping[str, Any],
    inventories: Sequence[Mapping[str, Any]],
) -> str:
    assay_plans = _mappings(plan.get("assays"))
    if not assay_plans:
        return ""
    families = _feature_family_counts(enrichment)
    choices: list[dict[str, Any]] = []
    measurements: list[tuple[str, str, str]] = []
    outcome_parts: list[str] = []
    for assay_plan in assay_plans:
        assay = str(assay_plan.get("assay") or "Assay")
        normalization = _mapping(assay_plan.get("normalizationParameters"))
        feature_parameters = _mapping(assay_plan.get("featureParameters"))
        inventory = _default_inventory_for_assay(inventories, assay)
        log_transform = normalization.get("logTransform") is True
        renormalize = normalization.get("renormalizeSubset") is True
        normalization_metrics = [
            "Log transform applied" if log_transform else "No log transform",
            (
                "Selected cells renormalized"
                if renormalize
                else "Existing normalization retained"
            ),
        ]
        choices.append(
            {
                "label": f"{_assay_label(assay)} normalization",
                "status": "Selected",
                "state": "selected",
                "metrics": normalization_metrics,
                "reason": "Used consistently for map construction.",
            }
        )
        excluded = [
            str(value)
            for value in _text_values(feature_parameters.get("excludeFamilies"))
        ]
        protected = [
            str(value)
            for value in _text_values(feature_parameters.get("protectFamilies"))
        ]
        if excluded:
            choices.append(
                {
                    "label": f"Exclude {_format_text_list([_feature_family_label(value) for value in excluded])}",
                    "status": "Excluded from map",
                    "state": "rejected",
                    "metrics": [
                        "Still available for marker testing",
                    ],
                    "reason": (
                        "Excluded only from map-building features to reduce "
                        "unwanted signal."
                    ),
                }
            )
        if protected:
            choices.append(
                {
                    "label": f"Protect {_format_text_list([_feature_family_label(value) for value in protected])}",
                    "status": "Preserved",
                    "state": "selected",
                    "metrics": ["Remained eligible for map construction"],
                    "reason": (
                        "Protected so biological structure was not removed as "
                        "technical noise."
                    ),
                }
            )
        if inventory:
            default_applied = inventory.get("appliedToSelectedRepresentation") is True
            match_count = inventory.get("matchCount")
            total_features = inventory.get("totalFeatures")
            choices.append(
                {
                    "label": "Exact Scarf default HVG blacklist",
                    "status": (
                        "Applied to selected representation"
                        if default_applied
                        else "Evaluated as reference"
                    ),
                    "state": "selected" if default_applied else "reviewed",
                    "metrics": (
                        [
                            f"Matched {int(match_count):,} of "
                            f"{int(total_features):,} genes"
                        ]
                        if isinstance(match_count, int)
                        and isinstance(total_features, int)
                        else []
                    ),
                    "reason": (
                        "Applied as the complete selected representation blacklist."
                        if default_applied
                        else (
                            "Not applied wholesale; the final policy used only the "
                            "families supported by the decision evidence."
                        )
                    ),
                }
            )
        outcome_parts.append(
            f"{_assay_label(assay)} log normalization"
            if log_transform
            else f"{_assay_label(assay)} normalization"
        )
        if excluded:
            outcome_parts.append(
                f"excluded {_format_text_list([_feature_family_label(value) for value in excluded])} from map construction"
            )
        if inventory and not inventory.get("appliedToSelectedRepresentation"):
            outcome_parts.append(
                "complete Scarf default blacklist not applied wholesale"
            )
        for family_name in dict.fromkeys([*excluded, *protected]):
            family = _mapping(families.get((assay, family_name)))
            count = family.get("count")
            skipped = family.get("skipped")
            action = (
                "Excluded from map construction"
                if family_name in excluded
                else "Protected and retained"
            )
            measurements.append(
                (
                    _feature_family_label(family_name).capitalize(),
                    (
                        "Not counted"
                        if skipped
                        else (
                            f"{int(count):,} identified features"
                            if isinstance(count, int)
                            else "Feature count unavailable"
                        )
                    ),
                    (
                        f"{action}. Inspection was skipped because "
                        f"{_label(skipped).lower()}."
                        if skipped
                        else f"{action}."
                    ),
                )
            )
        min_cells = feature_parameters.get("minCells")
        if isinstance(min_cells, int):
            measurements.append(
                (
                    f"{_assay_label(assay)} detection requirement",
                    f"Present in at least {min_cells:,} cells",
                    "Applied before variable-gene ranking.",
                )
            )
    inventory_markup = "".join(
        _render_default_inventory_summary(
            inventory,
            heading=f"{_assay_label(inventory.get('assay'))} default blacklist audit",
        )
        for inventory in inventories
    )
    return _render_evidence_panel(
        title="Normalization and feature policy",
        outcome="; ".join(outcome_parts),
        introduction=(
            "Normalization and feature-family rules were fixed before tuning. "
            "Representation exclusions changed the map-building features, not the "
            "genes available for marker analysis."
        ),
        body=_render_evidence_choices(choices) + inventory_markup,
        measurements=_render_evidence_measurements(measurements),
    )


def _render_batch_evidence(
    experimental: Mapping[str, Any],
    parameter: Mapping[str, Any],
    final: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> str:
    decision = _mapping(experimental.get("decision"))
    batch_plan = _mapping(decision.get("batchCorrection"))
    safety = _mappings(experimental.get("batchSafety"))
    if not batch_plan and not safety:
        return ""
    native_analyses = _mappings(final.get("nativeAnalyses"))
    if final.get("graphMethod") == "native" and final.get("primaryAssay"):
        native_analyses = [
            item
            for item in native_analyses
            if item.get("assay") == final.get("primaryAssay")
        ]
    adjusted = any(_present(item.get("batchCorrection")) for item in native_analyses)
    native_candidate, harmony_candidate = _harmony_candidate_pair(parameter, final)
    harmony_executed = _harmony_completed(native_candidate) and _harmony_completed(
        harmony_candidate
    )
    degraded = _degraded_protected_columns(native_candidate, harmony_candidate)
    coefficients = list(
        dict.fromkeys(
            _public_field_label(item.get("coefficient"))
            for item in safety
            if _public_field_label(item.get("coefficient"))
        )
    )
    unsafe = any(item.get("status") == "unsafe" for item in safety)
    correction_outcome = _active_decision(decisions, "correctionOutcome")
    correction_license = _active_decision(decisions, "correctionLicense")
    diagnostic_only = str(correction_license.get("selectedOptionId") or "").endswith(
        "unsafeConfounded"
    )
    native_metrics = _mapping(native_candidate.get("metrics"))
    harmony_metrics = _mapping(harmony_candidate.get("metrics"))
    native_batch = _mapping(native_metrics.get("batchMixing"))
    harmony_batch = _mapping(harmony_metrics.get("batchMixing"))
    harmony_choice_metrics: list[str] = []
    if harmony_candidate:
        parameters = _mapping(harmony_candidate.get("parameters"))
        harmony_choice_metrics.append(
            "Matched parameters: "
            f"{_scalar(parameters.get('dimensions'))} dimensions, "
            f"{_scalar(parameters.get('neighborsK'))} neighbors, "
            f"resolution {_scalar(parameters.get('leidenResolution'))}"
        )
    if harmony_executed:
        harmony_choice_metrics.insert(0, "Run status: completed")
        for column in dict.fromkeys([*native_batch, *harmony_batch]):
            harmony_choice_metrics.append(
                f"{_public_field_label(column).capitalize()} mixing: "
                f"{_score_transition(native_batch.get(column), harmony_batch.get(column))}"
            )
        if degraded:
            harmony_choice_metrics.append(
                "Protected evidence degraded: " + _format_text_list(degraded)
            )
    if coefficients:
        harmony_choice_metrics.append(
            f"Design-confounded comparisons: {_format_text_list(coefficients)}"
        )
    if diagnostic_only:
        harmony_choice_metrics.append("Selection license: diagnostic only")
    recorded_rationale = str(correction_outcome.get("rationale") or "").strip()
    if harmony_executed and degraded:
        harmony_reason = (
            "Rejected because protected evidence degraded for "
            f"{_format_text_list(degraded)}"
            + ("; the design license was diagnostic only." if diagnostic_only else ".")
        )
    elif harmony_executed:
        harmony_reason = "Executed as a matched diagnostic but not selected."
    elif unsafe:
        harmony_reason = (
            "Not run because library effects could not be separated safely from "
            "the protected study comparisons."
        )
    else:
        harmony_reason = "No completed matched Harmony diagnostic was recorded."
    choices = [
        {
            "label": "Use the unadjusted representation",
            "status": "Selected" if not adjusted else "Not selected",
            "state": "selected" if not adjusted else "rejected",
            "metrics": ["Protected biological comparisons remain intact"],
            "reason": (
                "Selected after the matched diagnostic retained more protected "
                "biological structure."
                if harmony_executed and not adjusted
                else "Selected because no safe, measurable correction was available."
                if not adjusted
                else "Not selected after the adjusted result showed a safe benefit."
            ),
        },
        {
            "label": "Apply Harmony correction",
            "status": (
                "Run and selected"
                if adjusted
                else (
                    "Run diagnostically; rejected"
                    if harmony_executed
                    else ("Not run" if unsafe else "Not selected")
                )
            ),
            "state": "rejected" if not adjusted else "selected",
            "metrics": harmony_choice_metrics,
            "reason": harmony_reason,
        },
    ]
    measurements: list[tuple[str, str, str]] = []
    for item in safety:
        estimability = _mapping(item.get("estimability"))
        coefficient = _public_field_label(item.get("coefficient"))
        estimable = estimability.get("coefficientEstimable") is True
        rows = estimability.get("rowsUsed")
        rank = estimability.get("rankTechnical")
        residual = estimability.get("residualDf")
        remaining = estimability.get("estimableDf")
        measurements.append(
            (
                f"Harmony safety for {coefficient}",
                "Estimable" if estimable else "Not estimable",
                "; ".join(
                    value
                    for value in (
                        f"Study units: {int(rows):,}" if isinstance(rows, int) else "",
                        f"Technical rank: {int(rank):,}"
                        if isinstance(rank, int)
                        else "",
                        f"Residual degrees of freedom: {int(residual):,}"
                        if isinstance(residual, int)
                        else "",
                        f"Remaining comparison capacity: {int(remaining):,}"
                        if isinstance(remaining, int)
                        else "",
                    )
                    if value
                ),
            )
        )
    measurements.extend(
        _covariate_pair_measurements(_mapping(experimental.get("characterization")))
    )
    outcome = (
        "Harmony completed and was selected"
        if adjusted
        else (
            "Diagnostic Harmony completed; rejected and native representation retained"
            if harmony_executed
            else (
                "Harmony not applied; protected comparisons were not independently estimable"
                if unsafe
                else "No batch correction was selected"
            )
        )
    )
    comparison_markup = (
        '<div class="subsection"><h3>Matched native versus Harmony metrics</h3>'
        + _table(
            _harmony_metric_rows(native_candidate, harmony_candidate),
            columns=(
                "category",
                "metric",
                "native",
                "Harmony",
                "change",
                "interpretation",
            ),
            empty="No matched Harmony measurements were recorded.",
        )
        + "</div>"
        if harmony_executed
        else ""
    )
    rationale_markup = (
        '<div class="callout subsection"><h3>Recorded correction decision</h3>'
        f"<p>{html.escape(recorded_rationale)}</p></div>"
        if recorded_rationale
        else ""
    )
    return _render_evidence_panel(
        title="Harmony and batch correction",
        outcome=outcome,
        introduction=(
            "Selection required measured technical improvement without material "
            "loss of the recorded protected study structure. A diagnostic "
            "run could still be completed when the design was not licensed for "
            "corrected-result selection."
        ),
        body=(_render_evidence_choices(choices) + comparison_markup + rationale_markup),
        measurements=_render_evidence_measurements(measurements),
    )


def _render_hvg_evidence(evidence: Mapping[str, Any]) -> str:
    rankings = _mappings(evidence.get("rankings"))
    candidates = _mappings(evidence.get("candidateMetrics"))
    default_counts = [
        int(value)
        for value in evidence.get("scarfDefaultReferenceCounts", [])
        if isinstance(value, int)
    ]
    if not rankings and not candidates:
        return ""
    selected_mode = evidence.get("selectedRankingMode")
    selected_count = evidence.get("selectedFeatureCount")
    ranking_choices: list[dict[str, Any]] = []
    for ranking in rankings:
        selected = ranking.get("rankingMode") == selected_mode
        ranking_choices.append(
            {
                "label": _hvg_ranking_label(ranking.get("rankingMode")),
                "status": "Selected" if selected else "Not selected",
                "state": "selected" if selected else "rejected",
                "metrics": [
                    "Mean coverage across libraries: "
                    f"{_analysis_percent(ranking.get('meanTechnicalGroupCoverage'))}",
                    "Genes recurring in at least two libraries: "
                    f"{_analysis_percent(ranking.get('recurrentInTwoGroupsFraction'))}",
                ],
                "reason": (
                    "Selected after combining recurrence, exact Scarf-default "
                    "overlap, technical association, and downstream stability."
                    if selected
                    else (
                        "Not selected after the combined upstream and downstream "
                        "comparison."
                    )
                ),
            }
        )
    if default_counts:
        ranking_choices.append(
            {
                "label": "Exact Scarf-default blacklist reference",
                "status": "Reference evaluated",
                "state": "reviewed",
                "metrics": [
                    "Executed set sizes: "
                    + ", ".join(f"{value:,}" for value in default_counts)
                ],
                "reason": (
                    "Used as a fixed comparison reference, not as a selectable "
                    "ranking mode."
                ),
            }
        )
    candidate_choices: list[dict[str, Any]] = []
    for candidate in candidates:
        count = candidate.get("featureCount")
        if not isinstance(count, int):
            continue
        selected = count == selected_count
        candidate_choices.append(
            {
                "label": f"{count:,} variable genes",
                "status": "Selected" if selected else "Not selected",
                "state": "selected" if selected else "rejected",
                "metrics": [
                    "Corrected variance captured: "
                    f"{_analysis_percent(candidate.get('varianceFraction'))}",
                    "Genes recurring across most libraries: "
                    f"{_analysis_percent(candidate.get('recurrentFraction'))}",
                ],
                "reason": (
                    "Selected as the best balance of captured variation and "
                    "cross-library reproducibility."
                    if selected
                    else (
                        "Captured less variation than the selected set."
                        if count < int(selected_count or 0)
                        else "Added genes with substantially lower reproducibility."
                    )
                ),
            }
        )
    measurements = [
        (
            "Eligible genes",
            f"{int(evidence['eligibleFeatureCount']):,}",
            "Genes available after detection and feature-family rules.",
        )
        if isinstance(evidence.get("eligibleFeatureCount"), int)
        else None,
        (
            "Libraries represented",
            f"{int(evidence['validTechnicalGroups']):,}",
            "Registered technical groups used to assess recurrence.",
        )
        if isinstance(evidence.get("validTechnicalGroups"), int)
        else None,
        (
            "Minimum detection",
            f"{int(evidence['minimumDetectedCells']):,} cells",
            "Required before a gene could enter the ranking.",
        )
        if isinstance(evidence.get("minimumDetectedCells"), int)
        else None,
        (
            "Excluded libraries",
            f"{int(evidence['excludedTechnicalGroupCount']):,}",
            "Libraries omitted from the group-aware ranking.",
        )
        if isinstance(evidence.get("excludedTechnicalGroupCount"), int)
        else None,
        (
            "HVG branches executed",
            f"{int(evidence['executedBranchCount']):,}",
            "Global, group-aware, and exact Scarf-default reference branches.",
        )
        if isinstance(evidence.get("executedBranchCount"), int)
        else None,
    ]
    body = (
        '<div class="subsection"><h3>Ranking method</h3>'
        f"{_render_evidence_choices(ranking_choices)}</div>"
        '<div class="subsection"><h3>Number of variable genes</h3>'
        f"{_render_evidence_choices(candidate_choices)}</div>"
    )
    outcome = (
        f"{_hvg_ranking_label(selected_mode)}; {int(selected_count):,} genes selected"
        if isinstance(selected_count, int)
        else f"{_hvg_ranking_label(selected_mode)} selected"
    )
    return _render_evidence_panel(
        title="Highly variable genes (HVGs)",
        outcome=outcome,
        introduction=(
            "The workflow first compared how genes were ranked, then compared three "
            "registered set sizes. Selection combined recurrence, exact "
            "Scarf-default overlap, technical association, and downstream "
            "stability rather than using one metric alone."
        ),
        body=body,
        measurements=_render_evidence_measurements(
            [item for item in measurements if item is not None]
        ),
        expanded=True,
    )


def _render_analysis_evidence(payload: Mapping[str, Any]) -> str:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    plan = _mapping(workflow_result.get("preprocessingPlan"))
    final = _mapping(workflow_result.get("finalAnalysis"))
    enrichment = _latest(reports, "data_enrichment")
    experimental = _latest(reports, "experimental_context")
    parameter = _latest(reports, "parameter_tuning")
    decisions = _mapping(payload.get("activeDecisions"))
    inventories = _mappings(payload.get("defaultFeatureInventories"))
    panels = [
        _render_filtering_evidence(experimental, plan),
        _render_covariate_evidence(experimental),
        _render_normalization_evidence(enrichment, plan, inventories),
        _render_batch_evidence(experimental, parameter, final, decisions),
        _render_hvg_evidence(_mapping(payload.get("hvgEvidence"))),
    ]
    panels = [panel for panel in panels if panel]
    if not panels:
        return ""
    return (
        '<section class="section" id="decision-evidence">'
        "<h2>Evidence behind the decisions</h2>"
        "<p>Open a section to compare the selected and rejected choices. "
        "Each section keeps denser thresholds and scores under Measurements.</p>"
        f'<div class="evidence-accordion">{"".join(panels)}</div></section>'
    )


def _narrative_items(value: Any, keys: Sequence[str]) -> list[str]:
    if not _is_sequence(value):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            text = next(
                (
                    _brief_text(item.get(key))
                    for key in keys
                    if _brief_text(item.get(key))
                ),
                "",
            )
        else:
            text = _brief_text(item)
        if text:
            items.append(text)
    return items


def _render_plain_list(items: Sequence[str]) -> str:
    if not items:
        return ""
    return '<ul class="plain-list">{}</ul>'.format(
        "".join(f"<li>{html.escape(item)}</li>" for item in items)
    )


def _render_analysis_biology(biology: Mapping[str, Any]) -> str:
    interpretations = _mappings(biology.get("clusterInterpretations"))
    observations = _narrative_items(
        biology.get("treatmentObservations"),
        ("observation",),
    )
    follow_ups = _narrative_items(
        biology.get("followUps"),
        ("question", "rationale"),
    )
    if not interpretations and not observations and not follow_ups:
        return ""

    cards = "".join(
        '<article class="interpretation-card">'
        f'<p class="summary-label">Cell group {html.escape(str(item.get("clusterId") or "unresolved"))}</p>'
        f"<h3>{html.escape(str(item.get('proposedIdentity') or 'Unresolved'))}</h3>"
        + (
            f"<p>{html.escape(_brief_text(item.get('rationale')))}</p>"
            if _brief_text(item.get("rationale"))
            else ""
        )
        + (
            '<span class="chip">Tentative interpretation</span>'
            if item.get("identityIsHypothesis") is True
            else ""
        )
        + "</article>"
        for item in interpretations
    )
    interpretation_markup = (
        f'<div class="interpretation-grid">{cards}</div>' if cards else ""
    )
    observation_markup = (
        '<div class="subsection"><h3>Observed group differences</h3>'
        f"{_render_plain_list(observations)}</div>"
        if observations
        else ""
    )
    follow_up_markup = (
        '<div class="subsection"><h3>Recommended follow-up</h3>'
        f"{_render_plain_list(follow_ups)}</div>"
        if follow_ups
        else ""
    )
    return f"""
  <section class="section">
    <h2>Biological interpretation</h2>
    {interpretation_markup}
    {observation_markup}
    {follow_up_markup}
  </section>
"""


def _render_column_list(items: Sequence[str]) -> str:
    if not items:
        return '<p class="empty">No matched feature names were recorded.</p>'
    return '<ul class="plain-list column-list">{}</ul>'.format(
        "".join(f"<li>{html.escape(item)}</li>" for item in items)
    )


def _render_qc_technical_audit(
    experimental: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    profiles = _mappings(experimental.get("qcProfiles"))
    if not profiles:
        return ""
    decision = _mapping(experimental.get("decision"))
    cell_qc = _mapping(plan.get("cellQc"))
    if not cell_qc:
        cell_qc = _mapping(decision.get("cellQc"))
    if not cell_qc:
        cell_qc = _mapping(experimental.get("cellQc"))
    selected = _selected_qc_profile(experimental, cell_qc)
    selected_id = selected.get("profileId")
    selected_name = selected.get("registeredProfile")
    profile_rows = _qc_profile_rows(profiles)
    for row, profile in zip(profile_rows, profiles, strict=True):
        row["_selected"] = bool(
            (selected_id and profile.get("profileId") == selected_id)
            or (
                not selected_id
                and selected_name
                and profile.get("registeredProfile") == selected_name
            )
        )
    selected_metrics = _qc_metric_rows(selected)
    return f"""
  <section class="section" id="qc-audit">
    <h2>Cell QC audit</h2>
    <p>Selected and alternative filtering profiles, diagnostic flags, and every persisted cutoff are shown below. A cutoff in a retain-with-flags profile is diagnostic and did not remove cells.</p>
    <div class="subsection"><h3>Profile comparison</h3>{_table(profile_rows, columns=("profile", "scope", "active cells", "retained cells", "removed cells", "flags", "failed libraries"))}</div>
    <div class="subsection"><h3>Selected-profile metric summary</h3>{_table(selected_metrics, columns=("metric", "scope", "median", "diagnostic reference", "lower cutoff", "upper cutoff", "high flag", "flagged cells"))}</div>
    <details><summary>All global and per-library cutoffs</summary>{_table(_qc_bound_rows(profiles), columns=("profile", "group", "metric", "median", "lower removal", "upper removal", "upper flag"), empty="No persisted QC cutoffs were recorded.")}</details>
  </section>
"""


def _render_feature_technical_audit(
    plan: Mapping[str, Any],
    inventories: Sequence[Mapping[str, Any]],
) -> str:
    if not inventories:
        return ""
    policy_rows: list[dict[str, Any]] = []
    for assay_plan in _mappings(plan.get("assays")):
        parameters = _mapping(assay_plan.get("featureParameters"))
        if not parameters:
            continue
        policy_rows.append(
            {
                "assay": assay_plan.get("assay"),
                "selected features": parameters.get("topN"),
                "minimum detected cells": parameters.get("minCells"),
                "excluded families": _text_values(parameters.get("excludeFamilies")),
                "protected families": _text_values(parameters.get("protectFamilies")),
                "complete default blacklist applied": (
                    parameters.get("useScarfDefaultBlacklist") is True
                ),
            }
        )
    inventory_markup: list[str] = []
    for inventory in inventories:
        assay = _assay_label(inventory.get("assay")) or "Assay"
        match_count = inventory.get("matchCount")
        names = _text_values(inventory.get("matchedFeatures"))
        inventory_markup.append(
            _render_default_inventory_summary(
                inventory,
                heading=f"{assay} exact Scarf-default blacklist",
            )
            + "<details><summary>All "
            + (
                f"{int(match_count):,}"
                if isinstance(match_count, int)
                else f"{len(names):,}"
            )
            + " matched feature names</summary>"
            + _render_column_list(names)
            + "</details>"
        )
    return f"""
  <section class="section" id="feature-audit">
    <h2>Normalization and feature-selection audit</h2>
    <p>The selected representation policy is separate from the exact Scarf-default blacklist reference. Genes excluded from map construction remained available to marker testing.</p>
    {_table(policy_rows, columns=("assay", "selected features", "minimum detected cells", "excluded families", "protected families", "complete default blacklist applied"))}
    {"".join(inventory_markup)}
  </section>
"""


def _render_harmony_technical_audit(
    experimental: Mapping[str, Any],
    parameter: Mapping[str, Any],
    final: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> str:
    native, harmony = _harmony_candidate_pair(parameter, final)
    if not native and not harmony:
        return ""
    outcome = _active_decision(decisions, "correctionOutcome")
    license_record = _active_decision(decisions, "correctionLicense")
    rationale = str(outcome.get("rationale") or "").strip()
    license_option = str(license_record.get("selectedOptionId") or "not recorded")
    license_label = _label(license_option.rpartition(":")[2])
    run_status = (
        "completed" if _harmony_completed(harmony) else _scalar(harmony.get("status"))
    )
    rationale_markup = (
        '<div class="callout subsection"><h3>Recorded rejection rationale</h3>'
        f"<p>{html.escape(rationale)}</p></div>"
        if rationale
        else ""
    )
    candidate_rows = [
        {
            "candidate": "Native",
            "status": native.get("status"),
            "eligible": native.get("eligible"),
            **_mapping(native.get("parameters")),
        },
        {
            "candidate": "Harmony",
            "status": harmony.get("status"),
            "eligible": harmony.get("eligible"),
            **_mapping(harmony.get("parameters")),
        },
    ]
    safety_rows: list[dict[str, Any]] = []
    for item in _mappings(experimental.get("batchSafety")):
        estimability = _mapping(item.get("estimability"))
        safety_rows.append(
            {
                "comparison": _public_field_label(item.get("coefficient")),
                "status": item.get("status"),
                "study units": estimability.get("rowsUsed"),
                "technical rank": estimability.get("rankTechnical"),
                "residual degrees of freedom": estimability.get("residualDf"),
                "remaining capacity": estimability.get("estimableDf"),
            }
        )
    return f"""
  <section class="section" id="harmony-audit">
    <h2>Harmony diagnostic audit</h2>
    <p>Run status: <strong>{html.escape(run_status)}</strong>. Selection license: <strong>{html.escape(license_label)}</strong>.</p>
    <div class="subsection"><h3>Matched candidates</h3>{_table(candidate_rows, columns=("candidate", "status", "eligible", "dimensions", "neighborsK", "leidenResolution", "useHarmony"))}</div>
    <div class="subsection"><h3>Native versus Harmony measurements</h3>{_table(_harmony_metric_rows(native, harmony), columns=("category", "metric", "native", "Harmony", "change", "interpretation"))}</div>
    <div class="subsection"><h3>Design safety</h3>{_table(safety_rows, columns=("comparison", "status", "study units", "technical rank", "residual degrees of freedom", "remaining capacity"), empty="No design-safety rows were recorded.")}</div>
    {rationale_markup}
  </section>
"""


def _analysis_limitations(payload: Mapping[str, Any]) -> list[str]:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    final = _mapping(workflow_result.get("finalAnalysis"))
    parameter = _latest(reports, "parameter_tuning")
    biology = _latest(reports, "biological_interpretation")
    interpretations = _mappings(biology.get("clusterInterpretations"))
    limitations: list[str] = []
    if not interpretations:
        limitations.append(
            "No biological cell-type interpretation was generated, so the cell "
            "groups should not be treated as named cell types."
        )
    elif any(item.get("identityIsHypothesis") is True for item in interpretations):
        limitations.append(
            "Cell-group identities are hypotheses based on observed marker patterns "
            "and need independent validation."
        )
    if _mappings(biology.get("treatmentObservations")):
        limitations.append(
            "Reported group differences are descriptive and do not establish cause "
            "and effect."
        )
    if parameter.get("totalCandidates"):
        limitations.append(
            "The final result was selected only from the analysis settings that "
            "were explicitly evaluated."
        )
    if _present(final.get("limitations")) or _present(parameter.get("limitations")):
        limitations.append(
            "Additional technical limitations are recorded in the technical report."
        )
    if _present(payload.get("plotNotes")):
        limitations.append(
            "Some optional visualizations were unavailable; the technical report "
            "records the reason."
        )
    return limitations


def _render_analysis_document(payload: Mapping[str, Any]) -> str:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    plan = _mapping(workflow_result.get("preprocessingPlan"))
    biology = _latest(reports, "biological_interpretation")
    cluster_counts = {
        str(key): int(value)
        for key, value in _mapping(payload.get("clusterCounts")).items()
    }
    plots = {
        str(key): str(value)
        for key, value in _mapping(payload.get("plotFiles")).items()
    }
    objective, organisms, tissues = _study_overview(payload)
    assays = _report_assays(plan)
    total_cells = sum(cluster_counts.values())
    metrics = _render_metrics(
        (
            ("Cells analyzed", total_cells or None),
            ("Cell groups", len(cluster_counts) or None),
            ("Data analyzed", _format_text_list(assays) or None),
        )
    )
    source = _biological_source(organisms, tissues) or "Not specified"
    source = source[:1].upper() + source[1:]
    tree_stages = _analysis_tree_stages(payload)
    selection_evidence = _render_selection_evidence(payload)
    decision_evidence = _render_analysis_evidence(payload)
    analysis_plots = _render_plots(
        plots,
        (),
        order=("umapClusters", "clusterComposition", "markerHeatmap"),
        titles={
            "umapClusters": (
                "Final cell map",
                "Each point is a cell, colored by its selected cell group.",
            ),
            "clusterComposition": (
                "Relative group sizes",
                "The relative size of each selected cell group.",
            ),
            "markerHeatmap": (
                "Marker patterns",
                "Features that help distinguish the selected cell groups.",
            ),
        },
        show_provenance=False,
        show_notes=False,
        empty_message=(
            "Visual results are unavailable for this report. Technical details "
            "record the reason."
        ),
    )
    diagnostic_plot_order = (
        "qcDistributionsBeforeFiltering",
        "qcDistributions",
        *(name for name in plots if name.startswith("qcDistributionDerived")),
        "hvgGlobal",
        "hvgBatchAware",
    )
    diagnostic_plots = (
        _render_plots(
            plots,
            (),
            order=diagnostic_plot_order,
            show_provenance=False,
            show_notes=False,
        )
        if any(name in plots for name in diagnostic_plot_order)
        else ""
    )
    diagnostic_section = (
        """
  <section class="section">
    <h2>Quality control and variable-gene diagnostics</h2>
    <p>QC panels show the selected cutoff annotations. HVG panels highlight the genes retained by each executed ranking at the selected feature count.</p>
    {plots}
  </section>
""".format(plots=diagnostic_plots)
        if diagnostic_plots
        else ""
    )
    limitations = _analysis_limitations(payload)
    biology_markup = _render_analysis_biology(biology)
    body = f"""  <p class="eyebrow">Analysis summary</p>
  <h1>The analysis, at a glance.</h1>
  <p class="lead">{html.escape(objective)}</p>
  {metrics}

  <section class="section">
    <h2>What was analyzed</h2>
    <div class="summary-grid">
      <article class="summary-card">
        <p class="summary-label">Biological source</p>
        <h3>{html.escape(source)}</h3>
      </article>
      <article class="summary-card">
        <p class="summary-label">Final result</p>
        <h3>{total_cells:,} cells organized into {len(cluster_counts):,} groups</h3>
      </article>
    </div>
  </section>

  <section class="section">
    <h2>Analysis decision tree</h2>
    <p>Each decision shows the selected branch, the alternatives considered, their measured values, and why the selected path continued.</p>
    {_render_decision_tree(tree_stages)}
  </section>

  {decision_evidence}

  {diagnostic_section}

  <section class="section">
    <h2>Why the final result was selected</h2>
    <p>These are the main measurements supporting the final cell map. Values closer to 1 indicate stronger agreement for the stability and coherence measures.</p>
    {selection_evidence or '<p class="empty">No final selection measurements were available.</p>'}
  </section>

  <section class="section">
    <h2>Visual results</h2>
    {analysis_plots}
  </section>

  {biology_markup}

  <section class="section">
    <h2>Limitations</h2>
    {_render_plain_list(limitations) if limitations else "<p>No additional user-facing limitations were recorded.</p>"}
  </section>

  <aside class="product-callout">
    <p><strong>Need the exact methods?</strong><br>The technical report contains parameters, comparisons, provenance, and the complete structured record.</p>
    <a class="pill" href="technical.html">Open technical details</a>
  </aside>
"""
    return _render_report_shell(
        title="Scarf analysis summary",
        active_page="analysis",
        body=body,
    )


def _render_technical_document(payload: Mapping[str, Any]) -> str:
    reports = _mapping(payload.get("reports"))
    workflow_result = _mapping(payload.get("workflowResult"))
    request = _mapping(payload.get("request"))
    final = _mapping(workflow_result.get("finalAnalysis"))
    plan = _mapping(workflow_result.get("preprocessingPlan"))
    enrichment = _latest(reports, "data_enrichment")
    experimental = _latest(reports, "experimental_context")
    parameter = _latest(reports, "parameter_tuning")
    biology = _latest(reports, "biological_interpretation")
    decisions = _mapping(payload.get("activeDecisions"))
    inventories = _mappings(payload.get("defaultFeatureInventories"))
    cluster_counts = {
        str(key): int(value)
        for key, value in _mapping(payload.get("clusterCounts")).items()
    }
    top_markers = _mappings(payload.get("topMarkers"))
    plots = {
        str(key): str(value)
        for key, value in _mapping(payload.get("plotFiles")).items()
    }
    plot_notes = [str(item) for item in payload.get("plotNotes", [])]
    attempts = _mappings(payload.get("stageAttempts"))
    resumes = _mappings(payload.get("workflowResumes"))
    workflow = _mapping(workflow_result.get("workflowRun"))
    workflow_id = str(workflow.get("workflowRunId") or "unavailable")
    total_cells = sum(cluster_counts.values())
    assay_plans = _mappings(plan.get("assays"))
    assays = [str(item.get("assay")) for item in assay_plans if item.get("assay")]
    doublet_evidence = _mapping(final.get("doubletEvidence"))
    marker_evidence = _mapping(final.get("markerEvidence"))
    metrics = [
        ("Final cells", total_cells or None),
        ("Final clusters", len(cluster_counts) or None),
        ("Assays", ", ".join(assays) or None),
        ("Candidates", parameter.get("totalCandidates")),
        ("Selected graph", final.get("graphMethod")),
        ("Marker assay", final.get("markerAssay")),
        ("Marker specificity", marker_evidence.get("specificityMedian")),
        ("Doublet capture coverage", doublet_evidence.get("captureCoverage")),
        (
            "Statistical test artifacts",
            len(_mappings(final.get("statisticalTests"))) or None,
        ),
    ]
    metric_markup = _render_metrics(metrics)
    interpretation = {
        "status": biology.get("status"),
        "clusterInterpretations": biology.get("clusterInterpretations"),
        "evidenceIds": biology.get("evidenceIds"),
        "stopReason": biology.get("stopReason"),
    }
    study = enrichment.get("studyContextSummary") or {
        "originalContext": request.get("studyContext")
    }
    enrichment_summary = {
        "status": enrichment.get("status"),
        "policies": enrichment.get("policies"),
        "inspections": enrichment.get("inspections"),
        "defaultFeatureEvidence": enrichment.get("defaultFeatureEvidence"),
        "evidenceIds": enrichment.get("evidenceIds"),
        "unresolvedQuestions": enrichment.get("unresolvedQuestions"),
    }
    experimental_summary = {
        "status": experimental.get("status"),
        "decision": experimental.get("decision"),
        "cellQc": experimental.get("cellQc"),
        "qcProfiles": experimental.get("qcProfiles"),
        "batchSafety": experimental.get("batchSafety"),
        "characterization": experimental.get("characterization"),
        "contrastPlans": experimental.get("contrastPlans"),
    }
    preprocessing_summary = {
        "primaryAssay": plan.get("primaryAssay"),
        "markerAssay": plan.get("markerAssay"),
        "pairedAssays": plan.get("pairedAssays"),
        "cellQc": plan.get("cellQc"),
        "assays": plan.get("assays"),
        "planChecksum": plan.get("planChecksum"),
    }
    limitations = {
        "Data Enrichment": enrichment.get("limitations"),
        "Experimental Context": experimental.get("notes"),
        "Parameter Tuning": parameter.get("limitations"),
        "Biological Interpretation": biology.get("limitations"),
        "Final analysis": final.get("limitations"),
        "Workflow": workflow_result.get("notes"),
        "Plots": plot_notes,
    }
    limitations = {key: value for key, value in limitations.items() if _present(value)}
    marker_columns = [
        key
        for key in (
            "group_id",
            "feature_name",
            "feature_id",
            "score",
            "frac_exp",
            "fold_change",
            "p_value",
        )
        if any(key in row for row in top_markers)
    ]
    provenance: list[dict[str, Any]] = [
        {"field": "Workflow run ID", "value": workflow_id},
        {"field": "Scarf version", "value": __version__},
        {"field": "Workspace", "value": workflow.get("workspace")},
        {"field": "Analysis store", "value": workflow.get("analysisStore")},
        {"field": "Dataset fingerprints", "value": workflow.get("datasetFingerprints")},
        {"field": "Generated at", "value": payload.get("generatedAt")},
        {"field": "Source path", "value": request.get("sourcePath")},
    ]
    raw_json = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, default=str
    )
    biology_nav = (
        '<a class="pill pill-outline" href="#biology">Biology</a>' if biology else ""
    )
    biology_markup = (
        f"""
  <section class="section" id="biology">
    <h2>Biological interpretation</h2>
    {_value(interpretation)}
    <div class="subsection"><h3>Treatment observations</h3>{_value(biology.get("treatmentObservations"))}</div>
    <div class="subsection"><h3>Follow-up recommendations</h3>{_value(biology.get("followUps"))}</div>
  </section>
"""
        if biology
        else ""
    )
    qc_audit = _render_qc_technical_audit(experimental, plan)
    feature_audit = _render_feature_technical_audit(plan, inventories)
    harmony_audit = _render_harmony_technical_audit(
        experimental,
        parameter,
        final,
        decisions,
    )
    qc_nav = '<a class="pill pill-outline" href="#qc-audit">QC</a>' if qc_audit else ""
    feature_nav = (
        '<a class="pill pill-outline" href="#feature-audit">Features</a>'
        if feature_audit
        else ""
    )
    harmony_nav = (
        '<a class="pill pill-outline" href="#harmony-audit">Harmony</a>'
        if harmony_audit
        else ""
    )
    title = f"Scarf agent report {workflow_id}"
    body = f"""  <p class="eyebrow">Technical report</p>
  <h1>Evidence from an automated analysis.</h1>
  <p class="lead">The workflow completed and its selected artifacts and decisions are summarized here.</p>
  <div class="pill-row">
    <span class="pill">Completed</span>
    <span class="pill pill-outline">{html.escape(_label(workflow_result.get("currentStage") or "completed"))}</span>
  </div>
  {metric_markup}
  <nav class="pill-row" aria-label="Report sections">
    <a class="pill pill-outline" href="#visuals">Visual results</a>
    {biology_nav}
    <a class="pill pill-outline" href="#context">Context</a>
    {qc_nav}
    {feature_nav}
    {harmony_nav}
    <a class="pill pill-outline" href="#tuning">Tuning</a>
    <a class="pill pill-outline" href="#workflow">Workflow</a>
  </nav>

  <aside class="product-callout">
    <p><strong>ScarfWeb</strong><br>Distributed, secure infrastructure for intuitive secondary analysis, browser-native.</p>
    <a class="pill" href="https://www.nygen.io/products/scarfweb" target="_blank" rel="noopener noreferrer">Explore ScarfWeb</a>
  </aside>

  <section class="section" id="visuals">
    <div class="section-heading"><h2>Visual results</h2><span class="pill pill-outline">Persisted artifacts</span></div>
    {_render_plots(plots, plot_notes)}
  </section>

  <section class="section">
    <h2>Final partition evidence</h2>
    <div class="subsection"><h3>Final cluster sizes</h3>{_render_clusters(cluster_counts)}</div>
    <div class="subsection"><h3>Top marker evidence</h3>{_table(top_markers, columns=marker_columns, empty="No marker table was available.")}</div>
    <div class="subsection"><h3>Marker-family summary</h3>{_value(marker_evidence)}</div>
    <div class="subsection"><h3>Advisory doublet summary</h3>{_value(doublet_evidence)}</div>
  </section>

  {biology_markup}

  {qc_audit}
  {feature_audit}
  {harmony_audit}

  <section class="section" id="context"><h2>Study context</h2>{_value(study)}</section>
  <section class="section"><h2>Data enrichment</h2>{_value(enrichment_summary)}</section>
  <section class="section"><h2>Experimental design</h2>{_value(experimental_summary)}</section>
  <section class="section"><h2>Preprocessing plan</h2>{_value(preprocessing_summary)}</section>

  <section class="section" id="tuning"><h2>Parameter tuning and graph selection</h2>{_render_parameter_tuning(parameter)}</section>
  <section class="section"><h2>Bounded analysis review and hypothesis tests</h2>{_value({"analysisEvidence": final.get("analysisEvidence"), "statisticalTests": final.get("statisticalTests")})}</section>
  <section class="section" id="workflow"><h2>Workflow execution</h2>{_render_timeline(attempts, resumes)}</section>
  <section class="section"><h2>Agent execution</h2>{_render_executions(reports)}</section>
  <section class="section"><h2>Limitations and workflow notes</h2>{_value(limitations) if limitations else '<p class="empty">No limitations were recorded.</p>'}</section>

  <section class="section">
    <h2>Technical provenance</h2>
    {_table(provenance)}
    <details><summary>Final immutable artifact references</summary>{_value(final)}</details>
    <details><summary>Structured report data</summary><pre>{html.escape(raw_json)}</pre></details>
  </section>
"""
    return _render_report_shell(
        title=title,
        active_page="technical",
        body=body,
    )
