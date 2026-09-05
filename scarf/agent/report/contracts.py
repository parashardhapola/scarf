"""Shared internal value contracts for agent report generation."""

import re
from collections.abc import Mapping, Sequence
from typing import Any


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _label(value: Any) -> str:
    text = str(value).replace("_", " ").strip()
    words: list[str] = []
    for index, character in enumerate(text):
        if (
            index
            and character.isupper()
            and not text[index - 1].isupper()
            and text[index - 1] != " "
        ):
            words.append(" ")
        words.append(character)
    text = "".join(words)
    return text[:1].upper() + text[1:]


def _scalar(value: Any) -> str:
    if value is None or value == "":
        return "Not provided"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 10_000:
            return f"{value:.3g}"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _is_leaf(value: Any) -> bool:
    return not isinstance(value, Mapping) and not _is_sequence(value)


def _is_simple(value: Any) -> bool:
    if _is_leaf(value):
        return True
    return _is_sequence(value) and all(_is_leaf(item) for item in value)


def _is_mapping_sequence(value: Any) -> bool:
    return (
        _is_sequence(value)
        and bool(value)
        and all(isinstance(item, Mapping) for item in value)
    )


def _latest(reports: Mapping[str, Any], agent_name: str) -> dict[str, Any]:
    values = reports.get(agent_name)
    if isinstance(values, Mapping):
        return dict(values)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        for value in reversed(values):
            if isinstance(value, Mapping):
                return dict(value)
    return {}


def _text_values(value: Any) -> list[str]:
    if not _is_sequence(value):
        return []
    return [str(item).strip() for item in value if _is_leaf(item) and str(item).strip()]


def _specific_references(values: Sequence[str]) -> list[str]:
    unique = list(dict.fromkeys(values))
    return [
        value
        for value in unique
        if not any(
            value.casefold() != other.casefold()
            and value.casefold() in other.casefold()
            for other in unique
        )
    ]


def _brief_text(value: Any, *, max_length: int = 240) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    text = re.sub(
        r"\b[0-9a-f]{64}\b",
        "recorded result",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "recorded value",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b([A-Za-z][A-Za-z0-9]*)_id\b",
        lambda match: _label(match.group(1)).lower(),
        text,
    )
    if not text:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    if len(first_sentence) <= max_length:
        return first_sentence
    shortened = first_sentence[: max_length - 3].rsplit(" ", 1)[0]
    return f"{shortened or first_sentence[: max_length - 3]}..."


def _format_text_list(values: Sequence[str]) -> str:
    items = [value for value in dict.fromkeys(values) if value]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _assay_label(value: Any) -> str:
    labels = {
        "RNA": "RNA",
        "ATAC": "chromatin accessibility",
        "ADT": "protein abundance",
        "HTO": "sample tags",
    }
    text = str(value or "").strip()
    return labels.get(text.upper(), _label(text).lower()) if text else ""


def _selected_qc_profile(
    experimental: Mapping[str, Any],
    cell_qc: Mapping[str, Any],
) -> dict[str, Any]:
    profiles = _mappings(experimental.get("qcProfiles"))
    profile_id = cell_qc.get("profileId")
    if profile_id:
        for profile in profiles:
            if profile.get("profileId") == profile_id:
                return profile
    return profiles[0] if len(profiles) == 1 else {}


def _feature_family_label(value: Any) -> str:
    labels = {
        "ribosomal": "ribosomal genes",
        "ribosomalProtein": "ribosomal protein genes",
        "mitochondrial": "mitochondrial genes",
        "mitoribosomal": "mitoribosomal genes",
        "sex": "sex-linked genes",
        "sexLinked": "sex-linked genes",
        "cellCycle": "cell-cycle genes",
        "cellCycleCcn": "CCN-prefixed genes",
        "hla": "HLA genes",
        "h2": "H2 genes",
        "histone": "histone genes",
    }
    text = str(value or "").strip()
    return labels.get(text, _label(text).lower()) if text else ""


def _public_field_label(value: Any) -> str:
    labels = {
        "T2D": "T2D status",
        "donor_id": "donor",
        "library_id": "library",
        "RNA_nCounts": "RNA counts",
        "RNA_nFeatures": "detected genes",
        "RNA_percentMito": "mitochondrial percentage",
        "RNA_percentRibo": "ribosomal percentage",
        "sample_id": "sample",
        "sex": "sex",
        "tissue": "tissue",
    }
    text = str(value or "").strip()
    if not text:
        return ""
    if text in labels:
        return labels[text]
    return _label(text.removesuffix("_id")).lower()


def _analysis_percent(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "Not available"
    return f"{float(value):.1%}"


def _analysis_number_range(values: Sequence[Any]) -> str:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numbers:
        return "Not available"
    low = min(numbers)
    high = max(numbers)

    def display(value: float) -> str:
        if abs(value) >= 100:
            return f"{value:,.0f}"
        return f"{value:,.3f}".rstrip("0").rstrip(".")

    if low == high:
        return display(low)
    return f"{display(low)} to {display(high)}"


def _qc_resolved_bounds(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct = _mappings(profile.get("resolvedBounds"))
    if direct:
        return direct
    return _mappings(_mapping(profile.get("parameters")).get("resolvedBounds"))
