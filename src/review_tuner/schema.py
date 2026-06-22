"""Record validation for code review fine-tuning and evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_SEVERITIES = {"blocker", "high", "medium", "low", "nit"}


class DatasetError(ValueError):
    """Raised when a dataset record does not match the expected schema."""


def _required_string(raw: dict[str, Any], name: str, prefix: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{prefix}{name} must be a non-empty string")
    return value


def _string_list(raw: Any, name: str, prefix: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise DatasetError(f"{prefix}{name} must be a list of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value.strip():
            raise DatasetError(
                f"{prefix}{name}[{index}] must be a non-empty string"
            )
        item = value.strip().lower()
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True)
class ReviewExample:
    """One training or golden evaluation example."""

    id: str
    diff: str
    file_path: str
    language: str
    context: str
    target_comment: str
    severity: str = "medium"
    tags: tuple[str, ...] = field(default_factory=tuple)
    must_mention: tuple[str, ...] = field(default_factory=tuple)
    avoid: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, row_number: int | None = None) -> ReviewExample:
        prefix = f"row {row_number}: " if row_number is not None else ""
        review_comment = raw.get("review_comment")
        expected_comment = raw.get("expected_comment")
        if review_comment is not None and expected_comment is not None:
            if review_comment != expected_comment:
                raise DatasetError(
                    f"{prefix}review_comment and expected_comment must not conflict"
                )
            target = review_comment
        else:
            target = review_comment if review_comment is not None else expected_comment
        if not isinstance(target, str) or not target.strip():
            raise DatasetError(
                f"{prefix}review_comment or expected_comment must be a non-empty string"
            )

        severity_value = raw.get("severity", "medium")
        if not isinstance(severity_value, str):
            raise DatasetError(f"{prefix}severity must be a string")
        severity = severity_value.strip().lower()
        if severity not in VALID_SEVERITIES:
            raise DatasetError(
                f"{prefix}severity must be one of {sorted(VALID_SEVERITIES)}, got {severity!r}"
            )

        rubric = raw.get("rubric", {})
        if rubric is None:
            rubric = {}
        if not isinstance(rubric, dict):
            raise DatasetError(f"{prefix}rubric must be an object when provided")

        return cls(
            id=_required_string(raw, "id", prefix).strip(),
            diff=_required_string(raw, "diff", prefix),
            file_path=_required_string(raw, "file_path", prefix).strip(),
            language=_required_string(raw, "language", prefix).strip().lower(),
            context=_required_string(raw, "context", prefix),
            target_comment=target,
            severity=severity,
            tags=_string_list(raw.get("tags"), "tags", prefix),
            must_mention=_string_list(
                rubric.get("must_mention"), "rubric.must_mention", prefix
            ),
            avoid=_string_list(rubric.get("avoid"), "rubric.avoid", prefix),
        )


@dataclass(frozen=True)
class Prediction:
    """Model output aligned to a golden example id."""

    id: str
    prediction: str
    severity: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, row_number: int | None = None) -> Prediction:
        prefix = f"row {row_number}: " if row_number is not None else ""
        prediction_id = _required_string(raw, "id", prefix).strip()
        prediction = _required_string(raw, "prediction", prefix)

        severity = raw.get("severity")
        if severity is not None:
            if not isinstance(severity, str):
                raise DatasetError(f"{prefix}severity must be a string or null")
            severity = severity.strip().lower()
            if severity not in VALID_SEVERITIES:
                raise DatasetError(
                    f"{prefix}severity must be one of {sorted(VALID_SEVERITIES)}, got {severity!r}"
                )

        return cls(
            id=prediction_id,
            prediction=prediction,
            severity=severity,
            tags=_string_list(raw.get("tags"), "tags", prefix),
        )
