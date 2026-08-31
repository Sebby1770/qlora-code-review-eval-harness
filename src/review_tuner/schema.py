"""Record validation for code review fine-tuning and evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SEVERITY_ORDER = ("blocker", "high", "medium", "low", "nit")
VALID_SEVERITIES = set(SEVERITY_ORDER)


class DatasetError(ValueError):
    """Raised when a dataset record does not match the expected schema."""


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
        target = raw.get("review_comment", raw.get("expected_comment"))
        required = {
            "id": raw.get("id"),
            "diff": raw.get("diff"),
            "file_path": raw.get("file_path"),
            "language": raw.get("language"),
            "context": raw.get("context"),
            "review_comment or expected_comment": target,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise DatasetError(f"{prefix}missing required field(s): {', '.join(missing)}")

        severity = str(raw.get("severity", "medium")).lower()
        if severity not in VALID_SEVERITIES:
            raise DatasetError(
                f"{prefix}severity must be one of {sorted(VALID_SEVERITIES)}, got {severity!r}"
            )

        rubric = raw.get("rubric") or {}
        if not isinstance(rubric, dict):
            raise DatasetError(f"{prefix}rubric must be an object when provided")

        return cls(
            id=str(raw["id"]),
            diff=str(raw["diff"]),
            file_path=str(raw["file_path"]),
            language=str(raw["language"]),
            context=str(raw["context"]),
            target_comment=str(target),
            severity=severity,
            tags=tuple(str(tag).lower() for tag in raw.get("tags", [])),
            must_mention=tuple(str(item).lower() for item in rubric.get("must_mention", [])),
            avoid=tuple(str(item).lower() for item in rubric.get("avoid", [])),
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
        if not raw.get("id") or not raw.get("prediction"):
            raise DatasetError(f"{prefix}prediction records require id and prediction")

        severity = raw.get("severity")
        if severity is not None:
            severity = str(severity).lower()
            if severity not in VALID_SEVERITIES:
                raise DatasetError(
                    f"{prefix}severity must be one of {sorted(VALID_SEVERITIES)}, got {severity!r}"
                )

        return cls(
            id=str(raw["id"]),
            prediction=str(raw["prediction"]),
            severity=severity,
            tags=tuple(str(tag).lower() for tag in raw.get("tags", [])),
        )
