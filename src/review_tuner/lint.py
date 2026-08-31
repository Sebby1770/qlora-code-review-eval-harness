"""Dataset linter for golden and training JSONL files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from review_tuner.schema import VALID_SEVERITIES


@dataclass(frozen=True)
class LintIssue:
    """One linter finding."""

    level: str
    rule: str
    message: str
    example_id: str | None = None
    row: int | None = None

    def format(self) -> str:
        location = []
        if self.row is not None:
            location.append(f"row {self.row}")
        if self.example_id:
            location.append(self.example_id)
        prefix = f"{':'.join(location)}: " if location else ""
        return f"{self.level}: {prefix}{self.message}"


def _as_phrases(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower() for item in value if str(item).strip()]


def lint_records(records: list[tuple[int, dict]]) -> list[LintIssue]:
    """Lint already-parsed JSON objects. ``records`` is ``(row_number, object)``."""

    issues: list[LintIssue] = []
    seen_ids: dict[str, int] = {}
    for row_number, raw in records:
        example_id = str(raw.get("id") or "").strip() or None

        if not example_id:
            issues.append(
                LintIssue("error", "missing_id", "missing id", row=row_number)
            )
        elif example_id in seen_ids:
            issues.append(
                LintIssue(
                    "error",
                    "duplicate_id",
                    f"duplicate id (first seen on row {seen_ids[example_id]})",
                    example_id=example_id,
                    row=row_number,
                )
            )
        else:
            seen_ids[example_id] = row_number

        diff = str(raw.get("diff") or "")
        if not diff.strip():
            issues.append(
                LintIssue(
                    "error",
                    "empty_diff",
                    "empty diff",
                    example_id=example_id,
                    row=row_number,
                )
            )

        comment = raw.get("review_comment", raw.get("expected_comment", ""))
        if not str(comment or "").strip():
            issues.append(
                LintIssue(
                    "error",
                    "empty_comment",
                    "empty comment",
                    example_id=example_id,
                    row=row_number,
                )
            )

        if "severity" in raw:
            severity = str(raw.get("severity") or "").strip().lower()
            if severity not in VALID_SEVERITIES:
                issues.append(
                    LintIssue(
                        "error",
                        "unknown_severity",
                        f"unknown severity {severity!r}; "
                        f"expected one of {sorted(VALID_SEVERITIES)}",
                        example_id=example_id,
                        row=row_number,
                    )
                )
        else:
            issues.append(
                LintIssue(
                    "warning",
                    "missing_severity",
                    "missing severity; default is medium",
                    example_id=example_id,
                    row=row_number,
                )
            )

        rubric = raw.get("rubric") or {}
        if rubric and not isinstance(rubric, dict):
            issues.append(
                LintIssue(
                    "error",
                    "invalid_rubric",
                    "rubric must be an object",
                    example_id=example_id,
                    row=row_number,
                )
            )
            must: list[str] = []
            avoid: list[str] = []
        else:
            must = _as_phrases((rubric or {}).get("must_mention", []))
            avoid = _as_phrases((rubric or {}).get("avoid", []))

        avoid_set = set(avoid)
        for phrase in must:
            if phrase in avoid_set:
                issues.append(
                    LintIssue(
                        "error",
                        "must_mention_in_avoid",
                        f"must_mention phrase {phrase!r} is also listed in avoid",
                        example_id=example_id,
                        row=row_number,
                    )
                )

        if not must:
            issues.append(
                LintIssue(
                    "warning",
                    "empty_must_mention",
                    "rubric.must_mention is empty",
                    example_id=example_id,
                    row=row_number,
                )
            )

        tags = raw.get("tags") or []
        if not tags:
            issues.append(
                LintIssue(
                    "warning",
                    "empty_tags",
                    "tags are empty",
                    example_id=example_id,
                    row=row_number,
                )
            )

        if not str(raw.get("context") or "").strip():
            issues.append(
                LintIssue(
                    "warning",
                    "empty_context",
                    "empty context",
                    example_id=example_id,
                    row=row_number,
                )
            )

    return issues


def lint_dataset(path: str | Path) -> list[LintIssue]:
    """Lint a JSONL dataset and return errors plus warnings."""

    path = Path(path)
    issues: list[LintIssue] = []
    records: list[tuple[int, dict]] = []

    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(
                    LintIssue("error", "invalid_json", f"invalid JSON: {exc}", row=row_number)
                )
                continue
            if not isinstance(value, dict):
                issues.append(
                    LintIssue("error", "invalid_json", "expected a JSON object", row=row_number)
                )
                continue
            records.append((row_number, value))

    issues.extend(lint_records(records))
    return issues


def lint_has_errors(issues: list[LintIssue]) -> bool:
    return any(issue.level == "error" for issue in issues)
