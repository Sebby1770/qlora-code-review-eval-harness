"""Dataset quality checks for golden and training JSONL files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from review_tuner.data import iter_jsonl
from review_tuner.metrics import missed_phrases, normalize_text
from review_tuner.schema import DatasetError, ReviewExample

MIN_COMMENT_LENGTH = 40
ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class LintIssue:
    """One dataset quality finding."""

    path: str
    row_number: int | None
    example_id: str | None
    code: str
    message: str
    severity: str

    def format(self) -> str:
        location = self.path
        if self.row_number is not None:
            location = f"{location}:{self.row_number}"
        identity = f" [{self.example_id}]" if self.example_id else ""
        return f"{location}{identity} {self.severity} {self.code}: {self.message}"


def _diff_has_changes(diff: str) -> bool:
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            return True
    return False


def lint_example(
    example: ReviewExample, *, path: str = "", row_number: int | None = None
) -> list[LintIssue]:
    """Return quality issues for one already-validated example."""

    issues: list[LintIssue] = []

    def add(code: str, message: str, severity: str = WARNING) -> None:
        issues.append(
            LintIssue(
                path=path,
                row_number=row_number,
                example_id=example.id,
                code=code,
                message=message,
                severity=severity,
            )
        )

    if not _diff_has_changes(example.diff):
        add("empty_diff", "diff has no added or removed lines", ERROR)
    if not example.must_mention and not example.avoid:
        add("missing_rubric", "rubric has no must_mention or avoid phrases")
    comment = example.target_comment.strip()
    if len(comment) < MIN_COMMENT_LENGTH:
        add(
            "short_comment",
            f"review comment is {len(comment)} characters; need at least {MIN_COMMENT_LENGTH}",
        )
    overlap = sorted(set(example.must_mention) & set(example.avoid))
    if overlap:
        add(
            "overlapping_rubric",
            "must_mention and avoid share phrases: " + ", ".join(overlap),
            ERROR,
        )
    missing_from_target = missed_phrases(example.target_comment, example.must_mention)
    if missing_from_target:
        add(
            "rubric_not_in_target",
            "must_mention phrases missing from the target comment: "
            + ", ".join(missing_from_target),
        )
    avoid_in_target = [
        phrase
        for phrase in example.avoid
        if normalize_text(phrase) in normalize_text(example.target_comment)
    ]
    if avoid_in_target:
        add(
            "avoid_in_target",
            "avoid phrases appear in the target comment: " + ", ".join(avoid_in_target),
        )
    return issues


def lint_path(path: str | Path) -> list[LintIssue]:
    """Lint one JSONL file, including parse errors and duplicate ids."""

    source = Path(path)
    issues: list[LintIssue] = []
    seen: dict[str, int] = {}
    try:
        rows = list(enumerate(iter_jsonl(source), start=1))
    except DatasetError as exc:
        return [
            LintIssue(
                path=str(source),
                row_number=None,
                example_id=None,
                code="parse_error",
                message=str(exc),
                severity=ERROR,
            )
        ]

    if not rows:
        issues.append(
            LintIssue(
                path=str(source),
                row_number=None,
                example_id=None,
                code="empty_dataset",
                message="dataset contains no records",
                severity=ERROR,
            )
        )
        return issues

    for row_number, record in rows:
        try:
            example = ReviewExample.from_dict(record, row_number=row_number)
        except DatasetError as exc:
            issues.append(
                LintIssue(
                    path=str(source),
                    row_number=row_number,
                    example_id=str(record["id"]) if isinstance(record.get("id"), str) else None,
                    code="parse_error",
                    message=str(exc),
                    severity=ERROR,
                )
            )
            continue
        previous = seen.get(example.id)
        if previous is not None:
            issues.append(
                LintIssue(
                    path=str(source),
                    row_number=row_number,
                    example_id=example.id,
                    code="duplicate_id",
                    message=f"id already appeared on row {previous}",
                    severity=ERROR,
                )
            )
        else:
            seen[example.id] = row_number
        issues.extend(lint_example(example, path=str(source), row_number=row_number))
    return issues


def lint_paths(paths: Sequence[str | Path]) -> list[LintIssue]:
    """Lint many JSONL files in the given order."""

    issues: list[LintIssue] = []
    for path in paths:
        issues.extend(lint_path(path))
    return issues


def _has_errors(issues: Sequence[LintIssue]) -> bool:
    return any(issue.severity == ERROR for issue in issues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="review-eval lint",
        description="Check review JSONL datasets for duplicates, empty diffs, and rubric issues.",
    )
    parser.add_argument("paths", nargs="+", help="JSONL files to lint.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable findings.")
    args = parser.parse_args(argv)
    issues = lint_paths(args.paths)
    if args.json:
        print(json.dumps([asdict(issue) for issue in issues], indent=2, sort_keys=True))
    else:
        if not issues:
            print(f"ok: {len(args.paths)} file(s), no issues")
        else:
            for issue in issues:
                print(issue.format())
            errors = sum(1 for issue in issues if issue.severity == ERROR)
            warnings = len(issues) - errors
            print(f"{errors} error(s), {warnings} warning(s)")
    if _has_errors(issues) or (args.strict and issues):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
