import json
from pathlib import Path

from review_tuner.evaluate import main
from review_tuner.lint import lint_dataset


def _row(**overrides) -> dict:
    row = {
        "id": "ex-1",
        "language": "python",
        "file_path": "a.py",
        "context": "ctx",
        "diff": "- old\n+ new",
        "expected_comment": "please add a test",
        "severity": "medium",
        "tags": ["tests"],
        "rubric": {"must_mention": ["test"], "avoid": ["style"]},
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_lint_duplicate_ids(tmp_path: Path):
    path = tmp_path / "dup.jsonl"
    _write_jsonl(path, [_row(id="same"), _row(id="same")])
    issues = lint_dataset(path)
    assert any(issue.rule == "duplicate_id" and issue.level == "error" for issue in issues)
    assert main(["lint", str(path)]) == 1


def test_lint_empty_diff(tmp_path: Path):
    path = tmp_path / "empty_diff.jsonl"
    _write_jsonl(path, [_row(diff="   ")])
    issues = lint_dataset(path)
    assert any(issue.rule == "empty_diff" and issue.level == "error" for issue in issues)
    assert main(["lint", str(path)]) == 1


def test_lint_empty_comment(tmp_path: Path):
    path = tmp_path / "empty_comment.jsonl"
    _write_jsonl(path, [_row(expected_comment="")])
    issues = lint_dataset(path)
    assert any(issue.rule == "empty_comment" and issue.level == "error" for issue in issues)
    assert main(["lint", str(path)]) == 1


def test_lint_must_mention_also_in_avoid(tmp_path: Path):
    path = tmp_path / "overlap.jsonl"
    _write_jsonl(
        path,
        [_row(rubric={"must_mention": ["style"], "avoid": ["style"]})],
    )
    issues = lint_dataset(path)
    assert any(
        issue.rule == "must_mention_in_avoid" and issue.level == "error" for issue in issues
    )
    assert main(["lint", str(path)]) == 1


def test_lint_unknown_severity(tmp_path: Path):
    path = tmp_path / "sev.jsonl"
    _write_jsonl(path, [_row(severity="critical")])
    issues = lint_dataset(path)
    assert any(issue.rule == "unknown_severity" and issue.level == "error" for issue in issues)
    assert main(["lint", str(path)]) == 1


def test_lint_warnings_only_exit_zero(tmp_path: Path):
    path = tmp_path / "warn.jsonl"
    _write_jsonl(
        path,
        [
            _row(
                tags=[],
                rubric={"must_mention": [], "avoid": ["style"]},
            )
        ],
    )
    issues = lint_dataset(path)
    assert issues
    assert all(issue.level == "warning" for issue in issues)
    assert main(["lint", str(path)]) == 0


def test_golden_set_lints_clean():
    assert main(["lint", "data/golden/code_review_golden.jsonl"]) == 0
