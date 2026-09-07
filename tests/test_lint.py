from review_tuner.data import write_jsonl
from review_tuner.lint import MIN_COMMENT_LENGTH, lint_example, lint_path, main
from review_tuner.schema import ReviewExample


def valid_row(**overrides) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "one",
        "language": "python",
        "file_path": "service.py",
        "context": "A focused example.",
        "diff": "- old\n+ new",
        "expected_comment": "Please verify the changed behavior with a regression test.",
        "severity": "medium",
        "tags": ["tests"],
        "rubric": {"must_mention": ["changed behavior"], "avoid": ["style"]},
    }
    record.update(overrides)
    return record


def test_lint_example_flags_empty_diff_and_overlap() -> None:
    empty = ReviewExample.from_dict(valid_row(diff="no line changes in this hunk"))
    overlap = ReviewExample.from_dict(
        valid_row(rubric={"must_mention": ["changed behavior"], "avoid": ["changed behavior"]})
    )
    short = ReviewExample.from_dict(valid_row(expected_comment="Too short."))
    missing = ReviewExample.from_dict(valid_row(rubric={}))

    empty_codes = {issue.code for issue in lint_example(empty)}
    assert "empty_diff" in empty_codes
    assert any(issue.code == "overlapping_rubric" for issue in lint_example(overlap))
    short_issues = lint_example(short)
    assert any(issue.code == "short_comment" for issue in short_issues)
    assert MIN_COMMENT_LENGTH == 40
    assert any(issue.code == "missing_rubric" for issue in lint_example(missing))


def test_lint_path_duplicate_ids_and_parse_errors(tmp_path) -> None:
    path = tmp_path / "golden.jsonl"
    write_jsonl(path, [valid_row(), valid_row(id="one")])
    issues = lint_path(path)
    assert any(issue.code == "duplicate_id" and issue.severity == "error" for issue in issues)

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    parse_issues = lint_path(bad)
    assert parse_issues[0].code == "parse_error"
    assert main([str(bad)]) == 1


def test_lint_clean_file_is_ok(tmp_path) -> None:
    path = tmp_path / "golden.jsonl"
    write_jsonl(path, [valid_row()])
    assert main([str(path)]) == 0
    assert main(["--json", str(path)]) == 0
