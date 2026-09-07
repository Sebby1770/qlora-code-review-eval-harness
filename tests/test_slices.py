from review_tuner.cli import main as cli_main
from review_tuner.data import write_jsonl
from review_tuner.evaluate import heuristic_prediction, iter_scores
from review_tuner.report import format_slices, slice_composites, slices_main
from review_tuner.schema import ReviewExample


def test_slice_composites_group_by_language_severity_and_tag() -> None:
    python_high = ReviewExample(
        id="py",
        diff="- token\n+ no token",
        file_path="auth.py",
        language="python",
        context="Auth.",
        target_comment="Restore token checks.",
        severity="high",
        tags=("security", "tests"),
        must_mention=("token",),
    )
    go_low = ReviewExample(
        id="go",
        diff="- x\n+ y",
        file_path="retry.go",
        language="go",
        context="Retry.",
        target_comment="Keep the retry budget.",
        severity="low",
        tags=("correctness",),
        must_mention=("retry budget",),
    )
    examples = [python_high, go_low]
    predictions = [heuristic_prediction(example) for example in examples]
    scores = list(iter_scores(examples, predictions))
    slices = slice_composites(examples, scores)

    assert set(slices["by_language"]) == {"python", "go"}
    assert set(slices["by_severity"]) == {"high", "low"}
    assert "security" in slices["by_tag"]
    assert "correctness" in slices["by_tag"]
    assert slices["by_language"]["python"] == scores[0].composite
    assert slices["by_severity"]["low"] == scores[1].composite
    rendered = format_slices(slices)
    assert "by language" in rendered
    assert "python" in rendered


def test_slices_cli(tmp_path, capsys) -> None:
    golden_path = tmp_path / "golden.jsonl"
    write_jsonl(
        golden_path,
        [
            {
                "id": "one",
                "language": "rust",
                "file_path": "lib.rs",
                "context": "Overflow.",
                "diff": "- checked_add\n+ plus",
                "expected_comment": "Keep checked_add to avoid overflow.",
                "severity": "high",
                "tags": ["correctness"],
                "rubric": {"must_mention": ["checked_add"]},
            }
        ],
    )
    assert slices_main(["--golden", str(golden_path)]) == 0
    out = capsys.readouterr().out
    assert "rust" in out
    assert cli_main(["slices", "--golden", str(golden_path), "--json"]) == 0
    json_out = capsys.readouterr().out
    assert "by_language" in json_out
