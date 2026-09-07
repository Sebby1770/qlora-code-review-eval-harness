import json

from review_tuner.data import write_jsonl
from review_tuner.evaluate import main as evaluate_main
from review_tuner.metrics import score_example
from review_tuner.report import (
    bootstrap_ci,
    build_eval_report,
    gate_status,
    letter_grade,
    render_html,
    render_markdown,
    report_main,
)
from review_tuner.schema import Prediction, ReviewExample


def _example(example_id: str, language: str = "python") -> ReviewExample:
    return ReviewExample(
        id=example_id,
        diff="- old\n+ new",
        file_path="service.py",
        language=language,
        context="Context.",
        target_comment="Restore the expires_at check and add a regression test.",
        severity="high",
        tags=("security", "tests"),
        must_mention=("expires_at check", "regression test"),
    )


def test_letter_grade_and_gate_boundaries() -> None:
    assert letter_grade(0.90) == "A"
    assert letter_grade(0.899) == "B"
    assert letter_grade(0.80) == "B"
    assert letter_grade(0.70) == "C"
    assert letter_grade(0.60) == "D"
    assert letter_grade(0.599) == "F"
    assert gate_status(0.60) == "pass"
    assert gate_status(0.55) == "weak"
    assert gate_status(0.49) == "fail"
    assert gate_status(0.75, threshold=0.80) == "weak"


def test_bootstrap_ci_is_deterministic_and_skipped_for_tiny_sets() -> None:
    values = [0.10, 0.20, 0.30, 0.80]
    first = bootstrap_ci(values)
    second = bootstrap_ci(values)
    assert first == second
    assert first["low"] <= first["high"]
    assert first["samples"] == 1000
    assert first["seed"] == 1337
    assert bootstrap_ci([0.1, 0.2]) == {}


def test_reports_include_grade_details_and_escape_html() -> None:
    example = _example("one")
    prediction = Prediction(
        id="one",
        prediction=(
            "Restore the expires_at check and add a regression test. "
            "<script>alert(1)</script>"
        ),
        severity="high",
        tags=("security", "tests"),
    )
    score = score_example(example, prediction)
    report = build_eval_report([example], [prediction], [score])
    details = [
        {
            **score.as_dict(),
            "language": example.language,
            "file_path": example.file_path,
            "severity_expected": example.severity,
            "diff": example.diff,
            "expected_comment": example.target_comment,
            "prediction": prediction.prediction,
            "missed_must_mention": [],
            "security_fail": 0.0,
        }
    ]
    markdown = render_markdown(report, details)
    html = render_html(report, details)
    assert f"Grade {report['letter_grade']}" in markdown
    assert "<details>" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "d add" in html


def test_evaluate_cli_writes_markdown_and_html(tmp_path) -> None:
    golden_path = tmp_path / "golden.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    write_jsonl(
        golden_path,
        [
            {
                "id": "one",
                "language": "python",
                "file_path": "auth.py",
                "context": "Refresh token flow.",
                "diff": "- expires_at check\n+ no check",
                "expected_comment": "Restore the expires_at check and add a regression test.",
                "severity": "high",
                "tags": ["security", "tests"],
                "rubric": {"must_mention": ["expires_at check", "regression test"]},
            }
        ],
    )
    write_jsonl(
        prediction_path,
        [
            {
                "id": "one",
                "prediction": "Restore the expires_at check and add a regression test.",
                "severity": "high",
                "tags": ["security", "tests"],
            }
        ],
    )
    aggregate_path = tmp_path / "eval.json"
    examples_path = tmp_path / "eval_examples.jsonl"
    md_path = tmp_path / "eval.md"
    html_path = tmp_path / "eval.html"
    assert (
        evaluate_main(
            [
                "--golden",
                str(golden_path),
                "--predictions",
                str(prediction_path),
                "--out",
                str(aggregate_path),
                "--per-example-out",
                str(examples_path),
                "--report-md",
                str(md_path),
                "--report-html",
                str(html_path),
            ]
        )
        == 0
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["letter_grade"] == "A"
    assert aggregate["gate"] == "pass"
    assert "bootstrap_ci" not in aggregate
    assert md_path.read_text(encoding="utf-8")
    assert "<details>" in html_path.read_text(encoding="utf-8")

    regenerated_md = tmp_path / "from_json.md"
    assert (
        report_main(
            [
                "--eval",
                str(aggregate_path),
                "--examples",
                str(examples_path),
                "--golden",
                str(golden_path),
                "--predictions",
                str(prediction_path),
                "--report-md",
                str(regenerated_md),
            ]
        )
        == 0
    )
    assert regenerated_md.read_text(encoding="utf-8")


def test_build_eval_report_omits_ci_for_two_examples() -> None:
    examples = [_example("one"), _example("two")]
    predictions = [
        Prediction(
            id=example.id,
            prediction=example.target_comment,
            severity="high",
            tags=("security", "tests"),
        )
        for example in examples
    ]
    scores = [
        score_example(example, prediction)
        for example, prediction in zip(examples, predictions, strict=True)
    ]
    report = build_eval_report(examples, predictions, scores)
    assert "bootstrap_ci" not in report
    assert report["letter_grade"] == "A"
