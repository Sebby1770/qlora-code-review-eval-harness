import json

from review_tuner.data import write_jsonl
from review_tuner.evaluate import main


def test_evaluate_cli_writes_reports(tmp_path) -> None:
    golden_path = tmp_path / "golden.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    aggregate_path = tmp_path / "eval.json"
    examples_path = tmp_path / "eval_examples.jsonl"

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

    exit_code = main(
        [
            "--golden",
            str(golden_path),
            "--predictions",
            str(prediction_path),
            "--out",
            str(aggregate_path),
            "--per-example-out",
            str(examples_path),
            "--fail-under",
            "0.8",
        ]
    )

    assert exit_code == 0
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["count"] == 1
    assert aggregate["composite"] > 0.8
    assert examples_path.read_text(encoding="utf-8")
