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


def test_evaluate_cli_compares_baseline_and_enforces_regression_gate(
    tmp_path, capsys
) -> None:
    golden_path = tmp_path / "golden.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
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
                "expected_comment": "Restore the expires_at check.",
                "severity": "high",
                "tags": ["security"],
                "rubric": {"must_mention": ["expires_at check"]},
            }
        ],
    )
    write_jsonl(
        candidate_path,
        [
            {
                "id": "one",
                "prediction": "Looks fine.",
                "severity": "low",
                "tags": ["style"],
            }
        ],
    )
    write_jsonl(
        baseline_path,
        [
            {
                "id": "one",
                "prediction": "Restore the expires_at check.",
                "severity": "high",
                "tags": ["security"],
            }
        ],
    )

    exit_code = main(
        [
            "--golden",
            str(golden_path),
            "--predictions",
            str(candidate_path),
            "--baseline-predictions",
            str(baseline_path),
            "--out",
            str(aggregate_path),
            "--per-example-out",
            str(examples_path),
            "--max-regression",
            "0.01",
        ]
    )

    assert exit_code == 2
    assert "composite regression" in capsys.readouterr().err
    report = json.loads(aggregate_path.read_text(encoding="utf-8"))
    [row] = [json.loads(line) for line in examples_path.read_text().splitlines()]
    assert report["count"] == 1
    assert report["comparison"]["baseline"]["composite"] > report["composite"]
    assert report["comparison"]["delta"]["composite"] < 0
    assert report["comparison"]["outcomes"]["candidate_losses"] == 1
    assert report["comparison"]["largest_regressions"][0]["id"] == "one"
    assert row["outcome"] == "loss"
    assert row["baseline"]["composite"] > row["composite"]


def test_evaluate_cli_rejects_missing_baseline_and_input_overwrites(
    tmp_path, capsys
) -> None:
    assert main(["--golden", "golden.jsonl", "--max-regression", "0"]) == 1
    assert "requires --baseline-predictions" in capsys.readouterr().err

    golden_path = tmp_path / "golden.jsonl"
    assert (
        main(
            [
                "--golden",
                str(golden_path),
                "--out",
                str(golden_path),
            ]
        )
        == 1
    )
    assert "must not overwrite --golden" in capsys.readouterr().err
