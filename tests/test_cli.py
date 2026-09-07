import json

from review_tuner.cli import HELP, main
from review_tuner.data import write_jsonl
from review_tuner.evaluate import main as evaluate_main


def _example_row() -> dict[str, object]:
    return {
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


def _write_pair(tmp_path):
    golden_path = tmp_path / "golden.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    write_jsonl(golden_path, [_example_row()])
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
    return golden_path, prediction_path


def test_cli_hyphen_dispatch_matches_evaluate(tmp_path) -> None:
    golden_path, prediction_path = _write_pair(tmp_path)
    out = tmp_path / "eval.json"
    examples = tmp_path / "eval_examples.jsonl"
    argv = [
        "--golden",
        str(golden_path),
        "--predictions",
        str(prediction_path),
        "--out",
        str(out),
        "--per-example-out",
        str(examples),
    ]
    assert main(argv) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["count"] == 1


def test_cli_eval_subcommand_and_help(tmp_path, capsys) -> None:
    golden_path, prediction_path = _write_pair(tmp_path)
    out = tmp_path / "eval.json"
    examples = tmp_path / "eval_examples.jsonl"
    assert (
        main(
            [
                "eval",
                "--golden",
                str(golden_path),
                "--predictions",
                str(prediction_path),
                "--out",
                str(out),
                "--per-example-out",
                str(examples),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text(encoding="utf-8"))["letter_grade"] in {"A", "B", "C", "D", "F"}

    assert main([]) == 0
    assert "commands:" in capsys.readouterr().out
    assert main(["help"]) == 0
    assert HELP.splitlines()[0] in capsys.readouterr().out
    assert main(["not-a-command"]) == 1
    err = capsys.readouterr().err
    assert "unknown command" in err


def test_cli_baseline_and_lint(tmp_path) -> None:
    golden_path, _prediction_path = _write_pair(tmp_path)
    baseline_path = tmp_path / "baseline.jsonl"
    assert main(["baseline", "--golden", str(golden_path), "--out", str(baseline_path)]) == 0
    assert "one" in baseline_path.read_text(encoding="utf-8")
    assert main(["lint", str(golden_path)]) == 0


def test_evaluate_entrypoint_still_importable() -> None:
    assert callable(evaluate_main)
