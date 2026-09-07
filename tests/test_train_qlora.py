from __future__ import annotations

import json

from review_tuner.config import AppConfig
from review_tuner.data import write_jsonl
from review_tuner.train_qlora import dry_run, main


def config() -> AppConfig:
    return AppConfig.from_mapping(
        {
            "model": {"name": "example/model"},
            "qlora": {},
            "training": {},
        }
    )


def test_dry_run_validates_both_datasets_and_prints_preview(tmp_path, capsys) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    row = {
        "id": "example-1",
        "language": "python",
        "file_path": "service.py",
        "context": "Context.",
        "diff": "+ changed",
        "review_comment": "Please verify this change.",
    }
    write_jsonl(train_path, [row])
    write_jsonl(eval_path, [{**row, "id": "example-2"}])

    dry_run(config(), str(train_path), str(eval_path))

    summary = json.loads(capsys.readouterr().out)
    assert summary["model"] == "example/model"
    assert summary["train_examples"] == 1
    assert summary["eval_examples"] == 1
    assert summary["first_train_id"] == "example-1"
    assert "Please verify this change" in summary["first_text_preview"]


def test_train_cli_reports_configuration_errors(capsys) -> None:
    assert main(["--config", "missing.json", "--train", "missing.jsonl"]) == 1
    assert "could not load config" in capsys.readouterr().err
