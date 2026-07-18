from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from review_tuner.config import AppConfig
from review_tuner.data import write_jsonl
from review_tuner.prompts import OUTPUT_CUE, build_prompt, build_target
from review_tuner.schema import ReviewExample
from review_tuner.train_qlora import (
    TrainingDataError,
    _dataset_from_generator,
    _dataset_rows,
    dry_run,
    main,
)


def config() -> AppConfig:
    return AppConfig.from_mapping(
        {
            "model": {"name": "example/model"},
            "qlora": {},
            "training": {},
        }
    )


class FakeTokenizer:
    eos_token = "<eos>"

    def encode(self, text, *, add_special_tokens):
        tokens = text.replace(self.eos_token, f" {self.eos_token} ").split()
        return ["<bos>", *tokens] if add_special_tokens else tokens

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return " ".join(tokens)


class FakeDatasetGenerationError(Exception):
    pass


class WrappingDataset:
    @staticmethod
    def from_generator(generator, *, gen_kwargs):
        try:
            return list(generator(**gen_kwargs))
        except Exception as exc:
            raise FakeDatasetGenerationError("dataset generation failed") from exc


def test_train_extra_requires_the_supported_trl_api_floor() -> None:
    project_metadata = (
        Path(__file__).parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    [requirement] = re.findall(
        r'^\s*"(trl[^"]*)",\s*$',
        project_metadata,
        flags=re.MULTILINE,
    )
    minimum = re.search(r">=\s*(\d+)\.(\d+)(?:\.(\d+))?", requirement)

    assert minimum is not None
    assert tuple(int(part or 0) for part in minimum.groups()) >= (0, 16, 0)


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


def test_training_rows_budget_fields_without_truncating_structured_target(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    row = {
        "id": "example-1",
        "language": "python",
        "file_path": "service.py",
        "context": " ".join(f"context-{index}" for index in range(100)),
        "diff": " ".join(f"diff-{index}" for index in range(200)),
        "review_comment": "Preserve this complete structured target.",
        "severity": "high",
        "tags": ["correctness", "tests"],
    }
    write_jsonl(path, [row])
    example = ReviewExample.from_dict(row)
    tokenizer = FakeTokenizer()
    target = build_target(example)
    protected_text = (
        f'{build_prompt(example, context="", diff="")}\n{target}'
        f"{tokenizer.eos_token}"
    )
    minimum = len(tokenizer.encode(protected_text, add_special_tokens=True))
    max_length = minimum + 20

    [formatted] = list(_dataset_rows(str(path), tokenizer, max_length))

    assert len(tokenizer.encode(formatted["text"], add_special_tokens=True)) <= max_length
    assert formatted["text"].endswith(f"{target}{tokenizer.eos_token}")
    assert OUTPUT_CUE in formatted["text"]
    assert "[truncated]" in formatted["text"]

    with pytest.raises(ValueError, match="complete structured target and EOS token"):
        list(_dataset_rows(str(path), tokenizer, minimum - 1))


def test_dataset_factory_unwraps_actionable_training_row_errors(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "too-large",
                "language": "python",
                "file_path": "service.py",
                "context": "Context.",
                "diff": "+ changed",
                "review_comment": "A complete target that cannot fit.",
            }
        ],
    )

    with pytest.raises(
        TrainingDataError,
        match=r"example 'too-large'.*training\.max_length",
    ):
        _dataset_from_generator(
            WrappingDataset,
            path=str(path),
            tokenizer=FakeTokenizer(),
            max_length=1,
            generation_error_type=FakeDatasetGenerationError,
        )


def test_dataset_factory_unwraps_training_schema_errors(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(
        path,
        [
            {
                "id": "invalid-tags",
                "language": "python",
                "file_path": "service.py",
                "context": "Context.",
                "diff": "+ changed",
                "review_comment": "Review this.",
                "tags": "tests",
            }
        ],
    )

    with pytest.raises(TrainingDataError, match=r"tags must be a list of strings"):
        _dataset_from_generator(
            WrappingDataset,
            path=str(path),
            tokenizer=FakeTokenizer(),
            max_length=200,
            generation_error_type=FakeDatasetGenerationError,
        )


def test_dataset_factory_does_not_hide_unrelated_generation_failures() -> None:
    class BrokenDataset:
        @staticmethod
        def from_generator(generator, *, gen_kwargs):
            del generator, gen_kwargs
            try:
                raise OSError("Arrow writer failed")
            except OSError as exc:
                raise FakeDatasetGenerationError("dataset generation failed") from exc

    with pytest.raises(FakeDatasetGenerationError) as raised:
        _dataset_from_generator(
            BrokenDataset,
            path="train.jsonl",
            tokenizer=FakeTokenizer(),
            max_length=100,
            generation_error_type=FakeDatasetGenerationError,
        )

    assert isinstance(raised.value.__cause__, OSError)


def test_train_cli_reports_configuration_errors(capsys) -> None:
    assert main(["--config", "missing.json", "--train", "missing.jsonl"]) == 1
    assert "could not load config" in capsys.readouterr().err


def test_train_cli_reports_training_data_errors_without_a_traceback(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "review_tuner.train_qlora.load_config",
        lambda _: config(),
    )

    def fail_with_training_data_error(*_args, **_kwargs) -> None:
        raise TrainingDataError("train.jsonl: example 'too-large' cannot fit")

    monkeypatch.setattr(
        "review_tuner.train_qlora.train",
        fail_with_training_data_error,
    )

    assert main(["--config", "config.yaml", "--train", "train.jsonl"]) == 1
    captured = capsys.readouterr()
    assert "example 'too-large' cannot fit" in captured.err
    assert "Traceback" not in captured.err
