from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from review_tuner.evaluate import evaluate_predictions
from review_tuner.infer import (
    InferenceLimits,
    batched,
    build_budgeted_prompt,
    generate_predictions,
    iter_prediction_rows,
    main,
    parse_prediction_output,
)
from review_tuner.prompts import OUTPUT_CUE, build_prompt
from review_tuner.schema import DatasetError, Prediction, ReviewExample


def test_batched_preserves_order_and_remainder() -> None:
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"max_input_tokens": 0}, "max_input_tokens"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
    ],
)
def test_inference_limits_are_validated_before_model_loading(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        generate_predictions(
            model_name="unused",
            golden_path="unused.jsonl",
            output_path="unused.jsonl",
            **kwargs,
        )


class FakeBatch(dict[str, object]):
    def __init__(self, width: int, batch_size: int) -> None:
        super().__init__(input_ids=SimpleNamespace(shape=(batch_size, width)))
        self.device: str | None = None

    def to(self, device: str) -> FakeBatch:
        self.device = device
        return self


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, text, *, add_special_tokens):
        tokens = text.split()
        return ["<bos>", *tokens] if add_special_tokens else tokens

    def __call__(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        width = max(len(self.encode(prompt, add_special_tokens=True)) for prompt in prompts)
        return FakeBatch(width=width, batch_size=len(prompts))

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        if tokens and isinstance(tokens[-1], int):
            return json.dumps(
                {
                    "prediction": f"generated-{tokens[-1]}",
                    "severity": "medium",
                    "tags": ["tests"],
                }
            )
        return " ".join(tokens)


class FakeModel:
    device = "test-device"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = kwargs["input_ids"].shape[0]
        input_width = kwargs["input_ids"].shape[-1]
        return [
            [10] * input_width + [20 + index]
            for index in range(batch_size)
        ]


def review_example(example_id: str) -> ReviewExample:
    return ReviewExample(
        id=example_id,
        language="python",
        file_path="service.py",
        context="Context.",
        diff="+ changed",
        target_comment="Review this.",
        tags=("tests",),
    )


def test_prediction_core_batches_and_preserves_ids() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    torch_module = SimpleNamespace(inference_mode=nullcontext)

    rows = list(
        iter_prediction_rows(
            [review_example("one"), review_example("two")],
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            limits=InferenceLimits(batch_size=1, max_input_tokens=200, max_new_tokens=9),
        )
    )

    assert rows == [
        {
            "id": "one",
            "prediction": "generated-20",
            "severity": "medium",
            "tags": ["tests"],
        },
        {
            "id": "two",
            "prediction": "generated-20",
            "severity": "medium",
            "tags": ["tests"],
        },
    ]
    assert all(call[1]["truncation"] is False for call in tokenizer.calls)
    assert all(OUTPUT_CUE in call[0][0] for call in tokenizer.calls)
    assert all(call["max_new_tokens"] == 9 for call in model.calls)


def test_budgeting_truncates_fields_but_preserves_the_output_cue() -> None:
    tokenizer = FakeTokenizer()
    example = ReviewExample(
        id="large",
        language="python",
        file_path="service.py",
        context=" ".join(f"context-{index}" for index in range(100)),
        diff=" ".join(f"diff-{index}" for index in range(200)),
        target_comment="Review this.",
        severity="high",
        tags=("security",),
    )
    shell_size = len(
        tokenizer.encode(
            build_prompt(example, context="", diff=""), add_special_tokens=True
        )
    )
    budget = shell_size + 30

    prompt = build_budgeted_prompt(
        example,
        tokenizer=tokenizer,
        max_input_tokens=budget,
    )

    assert len(tokenizer.encode(prompt, add_special_tokens=True)) <= budget
    assert prompt.endswith(OUTPUT_CUE)
    assert "[truncated]" in prompt
    assert '"severity": "high"' not in prompt

    with pytest.raises(ValueError, match="too small for the prompt"):
        build_budgeted_prompt(
            example,
            tokenizer=tokenizer,
            max_input_tokens=shell_size - 1,
        )


def test_structured_output_parser_validates_contract_and_supports_explicit_legacy_mode() -> None:
    prediction = parse_prediction_output(
        "one",
        '```json\n{"prediction":"Fix it.","severity":"HIGH","tags":["Security"]}\n```',
    )

    assert prediction == Prediction(
        id="one",
        prediction="Fix it.",
        severity="high",
        tags=("security",),
    )
    with pytest.raises(DatasetError, match="missing fields: severity, tags"):
        parse_prediction_output("one", '{"prediction": "Fix it."}')
    with pytest.raises(DatasetError, match="severity must be one of"):
        parse_prediction_output(
            "one",
            '{"prediction":"Fix it.","severity":"urgent","tags":[]}',
            allow_unstructured=True,
        )
    for invalid_severity in ("null", "[]", "{}"):
        with pytest.raises(DatasetError, match="severity must be a string"):
            parse_prediction_output(
                "one",
                f'{{"prediction":"Fix it.","severity":{invalid_severity},"tags":[]}}',
            )
    for invalid_tags in ("null", '"security"', "{}"):
        with pytest.raises(DatasetError, match="tags must be an array"):
            parse_prediction_output(
                "one",
                f'{{"prediction":"Fix it.","severity":"high","tags":{invalid_tags}}}',
            )
    assert parse_prediction_output(
        "one", "Legacy plain-text comment.", allow_unstructured=True
    ) == Prediction(id="one", prediction="Legacy plain-text comment.")

    empty_tags = parse_prediction_output(
        "one", '{"prediction":"Fix it.","severity":"low","tags":[]}'
    )
    assert empty_tags.tags == ()
    assert empty_tags.to_record()["tags"] == []


def test_generated_metadata_reaches_the_evaluator() -> None:
    tokenizer = FakeTokenizer()
    example = review_example("one")
    [row] = list(
        iter_prediction_rows(
            [example],
            tokenizer=tokenizer,
            model=FakeModel(),
            torch_module=SimpleNamespace(inference_mode=nullcontext),
            limits=InferenceLimits(max_input_tokens=200),
        )
    )

    aggregate, [score] = evaluate_predictions(
        [example], [Prediction.from_dict(row)]
    )

    assert aggregate["count"] == 1
    assert score["severity_accuracy"] == 1.0
    assert score["tag_f1"] == 1.0


def test_infer_cli_reports_dataset_errors(monkeypatch, capsys) -> None:
    def fail(**kwargs) -> None:
        raise DatasetError("bad golden data")

    monkeypatch.setattr("review_tuner.infer.generate_predictions", fail)

    assert main(["--model", "model", "--golden", "golden.jsonl"]) == 1
    assert "bad golden data" in capsys.readouterr().err
