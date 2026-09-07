from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from review_tuner.infer import (
    InferenceLimits,
    batched,
    generate_predictions,
    iter_prediction_rows,
    main,
)
from review_tuner.schema import DatasetError, ReviewExample


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
        self.calls: list[tuple[list[str], int]] = []

    def __call__(self, prompts, *, max_length, **kwargs):
        self.calls.append((prompts, max_length))
        return FakeBatch(width=2, batch_size=len(prompts))

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return f" generated-{tokens[-1]} "


class FakeModel:
    device = "test-device"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = kwargs["input_ids"].shape[0]
        return [[10, 11, 20 + index] for index in range(batch_size)]


def review_example(example_id: str) -> ReviewExample:
    return ReviewExample(
        id=example_id,
        language="python",
        file_path="service.py",
        context="Context.",
        diff="+ changed",
        target_comment="Review this.",
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
            limits=InferenceLimits(batch_size=1, max_input_tokens=17, max_new_tokens=9),
        )
    )

    assert rows == [
        {"id": "one", "prediction": "generated-20"},
        {"id": "two", "prediction": "generated-20"},
    ]
    assert [max_length for _, max_length in tokenizer.calls] == [17, 17]
    assert all(call["max_new_tokens"] == 9 for call in model.calls)


def test_infer_cli_reports_dataset_errors(monkeypatch, capsys) -> None:
    def fail(**kwargs) -> None:
        raise DatasetError("bad golden data")

    monkeypatch.setattr("review_tuner.infer.generate_predictions", fail)

    assert main(["--model", "model", "--golden", "golden.jsonl"]) == 1
    assert "bad golden data" in capsys.readouterr().err
