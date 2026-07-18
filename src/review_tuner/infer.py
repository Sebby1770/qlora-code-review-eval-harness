"""Generate predictions from a base model plus an optional PEFT adapter."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from review_tuner.config import QLoRAConfig
from review_tuner.data import iter_examples, write_jsonl
from review_tuner.modeling import (
    ensure_padding_token,
    generation_pad_token_id,
    make_quantization_config,
    missing_ml_dependency_error,
    torch_dtype,
)
from review_tuner.prompts import build_budgeted_prompt as build_budgeted_prompt
from review_tuner.schema import DatasetError, Prediction, ReviewExample

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class InferenceLimits:
    """Validated generation limits shared by loading and execution paths."""

    max_new_tokens: int = 160
    max_input_tokens: int = 1024
    batch_size: int = 1

    def __post_init__(self) -> None:
        for name in ("max_new_tokens", "max_input_tokens", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")


def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield bounded batches while preserving input order."""

    if size <= 0:
        raise ValueError("batch size must be greater than zero")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _strip_json_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def parse_prediction_output(
    example_id: str,
    generated: str,
    *,
    allow_unstructured: bool = False,
) -> Prediction:
    """Parse one generated JSON prediction, with an explicit legacy fallback."""

    stripped = generated.strip()
    if not stripped:
        raise DatasetError(f"model returned an empty prediction for id {example_id!r}")

    try:
        value = json.loads(_strip_json_fence(stripped))
    except json.JSONDecodeError as exc:
        if allow_unstructured:
            return Prediction(id=example_id, prediction=stripped)
        raise DatasetError(
            f"invalid structured model output for id {example_id!r}: {exc}; "
            "retrain with the current prompt contract or pass --allow-unstructured-output"
        ) from exc

    try:
        if not isinstance(value, dict):
            raise DatasetError("prediction output must be a JSON object")
        expected_fields = {"prediction", "severity", "tags"}
        missing = sorted(expected_fields - set(value))
        unexpected = sorted(set(value) - expected_fields)
        if missing:
            raise DatasetError(f"prediction output is missing fields: {', '.join(missing)}")
        if unexpected:
            raise DatasetError(
                f"prediction output has unexpected fields: {', '.join(unexpected)}"
            )
        if not isinstance(value["severity"], str):
            raise DatasetError("prediction output severity must be a string")
        if not isinstance(value["tags"], list):
            raise DatasetError("prediction output tags must be an array of strings")
        return Prediction.from_dict({"id": example_id, **value})
    except DatasetError as exc:
        raise DatasetError(
            f"invalid structured model output for id {example_id!r}: {exc}"
        ) from exc


def iter_prediction_rows(
    examples: Iterable[ReviewExample],
    *,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    limits: InferenceLimits,
    allow_unstructured_output: bool = False,
) -> Iterator[dict[str, object]]:
    """Generate ordered prediction rows using an already-loaded model runtime."""

    for example_batch in batched(examples, limits.batch_size):
        prompts = [
            build_budgeted_prompt(
                example,
                tokenizer=tokenizer,
                max_input_tokens=limits.max_input_tokens,
            )
            for example in example_batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)
        input_width = encoded["input_ids"].shape[-1]
        if input_width > limits.max_input_tokens:
            raise RuntimeError(
                "tokenizer produced an input wider than the validated prompt budget: "
                f"{input_width} > {limits.max_input_tokens}"
            )
        with torch_module.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=limits.max_new_tokens,
                do_sample=False,
                pad_token_id=generation_pad_token_id(tokenizer),
            )
        for example, tokens in zip(example_batch, output, strict=True):
            generated = tokenizer.decode(
                tokens[input_width:], skip_special_tokens=True
            ).strip()
            prediction = parse_prediction_output(
                example.id,
                generated,
                allow_unstructured=allow_unstructured_output,
            )
            yield prediction.to_record()


def generate_predictions(
    *,
    model_name: str,
    golden_path: str,
    output_path: str,
    adapter_path: str | None = None,
    max_new_tokens: int = 160,
    max_input_tokens: int = 1024,
    batch_size: int = 1,
    load_in_4bit: bool = True,
    allow_unstructured_output: bool = False,
) -> None:
    """Run model inference over a golden JSONL file and write predictions."""

    limits = InferenceLimits(
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
        batch_size=batch_size,
    )

    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        if adapter_path:
            from peft import PeftModel  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise missing_ml_dependency_error(exc) from exc

    qlora_config = QLoRAConfig(load_in_4bit=load_in_4bit)
    quantization_config = make_quantization_config(
        qlora_config,
        torch,
        BitsAndBytesConfig,
    )

    tokenizer = ensure_padding_token(AutoTokenizer.from_pretrained(model_name, use_fast=True))
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype(torch, qlora_config.bnb_4bit_compute_dtype)
        if load_in_4bit
        else "auto",
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    write_jsonl(
        output_path,
        iter_prediction_rows(
            iter_examples(golden_path),
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            limits=limits,
            allow_unstructured_output=allow_unstructured_output,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model id or local path.")
    parser.add_argument("--golden", required=True, help="Golden examples JSONL.")
    parser.add_argument(
        "--out",
        default="reports/predictions.jsonl",
        help="Prediction JSONL output.",
    )
    parser.add_argument("--adapter", help="Optional PEFT adapter path.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading for inference.",
    )
    parser.add_argument(
        "--allow-unstructured-output",
        action="store_true",
        help=(
            "Accept legacy plain-text model output without severity or tags. "
            "Those metrics will score as unavailable."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        generate_predictions(
            model_name=args.model,
            adapter_path=args.adapter,
            golden_path=args.golden,
            output_path=args.out,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            batch_size=args.batch_size,
            load_in_4bit=not args.no_4bit,
            allow_unstructured_output=args.allow_unstructured_output,
        )
        print(f"wrote predictions to {Path(args.out)}")
        return 0
    except (DatasetError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
