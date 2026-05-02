"""Generate predictions from a base model plus an optional PEFT adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from review_tuner.data import load_examples, write_jsonl
from review_tuner.prompts import build_prompt
from review_tuner.schema import DatasetError


def _missing_training_dependency_error(exc: ModuleNotFoundError) -> RuntimeError:
    return RuntimeError(
        f"Missing optional ML dependency {exc.name!r}. Install with: "
        "python -m pip install -e '.[train]'"
    )


def generate_predictions(
    *,
    model_name: str,
    golden_path: str,
    output_path: str,
    adapter_path: str | None = None,
    max_new_tokens: int = 160,
    load_in_4bit: bool = True,
) -> None:
    """Run model inference over a golden JSONL file and write predictions."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if adapter_path:
            from peft import PeftModel
    except ModuleNotFoundError as exc:
        raise _missing_training_dependency_error(exc) from exc

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if load_in_4bit else "auto",
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    rows = []
    for example in load_examples(golden_path):
        prompt = build_prompt(example)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        rows.append({"id": example.id, "prediction": generated})

    write_jsonl(output_path, rows)


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
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading for inference.",
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
            load_in_4bit=not args.no_4bit,
        )
        print(f"wrote predictions to {Path(args.out)}")
        return 0
    except (DatasetError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
