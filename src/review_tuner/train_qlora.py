"""QLoRA supervised fine-tuning entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from review_tuner.data import load_examples
from review_tuner.prompts import examples_to_sft_rows
from review_tuner.schema import DatasetError


def _missing_training_dependency_error(exc: ModuleNotFoundError) -> RuntimeError:
    return RuntimeError(
        f"Missing optional ML dependency {exc.name!r}. Install with: "
        "python -m pip install -e '.[train]'"
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML or JSON training config."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise _missing_training_dependency_error(exc) from exc
    return yaml.safe_load(text)


def _torch_dtype(name: str):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise _missing_training_dependency_error(exc) from exc

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def train(config: dict[str, Any], train_path: str, eval_path: str | None = None) -> None:
    """Run QLoRA fine-tuning using Hugging Face Transformers, PEFT, and TRL."""

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ModuleNotFoundError as exc:
        raise _missing_training_dependency_error(exc) from exc

    model_cfg = config["model"]
    qlora_cfg = config["qlora"]
    train_cfg = config["training"]

    train_rows = examples_to_sft_rows(load_examples(train_path))
    eval_rows = examples_to_sft_rows(load_examples(eval_path)) if eval_path else None

    compute_dtype = _torch_dtype(qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(qlora_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(qlora_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "quantization_config": quantization_config,
        "torch_dtype": compute_dtype,
    }
    if model_cfg.get("device_map") is not None:
        model_kwargs["device_map"] = model_cfg["device_map"]

    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **model_kwargs)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
    )

    target_modules = qlora_cfg.get("target_modules", "all-linear")
    lora_config = LoraConfig(
        r=int(qlora_cfg.get("r", 16)),
        lora_alpha=int(qlora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(qlora_cfg.get("lora_dropout", 0.05)),
        bias=qlora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    output_dir = train_cfg.get("output_dir", "outputs/code-review-adapter")
    sft_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=int(train_cfg.get("max_length", 1024)),
        packing=bool(train_cfg.get("packing", False)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 100)),
        eval_strategy="steps" if eval_rows else "no",
        eval_steps=int(train_cfg.get("eval_steps", 100)) if eval_rows else None,
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=compute_dtype is torch.bfloat16,
        fp16=compute_dtype is torch.float16,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        report_to=train_cfg.get("report_to", "none"),
        seed=int(train_cfg.get("seed", 42)),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def dry_run(config: dict[str, Any], train_path: str, eval_path: str | None = None) -> None:
    """Validate data/config and print a formatted SFT preview without importing ML stacks."""

    train_examples = load_examples(train_path)
    eval_examples = load_examples(eval_path) if eval_path else []
    rows = examples_to_sft_rows(train_examples)
    print(
        json.dumps(
            {
                "model": config.get("model", {}).get("name"),
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "first_train_id": rows[0]["id"] if rows else None,
                "first_text_preview": rows[0]["text"][:700] if rows else "",
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML or JSON config path.")
    parser.add_argument("--train", required=True, help="Training JSONL dataset.")
    parser.add_argument("--eval", help="Optional validation/golden JSONL dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.dry_run:
            dry_run(config, args.train, args.eval)
        else:
            train(config, args.train, args.eval)
        return 0
    except (DatasetError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
