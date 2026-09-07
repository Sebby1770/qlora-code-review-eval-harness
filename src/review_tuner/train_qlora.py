"""QLoRA supervised fine-tuning entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator

from review_tuner.config import AppConfig, ConfigError, load_config
from review_tuner.data import iter_examples
from review_tuner.modeling import (
    ensure_padding_token,
    make_quantization_config,
    missing_ml_dependency_error,
    torch_dtype,
)
from review_tuner.prompts import build_sft_text, iter_sft_rows
from review_tuner.schema import DatasetError


def _dataset_rows(path: str) -> Iterator[dict[str, str]]:
    yield from iter_sft_rows(iter_examples(path))


def train(config: AppConfig, train_path: str, eval_path: str | None = None) -> None:
    """Run QLoRA fine-tuning using Hugging Face Transformers, PEFT, and TRL."""

    try:
        import torch  # type: ignore[import-not-found]
        from datasets import Dataset  # type: ignore[import-not-found]
        from peft import (  # type: ignore[import-not-found]
            LoraConfig,
            prepare_model_for_kbit_training,
        )
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise missing_ml_dependency_error(exc) from exc

    model_cfg = config.model
    qlora_cfg = config.qlora
    train_cfg = config.training

    train_dataset = Dataset.from_generator(_dataset_rows, gen_kwargs={"path": train_path})
    eval_dataset = (
        Dataset.from_generator(_dataset_rows, gen_kwargs={"path": eval_path})
        if eval_path
        else None
    )

    compute_dtype = torch_dtype(torch, qlora_cfg.bnb_4bit_compute_dtype)
    quantization_config = make_quantization_config(
        qlora_cfg,
        torch,
        BitsAndBytesConfig,
    )

    tokenizer = ensure_padding_token(
        AutoTokenizer.from_pretrained(model_cfg.name, use_fast=True)
    )

    model_kwargs = {"torch_dtype": compute_dtype}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if model_cfg.device_map is not None:
        model_kwargs["device_map"] = model_cfg.device_map

    model = AutoModelForCausalLM.from_pretrained(model_cfg.name, **model_kwargs)
    model.config.use_cache = False
    if qlora_cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_cfg.gradient_checkpointing,
        )

    target_modules: str | list[str]
    if isinstance(qlora_cfg.target_modules, tuple):
        target_modules = list(qlora_cfg.target_modules)
    else:
        target_modules = qlora_cfg.target_modules
    lora_config = LoraConfig(
        r=qlora_cfg.r,
        lora_alpha=qlora_cfg.lora_alpha,
        lora_dropout=qlora_cfg.lora_dropout,
        bias=qlora_cfg.bias,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    output_dir = train_cfg.output_dir
    report_to = (
        list(train_cfg.report_to)
        if isinstance(train_cfg.report_to, tuple)
        else train_cfg.report_to
    )
    sft_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=train_cfg.max_length,
        packing=train_cfg.packing,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=train_cfg.eval_steps if eval_dataset is not None else None,
        save_total_limit=train_cfg.save_total_limit,
        bf16=compute_dtype is torch.bfloat16,
        fp16=compute_dtype is torch.float16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        report_to=report_to,
        seed=train_cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=train_cfg.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def dry_run(config: AppConfig, train_path: str, eval_path: str | None = None) -> None:
    """Validate data/config and print a formatted SFT preview without importing ML stacks."""

    train_count = 0
    first_train_id: str | None = None
    first_text_preview = ""
    for example in iter_examples(train_path):
        train_count += 1
        if first_train_id is None:
            first_train_id = example.id
            first_text_preview = build_sft_text(example)[:700]
    eval_count = sum(1 for _ in iter_examples(eval_path)) if eval_path else 0
    print(
        json.dumps(
            {
                "model": config.model.name,
                "train_examples": train_count,
                "eval_examples": eval_count,
                "first_train_id": first_train_id,
                "first_text_preview": first_text_preview,
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
    except (ConfigError, DatasetError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
