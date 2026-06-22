"""Shared optional-dependency and model-loading helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from review_tuner.config import QLoRAConfig


def missing_ml_dependency_error(exc: ModuleNotFoundError) -> RuntimeError:
    """Build the consistent installation error used by ML entry points."""

    return RuntimeError(
        f"Missing optional ML dependency {exc.name!r}. Install with: "
        "python -m pip install -e '.[train]'"
    )


def torch_dtype(torch_module: Any, name: str) -> Any:
    """Resolve a validated configuration dtype to its torch value."""

    mapping = {
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    return mapping[name]


def quantization_kwargs(config: QLoRAConfig, torch_module: Any) -> dict[str, Any]:
    """Build BitsAndBytesConfig keyword arguments from the canonical config."""

    return {
        "load_in_4bit": config.load_in_4bit,
        "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": config.bnb_4bit_use_double_quant,
        "bnb_4bit_compute_dtype": torch_dtype(
            torch_module, config.bnb_4bit_compute_dtype
        ),
    }


def make_quantization_config(
    config: QLoRAConfig,
    torch_module: Any,
    factory: Callable[..., Any],
) -> Any | None:
    """Create quantization configuration only when 4-bit loading is enabled."""

    if not config.load_in_4bit:
        return None
    return factory(**quantization_kwargs(config, torch_module))


def ensure_padding_token(tokenizer: Any) -> Any:
    """Use the EOS token for padding when the model has no explicit pad token."""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
