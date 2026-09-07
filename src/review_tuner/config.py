"""Typed configuration for training and model loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


_DTYPES = {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"config section {name!r} must be an object")
    return value


def _reject_unknown_keys(raw: Mapping[str, Any], model: Any, section: str) -> None:
    allowed = {field.name for field in fields(model)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown {section} option(s): {', '.join(unknown)}")


def _positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Base model loading configuration."""

    name: str
    device_map: str | dict[str, str | int] | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ModelConfig:
        _reject_unknown_keys(raw, cls, "model")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("model.name must be a non-empty string")
        device_map = raw.get("device_map")
        if device_map is not None and not isinstance(device_map, (str, dict)):
            raise ConfigError("model.device_map must be a string, object, or null")
        if isinstance(device_map, dict) and not all(
            isinstance(key, str)
            and isinstance(value, (str, int))
            and not isinstance(value, bool)
            for key, value in device_map.items()
        ):
            raise ConfigError(
                "model.device_map entries must map strings to strings or integers"
            )
        return cls(name=name.strip(), device_map=device_map)


@dataclass(frozen=True, slots=True)
class QLoRAConfig:
    """Quantization and adapter configuration shared by training and inference."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: str | tuple[str, ...] = "all-linear"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> QLoRAConfig:
        _reject_unknown_keys(raw, cls, "qlora")
        try:
            config = cls(
                load_in_4bit=_boolean(raw, "load_in_4bit", True),
                bnb_4bit_quant_type=_string(raw, "bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=_boolean(
                    raw, "bnb_4bit_use_double_quant", True
                ),
                bnb_4bit_compute_dtype=_string(
                    raw, "bnb_4bit_compute_dtype", "bfloat16"
                ).lower(),
                r=_integer(raw, "r", 16),
                lora_alpha=_integer(raw, "lora_alpha", 32),
                lora_dropout=_number(raw, "lora_dropout", 0.05),
                bias=_string(raw, "bias", "none"),
                target_modules=_target_modules(raw.get("target_modules", "all-linear")),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        _positive(config.r, "qlora.r")
        _positive(config.lora_alpha, "qlora.lora_alpha")
        if not 0.0 <= config.lora_dropout < 1.0:
            raise ConfigError("qlora.lora_dropout must be in [0, 1)")
        if config.bnb_4bit_compute_dtype not in _DTYPES:
            raise ConfigError(
                "qlora.bnb_4bit_compute_dtype must be one of "
                f"{sorted(_DTYPES)}, got {config.bnb_4bit_compute_dtype!r}"
            )
        return config


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """TRL training configuration used by the application."""

    output_dir: str = "outputs/code-review-adapter"
    max_length: int = 1024
    packing: bool = False
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    report_to: str | tuple[str, ...] = "none"
    seed: int = 42
    resume_from_checkpoint: str | bool | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> TrainingConfig:
        _reject_unknown_keys(raw, cls, "training")
        config = cls(
            output_dir=_string(raw, "output_dir", "outputs/code-review-adapter"),
            max_length=_integer(raw, "max_length", 1024),
            packing=_boolean(raw, "packing", False),
            num_train_epochs=_number(raw, "num_train_epochs", 1.0),
            per_device_train_batch_size=_integer(raw, "per_device_train_batch_size", 1),
            per_device_eval_batch_size=_integer(raw, "per_device_eval_batch_size", 1),
            gradient_accumulation_steps=_integer(raw, "gradient_accumulation_steps", 8),
            learning_rate=_number(raw, "learning_rate", 2e-4),
            lr_scheduler_type=_string(raw, "lr_scheduler_type", "cosine"),
            warmup_ratio=_number(raw, "warmup_ratio", 0.03),
            logging_steps=_integer(raw, "logging_steps", 10),
            save_steps=_integer(raw, "save_steps", 100),
            eval_steps=_integer(raw, "eval_steps", 100),
            save_total_limit=_integer(raw, "save_total_limit", 2),
            gradient_checkpointing=_boolean(raw, "gradient_checkpointing", True),
            report_to=_report_targets(raw.get("report_to", "none")),
            seed=_integer(raw, "seed", 42),
            resume_from_checkpoint=_checkpoint(raw.get("resume_from_checkpoint")),
        )
        for name in (
            "max_length",
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "logging_steps",
            "save_steps",
            "eval_steps",
            "save_total_limit",
        ):
            _positive(getattr(config, name), f"training.{name}")
        if not 0.0 <= config.warmup_ratio <= 1.0:
            raise ConfigError("training.warmup_ratio must be in [0, 1]")
        return config


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Complete application configuration."""

    model: ModelConfig
    qlora: QLoRAConfig
    training: TrainingConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AppConfig:
        unknown = sorted(set(raw) - {"model", "qlora", "training"})
        if unknown:
            raise ConfigError(f"unknown top-level section(s): {', '.join(unknown)}")
        return cls(
            model=ModelConfig.from_mapping(_section(raw, "model")),
            qlora=QLoRAConfig.from_mapping(_section(raw, "qlora")),
            training=TrainingConfig.from_mapping(_section(raw, "training")),
        )


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a JSON or safe YAML configuration file."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            try:
                import yaml  # type: ignore[import-untyped]
            except ModuleNotFoundError as exc:
                raise ConfigError(
                    "YAML configuration requires PyYAML; install with: "
                    "python -m pip install -e '.[train]'"
                ) from exc
            raw = yaml.safe_load(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not load config {config_path}: {exc}") from exc
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"could not parse config {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError("config root must be an object")
    return AppConfig.from_mapping(raw)


def _boolean(raw: Mapping[str, Any], name: str, default: bool) -> bool:
    value = raw.get(name, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _string(raw: Mapping[str, Any], name: str, default: str) -> str:
    value = raw.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(raw: Mapping[str, Any], name: str, default: int) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _number(raw: Mapping[str, Any], name: str, default: float) -> float:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    return float(value)


def _target_modules(value: Any) -> str | tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value and all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return tuple(item.strip() for item in value)
    raise ConfigError("target_modules must be a non-empty string or list of strings")


def _report_targets(value: Any) -> str | tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value and all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return tuple(item.strip() for item in value)
    raise ConfigError("report_to must be a non-empty string or list of non-empty strings")


def _checkpoint(value: Any) -> str | bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ConfigError("resume_from_checkpoint must be a string, boolean, or null")
