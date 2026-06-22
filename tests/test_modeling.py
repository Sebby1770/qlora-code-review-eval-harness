from types import SimpleNamespace

from review_tuner.config import QLoRAConfig
from review_tuner.modeling import (
    ensure_padding_token,
    make_quantization_config,
    missing_ml_dependency_error,
    quantization_kwargs,
    torch_dtype,
)


def test_modeling_helpers_share_dtype_and_quantization_policy() -> None:
    torch = SimpleNamespace(
        bfloat16="bf16-value",
        float16="fp16-value",
        float32="fp32-value",
    )
    config = QLoRAConfig(
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=False,
    )

    assert torch_dtype(torch, "fp32") == "fp32-value"
    assert quantization_kwargs(config, torch) == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": False,
        "bnb_4bit_compute_dtype": "fp16-value",
    }


def test_padding_and_dependency_errors_are_consistent() -> None:
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
    dependency_error = ModuleNotFoundError(name="transformers")

    assert ensure_padding_token(tokenizer) is tokenizer
    assert tokenizer.pad_token == "<eos>"
    assert ".[train]" in str(missing_ml_dependency_error(dependency_error))


def test_quantization_factory_is_skipped_when_4bit_is_disabled() -> None:
    torch = SimpleNamespace(
        bfloat16="bf16-value",
        float16="fp16-value",
        float32="fp32-value",
    )
    calls: list[dict[str, object]] = []

    def factory(**kwargs):
        calls.append(kwargs)
        return "quantization-config"

    assert make_quantization_config(QLoRAConfig(load_in_4bit=False), torch, factory) is None
    assert calls == []
    assert make_quantization_config(QLoRAConfig(), torch, factory) == "quantization-config"
    assert calls[0]["load_in_4bit"] is True
