import json

import pytest

from review_tuner.config import AppConfig, ConfigError, load_config


def test_config_applies_canonical_defaults() -> None:
    config = AppConfig.from_mapping(
        {
            "model": {"name": "example/model"},
            "qlora": {},
            "training": {},
        }
    )

    assert config.model.name == "example/model"
    assert config.qlora.bnb_4bit_quant_type == "nf4"
    assert config.qlora.bnb_4bit_compute_dtype == "bfloat16"
    assert config.training.max_length == 1024


def test_config_rejects_unknown_and_mistyped_options() -> None:
    with pytest.raises(ConfigError, match="unknown qlora option"):
        AppConfig.from_mapping(
            {
                "model": {"name": "example/model"},
                "qlora": {"lora_rank": 8},
                "training": {},
            }
        )

    with pytest.raises(ConfigError, match="load_in_4bit must be a boolean"):
        AppConfig.from_mapping(
            {
                "model": {"name": "example/model"},
                "qlora": {"load_in_4bit": "yes"},
                "training": {},
            }
        )


def test_load_json_config_validates_before_use(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model": {"name": "example/model", "device_map": "auto"},
                "qlora": {"r": 8},
                "training": {"learning_rate": 0.0001},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.device_map == "auto"
    assert config.qlora.r == 8
    assert config.training.learning_rate == 0.0001


def test_config_rejects_invalid_ranges() -> None:
    with pytest.raises(ConfigError, match="warmup_ratio"):
        AppConfig.from_mapping(
            {
                "model": {"name": "example/model"},
                "qlora": {},
                "training": {"warmup_ratio": 1.5},
            }
        )
