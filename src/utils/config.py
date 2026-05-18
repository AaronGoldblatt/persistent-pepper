"""
Tiny YAML config loader with explicit validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_CONFIG_FIELDS = {
    "textual_inversion": [
        "pretrained_model_name_or_path",
        "placeholder_token_string",
        "initializer_token_string",
        "processed_images_directory",
        "captions_file_path",
        "output_directory",
        "train_batch_size",
        "total_training_steps",
        "learning_rate",
    ],
    "dreambooth": [
        "pretrained_model_name_or_path",
        "subject_identifier_phrase",
        "class_noun",
        "processed_images_directory",
        "captions_file_path",
        "class_prior_images_directory",
        "output_directory",
        "train_batch_size",
        "total_training_steps",
        "learning_rate",
        "prior_loss_weight",
    ],
}


def load_yaml_config(
    config_file_path: str | Path,
) -> Dict[str, Any]:
    config_file_path = Path(config_file_path)
    if not config_file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file_path}")
    with config_file_path.open("r") as config_file_handle:
        parsed_config = yaml.safe_load(config_file_handle)
    if not isinstance(parsed_config, dict):
        raise ValueError(f"Config at {config_file_path} must be a YAML mapping.")
    return parsed_config


def validate_config_for_method(
    parsed_config: Dict[str, Any],
    method_name: str,
) -> None:
    if method_name not in REQUIRED_CONFIG_FIELDS:
        raise ValueError(
            f"Unknown training method {method_name!r}. "
            f"Expected one of: {list(REQUIRED_CONFIG_FIELDS)}"
        )
    missing_fields = [
        required_field_name
        for required_field_name in REQUIRED_CONFIG_FIELDS[method_name]
        if required_field_name not in parsed_config
    ]
    if missing_fields:
        raise ValueError(
            f"Config is missing required fields for method {method_name!r}: "
            f"{missing_fields}"
        )
