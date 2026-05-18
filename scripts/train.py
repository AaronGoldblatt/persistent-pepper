"""
Entry point for training for either method, routes to the appropriate training loop based on
``--method``

    python scripts/train.py --method textual_inversion --config configs/textual_inv_cfg.yaml
    python scripts/train.py --method dreambooth        --config configs/dreambooth_cfg.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.training import (  # noqa: E402
    run_dreambooth_training,
    run_textual_inversion_training,
)
from src.utils.config import (  # noqa: E402
    load_yaml_config,
    validate_config_for_method,
)


SUPPORTED_METHODS = (
    "textual_inversion",
    "dreambooth",
)


def parse_command_line_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=SUPPORTED_METHODS,
    )
    argument_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config for the chosen method.",
    )
    argument_parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Overrides the config's value (also respected via $PEPPER_SD_MODEL).",
    )
    argument_parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a single YAML config field. Repeatable (can be used multiple times in the command line to override multiple fields). "
            "Example: --override output_directory=/content/drive/MyDrive/pp/ckpts"
        ),
    )
    return argument_parser.parse_args()


def _resolve_pretrained_model_source(
    parsed_config: Dict[str, Any],
    command_line_override: str | None,
) -> str:
    if command_line_override is not None:
        return command_line_override
    environment_override = os.environ.get("PEPPER_SD_MODEL")
    if environment_override is not None:
        return environment_override
    return parsed_config["pretrained_model_name_or_path"]


def _coerce_override_string_to_native_type(
    raw_string_value: str,
) -> Any:
    """Tries to coerce the string value to a native type so YAML numeric / bool types survive overrides."""
    lowered_value = raw_string_value.strip().lower()
    if lowered_value in {"true", "false"}:
        return lowered_value == "true"
    if lowered_value in {"null", "none", "~"}:
        return None
    try:
        return int(raw_string_value)
    except ValueError:
        pass
    try:
        return float(raw_string_value)
    except ValueError:
        pass
    return raw_string_value


def _apply_command_line_overrides_to_config(
    parsed_config: Dict[str, Any],
    raw_override_strings: List[str],
) -> None:
    for single_override_string in raw_override_strings:
        if "=" not in single_override_string:
            raise ValueError(
                f"--override expects KEY=VALUE, got {single_override_string!r}"
            )
        override_key, override_raw_value = single_override_string.split("=", 1)
        parsed_config[override_key] = _coerce_override_string_to_native_type(
            override_raw_value
        )


def main() -> None:
    parsed_arguments = parse_command_line_arguments()
    parsed_config = load_yaml_config(parsed_arguments.config)
    _apply_command_line_overrides_to_config(
        parsed_config=parsed_config,
        raw_override_strings=parsed_arguments.override,
    )
    validate_config_for_method(parsed_config, parsed_arguments.method)

    parsed_config["pretrained_model_name_or_path"] = _resolve_pretrained_model_source(
        parsed_config=parsed_config,
        command_line_override=parsed_arguments.pretrained_model_name_or_path,
    )

    if parsed_arguments.method == "textual_inversion":
        outcome = run_textual_inversion_training(parsed_config)
        print(
            f"Textual Inversion done. Learned embedding for "
            f"{outcome.placeholder_token_string!r} saved to "
            f"{outcome.learned_embedding_output_path}."
        )
    elif parsed_arguments.method == "dreambooth":
        outcome = run_dreambooth_training(parsed_config)
        print(
            f"DreamBooth done after {outcome.number_of_training_steps_executed} steps. "
            f"UNet saved to {outcome.saved_unet_directory}. "
            f"Text encoder saved to {outcome.saved_text_encoder_directory}."
        )


if __name__ == "__main__":
    main()
