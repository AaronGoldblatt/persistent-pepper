"""
CLI wrapper around ``src.utils.captioning.generate_captions_for_directory``

Produces a ``captions.json`` mapping of
``{processed_image_filename: blip_caption}``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.utils.captioning import (  # noqa: E402
    DEFAULT_BLIP_MODEL_NAME,
    generate_captions_for_directory,
)


def parse_command_line_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--processed_images_directory",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "processed",
    )
    argument_parser.add_argument(
        "--captions_output_path",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "captions.json",
    )
    argument_parser.add_argument(
        "--blip_model_name",
        type=str,
        default=DEFAULT_BLIP_MODEL_NAME,
    )
    argument_parser.add_argument(
        "--conditional_caption_prefix",
        type=str,
        default=None,
        help="Optional prefix passed to BLIP to bias captions (e.g. 'a photo of').",
    )
    return argument_parser.parse_args()


def main() -> None:
    parsed_arguments = parse_command_line_arguments()
    image_filename_to_caption = generate_captions_for_directory(
        processed_images_directory=parsed_arguments.processed_images_directory,
        captions_output_path=parsed_arguments.captions_output_path,
        blip_model_name=parsed_arguments.blip_model_name,
        conditional_caption_prefix=parsed_arguments.conditional_caption_prefix,
    )
    print(
        f"Wrote {len(image_filename_to_caption)} captions "
        f"to {parsed_arguments.captions_output_path}."
    )


if __name__ == "__main__":
    main()
