"""
CLI wrapper around ``src.utils.preprocessing.preprocess_directory_of_images``.

Resizes every image in ``--raw_images_directory`` to a square of
``--target_resolution``, writing PNGs into ``--processed_images_directory``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.utils.preprocessing import preprocess_directory_of_images  # noqa: E402


def parse_command_line_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--raw_images_directory",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw",
    )
    argument_parser.add_argument(
        "--processed_images_directory",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "processed",
    )
    argument_parser.add_argument(
        "--target_resolution",
        type=int,
        default=512,
    )
    return argument_parser.parse_args()


def main() -> None:
    parsed_arguments = parse_command_line_arguments()
    written_paths = preprocess_directory_of_images(
        raw_images_directory=parsed_arguments.raw_images_directory,
        processed_images_directory=parsed_arguments.processed_images_directory,
        target_resolution=parsed_arguments.target_resolution,
    )
    print(
        f"Preprocessed {len(written_paths)} images "
        f"into {parsed_arguments.processed_images_directory} "
        f"at {parsed_arguments.target_resolution}x{parsed_arguments.target_resolution}."
    )


if __name__ == "__main__":
    main()
