"""
Image preprocessing: center-crop to a square and then resize to the target resolution.

Outputs are written as PNGs so the training-time transform only has to do a
``ToTensor`` and normalize transformation; no further resizing is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageOps


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def list_image_files_in_directory(
    source_directory: Path,
) -> List[Path]:
    return sorted(
        file_path
        for file_path in source_directory.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def center_crop_to_square(
    input_image: Image.Image,
) -> Image.Image:
    image_width, image_height = input_image.size
    shorter_side_length = min(image_width, image_height)
    left_pixel_offset = (
        (image_width - shorter_side_length)
        // 2
    )
    top_pixel_offset = (
        (image_height - shorter_side_length)
        // 2
    )
    return input_image.crop(
        (
            left_pixel_offset,
            top_pixel_offset,
            left_pixel_offset + shorter_side_length,
            top_pixel_offset + shorter_side_length,
        )
    )


def preprocess_single_image(
    source_image_path: Path,
    destination_image_path: Path,
    target_resolution: int,
) -> None:
    with Image.open(source_image_path) as opened_image:
        oriented_image = (
            ImageOps.exif_transpose(opened_image)
            .convert("RGB")
        )
        cropped_square_image = center_crop_to_square(oriented_image)
        resized_image = cropped_square_image.resize(
            (target_resolution, target_resolution),
            resample=Image.LANCZOS,
        )
        resized_image.save(destination_image_path, format="PNG")


def preprocess_directory_of_images(
    raw_images_directory: Path,
    processed_images_directory: Path,
    target_resolution: int,
) -> List[Path]:
    processed_images_directory.mkdir(parents=True, exist_ok=True)
    source_image_paths = list_image_files_in_directory(raw_images_directory)
    written_destination_paths: List[Path] = []
    for source_image_path in source_image_paths:
        destination_image_path = (
            processed_images_directory
            / f"{source_image_path.stem}.png"
        )
        preprocess_single_image(
            source_image_path=source_image_path,
            destination_image_path=destination_image_path,
            target_resolution=target_resolution,
        )
        written_destination_paths.append(destination_image_path)
    return written_destination_paths
