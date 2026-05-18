"""
PyTorch datasets for the two personalization methods.

``PepperImageDataset`` serves the real photos of Pepper paired with either:
  - a fixed template prompt containing the placeholder token (textual inversion), or
  - the BLIP caption rewritten with the subject identifier (dreambooth).

``ClassPriorDataset`` serves the synthetic class-prior images used by
DreamBooth's prior-preservation loss (``a photo of a dog``, etc.).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Default prompt templates (adapted from the Textual Inversion paper)
# One is picked at random each step so the learned embedding generalizes
DEFAULT_TEXTUAL_INVERSION_PROMPT_TEMPLATES: Sequence[str] = (
    "a photo of a {token}",
    "a rendering of a {token}",
    "a cropped photo of the {token}",
    "the photo of a {token}",
    "a clean photo of a {token}",
    "a photo of my {token}",
    "a close-up photo of a {token}",
    "a bright photo of the {token}",
    "a cropped photo of a {token}",
    "a photo of the {token}",
    "a good photo of the {token}",
    "a photo of one {token}",
    "a rendition of the {token}",
    "a photo of the cool {token}",
    "a photo of a nice {token}",
    "a rendition of a {token}",
    "a photo of a cool {token}",
    "a photo of a small {token}",
)


def build_image_to_tensor_transform(
    target_resolution: int,
    apply_random_horizontal_flip: bool,
) -> transforms.Compose:
    """Image to tensor in [-1, 1] transformation, as expected by the Stable Diffusion 1.5 VAE encoder."""
    transform_steps: List[Callable] = [
        transforms.Resize(
            target_resolution,
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        transforms.CenterCrop(target_resolution),
    ]
    if apply_random_horizontal_flip:
        transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(transform_steps)


@dataclass
class DatasetSample:
    """One training sample: an image tensor plus its prompt string."""

    image_tensor: torch.Tensor
    prompt_text: str
    source_image_filename: str


# Dataset for subject (Pepper images)
class PepperImageDataset(Dataset):
    """
    A small on-disk dataset of preprocessed Pepper photos.

    The caller decides how to build the prompt for each sample. Two factory
    functions below cover the two methods we implement.
    """

    def __init__(
        self,
        processed_images_directory: Path,
        prompt_builder: Callable[[str, str], str],
        captions_file_path: Optional[Path] = None,
        target_resolution: int = 512,
        apply_random_horizontal_flip: bool = True,
        dataset_repeat_count: int = 1,
    ) -> None:
        self._processed_images_directory = Path(processed_images_directory)
        if not self._processed_images_directory.is_dir():
            raise FileNotFoundError(
                f"Processed images directory not found: {self._processed_images_directory}"
            )

        self._image_paths_in_order: List[Path] = sorted(
            file_path
            for file_path in self._processed_images_directory.iterdir()
            if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if len(self._image_paths_in_order) == 0:
            raise ValueError(
                f"No processed images found under {self._processed_images_directory}."
            )

        self._image_filename_to_caption: Dict[str, str] = {}
        if captions_file_path is not None and Path(captions_file_path).is_file():
            with Path(captions_file_path).open("r") as captions_file_handle:
                self._image_filename_to_caption = json.load(captions_file_handle)

        self._prompt_builder = prompt_builder
        self._dataset_repeat_count = dataset_repeat_count
        self._image_to_tensor_transform = build_image_to_tensor_transform(
            target_resolution=target_resolution,
            apply_random_horizontal_flip=apply_random_horizontal_flip,
        )

    def __len__(self) -> int:
        return len(self._image_paths_in_order) * self._dataset_repeat_count

    def __getitem__(self, linear_index: int) -> DatasetSample:
        underlying_image_index = linear_index % len(self._image_paths_in_order)
        image_path = self._image_paths_in_order[underlying_image_index]

        with Image.open(image_path) as opened_image:
            rgb_image = opened_image.convert("RGB")
            image_tensor_in_minus_one_to_one = self._image_to_tensor_transform(rgb_image)

        caption_for_this_image = self._image_filename_to_caption.get(image_path.name, "")
        prompt_text_for_this_sample = self._prompt_builder(
            image_path.name,
            caption_for_this_image,
        )

        return DatasetSample(
            image_tensor=image_tensor_in_minus_one_to_one,
            prompt_text=prompt_text_for_this_sample,
            source_image_filename=image_path.name,
        )


# Prompt builder factories to build prompts for the subject dataset
def make_textual_inversion_prompt_builder(
    placeholder_token_string: str,
    prompt_templates: Sequence[str] = DEFAULT_TEXTUAL_INVERSION_PROMPT_TEMPLATES,
) -> Callable[[str, str], str]:
    """Each call returns a random template filled with the placeholder token."""

    def build_prompt_for_sample(
        image_filename: str,
        blip_caption_for_image: str,
    ) -> str:
        _ = image_filename, blip_caption_for_image
        selected_template = random.choice(list(prompt_templates))
        return selected_template.format(token=placeholder_token_string)

    return build_prompt_for_sample


def make_dreambooth_prompt_builder(
    subject_identifier_phrase: str,
    class_noun: str,
    use_blip_caption_when_available: bool = True,
) -> Callable[[str, str], str]:
    """
    DreamBooth-style prompt:
        "a photo of sks dog" (minimal)
    or, when a BLIP caption is available and we opt into it:
        "a photo of sks dog, <blip caption with mentions of 'dog' replaced>"
   Keeps the prompt simple and predictable: default is "a photo of {subject_id} {class_noun}".
    """

    minimal_prompt = f"a photo of {subject_identifier_phrase} {class_noun}"

    def build_prompt_for_sample(
        image_filename: str,
        blip_caption_for_image: str,
    ) -> str:
        _ = image_filename
        if not use_blip_caption_when_available or not blip_caption_for_image:
            return minimal_prompt
        caption_without_leading_whitespace = blip_caption_for_image.strip()
        return f"{minimal_prompt}, {caption_without_leading_whitespace}"

    return build_prompt_for_sample


# Class-prior dataset (DreamBooth only)
class ClassPriorDataset(Dataset):
    """
    Images and prompts for DreamBooth's prior-preservation loss.

    Prior images are typically generated from the same base model with a
    generic class prompt (e.g. ``a photo of a dog``) so the model doesn't
    forget what a generic dog looks like while it fits on Pepper's unique characteristics.
    """

    def __init__(
        self,
        class_prior_images_directory: Path,
        class_prior_prompt: str,
        target_resolution: int = 512,
        apply_random_horizontal_flip: bool = True,
    ) -> None:
        self._class_prior_images_directory = Path(class_prior_images_directory)
        if not self._class_prior_images_directory.is_dir():
            raise FileNotFoundError(
                f"Class prior directory not found: {self._class_prior_images_directory}"
            )

        self._image_paths_in_order: List[Path] = sorted(
            file_path
            for file_path in self._class_prior_images_directory.iterdir()
            if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if len(self._image_paths_in_order) == 0:
            raise ValueError(
                f"No class prior images found under {self._class_prior_images_directory}. "
                f"Generate them first (see scripts/train.py)."
            )

        self._class_prior_prompt = class_prior_prompt
        self._image_to_tensor_transform = build_image_to_tensor_transform(
            target_resolution=target_resolution,
            apply_random_horizontal_flip=apply_random_horizontal_flip,
        )

    def __len__(self) -> int:
        return len(self._image_paths_in_order)

    def __getitem__(self, linear_index: int) -> DatasetSample:
        image_path = self._image_paths_in_order[linear_index]
        with Image.open(image_path) as opened_image:
            rgb_image = opened_image.convert("RGB")
            image_tensor_in_minus_one_to_one = self._image_to_tensor_transform(rgb_image)
        return DatasetSample(
            image_tensor=image_tensor_in_minus_one_to_one,
            prompt_text=self._class_prior_prompt,
            source_image_filename=image_path.name,
        )


# Collate function to stack image tensors and preserve prompts as a list
def collate_dataset_samples_into_batch(
    batch_of_samples: Sequence[DatasetSample],
) -> Dict[str, torch.Tensor | List[str]]:
    """Simple collate function that stacks image tensors and preserves prompts as a list for the batch."""
    stacked_image_tensor = torch.stack(
        [single_sample.image_tensor for single_sample in batch_of_samples],
        dim=0,
    )
    return {
        "pixel_value_tensors": stacked_image_tensor,
        "prompt_texts": [single_sample.prompt_text for single_sample in batch_of_samples],
        "source_image_filenames": [
            single_sample.source_image_filename for single_sample in batch_of_samples
        ],
    }
