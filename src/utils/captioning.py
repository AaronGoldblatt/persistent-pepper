"""
BLIP-based caption generation for the Pepper dataset using the raw HF `BlipForConditionalGeneration` class directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
)


DEFAULT_BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"


def load_blip_caption_generator(
    blip_model_name: str = DEFAULT_BLIP_MODEL_NAME,
    working_device: Optional[torch.device] = None,
) -> tuple[BlipProcessor, BlipForConditionalGeneration, torch.device]:
    resolved_working_device = (
        working_device
        if working_device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    blip_processor = BlipProcessor.from_pretrained(blip_model_name)
    blip_caption_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
    blip_caption_model = blip_caption_model.to(resolved_working_device)
    blip_caption_model.eval()
    return blip_processor, blip_caption_model, resolved_working_device


@torch.no_grad()
def generate_caption_for_single_image(
    blip_processor: BlipProcessor,
    blip_caption_model: BlipForConditionalGeneration,
    working_device: torch.device,
    image_path: Path,
    conditional_caption_prefix: Optional[str] = None,
    max_new_tokens: int = 40,
) -> str:
    with Image.open(image_path) as opened_image:
        rgb_image = opened_image.convert("RGB")
        processor_inputs = blip_processor(
            images=rgb_image,
            text=conditional_caption_prefix,
            return_tensors="pt",
        ).to(working_device)
        generated_token_ids = blip_caption_model.generate(
            **processor_inputs,
            max_new_tokens=max_new_tokens,
        )
        decoded_caption = blip_processor.batch_decode(
            generated_token_ids,
            skip_special_tokens=True,
        )[0].strip()
    return decoded_caption


def generate_captions_for_directory(
    processed_images_directory: Path,
    captions_output_path: Path,
    blip_model_name: str = DEFAULT_BLIP_MODEL_NAME,
    conditional_caption_prefix: Optional[str] = None,
) -> Dict[str, str]:
    """
    Walks the processed-images directory, captions each file, and writes the
    ``{image_filename: caption}`` mapping as JSON.
    """

    blip_processor, blip_caption_model, working_device = load_blip_caption_generator(
        blip_model_name=blip_model_name,
    )

    image_filename_to_caption: Dict[str, str] = {}
    candidate_image_paths: List[Path] = sorted(
        file_path
        for file_path in processed_images_directory.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    for image_path in candidate_image_paths:
        generated_caption = generate_caption_for_single_image(
            blip_processor=blip_processor,
            blip_caption_model=blip_caption_model,
            working_device=working_device,
            image_path=image_path,
            conditional_caption_prefix=conditional_caption_prefix,
        )
        image_filename_to_caption[image_path.name] = generated_caption

    captions_output_path.parent.mkdir(parents=True, exist_ok=True)
    with captions_output_path.open("w") as captions_file_handle:
        json.dump(image_filename_to_caption, captions_file_handle, indent=2)

    return image_filename_to_caption
