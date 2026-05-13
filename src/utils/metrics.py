"""
CLIP-based evaluation metrics.

Two numbers we care about, following the DreamBooth / Textual Inversion papers:

- ``CLIP-I``: mean cosine similarity between CLIP image embeddings of the
  generated images and of the real reference images of Pepper. This proxies
  *identity preservation*.
- ``CLIP-T``: mean cosine similarity between CLIP image embeddings of the
  generated images and CLIP text embeddings of their prompts. This proxies
  *prompt alignment*.

We load raw `CLIPModel` / `CLIPProcessor` directly — no `pipeline(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


@dataclass
class ClipSimilarityReport:
    clip_image_to_image_similarity: float
    clip_image_to_text_similarity: float
    number_of_generated_images_considered: int
    number_of_reference_images_considered: int


def load_clip_evaluation_model(
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME,
    working_device: Optional[torch.device] = None,
) -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    resolved_working_device = (
        working_device
        if working_device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(resolved_working_device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    return clip_model, clip_processor, resolved_working_device


def extract_tensor_from_clip_output(
    clip_output,
    preferred_attribute_names: Sequence[str],
) -> torch.Tensor:
    """
    Transformers versions differ on whether CLIP helpers return a Tensor or a
    model-output object. Normalize those variants here.
    """
    if isinstance(clip_output, torch.Tensor):
        return clip_output[:, 0, :] if clip_output.ndim == 3 else clip_output
    for attribute_name in preferred_attribute_names:
        candidate_tensor = getattr(clip_output, attribute_name, None)
        if isinstance(candidate_tensor, torch.Tensor):
            return candidate_tensor[:, 0, :] if candidate_tensor.ndim == 3 else candidate_tensor
    if isinstance(clip_output, tuple):
        for candidate_tensor in clip_output:
            if isinstance(candidate_tensor, torch.Tensor):
                return candidate_tensor[:, 0, :] if candidate_tensor.ndim == 3 else candidate_tensor
    raise TypeError(
        "Could not extract a tensor from CLIP output of type "
        f"{type(clip_output)!r}."
    )


def project_clip_pooler_output_if_needed(
    clip_model: CLIPModel,
    clip_output,
    projection_attribute_name: str,
) -> torch.Tensor:
    """Fallback for versions where get_*_features returns a model output."""
    if isinstance(clip_output, torch.Tensor):
        return clip_output

    for projected_attribute_name in ["image_embeds", "text_embeds"]:
        projected_tensor = getattr(clip_output, projected_attribute_name, None)
        if isinstance(projected_tensor, torch.Tensor):
            return (
                projected_tensor[:, 0, :]
                if projected_tensor.ndim == 3
                else projected_tensor
            )

    pooled_hidden_state = extract_tensor_from_clip_output(
        clip_output=clip_output,
        preferred_attribute_names=["pooler_output"],
    )
    projection_layer = getattr(clip_model, projection_attribute_name, None)
    if projection_layer is None:
        return pooled_hidden_state
    return projection_layer(pooled_hidden_state)


@torch.no_grad()
def embed_pil_images_with_clip(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    working_device: torch.device,
    pil_images: Sequence[Image.Image],
) -> torch.Tensor:
    pixel_values = clip_processor(
        images=list(pil_images),
        return_tensors="pt",
    )["pixel_values"].to(working_device)
    batch_size = pixel_values.shape[0]
    # Use the full CLIPModel forward so we get the stable CLIPOutput.image_embeds
    # field regardless of how the sub-model APIs behave in this transformers version.
    dummy_input_ids = torch.zeros(
        (batch_size, 2), dtype=torch.long, device=working_device
    )
    clip_output = clip_model(pixel_values=pixel_values, input_ids=dummy_input_ids)
    return F.normalize(clip_output.image_embeds.float(), dim=-1)


@torch.no_grad()
def embed_text_prompts_with_clip(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    working_device: torch.device,
    prompt_texts: Sequence[str],
) -> torch.Tensor:
    text_inputs = clip_processor(
        text=list(prompt_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = text_inputs["input_ids"].to(working_device)
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(working_device)
    batch_size = input_ids.shape[0]
    image_size = clip_model.config.vision_config.image_size
    dummy_pixel_values = torch.zeros(
        (batch_size, 3, image_size, image_size), device=working_device
    )
    clip_output = clip_model(
        pixel_values=dummy_pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return F.normalize(clip_output.text_embeds.float(), dim=-1)


def load_pil_images_from_directory(
    images_directory: Path,
) -> List[Image.Image]:
    pil_images: List[Image.Image] = []
    for image_path in sorted(images_directory.iterdir()):
        if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            pil_images.append(Image.open(image_path).convert("RGB").copy())
    return pil_images


def compute_clip_similarity_report(
    generated_pil_images: Sequence[Image.Image],
    reference_pil_images: Sequence[Image.Image],
    prompts_used_for_generation: Sequence[str],
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME,
) -> ClipSimilarityReport:
    if len(generated_pil_images) != len(prompts_used_for_generation):
        raise ValueError(
            "Each generated image must be paired with the prompt that produced it."
        )

    clip_model, clip_processor, working_device = load_clip_evaluation_model(
        clip_model_name=clip_model_name,
    )

    generated_image_embeddings = embed_pil_images_with_clip(
        clip_model=clip_model,
        clip_processor=clip_processor,
        working_device=working_device,
        pil_images=generated_pil_images,
    )
    reference_image_embeddings = embed_pil_images_with_clip(
        clip_model=clip_model,
        clip_processor=clip_processor,
        working_device=working_device,
        pil_images=reference_pil_images,
    )
    prompt_text_embeddings = embed_text_prompts_with_clip(
        clip_model=clip_model,
        clip_processor=clip_processor,
        working_device=working_device,
        prompt_texts=prompts_used_for_generation,
    )

    image_to_image_similarity_matrix = generated_image_embeddings @ reference_image_embeddings.T
    mean_image_to_image_similarity = image_to_image_similarity_matrix.mean().item()

    mean_image_to_text_similarity = (
        (generated_image_embeddings * prompt_text_embeddings).sum(dim=-1).mean().item()
    )

    return ClipSimilarityReport(
        clip_image_to_image_similarity=mean_image_to_image_similarity,
        clip_image_to_text_similarity=mean_image_to_text_similarity,
        number_of_generated_images_considered=len(generated_pil_images),
        number_of_reference_images_considered=len(reference_pil_images),
    )


def format_report_as_human_readable_text(
    report: ClipSimilarityReport,
) -> str:
    return (
        f"CLIP-I (identity, gen vs refs):   {report.clip_image_to_image_similarity:.4f}\n"
        f"CLIP-T (prompt alignment):        {report.clip_image_to_text_similarity:.4f}\n"
        f"# generated images considered:    {report.number_of_generated_images_considered}\n"
        f"# reference images considered:    {report.number_of_reference_images_considered}"
    )
