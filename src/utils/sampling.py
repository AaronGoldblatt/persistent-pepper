"""
Manual DDIM sampler — no `StableDiffusionPipeline` involvement.

The core loop is:

    1. tokenize prompt (+ empty negative prompt) and encode with CLIPTextModel
    2. sample x_T ~ N(0, I) in latent space
    3. for t in reversed schedule:
           eps = UNet(concat(x_t, x_t), t, concat(uncond, cond))
           eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
           x_{t-1} = DDIM update(x_t, eps, t)
    4. decode latents via the VAE

We drive the DDIM update with a `DDIMScheduler` from `diffusers` because it
just holds the alpha schedule; all prompt handling, CFG, and loop control
lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from diffusers import DDIMScheduler
from PIL import Image

from ..models.loader import StableDiffusionComponents


VAE_LATENT_SCALING_FACTOR = 0.18215  # SD 1.5's pre-baked constant


@dataclass
class SamplingConfiguration:
    number_of_inference_steps: int = 50
    classifier_free_guidance_scale: float = 7.5
    output_resolution: int = 512
    random_seed: Optional[int] = None
    negative_prompt_text: str = ""


def encode_text_prompts_with_clip(
    stable_diffusion_components: StableDiffusionComponents,
    prompt_texts: Sequence[str],
) -> torch.Tensor:
    tokenizer_output = stable_diffusion_components.tokenizer(
        list(prompt_texts),
        padding="max_length",
        max_length=stable_diffusion_components.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    token_ids_on_device = tokenizer_output.input_ids.to(
        stable_diffusion_components.working_device,
    )
    with torch.no_grad():
        text_encoder_output = stable_diffusion_components.text_encoder(
            token_ids_on_device,
        )
    return text_encoder_output.last_hidden_state.to(
        dtype=stable_diffusion_components.working_dtype,
    )


def build_ddim_sampler(
    stable_diffusion_components: StableDiffusionComponents,
    number_of_inference_steps: int,
) -> DDIMScheduler:
    ddim_sampler = DDIMScheduler.from_config(
        stable_diffusion_components.noise_scheduler.config,
    )
    ddim_sampler.set_timesteps(number_of_inference_steps)
    return ddim_sampler


@torch.no_grad()
def decode_latents_to_pil_images(
    stable_diffusion_components: StableDiffusionComponents,
    final_latents: torch.Tensor,
) -> List[Image.Image]:
    unscaled_latents = final_latents / VAE_LATENT_SCALING_FACTOR
    decoded_pixel_tensor = stable_diffusion_components.variational_autoencoder.decode(
        unscaled_latents,
    ).sample
    decoded_pixel_tensor = (decoded_pixel_tensor / 2 + 0.5).clamp(0, 1)
    decoded_pixel_tensor = decoded_pixel_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    rendered_pil_images: List[Image.Image] = []
    for single_image_array in decoded_pixel_tensor:
        rendered_pil_images.append(
            Image.fromarray((single_image_array * 255).round().astype("uint8"))
        )
    return rendered_pil_images


@torch.no_grad()
def sample_images_from_prompts(
    stable_diffusion_components: StableDiffusionComponents,
    prompt_texts: Sequence[str],
    sampling_configuration: SamplingConfiguration = SamplingConfiguration(),
) -> List[Image.Image]:
    """
    Run the manual DDIM reverse-diffusion loop for a batch of prompts.
    Returns one PIL image per input prompt.
    """

    number_of_prompts = len(prompt_texts)
    ddim_sampler = build_ddim_sampler(
        stable_diffusion_components=stable_diffusion_components,
        number_of_inference_steps=sampling_configuration.number_of_inference_steps,
    )

    conditional_text_embeddings = encode_text_prompts_with_clip(
        stable_diffusion_components=stable_diffusion_components,
        prompt_texts=prompt_texts,
    )
    unconditional_text_embeddings = encode_text_prompts_with_clip(
        stable_diffusion_components=stable_diffusion_components,
        prompt_texts=[sampling_configuration.negative_prompt_text] * number_of_prompts,
    )
    concatenated_text_embeddings = torch.cat(
        [unconditional_text_embeddings, conditional_text_embeddings],
        dim=0,
    )

    latent_channel_count = stable_diffusion_components.denoising_unet.config.in_channels
    latent_spatial_size = sampling_configuration.output_resolution // 8

    random_generator = (
        torch.Generator(device=stable_diffusion_components.working_device)
        .manual_seed(sampling_configuration.random_seed)
        if sampling_configuration.random_seed is not None
        else None
    )

    current_latents = torch.randn(
        (
            number_of_prompts,
            latent_channel_count,
            latent_spatial_size,
            latent_spatial_size,
        ),
        generator=random_generator,
        device=stable_diffusion_components.working_device,
        dtype=stable_diffusion_components.working_dtype,
    )
    current_latents = current_latents * ddim_sampler.init_noise_sigma

    for current_timestep in ddim_sampler.timesteps:
        duplicated_latents_for_cfg = torch.cat([current_latents, current_latents], dim=0)
        duplicated_latents_for_cfg = ddim_sampler.scale_model_input(
            duplicated_latents_for_cfg,
            current_timestep,
        )

        predicted_noise_both_branches = stable_diffusion_components.denoising_unet(
            duplicated_latents_for_cfg,
            current_timestep,
            encoder_hidden_states=concatenated_text_embeddings,
        ).sample

        predicted_noise_unconditional, predicted_noise_conditional = predicted_noise_both_branches.chunk(2)
        guided_predicted_noise = (
            predicted_noise_unconditional
            + sampling_configuration.classifier_free_guidance_scale
            * (predicted_noise_conditional - predicted_noise_unconditional)
        )

        current_latents = ddim_sampler.step(
            guided_predicted_noise,
            current_timestep,
            current_latents,
        ).prev_sample

    return decode_latents_to_pil_images(
        stable_diffusion_components=stable_diffusion_components,
        final_latents=current_latents,
    )
