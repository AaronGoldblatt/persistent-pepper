"""
Loads the individual Stable Diffusion 1.5 sub-modules.

We deliberately avoid any high-level pipeline constructor (e.g. `StableDiffusionPipeline`)
so that every step of the forward/reverse diffusion process is implemented by us.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
)


@dataclass
class StableDiffusionComponents:
    """Bundle of the frozen / trainable modules we need from SD 1.5."""

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    variational_autoencoder: AutoencoderKL
    denoising_unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler
    working_device: torch.device
    working_dtype: torch.dtype

    def move_to_device(self) -> "StableDiffusionComponents":
        # Everything in fp32: avoids dtype mismatch bugs across training, sampling,
        # and the GradScaler, at the cost of ~25% slower training on a T4.
        self.text_encoder = self.text_encoder.to(self.working_device, dtype=torch.float32)
        self.variational_autoencoder = self.variational_autoencoder.to(self.working_device, dtype=torch.float32)
        self.denoising_unet = self.denoising_unet.to(self.working_device, dtype=torch.float32)
        return self


def auto_detect_best_available_device() -> torch.device:
    """
    Pick the best available accelerator:
        CUDA (Colab / local NVIDIA) > MPS (Apple Silicon) > CPU.

    On Apple Silicon we use MPS only in fp32; fp16 is unstable for SD 1.5 there,
    so fine-tuning should still happen on CUDA/Colab. MPS is mainly useful for
    quick local debugging of the dataset, sampler, or caption pipeline.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_stable_diffusion_components(
    pretrained_model_name_or_path: str,
    working_device: Optional[torch.device] = None,
    working_dtype: torch.dtype = torch.float32,
    revision: Optional[str] = None,
) -> StableDiffusionComponents:
    """
    Load each sub-module of Stable Diffusion 1.5 independently.

    `pretrained_model_name_or_path` can be either a HF hub id
    (e.g. ``stable-diffusion-v1-5/stable-diffusion-v1-5``) or a local directory
    with the canonical layout (``tokenizer/``, ``text_encoder/``, ``vae/``,
    ``unet/``, ``scheduler/``).
    """

    resolved_working_device = (
        working_device
        if working_device is not None
        else auto_detect_best_available_device()
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    variational_autoencoder = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=revision,
    )
    denoising_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=revision,
    )

    components = StableDiffusionComponents(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        variational_autoencoder=variational_autoencoder,
        denoising_unet=denoising_unet,
        noise_scheduler=noise_scheduler,
        working_device=resolved_working_device,
        working_dtype=working_dtype,
    )
    return components.move_to_device()
