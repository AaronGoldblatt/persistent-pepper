"""
Freeze /unfreeze helpers and token-injection logic.

These are the only places where ``requires_grad`` is touched on the Stable Diffusion 1.5 sub-modules. Centralizing it makes it easy to audit which method is training which parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from .loader import StableDiffusionComponents

# Generic freeze/unfreeze utilities
def freeze_module_parameters(module_to_freeze: nn.Module) -> None:
    for parameter in module_to_freeze.parameters():
        parameter.requires_grad = False
    module_to_freeze.eval()


def unfreeze_module_parameters(module_to_unfreeze: nn.Module) -> None:
    for parameter in module_to_unfreeze.parameters():
        parameter.requires_grad = True
    module_to_unfreeze.train()


# Textual Inversion
@dataclass
class PlaceholderTokenInjectionResult:
    """Describes a newly injected placeholder token and its initialization."""

    placeholder_token_string: str
    placeholder_token_id: int
    initializer_token_string: str
    initializer_token_id: int


def inject_new_placeholder_token(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    placeholder_token_string: str,
    initializer_token_string: str,
) -> PlaceholderTokenInjectionResult:
    """
    Adds a brand-new token (e.g. ``<pepper>``) to the tokenizer vocabulary and seeds its embedding with the embedding of an existing semantically similar token (e.g. ``"dog"``).
    """

    number_of_newly_added_tokens = tokenizer.add_tokens([placeholder_token_string])
    if number_of_newly_added_tokens != 1:
        raise ValueError(
            f"Placeholder token {placeholder_token_string!r} already exists in the "
            f"tokenizer vocabulary. Choose a unique string (e.g. '<pepper>')."
        )

    initializer_token_ids = tokenizer.encode(
        initializer_token_string,
        add_special_tokens=False,
    )
    if len(initializer_token_ids) != 1:
        raise ValueError(
            f"Initializer string {initializer_token_string!r} must tokenize to exactly "
            f"one token, but produced {len(initializer_token_ids)} tokens."
        )
    initializer_token_id = initializer_token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token_string)

    text_encoder.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        input_embedding_table = text_encoder.get_input_embeddings().weight
        input_embedding_table[placeholder_token_id] = (
            input_embedding_table[initializer_token_id].clone()
        )

    return PlaceholderTokenInjectionResult(
        placeholder_token_string=placeholder_token_string,
        placeholder_token_id=placeholder_token_id,
        initializer_token_string=initializer_token_string,
        initializer_token_id=initializer_token_id,
    )


def configure_textual_inversion_trainables(
    stable_diffusion_components: StableDiffusionComponents,
    placeholder_token_id: int,
) -> List[nn.Parameter]:
    """
    Freeze everything in Stable Diffusion 1.5 except the input embedding matrix of the text encoder.
    Gradients will still land on every row of that matrix, but a mask is applied to zero out all rows except the placeholder row inside the training step itself.

    Returns the list of parameters the optimizer should track for training.
    """

    freeze_module_parameters(stable_diffusion_components.variational_autoencoder)
    freeze_module_parameters(stable_diffusion_components.denoising_unet)
    freeze_module_parameters(stable_diffusion_components.text_encoder)

    input_embedding_table = stable_diffusion_components.text_encoder.get_input_embeddings()
    input_embedding_table.train()
    input_embedding_table.weight.requires_grad = True

    _ = placeholder_token_id  # only needed for the gradient mask

    return [input_embedding_table.weight]


# DreamBooth
def configure_dreambooth_trainables(
    stable_diffusion_components: StableDiffusionComponents,
    train_text_encoder: bool,
    enable_gradient_checkpointing: bool,
) -> List[nn.Parameter]:
    """
    Freeze the VAE (always), unfreeze the UNet, and optionally unfreeze the
    text encoder. Returns the concatenated list of trainable parameters.
    """

    freeze_module_parameters(stable_diffusion_components.variational_autoencoder)
    unfreeze_module_parameters(stable_diffusion_components.denoising_unet)

    if enable_gradient_checkpointing:
        stable_diffusion_components.denoising_unet.enable_gradient_checkpointing()

    trainable_parameters: List[nn.Parameter] = list(
        stable_diffusion_components.denoising_unet.parameters()
    )

    if train_text_encoder:
        unfreeze_module_parameters(stable_diffusion_components.text_encoder)
        if enable_gradient_checkpointing:
            stable_diffusion_components.text_encoder.gradient_checkpointing_enable()
        trainable_parameters.extend(stable_diffusion_components.text_encoder.parameters())
    else:
        freeze_module_parameters(stable_diffusion_components.text_encoder)

    return trainable_parameters


def iterate_trainable_parameters(
    parameter_iterable: Iterable[nn.Parameter],
) -> Iterable[nn.Parameter]:
    """Filters an iterable of parameters down to those with ``requires_grad``."""
    for parameter in parameter_iterable:
        if parameter.requires_grad:
            yield parameter
