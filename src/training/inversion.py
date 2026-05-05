"""
Manual Textual Inversion training loop.

Idea:
    - Freeze everything in SD 1.5 except the input embedding matrix of the
      text encoder.
    - Add a new token (e.g. ``<pepper>``), seed it from an existing token
      (e.g. ``dog``), and optimize *only its row* of the embedding matrix.
    - Everything else — VAE, UNet, scheduler — is identical to training and we
      don't touch its weights.

Only the placeholder row receives non-zero gradients; we enforce that
explicitly in ``_zero_out_non_placeholder_embedding_gradients``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..dataset import (
    PepperImageDataset,
    collate_dataset_samples_into_batch,
    make_textual_inversion_prompt_builder,
)
from ..models import (
    configure_textual_inversion_trainables,
    inject_new_placeholder_token,
    load_stable_diffusion_components,
)
from ..utils.loss import (
    compute_epsilon_prediction_mse_loss,
    prepare_noised_latents_and_target,
)
from ..utils.sampling import (
    SamplingConfiguration,
    VAE_LATENT_SCALING_FACTOR,
    sample_images_from_prompts,
)
from ..utils.training_logs import append_training_metrics_to_csv


@dataclass
class TextualInversionTrainingOutcome:
    placeholder_token_string: str
    placeholder_token_id: int
    final_learned_embedding_vector: torch.Tensor
    learned_embedding_output_path: Path


def _zero_out_non_placeholder_embedding_gradients(
    input_embedding_weight_tensor: torch.Tensor,
    placeholder_token_id: int,
) -> None:
    """
    Mask gradients so only the placeholder row actually updates. Called
    *after* backward() and *before* optimizer.step().
    """
    if input_embedding_weight_tensor.grad is None:
        return
    gradient_mask = torch.zeros_like(input_embedding_weight_tensor.grad)
    gradient_mask[placeholder_token_id] = 1.0
    input_embedding_weight_tensor.grad.mul_(gradient_mask)


def _tokenize_prompt_texts_for_training(
    tokenizer,
    prompt_texts,
    target_device: torch.device,
) -> torch.Tensor:
    tokenizer_output = tokenizer(
        prompt_texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return tokenizer_output.input_ids.to(target_device)


def _encode_pixel_values_to_latents(
    variational_autoencoder,
    pixel_value_tensors: torch.Tensor,
) -> torch.Tensor:
    vae_dtype = next(variational_autoencoder.parameters()).dtype
    with torch.no_grad():
        encoded_latent_distribution = variational_autoencoder.encode(
            pixel_value_tensors.to(dtype=vae_dtype)
        ).latent_dist
    return encoded_latent_distribution.sample() * VAE_LATENT_SCALING_FACTOR


def save_learned_embedding_to_disk(
    text_encoder,
    placeholder_token_string: str,
    placeholder_token_id: int,
    output_file_path: Path,
) -> None:
    """Saves just the single learned embedding vector (tiny file)."""
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    input_embedding_table = text_encoder.get_input_embeddings().weight.detach().cpu()
    learned_embedding_vector = input_embedding_table[placeholder_token_id].clone()
    torch.save(
        {placeholder_token_string: learned_embedding_vector},
        output_file_path,
    )


def save_textual_inversion_preview_image(
    stable_diffusion_components,
    preview_prompt_text: str,
    preview_output_path: Path,
    number_of_inference_steps: int,
    classifier_free_guidance_scale: float,
    random_seed: int,
) -> None:
    """Render one fixed prompt so identity progress is visible over time."""
    preview_output_path.parent.mkdir(parents=True, exist_ok=True)

    text_encoder_was_training = stable_diffusion_components.text_encoder.training
    denoising_unet_was_training = stable_diffusion_components.denoising_unet.training
    variational_autoencoder_was_training = (
        stable_diffusion_components.variational_autoencoder.training
    )

    stable_diffusion_components.text_encoder.eval()
    stable_diffusion_components.denoising_unet.eval()
    stable_diffusion_components.variational_autoencoder.eval()

    generated_preview_image = sample_images_from_prompts(
        stable_diffusion_components=stable_diffusion_components,
        prompt_texts=[preview_prompt_text],
        sampling_configuration=SamplingConfiguration(
            number_of_inference_steps=number_of_inference_steps,
            classifier_free_guidance_scale=classifier_free_guidance_scale,
            random_seed=random_seed,
        ),
    )[0]
    generated_preview_image.save(preview_output_path)

    stable_diffusion_components.text_encoder.train(text_encoder_was_training)
    stable_diffusion_components.denoising_unet.train(denoising_unet_was_training)
    stable_diffusion_components.variational_autoencoder.train(
        variational_autoencoder_was_training
    )


def run_textual_inversion_training(
    parsed_config: Dict[str, Any],
) -> TextualInversionTrainingOutcome:
    pretrained_model_name_or_path: str = parsed_config["pretrained_model_name_or_path"]
    placeholder_token_string: str = parsed_config["placeholder_token_string"]
    initializer_token_string: str = parsed_config["initializer_token_string"]
    processed_images_directory = Path(parsed_config["processed_images_directory"])
    captions_file_path = Path(parsed_config["captions_file_path"])
    output_directory = Path(parsed_config["output_directory"])

    train_batch_size: int = int(parsed_config["train_batch_size"])
    total_training_steps: int = int(parsed_config["total_training_steps"])
    learning_rate: float = float(parsed_config["learning_rate"])
    target_resolution: int = int(parsed_config.get("target_resolution", 512))
    dataset_repeat_count: int = int(parsed_config.get("dataset_repeat_count", 100))
    gradient_accumulation_steps: int = int(
        parsed_config.get("gradient_accumulation_steps", 1)
    )
    checkpoint_every_n_steps: int = int(parsed_config.get("checkpoint_every_n_steps", 250))
    log_every_n_steps: int = int(parsed_config.get("log_every_n_steps", 10))
    preview_every_n_steps: int = int(
        parsed_config.get("preview_every_n_steps", checkpoint_every_n_steps)
    )
    preview_prompt_template: str = str(
        parsed_config.get("preview_prompt_template", "a photo of {subject}")
    )
    preview_num_inference_steps: int = int(
        parsed_config.get("preview_num_inference_steps", 25)
    )
    preview_guidance_scale: float = float(
        parsed_config.get("preview_guidance_scale", 7.5)
    )
    preview_seed: int = int(parsed_config.get("preview_seed", 0))
    max_gradient_norm: float = float(parsed_config.get("max_gradient_norm", 1.0))
    use_mixed_precision_float16: bool = bool(
        parsed_config.get("use_mixed_precision_float16", True)
    )
    random_seed: int = int(parsed_config.get("random_seed", 42))

    torch.manual_seed(random_seed)

    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_csv_file_path = output_directory / "training_metrics.csv"
    preview_images_directory = output_directory / "previews"
    preview_prompt_text = preview_prompt_template.format(
        subject=placeholder_token_string
    )

    working_dtype = (
        torch.float16 if use_mixed_precision_float16 and torch.cuda.is_available()
        else torch.float32
    )

    stable_diffusion_components = load_stable_diffusion_components(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        working_dtype=working_dtype,
    )
    stable_diffusion_components.text_encoder.to(dtype=torch.float32)

    placeholder_token_info = inject_new_placeholder_token(
        tokenizer=stable_diffusion_components.tokenizer,
        text_encoder=stable_diffusion_components.text_encoder,
        placeholder_token_string=placeholder_token_string,
        initializer_token_string=initializer_token_string,
    )

    trainable_parameters = configure_textual_inversion_trainables(
        stable_diffusion_components=stable_diffusion_components,
        placeholder_token_id=placeholder_token_info.placeholder_token_id,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8,
    )

    prompt_builder_function = make_textual_inversion_prompt_builder(
        placeholder_token_string=placeholder_token_string,
    )
    pepper_training_dataset = PepperImageDataset(
        processed_images_directory=processed_images_directory,
        prompt_builder=prompt_builder_function,
        captions_file_path=captions_file_path,
        target_resolution=target_resolution,
        apply_random_horizontal_flip=True,
        dataset_repeat_count=dataset_repeat_count,
    )
    training_data_loader = DataLoader(
        pepper_training_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=int(parsed_config.get("num_workers", 0)),
        collate_fn=collate_dataset_samples_into_batch,
        drop_last=True,
    )

    working_device = stable_diffusion_components.working_device
    current_global_step = 0
    progress_bar = tqdm(total=total_training_steps, desc="textual_inversion")

    stable_diffusion_components.text_encoder.train()

    while current_global_step < total_training_steps:
        for training_batch in training_data_loader:
            pixel_value_tensors = training_batch["pixel_value_tensors"].to(
                working_device,
                dtype=working_dtype,
            )
            clean_latents = _encode_pixel_values_to_latents(
                variational_autoencoder=stable_diffusion_components.variational_autoencoder,
                pixel_value_tensors=pixel_value_tensors,
            ).to(dtype=working_dtype)

            (
                noised_latents,
                sampled_timesteps,
                prediction_target,
            ) = prepare_noised_latents_and_target(
                noise_scheduler=stable_diffusion_components.noise_scheduler,
                clean_latents=clean_latents,
            )

            prompt_token_id_tensor = _tokenize_prompt_texts_for_training(
                tokenizer=stable_diffusion_components.tokenizer,
                prompt_texts=training_batch["prompt_texts"],
                target_device=working_device,
            )
            text_encoder_hidden_states = stable_diffusion_components.text_encoder(
                prompt_token_id_tensor,
            ).last_hidden_state.to(dtype=working_dtype)

            predicted_noise = stable_diffusion_components.denoising_unet(
                noised_latents,
                sampled_timesteps,
                encoder_hidden_states=text_encoder_hidden_states,
            ).sample

            diffusion_loss_value = compute_epsilon_prediction_mse_loss(
                predicted_noise=predicted_noise,
                ground_truth_target=prediction_target,
            )
            if not torch.isfinite(diffusion_loss_value):
                progress_bar.write(
                    f"Skipping non-finite loss at step {current_global_step + 1}."
                )
                optimizer.zero_grad(set_to_none=True)
                current_global_step += 1
                progress_bar.update(1)
                continue

            (diffusion_loss_value / gradient_accumulation_steps).backward()

            if (current_global_step + 1) % gradient_accumulation_steps == 0:
                _zero_out_non_placeholder_embedding_gradients(
                    input_embedding_weight_tensor=(
                        stable_diffusion_components
                        .text_encoder
                        .get_input_embeddings()
                        .weight
                    ),
                    placeholder_token_id=placeholder_token_info.placeholder_token_id,
                )
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=max_gradient_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            current_global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{diffusion_loss_value.item():.4f}")

            should_log_this_step = (
                log_every_n_steps > 0
                and current_global_step % log_every_n_steps == 0
            )
            if should_log_this_step:
                append_training_metrics_to_csv(
                    csv_file_path=metrics_csv_file_path,
                    fieldnames=[
                        "step",
                        "loss",
                        "learning_rate",
                        "batch_size",
                        "preview_prompt",
                    ],
                    metric_row={
                        "step": current_global_step,
                        "loss": f"{diffusion_loss_value.item():.8f}",
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "batch_size": pixel_value_tensors.shape[0],
                        "preview_prompt": preview_prompt_text,
                    },
                )

            if current_global_step % checkpoint_every_n_steps == 0:
                intermediate_checkpoint_path = (
                    output_directory / f"learned_embedding_step_{current_global_step}.bin"
                )
                save_learned_embedding_to_disk(
                    text_encoder=stable_diffusion_components.text_encoder,
                    placeholder_token_string=placeholder_token_string,
                    placeholder_token_id=placeholder_token_info.placeholder_token_id,
                    output_file_path=intermediate_checkpoint_path,
                )

            should_save_preview_this_step = (
                preview_every_n_steps > 0
                and current_global_step % preview_every_n_steps == 0
            )
            if should_save_preview_this_step:
                save_textual_inversion_preview_image(
                    stable_diffusion_components=stable_diffusion_components,
                    preview_prompt_text=preview_prompt_text,
                    preview_output_path=(
                        preview_images_directory
                        / f"preview_step_{current_global_step:06d}.png"
                    ),
                    number_of_inference_steps=preview_num_inference_steps,
                    classifier_free_guidance_scale=preview_guidance_scale,
                    random_seed=preview_seed,
                )
                stable_diffusion_components.text_encoder.train()

            if current_global_step >= total_training_steps:
                break

    progress_bar.close()

    final_learned_embedding_path = output_directory / "learned_embedding_final.bin"
    save_learned_embedding_to_disk(
        text_encoder=stable_diffusion_components.text_encoder,
        placeholder_token_string=placeholder_token_string,
        placeholder_token_id=placeholder_token_info.placeholder_token_id,
        output_file_path=final_learned_embedding_path,
    )

    final_learned_embedding_vector = (
        stable_diffusion_components
        .text_encoder
        .get_input_embeddings()
        .weight[placeholder_token_info.placeholder_token_id]
        .detach()
        .cpu()
        .clone()
    )

    return TextualInversionTrainingOutcome(
        placeholder_token_string=placeholder_token_string,
        placeholder_token_id=placeholder_token_info.placeholder_token_id,
        final_learned_embedding_vector=final_learned_embedding_vector,
        learned_embedding_output_path=final_learned_embedding_path,
    )
