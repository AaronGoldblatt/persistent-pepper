"""
Manual DreamBooth training loop with prior-preservation.

Per step we compute two losses and combine them:

    subject_loss  = MSE( UNet(x_subject_t, t, text_enc("a photo of sks dog")),
                         noise_subject )
    prior_loss    = MSE( UNet(x_class_t,   t, text_enc("a photo of a dog")),
                         noise_class   )
    total_loss    = subject_loss + prior_loss_weight * prior_loss

Subject and class samples share the *same* timestep-sampling / noising
machinery and are concatenated into a single UNet call, following the
original DreamBooth paper's efficient implementation.

If ``class_prior_images_directory`` is empty, we first pre-generate class
prior images with the un-personalized base model via our manual DDIM sampler
(``src/utils/sampling.py``) — still no pipeline wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..dataset import (
    ClassPriorDataset,
    PepperImageDataset,
    collate_dataset_samples_into_batch,
    make_dreambooth_prompt_builder,
)
from ..models import (
    configure_dreambooth_trainables,
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
class DreamBoothTrainingOutcome:
    saved_unet_directory: Path
    saved_text_encoder_directory: Path | None
    number_of_training_steps_executed: int


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _infinite_iterator_over_data_loader(
    data_loader_to_wrap: DataLoader,
) -> Iterator[Dict[str, Any]]:
    while True:
        for yielded_batch in data_loader_to_wrap:
            yield yielded_batch


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


def _populate_class_prior_directory_if_empty(
    stable_diffusion_components,
    class_prior_images_directory: Path,
    class_prior_prompt: str,
    number_of_class_prior_images_to_generate: int,
    target_resolution: int,
) -> None:
    class_prior_images_directory.mkdir(parents=True, exist_ok=True)
    already_existing_class_prior_images = [
        file_path
        for file_path in class_prior_images_directory.iterdir()
        if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if len(already_existing_class_prior_images) >= number_of_class_prior_images_to_generate:
        return

    number_of_images_to_generate_now = (
        number_of_class_prior_images_to_generate - len(already_existing_class_prior_images)
    )

    sampling_configuration = SamplingConfiguration(
        number_of_inference_steps=50,
        classifier_free_guidance_scale=7.5,
        output_resolution=target_resolution,
    )

    batched_prompts = [class_prior_prompt] * number_of_images_to_generate_now
    max_prompts_per_batch = 4
    for batch_start_index in range(0, len(batched_prompts), max_prompts_per_batch):
        prompts_in_current_batch = batched_prompts[
            batch_start_index : batch_start_index + max_prompts_per_batch
        ]
        generated_pil_images = sample_images_from_prompts(
            stable_diffusion_components=stable_diffusion_components,
            prompt_texts=prompts_in_current_batch,
            sampling_configuration=sampling_configuration,
        )
        for offset_within_batch, generated_pil_image in enumerate(generated_pil_images):
            output_path = (
                class_prior_images_directory
                / f"class_prior_{batch_start_index + offset_within_batch:04d}.png"
            )
            generated_pil_image.save(output_path)


def save_dreambooth_preview_image(
    stable_diffusion_components,
    preview_prompt_text: str,
    preview_output_path: Path,
    number_of_inference_steps: int,
    classifier_free_guidance_scale: float,
    random_seed: int,
) -> None:
    """Render one fixed prompt so DreamBooth progress is visible over time."""
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


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def run_dreambooth_training(
    parsed_config: Dict[str, Any],
) -> DreamBoothTrainingOutcome:
    pretrained_model_name_or_path: str = parsed_config["pretrained_model_name_or_path"]
    subject_identifier_phrase: str = parsed_config["subject_identifier_phrase"]
    class_noun: str = parsed_config["class_noun"]
    processed_images_directory = Path(parsed_config["processed_images_directory"])
    captions_file_path = Path(parsed_config["captions_file_path"])
    class_prior_images_directory = Path(parsed_config["class_prior_images_directory"])
    output_directory = Path(parsed_config["output_directory"])

    train_batch_size: int = int(parsed_config["train_batch_size"])
    total_training_steps: int = int(parsed_config["total_training_steps"])
    learning_rate: float = float(parsed_config["learning_rate"])
    prior_loss_weight: float = float(parsed_config["prior_loss_weight"])

    target_resolution: int = int(parsed_config.get("target_resolution", 512))
    gradient_accumulation_steps: int = int(
        parsed_config.get("gradient_accumulation_steps", 1)
    )
    number_of_class_prior_images_to_generate: int = int(
        parsed_config.get("number_of_class_prior_images", 100)
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
    train_text_encoder: bool = bool(parsed_config.get("train_text_encoder", False))
    enable_gradient_checkpointing: bool = bool(
        parsed_config.get("enable_gradient_checkpointing", True)
    )
    use_mixed_precision_float16: bool = bool(
        parsed_config.get("use_mixed_precision_float16", True)
    )
    random_seed: int = int(parsed_config.get("random_seed", 42))

    torch.manual_seed(random_seed)
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_csv_file_path = output_directory / "training_metrics.csv"
    preview_images_directory = output_directory / "previews"
    preview_prompt_text = preview_prompt_template.format(
        subject=f"{subject_identifier_phrase} {class_noun}"
    )

    working_dtype = (
        torch.float16 if use_mixed_precision_float16 and torch.cuda.is_available()
        else torch.float32
    )

    stable_diffusion_components = load_stable_diffusion_components(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        working_dtype=working_dtype,
    )

    class_prior_prompt = f"a photo of a {class_noun}"
    _populate_class_prior_directory_if_empty(
        stable_diffusion_components=stable_diffusion_components,
        class_prior_images_directory=class_prior_images_directory,
        class_prior_prompt=class_prior_prompt,
        number_of_class_prior_images_to_generate=number_of_class_prior_images_to_generate,
        target_resolution=target_resolution,
    )

    trainable_parameters = configure_dreambooth_trainables(
        stable_diffusion_components=stable_diffusion_components,
        train_text_encoder=train_text_encoder,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    prompt_builder_function = make_dreambooth_prompt_builder(
        subject_identifier_phrase=subject_identifier_phrase,
        class_noun=class_noun,
        use_blip_caption_when_available=bool(
            parsed_config.get("use_blip_captions_in_prompts", False)
        ),
    )
    pepper_training_dataset = PepperImageDataset(
        processed_images_directory=processed_images_directory,
        prompt_builder=prompt_builder_function,
        captions_file_path=captions_file_path,
        target_resolution=target_resolution,
        apply_random_horizontal_flip=True,
        dataset_repeat_count=int(parsed_config.get("dataset_repeat_count", 1)),
    )
    class_prior_dataset = ClassPriorDataset(
        class_prior_images_directory=class_prior_images_directory,
        class_prior_prompt=class_prior_prompt,
        target_resolution=target_resolution,
        apply_random_horizontal_flip=True,
    )

    subject_training_data_loader = DataLoader(
        pepper_training_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=int(parsed_config.get("num_workers", 0)),
        collate_fn=collate_dataset_samples_into_batch,
        drop_last=True,
    )
    class_prior_data_loader = DataLoader(
        class_prior_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=int(parsed_config.get("num_workers", 0)),
        collate_fn=collate_dataset_samples_into_batch,
        drop_last=True,
    )
    class_prior_batch_iterator = _infinite_iterator_over_data_loader(class_prior_data_loader)

    working_device = stable_diffusion_components.working_device
    current_global_step = 0
    progress_bar = tqdm(total=total_training_steps, desc="dreambooth")

    stable_diffusion_components.denoising_unet.train()
    if train_text_encoder:
        stable_diffusion_components.text_encoder.train()

    while current_global_step < total_training_steps:
        for subject_training_batch in subject_training_data_loader:
            class_prior_batch = next(class_prior_batch_iterator)

            subject_pixel_values = subject_training_batch["pixel_value_tensors"].to(
                working_device,
                dtype=working_dtype,
            )
            class_prior_pixel_values = class_prior_batch["pixel_value_tensors"].to(
                working_device,
                dtype=working_dtype,
            )

            concatenated_pixel_values = torch.cat(
                [subject_pixel_values, class_prior_pixel_values],
                dim=0,
            )
            concatenated_clean_latents = _encode_pixel_values_to_latents(
                variational_autoencoder=stable_diffusion_components.variational_autoencoder,
                pixel_value_tensors=concatenated_pixel_values,
            )

            (
                concatenated_noised_latents,
                concatenated_sampled_timesteps,
                concatenated_prediction_target,
            ) = prepare_noised_latents_and_target(
                noise_scheduler=stable_diffusion_components.noise_scheduler,
                clean_latents=concatenated_clean_latents,
            )

            concatenated_prompt_texts = (
                subject_training_batch["prompt_texts"]
                + class_prior_batch["prompt_texts"]
            )
            prompt_token_id_tensor = _tokenize_prompt_texts_for_training(
                tokenizer=stable_diffusion_components.tokenizer,
                prompt_texts=concatenated_prompt_texts,
                target_device=working_device,
            )
            text_encoder_hidden_states = stable_diffusion_components.text_encoder(
                prompt_token_id_tensor,
            ).last_hidden_state.to(dtype=working_dtype)

            predicted_noise_concatenated = stable_diffusion_components.denoising_unet(
                concatenated_noised_latents,
                concatenated_sampled_timesteps,
                encoder_hidden_states=text_encoder_hidden_states,
            ).sample

            (
                predicted_noise_subject_half,
                predicted_noise_class_half,
            ) = predicted_noise_concatenated.chunk(2, dim=0)
            (
                prediction_target_subject_half,
                prediction_target_class_half,
            ) = concatenated_prediction_target.chunk(2, dim=0)

            subject_diffusion_loss = compute_epsilon_prediction_mse_loss(
                predicted_noise=predicted_noise_subject_half,
                ground_truth_target=prediction_target_subject_half,
            )
            class_prior_diffusion_loss = compute_epsilon_prediction_mse_loss(
                predicted_noise=predicted_noise_class_half,
                ground_truth_target=prediction_target_class_half,
            )
            total_diffusion_loss = (
                subject_diffusion_loss + prior_loss_weight * class_prior_diffusion_loss
            )
            if not torch.isfinite(total_diffusion_loss):
                progress_bar.write(
                    f"Skipping non-finite loss at step {current_global_step + 1}."
                )
                optimizer.zero_grad(set_to_none=True)
                current_global_step += 1
                progress_bar.update(1)
                continue

            (total_diffusion_loss / gradient_accumulation_steps).backward()

            if (current_global_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=max_gradient_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            current_global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                subject=f"{subject_diffusion_loss.item():.4f}",
                prior=f"{class_prior_diffusion_loss.item():.4f}",
            )

            should_log_this_step = (
                log_every_n_steps > 0
                and current_global_step % log_every_n_steps == 0
            )
            if should_log_this_step:
                append_training_metrics_to_csv(
                    csv_file_path=metrics_csv_file_path,
                    fieldnames=[
                        "step",
                        "total_loss",
                        "subject_loss",
                        "prior_loss",
                        "prior_loss_weight",
                        "learning_rate",
                        "batch_size",
                        "preview_prompt",
                    ],
                    metric_row={
                        "step": current_global_step,
                        "total_loss": f"{total_diffusion_loss.item():.8f}",
                        "subject_loss": f"{subject_diffusion_loss.item():.8f}",
                        "prior_loss": f"{class_prior_diffusion_loss.item():.8f}",
                        "prior_loss_weight": prior_loss_weight,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "batch_size": subject_pixel_values.shape[0],
                        "preview_prompt": preview_prompt_text,
                    },
                )

            if current_global_step % checkpoint_every_n_steps == 0:
                intermediate_checkpoint_directory = (
                    output_directory / f"step_{current_global_step}"
                )
                intermediate_checkpoint_directory.mkdir(parents=True, exist_ok=True)
                stable_diffusion_components.denoising_unet.save_pretrained(
                    intermediate_checkpoint_directory / "unet",
                )
                if train_text_encoder:
                    stable_diffusion_components.text_encoder.save_pretrained(
                        intermediate_checkpoint_directory / "text_encoder",
                    )

            should_save_preview_this_step = (
                preview_every_n_steps > 0
                and current_global_step % preview_every_n_steps == 0
            )
            if should_save_preview_this_step:
                save_dreambooth_preview_image(
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
                stable_diffusion_components.denoising_unet.train()
                if train_text_encoder:
                    stable_diffusion_components.text_encoder.train()

            if current_global_step >= total_training_steps:
                break

    progress_bar.close()

    final_checkpoint_directory = output_directory / "latest"
    final_checkpoint_directory.mkdir(parents=True, exist_ok=True)

    final_unet_directory = final_checkpoint_directory / "unet"
    stable_diffusion_components.denoising_unet.save_pretrained(final_unet_directory)

    final_text_encoder_directory: Path | None = None
    if train_text_encoder:
        final_text_encoder_directory = final_checkpoint_directory / "text_encoder"
        stable_diffusion_components.text_encoder.save_pretrained(final_text_encoder_directory)

    return DreamBoothTrainingOutcome(
        saved_unet_directory=final_unet_directory,
        saved_text_encoder_directory=final_text_encoder_directory,
        number_of_training_steps_executed=current_global_step,
    )
