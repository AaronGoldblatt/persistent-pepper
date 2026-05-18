"""
Manual epsilon-prediction diffusion loss implementation.

Relies on the scheduler to store the schedule, but does the forward-noising calculations explicitly.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler


def sample_random_timesteps(
    number_of_samples_in_batch: int,
    total_number_of_training_timesteps: int,
    target_device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=total_number_of_training_timesteps,
        size=(number_of_samples_in_batch,),
        device=target_device,
        dtype=torch.long,
    )


def add_forward_noise_to_latents(
    noise_scheduler: DDPMScheduler,
    clean_latents: torch.Tensor,
    gaussian_noise: torch.Tensor,
    sampled_timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

    Pull the cumulative-alpha values off the scheduler but apply them by hand so the forward process is fully explicit.
    """

    cumulative_alpha_values = noise_scheduler.alphas_cumprod.to(
        clean_latents.device,
        dtype=clean_latents.dtype,
    )
    alpha_bar_at_sampled_timesteps = cumulative_alpha_values[sampled_timesteps]
    sqrt_alpha_bar = alpha_bar_at_sampled_timesteps.sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_at_sampled_timesteps).sqrt().view(-1, 1, 1, 1)
    return sqrt_alpha_bar * clean_latents + sqrt_one_minus_alpha_bar * gaussian_noise


def select_prediction_target(
    noise_scheduler: DDPMScheduler,
    clean_latents: torch.Tensor,
    gaussian_noise: torch.Tensor,
    sampled_timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Stable Diffusion 1.5 uses epsilon-prediction. I still support v-prediction here in case of a need to swap in a v-prediction checkpoint.
    """

    scheduler_prediction_type = getattr(
        noise_scheduler.config,
        "prediction_type",
        "epsilon",
    )
    if scheduler_prediction_type == "epsilon":
        return gaussian_noise
    if scheduler_prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(
            clean_latents,
            gaussian_noise,
            sampled_timesteps,
        )
    raise ValueError(
        f"Unsupported scheduler prediction_type: {scheduler_prediction_type!r}"
    )


def compute_epsilon_prediction_mse_loss(
    predicted_noise: torch.Tensor,
    ground_truth_target: torch.Tensor,
    per_sample_loss_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Per-sample MSE, optionally reweighted (useful for prior-preservation where the aim is to blend subject loss and class prior loss).
    """

    if per_sample_loss_weights is None:
        return F.mse_loss(predicted_noise.float(), ground_truth_target.float(), reduction="mean")

    per_sample_mse = F.mse_loss(
        predicted_noise.float(),
        ground_truth_target.float(),
        reduction="none",
    ).mean(dim=(1, 2, 3))
    weighted_mean_loss = (per_sample_mse * per_sample_loss_weights.to(per_sample_mse.device)).mean()
    return weighted_mean_loss


def prepare_noised_latents_and_target(
    noise_scheduler: DDPMScheduler,
    clean_latents: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience helper: sample noise + timesteps, add noise, and return the target."""

    gaussian_noise = torch.randn_like(clean_latents)
    sampled_timesteps = sample_random_timesteps(
        number_of_samples_in_batch=clean_latents.shape[0],
        total_number_of_training_timesteps=noise_scheduler.config.num_train_timesteps,
        target_device=clean_latents.device,
    )
    noised_latents = add_forward_noise_to_latents(
        noise_scheduler=noise_scheduler,
        clean_latents=clean_latents,
        gaussian_noise=gaussian_noise,
        sampled_timesteps=sampled_timesteps,
    )
    prediction_target = select_prediction_target(
        noise_scheduler=noise_scheduler,
        clean_latents=clean_latents,
        gaussian_noise=gaussian_noise,
        sampled_timesteps=sampled_timesteps,
    )
    return noised_latents, sampled_timesteps, prediction_target
