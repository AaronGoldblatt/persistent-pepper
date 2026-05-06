"""
Evaluate a trained personalization (textual inversion or dreambooth) with the
manual DDIM sampler + CLIP metrics.

    python scripts/evaluate.py \\
        --method dreambooth \\
        --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \\
        --checkpoint_path checkpoints/dreambooth/latest \\
        --reference_images_directory data/processed \\
        --prompts_file configs/stress_test_prompts.txt \\
        --subject_identifier_phrase "sks dog" \\
        --output_directory outputs/dreambooth_eval

    python scripts/evaluate.py \\
        --method textual_inversion \\
        --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \\
        --checkpoint_path checkpoints/textual_inversion/learned_embedding_final.bin \\
        --prompt "a photo of {subject} wearing sunglasses" \\
        --prompt "an oil painting of {subject}" \\
        --subject_identifier_phrase "<pepper>" \\
        --output_directory outputs/textual_inversion_custom_prompts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

THIS_FILE = Path(__file__).resolve()
REPOSITORY_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from diffusers import UNet2DConditionModel  # noqa: E402
from transformers import CLIPTextModel  # noqa: E402

from src.models import load_stable_diffusion_components  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    compute_clip_similarity_report,
    format_report_as_human_readable_text,
    load_pil_images_from_directory,
)
from src.utils.sampling import (  # noqa: E402
    SamplingConfiguration,
    sample_images_from_prompts,
)


DEFAULT_PROMPTS_FILE = REPOSITORY_ROOT / "configs" / "stress_test_prompts.txt"


def parse_command_line_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--method",
        choices=("textual_inversion", "dreambooth"),
        required=True,
    )
    argument_parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
    )
    argument_parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help=(
            "For textual_inversion: path to learned_embedding_final.bin.\n"
            "For dreambooth: path to the 'latest' directory (containing 'unet/' and "
            "optionally 'text_encoder/')."
        ),
    )
    argument_parser.add_argument(
        "--reference_images_directory",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "processed",
    )
    argument_parser.add_argument(
        "--prompts_file",
        type=Path,
        default=DEFAULT_PROMPTS_FILE,
        help=(
            "File containing one prompt template per line. Used only when "
            "no --prompt arguments are provided."
        ),
    )
    argument_parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help=(
            "Single prompt template to render. Repeatable. "
            "Use {subject} where the learned identifier should be inserted."
        ),
    )
    argument_parser.add_argument(
        "--subject_identifier_phrase",
        type=str,
        required=True,
        help=(
            "Substitutes {subject} in each prompt. "
            "E.g. '<pepper>' for textual inversion or 'sks dog' for dreambooth."
        ),
    )
    argument_parser.add_argument(
        "--output_directory",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "eval",
    )
    argument_parser.add_argument(
        "--number_of_inference_steps",
        type=int,
        default=50,
    )
    argument_parser.add_argument(
        "--classifier_free_guidance_scale",
        type=float,
        default=7.5,
    )
    argument_parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
    )
    return argument_parser.parse_args()


def read_prompt_templates_from_file(
    prompts_file_path: Path,
) -> List[str]:
    raw_file_contents = prompts_file_path.read_text().splitlines()
    cleaned_prompt_templates: List[str] = []
    for raw_line in raw_file_contents:
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        cleaned_prompt_templates.append(stripped_line)
    return cleaned_prompt_templates


def resolve_prompt_templates_from_arguments(
    command_line_prompt_templates: List[str],
    prompts_file_path: Path,
) -> List[str]:
    """
    Prefer explicitly provided --prompt values. Fall back to the prompts file
    so the existing stress-test workflow keeps working unchanged.
    """
    if command_line_prompt_templates:
        return command_line_prompt_templates
    return read_prompt_templates_from_file(prompts_file_path)


def load_textual_inversion_embedding_into_components(
    stable_diffusion_components,
    learned_embedding_checkpoint_path: Path,
) -> str:
    saved_state_dictionary = torch.load(
        learned_embedding_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(saved_state_dictionary, dict) or len(saved_state_dictionary) != 1:
        raise ValueError(
            f"Expected a single-key dict in {learned_embedding_checkpoint_path}, "
            f"got: {type(saved_state_dictionary)}"
        )
    placeholder_token_string, learned_embedding_vector = next(
        iter(saved_state_dictionary.items())
    )

    stable_diffusion_components.tokenizer.add_tokens([placeholder_token_string])
    stable_diffusion_components.text_encoder.resize_token_embeddings(
        len(stable_diffusion_components.tokenizer)
    )
    placeholder_token_id = stable_diffusion_components.tokenizer.convert_tokens_to_ids(
        placeholder_token_string
    )
    with torch.no_grad():
        stable_diffusion_components.text_encoder.get_input_embeddings().weight[
            placeholder_token_id
        ] = learned_embedding_vector.to(
            stable_diffusion_components.text_encoder.get_input_embeddings().weight.device,
            dtype=stable_diffusion_components.text_encoder.get_input_embeddings().weight.dtype,
        )
    return placeholder_token_string


def load_dreambooth_weights_into_components(
    stable_diffusion_components,
    dreambooth_checkpoint_directory: Path,
) -> None:
    trained_unet_directory = dreambooth_checkpoint_directory / "unet"
    trained_text_encoder_directory = dreambooth_checkpoint_directory / "text_encoder"

    replacement_unet = UNet2DConditionModel.from_pretrained(trained_unet_directory).to(
        stable_diffusion_components.working_device,
        dtype=stable_diffusion_components.working_dtype,
    )
    stable_diffusion_components.denoising_unet = replacement_unet

    if trained_text_encoder_directory.is_dir():
        replacement_text_encoder = CLIPTextModel.from_pretrained(
            trained_text_encoder_directory
        ).to(
            stable_diffusion_components.working_device,
            dtype=stable_diffusion_components.working_dtype,
        )
        stable_diffusion_components.text_encoder = replacement_text_encoder


def main() -> None:
    parsed_arguments = parse_command_line_arguments()
    parsed_arguments.output_directory.mkdir(parents=True, exist_ok=True)

    stable_diffusion_components = load_stable_diffusion_components(
        pretrained_model_name_or_path=parsed_arguments.pretrained_model_name_or_path,
    )

    if parsed_arguments.method == "textual_inversion":
        load_textual_inversion_embedding_into_components(
            stable_diffusion_components=stable_diffusion_components,
            learned_embedding_checkpoint_path=parsed_arguments.checkpoint_path,
        )
    elif parsed_arguments.method == "dreambooth":
        load_dreambooth_weights_into_components(
            stable_diffusion_components=stable_diffusion_components,
            dreambooth_checkpoint_directory=parsed_arguments.checkpoint_path,
        )

    prompt_templates = resolve_prompt_templates_from_arguments(
        command_line_prompt_templates=parsed_arguments.prompt,
        prompts_file_path=parsed_arguments.prompts_file,
    )
    fully_substituted_prompts = [
        prompt_template.format(subject=parsed_arguments.subject_identifier_phrase)
        for prompt_template in prompt_templates
    ]

    sampling_configuration = SamplingConfiguration(
        number_of_inference_steps=parsed_arguments.number_of_inference_steps,
        classifier_free_guidance_scale=parsed_arguments.classifier_free_guidance_scale,
        random_seed=parsed_arguments.random_seed,
    )
    generated_pil_images = sample_images_from_prompts(
        stable_diffusion_components=stable_diffusion_components,
        prompt_texts=fully_substituted_prompts,
        sampling_configuration=sampling_configuration,
    )
    for prompt_index, generated_pil_image in enumerate(generated_pil_images):
        generated_pil_image.save(
            parsed_arguments.output_directory / f"gen_{prompt_index:02d}.png"
        )

    # Free SD model VRAM before loading CLIP to avoid OOM on constrained GPUs.
    del stable_diffusion_components
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        reference_pil_images = load_pil_images_from_directory(
            parsed_arguments.reference_images_directory,
        )
        clip_similarity_report = compute_clip_similarity_report(
            generated_pil_images=generated_pil_images,
            reference_pil_images=reference_pil_images,
            prompts_used_for_generation=fully_substituted_prompts,
        )
        human_readable_report_text = format_report_as_human_readable_text(
            clip_similarity_report,
        )
        print(human_readable_report_text)
        (parsed_arguments.output_directory / "clip_report.txt").write_text(
            human_readable_report_text
        )
    except Exception as clip_error:
        print(f"WARNING: CLIP evaluation failed and will be skipped: {clip_error}")
        print("Generated images were saved successfully.")


if __name__ == "__main__":
    main()
