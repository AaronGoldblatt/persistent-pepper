# Persistent Pepper

A comparative, from-scratch PyTorch implementation of two subject-personalization
techniques for text-to-image diffusion, applied to a single subject (my toy poodle,
Pepper):

1. **Textual Inversion** — learn a *new token embedding* while keeping every model
   weight frozen.
2. **DreamBooth** — fine-tune the *UNet* (and optionally the text encoder) with a
   prior-preservation loss so a rare identifier becomes bound to Pepper without
   destroying class-level knowledge.

Both loops are written by hand in PyTorch on top of the raw Stable Diffusion 1.5
building blocks (VAE, UNet, CLIP tokenizer, CLIP text encoder, DDPM scheduler).
We deliberately do **not** use `StableDiffusionPipeline` or any other high-level
pipeline wrapper — every step of the noising / denoising / loss / sampling process
is implemented explicitly.

---

## 1. Repository layout

```
persistent-pepper/
├── data/
│   ├── raw/                # Original high-res photos of Pepper
│   ├── processed/          # Center-cropped, resized (512x512), normalized tensors-on-disk
│   └── captions.json       # BLIP-generated captions, keyed by processed filename
├── src/
│   ├── dataset.py          # PepperImageDataset and ClassPriorDataset
│   ├── models/
│   │   ├── loader.py       # Loads SD 1.5 sub-modules (VAE, UNet, CLIP, scheduler)
│   │   └── wrappers.py     # Freeze/unfreeze + token-injection helpers
│   ├── training/
│   │   ├── dreambooth.py   # Manual DreamBooth loop with prior-preservation loss
│   │   └── inversion.py    # Manual Textual Inversion loop (embedding-only optim)
│   └── utils/
│       ├── loss.py         # Manual epsilon-prediction diffusion loss
│       ├── metrics.py      # CLIP image/text similarity for evaluation
│       ├── sampling.py     # Manual DDIM sampler (no pipeline wrapper)
│       ├── preprocessing.py# Image resizing/normalization utilities
│       ├── captioning.py   # BLIP caption generation for the raw dataset
│       └── config.py       # YAML config loader with light validation
├── configs/
│   ├── dreambooth_cfg.yaml
│   └── textual_inv_cfg.yaml
├── scripts/
│   ├── preprocess_images.py
│   ├── generate_captions.py
│   ├── train.py            # Dispatches to either training loop based on --method
│   ├── evaluate.py         # CLIP-based identity / prompt alignment evaluation
│   └── run_train.sh        # Quick shell trigger
├── notebooks/
│   └── pepper_inference.ipynb   # Colab-friendly inference & qualitative comparison
├── checkpoints/            # Saved weights (.pt / .safetensors / learned_embeddings.bin)
├── requirements.txt
└── README.md
```

---

## 2. Where training actually runs

All GPU-bound work (training, sampling, CLIP evaluation) is expected to run on
**Google Colab**. The local machine (Mac / Linux / etc.) is only used for code
editing, raw photo curation, and final report assembly. Nothing in the codebase
assumes CUDA is available locally — the loader auto-detects CUDA > MPS > CPU,
and training gates fp16 on CUDA specifically so it transparently falls back to
fp32 on anything else.

| Task                                                     | Where it runs     |
|----------------------------------------------------------|-------------------|
| Editing `src/`, debugging dataset + prompt builders      | Mac (Cursor)      |
| Curating raw Pepper photos into a Drive folder           | Mac               |
| Preprocessing, captioning, training, sampling, CLIP eval | **Colab**         |
| Inspecting saved checkpoints (`.bin` files are ~3 KB)    | Mac               |
| Final writeup / figure assembly                          | Mac               |

The canonical driver for the Colab side is
[`notebooks/pepper_colab_train.ipynb`](notebooks/pepper_colab_train.ipynb),
which owns the full flow: mount Drive → cache SD 1.5 → clone the repo →
preprocess → caption → train → stream checkpoints to Drive → evaluate.

---

## 3. Obtaining Stable Diffusion 1.5

The original `runwayml/stable-diffusion-v1-5` repository was taken down by the
uploader. The community-maintained mirror at
[`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
is the recommended source for a byte-identical copy.

All scripts accept a Hugging Face repo id or a local directory via
`--pretrained_model_name_or_path`, the `pretrained_model_name_or_path` YAML
field, or the `PEPPER_SD_MODEL` environment variable (env var wins over YAML,
CLI flag wins over env var).

### Colab (recommended): cache the weights in Drive once

`notebooks/pepper_colab_train.ipynb` does this for you on first run. Manually,
it's:

```python
from google.colab import drive; drive.mount('/content/drive')
import os
os.environ['PEPPER_SD_MODEL'] = '/content/drive/MyDrive/persistent-pepper-cache/sd-v1-5'
!huggingface-cli download \
    stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir $PEPPER_SD_MODEL --local-dir-use-symlinks False
```

Subsequent Colab sessions reuse the cached copy instead of re-downloading 4 GB.

### Local (optional, only needed for Mac-side debugging)

```bash
huggingface-cli download \
    stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir ./checkpoints/sd-v1-5 \
    --local-dir-use-symlinks False
export PEPPER_SD_MODEL="./checkpoints/sd-v1-5"
```

We load each SD 1.5 sub-module separately (`AutoencoderKL`,
`UNet2DConditionModel`, `CLIPTextModel`, `CLIPTokenizer`, `DDPMScheduler`)
rather than calling a pipeline constructor — see `src/models/loader.py`.

---

## 4. End-to-end workflow (Colab)

Open `notebooks/pepper_colab_train.ipynb` in Colab after pushing this repo to
GitHub, set the GPU runtime (Runtime → Change runtime type → GPU), and run
cells top-to-bottom. The notebook:

1. Sanity-checks the allocated GPU.
2. Mounts Drive and sets up a `persistent-pepper-cache/` folder for weights,
   processed images, captions, and checkpoints.
3. Reads a GitHub token from `MyDrive/cs5788/token.txt`, reads your GitHub
   username from `MyDrive/cs5788/git_username.txt`, clones the private repo
   into `/content/persistent-pepper`, and `pip install`s requirements.
4. Downloads SD 1.5 into the Drive cache (one-time).
5. Rsyncs raw photos from Drive to local SSD for fast reads.
6. Preprocesses + captions the dataset, writing outputs to the Drive cache.
7. Runs Textual Inversion training, streaming checkpoints to Drive via
   `--override output_directory=...`.
8. Runs DreamBooth training the same way.
9. Runs evaluation (manual DDIM sampler + CLIP-I / CLIP-T report).
10. Previews generated images inline.

During training, both methods also write lightweight progress artifacts into
their `output_directory`:

- `training_metrics.csv` records scalar losses every `log_every_n_steps`.
  Textual Inversion logs the embedding-training diffusion loss; DreamBooth logs
  total loss, subject loss, and class-prior loss separately.
- `previews/preview_step_*.png` stores one fixed-seed sample every
  `preview_every_n_steps`, using `preview_prompt_template`. Set
  `preview_every_n_steps: 0` to disable previews if Colab time is tight.

### CLI equivalents (also work locally if you do have a CUDA GPU)

```bash
python -m pip install -r requirements.txt
python scripts/preprocess_images.py
python scripts/generate_captions.py
python scripts/train.py --method textual_inversion --config configs/textual_inv_cfg.yaml
python scripts/train.py --method dreambooth       --config configs/dreambooth_cfg.yaml
python scripts/evaluate.py \
    --method dreambooth \
    --pretrained_model_name_or_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --checkpoint_path checkpoints/dreambooth/latest \
    --subject_identifier_phrase "sks dog"
```

If you want to render a few custom prompts without creating a file first, pass
repeatable `--prompt` args directly:

```bash
python scripts/evaluate.py \
    --method textual_inversion \
    --pretrained_model_name_or_path "$PEPPER_SD_MODEL" \
    --checkpoint_path checkpoints/textual_inversion/learned_embedding_final.bin \
    --prompt "a photo of {subject} wearing sunglasses" \
    --prompt "an oil painting of {subject} in the style of Van Gogh" \
    --subject_identifier_phrase "<pepper>" \
    --output_directory outputs/textual_inversion_custom_prompts
```

The `--override KEY=VALUE` flag on `scripts/train.py` (repeatable) lets you
redirect any YAML field at launch time — which is how the Colab notebook
pipes every output into Drive without editing the YAML files.

The `scripts/run_train.sh` helper is a one-line trigger that wraps the two
training commands.

`notebooks/pepper_inference.ipynb` is the sibling notebook for loading a
finished checkpoint from Drive and sampling new images (qualitative
comparison, stress-test prompts, etc.).

---

## 5. Hardware notes

- **Colab T4 (free tier, 16 GB)** is sufficient for both methods with the
  provided configs. DreamBooth takes ~25–40 min for 800 steps; Textual
  Inversion takes ~45–60 min for 3000 steps.
- **Colab L4 / A100 (Pro)** roughly 3× faster for DreamBooth; you can bump
  `train_batch_size` from 1 to 2 and drop `gradient_accumulation_steps`
  correspondingly.
- **Local Apple Silicon (MPS)** works for small-scale *inference* only. fp16
  is unstable on MPS for SD 1.5, so the loader forces fp32 there; training
  throughput is roughly an order of magnitude slower than a T4. Not
  recommended for real runs.
- **Local NVIDIA GPU with ≥12 GB** runs everything identically to Colab.

---

## 6. What is hand-implemented (open-source policy compliance)

The project relies on open-source building blocks but the *core process* is
authored here:

| Component                              | Source             | Hand-written here |
|----------------------------------------|--------------------|-------------------|
| VAE / UNet / CLIP text encoder weights | `diffusers`/`transformers` | loaded only      |
| Tokenizer vocabulary                   | `transformers`     | loaded only      |
| Forward noising schedule               | `diffusers` (coeffs) | **we index it manually** in `src/utils/loss.py` |
| Training loop (opt, clip, AMP, ckpt)  | —                  | **authored**      |
| Prior-preservation batching            | —                  | **authored** in `src/training/dreambooth.py` |
| Textual Inversion embedding masking    | —                  | **authored** in `src/training/inversion.py` |
| DDIM sampling loop                     | —                  | **authored** in `src/utils/sampling.py` |
| CLIP identity / prompt metrics         | —                  | **authored** in `src/utils/metrics.py` |

No `StableDiffusionPipeline`, `DiffusionPipeline`, `AutoPipelineForText2Image`, or
any equivalent high-level wrapper is imported anywhere in the project.
