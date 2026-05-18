# Pepper the Persistent Poodle

A comparative, written-from-scratch PyTorch implementation of two subject personalization/consistent character generation techniques for text-to-image diffusion, applied to a single subject - my toy poodle, Pepper.

1. **Textual Inversion** — learn a new token embedding in the model's CLIP space while keeping every model
   weight frozen
2. **DreamBooth** — fine-tune the UNet (and optionally the text encoder) with a
   prior-preservation loss so a rare identifier becomes bound to Pepper without
   destroying class-level knowledge

Both loops are written by hand in PyTorch on top of the raw Stable Diffusion 1.5
building blocks (VAE, UNet, CLIP tokenizer, CLIP text encoder, DDPM scheduler).
I did NOT use `StableDiffusionPipeline` or any other high-level
pipeline wrapper - every step of the noising, denoising, loss, sampling process, etc.
is implemented explicitly.

---

## 1. Repository layout

```
persistent-pepper/
├── data/
│   ├── raw/                # Original high-resolution smartphone photos of Pepper
│   ├── processed/          # Center-cropped, resized (512x512), normalized tensors-on-disk
│   ├── human_eval_ratings/ # Human subjective ratings per method (CSV exports of the Google Sheets)
│   └── captions.json       # BLIP-generated captions, keyed by processed filename
├── src/
│   ├── dataset.py          # PepperImageDataset and ClassPriorDataset
│   ├── models/
│   │   ├── loader.py       # Loads Stable Diffusion 1.5 sub-modules (VAE, UNet, CLIP, scheduler)
│   │   └── wrappers.py     # Freeze/unfreeze and token-injection helpers
│   ├── training/
│   │   ├── dreambooth.py   # Manual DreamBooth loop with prior-preservation loss
│   │   └── inversion.py    # Manual Textual Inversion loop (embedding-only optimization)
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
│   ├── evaluate.py         # CLIP-based identity/prompt alignment evaluation
│   └── run_train.sh        # Quick shell trigger
├── notebooks/
│   ├── pdfs/               # Exported notebook PDFs for demo appendix for paper
│   └── pepper_inference.ipynb   # Colab-friendly inference & qualitative comparison
├── checkpoints/            # Saved weights (.pt / .safetensors / learned_embeddings.bin)
├── requirements.txt
└── README.md
```

---

## 2. Where training actually runs

All GPU-bound work (training, sampling, CLIP evaluation) will be run on Google Colab with an NVIDIA A100 GPU. This will be done with Jupyter Notebooks designed in the codebase and connected to Colab via tokens.