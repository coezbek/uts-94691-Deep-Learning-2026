# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 2 — E5: Swap encoder to CLIP ViT-B/16
#
# CLIP was pretrained on 400M image-text pairs and produces image features
# that are aligned to natural-language descriptions — strictly better
# caption-relevant features than ImageNet-classification EfficientNet.
#
# Step 1 extracts the 196 patch-level features (14×14 grid, 768-dim) once
# over every unique image and caches them. Step 2 trains the standard
# Transformer decoder on those cached features, using the same code path as
# E0/E4/E7 — only `encoder_feat_dim` and `num_spatial_tokens` change.

# %%
import sys
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path("output").resolve()))
import importlib
import exp_runner
importlib.reload(exp_runner)
from exp_runner import ExperimentConfig, _get_image_reader, run_experiment

# %%
CACHE = Path("output/clip_vitb16_features.pt")

if not CACHE.exists():
    from transformers import CLIPModel, CLIPImageProcessor

    model_id = "openai/clip-vit-base-patch16"
    clip = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda").eval()
    proc = CLIPImageProcessor.from_pretrained(model_id)

    cfg_tmp = ExperimentConfig(run_name="clip_extract", use_cached_features=False)
    reader = _get_image_reader(cfg_tmp)
    df = pd.read_csv("output/processed_captions.csv")
    uniq = df[["image_id", "file_name"]].drop_duplicates().sort_values("image_id").reset_index(drop=True)
    N = len(uniq)
    D = clip.config.vision_config.hidden_size  # 768
    S = 196  # 14x14 patches after dropping CLS

    feats = torch.empty((N, S, D), dtype=torch.float16)
    image_to_idx: dict[int, int] = {}
    BATCH = 16

    batch_imgs, batch_pos = [], []
    with torch.no_grad():
        for i, row in tqdm(uniq.iterrows(), total=N, desc="CLIP extract"):
            img = reader.read(row["file_name"])
            batch_imgs.append(img)
            batch_pos.append(i)
            image_to_idx[int(row["image_id"])] = i
            if len(batch_imgs) == BATCH:
                px = proc(images=batch_imgs, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.float16)
                out = clip.vision_model(pixel_values=px).last_hidden_state
                out = out[:, 1:, :].cpu().to(torch.float16)  # drop CLS
                for k, p in enumerate(batch_pos):
                    feats[p] = out[k]
                batch_imgs, batch_pos = [], []
        if batch_imgs:
            px = proc(images=batch_imgs, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.float16)
            out = clip.vision_model(pixel_values=px).last_hidden_state
            out = out[:, 1:, :].cpu().to(torch.float16)
            for k, p in enumerate(batch_pos):
                feats[p] = out[k]

    torch.save({"features": feats, "image_to_idx": image_to_idx}, CACHE)
    print(f"Saved CLIP cache {tuple(feats.shape)} to {CACHE}")

    del clip
    import gc
    gc.collect()
    torch.cuda.empty_cache()
else:
    print(f"CLIP cache already at {CACHE}")

# %%
cfg = ExperimentConfig(
    run_name="phase2_e5_clip",
    output_dir="output/phase2_results",
    feature_cache=str(CACHE),
    encoder_kind="efficientnet_b0_cached",  # generic cached-feature head
    encoder_feat_dim=768,
    num_spatial_tokens=196,
    epochs=15,
    batch_size=128,
    lr=1e-4,
    decoding="beam",
    beam_width=5,
    length_penalty=0.7,
)
metrics_e5 = run_experiment(cfg)
metrics_e5
