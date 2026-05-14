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
# # E13 — Swap encoder to SigLIP2 ViT-B/16
#
# E05 already settled "use a vision-language pretrained encoder is the right
# call". SigLIP2 (Zhai et al., late 2024) is a strict drop-in upgrade — same
# patch count and embedding dim as CLIP ViT-B/16, but trained with the SigLIP
# objective which beats CLIP on captioning probes by 1–3 CIDEr in the
# literature. Extract patch features once, then train the same Transformer
# decoder we used in E05.

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
CACHE = Path("output/siglip2_b16_features.pt")

if not CACHE.exists():
    from transformers import AutoModel, AutoImageProcessor

    model_id = "google/siglip2-base-patch16-256"
    print(f"Loading {model_id} ...")
    siglip = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda").eval()
    proc = AutoImageProcessor.from_pretrained(model_id)

    cfg_tmp = ExperimentConfig(run_name="siglip2_extract", use_cached_features=False)
    reader = _get_image_reader(cfg_tmp)
    df = pd.read_csv("output/processed_captions.csv")
    uniq = df[["image_id", "file_name"]].drop_duplicates().sort_values("image_id").reset_index(drop=True)
    N = len(uniq)
    # SigLIP2-B/16-256: 256/16 = 16, so 16×16 = 256 patch tokens, no CLS.
    D = siglip.config.vision_config.hidden_size
    S = (siglip.config.vision_config.image_size // siglip.config.vision_config.patch_size) ** 2
    print(f"Cache shape will be ({N}, {S}, {D})")

    feats = torch.empty((N, S, D), dtype=torch.float16)
    image_to_idx: dict[int, int] = {}
    BATCH = 16
    batch_imgs, batch_pos = [], []
    with torch.no_grad():
        for i, row in tqdm(uniq.iterrows(), total=N, desc="SigLIP2 extract"):
            img = reader.read(row["file_name"])
            batch_imgs.append(img); batch_pos.append(i)
            image_to_idx[int(row["image_id"])] = i
            if len(batch_imgs) == BATCH:
                px = proc(images=batch_imgs, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.float16)
                out = siglip.vision_model(pixel_values=px).last_hidden_state  # (B, S, D) — no CLS for SigLIP2
                out = out.cpu().to(torch.float16)
                for k, p in enumerate(batch_pos):
                    feats[p] = out[k]
                batch_imgs, batch_pos = [], []
        if batch_imgs:
            px = proc(images=batch_imgs, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.float16)
            out = siglip.vision_model(pixel_values=px).last_hidden_state
            out = out.cpu().to(torch.float16)
            for k, p in enumerate(batch_pos):
                feats[p] = out[k]
    torch.save({"features": feats, "image_to_idx": image_to_idx}, CACHE)
    print(f"Saved SigLIP2 cache {tuple(feats.shape)} to {CACHE}")
    del siglip
    import gc; gc.collect(); torch.cuda.empty_cache()
else:
    print(f"SigLIP2 cache already at {CACHE}")

# %% [markdown]
# ## Train Transformer decoder on SigLIP2 features (same recipe as E05)

# %%
cache_blob = torch.load(CACHE, map_location="cpu", weights_only=False)
S = cache_blob["features"].shape[1]
D = cache_blob["features"].shape[2]
print(f"Loaded cache: S={S}, D={D}")

cfg = ExperimentConfig(
    run_name="phase2_e13_siglip2",
    output_dir="output/phase2_results",
    feature_cache=str(CACHE),
    encoder_kind="efficientnet_b0_cached",
    encoder_feat_dim=D,
    num_spatial_tokens=S,
    epochs=15,
    batch_size=128,
    lr=1e-4,
    decoding="beam", beam_width=5, length_penalty=0.7,
)
metrics_e13 = run_experiment(cfg)
metrics_e13
