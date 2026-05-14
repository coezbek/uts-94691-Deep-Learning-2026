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
# # E7b — img2img generative augmentation: visual probe of 50 images
#
# Before committing to a full augmentation run, generate 50 sample triplets
# `(original, low-denoise, high-denoise)` so we can eyeball whether SDXL-Turbo
# keeps the VizWiz "look" (blur, weird angles) at strength 0.3 and how far it
# drifts at strength 0.6. The prompt for both is the existing caption — so
# the synthetic image is supposed to still be a valid match for the same
# label.

# %%
import random
import time
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

OUT_DIR = Path("output/phase2_results/img2img_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
from diffusers import AutoPipelineForImage2Image

MODEL_ID = "stabilityai/sdxl-turbo"
print(f"Loading {MODEL_ID} ...")
t0 = time.time()
pipe = AutoPipelineForImage2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"Loaded in {time.time()-t0:.1f}s.")

# %%
df = pd.read_csv("output/processed_captions.csv")
train_pairs = (
    df[df["split"] == "train"][["image_id", "file_name", "caption_clean"]]
    .drop_duplicates("image_id")
    .reset_index(drop=True)
)
rng = random.Random(42)
indices = rng.sample(range(len(train_pairs)), 50)
sample = train_pairs.iloc[indices].reset_index(drop=True)
print(f"Sampled {len(sample)} unique training images.")

# %%
IMG_DIR = Path("data/val")
STRENGTHS = [0.3, 0.6]
# SDXL-Turbo expects num_inference_steps ≈ 4 for full generation;
# img2img uses round(steps * strength), so for strength 0.3 we need at
# least 4/0.3 ≈ 14 steps to actually run a few denoising steps.
NUM_STEPS = 8

results: list[dict] = []
t0 = time.time()
for i, row in sample.iterrows():
    orig = Image.open(IMG_DIR / row["file_name"]).convert("RGB").resize((512, 512), Image.BILINEAR)
    caption = str(row["caption_clean"])
    out_row = {"image_id": int(row["image_id"]), "file_name": row["file_name"],
               "caption": caption, "original": orig}
    for s in STRENGTHS:
        gen_t0 = time.time()
        with torch.inference_mode():
            out = pipe(
                prompt=caption,
                image=orig,
                strength=s,
                num_inference_steps=NUM_STEPS,
                guidance_scale=0.0,  # Turbo is trained for guidance_scale=0
            )
        gen_t = time.time() - gen_t0
        out_row[f"strength_{s}"] = out.images[0]
        out_row[f"strength_{s}_secs"] = gen_t
    results.append(out_row)
    if (i + 1) % 5 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) * len(STRENGTHS) / elapsed
        print(f"  {i+1}/{len(sample)}  rate={rate:.2f} gens/s  elapsed={elapsed:.0f}s", flush=True)

print(f"All generations done in {time.time()-t0:.1f}s.")

# %%
# Save individual PNGs for easy browsing
for r in results:
    base = f"{int(r['image_id'])}"
    r["original"].save(OUT_DIR / f"{base}_0_original.png")
    for s in STRENGTHS:
        r[f"strength_{s}"].save(OUT_DIR / f"{base}_s{s}.png")
print(f"Wrote individual PNGs to {OUT_DIR}")

# %%
# Composite grid: each row = one image. Columns = original / s=0.3 / s=0.6 + caption
N_PER_PAGE = 10
n_pages = (len(results) + N_PER_PAGE - 1) // N_PER_PAGE
for page in range(n_pages):
    sub = results[page * N_PER_PAGE : (page + 1) * N_PER_PAGE]
    fig, axes = plt.subplots(len(sub), 3, figsize=(16, 4.5 * len(sub)))
    if len(sub) == 1:
        axes = axes[None, :]
    for r, ax_row in zip(sub, axes):
        ax_row[0].imshow(r["original"]);          ax_row[0].set_title(f"original  id={r['image_id']}", fontsize=10)
        ax_row[1].imshow(r["strength_0.3"]);      ax_row[1].set_title(f"strength=0.3  ({r['strength_0.3_secs']:.1f}s)", fontsize=10)
        ax_row[2].imshow(r["strength_0.6"]);      ax_row[2].set_title(f"strength=0.6  ({r['strength_0.6_secs']:.1f}s)", fontsize=10)
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])
        ax_row[0].set_ylabel(r["caption"][:60], fontsize=9, color="#1a237e")
    fig.suptitle(f"SDXL-Turbo img2img probe — page {page+1}/{n_pages}", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    p = OUT_DIR / f"img2img_grid_page{page+1}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")
