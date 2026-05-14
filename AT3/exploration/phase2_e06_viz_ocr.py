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
# # E6 — visualise 16 images with their detected OCR
#
# Sanity-check EasyOCR's output: 16 images that have at least a few OCR
# tokens, displayed in a 4×4 grid with the detected text below each image.
# Helps explain why E6 didn't move the needle.

# %%
import ast
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec

OCR_PATH = Path("output/easyocr_per_image.json")
CAP_PATH = Path("output/processed_captions.csv")
IMG_DIR = Path("data/val")
OUT_PNG = Path("output/phase2_results/phase2_e6_ocr_viz_grid.png")
OUT_PDF = Path("output/phase2_results/phase2_e6_ocr_viz_grid.pdf")

assert OCR_PATH.exists(), f"Missing {OCR_PATH}"
ocr = json.loads(OCR_PATH.read_text(encoding="utf-8"))

df = pd.read_csv(CAP_PATH)
test_df = df[df["split"] == "test"][["image_id", "file_name"]].drop_duplicates()
# Build reference captions from the dataframe — first 3 only to keep titles short
ref_map = (
    df.groupby("image_id")["caption_clean"]
    .apply(list)
    .to_dict()
)

# Filter to images with ≥ 5 OCR words so the grid is informative
candidates = []
for _, row in test_df.iterrows():
    iid = str(int(row["image_id"]))
    text = ocr.get(iid, "")
    if len(text.split()) >= 5:
        candidates.append((row["image_id"], row["file_name"], text))
print(f"{len(candidates)} test images have ≥5 OCR tokens (of {len(test_df)} total)")

rng = random.Random(42)
rng.shuffle(candidates)
sample = candidates[:16]

# %%
def wrap(text: str, width: int = 50) -> str:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


fig = plt.figure(figsize=(20, 24))
outer = mgridspec.GridSpec(4, 4, figure=fig, hspace=0.18, wspace=0.06,
                           top=0.96, bottom=0.04, left=0.02, right=0.98)

for i, (iid, file_name, ocr_text) in enumerate(sample):
    inner = mgridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[i // 4, i % 4],
        height_ratios=[1.2, 4.0, 1.6], hspace=0.05,
    )
    ref_ax = fig.add_subplot(inner[0])
    img_ax = fig.add_subplot(inner[1])
    ocr_ax = fig.add_subplot(inner[2])

    # Top: first reference for context
    refs = ref_map.get(iid, [])
    ref_text = refs[0] if refs else ""
    ref_ax.set_axis_off()
    ref_ax.text(0.5, 0.5, f"caption: {wrap(ref_text, 45)}",
                ha="center", va="center", fontsize=10, color="#1a237e",
                transform=ref_ax.transAxes)

    # Middle: the image itself
    img = Image.open(IMG_DIR / file_name).convert("RGB")
    img_ax.imshow(img)
    img_ax.set_xticks([]); img_ax.set_yticks([])
    for s in img_ax.spines.values():
        s.set_visible(False)

    # Bottom: OCR text
    ocr_ax.set_axis_off()
    ocr_ax.text(0.5, 0.5, f"OCR: {wrap(ocr_text, 45)}",
                ha="center", va="center", fontsize=10, color="#b71c1c",
                transform=ocr_ax.transAxes, weight="bold")

fig.suptitle(
    "E6 — EasyOCR detections on 16 random test images with ≥5 OCR tokens\n"
    "blue = first reference caption,   red = OCR text fed into the model",
    fontsize=14, y=0.99,
)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
# PDF: vector for text, embedded raster for the imshow images, infinite zoom on captions.
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"Saved {OUT_PNG}")
print(f"Saved {OUT_PDF}")
plt.show()
