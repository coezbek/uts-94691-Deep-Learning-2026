"""Qwen3.5-9B VL captioning baseline on 100 VizWiz test images.

Compare a frontier vision-language model against Model 1 and Model 2 on the
same 100-image sample (seed=0). Reports BLEU-1..4 and CIDEr against the same
reference captions our supervised models were scored on.

Notes:
- Qwen3.5-9B is loaded via `transformers.AutoModelForImageTextToText`. Total
  ~9 B params; in fp16 that's ~18 GB, comfortable inside the Strix Halo's
  128 GB unified RAM. Inference is single-image at a time and uses the
  chat-template interface with thinking mode disabled (we want a one-line
  caption, not a reasoning trace).
- Output is post-processed (lowercase, drop punctuation, truncate to 25
  tokens) to match the cleaned-reference style our BLEU / CIDEr scorers
  expect.
- Runs deterministically against `seed=0`. The same 100 image_ids are used
  every time so the comparison is repeatable.

Run with:
    HSA_OVERRIDE_GFX_VERSION=11.5.1 .venv/bin/python -u phase2_qwen35_9b_baseline.py
"""
import json
import math
import re
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

# Reuse the same metrics implementations that the consolidated notebook uses,
# so the numbers reported here are directly comparable to Model 1 and Model 2.
sys.path.insert(0, str(Path("output").resolve()))
from exp_runner import corpus_bleu, corpus_cider  # noqa: E402

# --- Configuration ---------------------------------------------------------
MODEL_ID = "Qwen/Qwen3.5-9B"
N_IMAGES = 100
SEED = 0
MAX_NEW_TOKENS = 60
IMAGE_DIR = Path("data/val")
OUTPUT_DIR = Path("output") / "qwen35_9b_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Sample the same 100 test images every run ----------------------------
df = pd.read_csv("output/processed_captions.csv")
test_df = (
    df[df["split"] == "test"][["image_id", "file_name"]]
    .drop_duplicates()
    .sort_values("image_id")
    .reset_index(drop=True)
)
sample = test_df.sample(n=min(N_IMAGES, len(test_df)),
                        random_state=SEED).reset_index(drop=True)
references = json.loads(Path("output/reference_captions.json").read_text())
print(f"Sampled {len(sample)} / {len(test_df)} test images (seed={SEED}).")


# --- Caption post-processing ----------------------------------------------
# The references are lowercased, punctuation-stripped, and tokenised on
# whitespace. We apply the same transform to Qwen's output before scoring so
# we're comparing like with like.
def clean_qwen_caption(text: str, max_tokens: int = 25) -> str:
    # Strip thinking blocks if any leaked through, drop leading/trailing whitespace.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Lowercase and strip everything that isn't a letter, digit, apostrophe, or space.
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate to a reasonable caption length (references median ~10 words).
    return " ".join(text.split()[:max_tokens])


# --- Load Qwen3.5-9B -------------------------------------------------------
print(f"Loading {MODEL_ID} … (one-time download; weights ~18 GB at fp16)")
from transformers import AutoModelForImageTextToText, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map=device,
)
model.eval()
print(f"Loaded. device={device}")


# --- Caption-generation prompt --------------------------------------------
# VizWiz references are short (mean 11.7 words), descriptive, no punctuation.
# Tell Qwen explicitly to match that style — otherwise it produces full
# sentences with capital letters and periods.
PROMPT_TEXT = (
    "Write one short caption that describes the contents of this image. "
    "Match this style: 10 to 15 words, lowercase, no punctuation, "
    "no preamble. Just the caption text."
)


@torch.no_grad()
def caption_one(image_path: Path) -> str:
    """Run one Qwen forward pass + post-processing for a single image."""
    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        chat_template_kwargs={"enable_thinking": False},
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    raw = processor.decode(
        out[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    return clean_qwen_caption(raw)


# --- Main loop -------------------------------------------------------------
predictions = []
refs_for_eval = []
rows = []
t0 = time.time()
for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Qwen3.5-9B caption"):
    iid = int(row["image_id"])
    fname = row["file_name"]
    refs = references[str(iid)]
    pred = caption_one(IMAGE_DIR / fname)
    predictions.append(pred)
    refs_for_eval.append(refs)
    rows.append({"image_id": iid, "file_name": fname,
                 "prediction": pred, "references": refs})
elapsed = time.time() - t0
print(f"Done. {len(sample)} captions in {elapsed:.0f}s "
      f"({elapsed / max(len(sample), 1):.2f}s per image).")


# --- Score and save --------------------------------------------------------
bleu = corpus_bleu(predictions, refs_for_eval)
cider = corpus_cider(predictions, refs_for_eval)
metrics = {**bleu, "CIDEr": cider,
           "n_images": len(sample), "elapsed_sec": elapsed,
           "model": MODEL_ID, "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS}

pred_df = pd.DataFrame(rows)
pred_df.to_csv(OUTPUT_DIR / "qwen35_9b_predictions.csv", index=False)
json.dump(metrics, open(OUTPUT_DIR / "qwen35_9b_metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))


# --- Side-by-side with Model 1 / Model 2 on the same 100 images -----------
# Load the per-image predictions written by the consolidated notebook so we
# can rescore them on the SAME 100 image_ids. This makes the comparison fair
# even though the supervised models were also scored on the full test split.
def _rescore_on_sample(pred_path: Path, label: str) -> dict | None:
    if not pred_path.exists():
        print(f"  {label}: predictions file not found at {pred_path}, skipping.")
        return None
    base = pd.read_csv(pred_path)
    # The references column round-trips as a string repr of a list; parse it.
    base["references"] = base["references"].apply(lambda s: eval(s) if isinstance(s, str) else s)
    sub = base.merge(sample[["image_id"]], on="image_id", how="inner")
    p = sub["prediction"].astype(str).tolist()
    r = sub["references"].tolist()
    if len(p) == 0:
        print(f"  {label}: no overlap with sample, skipping.")
        return None
    sub_bleu = corpus_bleu(p, r)
    sub_cider = corpus_cider(p, r)
    return {"model": label, "n": len(p), **sub_bleu, "CIDEr": sub_cider}


rows_cmp = []
m1_path = Path("output/model1_26239780/model1_test_beam_predictions.csv")
m2_path = Path("output/model2_26239780/model2_test_predictions.csv")
m1_row = _rescore_on_sample(m1_path, "Model 1 (EffNet-B0 + 3L Tx, beam w=5 lp=0.7)")
m2_row = _rescore_on_sample(m2_path, "Model 2 (SigLIP2 + 2L small,  beam w=3 lp=0.7)")
if m1_row: rows_cmp.append(m1_row)
if m2_row: rows_cmp.append(m2_row)
rows_cmp.append({"model": f"Qwen3.5-9B (zero-shot, {N_IMAGES} images)",
                 "n": len(sample), **bleu, "CIDEr": cider})

cmp_df = pd.DataFrame(rows_cmp)
cmp_df.to_csv(OUTPUT_DIR / "qwen35_9b_vs_supervised.csv", index=False)
print("\nSide-by-side on the same 100 test images:")
print(cmp_df.to_string(index=False))
