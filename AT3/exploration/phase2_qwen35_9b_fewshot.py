"""Qwen3.5-9B few-shot captioning baseline on 100 VizWiz test images.

Same eval as `phase2_qwen35_9b_baseline.py` but with 5 (image, caption) pairs
from the **training split** prepended to every prompt as in-context
demonstrations of the VizWiz reference style. Plus an optional KV-cache reuse
across the 100 test images so the few-shot prefix is computed once instead of
100 times.

Run:
    HSA_OVERRIDE_GFX_VERSION=11.5.1 .venv/bin/python -u phase2_qwen35_9b_fewshot.py
"""
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

# Reduce CUDA memory fragmentation — same env var as the zero-shot baseline.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path("output").resolve()))
from exp_runner import corpus_bleu, corpus_cider  # noqa: E402

# --- Configuration --------------------------------------------------------
MODEL_ID = "Qwen/Qwen3.5-9B"
N_TEST_IMAGES = 100
TEST_SEED = 0
FEWSHOT_SEED = 0
N_FEWSHOT = 5
MAX_NEW_TOKENS = 40
MAX_IMAGE_SIDE = 448  # same cap as the zero-shot baseline to avoid OOM

# If True, attempt to reuse the few-shot prefix's KV cache across all 100
# target images. This saves ~1500 tokens of prefill per image. If False (or if
# caching errors on the first try), the prefix is recomputed every call.
TRY_KV_CACHE = True

IMAGE_DIR = Path("data/val")
OUTPUT_DIR = Path("output") / "qwen35_9b_fewshot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Sample test split (same 100 images as the zero-shot run) -------------
df = pd.read_csv("output/processed_captions.csv")
test_df = (
    df[df["split"] == "test"][["image_id", "file_name"]]
    .drop_duplicates()
    .sort_values("image_id")
    .reset_index(drop=True)
)
test_sample = test_df.sample(n=min(N_TEST_IMAGES, len(test_df)),
                             random_state=TEST_SEED).reset_index(drop=True)
references = json.loads(Path("output/reference_captions.json").read_text())
print(f"Sampled {len(test_sample)} / {len(test_df)} test images "
      f"(test_seed={TEST_SEED}).")


# --- Pick 5 few-shot examples from the train split ------------------------
# Diverse content matters more than fancy sampling: we want the model to see
# the *style* (short, lowercase, no punctuation, generic vocabulary) rather
# than memorise any one phrasing. Pick 5 random train images deterministically.
train_df_unique = (
    df[df["split"] == "train"][["image_id", "file_name"]]
    .drop_duplicates()
    .sort_values("image_id")
    .reset_index(drop=True)
)
fewshot = train_df_unique.sample(
    n=N_FEWSHOT, random_state=FEWSHOT_SEED
).reset_index(drop=True)
# For each few-shot image, the first reference caption is the demo target.
# Train captions live in df rows (one row per caption); we want the first
# *valid* reference per image, sourced from reference_captions.json so the
# demo string matches what BLEU would see at eval.
fewshot_examples = []
for _, row in fewshot.iterrows():
    iid = int(row["image_id"])
    refs = references[str(iid)]
    if not refs:
        continue
    fewshot_examples.append({
        "image_id": iid,
        "file_name": row["file_name"],
        "caption": refs[0],  # use the first valid reference as the demo
    })
print(f"Few-shot examples (n={len(fewshot_examples)}, "
      f"fewshot_seed={FEWSHOT_SEED}):")
for ex in fewshot_examples:
    print(f"  image_id={ex['image_id']}  caption: {ex['caption']}")


# --- Caption post-processing (same as zero-shot) --------------------------
def clean_qwen_caption(text: str, max_tokens: int = 25) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(text.split()[:max_tokens])


def resize_image(img: Image.Image) -> Image.Image:
    """Cap longest side to MAX_IMAGE_SIDE; keeps per-image patch count modest."""
    w, h = img.size
    scale = MAX_IMAGE_SIDE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


# --- Load Qwen3.5-9B ------------------------------------------------------
print(f"Loading {MODEL_ID} …")
from transformers import AutoModelForImageTextToText, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map=device,
)
model.eval()
print(f"Loaded. device={device}  dtype={next(model.parameters()).dtype}")


# --- Build the few-shot prefix (shared across all target images) ----------
SYSTEM_INSTRUCTION = (
    "You caption images in the VizWiz-Captions style. "
    "Each caption must be 10 to 15 words, lowercase, no punctuation, no preamble. "
    "Describe what is in the image, prefer common everyday words over brand names, "
    "and write a single line."
)


def build_prefix_messages():
    """Return the chat messages for system instruction + 5 demonstrations."""
    msgs = [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]},
    ]
    for ex in fewshot_examples:
        img = resize_image(Image.open(IMAGE_DIR / ex["file_name"]).convert("RGB"))
        msgs.append({"role": "user",
                     "content": [{"type": "image", "image": img},
                                 {"type": "text", "text": "Caption this image."}]})
        msgs.append({"role": "assistant",
                     "content": [{"type": "text", "text": ex["caption"]}]})
    return msgs


PREFIX_MESSAGES = build_prefix_messages()


def build_target_messages(img: Image.Image):
    """Few-shot prefix + one user turn with the target image."""
    return PREFIX_MESSAGES + [
        {"role": "user",
         "content": [{"type": "image", "image": img},
                     {"type": "text", "text": "Caption this image."}]}
    ]


# --- Try to cache the prefix's KV across calls ---------------------------
# Pattern: run the prefix-only chat through the model once with use_cache=True,
# stash past_key_values, then for each target call generate() with that cache.
# Qwen3.5's hybrid attention layers may not all support cache reuse; if the
# attempt errors on the very first target, fall back to no-caching.
cached_kv = None
cached_prefix_len = 0
if TRY_KV_CACHE:
    try:
        prefix_inputs = processor.apply_chat_template(
            PREFIX_MESSAGES,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(device)
        with torch.no_grad():
            prefix_out = model(**prefix_inputs, use_cache=True)
        cached_kv = prefix_out.past_key_values
        cached_prefix_len = prefix_inputs.input_ids.size(1)
        print(f"KV cache built. prefix_len={cached_prefix_len} tokens.")
        del prefix_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"KV cache build failed ({type(e).__name__}: {e}); falling back "
              f"to recomputing prefix per call.")
        cached_kv = None


@torch.no_grad()
def caption_one(image_path: Path) -> str:
    """Generate one caption with the few-shot prefix in place."""
    img = resize_image(Image.open(image_path).convert("RGB"))
    target_messages = build_target_messages(img)
    inputs = processor.apply_chat_template(
        target_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(device)

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    # If we have a cache, hand it to generate; the model will skip recomputing
    # KV for positions covered by the cache.
    if cached_kv is not None:
        gen_kwargs["past_key_values"] = cached_kv

    out = model.generate(**inputs, **gen_kwargs)
    raw = processor.decode(
        out[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    return clean_qwen_caption(raw)


# --- Main eval loop -------------------------------------------------------
predictions, refs_for_eval, rows = [], [], []
caching_failed_mid_run = False
t0 = time.time()
for _, row in tqdm(test_sample.iterrows(), total=len(test_sample),
                   desc="Qwen3.5-9B few-shot caption"):
    iid = int(row["image_id"])
    fname = row["file_name"]
    try:
        pred = caption_one(IMAGE_DIR / fname)
    except Exception as e:
        # If cache reuse breaks partway through, log once and retry without it.
        if cached_kv is not None and not caching_failed_mid_run:
            caching_failed_mid_run = True
            print(f"\nCache reuse failed on image_id={iid} "
                  f"({type(e).__name__}: {e}); disabling cache and retrying.")
            cached_kv = None
            pred = caption_one(IMAGE_DIR / fname)
        else:
            raise
    refs = references[str(iid)]
    predictions.append(pred)
    refs_for_eval.append(refs)
    rows.append({"image_id": iid, "file_name": fname,
                 "prediction": pred, "references": refs})
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f"Done. {len(test_sample)} captions in {elapsed:.0f}s "
      f"({elapsed / max(len(test_sample), 1):.2f}s per image). "
      f"KV cache used: {cached_kv is not None and not caching_failed_mid_run}.")


# --- Score and save --------------------------------------------------------
bleu = corpus_bleu(predictions, refs_for_eval)
cider = corpus_cider(predictions, refs_for_eval)
metrics = {**bleu, "CIDEr": cider,
           "n_images": len(test_sample), "elapsed_sec": elapsed,
           "model": MODEL_ID, "test_seed": TEST_SEED, "fewshot_seed": FEWSHOT_SEED,
           "n_fewshot": len(fewshot_examples),
           "kv_cache_used": cached_kv is not None and not caching_failed_mid_run,
           "max_new_tokens": MAX_NEW_TOKENS}

pred_df = pd.DataFrame(rows)
pred_df.to_csv(OUTPUT_DIR / "qwen35_9b_fewshot_predictions.csv", index=False)
json.dump(metrics, open(OUTPUT_DIR / "qwen35_9b_fewshot_metrics.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))


# --- Side-by-side with the zero-shot Qwen + supervised models -------------
def _rescore_on_sample(pred_path: Path, label: str):
    if not pred_path.exists():
        return None
    base = pd.read_csv(pred_path)
    base["references"] = base["references"].apply(
        lambda s: eval(s) if isinstance(s, str) else s
    )
    sub = base.merge(test_sample[["image_id"]], on="image_id", how="inner")
    p = sub["prediction"].astype(str).tolist()
    r = sub["references"].tolist()
    if not p:
        return None
    return {"model": label, "n": len(p),
            **corpus_bleu(p, r), "CIDEr": corpus_cider(p, r)}


cmp_rows = []
for path, label in [
    (Path("output/model1_26239780/model1_test_beam_predictions.csv"),
     "Model 1 (EffNet-B0 + 3L Tx, beam w=5 lp=0.7)"),
    (Path("output/model2_26239780/model2_test_predictions.csv"),
     "Model 2 (SigLIP2 + 2L small,  beam w=3 lp=0.7)"),
    (Path("output/qwen35_9b_baseline/qwen35_9b_predictions.csv"),
     "Qwen3.5-9B (zero-shot)"),
]:
    row = _rescore_on_sample(path, label)
    if row:
        cmp_rows.append(row)
cmp_rows.append({"model": f"Qwen3.5-9B ({N_FEWSHOT}-shot, KV-cached={metrics['kv_cache_used']})",
                 "n": len(test_sample), **bleu, "CIDEr": cider})

cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(OUTPUT_DIR / "qwen35_9b_fewshot_vs_others.csv", index=False)
print("\nSide-by-side on the same 100 test images:")
print(cmp_df.to_string(index=False))
