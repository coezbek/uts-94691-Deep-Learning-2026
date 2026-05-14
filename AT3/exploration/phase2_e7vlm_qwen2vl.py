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
# # Phase 2 — E7-VLM: VLM-grounded caption augmentation
#
# E7 (text-only) used Qwen2.5 to paraphrase existing captions. That added
# paraphrastic variety but no new grounding — the LLM only saw the caption
# text. Here we use **Qwen2-VL-2B-Instruct** to caption images directly and
# append those captions to the training set, then retrain.
#
# Risk: Qwen2-VL might describe things the human references don't mention,
# which can hurt BLEU. We guide it with the existing references to keep
# style and detail level consistent.

# %%
import sys
import json
import re
import time
import gc
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
AUG_CSV = Path("output/processed_captions_e7vlm.csv")

if not AUG_CSV.exists():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    proc = AutoProcessor.from_pretrained(model_id)
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    print("Qwen2-VL loaded.")

    cfg_tmp = ExperimentConfig(run_name="vlm_extract", use_cached_features=False)
    reader = _get_image_reader(cfg_tmp)

    df = pd.read_csv("output/processed_captions.csv")
    train = df[df["split"] == "train"].reset_index(drop=True)
    # Sample one row per training image to bound runtime.
    train_unique = train.drop_duplicates("image_id").reset_index(drop=True)
    print(f"Will VLM-caption {len(train_unique)} unique training images.")

    SYS_PROMPT = (
        "You write short image captions for blind users (VizWiz style). "
        "Reply with ONE line, lowercase, no punctuation other than apostrophes, "
        "max 15 words. Stay grounded in the image — do NOT invent."
    )
    new_captions: list[str] = []
    t0 = time.time()
    for i, row in tqdm(train_unique.iterrows(), total=len(train_unique), desc="VLM caption"):
        pil = reader.read(row["file_name"])
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": "Caption this image in one short line, lowercase."},
            ]},
        ]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[pil], padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = vlm.generate(**inputs, max_new_tokens=32, do_sample=False)
        gen = out_ids[:, inputs["input_ids"].shape[1]:]
        decoded = proc.batch_decode(gen, skip_special_tokens=True)[0]
        new_captions.append(decoded.strip().splitlines()[0] if decoded.strip() else "")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(train_unique)}  elapsed={time.time()-t0:.0f}s")

    new_rows = []
    for (_, row), cap in zip(train_unique.iterrows(), new_captions):
        if not cap:
            continue
        r = row.copy()
        r["caption"] = cap
        c = re.sub(r"[^a-z0-9' ]+", " ", cap.lower()).strip()
        c = re.sub(r"\s+", " ", c).strip()
        r["caption_clean"] = c
        r["caption_word_count"] = len(c.split())
        new_rows.append(r)
    new_df = pd.DataFrame(new_rows)
    new_df = new_df[new_df["caption_word_count"] > 0]
    print(f"Added {len(new_df)} VLM-generated training rows")
    aug = pd.concat([train, new_df], ignore_index=True)
    full_df = pd.concat([aug, df[df["split"] != "train"]], ignore_index=True)
    full_df.to_csv(AUG_CSV, index=False)
    print(f"Saved {AUG_CSV} with {len(full_df):,} total rows")

    del vlm
    gc.collect()
    torch.cuda.empty_cache()
else:
    print(f"Augmented CSV already at {AUG_CSV}")

# %%
cfg = ExperimentConfig(
    run_name="phase2_e7vlm_qwen2vl",
    output_dir="output/phase2_results",
    captions_csv=str(AUG_CSV),
    epochs=12,
    batch_size=128,
    lr=1e-4,
    decoding="beam",
    beam_width=5,
    length_penalty=0.7,
)
metrics_e7vlm = run_experiment(cfg)
metrics_e7vlm
