"""Regenerate processed_captions / vocab / reference_captions / utils using Raina's prep.

We do NOT need raw images or annotations.zip — only the existing processed_captions.csv
(which already contains the raw `caption` column and the image-level split column).
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

NB_DIR = Path(__file__).parent
OUTPUT_DIR = NB_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SRC_CSV = OUTPUT_DIR / "processed_captions.csv"
assert SRC_CSV.exists(), f"Missing {SRC_CSV} — needed as source of raw captions."

print(f"Loading {SRC_CSV} ...")
df = pd.read_csv(SRC_CSV)
print(f"  rows: {len(df):,}  cols: {list(df.columns)}")

# --- Raina's clean_caption (no <num> mapping) ---
def clean_caption(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["caption_clean"] = df["caption"].apply(clean_caption)
df["caption_word_count"] = df["caption_clean"].str.split().apply(len)

before = len(df)
df = df[df["caption_word_count"] > 0].copy()
print(f"  dropped {before - len(df)} empty-after-clean rows")

# --- Build vocab from internal train split only ---
MIN_WORD_FREQ = 3
SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]

word_counter: Counter[str] = Counter()
for tokens in df.loc[df["split"] == "train", "caption_clean"].str.split():
    word_counter.update(tokens)

vocab_words = [w for w, c in sorted(word_counter.items()) if c >= MIN_WORD_FREQ and w not in SPECIAL_TOKENS]
idx2word = SPECIAL_TOKENS + vocab_words
word2idx = {w: i for i, w in enumerate(idx2word)}

vocab = {
    "word2idx": word2idx,
    "idx2word": idx2word,
    "min_word_freq": MIN_WORD_FREQ,
    "special_tokens": SPECIAL_TOKENS,
    "pad_idx": word2idx["<pad>"],
    "start_idx": word2idx["<start>"],
    "end_idx": word2idx["<end>"],
    "unk_idx": word2idx["<unk>"],
}

print(f"Vocab size: {len(idx2word)}  (specials: {SPECIAL_TOKENS})")
print(f"Has <num>? {'<num>' in word2idx}")

with open(OUTPUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)
print(f"Saved {OUTPUT_DIR/'vocab.json'}")

# --- Reference captions ---
refs = (
    df.groupby("image_id")["caption_clean"].apply(list).to_dict()
)
refs = {str(k): v for k, v in refs.items()}
with open(OUTPUT_DIR / "reference_captions.json", "w", encoding="utf-8") as f:
    json.dump(refs, f)
print(f"Saved {OUTPUT_DIR/'reference_captions.json'} ({len(refs)} images)")

# --- Save processed_captions.csv (overwrite with Raina-clean version) ---
shutil.copy(SRC_CSV, OUTPUT_DIR / "processed_captions.co_backup.csv")
df.to_csv(SRC_CSV, index=False)
print(f"Wrote {SRC_CSV} (backup at processed_captions.co_backup.csv)")

# Counts per split
print("Rows per split:")
print(df.groupby("split").size())
print("Unique images per split:")
print(df.groupby("split")["image_id"].nunique())
