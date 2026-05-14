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
# # E12 — Smaller, more-regularised decoder on CLIP features
#
# The overfitting analysis pinned ~0.6-nat best-epoch gap with E05's 18 M
# parameter decoder on ~26 k caption rows. Shrink the decoder and crank
# regularisation:
#
# * **2 layers × d_model=384** (≈ 6 M params, your own suggestion)
# * **Dropout 0.3** (up from 0.2)
# * **Label smoothing 0.15**
# * **Caption-side word dropout 0.10** — randomly replace 10 % of training
#   token ids with `<unk>` (forces the decoder to lean on context, not memorise rare words)
# * **Early-stop on val BLEU-4** rather than val cross-entropy
#
# Same CLIP ViT-B/16 cached features as E05 so the only thing changing is the
# decoder + training recipe. Likely smaller per-model BLEU than E05 but a
# **different bias** → useful as a new ensemble member.

# %%
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path("output").resolve()))
import importlib
import exp_runner
importlib.reload(exp_runner)
from exp_runner import (
    ExperimentConfig,
    CaptioningModel,
    load_data,
    CachedFeatureCaptionDataset,
    CachedFeatureImageDataset,
    caption_collate,
    image_collate,
    evaluate_split,
    corpus_bleu,
    corpus_cider,
    make_progress_logger,
)

# %% [markdown]
# ## 1. Config — smaller decoder over CLIP features

# %%
cfg = ExperimentConfig(
    run_name="phase2_e12_small_decoder",
    output_dir="output/phase2_results",
    feature_cache="output/clip_vitb16_features.pt",
    encoder_kind="efficientnet_b0_cached",
    encoder_feat_dim=768,
    num_spatial_tokens=196,
    # Decoder size
    embed_dim=384,
    num_heads=6,
    num_decoder_layers=2,
    ffn_dim=1536,
    dropout=0.3,
    # Training recipe
    epochs=20,
    batch_size=128,
    lr=1e-4,
    weight_decay=0.02,
    optimizer="adamw",
    label_smoothing=0.15,
    early_stop_patience=5,
    grad_clip=1.0,
    decoding="beam", beam_width=5, length_penalty=0.7,
)
out_root = Path(cfg.output_dir) / cfg.run_name
out_root.mkdir(parents=True, exist_ok=True)
log = make_progress_logger(out_root)
device = torch.device("cuda")
log(f"[{cfg.run_name}] device={device}  decoder=2L×d={cfg.embed_dim}  dropout={cfg.dropout}  LS={cfg.label_smoothing}  WD={cfg.weight_decay}")

# %% [markdown]
# ## 2. Build model and data
#
# Caption-side word dropout is applied **only to the decoder input** inside the
# training loop below (not in the dataset). Masking the full caption tensor
# before splitting into input/target would corrupt the loss target — we'd train
# the decoder to *emit* `<unk>` 10% of the time. The standard formulation keeps
# the target clean and only adds noise to the teacher-forced input.

# %%
P_UNK = 0.10                       # word-dropout rate on decoder input

# %%
df, vocab, references, features, image_to_idx = load_data(cfg)
word2idx = vocab["word2idx"]; idx2word = vocab["idx2word"]
PAD = word2idx["<pad>"]; UNK = word2idx["<unk>"]; START = word2idx["<start>"]; END = word2idx["<end>"]
vocab_size = len(idx2word)

model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=PAD).to(device)
log(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=cfg.lr, weight_decay=cfg.weight_decay,
)
criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.label_smoothing)

train_ds = CachedFeatureCaptionDataset(df[df["split"] == "train"], word2idx, references, features, image_to_idx)
SPECIAL_IDS = (PAD, START, END, UNK)
val_ds = CachedFeatureCaptionDataset(df[df["split"] == "val"], word2idx, references, features, image_to_idx)
val_img_ds = CachedFeatureImageDataset(df[df["split"] == "val"], references, features, image_to_idx)
test_img_ds = CachedFeatureImageDataset(df[df["split"] == "test"], references, features, image_to_idx)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=caption_collate, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=caption_collate, pin_memory=True)
val_img_loader = DataLoader(val_img_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate, pin_memory=True)
test_img_loader = DataLoader(test_img_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate, pin_memory=True)

# %% [markdown]
# ## 3. Training loop with caption-input word dropout, early-stop on **val BLEU-4** (greedy)

# %%
@torch.no_grad()
def quick_greedy_bleu(loader):
    model.eval()
    preds, refs_all = [], []
    for feats, image_ids, file_names, refs in loader:
        feats = feats.to(device, non_blocking=True)
        memory = model.encode(feats)
        B = memory.size(0)
        toks = torch.full((B, 1), START, device=device, dtype=torch.long)
        fin = torch.zeros(B, dtype=torch.bool, device=device)
        for step in range(cfg.gen_max_len):
            logits = model.decoder(memory, toks)[:, -1, :].clone()
            logits[:, [PAD, START, UNK]] = -float("inf")
            if step + 1 < cfg.gen_min_len:
                logits[:, END] = -float("inf")
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            nxt = torch.where(fin.unsqueeze(1), torch.full_like(nxt, PAD), nxt)
            toks = torch.cat([toks, nxt], dim=1)
            fin = fin | (nxt.squeeze(1) == END)
            if fin.all():
                break
        for i in range(B):
            words = []
            for tid in toks[i].tolist()[1:]:
                if tid in (END, PAD):
                    break
                words.append(idx2word[tid])
            preds.append(" ".join(words))
            refs_all.append(refs[i])
    return corpus_bleu(preds, refs_all)["BLEU-4"]


best_bleu = -1.0; bad = 0; ckpt = out_root / "best.pt"; history = []
log(f"Starting training: epochs={cfg.epochs}  batches/epoch={len(train_loader)}")

for epoch in range(cfg.epochs):
    model.train()
    t0 = time.time()
    tl = 0.0; n = 0; last_hb = t0
    for i, (feats, caps, *_) in enumerate(train_loader):
        feats = feats.to(device, non_blocking=True)
        caps = caps[:, : cfg.max_caption_len].to(device, non_blocking=True)
        targets = caps[:, 1:]
        caps_in = caps[:, :-1].clone()
        if P_UNK > 0:
            mask = torch.rand_like(caps_in, dtype=torch.float) < P_UNK
            for s in SPECIAL_IDS:
                mask &= (caps_in != s)
            caps_in = torch.where(mask, torch.full_like(caps_in, UNK), caps_in)
        memory = model.encode(feats)
        logits = model.decoder(memory, caps_in)
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); optimizer.step()
        tl += loss.item(); n += 1
        now = time.time()
        if now - last_hb >= 15:
            log(f"  ep{epoch+1:02d} batch {i+1}/{len(train_loader)} train_loss={tl/n:.4f} elapsed={now-t0:.1f}s")
            last_hb = now
    train_loss = tl / max(n, 1)

    model.eval()
    vl = 0.0; vn = 0
    with torch.no_grad():
        for feats, caps, *_ in val_loader:
            feats = feats.to(device, non_blocking=True)
            caps = caps[:, : cfg.max_caption_len].to(device, non_blocking=True)
            logits = model(feats, caps)
            targets = caps[:, 1:]
            vl += criterion(logits.reshape(-1, vocab_size), targets.reshape(-1)).item(); vn += 1
    val_loss = vl / max(vn, 1)
    val_bleu = quick_greedy_bleu(val_img_loader)

    dt = time.time() - t0
    history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss,
                    "val_BLEU-4_greedy": val_bleu, "time": dt})
    log(f"ep{epoch+1:02d}/{cfg.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
        f"val_BLEU-4(g)={val_bleu:.4f}  t={dt:.1f}s")

    if val_bleu > best_bleu:
        best_bleu = val_bleu; bad = 0
        torch.save(model.state_dict(), ckpt)
        log(f"  ! new best val_BLEU-4={val_bleu:.4f} — saved")
    else:
        bad += 1
    if bad >= cfg.early_stop_patience:
        log(f"early stop at epoch {epoch+1}")
        break

pd.DataFrame(history).to_csv(out_root / "history.csv", index=False)

# %% [markdown]
# ## 4. Beam eval

# %%
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
model.eval()
val_metrics, val_preds = evaluate_split(model, val_img_loader, vocab, cfg, split="val")
test_metrics, test_preds = evaluate_split(model, test_img_loader, vocab, cfg, split="test")

metrics = {
    "run_name": cfg.run_name,
    "best_val_BLEU-4_greedy": best_bleu,
    "epochs_run": len(history),
    "decoding": cfg.decoding,
    "beam_width": cfg.beam_width,
    "length_penalty": cfg.length_penalty,
    **{f"val_{k}": v for k, v in val_metrics.items()},
    **{f"test_{k}": v for k, v in test_metrics.items()},
}
val_preds.to_csv(out_root / "val_predictions.csv", index=False)
test_preds.to_csv(out_root / "test_predictions.csv", index=False)
json.dump(metrics, open(out_root / "metrics.json", "w"), indent=2)
log(f"DONE  val_BLEU-4={val_metrics['BLEU-4']:.4f}  test_BLEU-4={test_metrics['BLEU-4']:.4f}  test_CIDEr={test_metrics['CIDEr']:.4f}")
metrics
