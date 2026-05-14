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
# # Phase 2 — E3: Unfreeze last MBConv blocks (discriminative LR)
#
# Unfreeze the **last 2 MBConv blocks** of EfficientNet-B0 and train them with
# a small LR (`1e-5`). The decoder + projection head use the standard LR
# (`1e-4`). Train-time augmentation is on. Eval uses beam (w=5, lp=0.7).
#
# This is meant to be the largest single training-side win — Phase 1's
# frozen-trunk recipe leaves caption-relevant features on the table.

# %%
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path("output").resolve()))
import importlib
import exp_runner
importlib.reload(exp_runner)
from exp_runner import (
    ExperimentConfig,
    CaptioningModel,
    load_data,
    RawImageCaptionDataset,
    RawImageImageDataset,
    InMemoryImageCache,
    InMemoryReader,
    imagenet_train_transform,
    imagenet_eval_transform,
    caption_collate,
    image_collate,
    evaluate_split,
    make_progress_logger,
)

# %%
cfg = ExperimentConfig(
    run_name="phase2_e3_unfreeze",
    output_dir="output/phase2_results",
    use_cached_features=False,
    encoder_kind="efficientnet_b0_raw",
    freeze_encoder=False,
    unfreeze_last_k_blocks=2,
    augment=True,
    epochs=10,
    batch_size=48,
    lr=1e-4,
    decoding="beam",
    beam_width=5,
    length_penalty=0.7,
)
out_root = Path(cfg.output_dir) / cfg.run_name
out_root.mkdir(parents=True, exist_ok=True)
log = make_progress_logger(out_root)
device = torch.device("cuda")
log(f"[{cfg.run_name}] device={device}")

# %%
df, vocab, references, _, _ = load_data(cfg)
pad_idx = vocab["pad_idx"]
vocab_size = len(vocab["idx2word"])

model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)

# Discriminative LR: unfrozen backbone params at 1e-5, everything else at 1e-4
bb_params = [p for p in model.encoder.features.parameters() if p.requires_grad]
other_params = [p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith("encoder.features.")]
print(f"Backbone trainable: {sum(p.numel() for p in bb_params):,}")
print(f"Other trainable:    {sum(p.numel() for p in other_params):,}")

optimizer = torch.optim.Adam([
    {"params": bb_params, "lr": 1e-5},
    {"params": other_params, "lr": 1e-4},
])
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# %%
mem_cache = InMemoryImageCache(cfg.image_dir)
reader = InMemoryReader(mem_cache)
train_tx = imagenet_train_transform()
eval_tx = imagenet_eval_transform()

train_ds = RawImageCaptionDataset(df[df["split"] == "train"], vocab["word2idx"], references, reader, train_tx)
val_ds   = RawImageCaptionDataset(df[df["split"] == "val"],   vocab["word2idx"], references, reader, eval_tx)
val_img_ds  = RawImageImageDataset(df[df["split"] == "val"],   references, reader, eval_tx)
test_img_ds = RawImageImageDataset(df[df["split"] == "test"],  references, reader, eval_tx)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=0,  # in-memory cache; workers would just CoW it
                          collate_fn=caption_collate, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0, collate_fn=caption_collate, pin_memory=True)
val_img_loader = DataLoader(val_img_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, collate_fn=image_collate, pin_memory=True)
test_img_loader = DataLoader(test_img_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, collate_fn=image_collate, pin_memory=True)

# %%
best = float("inf")
bad = 0
ckpt_path = out_root / "best.pt"
history = []

log(f"[{cfg.run_name}] training: epochs={cfg.epochs} batches/epoch={len(train_loader)}")
for epoch in range(cfg.epochs):
    model.train()
    t0 = time.time()
    train_loss = 0.0
    n = 0
    last_heartbeat = time.time()
    for i, (imgs, caps, *_) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        caps = caps[:, : cfg.max_caption_len].to(device, non_blocking=True)
        logits = model(imgs, caps)
        targets = caps[:, 1:]
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        train_loss += loss.item()
        n += 1
        now = time.time()
        if now - last_heartbeat >= 15:
            log(f"[{cfg.run_name}] ep{epoch+1:02d}  batch {i+1}/{len(train_loader)}  "
                f"train_loss(running)={train_loss/n:.4f}  elapsed={now-t0:.1f}s")
            last_heartbeat = now
    train_loss /= max(n, 1)

    model.eval()
    val_loss = 0.0
    vn = 0
    with torch.no_grad():
        for imgs, caps, *_ in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            caps = caps[:, : cfg.max_caption_len].to(device, non_blocking=True)
            logits = model(imgs, caps)
            targets = caps[:, 1:]
            val_loss += criterion(logits.reshape(-1, vocab_size), targets.reshape(-1)).item()
            vn += 1
    val_loss /= max(vn, 1)

    dt = time.time() - t0
    history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "time": dt})
    log(f"[{cfg.run_name}] ep{epoch + 1:02d}/{cfg.epochs} train={train_loss:.4f} val={val_loss:.4f} t={dt:.1f}s")

    if val_loss < best:
        best = val_loss
        bad = 0
        torch.save(model.state_dict(), ckpt_path)
        log(f"[{cfg.run_name}]   ! best so far ({val_loss:.4f}) — checkpoint saved")
    else:
        bad += 1
    if bad >= cfg.early_stop_patience:
        log(f"[{cfg.run_name}] early stop at epoch {epoch + 1}")
        break

pd.DataFrame(history).to_csv(out_root / "history.csv", index=False)

# %%
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
model.eval()
val_metrics, val_preds = evaluate_split(model, val_img_loader, vocab, cfg, split="val")
test_metrics, test_preds = evaluate_split(model, test_img_loader, vocab, cfg, split="test")

metrics_e3 = {
    "run_name": cfg.run_name,
    "best_val_loss": best,
    "epochs_run": len(history),
    "decoding": cfg.decoding,
    "beam_width": cfg.beam_width,
    "length_penalty": cfg.length_penalty,
    **{f"val_{k}": v for k, v in val_metrics.items()},
    **{f"test_{k}": v for k, v in test_metrics.items()},
}
val_preds.to_csv(out_root / "val_predictions.csv", index=False)
test_preds.to_csv(out_root / "test_predictions.csv", index=False)
json.dump(metrics_e3, open(out_root / "metrics.json", "w"), indent=2)
metrics_e3
