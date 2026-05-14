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
# # E10 — SCST with corpus-IDF CIDEr reward, on the E05 CLIP checkpoint
#
# E08 collapsed because the CIDEr reward used the *local* (per-image) document
# frequency built from only the 5 references for that image. That over-weights
# rare-within-5-refs n-grams and pushes captions toward narrow, repetitive
# outputs. Here we **precompute the document frequency from the full training
# caption corpus once**, then run REINFORCE with a greedy-decode baseline for
# 2 epochs. Warm-start from the E05 checkpoint (CLIP ViT-B/16 + Transformer)
# because that's our strongest MLE model.
#
# Reward: corpus-CIDEr-D. Loss: `-(sample_reward - baseline_reward) · log_p`
# averaged per image. Optional KL anchor toward the MLE policy can be added
# if we see drift.

# %%
import json, math, time
from pathlib import Path
from collections import Counter

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
    CachedFeatureImageDataset,
    image_collate,
    evaluate_split,
    corpus_bleu,
    make_progress_logger,
)

# %% [markdown]
# ## 1. Config — warm-start from E05 (best MLE)

# %%
cfg = ExperimentConfig(
    run_name="phase2_e10_scst_corpus_idf",
    output_dir="output/phase2_results",
    feature_cache="output/clip_vitb16_features.pt",
    encoder_kind="efficientnet_b0_cached",     # cached-feature head over CLIP
    encoder_feat_dim=768,
    num_spatial_tokens=196,
    epochs=2,
    batch_size=32,
    lr=5e-6,
    grad_clip=1.0,
    decoding="beam", beam_width=5, length_penalty=0.7,
    pretrained_checkpoint="output/phase2_results/phase2_e05_clip/best.pt",
)
out_root = Path(cfg.output_dir) / cfg.run_name
out_root.mkdir(parents=True, exist_ok=True)
log = make_progress_logger(out_root)
device = torch.device("cuda")
log(f"[{cfg.run_name}] device={device}")

# %% [markdown]
# ## 2. Precompute corpus IDF from the training references
#
# `df[n][ngram]` counts the number of *training reference documents* that
# contain `ngram`. The number of documents is the count of unique references
# in the training split. Same definition as the COCO CIDEr-D scorer.

# %%
df_all, vocab, references, features, image_to_idx = load_data(cfg)
word2idx = vocab["word2idx"]
idx2word = vocab["idx2word"]
PAD = word2idx["<pad>"]; START = word2idx["<start>"]; END = word2idx["<end>"]; UNK = word2idx["<unk>"]
vocab_size = len(idx2word)
N_MAX = 4

train_refs: list[list[str]] = []
for refs_for_img in df_all[df_all["split"] == "train"].groupby("image_id")["caption_clean"].apply(list):
    train_refs.extend(refs_for_img)
log(f"Corpus IDF over {len(train_refs)} training reference captions.")

corpus_df: list[Counter] = [Counter() for _ in range(N_MAX)]
for ref in train_refs:
    toks = ref.split()
    for n in range(1, N_MAX + 1):
        seen = set(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
        for ng in seen:
            corpus_df[n - 1][ng] += 1

num_docs = len(train_refs)
log_num_docs = math.log(max(num_docs, 1))

# Persist for downstream notebooks (E14 MBR pairwise CIDEr re-uses these).
torch.save(
    {"corpus_df": corpus_df, "log_num_docs": log_num_docs, "N_MAX": N_MAX, "num_docs": num_docs},
    out_root / "corpus_df.pt",
)
log(f"Saved corpus DF ({sum(len(c) for c in corpus_df):,} n-grams across n=1..{N_MAX}) to {out_root / 'corpus_df.pt'}")

# %% [markdown]
# ## 3. Fast corpus-IDF CIDEr-D
#
# Same formula as in the captioning literature, but using the precomputed
# `corpus_df` instead of building DF from each image's 5 refs.

# %%
SIGMA = 6.0
EPS = 1e-12


def _ngrams_at(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def tfidf_vec(tokens, n):
    counts = Counter(_ngrams_at(tokens, n))
    total = sum(counts.values())
    if total == 0:
        return {}
    out = {}
    for ng, c in counts.items():
        tf = c / total
        idf = log_num_docs - math.log(max(corpus_df[n - 1].get(ng, 0), 1))
        out[ng] = tf * idf
    return out


def cidered_one(pred_tokens, ref_tokens_lists):
    per_n = []
    for n in range(1, N_MAX + 1):
        p_vec = tfidf_vec(pred_tokens, n)
        if not p_vec:
            per_n.append(0.0); continue
        sims = []
        for rt in ref_tokens_lists:
            r_vec = tfidf_vec(rt, n)
            if not r_vec:
                sims.append(0.0); continue
            common = set(p_vec) & set(r_vec)
            num = sum(p_vec[ng] * r_vec[ng] for ng in common)
            pn = math.sqrt(sum(v * v for v in p_vec.values())) + EPS
            rn = math.sqrt(sum(v * v for v in r_vec.values())) + EPS
            cos = num / (pn * rn)
            delta = len(pred_tokens) - len(rt)
            penalty = math.exp(-(delta * delta) / (2 * SIGMA * SIGMA))
            sims.append(cos * penalty)
        per_n.append(10.0 * (sum(sims) / max(len(sims), 1)))
    return sum(per_n) / N_MAX


def corpus_cider_idf(predictions, references_list):
    return float(np.mean([cidered_one(str(p).split(), [str(r).split() for r in refs])
                          for p, refs in zip(predictions, references_list)]))


# %% [markdown]
# ## 4. Model — load E05 checkpoint

# %%
model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=PAD).to(device)
sd = torch.load(cfg.pretrained_checkpoint, map_location=device, weights_only=False)
model.load_state_dict(sd)
log(f"Warm-started from {cfg.pretrained_checkpoint}")

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

# %% [markdown]
# ## 5. SCST training loop

# %%
train_img_df = df_all[df_all["split"] == "train"][["image_id", "file_name"]].drop_duplicates()
train_img_ds = CachedFeatureImageDataset(train_img_df, references, features, image_to_idx)
train_img_loader = DataLoader(train_img_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, collate_fn=image_collate, pin_memory=True)


@torch.no_grad()
def greedy_baseline_caps(memory):
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
    return toks


def sample_caps(memory):
    B = memory.size(0)
    toks = torch.full((B, 1), START, device=device, dtype=torch.long)
    logp_sum = torch.zeros(B, device=device)
    fin = torch.zeros(B, dtype=torch.bool, device=device)
    for step in range(cfg.gen_max_len):
        logits = model.decoder(memory, toks)[:, -1, :].clone()
        logits[:, [PAD, START, UNK]] = -float("inf")
        if step + 1 < cfg.gen_min_len:
            logits[:, END] = -float("inf")
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        nxt = torch.multinomial(probs, 1)
        nxt_logp = log_probs.gather(1, nxt).squeeze(1)
        logp_sum = logp_sum + torch.where(fin, torch.zeros_like(nxt_logp), nxt_logp)
        nxt = torch.where(fin.unsqueeze(1), torch.full_like(nxt, PAD), nxt)
        toks = torch.cat([toks, nxt], dim=1)
        fin = fin | (nxt.squeeze(1) == END)
        if fin.all():
            break
    return toks, logp_sum


def ids_to_str(toks):
    out = []
    for row in toks.tolist():
        words = []
        for tid in row[1:]:
            if tid in (END, PAD):
                break
            words.append(idx2word[tid])
        out.append(" ".join(words))
    return out


history = []
log(f"Starting SCST: epochs={cfg.epochs}  batches/epoch={len(train_img_loader)}  lr={cfg.lr}")

for epoch in range(cfg.epochs):
    model.train()
    t0 = time.time()
    n_b = 0
    total_r = 0.0
    last_heartbeat = t0
    for feats, image_ids, file_names, refs in train_img_loader:
        feats = feats.to(device, non_blocking=True)
        memory = model.encode(feats)

        with torch.no_grad():
            base_toks = greedy_baseline_caps(memory)
            base_strs = ids_to_str(base_toks)

        samp_toks, samp_logp = sample_caps(memory)
        samp_strs = ids_to_str(samp_toks)

        r_sample = np.array([cidered_one(s.split(), [str(r).split() for r in refs[i]])
                             for i, s in enumerate(samp_strs)])
        r_baseline = np.array([cidered_one(s.split(), [str(r).split() for r in refs[i]])
                               for i, s in enumerate(base_strs)])
        advantage = torch.tensor(r_sample - r_baseline, device=device, dtype=torch.float32)
        loss = -(advantage * samp_logp).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_r += float(r_sample.mean())
        n_b += 1
        if time.time() - last_heartbeat >= 15:
            log(f"  ep{epoch+1} batch {n_b}/{len(train_img_loader)}  "
                f"mean_sample_CIDEr={total_r/n_b:.3f}  "
                f"mean_baseline_CIDEr={float(r_baseline.mean()):.3f}  "
                f"loss={loss.item():.4f}")
            last_heartbeat = time.time()

    mean_r = total_r / max(n_b, 1)
    dt = time.time() - t0
    history.append({"epoch": epoch + 1, "mean_sample_cider": mean_r, "time": dt})
    log(f"SCST ep{epoch+1}/{cfg.epochs}  mean_sample_CIDEr={mean_r:.3f}  t={dt:.1f}s")

torch.save(model.state_dict(), out_root / "best.pt")
pd.DataFrame(history).to_csv(out_root / "history.csv", index=False)

# %% [markdown]
# ## 6. Beam eval

# %%
val_df = df_all[df_all["split"] == "val"][["image_id", "file_name"]].drop_duplicates()
test_df = df_all[df_all["split"] == "test"][["image_id", "file_name"]].drop_duplicates()
val_loader = DataLoader(CachedFeatureImageDataset(val_df, references, features, image_to_idx),
                        batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate)
test_loader = DataLoader(CachedFeatureImageDataset(test_df, references, features, image_to_idx),
                         batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate)
model.eval()
val_metrics, val_preds = evaluate_split(model, val_loader, vocab, cfg, split="val")
test_metrics, test_preds = evaluate_split(model, test_loader, vocab, cfg, split="test")

metrics = {
    "run_name": cfg.run_name,
    "epochs_run": cfg.epochs,
    "decoding": cfg.decoding,
    "beam_width": cfg.beam_width,
    "length_penalty": cfg.length_penalty,
    **{f"val_{k}": v for k, v in val_metrics.items()},
    **{f"test_{k}": v for k, v in test_metrics.items()},
}
val_preds.to_csv(out_root / "val_predictions.csv", index=False)
test_preds.to_csv(out_root / "test_predictions.csv", index=False)
json.dump(metrics, open(out_root / "metrics.json", "w"), indent=2)
log(f"DONE val_BLEU-4={val_metrics['BLEU-4']:.4f}  test_BLEU-4={test_metrics['BLEU-4']:.4f}  "
    f"test_CIDEr={test_metrics['CIDEr']:.4f}")
metrics
