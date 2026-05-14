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
# # E10b — SCST with KL anchor (fixes E10's reward hacking)
#
# E10 regressed (test BLEU-4 0.290 vs E05's 0.389) despite the corpus-IDF fix.
# Diagnosis from the val metrics: BLEU-1 fell 0.638→0.301, i.e. the captions'
# core vocabulary drifted off the references. With no anchor and 2 epochs the
# policy learned to chase per-reference idiosyncratic words that happened to
# carry huge corpus-IDF weight (rare-in-training words appearing in only 1 of
# 5 in-image references → big reward spike). This is reward hacking.
#
# **Two changes vs E10:**
#
# 1. **KL anchor toward frozen E05.** A frozen copy of E05 acts as the reference
#    policy π_ref. Loss = −(advantage · log π_θ(sample)) + α · KL(π_θ || π_ref).
#    We use the cheap sample-based KL estimate: log π_θ(sample) − log π_ref(sample).
#    α=0.1 is a conservative default; can lower if reward stalls, raise if drift.
# 2. **One epoch only.** E10's per-epoch trace shows epoch 1 was already drifting;
#    epoch 2 made it worse. Cap at one epoch.
#
# Same corpus-IDF reward (loaded from E10's saved `corpus_df.pt`) and same
# warm-start (E05 best.pt).

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
    make_progress_logger,
)

# %% [markdown]
# ## 1. Config — KL-anchored SCST on E05

# %%
cfg = ExperimentConfig(
    run_name="phase2_e10b_scst_kl",
    output_dir="output/phase2_results",
    feature_cache="output/clip_vitb16_features.pt",
    encoder_kind="efficientnet_b0_cached",
    encoder_feat_dim=768,
    num_spatial_tokens=196,
    epochs=1,                              # cap at one epoch
    batch_size=32,
    lr=5e-6,
    grad_clip=1.0,
    decoding="beam", beam_width=5, length_penalty=0.7,
    pretrained_checkpoint="output/phase2_results/phase2_e05_clip/best.pt",
)
KL_ALPHA = 0.1                              # KL penalty weight

out_root = Path(cfg.output_dir) / cfg.run_name
out_root.mkdir(parents=True, exist_ok=True)
log = make_progress_logger(out_root)
device = torch.device("cuda")
log(f"[{cfg.run_name}] device={device}  KL_alpha={KL_ALPHA}  epochs={cfg.epochs}  lr={cfg.lr}")

# %% [markdown]
# ## 2. Load corpus IDF from E10

# %%
df_all, vocab, references, features, image_to_idx = load_data(cfg)
word2idx = vocab["word2idx"]; idx2word = vocab["idx2word"]
PAD = word2idx["<pad>"]; START = word2idx["<start>"]; END = word2idx["<end>"]; UNK = word2idx["<unk>"]
vocab_size = len(idx2word)

E10_DF_PATH = Path("output/phase2_results/phase2_e10_scst_corpus_idf/corpus_df.pt")
blob = torch.load(E10_DF_PATH, map_location="cpu", weights_only=False)
corpus_df = blob["corpus_df"]; log_num_docs = blob["log_num_docs"]; N_MAX = blob["N_MAX"]
log(f"Loaded corpus DF from E10: {sum(len(c) for c in corpus_df):,} n-grams across n=1..{N_MAX}")

SIGMA = 6.0
EPS = 1e-12


def _ngrams_at(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _tfidf_vec(tokens, n):
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
        p_vec = _tfidf_vec(pred_tokens, n)
        if not p_vec:
            per_n.append(0.0); continue
        sims = []
        for rt in ref_tokens_lists:
            r_vec = _tfidf_vec(rt, n)
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


# %% [markdown]
# ## 3. Build trainable model (warm-start E05) and frozen reference copy

# %%
model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=PAD).to(device)
ref_model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=PAD).to(device)

sd = torch.load(cfg.pretrained_checkpoint, map_location=device, weights_only=False)
model.load_state_dict(sd)
ref_model.load_state_dict(sd)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False
log(f"Loaded E05 weights into both trainable and frozen-reference models")

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

# %% [markdown]
# ## 4. SCST training loop with KL anchor
#
# For each batch:
#
# 1. Encode image with the current model and (independently) with the frozen
#    reference. Both encoders see the same features.
# 2. Get greedy baseline caption from current model.
# 3. Sample caption from current model, recording per-token log π_θ.
# 4. Score sample and baseline with corpus-IDF CIDEr (rewards).
# 5. Compute log π_ref of the same sampled token sequence (teacher-forced).
# 6. Loss = −(advantage · log π_θ(sample)).mean()  +  α · (log π_θ − log π_ref).mean()

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


@torch.no_grad()
def ref_logp_of(ref_memory, sample_toks):
    """Teacher-forced log π_ref(sample_toks[1:] | sample_toks[:-1]), summed per row,
    masking PAD targets so they don't contribute."""
    inp = sample_toks[:, :-1]
    tgt = sample_toks[:, 1:]
    logits = ref_model.decoder(ref_memory, inp)
    logp = F.log_softmax(logits, dim=-1)
    gathered = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    mask = (tgt != PAD).float()
    return (gathered * mask).sum(dim=1)


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
log(f"Starting KL-anchored SCST: epochs={cfg.epochs}  batches/epoch={len(train_img_loader)}  lr={cfg.lr}  alpha={KL_ALPHA}")

for epoch in range(cfg.epochs):
    model.train()
    t0 = time.time()
    n_b = 0
    total_r = 0.0
    total_kl = 0.0
    last_heartbeat = t0
    for feats, image_ids, file_names, refs in train_img_loader:
        feats = feats.to(device, non_blocking=True)
        memory = model.encode(feats)
        with torch.no_grad():
            ref_memory = ref_model.encode(feats)

        with torch.no_grad():
            base_toks = greedy_baseline_caps(memory)
            base_strs = ids_to_str(base_toks)

        samp_toks, samp_logp = sample_caps(memory)
        samp_strs = ids_to_str(samp_toks)

        # log π_ref on the sampled token sequence (teacher-forced)
        ref_logp = ref_logp_of(ref_memory, samp_toks)

        r_sample = np.array([cidered_one(s.split(), [str(r).split() for r in refs[i]])
                             for i, s in enumerate(samp_strs)])
        r_baseline = np.array([cidered_one(s.split(), [str(r).split() for r in refs[i]])
                               for i, s in enumerate(base_strs)])
        advantage = torch.tensor(r_sample - r_baseline, device=device, dtype=torch.float32)

        rl_loss = -(advantage * samp_logp).mean()
        kl = (samp_logp - ref_logp).mean()        # biased sample-based KL estimate
        loss = rl_loss + KL_ALPHA * kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total_r += float(r_sample.mean())
        total_kl += float(kl.item())
        n_b += 1
        if time.time() - last_heartbeat >= 15:
            log(f"  ep{epoch+1} batch {n_b}/{len(train_img_loader)}  "
                f"sample_CIDEr={total_r/n_b:.3f}  baseline_CIDEr={float(r_baseline.mean()):.3f}  "
                f"KL={total_kl/n_b:.3f}  rl_loss={rl_loss.item():.4f}  total_loss={loss.item():.4f}")
            last_heartbeat = time.time()

    mean_r = total_r / max(n_b, 1)
    mean_kl = total_kl / max(n_b, 1)
    dt = time.time() - t0
    history.append({"epoch": epoch + 1, "mean_sample_cider": mean_r, "mean_kl": mean_kl, "time": dt})
    log(f"SCST ep{epoch+1}/{cfg.epochs}  mean_sample_CIDEr={mean_r:.3f}  mean_KL={mean_kl:.3f}  t={dt:.1f}s")

torch.save(model.state_dict(), out_root / "best.pt")
pd.DataFrame(history).to_csv(out_root / "history.csv", index=False)

# %% [markdown]
# ## 5. Beam eval

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
    "kl_alpha": KL_ALPHA,
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
