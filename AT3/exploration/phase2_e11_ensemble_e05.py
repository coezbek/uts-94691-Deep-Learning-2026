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
# # E11 — E05-anchored logit ensemble
#
# E09 averaged E00 + E04 + E07 (all EfficientNet-backed) and beat its best
# member by +0.010 BLEU-4. Repeat with **E05 as the anchor** plus the
# strongest diverse signal we have (E07 paraphrase-augmented training data).
# E05 brings CLIP visual features; E07 brings training-data diversity. Each
# contributes a different inductive bias, so logit-averaging their next-token
# distributions during beam search should extract more than any single seed
# of E05 would.
#
# **E06 was excluded.** E06's defining signal is OCR memory tokens, but the
# `CaptioningModel` class we instantiate here does not have an OCR branch —
# loading E06's checkpoint with `strict=False` silently drops every OCR-
# specific weight, leaving an EfficientNet+Transformer that is essentially a
# differently-seeded copy of E07. Not the orthogonal signal we wanted; better
# to use a clean two-member ensemble than a misleading three-member one.
# Adding E10 (SCST-tuned policy), E12 (smaller decoder), or E13 (SigLIP2)
# once those checkpoints exist is the right way to widen the ensemble.
#
# **Caveat on dims:** E05 was trained on 196 CLIP patch tokens (d_model=512),
# E07 on 49 EfficientNet patch tokens (d_model=512, feat=1280). They use the
# **same vocabulary** so the *output* logits over a 4708-token vocab are
# directly averageable, but the encoders are different. The ensemble works
# at the logit level after each model has run its own forward pass — we
# don't try to share memory.

# %%
import json
import math
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import pandas as pd
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
    corpus_bleu,
    corpus_cider,
    make_progress_logger,
)

OUT_ROOT = Path("output/phase2_results/phase2_e11_ensemble_e05")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
log = make_progress_logger(OUT_ROOT)
device = torch.device("cuda")

# %% [markdown]
# ## 1. Per-model configs
#
# Each member has its own feature cache + encoder dimensions. We instantiate
# three `CaptioningModel`s with the right config, load each checkpoint, and
# index into the matching feature tensor per image.

# %%
MEMBERS = [
    {
        "name": "E05",
        "ckpt": "output/phase2_results/phase2_e05_clip/best.pt",
        "feature_cache": "output/clip_vitb16_features.pt",
        "encoder_feat_dim": 768,
        "num_spatial_tokens": 196,
    },
    {
        "name": "E07",
        "ckpt": "output/phase2_results/phase2_e07_text_paraphrase/best.pt",
        "feature_cache": "output/model1_26239780/efficientnet_b0_raw_features.pt",
        "encoder_feat_dim": 1280,
        "num_spatial_tokens": 49,
    },
    {
        "name": "E12",
        "ckpt": "output/phase2_results/phase2_e12_small_decoder/best.pt",
        "feature_cache": "output/clip_vitb16_features.pt",
        "encoder_feat_dim": 768,
        "num_spatial_tokens": 196,
        # Heterogeneous decoder: smaller, more-regularised
        "embed_dim": 384, "num_heads": 6, "num_decoder_layers": 2,
        "ffn_dim": 1536, "dropout": 0.3,
    },
    {
        "name": "E13",
        "ckpt": "output/phase2_results/phase2_e13_siglip2/best.pt",
        "feature_cache": "output/siglip2_b16_features.pt",
        "encoder_feat_dim": 768,
        "num_spatial_tokens": 256,
    },
]


def build_model(m):
    # Heterogeneous decoder hyperparameters: members may carry their own embed_dim,
    # num_decoder_layers, etc. Defaults match E05/E07 (the original recipe).
    cfg = ExperimentConfig(
        run_name="ensemble_member_" + m["name"],
        output_dir="output/phase2_results",
        feature_cache=m["feature_cache"],
        encoder_kind="efficientnet_b0_cached",
        encoder_feat_dim=m["encoder_feat_dim"],
        num_spatial_tokens=m["num_spatial_tokens"],
        embed_dim=m.get("embed_dim", 512),
        num_heads=m.get("num_heads", 8),
        num_decoder_layers=m.get("num_decoder_layers", 3),
        ffn_dim=m.get("ffn_dim", 2048),
        dropout=m.get("dropout", 0.2),
    )
    # Use vocab/captions/refs from any cfg — they're the same across members.
    df, vocab, references, features, image_to_idx = load_data(cfg)
    pad_idx = vocab["pad_idx"]; vocab_size = len(vocab["idx2word"])
    model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)
    sd = torch.load(m["ckpt"], map_location=device, weights_only=False)
    # Every current member is a plain CaptioningModel, so strict-load is expected
    # to succeed. We keep the strict=False fallback only so the loader degrades
    # loudly (logging missing/unexpected keys) instead of crashing if a future
    # member ships extra layers — but we DO want to see those warnings.
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"  WARNING: {m['name']} loaded partial — missing={len(missing)} unexpected={len(unexpected)}; "
            f"its training-time inductive bias may not transfer.")
    model.eval()
    return model, features, image_to_idx, vocab, df, references


log("Loading 3 ensemble members ...")
members = []
shared_vocab = None
shared_df = None
shared_refs = None
for m in MEMBERS:
    model, features, image_to_idx, vocab, df, refs = build_model(m)
    members.append({**m, "model": model, "features": features, "image_to_idx": image_to_idx})
    shared_vocab = vocab
    shared_df = df
    shared_refs = refs
    log(f"  loaded {m['name']}  encoder_feat_dim={m['encoder_feat_dim']}  "
        f"num_spatial={m['num_spatial_tokens']}  trainable_ckpt={Path(m['ckpt']).stat().st_size/1e6:.1f}MB")

vocab = shared_vocab
word2idx = vocab["word2idx"]; idx2word = vocab["idx2word"]
PAD = word2idx["<pad>"]; START = word2idx["<start>"]; END = word2idx["<end>"]; UNK = word2idx["<unk>"]

BEAM = 5
LP = 0.7
GEN_MAX = 20
GEN_MIN = 4

# %% [markdown]
# ## 2. Ensemble beam decoder — averages log-probs across members

# %%
@torch.no_grad()
def ensemble_beam(image_id: int) -> str:
    # Encode the image once per member
    memories = []
    for m in members:
        feat = m["features"][m["image_to_idx"][int(image_id)]].float().unsqueeze(0).to(device)
        memories.append(m["model"].encode(feat))

    beams = [([START], 0.0, False)]
    for step in range(GEN_MAX):
        if all(b[2] for b in beams):
            break
        active = [(i, b) for i, b in enumerate(beams) if not b[2]]
        seqs = torch.tensor([b[0] for _, b in active], device=device, dtype=torch.long)
        avg_log = None
        for m, mem in zip(members, memories):
            mem_e = mem.expand(seqs.size(0), -1, -1).contiguous()
            logits = m["model"].decoder(mem_e, seqs)[:, -1, :]
            lp = F.log_softmax(logits, dim=-1)
            avg_log = lp if avg_log is None else avg_log + lp
        avg_log = avg_log / len(members)
        avg_log[:, [PAD, START, UNK]] = -float("inf")
        if step + 1 < GEN_MIN:
            avg_log[:, END] = -float("inf")
        topk_lp, topk_id = avg_log.topk(BEAM, dim=-1)
        cands = []
        for ai, (_, (toks, sc, _)) in enumerate(active):
            for k in range(BEAM):
                tid = int(topk_id[ai, k].item())
                s = sc + float(topk_lp[ai, k].item())
                cands.append((toks + [tid], s, tid == END))
        for b in beams:
            if b[2]:
                cands.append(b)
        def sf(it):
            t, s, _ = it
            return s / (max(len(t) - 1, 1) ** LP)
        cands.sort(key=sf, reverse=True)
        beams = cands[:BEAM]
    def sf(it):
        t, s, _ = it; return s / (max(len(t) - 1, 1) ** LP)
    best_seq = max(beams, key=sf)[0][1:]
    words = []
    for tid in best_seq:
        if tid in (END, PAD):
            break
        words.append(idx2word[tid])
    return " ".join(words)


# %% [markdown]
# ## 3. Evaluate val + test

# %%
def eval_split(split):
    sdf = shared_df[shared_df["split"] == split][["image_id", "file_name"]].drop_duplicates()
    preds = []
    refs_all = []
    rows = []
    log(f"Eval {split}: {len(sdf)} images")
    import time as _time
    t0 = _time.time()
    last_log = t0
    for i, row in sdf.reset_index(drop=True).iterrows():
        p = ensemble_beam(int(row["image_id"]))
        preds.append(p)
        r = shared_refs[str(int(row["image_id"]))]
        refs_all.append(r)
        rows.append({"image_id": int(row["image_id"]),
                     "file_name": row["file_name"],
                     "prediction": p,
                     "references": r})
        now = _time.time()
        if now - last_log >= 15:
            log(f"  {split} {i+1}/{len(sdf)}  elapsed={now-t0:.0f}s")
            last_log = now
    bm = corpus_bleu(preds, refs_all); bm["CIDEr"] = corpus_cider(preds, refs_all)
    return bm, pd.DataFrame(rows)


val_m, val_p = eval_split("val")
log(f"val: {val_m}")
test_m, test_p = eval_split("test")
log(f"test: {test_m}")

# %% [markdown]
# ## 4. Save

# %%
metrics = {
    "run_name": "phase2_e11_ensemble_e05",
    "ensembled_runs": [m["name"] for m in MEMBERS],
    "decoding": "beam", "beam_width": BEAM, "length_penalty": LP,
    **{f"val_{k}": v for k, v in val_m.items()},
    **{f"test_{k}": v for k, v in test_m.items()},
}
val_p.to_csv(OUT_ROOT / "val_predictions.csv", index=False)
test_p.to_csv(OUT_ROOT / "test_predictions.csv", index=False)
json.dump(metrics, open(OUT_ROOT / "metrics.json", "w"), indent=2)
log(f"DONE  val_BLEU-4={val_m['BLEU-4']:.4f}  test_BLEU-4={test_m['BLEU-4']:.4f}  test_CIDEr={test_m['CIDEr']:.4f}")
metrics
