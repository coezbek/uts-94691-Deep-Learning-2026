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
# # E14 — Minimum Bayes Risk (MBR) decoding over the E11 ensemble
#
# For each image:
#
# 1. Generate **K = 16** beam candidates from the E11 ensemble (E05 + E07) —
#    same logit-averaging trick as E11.
# 2. For every candidate, compute the **mean sentence-CIDEr against the other
#    K−1 candidates** using the **corpus IDF precomputed by E10**. That's the
#    empirical risk under the (proper) CIDEr-D utility — not a uniform-IDF
#    approximation, so common stopwords don't dominate the consensus.
# 3. Pick the candidate with the highest mean consensus.
#
# MBR is complementary to logit-averaging: averaging picks the next-token
# distribution that all members agree on; MBR picks the *whole sequence*
# that's closest to the other sequences. Together they typically add
# ~+0.005–0.010 BLEU-4 over plain beam ensembling.
#
# **Prerequisite:** run E10 first so that `corpus_df.pt` exists. If it's
# missing, this notebook rebuilds the DF from the training references — same
# definition as E10 — at startup cost ~15 s.

# %%
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
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
    corpus_bleu,
    corpus_cider,
    make_progress_logger,
)

OUT_ROOT = Path("output/phase2_results/phase2_e14_mbr")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
log = make_progress_logger(OUT_ROOT)
device = torch.device("cuda")

# Mirror E11's member set: E05 (CLIP anchor) + E07 (paraphrase-augmented).
# E06 is intentionally excluded — see E11 for the rationale.
MEMBERS = [
    {"name": "E05", "ckpt": "output/phase2_results/phase2_e05_clip/best.pt",
     "feature_cache": "output/clip_vitb16_features.pt", "encoder_feat_dim": 768, "num_spatial_tokens": 196},
    {"name": "E07", "ckpt": "output/phase2_results/phase2_e07_text_paraphrase/best.pt",
     "feature_cache": "output/model1_26239780/efficientnet_b0_raw_features.pt", "encoder_feat_dim": 1280, "num_spatial_tokens": 49},
    {"name": "E12", "ckpt": "output/phase2_results/phase2_e12_small_decoder/best.pt",
     "feature_cache": "output/clip_vitb16_features.pt", "encoder_feat_dim": 768, "num_spatial_tokens": 196,
     "embed_dim": 384, "num_heads": 6, "num_decoder_layers": 2, "ffn_dim": 1536, "dropout": 0.3},
    {"name": "E13", "ckpt": "output/phase2_results/phase2_e13_siglip2/best.pt",
     "feature_cache": "output/siglip2_b16_features.pt", "encoder_feat_dim": 768, "num_spatial_tokens": 256},
]

K = 16   # candidates per image
LP = 0.7
GEN_MAX = 20
GEN_MIN = 4


def build_model(m):
    # Heterogeneous decoder hyperparameters: members may carry their own embed_dim,
    # num_decoder_layers, etc. Defaults match E05/E07 (the original recipe).
    cfg = ExperimentConfig(
        run_name="mbr_member_" + m["name"],
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
    df, vocab, references, features, image_to_idx = load_data(cfg)
    pad_idx = vocab["pad_idx"]; vocab_size = len(vocab["idx2word"])
    model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)
    sd = torch.load(m["ckpt"], map_location=device, weights_only=False)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"  WARNING: {m['name']} partial-loaded — missing={len(missing)} unexpected={len(unexpected)}; "
            f"its training-time inductive bias may not transfer.")
    model.eval()
    return model, features, image_to_idx, vocab, df, references


log(f"Building ensemble for MBR (K={K} candidates/image)")
members = []
shared = None
for m in MEMBERS:
    model, features, image_to_idx, vocab, df, refs = build_model(m)
    members.append({**m, "model": model, "features": features, "image_to_idx": image_to_idx})
    shared = (vocab, df, refs)
vocab, shared_df, shared_refs = shared
word2idx = vocab["word2idx"]; idx2word = vocab["idx2word"]
PAD = word2idx["<pad>"]; START = word2idx["<start>"]; END = word2idx["<end>"]; UNK = word2idx["<unk>"]

# ---------------------------------------------------------------------------
# Corpus IDF for proper MBR-CIDEr: load from E10 if present, otherwise rebuild.
# ---------------------------------------------------------------------------
N_MAX = 4
E10_DF_PATH = Path("output/phase2_results/phase2_e10_scst_corpus_idf/corpus_df.pt")
if E10_DF_PATH.exists():
    blob = torch.load(E10_DF_PATH, map_location="cpu", weights_only=False)
    corpus_df = blob["corpus_df"]
    log_num_docs = blob["log_num_docs"]
    N_MAX = blob["N_MAX"]
    log(f"Loaded E10 corpus DF: {sum(len(c) for c in corpus_df):,} n-grams across n=1..{N_MAX}, "
        f"log_num_docs={log_num_docs:.3f}")
else:
    log(f"E10 corpus DF not found at {E10_DF_PATH}; rebuilding from training references")
    train_refs: list[str] = []
    for refs_for_img in shared_df[shared_df["split"] == "train"].groupby("image_id")["caption_clean"].apply(list):
        train_refs.extend(refs_for_img)
    corpus_df = [Counter() for _ in range(N_MAX)]
    for ref in train_refs:
        toks = str(ref).split()
        for n in range(1, N_MAX + 1):
            seen = set(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
            for ng in seen:
                corpus_df[n - 1][ng] += 1
    log_num_docs = math.log(max(len(train_refs), 1))
    log(f"Built corpus DF in-place: {sum(len(c) for c in corpus_df):,} n-grams, log_num_docs={log_num_docs:.3f}")

# %% [markdown]
# ## Beam search that returns the **top-K finished candidates**, not just argmax
#
# Same machinery as E11 but we keep K candidates rather than reducing to 1.

# %%
@torch.no_grad()
def topk_beam_candidates(image_id: int, k: int = K) -> list[str]:
    memories = []
    for m in members:
        feat = m["features"][m["image_to_idx"][int(image_id)]].float().unsqueeze(0).to(device)
        memories.append(m["model"].encode(feat))

    BEAM = max(k, 8)  # widen beam so finished candidates are diverse
    beams = [([START], 0.0, False)]
    finished = []
    for step in range(GEN_MAX):
        active = [(i, b) for i, b in enumerate(beams) if not b[2]]
        if not active:
            break
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
            for kk in range(BEAM):
                tid = int(topk_id[ai, kk].item())
                s = sc + float(topk_lp[ai, kk].item())
                done = (tid == END)
                new = (toks + [tid], s, done)
                if done:
                    finished.append(new)
                else:
                    cands.append(new)
        if not cands:
            break
        def sf(it):
            t, s, _ = it; return s / (max(len(t) - 1, 1) ** LP)
        cands.sort(key=sf, reverse=True)
        beams = cands[:BEAM]

    # Also let any still-active beams count as finished (force-end)
    finished.extend(beams)

    def sf(it):
        t, s, _ = it; return s / (max(len(t) - 1, 1) ** LP)
    finished.sort(key=sf, reverse=True)

    # De-duplicate by token sequence, keep top-k
    seen = set(); out = []
    for toks, s, _ in finished:
        words = []
        for tid in toks[1:]:
            if tid in (END, PAD):
                break
            words.append(idx2word[tid])
        cap = " ".join(words)
        if cap in seen:
            continue
        seen.add(cap); out.append(cap)
        if len(out) >= k:
            break
    return out


# %% [markdown]
# ## CIDEr-D between two single captions, using the corpus IDF
#
# This is the same formula as E10's `cidered_one`, just specialised to one
# reference. Using the corpus IDF (rather than uniform / no IDF) makes the
# MBR consensus penalty downweight frequent stopwords — without it the
# pairwise score is dominated by `a / the / with / and` and the MBR pick
# collapses toward the most generic candidate.

# %%
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


def pairwise_cider(pred_tokens, ref_tokens):
    """Sentence-level CIDEr-D between two sentences using the corpus IDF."""
    per_n = []
    for n in range(1, N_MAX + 1):
        p_vec = _tfidf_vec(pred_tokens, n)
        r_vec = _tfidf_vec(ref_tokens, n)
        if not p_vec or not r_vec:
            per_n.append(0.0); continue
        common = set(p_vec) & set(r_vec)
        num = sum(p_vec[ng] * r_vec[ng] for ng in common)
        pn = math.sqrt(sum(v * v for v in p_vec.values())) + EPS
        rn = math.sqrt(sum(v * v for v in r_vec.values())) + EPS
        cos = num / (pn * rn)
        delta = len(pred_tokens) - len(ref_tokens)
        penalty = math.exp(-(delta * delta) / (2 * SIGMA * SIGMA))
        per_n.append(cos * penalty)
    return 10.0 * (sum(per_n) / N_MAX)


def mbr_pick(candidates: list[str]) -> str:
    """Pick the candidate with max mean pairwise CIDEr against the others."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    toks_list = [c.split() for c in candidates]
    best_i = 0; best_score = -1.0
    for i in range(len(candidates)):
        score = 0.0
        for j in range(len(candidates)):
            if i == j:
                continue
            score += pairwise_cider(toks_list[i], toks_list[j])
        score /= (len(candidates) - 1)
        if score > best_score:
            best_score = score; best_i = i
    return candidates[best_i]


# %% [markdown]
# ## Evaluate val + test

# %%
def eval_split(split):
    sdf = shared_df[shared_df["split"] == split][["image_id", "file_name"]].drop_duplicates()
    rows, preds, refs_all = [], [], []
    log(f"MBR eval {split}: {len(sdf)} images, K={K} candidates each")
    import time as _time
    t0 = _time.time(); last_hb = t0
    for i, row in sdf.reset_index(drop=True).iterrows():
        cands = topk_beam_candidates(int(row["image_id"]), k=K)
        pick = mbr_pick(cands)
        preds.append(pick)
        r = shared_refs[str(int(row["image_id"]))]
        refs_all.append(r)
        rows.append({"image_id": int(row["image_id"]), "file_name": row["file_name"],
                     "prediction": pick, "candidates": cands, "references": r})
        if _time.time() - last_hb >= 20:
            log(f"  {split} {i+1}/{len(sdf)}  elapsed={_time.time()-t0:.0f}s")
            last_hb = _time.time()
    bm = corpus_bleu(preds, refs_all); bm["CIDEr"] = corpus_cider(preds, refs_all)
    return bm, pd.DataFrame(rows)


val_m, val_p = eval_split("val")
test_m, test_p = eval_split("test")
metrics = {
    "run_name": "phase2_e14_mbr",
    "decoding": "MBR(K=16) over E11 ensemble",
    **{f"val_{k}": v for k, v in val_m.items()},
    **{f"test_{k}": v for k, v in test_m.items()},
}
val_p.to_csv(OUT_ROOT / "val_predictions.csv", index=False)
test_p.to_csv(OUT_ROOT / "test_predictions.csv", index=False)
json.dump(metrics, open(OUT_ROOT / "metrics.json", "w"), indent=2)
log(f"DONE  val_BLEU-4={val_m['BLEU-4']:.4f}  test_BLEU-4={test_m['BLEU-4']:.4f}  test_CIDEr={test_m['CIDEr']:.4f}")
metrics
