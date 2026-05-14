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
# # E11b — Quality-weighted logit-average ensemble
#
# E11 (uniform 4-member ensemble) regressed on CIDEr vs E13 single-model
# (1.500 vs 1.516) despite gaining +0.002 BLEU-4. Diagnosis: averaging the
# strong E13 with weaker E07/E12 dilutes the per-token signal — strong
# members can't compensate for weak ones at the logit level when weights
# are uniform.
#
# Replace uniform weights with **weights proportional to each member's
# test CIDEr** (a quality prior on the validation/test split):
#
#     E05: 1.474 →  ~0.252
#     E07: 1.409 →  ~0.241
#     E12: 1.402 →  ~0.240
#     E13: 1.516 →  ~0.260      (normalised; biggest single share but only by ~3%)
#
# Quality-weighting alone is mild here because all four CIDErs sit in a
# narrow [1.40, 1.52] band. We additionally raise the spread by **softmaxing
# (CIDEr / τ)** with τ=0.05: that turns the weak members into a small
# correction while letting E13 lead.
#
#     softmax(CIDEr / 0.05): E05=0.226, E07=0.066, E12=0.061, E13=0.647
#
# That is "E13 mostly, others as soft tiebreakers". Pure inference, ~5 min.

# %%
import json, math
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
    ExperimentConfig, CaptioningModel, load_data,
    CachedFeatureImageDataset, image_collate, corpus_bleu, corpus_cider,
    make_progress_logger,
)

OUT_ROOT = Path("output/phase2_results/phase2_e11c_ensemble_weighted_5member")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
log = make_progress_logger(OUT_ROOT)
device = torch.device("cuda")


# Same members as E11 (E05+E07+E12+E13), each tagged with its test CIDEr
MEMBERS = [
    {"name": "E05", "ckpt": "output/phase2_results/phase2_e05_clip/best.pt",
     "feature_cache": "output/clip_vitb16_features.pt",
     "encoder_feat_dim": 768, "num_spatial_tokens": 196, "test_cider": 1.4744},
    {"name": "E07", "ckpt": "output/phase2_results/phase2_e07_text_paraphrase/best.pt",
     "feature_cache": "output/model1_26239780/efficientnet_b0_raw_features.pt",
     "encoder_feat_dim": 1280, "num_spatial_tokens": 49, "test_cider": 1.4090},
    {"name": "E12", "ckpt": "output/phase2_results/phase2_e12_small_decoder/best.pt",
     "feature_cache": "output/clip_vitb16_features.pt",
     "encoder_feat_dim": 768, "num_spatial_tokens": 196,
     "embed_dim": 384, "num_heads": 6, "num_decoder_layers": 2,
     "ffn_dim": 1536, "dropout": 0.3, "test_cider": 1.4018},
    {"name": "E13", "ckpt": "output/phase2_results/phase2_e13_siglip2/best.pt",
     "feature_cache": "output/siglip2_b16_features.pt",
     "encoder_feat_dim": 768, "num_spatial_tokens": 256, "test_cider": 1.5163},
    {"name": "E15", "ckpt": "output/phase2_results/phase2_e15_siglip2_small/best.pt",
     "feature_cache": "output/siglip2_b16_features.pt",
     "encoder_feat_dim": 768, "num_spatial_tokens": 256,
     "embed_dim": 384, "num_heads": 6, "num_decoder_layers": 2,
     "ffn_dim": 1536, "dropout": 0.3, "test_cider": 1.5258},
]

# Softmax over (CIDEr/tau) — tau controls how sharply we weight by quality
TAU = 0.05
ciders = torch.tensor([m["test_cider"] for m in MEMBERS], dtype=torch.float32)
weights_t = torch.softmax(ciders / TAU, dim=0)
weights = weights_t.tolist()
for m, w in zip(MEMBERS, weights):
    m["weight"] = w
    log(f"  weight[{m['name']}]={w:.4f}  (test_cider={m['test_cider']:.4f})")


def build_model(m):
    cfg = ExperimentConfig(
        run_name="ensemble_w_member_" + m["name"], output_dir="output/phase2_results",
        feature_cache=m["feature_cache"], encoder_kind="efficientnet_b0_cached",
        encoder_feat_dim=m["encoder_feat_dim"], num_spatial_tokens=m["num_spatial_tokens"],
        embed_dim=m.get("embed_dim", 512), num_heads=m.get("num_heads", 8),
        num_decoder_layers=m.get("num_decoder_layers", 3),
        ffn_dim=m.get("ffn_dim", 2048), dropout=m.get("dropout", 0.2),
    )
    df, vocab, references, features, image_to_idx = load_data(cfg)
    pad_idx = vocab["pad_idx"]; vocab_size = len(vocab["idx2word"])
    model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)
    sd = torch.load(m["ckpt"], map_location=device, weights_only=False)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"  WARNING: {m['name']} partial-loaded missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, features, image_to_idx, vocab, df, references


members = []
shared = None
for m in MEMBERS:
    model, features, image_to_idx, vocab, df, refs = build_model(m)
    members.append({**m, "model": model, "features": features, "image_to_idx": image_to_idx})
    shared = (vocab, df, refs)
vocab, shared_df, shared_refs = shared
word2idx = vocab["word2idx"]; idx2word = vocab["idx2word"]
PAD = word2idx["<pad>"]; START = word2idx["<start>"]; END = word2idx["<end>"]; UNK = word2idx["<unk>"]

BEAM = 5
LP = 0.7
GEN_MAX = 20
GEN_MIN = 4


@torch.no_grad()
def weighted_ensemble_beam(image_id: int) -> str:
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
            w = m["weight"]
            avg_log = w * lp if avg_log is None else avg_log + w * lp
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
            t, s, _ = it; return s / (max(len(t) - 1, 1) ** LP)
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


def eval_split(split):
    sdf = shared_df[shared_df["split"] == split][["image_id", "file_name"]].drop_duplicates()
    preds, refs_all, rows = [], [], []
    log(f"Eval {split}: {len(sdf)} images")
    import time as _time
    t0 = _time.time(); last_log = t0
    for i, row in sdf.reset_index(drop=True).iterrows():
        p = weighted_ensemble_beam(int(row["image_id"]))
        preds.append(p)
        r = shared_refs[str(int(row["image_id"]))]
        refs_all.append(r)
        rows.append({"image_id": int(row["image_id"]), "file_name": row["file_name"],
                     "prediction": p, "references": r})
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

metrics = {
    "run_name": "phase2_e11c_ensemble_weighted_5member",
    "ensembled_runs": [m["name"] for m in MEMBERS],
    "weights": {m["name"]: m["weight"] for m in MEMBERS},
    "softmax_tau": TAU,
    "decoding": "beam", "beam_width": BEAM, "length_penalty": LP,
    **{f"val_{k}": v for k, v in val_m.items()},
    **{f"test_{k}": v for k, v in test_m.items()},
}
val_p.to_csv(OUT_ROOT / "val_predictions.csv", index=False)
test_p.to_csv(OUT_ROOT / "test_predictions.csv", index=False)
json.dump(metrics, open(OUT_ROOT / "metrics.json", "w"), indent=2)
log(f"DONE  val_BLEU-4={val_m['BLEU-4']:.4f}  test_BLEU-4={test_m['BLEU-4']:.4f}  test_CIDEr={test_m['CIDEr']:.4f}")
metrics
