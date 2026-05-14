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
# # E11d — Multi-seed self-ensemble of E15 (uniform avg over 3 seeds)
#
# E15 single (seed=42)   → test BLEU-4 0.396 / CIDEr 1.526
# E15-s2 single (seed=123) → test BLEU-4 0.403 / CIDEr 1.549
# E15-s3 single (seed=456) → test BLEU-4 ?
#
# Identical recipe, identical encoder, identical hyperparameters — *only the
# random seed changes*. The seed effect was much larger than expected
# (+0.023 CIDEr from one seed swap), so a 3-seed uniform-average ensemble
# is the most principled way to extract that variance:
#
# * Members have the same architecture and decoder hyperparameters, so
#   uniform weights are unambiguously right (no quality-weighting needed).
# * Disagreement comes purely from where in the parameter space each model
#   landed, which is exactly the noise SCST / data augmentation tries to
#   exploit — but multi-seed ensembling does it for free at inference.
#
# Predicted gain over the best single seed: +0.003-0.010 BLEU-4 typical
# for caption ensembles with this much per-seed variance.

# %%
import json, math
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd

import sys
sys.path.insert(0, str(Path("output").resolve()))
import importlib
import exp_runner
importlib.reload(exp_runner)
from exp_runner import (
    ExperimentConfig, CaptioningModel, load_data,
    corpus_bleu, corpus_cider, make_progress_logger,
)

OUT_ROOT = Path("output/phase2_results/phase2_e11d_multiseed_e15")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
log = make_progress_logger(OUT_ROOT)
device = torch.device("cuda")


# Three seeds of the identical E15 recipe (SigLIP2 + small reg decoder)
MEMBERS = [
    {"name": "E15",    "ckpt": "output/phase2_results/phase2_e15_siglip2_small/best.pt", "seed": 42},
    {"name": "E15-s2", "ckpt": "output/phase2_results/phase2_e15s2_siglip2_small_seed2/best.pt", "seed": 123},
    {"name": "E15-s3", "ckpt": "output/phase2_results/phase2_e15s3_siglip2_small_seed3/best.pt", "seed": 456},
]

# Shared by all members (identical encoder + decoder hyperparameters)
SHARED = dict(
    feature_cache="output/siglip2_b16_features.pt",
    encoder_feat_dim=768, num_spatial_tokens=256,
    embed_dim=384, num_heads=6, num_decoder_layers=2,
    ffn_dim=1536, dropout=0.3,
)


def build_model(m):
    cfg = ExperimentConfig(
        run_name="multiseed_" + m["name"],
        output_dir="output/phase2_results",
        encoder_kind="efficientnet_b0_cached",
        **SHARED,
    )
    df, vocab, references, features, image_to_idx = load_data(cfg)
    pad_idx = vocab["pad_idx"]; vocab_size = len(vocab["idx2word"])
    model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)
    sd = torch.load(m["ckpt"], map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, features, image_to_idx, vocab, df, references


log("Loading 3 E15 seeds ...")
members = []
shared = None
for m in MEMBERS:
    model, features, image_to_idx, vocab, df, refs = build_model(m)
    members.append({**m, "model": model, "features": features, "image_to_idx": image_to_idx})
    shared = (vocab, df, refs)
    log(f"  loaded {m['name']} (seed={m['seed']})")

vocab, shared_df, shared_refs = shared
word2idx = vocab["word2idx"]; idx2word = vocab["idx2word"]
PAD = word2idx["<pad>"]; START = word2idx["<start>"]; END = word2idx["<end>"]; UNK = word2idx["<unk>"]

BEAM = 5
LP = 0.7
GEN_MAX = 20
GEN_MIN = 4


@torch.no_grad()
def multiseed_ensemble_beam(image_id: int) -> str:
    # All three members share the same feature cache, so we look up once.
    feat = members[0]["features"][members[0]["image_to_idx"][int(image_id)]].float().unsqueeze(0).to(device)
    memories = [m["model"].encode(feat) for m in members]

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
        p = multiseed_ensemble_beam(int(row["image_id"]))
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
    "run_name": "phase2_e11d_multiseed_e15",
    "ensembled_runs": [m["name"] for m in MEMBERS],
    "seeds": [m["seed"] for m in MEMBERS],
    "decoding": "beam-uniform-logit-avg", "beam_width": BEAM, "length_penalty": LP,
    **{f"val_{k}": v for k, v in val_m.items()},
    **{f"test_{k}": v for k, v in test_m.items()},
}
val_p.to_csv(OUT_ROOT / "val_predictions.csv", index=False)
test_p.to_csv(OUT_ROOT / "test_predictions.csv", index=False)
json.dump(metrics, open(OUT_ROOT / "metrics.json", "w"), indent=2)
log(f"DONE  val_BLEU-4={val_m['BLEU-4']:.4f}  test_BLEU-4={test_m['BLEU-4']:.4f}  test_CIDEr={test_m['CIDEr']:.4f}")
metrics
