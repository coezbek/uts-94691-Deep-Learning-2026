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
# # E15-s2 narrow-beam refinement: greedy and w=2
#
# The wider beam sweep showed w=3 beats w=5 beats w=10 on E15-s2. To pin
# down the optimum, also try the two narrower options:
#
#   * greedy (= effective w=1)
#   * beam w=2

# %%
import json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path("output").resolve()))
import importlib
import exp_runner
importlib.reload(exp_runner)
from exp_runner import (
    ExperimentConfig, CaptioningModel, load_data,
    CachedFeatureImageDataset, image_collate,
    evaluate_split, make_progress_logger,
)

OUT_ROOT = Path("output/phase2_results/phase2_e15s2_narrow_sweep")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
log = make_progress_logger(OUT_ROOT)
device = torch.device("cuda")

CKPT = "output/phase2_results/phase2_e15s2_siglip2_small_seed2/best.pt"

SHARED_CFG_KW = dict(
    output_dir="output/phase2_results",
    feature_cache="output/siglip2_b16_features.pt",
    encoder_kind="efficientnet_b0_cached",
    encoder_feat_dim=768, num_spatial_tokens=256,
    embed_dim=384, num_heads=6, num_decoder_layers=2,
    ffn_dim=1536, dropout=0.3,
    optimizer="adamw", weight_decay=0.02, label_smoothing=0.15,
    batch_size=128,
)


SETTINGS = [
    {"decoding": "greedy", "beam_width": 1, "length_penalty": 0.7},
    {"decoding": "beam",   "beam_width": 2, "length_penalty": 0.7},
]


all_results = []
for s in SETTINGS:
    tag = f"{s['decoding']}_w{s['beam_width']}_lp{s['length_penalty']:g}"
    log(f"=== {tag} ===")
    cfg = ExperimentConfig(
        run_name=f"e15s2_narrow_{tag}",
        decoding=s["decoding"], beam_width=s["beam_width"], length_penalty=s["length_penalty"],
        **SHARED_CFG_KW,
    )
    df, vocab, references, features, image_to_idx = load_data(cfg)
    pad_idx = vocab["pad_idx"]; vocab_size = len(vocab["idx2word"])
    model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)
    sd = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()

    val_df = df[df["split"] == "val"][["image_id", "file_name"]].drop_duplicates()
    test_df = df[df["split"] == "test"][["image_id", "file_name"]].drop_duplicates()
    val_loader = DataLoader(CachedFeatureImageDataset(val_df, references, features, image_to_idx),
                            batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate)
    test_loader = DataLoader(CachedFeatureImageDataset(test_df, references, features, image_to_idx),
                             batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate)

    val_m, val_p = evaluate_split(model, val_loader, vocab, cfg, split="val")
    test_m, test_p = evaluate_split(model, test_loader, vocab, cfg, split="test")
    log(f"  {tag}: val_BLEU-4={val_m['BLEU-4']:.4f}  test_BLEU-4={test_m['BLEU-4']:.4f}  test_CIDEr={test_m['CIDEr']:.4f}")
    all_results.append({
        "tag": tag, "decoding": s["decoding"], "beam_width": s["beam_width"], "length_penalty": s["length_penalty"],
        "val_BLEU-4": val_m["BLEU-4"], "val_CIDEr": val_m["CIDEr"],
        "test_BLEU-4": test_m["BLEU-4"], "test_CIDEr": test_m["CIDEr"],
    })
    del model; torch.cuda.empty_cache()


df_res = pd.DataFrame(all_results)
df_res.to_csv(OUT_ROOT / "sweep_results.csv", index=False)
json.dump(all_results, open(OUT_ROOT / "metrics.json", "w"), indent=2)
log("DONE narrow sweep:")
log(df_res.to_string(index=False))
