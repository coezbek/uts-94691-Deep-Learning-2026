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
# # E15 — SigLIP2 features + E12's small-regularised decoder recipe
#
# We've found two independent wins in Phase 3:
#
# * **E13** showed that SigLIP2 ViT-B/16-256 features beat CLIP ViT-B/16
#   (test BLEU-4 0.391 / CIDEr 1.516 vs 0.389 / 1.474).
# * **E12** showed that a smaller decoder with stronger regularisation
#   (2L × d=384, dropout 0.3, label smoothing 0.15, caption-input word
#   dropout 0.10, AdamW WD 0.02, early-stop on val BLEU-4) reaches E07-
#   class performance (0.358 / 1.402) at one-third the parameters and is
#   useful in the ensemble as a "recipe diversity" axis.
#
# This experiment combines the two: SigLIP2 features + E12's decoder
# recipe. Two hypotheses:
#
# 1. The smaller decoder might generalise better on SigLIP2 features and
#    set a new single-model best (E13 was 0.391/1.516; ceiling unclear).
# 2. Even if it doesn't beat E13 single-model, it's an ensemble member
#    that disagrees with E13 along the decoder axis but agrees on the
#    encoder, giving the E11 ensemble a 5th genuinely-different bias.
#
# Same cached SigLIP2 features as E13 (no re-extraction needed).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path("output").resolve()))
import importlib
import exp_runner
importlib.reload(exp_runner)
from exp_runner import ExperimentConfig, run_experiment

# %%
cfg = ExperimentConfig(
    run_name="phase2_e15_siglip2_small",
    output_dir="output/phase2_results",
    feature_cache="output/siglip2_b16_features.pt",
    encoder_kind="efficientnet_b0_cached",
    encoder_feat_dim=768,
    num_spatial_tokens=256,
    # E12-style decoder
    embed_dim=384,
    num_heads=6,
    num_decoder_layers=2,
    ffn_dim=1536,
    dropout=0.3,
    # E12-style training
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
metrics_e15 = run_experiment(cfg)
metrics_e15
