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
# # E15-s2 — Second seed of E15 (same recipe, different seed)
#
# E15 is the Phase 3 best (SigLIP2 + small regularised decoder). All ensemble
# variants we tried (uniform E11, weighted E11b/E11c) underperform E15 alone.
# One untested ensemble direction: **multi-seed self-ensemble** — train the
# same recipe with different seeds, average their logits. With members
# that share recipe and encoder but differ only in random init / data order,
# the *only* axis of disagreement is sampling noise. Sometimes this is
# enough to add ~+0.005 BLEU-4 / +0.01 CIDEr.

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
    run_name="phase2_e15s2_siglip2_small_seed2",
    output_dir="output/phase2_results",
    feature_cache="output/siglip2_b16_features.pt",
    encoder_kind="efficientnet_b0_cached",
    encoder_feat_dim=768,
    num_spatial_tokens=256,
    embed_dim=384, num_heads=6, num_decoder_layers=2,
    ffn_dim=1536, dropout=0.3,
    epochs=20, batch_size=128, lr=1e-4, weight_decay=0.02,
    optimizer="adamw", label_smoothing=0.15,
    early_stop_patience=5, grad_clip=1.0,
    decoding="beam", beam_width=5, length_penalty=0.7,
    seed=123,                                   # <— only thing changed
)
metrics_e15s2 = run_experiment(cfg)
metrics_e15s2
