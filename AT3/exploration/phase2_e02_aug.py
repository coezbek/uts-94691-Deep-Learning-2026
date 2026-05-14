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
# # Phase 2 — E2: Re-enable train-time augmentation
#
# Train on raw images with random crop + horizontal flip (no feature cache at
# train time). The EfficientNet trunk stays frozen — only the projection head
# and the Transformer decoder train. Eval uses deterministic center crop and
# beam search with the best decoding config from E1 (w=5, lp=0.7).
#
# The point of this experiment is to check whether the augmentation we had to
# drop when we adopted the feature cache buys us anything back. With ~5,400
# training images and only one cleaned caption per (image, row), random
# crops/flips at train time act as a cheap regulariser.

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
    run_name="phase2_e2_aug",
    output_dir="output/phase2_results",
    use_cached_features=False,
    encoder_kind="efficientnet_b0_raw",
    freeze_encoder=True,
    augment=True,
    epochs=12,
    batch_size=128,
    lr=1e-4,
    decoding="beam",
    beam_width=5,
    length_penalty=0.7,
)
cfg

# %%
metrics_e2 = run_experiment(cfg)
metrics_e2
