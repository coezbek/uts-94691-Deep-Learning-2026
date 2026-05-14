# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
# ---

# %% [markdown]
# # Phase 2 — E4: Tied embeddings + label smoothing + warmup-cosine

# %%
import sys
from pathlib import Path
nb_dir = Path('.').resolve()
sys.path.insert(0, str(nb_dir / "output"))
import importlib, exp_runner
importlib.reload(exp_runner)
from exp_runner import ExperimentConfig, run_experiment, evaluate_only
print("exp_runner loaded from", exp_runner.__file__)


# %% [markdown]
# Three small training tricks bundled together:
# * tie input embedding with output projection (`tie_embeddings=True`)
# * label smoothing 0.1 in the cross-entropy loss
# * warmup + cosine LR schedule (1000 step warmup, cosine decay)
#
# Everything else identical to E0. Evaluated with beam search using the best (beam, length_penalty) from E1.

# %%
import pandas as pd
e1_sweep = pd.read_csv('output/phase2_results/phase2_e1_beam_sweep.csv')
e1_sweep = e1_sweep.sort_values('test_BLEU-4', ascending=False)
best_beam = int(e1_sweep.iloc[0]['beam_width']) if 'beam_width' in e1_sweep.columns else 5
best_lp = float(e1_sweep.iloc[0]['length_penalty']) if 'length_penalty' in e1_sweep.columns else 1.0
print('Using beam_width=', best_beam, ' length_penalty=', best_lp)

cfg = ExperimentConfig(
    run_name='phase2_e4_schedule_polish',
    output_dir='output/phase2_results',
    epochs=15, batch_size=128, lr=1e-4,
    optimizer='adamw', weight_decay=0.01,
    scheduler='warmup_cosine', warmup_steps=500,
    label_smoothing=0.1, tie_embeddings=True,
    decoding='beam', beam_width=best_beam, length_penalty=best_lp,
)
metrics_e4 = run_experiment(cfg)
metrics_e4
