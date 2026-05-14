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
# # Phase 2 — E1: Beam search inference (no retraining)

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
# Same checkpoint as E0. We sweep beam width and length penalty and report the best. No retraining.

# %%
from pathlib import Path
import pandas as pd
import json
ckpt = 'output/phase2_results/phase2_e0_baseline_raina_greedy/best.pt'
assert Path(ckpt).exists(), f'Missing {ckpt} - run E0 first'
sweep = []
for bw in [3, 5]:
    for lp in [0.7, 1.0]:
        cfg = ExperimentConfig(
            run_name=f'phase2_e1_beam_w{bw}_lp{lp}',
            output_dir='output/phase2_results',
            decoding='beam', beam_width=bw, length_penalty=lp,
        )
        m = evaluate_only(cfg, checkpoint_path=ckpt)
        sweep.append(m)
df = pd.DataFrame(sweep)
df = df.sort_values('test_BLEU-4', ascending=False)
print(df[['run_name','val_BLEU-4','test_BLEU-4','test_CIDEr']])
df.to_csv('output/phase2_results/phase2_e1_beam_sweep.csv', index=False)
best = df.iloc[0]
print('BEST:', best['run_name'])
metrics_e1 = best.to_dict()
metrics_e1
