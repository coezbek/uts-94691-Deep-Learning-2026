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
# # Phase 2 — E0: Baseline with Raina's prep (greedy)

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
# Re-run of the Phase 1 architecture (EfficientNet B0 + Transformer with 3 layers, 8 heads, d_model=512) using **Raina's data prep** (no `<num>` token). Greedy decoding. This is the new baseline for all Phase 2 deltas.

# %%
cfg = ExperimentConfig(
    run_name='phase2_e0_baseline_raina_greedy',
    output_dir='output/phase2_results',
    epochs=15,
    batch_size=128,
    lr=1e-4,
    decoding='greedy',
)
metrics_e0 = run_experiment(cfg)
metrics_e0
