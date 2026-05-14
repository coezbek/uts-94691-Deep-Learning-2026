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
# # Phase 2 — E7 (text-only): LLM paraphrase augmentation

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
# We have no images on this machine, so a VLM cannot be used. As a fallback we use a **text-only LLM (Qwen2.5-1.5B-Instruct)** to paraphrase each existing training caption into N new variants. This teaches paraphrastic invariance — it does not add new grounding but does reduce overfitting on small caption sets.
#
# If the model download fails (no internet), we **skip** this experiment gracefully and continue.

# %%
# Try to load Qwen; if unavailable just skip the experiment
import importlib, json, pandas as pd
from pathlib import Path
metrics_e7 = None
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map='cuda')
    print('Qwen loaded:', MODEL_ID)
    can_paraphrase = True
except Exception as e:
    print('Could not load Qwen — skipping E7. Reason:', e)
    can_paraphrase = False

if can_paraphrase:
    import numpy as np, re, time
    df = pd.read_csv('output/processed_captions.csv')
    train = df[df['split']=='train'].copy().reset_index(drop=True)
    # Sample 5000 captions to paraphrase (rather than all 26k) — keeps E7 runtime under control.
    SAMPLE_N = min(5000, len(train))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(train), size=SAMPLE_N, replace=False)
    sampled = train.iloc[sample_idx].reset_index(drop=True)
    cap_list = sampled['caption_clean'].tolist()
    BATCH = 32
    print(f'Paraphrasing {len(cap_list)} sampled captions...')
    t0 = time.time()
    SYS_PROMPT = 'You rewrite image captions. Output ONLY the rewritten caption, lowercase, no punctuation other than apostrophes, no numbering, on one line. Keep it under 20 words.'
    def paraphrase_batch(captions):
        prompts = []
        for c in captions:
            msg = [{'role':'system','content':SYS_PROMPT},
                   {'role':'user','content':f'Rewrite this caption keeping the same meaning: {c}'}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        inputs = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda')
        gen = llm.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)
        outs = tok.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        result = []
        for orig, o in zip(captions, outs):
            lines = [ln for ln in o.strip().splitlines() if ln.strip()]
            result.append((lines[0] if lines else orig).lower())
        return result
    new_caps = [None]*len(cap_list)
    for i in range(0, len(cap_list), BATCH):
        batch = cap_list[i:i+BATCH]
        try:
            outs = paraphrase_batch(batch)
        except Exception as ex:
            print(f'  batch {i//BATCH} failed: {ex}; reusing original captions')
            outs = batch
        for j, o in enumerate(outs):
            new_caps[i+j] = o
        if (i // BATCH) % 10 == 0:
            elapsed = time.time()-t0
            print(f'  batch {i//BATCH}/{len(cap_list)//BATCH}  elapsed={elapsed:.0f}s')
    print(f'Paraphrasing complete in {time.time()-t0:.0f}s')
    # Build augmented training rows
    new_rows = []
    for (_, row), p in zip(sampled.iterrows(), new_caps):
        if not p:
            continue
        r = row.copy()
        r['caption'] = p
        c = re.sub(r'[^a-z0-9\' ]+', ' ', p.lower()).strip()
        c = re.sub(r'\s+', ' ', c).strip()
        r['caption_clean'] = c
        r['caption_word_count'] = len(c.split())
        new_rows.append(r)
    new_df = pd.DataFrame(new_rows)
    new_df = new_df[new_df['caption_word_count']>0]
    print(f'Added {len(new_df)} paraphrased rows')
    aug = pd.concat([train, new_df], ignore_index=True)
    full_df = pd.concat([aug, df[df['split']!='train']], ignore_index=True)
    aug_csv = 'output/processed_captions_e7.csv'
    full_df.to_csv(aug_csv, index=False)
    print(f'Saved augmented CSV: {aug_csv}  total rows: {len(full_df):,}')
    del llm  # free GPU
    import gc; gc.collect(); torch.cuda.empty_cache()
    cfg = ExperimentConfig(
        run_name='phase2_e7_text_paraphrase',
        output_dir='output/phase2_results',
        captions_csv=aug_csv,
        epochs=12, batch_size=128, lr=1e-4,
        decoding='beam', beam_width=5, length_penalty=0.7,
    )
    metrics_e7 = run_experiment(cfg)
metrics_e7
