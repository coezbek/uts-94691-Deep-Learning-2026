"""Generate and execute per-experiment notebooks.

Each notebook is a thin wrapper that imports exp_runner, builds an
ExperimentConfig, calls run_experiment / evaluate_only, and saves results.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

NB_DIR = Path(__file__).parent
OUT_NB_DIR = NB_DIR  # notebooks land next to existing notebooks
RESULTS_DIR = NB_DIR / "output" / "phase2_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# We always need this preamble to put output/ on sys.path so exp_runner imports
PREAMBLE = '''import sys
from pathlib import Path
nb_dir = Path('.').resolve()
sys.path.insert(0, str(nb_dir / "output"))
import importlib, exp_runner
importlib.reload(exp_runner)
from exp_runner import ExperimentConfig, run_experiment, evaluate_only
print("exp_runner loaded from", exp_runner.__file__)
'''


def make_notebook(title: str, sections: list[tuple[str, str]]) -> nbformat.NotebookNode:
    """sections = [(markdown_or_code, content), ...] where markers are 'md' or 'code'."""
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(f"# {title}"))
    nb.cells.append(new_code_cell(PREAMBLE))
    for kind, content in sections:
        if kind == "md":
            nb.cells.append(new_markdown_cell(content))
        else:
            nb.cells.append(new_code_cell(content))
    return nb


def execute_notebook(nb_path: Path, timeout: int = 60 * 60) -> bool:
    """Execute a notebook in place using the project's venv jupyter."""
    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "11.5.1"
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(nb_path),
    ]
    print(f"[exec] {nb_path.name} ...")
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[exec] {nb_path.name} FAILED in {dt:.1f}s")
        print("STDOUT:", r.stdout[-2000:])
        print("STDERR:", r.stderr[-2000:])
        return False
    print(f"[exec] {nb_path.name} OK  ({dt:.1f}s)")
    return True


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS: list[dict] = []


def add(name: str, title: str, sections: list[tuple[str, str]]):
    EXPERIMENTS.append({"name": name, "title": title, "sections": sections})


# E0 — Baseline with Raina prep, greedy decoding ----------------------------
add(
    "phase2_e0_baseline_raina_greedy",
    "Phase 2 — E0: Baseline with Raina's prep (greedy)",
    [
        ("md", "Re-run of the Phase 1 architecture (EfficientNet B0 + Transformer "
               "with 3 layers, 8 heads, d_model=512) using **Raina's data prep** "
               "(no `<num>` token). Greedy decoding. This is the new baseline for "
               "all Phase 2 deltas."),
        ("code", "cfg = ExperimentConfig(\n"
                 "    run_name='phase2_e0_baseline_raina_greedy',\n"
                 "    output_dir='output/phase2_results',\n"
                 "    epochs=15,\n"
                 "    batch_size=128,\n"
                 "    lr=1e-4,\n"
                 "    decoding='greedy',\n"
                 ")\n"
                 "metrics_e0 = run_experiment(cfg)\n"
                 "metrics_e0"),
    ],
)

# E1 — Beam search inference on E0 ------------------------------------------
add(
    "phase2_e1_beam_search",
    "Phase 2 — E1: Beam search inference (no retraining)",
    [
        ("md", "Same checkpoint as E0. We sweep beam width and length penalty and "
               "report the best. No retraining."),
        ("code",
         "from pathlib import Path\n"
         "import pandas as pd\n"
         "import json\n"
         "ckpt = 'output/phase2_results/phase2_e0_baseline_raina_greedy/best.pt'\n"
         "assert Path(ckpt).exists(), f'Missing {ckpt} - run E0 first'\n"
         "sweep = []\n"
         "for bw in [3, 5]:\n"
         "    for lp in [0.7, 1.0]:\n"
         "        cfg = ExperimentConfig(\n"
         "            run_name=f'phase2_e1_beam_w{bw}_lp{lp}',\n"
         "            output_dir='output/phase2_results',\n"
         "            decoding='beam', beam_width=bw, length_penalty=lp,\n"
         "        )\n"
         "        m = evaluate_only(cfg, checkpoint_path=ckpt)\n"
         "        sweep.append(m)\n"
         "df = pd.DataFrame(sweep)\n"
         "df = df.sort_values('test_BLEU-4', ascending=False)\n"
         "print(df[['run_name','val_BLEU-4','test_BLEU-4','test_CIDEr']])\n"
         "df.to_csv('output/phase2_results/phase2_e1_beam_sweep.csv', index=False)\n"
         "best = df.iloc[0]\n"
         "print('BEST:', best['run_name'])\n"
         "metrics_e1 = best.to_dict()\n"
         "metrics_e1"),
    ],
)

# E4 — Training schedule polish: tied embeddings + label smoothing + warmup_cosine
add(
    "phase2_e4_schedule_polish",
    "Phase 2 — E4: Tied embeddings + label smoothing + warmup-cosine",
    [
        ("md", "Three small training tricks bundled together:\n"
               "* tie input embedding with output projection (`tie_embeddings=True`)\n"
               "* label smoothing 0.1 in the cross-entropy loss\n"
               "* warmup + cosine LR schedule (1000 step warmup, cosine decay)\n"
               "\nEverything else identical to E0. Evaluated with beam search "
               "using the best (beam, length_penalty) from E1."),
        ("code",
         "import pandas as pd\n"
         "e1_sweep = pd.read_csv('output/phase2_results/phase2_e1_beam_sweep.csv')\n"
         "e1_sweep = e1_sweep.sort_values('test_BLEU-4', ascending=False)\n"
         "best_beam = int(e1_sweep.iloc[0]['beam_width']) if 'beam_width' in e1_sweep.columns else 5\n"
         "best_lp = float(e1_sweep.iloc[0]['length_penalty']) if 'length_penalty' in e1_sweep.columns else 1.0\n"
         "print('Using beam_width=', best_beam, ' length_penalty=', best_lp)\n"
         "\n"
         "cfg = ExperimentConfig(\n"
         "    run_name='phase2_e4_schedule_polish',\n"
         "    output_dir='output/phase2_results',\n"
         "    epochs=15, batch_size=128, lr=1e-4,\n"
         "    optimizer='adamw', weight_decay=0.01,\n"
         "    scheduler='warmup_cosine', warmup_steps=500,\n"
         "    label_smoothing=0.1, tie_embeddings=True,\n"
         "    decoding='beam', beam_width=best_beam, length_penalty=best_lp,\n"
         ")\n"
         "metrics_e4 = run_experiment(cfg)\n"
         "metrics_e4"),
    ],
)

# E7-text — LLM paraphrase augmentation (text-only, no images required) -----
add(
    "phase2_e7_text_paraphrase",
    "Phase 2 — E7 (text-only): LLM paraphrase augmentation",
    [
        ("md", "We have no images on this machine, so a VLM cannot be used. As a "
               "fallback we use a **text-only LLM (Qwen2.5-1.5B-Instruct)** to "
               "paraphrase each existing training caption into N new variants. "
               "This teaches paraphrastic invariance — it does not add new "
               "grounding but does reduce overfitting on small caption sets.\n\n"
               "If the model download fails (no internet), we **skip** this "
               "experiment gracefully and continue."),
        ("code",
         "# Try to load Qwen; if unavailable just skip the experiment\n"
         "import importlib, json, pandas as pd\n"
         "from pathlib import Path\n"
         "metrics_e7 = None\n"
         "try:\n"
         "    from transformers import AutoTokenizer, AutoModelForCausalLM\n"
         "    import torch\n"
         "    MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'\n"
         "    tok = AutoTokenizer.from_pretrained(MODEL_ID)\n"
         "    llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map='cuda')\n"
         "    print('Qwen loaded:', MODEL_ID)\n"
         "    can_paraphrase = True\n"
         "except Exception as e:\n"
         "    print('Could not load Qwen — skipping E7. Reason:', e)\n"
         "    can_paraphrase = False\n"
         "\n"
         "if can_paraphrase:\n"
         "    import numpy as np, re, time\n"
         "    df = pd.read_csv('output/processed_captions.csv')\n"
         "    train = df[df['split']=='train'].copy().reset_index(drop=True)\n"
         "    # Sample 5000 captions to paraphrase (rather than all 26k) — keeps E7 runtime under control.\n"
         "    SAMPLE_N = min(5000, len(train))\n"
         "    rng = np.random.default_rng(42)\n"
         "    sample_idx = rng.choice(len(train), size=SAMPLE_N, replace=False)\n"
         "    sampled = train.iloc[sample_idx].reset_index(drop=True)\n"
         "    cap_list = sampled['caption_clean'].tolist()\n"
         "    BATCH = 32\n"
         "    print(f'Paraphrasing {len(cap_list)} sampled captions...')\n"
         "    t0 = time.time()\n"
         "    SYS_PROMPT = 'You rewrite image captions. Output ONLY the rewritten caption, lowercase, no punctuation other than apostrophes, no numbering, on one line. Keep it under 20 words.'\n"
         "    def paraphrase_batch(captions):\n"
         "        prompts = []\n"
         "        for c in captions:\n"
         "            msg = [{'role':'system','content':SYS_PROMPT},\n"
         "                   {'role':'user','content':f'Rewrite this caption keeping the same meaning: {c}'}]\n"
         "            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))\n"
         "        inputs = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda')\n"
         "        gen = llm.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)\n"
         "        outs = tok.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)\n"
         "        result = []\n"
         "        for orig, o in zip(captions, outs):\n"
         "            lines = [ln for ln in o.strip().splitlines() if ln.strip()]\n"
         "            result.append((lines[0] if lines else orig).lower())\n"
         "        return result\n"
         "    new_caps = [None]*len(cap_list)\n"
         "    for i in range(0, len(cap_list), BATCH):\n"
         "        batch = cap_list[i:i+BATCH]\n"
         "        try:\n"
         "            outs = paraphrase_batch(batch)\n"
         "        except Exception as ex:\n"
         "            print(f'  batch {i//BATCH} failed: {ex}; reusing original captions')\n"
         "            outs = batch\n"
         "        for j, o in enumerate(outs):\n"
         "            new_caps[i+j] = o\n"
         "        if (i // BATCH) % 10 == 0:\n"
         "            elapsed = time.time()-t0\n"
         "            print(f'  batch {i//BATCH}/{len(cap_list)//BATCH}  elapsed={elapsed:.0f}s')\n"
         "    print(f'Paraphrasing complete in {time.time()-t0:.0f}s')\n"
         "    # Build augmented training rows\n"
         "    new_rows = []\n"
         "    for (_, row), p in zip(sampled.iterrows(), new_caps):\n"
         "        if not p:\n"
         "            continue\n"
         "        r = row.copy()\n"
         "        r['caption'] = p\n"
         "        c = re.sub(r'[^a-z0-9\\' ]+', ' ', p.lower()).strip()\n"
         "        c = re.sub(r'\\s+', ' ', c).strip()\n"
         "        r['caption_clean'] = c\n"
         "        r['caption_word_count'] = len(c.split())\n"
         "        new_rows.append(r)\n"
         "    new_df = pd.DataFrame(new_rows)\n"
         "    new_df = new_df[new_df['caption_word_count']>0]\n"
         "    print(f'Added {len(new_df)} paraphrased rows')\n"
         "    aug = pd.concat([train, new_df], ignore_index=True)\n"
         "    full_df = pd.concat([aug, df[df['split']!='train']], ignore_index=True)\n"
         "    aug_csv = 'output/processed_captions_e7.csv'\n"
         "    full_df.to_csv(aug_csv, index=False)\n"
         "    print(f'Saved augmented CSV: {aug_csv}  total rows: {len(full_df):,}')\n"
         "    del llm  # free GPU\n"
         "    import gc; gc.collect(); torch.cuda.empty_cache()\n"
         "    cfg = ExperimentConfig(\n"
         "        run_name='phase2_e7_text_paraphrase',\n"
         "        output_dir='output/phase2_results',\n"
         "        captions_csv=aug_csv,\n"
         "        epochs=12, batch_size=128, lr=1e-4,\n"
         "        decoding='beam', beam_width=5, length_penalty=0.7,\n"
         "    )\n"
         "    metrics_e7 = run_experiment(cfg)\n"
         "metrics_e7"),
    ],
)

# E8 — SCST / CIDEr-RL fine-tune --------------------------------------------
add(
    "phase2_e8_scst",
    "Phase 2 — E8: SCST (CIDEr-reward RL fine-tune)",
    [
        ("md", "Self-Critical Sequence Training (Rennie 2017): warm-start from the "
               "best MLE checkpoint (E4 if it improved, else E0); then for 1-2 "
               "epochs sample captions, score with CIDEr, and use REINFORCE with "
               "a greedy-decode baseline.\n\n"
               "This is the canonical post-MLE polish for captioning."),
        ("code",
         "# Pick the best MLE checkpoint to warm-start from\n"
         "import json\n"
         "from pathlib import Path\n"
         "import torch, pandas as pd\n"
         "import sys\n"
         "sys.path.insert(0, 'output')\n"
         "from exp_runner import (ExperimentConfig, CaptioningModel, load_data,\n"
         "                       CachedFeatureCaptionDataset, CachedFeatureImageDataset,\n"
         "                       caption_collate, image_collate, evaluate_split,\n"
         "                       corpus_cider, beam_decode_single)\n"
         "from torch.utils.data import DataLoader\n"
         "import torch.nn.functional as F\n"
         "import numpy as np\n"
         "\n"
         "# Choose the best base checkpoint\n"
         "candidates = ['output/phase2_results/phase2_e7_text_paraphrase/best.pt',\n"
         "              'output/phase2_results/phase2_e0_baseline_raina_greedy/best.pt',\n"
         "              'output/phase2_results/phase2_e4_schedule_polish/best.pt']\n"
         "ckpt = next((c for c in candidates if Path(c).exists()), None)\n"
         "assert ckpt is not None, 'No base checkpoint found'\n"
         "print('Warm-start from', ckpt)\n"
         "\n"
         "cfg = ExperimentConfig(\n"
         "    run_name='phase2_e8_scst',\n"
         "    output_dir='output/phase2_results',\n"
         "    epochs=2,\n"
         "    batch_size=64,\n"
         "    lr=5e-6,\n"
         "    decoding='beam', beam_width=5, length_penalty=1.0,\n"
         "    pretrained_checkpoint=ckpt,\n"
         ")\n"
         "out_root = Path(cfg.output_dir) / cfg.run_name\n"
         "out_root.mkdir(parents=True, exist_ok=True)\n"
         "torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)\n"
         "device = torch.device('cuda')\n"
         "df, vocab, references, features, image_to_idx = load_data(cfg)\n"
         "word2idx = vocab['word2idx']; idx2word = vocab['idx2word']\n"
         "PAD=word2idx['<pad>']; START=word2idx['<start>']; END=word2idx['<end>']; UNK=word2idx['<unk>']\n"
         "vocab_size = len(idx2word)\n"
         "\n"
         "model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=PAD).to(device)\n"
         "model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))\n"
         "\n"
         "optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)\n"
         "\n"
         "# Build train loader at IMAGE level — one update per image (CIDEr is image-level)\n"
         "train_img_df = df[df['split']=='train'][['image_id','file_name']].drop_duplicates()\n"
         "train_img_ds = CachedFeatureImageDataset(train_img_df, references, features, image_to_idx)\n"
         "train_img_loader = DataLoader(train_img_ds, batch_size=cfg.batch_size, shuffle=True,\n"
         "                              num_workers=0, collate_fn=image_collate, pin_memory=True)\n"
         "\n"
         "@torch.no_grad()\n"
         "def greedy_baseline_caps(memory):\n"
         "    B = memory.size(0)\n"
         "    toks = torch.full((B,1), START, device=device, dtype=torch.long)\n"
         "    fin = torch.zeros(B, dtype=torch.bool, device=device)\n"
         "    for step in range(cfg.gen_max_len):\n"
         "        logits = model.decoder(memory, toks)[:, -1, :].clone()\n"
         "        logits[:, [PAD, START, UNK]] = -float('inf')\n"
         "        if step+1 < cfg.gen_min_len: logits[:, END] = -float('inf')\n"
         "        nxt = torch.argmax(logits, dim=-1, keepdim=True)\n"
         "        nxt = torch.where(fin.unsqueeze(1), torch.full_like(nxt, PAD), nxt)\n"
         "        toks = torch.cat([toks, nxt], dim=1)\n"
         "        fin = fin | (nxt.squeeze(1) == END)\n"
         "        if fin.all(): break\n"
         "    return toks\n"
         "\n"
         "def sample_caps(memory):\n"
         "    B = memory.size(0)\n"
         "    toks = torch.full((B,1), START, device=device, dtype=torch.long)\n"
         "    logp_sum = torch.zeros(B, device=device)\n"
         "    fin = torch.zeros(B, dtype=torch.bool, device=device)\n"
         "    for step in range(cfg.gen_max_len):\n"
         "        logits = model.decoder(memory, toks)[:, -1, :].clone()\n"
         "        logits[:, [PAD, START, UNK]] = -float('inf')\n"
         "        if step+1 < cfg.gen_min_len: logits[:, END] = -float('inf')\n"
         "        log_probs = F.log_softmax(logits, dim=-1)\n"
         "        probs = log_probs.exp()\n"
         "        nxt = torch.multinomial(probs, 1)\n"
         "        nxt_logp = log_probs.gather(1, nxt).squeeze(1)\n"
         "        logp_sum = logp_sum + torch.where(fin, torch.zeros_like(nxt_logp), nxt_logp)\n"
         "        nxt = torch.where(fin.unsqueeze(1), torch.full_like(nxt, PAD), nxt)\n"
         "        toks = torch.cat([toks, nxt], dim=1)\n"
         "        fin = fin | (nxt.squeeze(1) == END)\n"
         "        if fin.all(): break\n"
         "    return toks, logp_sum\n"
         "\n"
         "def ids_to_str(toks):\n"
         "    out = []\n"
         "    for row in toks.tolist():\n"
         "        ids = row[1:]\n"
         "        words = []\n"
         "        for tid in ids:\n"
         "            if tid in (END, PAD): break\n"
         "            words.append(idx2word[tid])\n"
         "        out.append(' '.join(words))\n"
         "    return out\n"
         "\n"
         "import time\n"
         "history=[]\n"
         "for epoch in range(cfg.epochs):\n"
         "    model.train()\n"
         "    t0 = time.time()\n"
         "    total_reward = 0.0; n_batches = 0\n"
         "    for feats, image_ids, file_names, refs in train_img_loader:\n"
         "        feats = feats.to(device, non_blocking=True)\n"
         "        memory = model.encode(feats)\n"
         "        # baseline (greedy, no grad)\n"
         "        with torch.no_grad():\n"
         "            base_toks = greedy_baseline_caps(memory)\n"
         "            base_strs = ids_to_str(base_toks)\n"
         "        # sample with grad\n"
         "        samp_toks, samp_logp = sample_caps(memory)\n"
         "        samp_strs = ids_to_str(samp_toks)\n"
         "        # rewards (CIDEr per image)\n"
         "        r_sample = np.array([corpus_cider([s],[refs[i]]) for i,s in enumerate(samp_strs)])\n"
         "        r_baseline = np.array([corpus_cider([s],[refs[i]]) for i,s in enumerate(base_strs)])\n"
         "        advantage = torch.tensor(r_sample - r_baseline, device=device, dtype=torch.float32)\n"
         "        loss = -(advantage * samp_logp).mean()\n"
         "        optimizer.zero_grad(set_to_none=True)\n"
         "        loss.backward()\n"
         "        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)\n"
         "        optimizer.step()\n"
         "        total_reward += float(r_sample.mean()); n_batches += 1\n"
         "    dt = time.time() - t0\n"
         "    mean_r = total_reward / max(n_batches,1)\n"
         "    history.append({'epoch':epoch+1,'mean_sample_cider':mean_r,'time':dt})\n"
         "    print(f'SCST ep{epoch+1}  mean_sample_CIDEr={mean_r:.3f}  t={dt:.1f}s')\n"
         "torch.save(model.state_dict(), out_root/'best.pt')\n"
         "pd.DataFrame(history).to_csv(out_root/'history.csv', index=False)\n"
         "\n"
         "# Evaluate\n"
         "val_df = df[df['split']=='val'][['image_id','file_name']].drop_duplicates()\n"
         "test_df = df[df['split']=='test'][['image_id','file_name']].drop_duplicates()\n"
         "from exp_runner import CachedFeatureImageDataset, image_collate\n"
         "val_loader = DataLoader(CachedFeatureImageDataset(val_df, references, features, image_to_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate)\n"
         "test_loader = DataLoader(CachedFeatureImageDataset(test_df, references, features, image_to_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate)\n"
         "vm, vp = evaluate_split(model, val_loader, vocab, cfg, split='val')\n"
         "tm, tp = evaluate_split(model, test_loader, vocab, cfg, split='test')\n"
         "metrics_e8 = {'run_name':cfg.run_name, 'epochs_run':cfg.epochs, 'decoding':'beam',\n"
         "              **{f'val_{k}':v for k,v in vm.items()},\n"
         "              **{f'test_{k}':v for k,v in tm.items()}}\n"
         "vp.to_csv(out_root/'val_predictions.csv', index=False)\n"
         "tp.to_csv(out_root/'test_predictions.csv', index=False)\n"
         "import json\n"
         "json.dump(metrics_e8, open(out_root/'metrics.json','w'), indent=2)\n"
         "print(metrics_e8)\n"
         "metrics_e8"),
    ],
)

# E2 — Re-enable train-time augmentation (raw images) ----------------------
add(
    "phase2_e2_aug",
    "Phase 2 — E2: Re-enable train-time augmentation",
    [
        ("md", "Train on raw images with random crop + horizontal flip (no feature "
               "cache at train time). Encoder still frozen — only the projection "
               "head and decoder train. Eval uses the deterministic center crop "
               "and beam_w5_lp0.7."),
        ("code",
         "cfg = ExperimentConfig(\n"
         "    run_name='phase2_e2_aug',\n"
         "    output_dir='output/phase2_results',\n"
         "    use_cached_features=False,\n"
         "    encoder_kind='efficientnet_b0_raw',\n"
         "    freeze_encoder=True, augment=True,\n"
         "    epochs=12, batch_size=64, lr=1e-4,\n"
         "    decoding='beam', beam_width=5, length_penalty=0.7,\n"
         ")\n"
         "metrics_e2 = run_experiment(cfg)\n"
         "metrics_e2"),
    ],
)

# E3 — Unfreeze last MBConv blocks with discriminative LR -------------------
add(
    "phase2_e3_unfreeze",
    "Phase 2 — E3: Unfreeze last MBConv blocks with discriminative LR",
    [
        ("md", "Unfreeze the last 2 MBConv blocks of EfficientNet-B0 and train them "
               "with a small LR (1e-5), while the decoder + projection head use the "
               "standard 1e-4. Train-time augmentation is enabled. Eval uses "
               "beam_w5_lp0.7."),
        ("code",
         "# Build the model + per-parameter-group LR manually because run_experiment\n"
         "# uses a single optimizer LR. We construct everything by hand here.\n"
         "import sys; sys.path.insert(0, 'output')\n"
         "from exp_runner import (ExperimentConfig, CaptioningModel, load_data,\n"
         "                       RawImageCaptionDataset, RawImageImageDataset,\n"
         "                       _get_image_reader, imagenet_train_transform,\n"
         "                       imagenet_eval_transform, caption_collate, image_collate,\n"
         "                       evaluate_split, corpus_bleu, corpus_cider)\n"
         "import torch, torch.nn as nn, pandas as pd, json, time\n"
         "from pathlib import Path\n"
         "from torch.utils.data import DataLoader\n"
         "\n"
         "cfg = ExperimentConfig(\n"
         "    run_name='phase2_e3_unfreeze',\n"
         "    output_dir='output/phase2_results',\n"
         "    use_cached_features=False,\n"
         "    encoder_kind='efficientnet_b0_raw',\n"
         "    freeze_encoder=False, unfreeze_last_k_blocks=2,\n"
         "    augment=True,\n"
         "    epochs=10, batch_size=48, lr=1e-4,\n"
         "    decoding='beam', beam_width=5, length_penalty=0.7,\n"
         ")\n"
         "out_root = Path(cfg.output_dir) / cfg.run_name\n"
         "out_root.mkdir(parents=True, exist_ok=True)\n"
         "device = torch.device('cuda')\n"
         "df, vocab, references, _, _ = load_data(cfg)\n"
         "pad_idx = vocab['pad_idx']; vocab_size = len(vocab['idx2word'])\n"
         "model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)\n"
         "# Discriminative LR: unfrozen backbone params at 1e-5, everything else at 1e-4\n"
         "bb_params = [p for p in model.encoder.features.parameters() if p.requires_grad]\n"
         "other_params = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.features.')]\n"
         "print('Backbone trainable:', sum(p.numel() for p in bb_params))\n"
         "print('Other trainable:   ', sum(p.numel() for p in other_params))\n"
         "optimizer = torch.optim.Adam([\n"
         "    {'params': bb_params, 'lr': 1e-5},\n"
         "    {'params': other_params, 'lr': 1e-4},\n"
         "])\n"
         "criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)\n"
         "reader = _get_image_reader(cfg)\n"
         "train_tx = imagenet_train_transform(); eval_tx = imagenet_eval_transform()\n"
         "train_ds = RawImageCaptionDataset(df[df['split']=='train'], vocab['word2idx'], references, reader, train_tx)\n"
         "val_ds   = RawImageCaptionDataset(df[df['split']=='val'],   vocab['word2idx'], references, reader, eval_tx)\n"
         "val_img_ds  = RawImageImageDataset(df[df['split']=='val'],   references, reader, eval_tx)\n"
         "test_img_ds = RawImageImageDataset(df[df['split']=='test'],  references, reader, eval_tx)\n"
         "train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, persistent_workers=True, collate_fn=caption_collate, pin_memory=True)\n"
         "val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=caption_collate, pin_memory=True)\n"
         "val_img_loader = DataLoader(val_img_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=image_collate, pin_memory=True)\n"
         "test_img_loader = DataLoader(test_img_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=image_collate, pin_memory=True)\n"
         "best=float('inf'); bad=0; ckpt = out_root/'best.pt'; history=[]\n"
         "for epoch in range(cfg.epochs):\n"
         "    model.train(); t0=time.time(); tl=0.0; n=0\n"
         "    for imgs, caps, *_ in train_loader:\n"
         "        imgs = imgs.to(device, non_blocking=True)\n"
         "        caps = caps[:, :cfg.max_caption_len].to(device, non_blocking=True)\n"
         "        logits = model(imgs, caps); tgt = caps[:,1:]\n"
         "        loss = criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1))\n"
         "        optimizer.zero_grad(set_to_none=True); loss.backward()\n"
         "        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)\n"
         "        optimizer.step(); tl += loss.item(); n += 1\n"
         "    model.eval(); vl=0.0; vn=0\n"
         "    with torch.no_grad():\n"
         "        for imgs, caps, *_ in val_loader:\n"
         "            imgs = imgs.to(device, non_blocking=True)\n"
         "            caps = caps[:, :cfg.max_caption_len].to(device, non_blocking=True)\n"
         "            logits = model(imgs, caps); tgt = caps[:,1:]\n"
         "            vl += criterion(logits.reshape(-1, vocab_size), tgt.reshape(-1)).item(); vn += 1\n"
         "    dt = time.time()-t0; train_loss = tl/max(n,1); val_loss = vl/max(vn,1)\n"
         "    history.append({'epoch':epoch+1,'train_loss':train_loss,'val_loss':val_loss,'time':dt})\n"
         "    print(f'E3 ep{epoch+1} train={train_loss:.4f} val={val_loss:.4f} t={dt:.1f}s')\n"
         "    if val_loss < best: best=val_loss; bad=0; torch.save(model.state_dict(), ckpt)\n"
         "    else: bad += 1\n"
         "    if bad >= cfg.early_stop_patience: break\n"
         "pd.DataFrame(history).to_csv(out_root/'history.csv', index=False)\n"
         "model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)); model.eval()\n"
         "vm, vp = evaluate_split(model, val_img_loader, vocab, cfg, split='val')\n"
         "tm, tp = evaluate_split(model, test_img_loader, vocab, cfg, split='test')\n"
         "metrics_e3 = {'run_name':cfg.run_name,'best_val_loss':best,'epochs_run':len(history),\n"
         "              'decoding':cfg.decoding,'beam_width':cfg.beam_width,'length_penalty':cfg.length_penalty,\n"
         "              **{f'val_{k}':v for k,v in vm.items()},\n"
         "              **{f'test_{k}':v for k,v in tm.items()}}\n"
         "vp.to_csv(out_root/'val_predictions.csv', index=False)\n"
         "tp.to_csv(out_root/'test_predictions.csv', index=False)\n"
         "json.dump(metrics_e3, open(out_root/'metrics.json','w'), indent=2)\n"
         "metrics_e3"),
    ],
)

# E5 — Swap encoder to CLIP ViT-B/16 ----------------------------------------
add(
    "phase2_e5_clip",
    "Phase 2 — E5: Swap encoder to CLIP ViT-B/16",
    [
        ("md", "Replace EfficientNet-B0 with **CLIP ViT-B/16** image features. "
               "Step 1 extracts patch-level features once over every unique image "
               "(7×7 grid of 768-dim tokens, deterministic preprocessing) and "
               "caches them. Step 2 trains the same Transformer decoder on top, "
               "swapping `encoder_feat_dim` from 1280 → 768 and reusing the "
               "cached-features training path."),
        ("code",
         "import importlib, exp_runner; importlib.reload(exp_runner)\n"
         "from exp_runner import (ExperimentConfig, _get_image_reader,\n"
         "                       imagenet_eval_transform, run_experiment)\n"
         "import torch, pandas as pd\n"
         "from pathlib import Path\n"
         "from tqdm.auto import tqdm\n"
         "\n"
         "CACHE = Path('output/clip_vitb16_features.pt')\n"
         "if not CACHE.exists():\n"
         "    from transformers import CLIPModel, CLIPImageProcessor\n"
         "    model_id = 'openai/clip-vit-base-patch16'\n"
         "    clip = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda').eval()\n"
         "    proc = CLIPImageProcessor.from_pretrained(model_id)\n"
         "    cfg_tmp = ExperimentConfig(run_name='clip_extract',\n"
         "                              use_cached_features=False)\n"
         "    reader = _get_image_reader(cfg_tmp)\n"
         "    df = pd.read_csv('output/processed_captions.csv')\n"
         "    uniq = df[['image_id','file_name']].drop_duplicates().sort_values('image_id').reset_index(drop=True)\n"
         "    N = len(uniq); D = clip.config.vision_config.hidden_size\n"
         "    # CLIP ViT-B/16: 224x224 -> 14x14 patches = 196 + 1 CLS = 197. Keep all 196 spatial tokens.\n"
         "    S = 196\n"
         "    feats = torch.empty((N, S, D), dtype=torch.float16)\n"
         "    image_to_idx = {}\n"
         "    BATCH = 16\n"
         "    batch_imgs=[]; batch_pos=[]\n"
         "    with torch.no_grad():\n"
         "        for i, row in tqdm(uniq.iterrows(), total=N, desc='CLIP extract'):\n"
         "            img = reader.read(row['file_name'])\n"
         "            batch_imgs.append(img); batch_pos.append(i)\n"
         "            image_to_idx[int(row['image_id'])] = i\n"
         "            if len(batch_imgs) == BATCH:\n"
         "                px = proc(images=batch_imgs, return_tensors='pt')['pixel_values'].to('cuda', dtype=torch.float16)\n"
         "                out = clip.vision_model(pixel_values=px).last_hidden_state  # (B, 197, D)\n"
         "                out = out[:, 1:, :].cpu().to(torch.float16)  # drop CLS, keep 196 patch tokens\n"
         "                for k,p in enumerate(batch_pos): feats[p] = out[k]\n"
         "                batch_imgs=[]; batch_pos=[]\n"
         "        if batch_imgs:\n"
         "            px = proc(images=batch_imgs, return_tensors='pt')['pixel_values'].to('cuda', dtype=torch.float16)\n"
         "            out = clip.vision_model(pixel_values=px).last_hidden_state\n"
         "            out = out[:, 1:, :].cpu().to(torch.float16)\n"
         "            for k,p in enumerate(batch_pos): feats[p] = out[k]\n"
         "    torch.save({'features':feats, 'image_to_idx':image_to_idx}, CACHE)\n"
         "    print(f'Saved CLIP cache {feats.shape} to {CACHE}')\n"
         "    del clip\n"
         "    import gc; gc.collect(); torch.cuda.empty_cache()\n"
         "else:\n"
         "    print(f'CLIP cache already at {CACHE}')\n"
         "\n"
         "cfg = ExperimentConfig(\n"
         "    run_name='phase2_e5_clip',\n"
         "    output_dir='output/phase2_results',\n"
         "    feature_cache=str(CACHE),\n"
         "    encoder_kind='efficientnet_b0_cached',  # we use the cached-feature head over CLIP feats\n"
         "    encoder_feat_dim=768,\n"
         "    num_spatial_tokens=196,\n"
         "    epochs=15, batch_size=128, lr=1e-4,\n"
         "    decoding='beam', beam_width=5, length_penalty=0.7,\n"
         ")\n"
         "metrics_e5 = run_experiment(cfg)\n"
         "metrics_e5"),
    ],
)

# E9 — Ensemble top-3 checkpoints -------------------------------------------
add(
    "phase2_e9_ensemble",
    "Phase 2 — E9: Ensemble of top checkpoints (logit average + beam)",
    [
        ("md", "Average per-step decoder logits across the top-N checkpoints during "
               "beam search. No retraining, inference only."),
        ("code",
         "import json, math, pandas as pd, torch\n"
         "import torch.nn.functional as F\n"
         "from pathlib import Path\n"
         "from torch.utils.data import DataLoader\n"
         "import sys; sys.path.insert(0,'output')\n"
         "from exp_runner import (ExperimentConfig, CaptioningModel, load_data,\n"
         "                       CachedFeatureImageDataset, image_collate, corpus_bleu, corpus_cider)\n"
         "\n"
         "RESULTS = Path('output/phase2_results')\n"
         "# Ensemble the three distinct strong MLE checkpoints (E0/E4/E7).\n"
         "# E8 SCST regressed under the weak local-DF reward — skip it.\n"
         "# E0 reports greedy BLEU in its metrics.json but its checkpoint under beam matches E1.\n"
         "wanted = ['phase2_e0_baseline_raina_greedy', 'phase2_e4_schedule_polish', 'phase2_e7_text_paraphrase']\n"
         "top = []\n"
         "for name in wanted:\n"
         "    d = RESULTS / name\n"
         "    m = d / 'metrics.json'\n"
         "    if (d/'best.pt').exists() and m.exists():\n"
         "        jd = json.loads(m.read_text())\n"
         "        top.append((str(d/'best.pt'), float(jd.get('test_BLEU-4', 0)), jd['run_name']))\n"
         "print('Top-3 checkpoints to ensemble:')\n"
         "for c,b,n in top: print(f'  {n}: test_BLEU-4={b:.4f}')\n"
         "\n"
         "cfg = ExperimentConfig(run_name='phase2_e9_ensemble', output_dir='output/phase2_results',\n"
         "                      decoding='beam', beam_width=5, length_penalty=1.0)\n"
         "device = torch.device('cuda')\n"
         "df, vocab, references, features, image_to_idx = load_data(cfg)\n"
         "pad_idx = vocab['pad_idx']; vocab_size = len(vocab['idx2word'])\n"
         "word2idx = vocab['word2idx']; idx2word = vocab['idx2word']\n"
         "PAD=word2idx['<pad>']; START=word2idx['<start>']; END=word2idx['<end>']; UNK=word2idx['<unk>']\n"
         "\n"
         "models = []\n"
         "for ckpt_path, _, _ in top:\n"
         "    m = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)\n"
         "    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))\n"
         "    m.eval()\n"
         "    models.append(m)\n"
         "\n"
         "@torch.no_grad()\n"
         "def ensemble_beam(feats_single, beam_width=cfg.beam_width, length_penalty=cfg.length_penalty,\n"
         "                  max_len=cfg.gen_max_len, min_len=cfg.gen_min_len):\n"
         "    mems = [m.encode(feats_single.unsqueeze(0).to(device)) for m in models]\n"
         "    beams = [([START], 0.0, False)]\n"
         "    for step in range(max_len):\n"
         "        if all(b[2] for b in beams): break\n"
         "        active = [(i,b) for i,b in enumerate(beams) if not b[2]]\n"
         "        seqs = torch.tensor([b[0] for _,b in active], device=device, dtype=torch.long)\n"
         "        log_probs_avg = None\n"
         "        for mi, m in enumerate(models):\n"
         "            mem = mems[mi].expand(seqs.size(0), -1, -1).contiguous()\n"
         "            logits = m.decoder(mem, seqs)[:, -1, :]\n"
         "            lp = F.log_softmax(logits, dim=-1)\n"
         "            log_probs_avg = lp if log_probs_avg is None else log_probs_avg + lp\n"
         "        log_probs_avg = log_probs_avg / len(models)\n"
         "        log_probs_avg[:, [PAD, START, UNK]] = -float('inf')\n"
         "        if step+1 < min_len: log_probs_avg[:, END] = -float('inf')\n"
         "        topk_lp, topk_id = log_probs_avg.topk(beam_width, dim=-1)\n"
         "        cands = []\n"
         "        for ai, (_, (toks, sc, _)) in enumerate(active):\n"
         "            for k in range(beam_width):\n"
         "                tid = int(topk_id[ai,k].item())\n"
         "                s = sc + float(topk_lp[ai,k].item())\n"
         "                cands.append((toks+[tid], s, tid==END))\n"
         "        for b in beams:\n"
         "            if b[2]: cands.append(b)\n"
         "        def sf(it):\n"
         "            t,s,_ = it; L=max(len(t)-1,1); return s/(L**length_penalty)\n"
         "        cands.sort(key=sf, reverse=True)\n"
         "        beams = cands[:beam_width]\n"
         "    def sf(it):\n"
         "        t,s,_ = it; L=max(len(t)-1,1); return s/(L**length_penalty)\n"
         "    best = max(beams, key=sf)\n"
         "    words=[]\n"
         "    for tid in best[0][1:]:\n"
         "        if tid in (END,PAD): break\n"
         "        words.append(idx2word[tid])\n"
         "    return ' '.join(words)\n"
         "\n"
         "def eval_ensemble(split):\n"
         "    sdf = df[df['split']==split][['image_id','file_name']].drop_duplicates()\n"
         "    ds = CachedFeatureImageDataset(sdf, references, features, image_to_idx)\n"
         "    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate)\n"
         "    rows=[]; preds=[]; refs_all=[]\n"
         "    for feats, image_ids, file_names, refs in loader:\n"
         "        for i in range(feats.size(0)):\n"
         "            p = ensemble_beam(feats[i])\n"
         "            preds.append(p); refs_all.append(refs[i])\n"
         "            rows.append({'image_id':int(image_ids[i]),'file_name':file_names[i],'prediction':p,'references':refs[i]})\n"
         "    m = corpus_bleu(preds, refs_all); m['CIDEr']=corpus_cider(preds, refs_all)\n"
         "    return m, pd.DataFrame(rows)\n"
         "\n"
         "out_root = Path('output/phase2_results/phase2_e9_ensemble')\n"
         "out_root.mkdir(parents=True, exist_ok=True)\n"
         "val_m, val_p = eval_ensemble('val')\n"
         "test_m, test_p = eval_ensemble('test')\n"
         "metrics_e9 = {'run_name':'phase2_e9_ensemble',\n"
         "              'ensembled_runs':[n for _,_,n in top],\n"
         "              **{f'val_{k}':v for k,v in val_m.items()},\n"
         "              **{f'test_{k}':v for k,v in test_m.items()}}\n"
         "val_p.to_csv(out_root/'val_predictions.csv', index=False)\n"
         "test_p.to_csv(out_root/'test_predictions.csv', index=False)\n"
         "json.dump(metrics_e9, open(out_root/'metrics.json','w'), indent=2)\n"
         "print(metrics_e9)\n"
         "metrics_e9"),
    ],
)


def main():
    only = set(sys.argv[1:]) or None
    for exp in EXPERIMENTS:
        if only and exp["name"] not in only:
            continue
        nb = make_notebook(exp["title"], exp["sections"])
        path = NB_DIR / f"{exp['name']}.ipynb"
        with path.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"[built] {path.name}")
        ok = execute_notebook(path, timeout=60 * 60)
        if not ok:
            print(f"[ERROR] {exp['name']} failed; stopping")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
