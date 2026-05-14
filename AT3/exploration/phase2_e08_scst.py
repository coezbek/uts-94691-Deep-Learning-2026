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
# # Phase 2 — E8: SCST (CIDEr-reward RL fine-tune)

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
# Self-Critical Sequence Training (Rennie 2017): warm-start from the best MLE checkpoint (E4 if it improved, else E0); then for 1-2 epochs sample captions, score with CIDEr, and use REINFORCE with a greedy-decode baseline.
#
# This is the canonical post-MLE polish for captioning.

# %%
# Pick the best MLE checkpoint to warm-start from
import json
from pathlib import Path
import torch, pandas as pd
import sys
sys.path.insert(0, 'output')
from exp_runner import (ExperimentConfig, CaptioningModel, load_data,
                       CachedFeatureCaptionDataset, CachedFeatureImageDataset,
                       caption_collate, image_collate, evaluate_split,
                       corpus_cider, beam_decode_single)
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

# Choose the best base checkpoint
candidates = ['output/phase2_results/phase2_e7_text_paraphrase/best.pt',
              'output/phase2_results/phase2_e0_baseline_raina_greedy/best.pt',
              'output/phase2_results/phase2_e4_schedule_polish/best.pt']
ckpt = next((c for c in candidates if Path(c).exists()), None)
assert ckpt is not None, 'No base checkpoint found'
print('Warm-start from', ckpt)

cfg = ExperimentConfig(
    run_name='phase2_e8_scst',
    output_dir='output/phase2_results',
    epochs=2,
    batch_size=64,
    lr=5e-6,
    decoding='beam', beam_width=5, length_penalty=1.0,
    pretrained_checkpoint=ckpt,
)
out_root = Path(cfg.output_dir) / cfg.run_name
out_root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
device = torch.device('cuda')
df, vocab, references, features, image_to_idx = load_data(cfg)
word2idx = vocab['word2idx']; idx2word = vocab['idx2word']
PAD=word2idx['<pad>']; START=word2idx['<start>']; END=word2idx['<end>']; UNK=word2idx['<unk>']
vocab_size = len(idx2word)

model = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=PAD).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

# Build train loader at IMAGE level — one update per image (CIDEr is image-level)
train_img_df = df[df['split']=='train'][['image_id','file_name']].drop_duplicates()
train_img_ds = CachedFeatureImageDataset(train_img_df, references, features, image_to_idx)
train_img_loader = DataLoader(train_img_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, collate_fn=image_collate, pin_memory=True)

@torch.no_grad()
def greedy_baseline_caps(memory):
    B = memory.size(0)
    toks = torch.full((B,1), START, device=device, dtype=torch.long)
    fin = torch.zeros(B, dtype=torch.bool, device=device)
    for step in range(cfg.gen_max_len):
        logits = model.decoder(memory, toks)[:, -1, :].clone()
        logits[:, [PAD, START, UNK]] = -float('inf')
        if step+1 < cfg.gen_min_len: logits[:, END] = -float('inf')
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        nxt = torch.where(fin.unsqueeze(1), torch.full_like(nxt, PAD), nxt)
        toks = torch.cat([toks, nxt], dim=1)
        fin = fin | (nxt.squeeze(1) == END)
        if fin.all(): break
    return toks

def sample_caps(memory):
    B = memory.size(0)
    toks = torch.full((B,1), START, device=device, dtype=torch.long)
    logp_sum = torch.zeros(B, device=device)
    fin = torch.zeros(B, dtype=torch.bool, device=device)
    for step in range(cfg.gen_max_len):
        logits = model.decoder(memory, toks)[:, -1, :].clone()
        logits[:, [PAD, START, UNK]] = -float('inf')
        if step+1 < cfg.gen_min_len: logits[:, END] = -float('inf')
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        nxt = torch.multinomial(probs, 1)
        nxt_logp = log_probs.gather(1, nxt).squeeze(1)
        logp_sum = logp_sum + torch.where(fin, torch.zeros_like(nxt_logp), nxt_logp)
        nxt = torch.where(fin.unsqueeze(1), torch.full_like(nxt, PAD), nxt)
        toks = torch.cat([toks, nxt], dim=1)
        fin = fin | (nxt.squeeze(1) == END)
        if fin.all(): break
    return toks, logp_sum

def ids_to_str(toks):
    out = []
    for row in toks.tolist():
        ids = row[1:]
        words = []
        for tid in ids:
            if tid in (END, PAD): break
            words.append(idx2word[tid])
        out.append(' '.join(words))
    return out

import time
history=[]
for epoch in range(cfg.epochs):
    model.train()
    t0 = time.time()
    total_reward = 0.0; n_batches = 0
    for feats, image_ids, file_names, refs in train_img_loader:
        feats = feats.to(device, non_blocking=True)
        memory = model.encode(feats)
        # baseline (greedy, no grad)
        with torch.no_grad():
            base_toks = greedy_baseline_caps(memory)
            base_strs = ids_to_str(base_toks)
        # sample with grad
        samp_toks, samp_logp = sample_caps(memory)
        samp_strs = ids_to_str(samp_toks)
        # rewards (CIDEr per image)
        r_sample = np.array([corpus_cider([s],[refs[i]]) for i,s in enumerate(samp_strs)])
        r_baseline = np.array([corpus_cider([s],[refs[i]]) for i,s in enumerate(base_strs)])
        advantage = torch.tensor(r_sample - r_baseline, device=device, dtype=torch.float32)
        loss = -(advantage * samp_logp).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total_reward += float(r_sample.mean()); n_batches += 1
    dt = time.time() - t0
    mean_r = total_reward / max(n_batches,1)
    history.append({'epoch':epoch+1,'mean_sample_cider':mean_r,'time':dt})
    print(f'SCST ep{epoch+1}  mean_sample_CIDEr={mean_r:.3f}  t={dt:.1f}s')
torch.save(model.state_dict(), out_root/'best.pt')
pd.DataFrame(history).to_csv(out_root/'history.csv', index=False)

# Evaluate
val_df = df[df['split']=='val'][['image_id','file_name']].drop_duplicates()
test_df = df[df['split']=='test'][['image_id','file_name']].drop_duplicates()
from exp_runner import CachedFeatureImageDataset, image_collate
val_loader = DataLoader(CachedFeatureImageDataset(val_df, references, features, image_to_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate)
test_loader = DataLoader(CachedFeatureImageDataset(test_df, references, features, image_to_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate)
vm, vp = evaluate_split(model, val_loader, vocab, cfg, split='val')
tm, tp = evaluate_split(model, test_loader, vocab, cfg, split='test')
metrics_e8 = {'run_name':cfg.run_name, 'epochs_run':cfg.epochs, 'decoding':'beam',
              **{f'val_{k}':v for k,v in vm.items()},
              **{f'test_{k}':v for k,v in tm.items()}}
vp.to_csv(out_root/'val_predictions.csv', index=False)
tp.to_csv(out_root/'test_predictions.csv', index=False)
import json
json.dump(metrics_e8, open(out_root/'metrics.json','w'), indent=2)
print(metrics_e8)
metrics_e8
