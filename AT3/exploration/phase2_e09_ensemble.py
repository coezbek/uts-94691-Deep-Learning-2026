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
# # Phase 2 — E9: Ensemble of top checkpoints (logit average + beam)

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
# Average per-step decoder logits across the top-N checkpoints during beam search. No retraining, inference only.

# %%
import json, math, pandas as pd, torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
import sys; sys.path.insert(0,'output')
from exp_runner import (ExperimentConfig, CaptioningModel, load_data,
                       CachedFeatureImageDataset, image_collate, corpus_bleu, corpus_cider)

RESULTS = Path('output/phase2_results')
# Ensemble the three distinct strong MLE checkpoints (E0/E4/E7).
# E8 SCST regressed under the weak local-DF reward — skip it.
# E0 reports greedy BLEU in its metrics.json but its checkpoint under beam matches E1.
wanted = ['phase2_e0_baseline_raina_greedy', 'phase2_e4_schedule_polish', 'phase2_e7_text_paraphrase']
top = []
for name in wanted:
    d = RESULTS / name
    m = d / 'metrics.json'
    if (d/'best.pt').exists() and m.exists():
        jd = json.loads(m.read_text())
        top.append((str(d/'best.pt'), float(jd.get('test_BLEU-4', 0)), jd['run_name']))
print('Top-3 checkpoints to ensemble:')
for c,b,n in top: print(f'  {n}: test_BLEU-4={b:.4f}')

cfg = ExperimentConfig(run_name='phase2_e9_ensemble', output_dir='output/phase2_results',
                      decoding='beam', beam_width=5, length_penalty=1.0)
device = torch.device('cuda')
df, vocab, references, features, image_to_idx = load_data(cfg)
pad_idx = vocab['pad_idx']; vocab_size = len(vocab['idx2word'])
word2idx = vocab['word2idx']; idx2word = vocab['idx2word']
PAD=word2idx['<pad>']; START=word2idx['<start>']; END=word2idx['<end>']; UNK=word2idx['<unk>']

models = []
for ckpt_path, _, _ in top:
    m = CaptioningModel(cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    m.eval()
    models.append(m)

@torch.no_grad()
def ensemble_beam(feats_single, beam_width=cfg.beam_width, length_penalty=cfg.length_penalty,
                  max_len=cfg.gen_max_len, min_len=cfg.gen_min_len):
    mems = [m.encode(feats_single.unsqueeze(0).to(device)) for m in models]
    beams = [([START], 0.0, False)]
    for step in range(max_len):
        if all(b[2] for b in beams): break
        active = [(i,b) for i,b in enumerate(beams) if not b[2]]
        seqs = torch.tensor([b[0] for _,b in active], device=device, dtype=torch.long)
        log_probs_avg = None
        for mi, m in enumerate(models):
            mem = mems[mi].expand(seqs.size(0), -1, -1).contiguous()
            logits = m.decoder(mem, seqs)[:, -1, :]
            lp = F.log_softmax(logits, dim=-1)
            log_probs_avg = lp if log_probs_avg is None else log_probs_avg + lp
        log_probs_avg = log_probs_avg / len(models)
        log_probs_avg[:, [PAD, START, UNK]] = -float('inf')
        if step+1 < min_len: log_probs_avg[:, END] = -float('inf')
        topk_lp, topk_id = log_probs_avg.topk(beam_width, dim=-1)
        cands = []
        for ai, (_, (toks, sc, _)) in enumerate(active):
            for k in range(beam_width):
                tid = int(topk_id[ai,k].item())
                s = sc + float(topk_lp[ai,k].item())
                cands.append((toks+[tid], s, tid==END))
        for b in beams:
            if b[2]: cands.append(b)
        def sf(it):
            t,s,_ = it; L=max(len(t)-1,1); return s/(L**length_penalty)
        cands.sort(key=sf, reverse=True)
        beams = cands[:beam_width]
    def sf(it):
        t,s,_ = it; L=max(len(t)-1,1); return s/(L**length_penalty)
    best = max(beams, key=sf)
    words=[]
    for tid in best[0][1:]:
        if tid in (END,PAD): break
        words.append(idx2word[tid])
    return ' '.join(words)

def eval_ensemble(split):
    sdf = df[df['split']==split][['image_id','file_name']].drop_duplicates()
    ds = CachedFeatureImageDataset(sdf, references, features, image_to_idx)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=image_collate)
    rows=[]; preds=[]; refs_all=[]
    for feats, image_ids, file_names, refs in loader:
        for i in range(feats.size(0)):
            p = ensemble_beam(feats[i])
            preds.append(p); refs_all.append(refs[i])
            rows.append({'image_id':int(image_ids[i]),'file_name':file_names[i],'prediction':p,'references':refs[i]})
    m = corpus_bleu(preds, refs_all); m['CIDEr']=corpus_cider(preds, refs_all)
    return m, pd.DataFrame(rows)

out_root = Path('output/phase2_results/phase2_e9_ensemble')
out_root.mkdir(parents=True, exist_ok=True)
val_m, val_p = eval_ensemble('val')
test_m, test_p = eval_ensemble('test')
metrics_e9 = {'run_name':'phase2_e9_ensemble',
              'ensembled_runs':[n for _,_,n in top],
              **{f'val_{k}':v for k,v in val_m.items()},
              **{f'test_{k}':v for k,v in test_m.items()}}
val_p.to_csv(out_root/'val_predictions.csv', index=False)
test_p.to_csv(out_root/'test_predictions.csv', index=False)
json.dump(metrics_e9, open(out_root/'metrics.json','w'), indent=2)
print(metrics_e9)
metrics_e9
