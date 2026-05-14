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
# # Phase 2 — E6: OCR-augmented memory bank
#
# VizWiz captions describe a *lot* of on-image text — product labels, screens,
# packaging, signs. EfficientNet features encode "there is text" but not
# "*what* the text says." We bolt an OCR signal onto the cross-attention
# memory:
#
# 1. Run **EasyOCR** over every unique image once, save the concatenated
#    string per image.
# 2. Tokenise each OCR string with the captioning vocab (`<unk>` for missing),
#    pad/truncate to a fixed length M.
# 3. The captioning model encodes the OCR ids with the *same* token embedding
#    (which lets the decoder copy OCR tokens into the output cheaply) plus a
#    learned **segment embedding** that flags visual vs OCR tokens.
# 4. Concatenate visual memory (49) and OCR memory (M) → cross-attention
#    memory of length 49+M. Padding mask covers OCR pad positions.

# %%
import sys
import json
import time
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path("output").resolve()))
import importlib
import exp_runner
importlib.reload(exp_runner)
from exp_runner import (
    ExperimentConfig,
    CachedFeatureEncoder,
    TransformerDecoder,
    load_data,
    caption_collate,
    image_collate,
    corpus_bleu,
    corpus_cider,
)

# %% [markdown]
# ## 1. OCR extraction

# %%
OCR_CACHE = Path("output/easyocr_per_image.json")
# Live progress + crash-recovery log lives next to the cache.
OCR_LOG_PATH = Path("output/easyocr_progress.log")
OCR_PARTIAL_PATH = Path("output/easyocr_per_image.partial.json")


def _log_ocr(msg: str):
    print(msg, flush=True)
    with open(OCR_LOG_PATH, "a", buffering=1, encoding="utf-8") as f:
        f.write(msg + "\n")


if not OCR_CACHE.exists():
    try:
        import easyocr
    except ImportError:
        print("easyocr not installed; install with `uv add easyocr`")
        raise

    _log_ocr("=== OCR extraction starting ===")
    reader_ocr = easyocr.Reader(["en"], gpu=True)
    _log_ocr("EasyOCR Reader loaded.")
    cfg_tmp = ExperimentConfig(run_name="ocr_extract", use_cached_features=False)
    from exp_runner import _get_image_reader, InMemoryImageCache, InMemoryReader  # late import
    # Use the in-memory image cache so we don't pay JPG decode per image on every iter.
    mem = InMemoryImageCache(cfg_tmp.image_dir)
    img_reader = InMemoryReader(mem)

    df = pd.read_csv("output/processed_captions.csv")
    uniq = df[["image_id", "file_name"]].drop_duplicates().sort_values("image_id").reset_index(drop=True)
    total = len(uniq)
    _log_ocr(f"Will OCR {total} unique images.")

    # Resume from partial cache if available
    if OCR_PARTIAL_PATH.exists():
        out: dict[str, str] = json.loads(OCR_PARTIAL_PATH.read_text(encoding="utf-8"))
        _log_ocr(f"Resuming from partial cache with {len(out)} images already done.")
    else:
        out = {}

    import time as _time
    t0 = _time.time()
    last_log = t0
    SAVE_EVERY = 500
    LOG_EVERY_SECONDS = 15
    for i, row in uniq.iterrows():
        key = str(int(row["image_id"]))
        if key in out:
            continue  # resumed
        pil = img_reader.read(row["file_name"])
        arr = np.array(pil)
        detections = reader_ocr.readtext(arr, detail=0, paragraph=True)
        out[key] = " ".join(detections).strip()

        now = _time.time()
        done = len(out)
        if now - last_log >= LOG_EVERY_SECONDS:
            rate = done / max(now - t0, 1e-9)
            eta_s = (total - done) / max(rate, 1e-9)
            _log_ocr(
                f"  OCR  {done}/{total}  ({100*done/total:.1f}%)   "
                f"rate={rate:.2f} img/s   elapsed={now - t0:.0f}s   eta={eta_s:.0f}s"
            )
            last_log = now
        if done % SAVE_EVERY == 0:
            OCR_PARTIAL_PATH.write_text(json.dumps(out), encoding="utf-8")

    OCR_CACHE.write_text(json.dumps(out), encoding="utf-8")
    if OCR_PARTIAL_PATH.exists():
        OCR_PARTIAL_PATH.unlink()
    _log_ocr(f"Saved OCR cache for {len(out)} images to {OCR_CACHE}  "
             f"(total time {(_time.time()-t0)/60:.1f} min)")
else:
    print(f"OCR cache already at {OCR_CACHE}")

OCR_TEXT = json.loads(OCR_CACHE.read_text(encoding="utf-8"))

# %% [markdown]
# ## 2. Tokenise OCR strings with the captioning vocab

# %%
M = 20  # OCR positions in memory


def clean(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise_ocr(s: str, word2idx: dict, pad: int, length: int = M) -> tuple[torch.Tensor, torch.Tensor]:
    toks = clean(s).split()[:length]
    ids = [word2idx.get(t, word2idx["<unk>"]) for t in toks]
    pad_n = length - len(ids)
    ids = ids + [pad] * pad_n
    mask = [False] * len(toks) + [True] * pad_n
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


# %% [markdown]
# ## 3. Model: cached visual features + OCR memory bank

# %%
cfg = ExperimentConfig(
    run_name="phase2_e6_ocr",
    output_dir="output/phase2_results",
    feature_cache="output/model1_26239780/efficientnet_b0_raw_features.pt",
    encoder_kind="efficientnet_b0_cached",
    encoder_feat_dim=1280,
    num_spatial_tokens=49,
    epochs=15,
    batch_size=128,
    lr=1e-4,
    decoding="beam",
    beam_width=5,
    length_penalty=0.7,
)
out_root = Path(cfg.output_dir) / cfg.run_name
out_root.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")

df, vocab, references, features, image_to_idx = load_data(cfg)
word2idx = vocab["word2idx"]
idx2word = vocab["idx2word"]
pad_idx = vocab["pad_idx"]
vocab_size = len(idx2word)


class OcrAugmentedModel(nn.Module):
    """Cached EfficientNet visual head + a small OCR token embedder, concatenated."""

    def __init__(self):
        super().__init__()
        self.visual = CachedFeatureEncoder(cfg.encoder_feat_dim, cfg.embed_dim, cfg.num_spatial_tokens)
        self.token_embed = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=pad_idx)
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        self.ocr_pos = nn.Parameter(torch.zeros(1, M, cfg.embed_dim))
        nn.init.trunc_normal_(self.ocr_pos, std=0.02)
        self.segment = nn.Embedding(2, cfg.embed_dim)  # 0 = visual, 1 = OCR
        nn.init.trunc_normal_(self.segment.weight, std=0.02)
        self.ocr_norm = nn.LayerNorm(cfg.embed_dim)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size, pad_idx=pad_idx,
            embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
            num_layers=cfg.num_decoder_layers, ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout, tie_embeddings=cfg.tie_embeddings,
        )

    def encode(self, raw_feats, ocr_ids, ocr_mask):
        # raw_feats: (B, 49, 1280), ocr_ids: (B, M), ocr_mask: (B, M) True=PAD
        v = self.visual(raw_feats)                                # (B, 49, D)
        v = v + self.segment.weight[0]                            # add visual segment id
        o = self.token_embed(ocr_ids) + self.ocr_pos              # (B, M, D)
        o = self.ocr_norm(o) + self.segment.weight[1]             # OCR segment id
        memory = torch.cat([v, o], dim=1)                         # (B, 49+M, D)
        # padding mask: visual is never padded, OCR uses ocr_mask
        vis_pad = torch.zeros(v.size(0), v.size(1), dtype=torch.bool, device=v.device)
        memory_key_padding_mask = torch.cat([vis_pad, ocr_mask], dim=1)
        return memory, memory_key_padding_mask

    def forward(self, raw_feats, ocr_ids, ocr_mask, captions_in):
        memory, mem_pad = self.encode(raw_feats, ocr_ids, ocr_mask)
        # call the decoder layers directly to pass memory_key_padding_mask
        tgt = self.decoder.embedding(captions_in) * (cfg.embed_dim ** 0.5)
        tgt = self.decoder.pos_enc(tgt)
        T = captions_in.size(1)
        out = self.decoder.decoder(
            tgt=tgt, memory=memory,
            tgt_mask=self.decoder.causal_mask(T, captions_in.device),
            tgt_key_padding_mask=(captions_in == pad_idx),
            memory_key_padding_mask=mem_pad,
        )
        return self.decoder.fc(out)


model = OcrAugmentedModel().to(device)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# %% [markdown]
# ## 4. Datasets

# %%
class CapDsWithOcr(Dataset):
    def __init__(self, sub_df):
        self.df = sub_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        iid = int(row["image_id"])
        feat = features[image_to_idx[iid]].float()
        ocr_ids, ocr_mask = tokenise_ocr(OCR_TEXT.get(str(iid), ""), word2idx, pad_idx, M)
        tokens = str(row["caption_clean"]).split()
        ids = [word2idx["<start>"]] + [word2idx.get(t, word2idx["<unk>"]) for t in tokens] + [word2idx["<end>"]]
        cap = torch.tensor(ids, dtype=torch.long)
        return feat, ocr_ids, ocr_mask, cap, iid, row["file_name"], references[str(iid)]


class ImgDsWithOcr(Dataset):
    def __init__(self, sub_df):
        sub_df = sub_df[["image_id", "file_name"]].drop_duplicates().reset_index(drop=True)
        self.df = sub_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        iid = int(row["image_id"])
        feat = features[image_to_idx[iid]].float()
        ocr_ids, ocr_mask = tokenise_ocr(OCR_TEXT.get(str(iid), ""), word2idx, pad_idx, M)
        return feat, ocr_ids, ocr_mask, iid, row["file_name"], references[str(iid)]


def collate_cap(batch):
    feats, oids, omask, caps, image_ids, file_names, refs = zip(*batch)
    feats = torch.stack(feats); oids = torch.stack(oids); omask = torch.stack(omask)
    padded = torch.nn.utils.rnn.pad_sequence(caps, batch_first=True, padding_value=pad_idx)
    return feats, oids, omask, padded, list(image_ids), list(file_names), list(refs)


def collate_img(batch):
    feats, oids, omask, image_ids, file_names, refs = zip(*batch)
    feats = torch.stack(feats); oids = torch.stack(oids); omask = torch.stack(omask)
    return feats, oids, omask, list(image_ids), list(file_names), list(refs)


train_loader = DataLoader(CapDsWithOcr(df[df["split"] == "train"]), batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=collate_cap, pin_memory=True)
val_loader = DataLoader(CapDsWithOcr(df[df["split"] == "val"]), batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_cap, pin_memory=True)
val_img_loader = DataLoader(ImgDsWithOcr(df[df["split"] == "val"]), batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate_img, pin_memory=True)
test_img_loader = DataLoader(ImgDsWithOcr(df[df["split"] == "test"]), batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_img, pin_memory=True)

# %% [markdown]
# ## 5. Training loop

# %%
best = float("inf"); bad = 0; ckpt = out_root / "best.pt"; history = []
for epoch in range(cfg.epochs):
    model.train(); t0 = time.time(); tl = 0.0; n = 0
    for feats, oids, omask, caps, *_ in train_loader:
        feats = feats.to(device, non_blocking=True)
        oids = oids.to(device, non_blocking=True)
        omask = omask.to(device, non_blocking=True)
        caps = caps[:, : cfg.max_caption_len].to(device, non_blocking=True)
        logits = model(feats, oids, omask, caps[:, :-1])
        targets = caps[:, 1:]
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        tl += loss.item(); n += 1
    model.eval(); vl = 0.0; vn = 0
    with torch.no_grad():
        for feats, oids, omask, caps, *_ in val_loader:
            feats = feats.to(device, non_blocking=True)
            oids = oids.to(device, non_blocking=True)
            omask = omask.to(device, non_blocking=True)
            caps = caps[:, : cfg.max_caption_len].to(device, non_blocking=True)
            logits = model(feats, oids, omask, caps[:, :-1])
            targets = caps[:, 1:]
            vl += criterion(logits.reshape(-1, vocab_size), targets.reshape(-1)).item(); vn += 1
    dt = time.time() - t0; train_loss = tl / max(n, 1); val_loss = vl / max(vn, 1)
    history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "time": dt})
    print(f"E6 ep{epoch+1:02d}/{cfg.epochs} train={train_loss:.4f} val={val_loss:.4f} t={dt:.1f}s")
    if val_loss < best:
        best = val_loss; bad = 0; torch.save(model.state_dict(), ckpt)
    else:
        bad += 1
    if bad >= cfg.early_stop_patience:
        break
pd.DataFrame(history).to_csv(out_root / "history.csv", index=False)

# %% [markdown]
# ## 6. Beam-search evaluation

# %%
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
model.eval()

START = word2idx["<start>"]; END = word2idx["<end>"]; UNK = word2idx["<unk>"]


@torch.no_grad()
def beam_one(feats_single, oids_single, omask_single, beam=cfg.beam_width, lp=cfg.length_penalty,
             maxlen=cfg.gen_max_len, minlen=cfg.gen_min_len):
    feats = feats_single.unsqueeze(0).to(device)
    oids = oids_single.unsqueeze(0).to(device)
    omask = omask_single.unsqueeze(0).to(device)
    memory, mem_pad = model.encode(feats, oids, omask)
    beams = [([START], 0.0, False)]
    for step in range(maxlen):
        if all(b[2] for b in beams):
            break
        active = [(i, b) for i, b in enumerate(beams) if not b[2]]
        seqs = torch.tensor([b[0] for _, b in active], device=device, dtype=torch.long)
        mem = memory.expand(seqs.size(0), -1, -1).contiguous()
        mp = mem_pad.expand(seqs.size(0), -1).contiguous()
        tgt = model.decoder.embedding(seqs) * (cfg.embed_dim ** 0.5)
        tgt = model.decoder.pos_enc(tgt)
        T = seqs.size(1)
        out = model.decoder.decoder(
            tgt=tgt, memory=mem,
            tgt_mask=model.decoder.causal_mask(T, device),
            tgt_key_padding_mask=(seqs == pad_idx),
            memory_key_padding_mask=mp,
        )
        logits = model.decoder.fc(out)[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1).clone()
        log_probs[:, [pad_idx, START, UNK]] = -float("inf")
        if step + 1 < minlen:
            log_probs[:, END] = -float("inf")
        topk_lp, topk_id = log_probs.topk(beam, dim=-1)
        cands = []
        for ai, (_, (toks, sc, _)) in enumerate(active):
            for k in range(beam):
                tid = int(topk_id[ai, k].item())
                s = sc + float(topk_lp[ai, k].item())
                cands.append((toks + [tid], s, tid == END))
        for b in beams:
            if b[2]:
                cands.append(b)

        def sf(it):
            t, s, _ = it
            return s / (max(len(t) - 1, 1) ** lp)
        cands.sort(key=sf, reverse=True)
        beams = cands[:beam]

    def sf(it):
        t, s, _ = it
        return s / (max(len(t) - 1, 1) ** lp)
    best_seq = max(beams, key=sf)[0][1:]
    words = []
    for tid in best_seq:
        if tid in (END, pad_idx):
            break
        words.append(idx2word[tid])
    return " ".join(words)


def eval_split(loader):
    rows, preds, refs_all = [], [], []
    for feats, oids, omask, image_ids, file_names, refs in loader:
        for i in range(feats.size(0)):
            p = beam_one(feats[i], oids[i], omask[i])
            preds.append(p); refs_all.append(refs[i])
            rows.append({"image_id": int(image_ids[i]), "file_name": file_names[i],
                         "prediction": p, "references": refs[i]})
    m = corpus_bleu(preds, refs_all); m["CIDEr"] = corpus_cider(preds, refs_all)
    return m, pd.DataFrame(rows)


vm, vp = eval_split(val_img_loader)
tm, tp = eval_split(test_img_loader)
metrics_e6 = {
    "run_name": cfg.run_name,
    "best_val_loss": best,
    "epochs_run": len(history),
    "decoding": cfg.decoding,
    "beam_width": cfg.beam_width,
    "length_penalty": cfg.length_penalty,
    **{f"val_{k}": v for k, v in vm.items()},
    **{f"test_{k}": v for k, v in tm.items()},
}
vp.to_csv(out_root / "val_predictions.csv", index=False)
tp.to_csv(out_root / "test_predictions.csv", index=False)
json.dump(metrics_e6, open(out_root / "metrics.json", "w"), indent=2)
metrics_e6
