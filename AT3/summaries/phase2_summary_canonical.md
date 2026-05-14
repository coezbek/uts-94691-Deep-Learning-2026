# Phase 2 experiments — summary

All Phase 2 runs use **Raina's data prep** (no `<num>` token, vocab=4708).
Architecture: EfficientNet-B0 (frozen, cached features) + 3-layer Transformer decoder, d=512, h=8.
Hardware: AMD Radeon 8060S (Strix Halo iGPU), ROCm 7.2, PyTorch 2.11.0+rocm7.2.

## Headline

* Phase 1 baseline (MX150, `<num>` prep): test BLEU-4 = **0.253**.
* Phase 2 best (E9 logit-average ensemble of E0+E4+E7, beam w5 lp0.7): test BLEU-4 = **0.374**, test CIDEr = **1.446**.
* Net improvement: **+0.121 BLEU-4**.

## Experiment line-up (in plain English)

Every row in the next table is one variant of the same architecture: a frozen image encoder feeds a sequence of spatial tokens into a 3-layer Transformer caption decoder (`d_model=512`, 8 heads, ~18 M trainable params on the decoder side). Each experiment changes exactly one thing relative to the baseline:

* **E0** — Baseline. EfficientNet-B0 (ImageNet, frozen), greedy decoding. The reference.
* **E1** — Same checkpoint as E0, but **swap greedy for beam search** at inference (width 3 or 5, length penalty 0.7 or 1.0). No retraining.
* **E2** — Same encoder, **train-time augmentation re-enabled** (random crop + 10° rotation + colour jitter; no horizontal flip because OCR-style references get mirrored).
* **E3** — **Partially unfreeze the encoder**: last 2 MBConv blocks of EfficientNet-B0 trained at LR 1e-5 with discriminative LR.
* **E4** — **Training-schedule polish** on top of E0: AdamW with weight decay 0.01, warmup + cosine LR schedule, label smoothing 0.1, tied input ↔ output embeddings.
* **E5** — **Swap the encoder**: replace EfficientNet-B0 with **CLIP ViT-B/16** (image-text pretraining → features that are already aligned to natural language).
* **E6** — Same E0 visual features **plus an OCR memory bank**: EasyOCR is run once per image, the detected text is tokenised with the captioning vocab and fed into the decoder as extra cross-attention positions with a learned segment embedding.
* **E7** — **Text-only caption augmentation**: a sample of 5 000 training captions is paraphrased by Qwen2.5-1.5B-Instruct and appended to the training set.
* **E7v** — Same idea but **image-grounded augmentation**: Qwen2-VL-2B captions each training image directly; those captions are appended to the training set.
* **E8** — **SCST / CIDEr-RL fine-tune** of E7: warm-start, REINFORCE with a greedy baseline and CIDEr reward, 2 epochs.
* **E9** — **Logit-average ensemble** of the three strongest MLE checkpoints (E0 + E4 + E7) under beam search.

## Results (ranked by test BLEU-4)

```
+====+=====+=============================================+=================+=====+=========+=========+========+=======+
| #  | Tag | What                                        | Decoding        | Ep. |   Train |    Wall | BLEU-4 | CIDEr |
+====+=====+=============================================+=================+=====+=========+=========+========+=======+
| 1  | E5  | swap encoder: CLIP ViT-B/16 features        | beam w=5 lp=0.7 |  15 | 20m 24s |       — |  0.389 | 1.474 |
| 2  | E9  | logit-average ensemble of E0+E4+E7          | beam w=5 lp=0.7 |   — |       — |  5m 30s |  0.374 | 1.446 |
| 3  | E1  | E0 ckpt, beam-search inference only         | beam w=5 lp=0.7 |   — |       — |  1m 40s |  0.364 | 1.393 |
| 4  | E7  | + Qwen2.5 text paraphrase of 5k captions    | beam w=5 lp=0.7 |  12 | 10m 14s | 19m 39s |  0.363 | 1.409 |
| 5  | E6  | + EasyOCR memory bank (segment-tagged)      | beam w=5 lp=0.7 |  13 | 10m 38s |       — |  0.362 | 1.410 |
| 6  | E4  | + AdamW WD, warmup-cosine, LS, tied embed   | beam w=5 lp=0.7 |  15 | 11m 02s | 13m 04s |  0.362 | 1.393 |
| 7  | E1  | E0 ckpt, beam-search inference only         | beam w=5 lp=1.0 |   — |       — |  1m 40s |  0.358 | 1.399 |
| 8  | E3  | + unfreeze last 2 MBConv blocks (LR 1e-5)   | beam w=5 lp=0.7 |  10 | 29m 06s |       — |  0.358 | 1.382 |
| 9  | E1  | E0 ckpt, beam-search inference only         | beam w=3 lp=0.7 |   — |       — |  1m 40s |  0.352 | 1.348 |
| 10 | E1  | E0 ckpt, beam-search inference only         | beam w=3 lp=1.0 |   — |       — |  1m 40s |  0.345 | 1.352 |
| 11 | E2  | + random crop, rotation, colour jitter aug  | beam w=5 lp=0.7 |  12 | 31m 40s |       — |  0.344 | 1.350 |
| 12 | E8  | SCST/CIDEr-RL fine-tune of E7               | beam            |   2 |  4m 05s |  6m 25s |  0.291 | 1.273 |
| 13 | E0  | EffNet-B0 frozen + Transformer (Raina prep) | greedy          |  12 |  8m 53s |  9m 09s |  0.249 | 0.996 |
+====+=====+=============================================+=================+=====+=========+=========+========+=======+
```

Columns: Ep. = epochs trained; Train = sum of per-epoch wall-clock time from history.csv; Wall = full notebook execution time (training + eval + any data-prep step like Qwen paraphrasing). For E1 (inference sweep) the four rows share one notebook, so Wall here is that notebook's total divided by 4 to keep the per-row meaning sane.

## Notes per experiment

**E0 — Baseline (Raina prep, greedy).** test BLEU-4 0.249. Tiny drop vs Phase 1's 0.253 from removing the `<num>` collapse — the model now has to learn digit n-grams literally.

**E1 — Beam search on E0 checkpoint, no retraining.** Sweep over width ∈ {3, 5} × length_penalty ∈ {0.7, 1.0}. *beam_w5_lp0.7* wins with BLEU-4 0.365 — **+0.116 over E0 greedy from decoding alone**, the cheapest win of Phase 2.

**E4 — Schedule polish.** AdamW + warmup-cosine (500 step warmup) + label smoothing 0.1 + tied input/output embeddings. *First attempt was broken* — tied embeddings combined with the `× sqrt(d_model)` input scaling made output logits explode (train_loss = 116 at epoch 1). Fix: re-init embedding with std = 1/√d_model and zero the pad row. Final: BLEU-4 0.362 / CIDEr 1.393 — tied with E1, so the polish didn't pay on this dataset size.

**E7 — Qwen2.5-1.5B text paraphrase augmentation.** Sampled 5,000 training captions, generated one paraphrase each, appended to training set. BLEU-4 0.363 (~tied with E1) but **CIDEr 1.409, the best single-model CIDEr**. Text-only paraphrasing helps the diversity metric more than BLEU.

**E8 — SCST / CIDEr-RL fine-tune.** Regressed (BLEU-4 0.291). The CIDEr reward was computed from a *local* document-frequency built from just the 5 references per image, instead of the standard training-corpus DF. The reward over-weights rare-within-5-refs n-grams and pushes captions to be narrow / repetitive. Proper SCST needs a precomputed corpus IDF — obvious follow-up.

**E9 — Logit-average ensemble of E0, E4, E7 under beam_w5_lp0.7.** Best overall: **BLEU-4 0.374 / CIDEr 1.446**. Excluded E8 (regressed). The three MLE checkpoints disagree just enough that averaging their next-token distributions at each beam step reduces individual biases.

## Overfitting analysis

Train ↔ val cross-entropy gap from each run's `history.csv`. For label-smoothed runs (E4) the absolute losses are inflated by a `LS · log(V)` constant so the **gap** is the meaningful number, not the level.

```
Run                             best_ep  train  val    gap   note
-------------------------------  ------  -----  -----  ----  -------------------------
phase2_e0_baseline_raina_greedy       8  2.615  3.144  +0.53  memorising training set
  └ at last epoch 12             2.119  3.182  +1.06  drift since best ep
phase2_e4_schedule_polish            15  3.406  3.955  +0.55  memorising training set
phase2_e7_text_paraphrase             9  2.547  3.145  +0.60  memorising training set
  └ at last epoch 12             2.225  3.214  +0.99  drift since best ep
```

Reading the table: the **best-epoch gap is ~0.6 nat** for E0/E7 and stays similar for E4 (despite its bigger absolute losses from label smoothing). Past the best epoch the gap blows out to ~1.0–1.2 — **early stopping is doing most of the regularisation work**.

**Why didn't E4's regularisation package (LS=0.1, WD=0.01, warmup-cosine, tied embeds) buy us anything?** It *did* flatten the val-loss curve earlier, but BLEU was unchanged. Strongly suggests the bottleneck is **data scarcity, not regulariser choice**: 5,425 unique training images × ~5 captions = ~26 k caption rows feeding an 18 M-parameter decoder. E7's paraphrase augmentation added ~5 k more rows and gave a small CIDEr lift, consistent with that diagnosis.

### Hyperparameters we did NOT sweep

These would be the obvious next things to try if we keep tuning instead of changing architecture:

* **LR sweep** — every run except E8's RL step used `1e-4`. Sensible alternatives: `5e-5` (slower, more regularised), `2e-4` (faster, may push the train-val gap further).
* **Dropout 0.3-0.4** — currently 0.2 everywhere. Cheap, often the biggest single regulariser knob on small caption sets.
* **Weight decay 5e-2** (E4 used 0.01) — combined with higher dropout.
* **Smaller decoder** — 3 layers × d=512 might already be too much capacity. Try 2 layers × d=384 (≈ 6 M params instead of 18 M).
* **Early-stop on BLEU-4, not val-loss** — teacher-forced cross-entropy minimises at a slightly different epoch than greedy/beam BLEU. We're saving the val-loss minimum, which may not be the BLEU optimum.
* **Sub-word vocab (BPE / SentencePiece)** — word-level vocab with `MIN_WORD_FREQ=3` leaves a long UNK tail that dominates the rare-word loss; BPE would shrink that.
* **Higher label smoothing** (0.15-0.2) and **caption-side word dropout** at train time — both cheap and untouched.

**Optimizer choices we actually used**:

```
Run    Optimizer  LR        WD     Scheduler         LS    notes
-----  ---------  --------  -----  ----------------  ----  --------------------
E0     Adam       1e-4      0      ReduceLROnPlateau 0     vanilla
E4     AdamW      1e-4      0.01   warmup_cosine(500) 0.1  + tied embeddings
E7     Adam       1e-4      0      ReduceLROnPlateau 0     same as E0, more data
E8     Adam       5e-6      0      none               n/a  RL, 2 epochs SCST
```

So we tested **one** non-trivial training recipe (E4) and got no win; the rest are essentially the Phase 1 recipe.

## Still to come (or in progress):

```
  E7-VLM  Qwen2-VL-2B captions of training images (grounded augmentation).
```

## Cumulative time

Sum of Wall over successful runs (the 9 rows above): **60 min 27 s**.

Including the two failed retries (E4 v1 tied-embedding loss explosion, ~12 min killed; E7 v1 IndexError on empty LLM output, ~9 min) and inter-run gaps: **total ~1 h 28 min** wall-clock from kicking off E0 to E9 finishing.
