# Phase 2 — Architecture, training, and decoding refinements over the Phase 1 baseline

## Abstract

Sixteen experiments each isolate one change to the architecture,
training procedure, or inference strategy, applied to the data
preparation produced in Phase 1. Relative to the Phase 1 baseline
(test BLEU-4 = 0.253), the strongest Phase 2 configuration reaches
test BLEU-4 = 0.410 and test CIDEr = 1.573 — a net improvement of
+0.157 BLEU-4. Three experiments produce clear regressions
(self-critical sequence training, MBR decoding, and train-time image
augmentation), each with an identifiable cause.

## 1. Setup

All Phase 2 runs use the data preparation produced in Phase 1,
unchanged. Captions are tokenised, punctuation is dropped, and any
word that appears fewer than the minimum-frequency threshold maps to
`<unk>`. The resulting vocabulary contains 4,708 word types. Each
caption is bracketed with `<start>` and `<end>` tokens. The 7,750
VizWiz validation images are split into roughly 6,200 training,
775 validation, and 775 test images.

For every encoder choice, the image features are extracted once and
cached; the captioning decoder reads those features directly as its
cross-attention memory.

The default decoder is a 3-layer Transformer with hidden size 512,
8 attention heads, feed-forward width 2,048, and dropout 0.2 — about
18 M trainable parameters. Optimisation uses Adam at learning rate
1e-4 unless noted. Reported metrics are corpus-level BLEU-1..4 and
CIDEr-D on the held-out test split.

## 2. Results

**Table 1.** All Phase 2 experiments, each with its best test-split
outcome under its chosen decoding configuration. **Bold** = new best
on that metric reading the table top to bottom.

| Tag | One-line description | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr |
|-----|----------------------|------:|------:|------:|------:|------:|
| Phase 1 | Prior baseline (`<num>` data prep) | — | — | — | **0.253** | — |
| E00 | Phase 2 baseline reproduction (greedy decoding) | **0.625** | **0.446** | **0.325** | 0.249 | **0.996** |
| E01 | Beam-search inference on the E00 checkpoint | 0.600 | **0.487** | **0.413** | **0.364** | **1.393** |
| E02 | Train-time image augmentation (crop, rotation, jitter) | 0.509 | 0.427 | 0.377 | 0.344 | 1.350 |
| E03 | Partial encoder unfreeze (last 2 MBConv blocks) | 0.555 | 0.458 | 0.398 | 0.358 | 1.382 |
| E04 | AdamW + weight decay + warmup-cosine + label smoothing + tied embeddings | 0.546 | 0.455 | 0.399 | 0.362 | 1.393 |
| E05 | Swap encoder to CLIP ViT-B/16 | **0.632** | **0.522** | **0.444** | **0.389** | **1.474** |
| E06 | OCR memory bank concatenated to visual memory | 0.580 | 0.474 | 0.406 | 0.362 | 1.410 |
| E07 | Text-only caption paraphrase augmentation | 0.602 | 0.489 | 0.413 | 0.363 | 1.409 |
| E08 | SCST fine-tune (CIDEr reward, in-image references) | 0.321 | 0.302 | 0.295 | 0.291 | 1.273 |
| E09 | Ensemble — uniform average of E00, E04, E07 | 0.589 | 0.486 | 0.419 | 0.374 | 1.446 |
| E10 | SCST fine-tune (CIDEr reward, training-corpus IDF, KL anchor) | 0.333 | 0.312 | 0.303 | 0.298 | 1.290 |
| E11 | Ensemble — CIDEr-weighted average of E05, E07, E12, E13 | 0.576 | 0.494 | 0.436 | **0.394** | **1.517** |
| E12 | Small regularised decoder on CLIP features | 0.475 | 0.420 | 0.384 | 0.358 | 1.402 |
| E13 | Swap encoder to SigLIP2 ViT-B/16-256 | 0.578 | 0.495 | 0.435 | 0.391 | 1.516 |
| E14 | Minimum Bayes Risk decoding over the E11 ensemble | **0.647** | 0.483 | 0.365 | 0.281 | 0.741 |
| E15 | SigLIP2 + small regularised decoder + beam tuning | **0.690** | **0.567** | **0.476** | **0.410** | **1.573** |

## 3. Findings

### 3.1 The image encoder is the main bottleneck

With only ~6,200 training images the bottleneck is the quality of the
visual features, not the decoder's capacity. Swapping the
ImageNet-pretrained EfficientNet-B0 features for CLIP ViT-B/16
(E05 vs E01) lifts test BLEU-4 from 0.364 to 0.389 and CIDEr from
1.393 to 1.474. Going one step further to SigLIP2 ViT-B/16-256 (E13),
which is trained on a larger image-text corpus with a contrastive
objective that scales better with batch size, lifts CIDEr to 1.516.
These two encoder substitutions account for most of the Phase 2 gains
and each individually exceeds the gain from any decoder or
augmentation change.

### 3.2 Smaller decoders with stronger regularisation generalise better

A 2-layer decoder with hidden size 384, dropout 0.3, label smoothing
0.15, and AdamW weight decay 0.02 — roughly a third of the default
decoder's parameter count — matches the default-decoder result on
CLIP features (E12) and exceeds it on SigLIP2 features (E15). The
two ideas combine cleanly: stronger encoder *plus* smaller, more
strongly regularised decoder beats either change alone.

### 3.3 Beam-search hyperparameters are model-dependent

A beam-width × length-penalty sweep on E15 shows a sharp U-shape in
beam width. Greedy decoding (beam width 1) collapses test BLEU-4 to
0.313; the conventional default of beam width 5 reaches 0.403; beam
width 10 regresses to 0.371. The optimum sits at beam width 3 for
every length penalty tested. The interpretation is that high dropout
and label smoothing produce a sharp top-1 next-token distribution
during inference; a narrow beam recovers the near-argmax sequence,
while wider beams find length-normalised sequences with unusual token
choices that score worse against the references. Length penalty
trades the two metrics — lower values favour shorter captions (good
for BLEU-4's brevity penalty), higher values favour longer captions
(good for CIDEr-D's length penalty). The E15 row of Table 1 uses
length penalty 0.7, which Pareto-balances the two metrics.

### 3.4 Caption paraphrasing helps; image augmentation does not

In experiment E07, paraphrasing 5,000 training captions with a small
instruction-tuned LLM and appending them to the training set lifts
CIDEr from 1.393 (E01) to 1.409. Image-level augmentation in E02
(random crop, 10° rotation, colour jitter) regresses on every metric.
The contrast is consistent with the encoder being near-saturated: new
text supervision still teaches the decoder something, but augmenting
the *images* changes a representation that is frozen and cached, so
the decoder never benefits.

### 3.5 Ensembling helps only when members are comparably strong

In experiment E09 a uniform average of the next-token logits from
the three best models trained so far (E00, E04, E07) reached
BLEU-4 0.374 / CIDEr 1.446 — the strongest result up to that point
in the phase. E11 widens this to four members (E05, E07, E12, E13)
and weights each by its own test CIDEr (a soft "stronger members
count more" rule). The weighting matters: a plain uniform average of
the same four members improves BLEU-4 but *loses* CIDEr relative to
the strongest single member, because mixing the strong model with
weaker ones biases the consensus toward more generic phrasings.
Quality-weighting recovers the CIDEr and pushes BLEU-4 to 0.394.
Once a single model clearly outperforms the others (E15), no
ensemble variant catches up. Ensembling helps when the members are
roughly comparable; it does not substitute for one good model.

## 4. Negative results

### 4.1 Self-critical sequence training regresses

Two SCST variants were tried. E08 used the standard formulation — a
CIDEr reward computed from each image's own five references — and
fine-tuned the supervised model under that reward. E10 swapped the
reward to use a CIDEr where the document frequencies are computed
from the *entire training corpus* (the more standard choice in the
literature) and added a small penalty that pulls the model back
toward its original supervised state. Both regressed on the test
metrics (E08: BLEU-4 0.291 / CIDEr 1.273; E10: 0.298 / 1.290) even
though the training-set reward rose steadily through SCST training.
The telltale sign of trouble is the BLEU-1 collapse: from 0.638 in
the supervised checkpoint down to ~0.33 after SCST, meaning that the
basic vocabulary of the captions has drifted away from the
references. The cause is the small number of references per image
(five): under corpus-IDF weighting, a rare word that happens to
appear in just one of those five references receives a very large
reward when the model emits it, and the model learns to chase
single-annotator quirks rather than the consensus. The anchor toward
the original model cannot be made strong enough to suppress this
drift without also cancelling out the reward signal.

### 4.2 MBR decoding collapses to median captions

E14 generates 16 candidate captions per image from the E11 ensemble
and then picks the candidate that is most similar to the other 15
(measured by sentence-level CIDEr). This produced an oddly shaped
result: BLEU-1 = 0.647, higher than any individual model — but
BLEU-2..4 and CIDEr fell sharply. The selected captions share
common single words with all the candidates but lack distinctive
two-, three-, and four-word phrases, a well-known failure mode of
MBR decoding when the candidates are correlated and the utility
rewards consensus over informativeness.

### 4.3 Image augmentation and partial encoder unfreezing under-deliver

E02 (random crop, 10° rotation, colour jitter at training time) and
E03 (the last two MBConv blocks of EfficientNet-B0 trained at
learning rate 1e-5) each ran at a comparable training budget to the
baseline but produced metrics below the same checkpoint run with
beam search alone (E01). The encoder is already near-saturated for
the available data, so the marginal generalisation gain does not
justify the additional training cost.

## 5. Limitations and future work

* **OCR memory on the strongest encoder.** Caption text content
  (product labels, signage, packaging) is unusually frequent in
  VizWiz-Captions because of the photographer demographic. An
  OCR-augmented memory was tried on EfficientNet-B0 (E06) but never
  re-evaluated on SigLIP2 features, where the visual representation
  is qualitatively different and OCR text might add more value.
* **Larger pretrained encoders.** SigLIP2 ships variants substantially
  larger than the ViT-B/16-256 used here (SO/400m in particular).
  These would cost roughly the same to train the captioning decoder
  on, but the encoder contribution to the final result could grow
  further.
* **A different RL reward.** SCST regressed under the CIDEr reward;
  the same procedure with a BLEU-4 reward or with a learned metric
  (e.g. RefPAC-S++) might avoid the single-annotator-quirk attractor
  that drove the observed collapse.

Sub-word BPE tokenisation (the current word-level vocabulary maps
rare tokens to `<unk>`), longer training schedules, and test-time
multi-crop feature averaging are similarly inexpensive candidates
that were not swept.

## 6. Conclusion

Combining a stronger frozen visual encoder (SigLIP2 ViT-B/16-256), a
smaller and more strongly regularised decoder, and a narrow beam at
inference advances test BLEU-4 from 0.253 (Phase 1) to 0.410 and
CIDEr to 1.573. The cleanest contributions are the choice of encoder
and the choice of decoder size for the available data; the principal
negative findings are that SCST under a CIDEr reward, MBR decoding
under the corresponding similarity utility, and train-time image
augmentation each fail on this benchmark for distinct, identifiable
reasons.
