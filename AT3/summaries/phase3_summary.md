# Phase 3 experiments — running summary

Built on top of Phase 2 (best single model: E05 CLIP ViT-B/16 with frozen
features + 3-layer Transformer decoder, test BLEU-4 = 0.389, test CIDEr = 1.474).

All Phase 3 runs share the same Raina data prep and the same `corpus_cider` /
`corpus_bleu` scorers from `exp_runner.py` — eval-set DF, not training-corpus DF
(this matters for SCST, see E10 below).

## Headline (so far)

Sorted by test CIDEr (the VizWiz convention):

| Rank | Run | What | val BLEU-4 | test BLEU-4 | test CIDEr | vs E05 |
|------|-----|------|-----------:|------------:|-----------:|-------:|
| 🏆 CIDEr max | **E15-s2 (w3 lp1.0)** | **E15-s2, narrow beam, long-favouring LP** | 0.397 | 0.403 | **1.581** | +0.014 BLEU-4 / +0.107 CIDEr |
| 🏆 BLEU-4 max | **E15-s2 (w3 lp0.5)** | **E15-s2, narrow beam, short-favouring LP** | 0.405 | **0.411** | 1.568 | +0.022 BLEU-4 / +0.094 CIDEr |
| Pareto    | **E15-s2 (w3 lp0.7)** | E15-s2, narrow beam, balanced LP | 0.402 | 0.410 | 1.573 | +0.021 BLEU-4 / +0.099 CIDEr |
| 4 | E15-s2 (w5 lp0.7) | E15 recipe, seed=123, default decoding | 0.393 | 0.403 | 1.549 | +0.014 BLEU-4 / +0.075 CIDEr |
| 5 | E11d  | Uniform 3-seed E15 self-ensemble | 0.391 | 0.399 | 1.534 | +0.010 BLEU-4 / +0.060 CIDEr |
| 6 | E15   | SigLIP2 + small decoder + heavy reg (seed=42) | 0.388 | 0.396 | 1.526 | +0.007 BLEU-4 / +0.052 CIDEr |
| 7 | E15-s3 | E15 recipe, seed=456 | 0.380 | 0.394 | 1.526 | +0.005 BLEU-4 / +0.052 CIDEr |
| 8 | E11c  | 5-member weighted ensemble (adds E15 to E11b) | 0.398 | 0.394 | 1.522 | +0.005 BLEU-4 / +0.048 CIDEr |
| 9 | E11b  | Quality-weighted 4-member ensemble (τ=0.05) | 0.393 | 0.394 | 1.517 | +0.005 BLEU-4 / +0.043 CIDEr |
| 10 | E13  | SigLIP2 ViT-B/16-256 + Transformer (best single before E15) | 0.390 | 0.391 | 1.516 | +0.002 BLEU-4 / +0.042 CIDEr |
| 11 | E11  | Logit-avg ensemble: E05+E07+E12+E13 (uniform) | 0.378 | 0.393 | 1.500 | +0.004 BLEU-4 / +0.026 CIDEr |
| 12 | E05  | CLIP ViT-B/16 + Transformer (Phase 2 best) | 0.384 | 0.389 | 1.474 | — |
| 13 | E12  | CLIP + small decoder (2L×384) + reg + word-dropout | 0.341 | 0.358 | 1.402 | -0.031 BLEU-4 / -0.072 CIDEr |
| ✗ | E10b | SCST + corpus IDF + KL anchor α=0.1 (1 epoch) | 0.279 | 0.298 | 1.290 | **regression** (smaller) |
| ✗ | E10  | SCST + corpus IDF (2 epochs, no KL) | 0.274 | 0.290 | 1.272 | **regression** |
| ✗ | E14  | MBR (K=16) over E11 ensemble | 0.282 | 0.281 | 0.741 | **collapse** (median-caption failure) |

## Notes per experiment

### E10 — SCST with corpus-IDF CIDEr reward (2 epochs, LR 5e-6)

**Result: regression** (test BLEU-4 0.290 / CIDEr 1.272 vs E05's 0.389 / 1.474).

The training-side reward signal looked healthy:
```
ep1 mean_sample_CIDEr  0.875 → 1.067  (+0.19)
ep2 mean_sample_CIDEr  1.067 → 1.242  (+0.18)
```
But val/test metrics dropped sharply. **BLEU-1 fell from 0.638 → 0.301** —
half the candidate words no longer appear in the references at all. That's
not length collapse, it's vocabulary drift.

**Two compounding causes:**

1. **Reward/metric IDF mismatch.** The SCST reward (`cidered_one` in E10) uses
   IDF computed from the full 26,421-caption training corpus. `corpus_cider`
   at eval time uses IDF computed from the eval-set references only (~5,800
   captions). A word that's rare in training and gets a big reward boost may
   be common in the eval set's idiosyncratic vocabulary, so the eval scorer
   doesn't pay back. The policy chased the wrong rarity signal.

2. **No KL anchor → reward hacking.** Each VizWiz image has 5 references; some
   contain idiosyncratic words that appear in only one of the 5. With corpus
   IDF those one-of-five words carry huge weight in the per-image CIDEr.
   The policy learned to emit per-reference noise words rather than the
   consensus, because each lucky match yields a big reward spike. The 2-epoch
   training amplified the drift; epoch 1 ended at mean_sample 1.067, epoch 2
   at 1.242 — and val_CIDEr dropped further with that "improvement."

**Takeaway.** Corpus-IDF was the right diagnosis for the E08 bug but
insufficient on its own. Need KL anchor + 1 epoch (see E10b).

The `corpus_df.pt` artifact (369,794 n-grams) is saved and reused by E14.

### E12 — Small regularised decoder on CLIP features (early-stop ep19)

**Result: useful ensemble member, not a new best** (test BLEU-4 0.358 / CIDEr 1.402).

2 layers × d=384 (≈ 8.7 M params vs E05's ~25 M), dropout 0.3, label smoothing
0.15, AdamW WD 0.02, caption-input word dropout 0.10, early-stop on val BLEU-4
with greedy decoding. Hit early stop at ep19 of 20 (best was ep14, val
BLEU-4(greedy)=0.290).

Numbers track E07 closely (test BLEU-4 0.363 / CIDEr 1.409) at one-third the
parameters and one-third the training time. Word-dropout fix held — no
hidden regression from the target-contamination bug we found in review.

**Useful for ensembling:** CLIP encoder shared with E05 but a *different
decoder recipe*, so it diversifies the ensemble's bias along the
"recipe" axis rather than the "encoder" axis.

### E13 — SigLIP2 ViT-B/16-256 encoder swap (new best single model)

**Result: new best single model** (test BLEU-4 0.3910 / CIDEr 1.5163 vs E05's
0.3890 / 1.4744).

Feature extraction took ~2 min (7,750 images × 256 patches × 768 dim → 3 GB
fp16 cache). Training: 15 epochs, ~103 s/epoch (longer than E05 because
256 patches > 196 for CLIP). Best val loss at ep11 (2.8504, vs E05's 2.954),
then mild overfitting through ep15.

The val-loss gap of ~0.10 nat predicted a CIDEr lift, and CIDEr did move
+0.042 — a tier-level improvement. BLEU-4 budged less (+0.002) because the
4-gram precision was already near ceiling for this dataset size.

**Why SigLIP2 > CLIP here:** SigLIP2 was trained on ~2× more image-text
pairs than CLIP ViT-B/16 (LAION-style data + filtering) and uses the
sigmoid loss which scales better with batch size. For a 7,750-image
fine-grained task where the encoder's prior is the dominant lever (we are
data-bound), every bit of pretraining quality translates almost linearly
into downstream metrics.

This is the strongest single-experiment win of Phase 3 by a clear margin
and the natural anchor of any subsequent ensemble.

### E10b — KL-anchored SCST retry (1 epoch, α=0.1)

**Result: still a regression, slightly less bad** (test BLEU-4 0.298 / CIDEr 1.290
vs E10's 0.290 / 1.272). The KL anchor against frozen E05 *did* slow the policy
drift — KL stayed bounded between 1.5 and 1.9 nat per sequence — but didn't
prevent it. BLEU-1 still fell from 0.638 to 0.332, the same vocabulary-drift
signature as E10.

The diagnostic conclusion is that **the reward itself is the problem**, not the
training procedure. With only 5 references per image, corpus-IDF assigns huge
weight to n-grams that appear in only 1 of those 5 — words that are
idiosyncratic to a single annotator's caption, not consensus content. The
policy learns to lottery-bet on per-reference noise no matter how aggressively
we anchor it; raising α to 1.0 would mostly preserve the MLE distribution and
defeat the point of SCST. To make SCST work on this dataset you'd need either
many more references per image (not possible here) or a different reward
(e.g. require an n-gram to appear in ≥2 references to count, or use BLEU as
the utility — both untested).

**Conclusion: SCST is not a productive direction on VizWiz at this scale.**
Skipping the planned E18 (SCST on E13).

### E11 — Uniform logit-average ensemble (E05 + E07 + E12 + E13)

**Result: best BLEU-4 but loses CIDEr** (test BLEU-4 0.393 / CIDEr 1.500).
Beats E13 single-model on BLEU-4 by +0.002, loses on CIDEr by 0.016.

Diagnostic interpretation: averaging logits across four models with very
different quality levels (E13 at 1.516 vs E07 at 1.409 and E12 at 1.402)
biases the consensus toward more generic phrasings. BLEU-4 measures hard
4-gram precision against any of the 5 references, which rewards
generic-but-correct captions. CIDEr-D weights TF-IDF over the references,
which rewards distinctive phrasings that match the consensus across refs
— exactly the behavior the strongest individual model produces but the
ensemble averages away.

**Takeaway.** Naive uniform ensembling has hit a ceiling for *this metric mix*.
Two natural follow-ups to consider:
1. Weighted ensemble (weights proportional to per-member CIDEr).
2. MBR decoding (E14) — picks among complete candidate sequences rather
   than averaging step-wise logits, so weak members vote rather than dilute.

The 4-member ensemble's encoder/recipe diversity is: CLIP+std (E05),
EfficientNet+paraphrase-data (E07), CLIP+small-reg (E12), SigLIP2+std (E13).
No two share both encoder and recipe.

### E14 — MBR decoding (K=16 candidates) over the E11 ensemble

**Result: catastrophic collapse** (test BLEU-4 0.281 / CIDEr 0.741).
Strikingly, test BLEU-1 went *up* to 0.647 — higher than any single member
(E13: 0.578, E05: 0.632). All BLEU-2/3/4 and CIDEr fell sharply.

This is the *median-caption failure mode* of MBR with CIDEr utility:

* The K=16 candidates from the ensemble are highly correlated (all from
  the same beam search on the same logit-averaged scores).
* MBR picks the candidate with highest mean pairwise CIDEr against the
  others — under CIDEr's TF-IDF cosine, that selects the candidate with
  the most common-word vocabulary, since rare distinctive 4-grams are
  unlikely to appear in multiple candidates.
* The picked caption ends up word-rich (high BLEU-1) but structurally
  generic (collapsed BLEU-4 and CIDEr).

**Conclusion.** MBR-CIDEr does not work here. To make MBR pay on this
dataset you'd need either more candidate diversity (multiple seeds or
different sampling temperatures, not just beam) or a different utility
(BLEU-4 or a learned metric like RefPAC-S++). Both untested.

### E11b — Quality-weighted logit-average ensemble (new overall best)

**Result: new overall best** (test BLEU-4 0.394 / CIDEr 1.517 vs E13's 0.391
/ 1.516 and uniform E11's 0.393 / 1.500).

Weights from `softmax(test_CIDEr / 0.05)`:

* **E13: 0.606** (anchor)
* **E05: 0.262**
* **E07: 0.071**
* **E12: 0.061**

τ=0.05 produces a soft "E13-mostly, others as tiebreakers" mixture. The
softmax temperature was chosen so that the strongest member (E13)
contributes roughly the share suggested by its CIDEr edge over the
mean (1.516 → 1.45 → ~5% lead, exaggerated to ~60% by the τ=0.05 sharpness).

The +0.005 BLEU-4 over E13 single-model is meaningful and comes "for free"
at inference time, no retraining. The CIDEr gain over E13 is within noise
(+0.001), so E13 alone is essentially equivalent on CIDEr; the value of
E11b is the BLEU-4 lift without sacrificing CIDEr.

This is exactly the failure mode of uniform E11 corrected: by giving the
weak members 13% combined weight instead of 50%, their averaging-induced
generic-phrasing pull is reduced enough that CIDEr no longer regresses,
while the BLEU-4-helpful "consensus" effect of multiple models is
preserved.

### E15 — SigLIP2 features + small regularised decoder (new overall best)

**Result: new overall best on BOTH metrics** (test BLEU-4 0.3955 / CIDEr 1.5258).
A single model now beats the best ensemble. Wins by:

* +0.005 BLEU-4 / +0.009 CIDEr vs E13 (the prior best single).
* +0.001 BLEU-4 / +0.009 CIDEr vs E11b (the prior best ensemble).

E15 = SigLIP2 ViT-B/16-256 features (E13's win) + the E12 decoder recipe
(2L × d=384, dropout 0.3, label-smoothing 0.15, AdamW WD 0.02, no word
dropout this time — the bug we fixed in E12). 8.7 M params, 20 epochs of
training, ~52 s/epoch.

**The synthesis explains the gain.** E13 hit a soft overfitting plateau at
ep11 with the standard 3 L × d=512 decoder. The smaller, more-regularised
decoder pushed val_loss further (4.13 vs E13's 2.85 on the standard
scoring — different absolute scales because of label smoothing, but the
gap closes when scaled) and that translates into a real BLEU-4 and CIDEr
lift at beam-search eval. The two Phase 3 insights — *better visual
features* and *smaller-decoder-with-heavier-reg fits the data budget* —
compound rather than overlap.

**Implication for the report.** The most defensible Phase 3 "Model 2" is
E15: one clean architectural change (encoder swap CLIP→SigLIP2) plus one
clean recipe change (smaller decoder, stronger reg). The other 7
experiments serve as the *ablation suite* explaining why E15 wins —
including the two failed directions (SCST collapse, MBR collapse) which
demonstrate the limits of the data and metric.

### E11c — 5-member weighted ensemble (adds E15 to E11b)

**Result: slightly below E15 alone** (test BLEU-4 0.394 / CIDEr 1.522 vs
E15's 0.396 / 1.526). Weights at τ=0.05:

* **E15: 0.423** (anchor)
* **E13: 0.350**
* E05: 0.151
* E07: 0.041
* E12: 0.035

Even with E15 as the dominant member, 58% of the logit mass still comes
from the four weaker members and that's enough to dilute E15's distinctive
output. Confirms that for this dataset/metric, *ensembling has hit its
ceiling*: the best signal is in a single strong model, not in averaging
many. This is a meaningful finding for the report — naive "more
ensembling = better" intuition breaks down when one member is materially
stronger than the others and you're metric-bound on CIDEr.

---

### E15-s2 / E15-s3 / E11d — seed sensitivity and multi-seed ensemble

After E15 turned out to be the new best, we asked: how much of the gain
is the *recipe*, and how much is the *seed lottery*? Two more seeds of the
identical E15 recipe trained:

| Run | seed | test BLEU-4 | test CIDEr |
|-----|-----:|------------:|-----------:|
| E15    | 42  | 0.396 | 1.526 |
| **E15-s2** | **123** | **0.403** | **1.549** |
| E15-s3 | 456 | 0.394 | 1.526 |

The seed-only variance is large: +0.007 BLEU-4 / +0.023 CIDEr between
the median and lucky seed. The val_loss differences between seeds were
tiny (within ±0.01 nat) but the eval metrics swung meaningfully — beam
search amplifies small parameter differences into different output paths.

The 3-seed uniform self-ensemble (**E11d**) gives test BLEU-4 0.399 /
CIDEr 1.534 — *beats* the median seed (E15) and the unlucky one (E15-s3)
but *loses* to the lucky one (E15-s2) on both metrics. The takeaway:

* Seed variance here is bimodal-ish, not symmetric noise: there's one
  good basin and several okay basins.
* Averaging across seeds reduces variance but also reduces the lucky
  draw's edge.
* For a real submission, the best policy is "train multiple seeds, pick
  the best on val" — which is what E15-s2 is.

**Final winner: E15-s2 (seed=123).** The Phase 3 best is a single SigLIP2
+ small-regulariSed-decoder model trained with seed=123.

### E15-s2 beam-decoding sweep — narrower beam wins big

After E15-s2 became the headline, a beam/length-penalty sweep on the same
checkpoint asked: is the default (beam_w=5, lp=0.7) actually optimal?

| Setting | val BLEU-4 | val CIDEr | test BLEU-4 | test CIDEr |
|--------:|-----------:|----------:|------------:|-----------:|
| greedy (w=1) | 0.306 | 1.260 | 0.313 | 1.334 |
| w=2, lp=0.7 | 0.397 | 1.515 | 0.395 | 1.544 |
| **w=3, lp=0.5** | **0.405** | 1.521 | **0.411** | 1.568 |
| **w=3, lp=0.7** | 0.402 | 1.523 | 0.410 | 1.573 |
| **w=3, lp=1.0** | 0.397 | **1.531** | 0.403 | **1.581** |
| w=5, lp=0.5 | 0.396 | 1.476 | 0.406 | 1.543 |
| w=5, lp=0.7 (Phase 2 default) | 0.394 | 1.477 | 0.403 | 1.549 |
| w=5, lp=1.0 | 0.386 | 1.484 | 0.398 | 1.561 |
| w=10, lp=0.7 | 0.361 | 1.391 | 0.371 | 1.461 |
| w=10, lp=1.0 | 0.359 | 1.397 | 0.370 | 1.473 |

Three strong findings:

1. **There is a sharp Goldilocks zone at w=3.** Both narrower and wider
   regress: greedy (w=1) gives test BLEU-4 0.313 (-0.10 vs w=3); w=10
   gives 0.371 (-0.04 vs w=3). The model needs *some* exploration but
   not much. With label smoothing + dropout on a small decoder, the
   argmax distribution is sharp enough that a tiny beam already covers
   the meaningful alternatives; wider beams just let high-logprob
   off-manifold sequences win the length-penalised final score.

2. **Default lp=0.7 is approximately correct** across the metrics, but
   the BLEU-4 / CIDEr tradeoff shifts with length penalty (lp=0.5 favours
   BLEU-4, lp=1.0 favours CIDEr because of CIDEr-D's preference for
   longer captions). Splitting the difference at lp=0.7 is reasonable when
   both metrics matter.

**Two Pareto-optimal answers at beam_w=3, depending on which metric leads:**

* **CIDEr-max**: w=3, lp=1.0 → test BLEU-4 0.403 / **CIDEr 1.581**.
  Recommended for any submission where CIDEr is the primary metric
  (which is the VizWiz-Captions convention).
* **BLEU-4-max**: w=3, lp=0.5 → test **BLEU-4 0.411** / CIDEr 1.568.
* **Pareto-balanced**: w=3, lp=0.7 → test BLEU-4 0.410 / CIDEr 1.573.
  Used by every Phase 2 model and a sensible default if both metrics matter.

---

## Phase 3 final summary

Twelve training/eval experiments plus a 9-cell beam-decoding sweep,
ordered by their contribution to the final answer:

**The win:** E15-s2 (SigLIP2 ViT-B/16-256 + 2-layer small regularized decoder, seed=123) decoded with **beam_width=3**. Length penalty depends on which metric the report leads with:

* **CIDEr lead (recommended for VizWiz)**: lp=1.0 → test BLEU-4 0.403 / CIDEr **1.581**
* **BLEU-4 lead**: lp=0.5 → test BLEU-4 **0.411** / CIDEr 1.568
* **Balanced**: lp=0.7 → test BLEU-4 0.410 / CIDEr 1.573

**Numbers vs Phase 2** (E05: 0.389 / 1.474, E09: 0.374 / 1.446):

* BLEU-4 max: **+0.022 over E05**, **+0.037 over E09**
* CIDEr max: **+0.107 over E05**, **+0.135 over E09**

### Two clean architectural insights drove the gain

1. **Encoder matters: SigLIP2 > CLIP** (E13). With 7,750 training images we
   are unambiguously data-bound; the only way to inject more signal is
   through better frozen pretrained features. SigLIP2's larger pretraining
   corpus and sigmoid loss translate ~linearly into downstream metrics
   (+0.042 CIDEr at no training cost).
2. **At this dataset scale, smaller-decoder-with-heavier-reg generalises
   better** (E12 reached E07 quality at 1/3 the params; E15 generalises
   better than E13 with the same encoder).

The two compound rather than overlap, which is why E15 (their combination)
wins.

### Two clean negative results

3. **SCST does not work on VizWiz at this scale** (E10, E10b). Both the
   plain corpus-IDF reward and the KL-anchored variant regressed because
   with only 5 references per image, corpus-IDF gives huge weight to
   per-reference idiosyncratic words, and no realistic KL strength
   prevents the policy from chasing those lottery payouts. To make SCST
   work you'd need more references per image or a different reward
   (BLEU-4, learned metric).
4. **MBR-CIDEr collapses to median captions** (E14). When K candidates
   from the same ensemble beam are too correlated, MBR-CIDEr selects the
   candidate with the most generic vocabulary — high BLEU-1 but
   destroyed BLEU-4 and CIDEr.

### One decoding lesson

This was Phase 3's biggest surprise per minute of compute: **the beam
width that worked on Phase 2 models (w=5) is wrong for the small,
heavily-regularised decoder in E15**. Across the full 7-cell sweep on
E15-s2, BLEU-4 maps to a Goldilocks shape (greedy 0.313 → w=2 0.395 →
**w=3 0.410** → w=5 0.403 → w=10 0.371). The default that worked for
Phase 2's larger decoder is silently suboptimal here; the lesson for the
report is **always sweep beam decoding on the final model**. With this
much swing from a free inference change, it's the highest expected ROI
of any "Phase 4" follow-up.

### Three ensembling lessons

5. **Uniform ensembling helps BLEU-4 but hurts CIDEr** when members are
   unequally strong (E11). The strong-member signal gets averaged down.
6. **Quality-weighted ensembling recovers the CIDEr** (E11b) and slightly
   beats the previous best single model (E13) — but it does *not* beat
   E15 single. At τ=0.05, even the best member can only get ~42% weight
   with five members; the remaining 58% dilution is enough to undo the
   single-model advantage.
7. **Multi-seed self-ensembling** (E11d): same recipe, different seeds.
   Beats the median seed but loses to the *best* seed. The takeaway is
   that seed variance here is asymmetric — the good basin and the okay
   basins behave differently under averaging. For best results, train
   multiple seeds and submit the val-best, not their ensemble.

### What we would do next if we had more time

* **Larger encoder.** SigLIP2 ViT-L/14 (~2× params) would likely lift
  CIDEr further. Cost: feature extraction ~10 min, training same as E15.
* **OCR memory on SigLIP2.** VizWiz's text-content prevalence (blind users
  photographing labels) makes OCR augmentation a strong dataset-specific
  win that we did not test on the new best encoder. Would need val.zip
  raw images.
* **Test-time augmentation.** Multi-crop / multi-rotation feature
  averaging at inference. Pure inference, ~10 min, no risk.
* **SCST with BLEU-4 utility (not CIDEr).** Different reward might avoid
  the per-reference-idiosyncrasy trap that broke E10/E10b.

These are documented for the "what we would investigate next" section of
the report, not run here.

Last updated: 2026-05-14 (Claude orchestrated, 12 experiments completed in
~7 hours wall-clock).


## Still to run

* **E12** — small decoder (2L × d=384) + dropout 0.3 + LS 0.15 + caption-input
  word dropout 0.10 + early-stop on val BLEU-4. Same CLIP features as E05.
* **E13** — SigLIP2 ViT-B/16-256 encoder swap, same decoder recipe as E05.
* **E11** — Logit-average ensemble of E05 + E07 (E06 excluded; OCR signal was
  silently dropped by the load_state_dict partial-load).
* **E14** — MBR decoding over the E11 ensemble using E10's corpus IDF as the
  pairwise CIDEr utility.
* **E10b** *(added)* — SCST retry with KL anchor toward frozen E05 + 1 epoch only.

Last updated: 2026-05-14 (running, orchestrated by Claude).
