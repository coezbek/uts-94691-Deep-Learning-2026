# AT3 — Personal exploration archive

Working artefacts from UTS 94691 Assignment 3 (Image Captioning on
VizWiz-Captions, May 2026). The official group submission lives in
[`Rainbow6174/DeepLearning_AT3_workload`](https://github.com/Rainbow6174/DeepLearning_AT3_workload);
this folder is just my personal trail of exploration that didn't make it
into the submission notebook.

## Contents

- **`02_26239780_christopher_ozbek.py`** — copy of my consolidated submission
  notebook (Model 1: EfficientNet-B0 + 3-layer Transformer; Model 2:
  SigLIP2 ViT-B/16-256 + 2-layer small regularised decoder + beam tuning).
  The executed `.ipynb` with cell outputs lives in the group repo.

- **`01_shared_data_preparation.co_backup.{py,ipynb}`** — my own first cut
  of the shared data-prep notebook (with a `<num>` token collapse). The
  group adopted Raina's cleaner version instead, so this is just a
  reference for the difference.

- **`exploration/`** — every per-experiment script I ran from E0 (greedy
  baseline reproduction) through E15 (SigLIP2 + small decoder), plus the
  seed-replicate runs (E15-s2, E15-s3), the beam-decoding sweeps, and the
  Qwen3.5-9B zero-shot comparison script. None of these are part of the
  submission; they are the scaffolding behind the numbers in the
  comparison table and the Phase 3 discussion of the report.

- **`summaries/`** — running summary documents I kept while orchestrating
  the experiments:
  - `phase2_summary_canonical.md` — the headline Phase 2 result table the
    group used as a reference.
  - `phase3_summary.md` — running log of E10–E15 + sweep results, with
    per-experiment notes on what worked and what didn't.
  - `phase2_summary_version2.md`, `phase2_summary2.md`, `research-loop.md`
    — a research-paper-style consolidated summary and a meta-document on
    how the long-running experiment loop was orchestrated.

- **`helpers/`** — orchestration / figure-rendering scripts. Includes the
  scripts that produced the two figures in the report, the multi-experiment
  chain runner, and the auto-generator that built the per-experiment
  `phase2_e*.ipynb` files in `exploration/`.

- **`figures/`** — `figure_1_dataset_samples.png` (2×2 sample gallery with
  reference captions) and `figure_2_beam_sweep.png` (beam-width sweep on
  Model 2). Both used in the group report.

## What's NOT in this folder

- The trained checkpoints (`*.pt`), feature caches (`efficientnet_b0_raw_features.pt`,
  `clip_vitb16_features.pt`, `siglip2_b16_features.pt`) — ~6 GB total,
  rebuildable from the submission notebook.
- The VizWiz dataset images and annotations — downloaded by
  `01_shared_data_preparation.ipynb` on first run.
- The `output/phase2_results/` per-experiment outputs (`metrics.json`,
  `predictions.csv`, `history.csv`, `best.pt`, …) — ~few hundred MB,
  derivable from `exploration/` if needed.

## Tooling note

The work in this folder was orchestrated with substantial support from
Claude 4.7 (Anthropic), including the per-experiment design,
docstring / comment writing, and the long-running experiment loop that
produced the Phase 3 results.
