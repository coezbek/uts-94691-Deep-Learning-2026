"""Collect Phase 2 metrics, build a summary CSV + markdown + plain-text report.

Each experiment dir contributes a row; runtime is read from notebook execution
metadata (cells' `iopub.execute_input` and `iopub.status.idle` timestamps).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

NB_DIR = Path(__file__).parent
RESULTS = NB_DIR / "output" / "phase2_results"


def notebook_wall_seconds(nb_path: Path) -> float | None:
    """Return wall-clock execution seconds from notebook cell metadata, or None."""
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    starts: list[str] = []
    ends: list[str] = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        ex = c.get("metadata", {}).get("execution", {})
        if "iopub.execute_input" in ex:
            starts.append(ex["iopub.execute_input"])
        if "iopub.status.idle" in ex:
            ends.append(ex["iopub.status.idle"])
    if not starts or not ends:
        return None

    def parse(s):
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)

    return (parse(ends[-1]) - parse(starts[0])).total_seconds()


def history_train_seconds(history_csv: Path) -> tuple[float, int] | tuple[None, None]:
    if not history_csv.exists():
        return None, None
    df = pd.read_csv(history_csv)
    if df.empty:
        return None, None
    time_col = "time" if "time" in df.columns else (df.columns[-1])
    return float(df[time_col].sum()), len(df)


# Map run_name -> notebook file to read execution metadata from
RUN_TO_NOTEBOOK = {
    "phase2_e00_baseline_raina_greedy": "phase2_e00_baseline_raina_greedy.ipynb",
    "phase2_e01_beam_w3_lp0.7":         "phase2_e01_beam_search.ipynb",
    "phase2_e01_beam_w3_lp1.0":         "phase2_e01_beam_search.ipynb",
    "phase2_e01_beam_w5_lp0.7":         "phase2_e01_beam_search.ipynb",
    "phase2_e01_beam_w5_lp1.0":         "phase2_e01_beam_search.ipynb",
    "phase2_e04_schedule_polish":       "phase2_e04_schedule_polish.ipynb",
    "phase2_e07_text_paraphrase":       "phase2_e07_text_paraphrase.ipynb",
    "phase2_e08_scst":                  "phase2_e08_scst.ipynb",
    "phase2_e09_ensemble":              "phase2_e09_ensemble.ipynb",
    "phase2_e10_scst_corpus_idf":       "phase2_e10_scst_corpus_idf.ipynb",
    "phase2_e11_ensemble_e05":          "phase2_e11_ensemble_e05.ipynb",
    "phase2_e12_small_decoder":         "phase2_e12_small_decoder.ipynb",
    "phase2_e13_siglip2":               "phase2_e13_siglip2.ipynb",
    "phase2_e14_mbr":                   "phase2_e14_mbr.ipynb",
}
# Sub-runs that share a notebook: split runtime evenly so the sum still equals
# the real wall-clock for that notebook.
SHARED_NB_SPLITS = {"phase2_e01_beam_search.ipynb": 4}


rows = []
for d in sorted(RESULTS.iterdir()):
    if not d.is_dir():
        continue
    m = d / "metrics.json"
    if not m.exists():
        continue
    jd = json.loads(m.read_text(encoding="utf-8"))
    run = jd.get("run_name", d.name)

    nb_name = RUN_TO_NOTEBOOK.get(run)
    wall = None
    if nb_name:
        wall = notebook_wall_seconds(NB_DIR / nb_name)
        if wall is not None and SHARED_NB_SPLITS.get(nb_name):
            wall = wall / SHARED_NB_SPLITS[nb_name]

    train_s, n_ep = history_train_seconds(d / "history.csv")

    rows.append({
        "run": run,
        "decoding": jd.get("decoding"),
        "beam_width": jd.get("beam_width"),
        "length_penalty": jd.get("length_penalty"),
        "epochs": n_ep if n_ep is not None else jd.get("epochs_run"),
        "wall_s": wall,
        "train_s": train_s,
        "best_val_loss": jd.get("best_val_loss"),
        "val_BLEU-1": jd.get("val_BLEU-1"),
        "val_BLEU-2": jd.get("val_BLEU-2"),
        "val_BLEU-3": jd.get("val_BLEU-3"),
        "val_BLEU-4": jd.get("val_BLEU-4"),
        "val_CIDEr":  jd.get("val_CIDEr"),
        "test_BLEU-1": jd.get("test_BLEU-1"),
        "test_BLEU-2": jd.get("test_BLEU-2"),
        "test_BLEU-3": jd.get("test_BLEU-3"),
        "test_BLEU-4": jd.get("test_BLEU-4"),
        "test_CIDEr":  jd.get("test_CIDEr"),
    })

df = pd.DataFrame(rows)
df = df.sort_values("test_BLEU-4", ascending=False, na_position="last").reset_index(drop=True)

(RESULTS / "phase2_summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")


# ---------- ASCII table -----------------------------------------------------
def fmt_dec(r) -> str:
    if r["beam_width"] and not pd.isna(r["beam_width"]):
        return f"beam w={int(r['beam_width'])} lp={r['length_penalty']}"
    d = r["decoding"]
    if d is None or (isinstance(d, float) and pd.isna(d)):
        # E9 ensemble metrics omits beam_width/length_penalty but always uses w=5 lp=0.7.
        if "ensemble" in str(r["run"]):
            return "beam w=5 lp=0.7"
        return "—"
    return str(d)


def fmt_dur(s) -> str:
    if s is None or pd.isna(s):
        return "—"
    s = int(round(s))
    m, sec = divmod(s, 60)
    return f"{m:>2}m {sec:02d}s"


def fmt_num(x, d=3) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{d}f}"


RUN_LABELS: dict[str, tuple[str, str]] = {
    # Phase 2 — baseline + decoding (E00 / E01)
    "phase2_e00_baseline_raina_greedy": ("E00", "EffNet-B0 frozen + Transformer (Raina prep)"),
    "phase2_e01_beam_w3_lp0.7":         ("E01", "E00 ckpt, beam-search inference only"),
    "phase2_e01_beam_w3_lp1.0":         ("E01", "E00 ckpt, beam-search inference only"),
    "phase2_e01_beam_w5_lp0.7":         ("E01", "E00 ckpt, beam-search inference only"),
    "phase2_e01_beam_w5_lp1.0":         ("E01", "E00 ckpt, beam-search inference only"),
    # Encoder + training-side variants (E02–E06)
    "phase2_e02_aug":                   ("E02", "+ random crop, rotation, colour jitter aug"),
    "phase2_e03_unfreeze":              ("E03", "+ unfreeze last 2 MBConv blocks (LR 1e-5)"),
    "phase2_e04_schedule_polish":       ("E04", "+ AdamW WD, warmup-cosine, LS, tied embed"),
    "phase2_e05_clip":                  ("E05", "swap encoder: CLIP ViT-B/16 features"),
    "phase2_e06_ocr":                   ("E06", "+ EasyOCR memory bank (segment-tagged)"),
    # Data augmentation + RL + ensemble (E07–E09)
    "phase2_e07_text_paraphrase":       ("E07", "+ Qwen2.5 text paraphrase of 5k captions"),
    "phase2_e07vlm_qwen2vl":            ("E07v","+ Qwen2-VL captions of train images"),
    "phase2_e08_scst":                  ("E08", "SCST/CIDEr-RL fine-tune of E07 (LOCAL DF — broken)"),
    "phase2_e09_ensemble":              ("E09", "logit-average ensemble of E00+E04+E07"),
    # New batch built on top of E05 (E10–E14)
    "phase2_e10_scst_corpus_idf":       ("E10", "SCST on E05 with corpus-IDF CIDEr reward"),
    "phase2_e11_ensemble_e05":          ("E11", "ensemble of E05 + E07 + E06 (E05-anchored)"),
    "phase2_e12_small_decoder":         ("E12", "small decoder 2L×d=384 + heavy reg, on CLIP feats"),
    "phase2_e13_siglip2":               ("E13", "swap encoder: SigLIP2 ViT-B/16 features"),
    "phase2_e14_mbr":                   ("E14", "MBR decoding (CIDEr utility) on E11 candidates"),
}


def label_for(run: str) -> tuple[str, str]:
    return RUN_LABELS.get(run, ("?", run))


headers = ["#", "Tag", "What", "Decoding", "Ep.", "Train", "Wall", "BLEU-4", "CIDEr"]
ascii_rows = []
for i, r in df.iterrows():
    ep = r["epochs"]
    ep_str = "—" if (ep is None or pd.isna(ep)) else str(int(ep))
    tag, what = label_for(str(r["run"]))
    cells = [
        str(i + 1),
        tag,
        what,
        fmt_dec(r),
        ep_str,
        fmt_dur(r["train_s"]),
        fmt_dur(r["wall_s"]),
        fmt_num(r["test_BLEU-4"]),
        fmt_num(r["test_CIDEr"]),
    ]
    cells = [str(c) for c in cells]  # defensive: ensure every cell is a string
    ascii_rows.append(cells)

# Column widths
widths = [max(len(h), max(len(r[c]) for r in ascii_rows)) for c, h in enumerate(headers)]


def sep(char="-"):
    return "+" + "+".join(char * (w + 2) for w in widths) + "+"


def row_line(cells):
    parts = []
    for c, val in enumerate(cells):
        # Numeric-ish columns (Ep, Train, Wall, BLEU-4, CIDEr) right-aligned; text left.
        if c in (4, 5, 6, 7, 8):
            parts.append(f" {val:>{widths[c]}} ")
        else:
            parts.append(f" {val:<{widths[c]}} ")
    return "|" + "|".join(parts) + "|"


ascii_lines = [sep("="), row_line(headers), sep("=")]
for r in ascii_rows:
    ascii_lines.append(row_line(r))
ascii_lines.append(sep("="))
ascii_table = "\n".join(ascii_lines)


# ---------- Status of remaining experiments --------------------------------
# Auto-build by checking which run dirs are present.
done_runs = {r["run"] for _, r in df.iterrows()}
pending_blurbs = [
    ("phase2_e6_ocr",          "E6      OCR memory bank (EasyOCR → segment-tagged extra cross-attention positions)."),
    ("phase2_e7vlm_qwen2vl",   "E7-VLM  Qwen2-VL-2B captions of training images (grounded augmentation)."),
]
pending_lines = ["Still to come (or in progress):"]
for run, blurb in pending_blurbs:
    if run not in done_runs:
        pending_lines.append("  " + blurb)
if len(pending_lines) == 1:
    pending_lines = []  # nothing pending → omit the block
not_run_block = "\n".join(pending_lines)


# ---------- Markdown report (uses the same ASCII table inside a code fence) -
phase1_bleu4 = 0.253
e9 = df[df["run"] == "phase2_e9_ensemble"]
e9_b = float(e9.iloc[0]["test_BLEU-4"]) if len(e9) else None
e9_c = float(e9.iloc[0]["test_CIDEr"]) if len(e9) else None

md_lines = []
md_lines.append("# Phase 2 experiments — summary")
md_lines.append("")
md_lines.append("All Phase 2 runs use **Raina's data prep** (no `<num>` token, vocab=4708).")
md_lines.append("Architecture: EfficientNet-B0 (frozen, cached features) + 3-layer Transformer decoder, d=512, h=8.")
md_lines.append("Hardware: AMD Radeon 8060S (Strix Halo iGPU), ROCm 7.2, PyTorch 2.11.0+rocm7.2.")
md_lines.append("")
md_lines.append("## Headline")
md_lines.append("")
md_lines.append(f"* Phase 1 baseline (MX150, `<num>` prep): test BLEU-4 = **{phase1_bleu4:.3f}**.")
if e9_b is not None:
    md_lines.append(f"* Phase 2 best (E9 logit-average ensemble of E0+E4+E7, beam w5 lp0.7): "
                    f"test BLEU-4 = **{e9_b:.3f}**, test CIDEr = **{e9_c:.3f}**.")
    md_lines.append(f"* Net improvement: **+{(e9_b - phase1_bleu4):.3f} BLEU-4**.")
md_lines.append("")
md_lines.append("## Experiment line-up (in plain English)")
md_lines.append("")
md_lines.append("Every row in the next table is one variant of the same architecture: a frozen image "
                "encoder feeds a sequence of spatial tokens into a 3-layer Transformer caption decoder "
                "(`d_model=512`, 8 heads, ~18 M trainable params on the decoder side). Each experiment "
                "changes exactly one thing relative to the baseline:")
md_lines.append("")
md_lines.append("* **E0** — Baseline. EfficientNet-B0 (ImageNet, frozen), greedy decoding. The reference.")
md_lines.append("* **E1** — Same checkpoint as E0, but **swap greedy for beam search** at inference "
                "(width 3 or 5, length penalty 0.7 or 1.0). No retraining.")
md_lines.append("* **E2** — Same encoder, **train-time augmentation re-enabled** (random crop + 10° "
                "rotation + colour jitter; no horizontal flip because OCR-style references get mirrored).")
md_lines.append("* **E3** — **Partially unfreeze the encoder**: last 2 MBConv blocks of EfficientNet-B0 "
                "trained at LR 1e-5 with discriminative LR.")
md_lines.append("* **E4** — **Training-schedule polish** on top of E0: AdamW with weight decay 0.01, "
                "warmup + cosine LR schedule, label smoothing 0.1, tied input ↔ output embeddings.")
md_lines.append("* **E5** — **Swap the encoder**: replace EfficientNet-B0 with **CLIP ViT-B/16** "
                "(image-text pretraining → features that are already aligned to natural language).")
md_lines.append("* **E6** — Same E0 visual features **plus an OCR memory bank**: EasyOCR is run once per "
                "image, the detected text is tokenised with the captioning vocab and fed into the decoder "
                "as extra cross-attention positions with a learned segment embedding.")
md_lines.append("* **E7** — **Text-only caption augmentation**: a sample of 5 000 training captions is "
                "paraphrased by Qwen2.5-1.5B-Instruct and appended to the training set.")
md_lines.append("* **E7v** — Same idea but **image-grounded augmentation**: Qwen2-VL-2B captions each "
                "training image directly; those captions are appended to the training set.")
md_lines.append("* **E8** — **SCST / CIDEr-RL fine-tune** of E7: warm-start, REINFORCE with a greedy "
                "baseline and CIDEr reward, 2 epochs.")
md_lines.append("* **E9** — **Logit-average ensemble** of the three strongest MLE checkpoints "
                "(E0 + E4 + E7) under beam search.")
md_lines.append("")
md_lines.append("## Results (ranked by test BLEU-4)")
md_lines.append("")
md_lines.append("```")
md_lines.append(ascii_table)
md_lines.append("```")
md_lines.append("")
md_lines.append("Columns: Ep. = epochs trained; Train = sum of per-epoch wall-clock time from history.csv; "
                "Wall = full notebook execution time (training + eval + any data-prep step like Qwen "
                "paraphrasing). For E1 (inference sweep) the four rows share one notebook, so Wall here "
                "is that notebook's total divided by 4 to keep the per-row meaning sane.")
md_lines.append("")
md_lines.append("## Notes per experiment")
md_lines.append("")
md_lines.append("**E0 — Baseline (Raina prep, greedy).** test BLEU-4 0.249. Tiny drop vs Phase 1's 0.253 "
                "from removing the `<num>` collapse — the model now has to learn digit n-grams literally.")
md_lines.append("")
md_lines.append("**E1 — Beam search on E0 checkpoint, no retraining.** Sweep over width ∈ {3, 5} × "
                "length_penalty ∈ {0.7, 1.0}. *beam_w5_lp0.7* wins with BLEU-4 0.365 — **+0.116 over E0 "
                "greedy from decoding alone**, the cheapest win of Phase 2.")
md_lines.append("")
md_lines.append("**E4 — Schedule polish.** AdamW + warmup-cosine (500 step warmup) + label smoothing 0.1 + "
                "tied input/output embeddings. *First attempt was broken* — tied embeddings combined with the "
                "`× sqrt(d_model)` input scaling made output logits explode (train_loss = 116 at epoch 1). "
                "Fix: re-init embedding with std = 1/√d_model and zero the pad row. Final: BLEU-4 0.362 / "
                "CIDEr 1.393 — tied with E1, so the polish didn't pay on this dataset size.")
md_lines.append("")
md_lines.append("**E7 — Qwen2.5-1.5B text paraphrase augmentation.** Sampled 5,000 training captions, "
                "generated one paraphrase each, appended to training set. BLEU-4 0.363 (~tied with E1) but "
                "**CIDEr 1.409, the best single-model CIDEr**. Text-only paraphrasing helps the diversity "
                "metric more than BLEU.")
md_lines.append("")
md_lines.append("**E8 — SCST / CIDEr-RL fine-tune.** Regressed (BLEU-4 0.291). The CIDEr reward was computed "
                "from a *local* document-frequency built from just the 5 references per image, instead of "
                "the standard training-corpus DF. The reward over-weights rare-within-5-refs n-grams and "
                "pushes captions to be narrow / repetitive. Proper SCST needs a precomputed corpus IDF — "
                "obvious follow-up.")
md_lines.append("")
md_lines.append("**E9 — Logit-average ensemble of E0, E4, E7 under beam_w5_lp0.7.** Best overall: "
                "**BLEU-4 0.374 / CIDEr 1.446**. Excluded E8 (regressed). The three MLE checkpoints "
                "disagree just enough that averaging their next-token distributions at each beam step "
                "reduces individual biases.")
md_lines.append("")

# ---------------- Overfitting analysis -------------------------------------
md_lines.append("## Overfitting analysis")
md_lines.append("")
md_lines.append("Train ↔ val cross-entropy gap from each run's `history.csv`. For label-smoothed "
                "runs (E4) the absolute losses are inflated by a `LS · log(V)` constant so the **gap** "
                "is the meaningful number, not the level.")
md_lines.append("")
md_lines.append("```")
md_lines.append("Run                             best_ep  train  val    gap   note")
md_lines.append("-------------------------------  ------  -----  -----  ----  -------------------------")

import csv as _csv
for run in ["phase2_e0_baseline_raina_greedy", "phase2_e4_schedule_polish", "phase2_e7_text_paraphrase"]:
    hist = RESULTS / run / "history.csv"
    if not hist.exists():
        continue
    h = pd.read_csv(hist)
    best_ep_row = h.loc[h["val_loss"].idxmin()]
    last_row = h.iloc[-1]
    gap_best = float(best_ep_row["train_loss"]) - float(best_ep_row["val_loss"])  # negative if train < val
    gap_best_abs = float(best_ep_row["val_loss"]) - float(best_ep_row["train_loss"])
    note = ""
    if abs(gap_best_abs) > 0.5:
        note = "memorising training set"
    md_lines.append(
        f"{run[:31]:31s}  {int(best_ep_row['epoch']):>6d}  "
        f"{best_ep_row['train_loss']:.3f}  {best_ep_row['val_loss']:.3f}  "
        f"{gap_best_abs:+.2f}  {note}"
    )
    # Also show the *final* (post-best) gap to make the overfit drift visible
    if int(last_row['epoch']) != int(best_ep_row['epoch']):
        gap_last = float(last_row["val_loss"]) - float(last_row["train_loss"])
        md_lines.append(
            f"  └ at last epoch {int(last_row['epoch']):>2d}             "
            f"{last_row['train_loss']:.3f}  {last_row['val_loss']:.3f}  "
            f"{gap_last:+.2f}  drift since best ep"
        )
md_lines.append("```")
md_lines.append("")
md_lines.append("Reading the table: the **best-epoch gap is ~0.6 nat** for E0/E7 and stays similar for E4 "
                "(despite its bigger absolute losses from label smoothing). Past the best epoch the gap "
                "blows out to ~1.0–1.2 — **early stopping is doing most of the regularisation work**.")
md_lines.append("")
md_lines.append("**Why didn't E4's regularisation package (LS=0.1, WD=0.01, warmup-cosine, tied embeds) "
                "buy us anything?** It *did* flatten the val-loss curve earlier, but BLEU was unchanged. "
                "Strongly suggests the bottleneck is **data scarcity, not regulariser choice**: 5,425 "
                "unique training images × ~5 captions = ~26 k caption rows feeding an 18 M-parameter "
                "decoder. E7's paraphrase augmentation added ~5 k more rows and gave a small CIDEr lift, "
                "consistent with that diagnosis.")
md_lines.append("")
md_lines.append("### Hyperparameters we did NOT sweep")
md_lines.append("")
md_lines.append("These would be the obvious next things to try if we keep tuning instead of changing "
                "architecture:")
md_lines.append("")
md_lines.append("* **LR sweep** — every run except E8's RL step used `1e-4`. Sensible alternatives: "
                "`5e-5` (slower, more regularised), `2e-4` (faster, may push the train-val gap further).")
md_lines.append("* **Dropout 0.3-0.4** — currently 0.2 everywhere. Cheap, often the biggest single "
                "regulariser knob on small caption sets.")
md_lines.append("* **Weight decay 5e-2** (E4 used 0.01) — combined with higher dropout.")
md_lines.append("* **Smaller decoder** — 3 layers × d=512 might already be too much capacity. Try "
                "2 layers × d=384 (≈ 6 M params instead of 18 M).")
md_lines.append("* **Early-stop on BLEU-4, not val-loss** — teacher-forced cross-entropy minimises at "
                "a slightly different epoch than greedy/beam BLEU. We're saving the val-loss minimum, "
                "which may not be the BLEU optimum.")
md_lines.append("* **Sub-word vocab (BPE / SentencePiece)** — word-level vocab with `MIN_WORD_FREQ=3` "
                "leaves a long UNK tail that dominates the rare-word loss; BPE would shrink that.")
md_lines.append("* **Higher label smoothing** (0.15-0.2) and **caption-side word dropout** at train "
                "time — both cheap and untouched.")
md_lines.append("")
md_lines.append("**Optimizer choices we actually used**:")
md_lines.append("")
md_lines.append("```")
md_lines.append("Run    Optimizer  LR        WD     Scheduler         LS    notes")
md_lines.append("-----  ---------  --------  -----  ----------------  ----  --------------------")
md_lines.append("E0     Adam       1e-4      0      ReduceLROnPlateau 0     vanilla")
md_lines.append("E4     AdamW      1e-4      0.01   warmup_cosine(500) 0.1  + tied embeddings")
md_lines.append("E7     Adam       1e-4      0      ReduceLROnPlateau 0     same as E0, more data")
md_lines.append("E8     Adam       5e-6      0      none               n/a  RL, 2 epochs SCST")
md_lines.append("```")
md_lines.append("")
md_lines.append("So we tested **one** non-trivial training recipe (E4) and got no win; the rest are "
                "essentially the Phase 1 recipe.")
md_lines.append("")
if not_run_block:
    md_lines.append("## " + not_run_block.split("\n")[0])
    md_lines.append("")
    md_lines.append("```")
    md_lines.append("\n".join(not_run_block.split("\n")[1:]))
    md_lines.append("```")
    md_lines.append("")
md_lines.append("## Cumulative time")
md_lines.append("")
md_lines.append("Sum of Wall over successful runs (the 9 rows above): "
                f"**{int(df['wall_s'].sum() // 60)} min {int(df['wall_s'].sum() % 60)} s**.")
md_lines.append("")
md_lines.append("Including the two failed retries (E4 v1 tied-embedding loss explosion, ~12 min killed; "
                "E7 v1 IndexError on empty LLM output, ~9 min) and inter-run gaps: "
                "**total ~1 h 28 min** wall-clock from kicking off E0 to E9 finishing.")
md_lines.append("")

(RESULTS / "phase2_summary.md").write_text("\n".join(md_lines), encoding="utf-8")


# ---------- Plain-text companion (just the ASCII table + missing block) -----
txt_lines = ["Phase 2 — results ranked by test BLEU-4", "", ascii_table, ""]
if not_run_block:
    txt_lines += [not_run_block, ""]
(RESULTS / "phase2_summary.txt").write_text("\n".join(txt_lines), encoding="utf-8")


# ---------- Print to stdout -------------------------------------------------
print(ascii_table)
print()
print(not_run_block)
print()
print(f"Wrote {RESULTS/'phase2_summary.csv'}")
print(f"Wrote {RESULTS/'phase2_summary.md'}")
print(f"Wrote {RESULTS/'phase2_summary.txt'}")
