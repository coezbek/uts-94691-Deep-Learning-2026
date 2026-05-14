"""Render Figure 1: 2x2 grid of representative VizWiz-Captions samples
with their reference captions, drawn from the test split.

Saves the result to:
    /home/coezbek/dev/2026/DeepLearning_AT3_workload/figure_1_dataset_samples.png
"""
import json
import textwrap
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

OUTPUT_PNG = Path("/home/coezbek/dev/2026/DeepLearning_AT3_workload/figure_1_dataset_samples.png")
IMAGE_DIR = Path("data/val")

# Load split + references
df = pd.read_csv("output/processed_captions.csv")
test_df = df[df["split"] == "test"][["image_id", "file_name"]].drop_duplicates()
refs = json.loads(Path("output/reference_captions.json").read_text())

# Pick four representative images by keyword search across the cleaned captions.
# Each tile illustrates a defining property of the VizWiz-Captions data.
test_ids = set(test_df["image_id"].astype(int))

used_ids = set()

def first_test_match(predicate, label):
    """Return (image_id, file_name, refs) for the first test image whose
    references satisfy predicate, skipping any image already used in another tile."""
    for image_id, file_name in test_df.itertuples(index=False):
        iid = int(image_id)
        if iid not in test_ids or iid in used_ids:
            continue
        captions = refs.get(str(iid), [])
        if predicate(captions):
            used_ids.add(iid)
            return iid, file_name, captions
    raise RuntimeError(f"No test image matched {label}")


def has_kw(kws):
    """Return a predicate that fires if any reference contains any of kws."""
    return lambda caps: any(any(k in c.lower() for k in kws) for c in caps)


# (1) "Clear" — caption mentions a centered everyday object, no quality issues.
clear_pick = first_test_match(
    lambda caps: (
        all("quality issues" not in c.lower() for c in caps)
        and any(len(c.split()) >= 8 for c in caps)
        and not any(k in " ".join(caps).lower()
                    for k in ["blurry", "blurr", "hand", "label", "bottle",
                              "box", "package", "writing", "dark", "out of focus"])
    ),
    "clear",
)

# (2) "Blurry / motion blur" — references explicitly call it out.
blurry_pick = first_test_match(has_kw(["blurry", "blurr", "out of focus"]), "blurry")

# (3) "Product packaging / OCR content" — references describe text/label.
label_pick = first_test_match(
    has_kw(["label", "writing", "package", "instructions", "expiration", "ingredient"]),
    "label",
)

# (4) "Hand visible / off-frame" — references mention a hand holding the object.
hand_pick = first_test_match(has_kw(["hand", "holding"]), "hand")

picks = [
    ("Clear, well-framed subject", clear_pick),
    ("Motion blur / out-of-focus", blurry_pick),
    ("Text-bearing object (label / package)", label_pick),
    ("Hand visible / off-frame", hand_pick),
]

for label, (iid, fn, _) in picks:
    print(f"{label}: image_id={iid}  file={fn}")

# Render the 2x2 grid.
fig, axes = plt.subplots(2, 2, figsize=(13, 13))

for ax, (label, (iid, fn, captions)) in zip(axes.flat, picks):
    img = Image.open(IMAGE_DIR / fn).convert("RGB")
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    # Force a uniform square box for each cell so left/right column widths
    # are equal regardless of the underlying image's aspect ratio.
    ax.set_box_aspect(1)
    # No frame around the image.
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = f"image_id={iid}  ({fn})"
    ax.set_title(title, fontsize=10, loc="center", pad=6)

    # Wrap each reference caption to a reasonable line length and stack below.
    lines = []
    for i, c in enumerate(captions, 1):
        wrapped = textwrap.fill(f"{i}. {c}", width=80,
                                subsequent_indent="   ")
        lines.append(wrapped)
    caption_block = "\n".join(lines)
    ax.text(
        0.0, -0.02, caption_block,
        transform=ax.transAxes,
        fontsize=8, ha="left", va="top",
        family="monospace",
    )

plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.04,
                    wspace=0.15, hspace=0.45)
OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_PNG}")
