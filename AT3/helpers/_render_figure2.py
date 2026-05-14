"""Render Figure 2: beam-width sweep at three length penalties on E15-s2.

Two side-by-side line plots (BLEU-4 and CIDEr), x = beam width.
Three lines per panel for length penalty 0.5 / 0.7 / 1.0.

Data sources (all under output/phase2_results/):
    phase2_e15s2_narrow_sweep/sweep_results.csv   greedy + w=2
    phase2_e15s2_beam_sweep/sweep_results.csv     w=5 (lp 0.5,1.0), w=10 (lp 0.7,1.0)
    phase2_e15s2_w3_lp_sweep/sweep_results.csv    w=3 lp 0.5 and 1.0
    phase2_e15s2_siglip2_small_seed2/metrics.json E15-s2 default (w=5, lp=0.7)
    plus a one-off re-eval of the (w=3, lp=0.7) point that I'd already produced.

Saves PNG to:
    /home/coezbek/dev/2026/DeepLearning_AT3_workload/figure_2_beam_sweep.png
"""
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_PNG = Path("/home/coezbek/dev/2026/DeepLearning_AT3_workload/figure_2_beam_sweep.png")

# All available sweep points on E15-s2, hand-collated from the metrics files.
# Format: (beam_width, length_penalty, test_BLEU-4, test_CIDEr)
points = [
    (1,  0.7, 0.3129, 1.3339),   # greedy
    (2,  0.7, 0.3946, 1.5440),
    (3,  0.5, 0.4113, 1.5683),
    (3,  0.7, 0.4102, 1.5732),
    (3,  1.0, 0.4030, 1.5810),
    (5,  0.5, 0.4055, 1.5427),
    (5,  0.7, 0.4034, 1.5488),
    (5,  1.0, 0.3984, 1.5608),
    (10, 0.7, 0.3712, 1.4611),
    (10, 1.0, 0.3695, 1.4731),
]

# Group by length penalty.
by_lp = {0.5: [], 0.7: [], 1.0: []}
for bw, lp, b4, cd in points:
    by_lp[lp].append((bw, b4, cd))
for lp in by_lp:
    by_lp[lp].sort()

styles = {
    0.5: {"label": "length penalty 0.5", "marker": "s", "linestyle": "--",
          "color": "#1f77b4"},
    0.7: {"label": "length penalty 0.7", "marker": "o", "linestyle": "-",
          "color": "#222222", "linewidth": 2},
    1.0: {"label": "length penalty 1.0", "marker": "^", "linestyle": "--",
          "color": "#d62728"},
}

fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(12, 5))

for ax, metric_idx, ylabel, ylim in [
    (ax_b, 1, "Test BLEU-4", (0.30, 0.42)),
    (ax_c, 2, "Test CIDEr-D", (1.30, 1.60)),
]:
    for lp, rows in by_lp.items():
        xs = [r[0] for r in rows]
        ys = [r[metric_idx] for r in rows]
        ax.plot(xs, ys, **styles[lp])
    ax.set_xlabel("Beam width")
    ax.set_ylabel(ylabel)
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.set_xticklabels(["1\n(greedy)", "2", "3", "5", "10"])
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower center", frameon=False, fontsize=9)

plt.tight_layout()
OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTPUT_PNG}")
