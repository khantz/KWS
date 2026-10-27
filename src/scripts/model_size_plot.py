import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
JSON_PATH = r"C:\Users\M S I\PycharmProjects\KWS\evaluation_results\tflite_evaluation_results.json"
OUT_DIR   = r"C:\Users\M S I\PycharmProjects\KWS\results\tflite_evaluation\plots"
os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

results = data["results"]

# ---------------- Blue palette ----------------
BLUE_LIGHT = "#9ecae1"
BLUE_DARK  = "#08519c"
TEXT_COLOR = "#08306b"
GRID_COLOR = "#d9e6f2"

# ---------------- Organize by model ----------------
model_order = []
for r in results:
    if r["model"] not in model_order:
        model_order.append(r["model"])

lookup = {m: {} for m in model_order}
for r in results:
    lookup[r["model"]][r["quantization"]] = r

models    = model_order
fp32_kb   = [lookup[m]["FP32"]["model_size_kb"] for m in models]
int8_kb   = [lookup[m]["INT8"]["model_size_kb"] for m in models]
fp32_acc  = [lookup[m]["FP32"]["accuracy"] * 100 for m in models]
int8_acc  = [lookup[m]["INT8"]["accuracy"] * 100 for m in models]
reduction = [(f - i) / f * 100 for f, i in zip(fp32_kb, int8_kb)]


# ============================================================
# Plot 1 — Model size comparison (log scale)
# ============================================================
x = np.arange(len(models))
width = 0.38

fig, ax = plt.subplots(figsize=(9, 5.5))

bars_fp32 = ax.bar(
    x - width/2, fp32_kb, width,
    label="FP32", color=BLUE_LIGHT,
    edgecolor=TEXT_COLOR, linewidth=0.8
)
bars_int8 = ax.bar(
    x + width/2, int8_kb, width,
    label="INT8", color=BLUE_DARK,
    edgecolor=TEXT_COLOR, linewidth=0.8
)

for bars in (bars_fp32, bars_int8):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f} KB",
            xy=(bar.get_x() + bar.get_width()/2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, color=TEXT_COLOR, fontweight="bold"
        )

for i, r in enumerate(reduction):
    ax.annotate(
        f"−{r:.1f}%",
        xy=(x[i], max(fp32_kb[i], int8_kb[i]) * 1.35),
        ha="center", va="bottom",
        fontsize=10, color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.25",
                  fc="#e7f0fa", ec=BLUE_DARK, lw=0.8)
    )

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, color=TEXT_COLOR)
ax.set_ylabel("Model size (KB, log scale)", fontsize=11, color=TEXT_COLOR)
ax.set_title("Model Size Comparison: FP32 vs INT8",
             fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)

ax.yaxis.grid(True, which="both", linestyle="--",
              color=GRID_COLOR, linewidth=0.7)
ax.set_axisbelow(True)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(TEXT_COLOR)

ax.tick_params(axis="y", colors=TEXT_COLOR)
ax.tick_params(axis="x", colors=TEXT_COLOR)

legend = ax.legend(loc="upper left", frameon=True, fontsize=10,
                   facecolor="white", edgecolor=BLUE_DARK)
for text in legend.get_texts():
    text.set_color(TEXT_COLOR)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "model_size_comparison.png")
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {out1}")


# ============================================================
# Plot 2 — Accuracy vs Size trade-off
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

for i, m in enumerate(models):
    ax.scatter(fp32_kb[i], fp32_acc[i], s=220, c=BLUE_LIGHT,
               edgecolor=TEXT_COLOR, linewidth=1.2, zorder=3)
    ax.scatter(int8_kb[i], int8_acc[i], s=220, c=BLUE_DARK,
               edgecolor=TEXT_COLOR, linewidth=1.2, zorder=3)
    ax.annotate(f"{m} FP32", (fp32_kb[i], fp32_acc[i]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=9, color=TEXT_COLOR)
    ax.annotate(f"{m} INT8", (int8_kb[i], int8_acc[i]),
                xytext=(8, -14), textcoords="offset points",
                fontsize=9, color=TEXT_COLOR)

ax.set_xscale("log")
ax.set_xlabel("Model size (KB, log scale)", fontsize=11, color=TEXT_COLOR)
ax.set_ylabel("Accuracy (%)", fontsize=11, color=TEXT_COLOR)
ax.set_title("Accuracy vs Model Size Trade-off",
             fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)

ax.grid(True, which="both", linestyle="--",
        color=GRID_COLOR, linewidth=0.7)
ax.set_axisbelow(True)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(TEXT_COLOR)

ax.tick_params(colors=TEXT_COLOR)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "accuracy_vs_size.png")
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {out2}")


# ============================================================
# Plot 3 — FP32 → INT8 accuracy drop per model
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))

drop = [f - i for f, i in zip(fp32_acc, int8_acc)]
colors = [BLUE_DARK if d > 5 else BLUE_LIGHT for d in drop]

bars = ax.bar(models, drop, color=colors,
              edgecolor=TEXT_COLOR, linewidth=0.9, width=0.55)

for bar, d in zip(bars, drop):
    h = bar.get_height()
    ax.annotate(f"{d:.2f} pp",
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, color=TEXT_COLOR, fontweight="bold")

ax.set_ylabel("Accuracy drop (percentage points)",
              fontsize=11, color=TEXT_COLOR)
ax.set_title("Accuracy Degradation after INT8 Quantization",
             fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)

ax.yaxis.grid(True, linestyle="--", color=GRID_COLOR, linewidth=0.7)
ax.set_axisbelow(True)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(TEXT_COLOR)

ax.tick_params(colors=TEXT_COLOR)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "accuracy_drop.png")
plt.savefig(out3, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {out3}")