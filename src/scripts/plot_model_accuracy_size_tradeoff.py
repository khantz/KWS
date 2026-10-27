"""
Thesis-grade model comparison figures
=====================================
Clean layout, no overlapping labels, blue palette.

Outputs (PNG + PDF):
    Figure_1_Model_Comparison.png
    Figure_2_Accuracy_Size_Tradeoff.png
    Figure_3_Quantization_Effect.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ============================================================
# PATHS
# ============================================================
OUTPUT_DIR = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\results"
    r"\tflite_evaluation\plots"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 600

# ============================================================
# DATA
# ============================================================
df = pd.DataFrame([
    {"Arch": "CNN1D",       "Quant": "FP32", "Acc": 93.8594, "F1": 94.1671, "Size": 115.5547},
    {"Arch": "CNN1D",       "Quant": "INT8", "Acc": 93.8594, "F1": 94.1602, "Size": 39.8672},
    {"Arch": "DS-CNN",      "Quant": "FP32", "Acc": 89.5249, "F1": 89.9246, "Size": 47.8672},
    {"Arch": "DS-CNN",      "Quant": "INT8", "Acc": 89.4415, "F1": 89.8284, "Size": 25.4531},
    {"Arch": "MobileNetV2", "Quant": "FP32", "Acc": 55.2654, "F1": 54.5219, "Size": 1629.7617},
    {"Arch": "MobileNetV2", "Quant": "INT8", "Acc": 24.5068, "F1": 19.1743, "Size": 646.6641},
])

ARCH_ORDER = ["CNN1D", "DS-CNN", "MobileNetV2"]

# ============================================================
# BLUE PALETTE  (one color per quantization, one hatch per arch)
# ============================================================
C_FP32 = "#9ECAE1"   # light blue  -> FP32
C_INT8 = "#08519C"   # dark blue   -> INT8
C_TEXT = "#08306B"
C_GRID = "#D9E6F2"
C_EDGE = "#08306B"

ARCH_HATCH = {
    "CNN1D":       "",
    "DS-CNN":      "//",
    "MobileNetV2": "..",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def save_fig(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf")


# ============================================================
# FIGURE 1 — Two-panel horizontal bars
# Left: model size.  Right: accuracy.
# ============================================================
def figure1_comparison():
    fig, (ax_size, ax_acc) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"wspace": 0.35}
    )

    y = np.arange(len(ARCH_ORDER))
    h = 0.36

    for i, arch in enumerate(ARCH_ORDER):
        sub = df[df["Arch"] == arch]
        fp = sub[sub["Quant"] == "FP32"].iloc[0]
        q8 = sub[sub["Quant"] == "INT8"].iloc[0]

        # ---- Size panel ----
        b1 = ax_size.barh(
            i + h/2, fp["Size"], h,
            color=C_FP32, edgecolor=C_EDGE, linewidth=0.9,
            hatch=ARCH_HATCH[arch],
        )
        b2 = ax_size.barh(
            i - h/2, q8["Size"], h,
            color=C_INT8, edgecolor=C_EDGE, linewidth=0.9,
            hatch=ARCH_HATCH[arch],
        )

        for bar, val in [(b1, fp["Size"]), (b2, q8["Size"])]:
            ax_size.text(
                val * 1.05, bar.get_y() + bar.get_height()/2,
                f"{val:,.1f}", va="center", ha="left",
                fontsize=9, color=C_TEXT, fontweight="bold",
            )

        # ---- Accuracy panel ----
        b3 = ax_acc.barh(
            i + h/2, fp["Acc"], h,
            color=C_FP32, edgecolor=C_EDGE, linewidth=0.9,
            hatch=ARCH_HATCH[arch],
        )
        b4 = ax_acc.barh(
            i - h/2, q8["Acc"], h,
            color=C_INT8, edgecolor=C_EDGE, linewidth=0.9,
            hatch=ARCH_HATCH[arch],
        )
        for bar, val in [(b3, fp["Acc"]), (b4, q8["Acc"])]:
            ax_acc.text(
                val + 0.6, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", ha="left",
                fontsize=9, color=C_TEXT, fontweight="bold",
            )

    # ---- Size axis ----
    ax_size.set_xscale("log")
    ax_size.set_xlim(15, 4000)
    ax_size.set_xlabel("TFLite model size (KB, log scale)")
    ax_size.set_title("(a) Model size", loc="left",
                      color=C_TEXT, fontweight="bold", pad=10)
    ax_size.set_yticks(y)
    ax_size.set_yticklabels(ARCH_ORDER)
    ax_size.grid(axis="x", linestyle=":", color=C_GRID, linewidth=0.7)
    ax_size.set_axisbelow(True)

    # ---- Accuracy axis ----
    ax_acc.set_xlim(0, 105)
    ax_acc.set_xlabel("Test accuracy (%)")
    ax_acc.set_title("(b) Accuracy", loc="left",
                     color=C_TEXT, fontweight="bold", pad=10)
    ax_acc.set_yticks(y)
    ax_acc.set_yticklabels([])
    ax_acc.grid(axis="x", linestyle=":", color=C_GRID, linewidth=0.7)
    ax_acc.set_axisbelow(True)

    for ax in (ax_size, ax_acc):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single shared legend at the top.
    handles = [
        Patch(facecolor=C_FP32, edgecolor=C_EDGE, label="FP32"),
        Patch(facecolor=C_INT8, edgecolor=C_EDGE, label="INT8"),
        Patch(facecolor="white", edgecolor=C_EDGE, hatch="",
              label="CNN1D"),
        Patch(facecolor="white", edgecolor=C_EDGE, hatch="//",
              label="DS-CNN"),
        Patch(facecolor="white", edgecolor=C_EDGE, hatch="..",
              label="MobileNetV2"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False)

    save_fig(fig, "Figure_1_Model_Comparison")


# ============================================================
# FIGURE 2 — Accuracy–size scatter, numbered markers
# Labels are removed from the plot and shown in a key table below.
# ============================================================
def figure2_tradeoff():
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.15)
    ax = fig.add_subplot(gs[0])
    ax_key = fig.add_subplot(gs[1])
    ax_key.axis("off")

    # Sort by size to number consistently.
    ordered = df.sort_values("Size").reset_index(drop=True)

    for idx, row in ordered.iterrows():
        marker = "o" if row["Quant"] == "FP32" else "s"
        ax.scatter(
            row["Size"], row["Acc"],
            s=180, marker=marker,
            color=C_FP32 if row["Quant"] == "FP32" else C_INT8,
            edgecolor=C_EDGE, linewidth=1.2, zorder=3,
        )
        ax.annotate(
            f"{idx+1}",
            (row["Size"], row["Acc"]),
            xytext=(0, 0), textcoords="offset points",
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="white" if row["Quant"] == "INT8" else C_TEXT,
            zorder=4,
        )

    # Dashed arrows FP32 -> INT8 per architecture.
    for arch in ARCH_ORDER:
        sub = ordered[ordered["Arch"] == arch]
        fp = sub[sub["Quant"] == "FP32"]
        q8 = sub[sub["Quant"] == "INT8"]
        if fp.empty or q8.empty:
            continue
        x1, y1 = fp.iloc[0]["Size"], fp.iloc[0]["Acc"]
        x2, y2 = q8.iloc[0]["Size"], q8.iloc[0]["Acc"]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "color": C_TEXT, "linewidth": 1.4,
                "linestyle": "--", "alpha": 0.7,
                "shrinkA": 12, "shrinkB": 12,
            },
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlim(15, 4000)
    ax.set_ylim(0, 105)
    ax.set_xlabel("TFLite model size (KB, log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(
        "Accuracy vs. Model Size Trade-off",
        loc="left", color=C_TEXT, fontweight="bold", pad=10,
    )
    ax.grid(True, which="both", linestyle=":",
            color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Key table below the plot.
    key_lines = []
    for idx, row in ordered.iterrows():
        key_lines.append(
            f"[{idx+1}]  {row['Arch']:<12} {row['Quant']:<5}   "
            f"{row['Size']:>9,.2f} KB    {row['Acc']:>6.2f}%"
        )
    ax_key.text(
        0.02, 0.9,
        "  ".join(["[#]  Architecture  Format      Size          Accuracy"]),
        fontsize=9, color=C_TEXT, fontweight="bold",
        family="DejaVu Sans Mono",
        transform=ax_key.transAxes,
    )
    for i, line in enumerate(key_lines):
        ax_key.text(
            0.02, 0.72 - i * 0.17,
            "  ".join([line]),
            fontsize=9, color=C_TEXT,
            family="DejaVu Sans Mono",
            transform=ax_key.transAxes,
        )

    save_fig(fig, "Figure_2_Accuracy_Size_Tradeoff")


# ============================================================
# FIGURE 3 — Quantization effect (size reduction + accuracy drop)
# ============================================================
def figure3_quantization():
    records = []
    for arch in ARCH_ORDER:
        sub = df[df["Arch"] == arch]
        fp = sub[sub["Quant"] == "FP32"].iloc[0]
        q8 = sub[sub["Quant"] == "INT8"].iloc[0]
        records.append({
            "Arch": arch,
            "SizeReduction": (fp["Size"] - q8["Size"]) / fp["Size"] * 100,
            "AccDrop": fp["Acc"] - q8["Acc"],
        })
    res = pd.DataFrame(records)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.8),
        gridspec_kw={"wspace": 0.30}
    )

    y = np.arange(len(res))

    # (a) Size reduction
    bars = ax1.barh(y, res["SizeReduction"],
                    color=C_INT8, edgecolor=C_EDGE,
                    linewidth=0.9, height=0.55)
    for bar, val in zip(bars, res["SizeReduction"]):
        ax1.text(
            val + 1.0, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", ha="left",
            fontsize=10, color=C_TEXT, fontweight="bold",
        )
    ax1.set_yticks(y)
    ax1.set_yticklabels(res["Arch"])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Model-size reduction after INT8 (%)")
    ax1.set_title("(a) Size reduction", loc="left",
                  color=C_TEXT, fontweight="bold", pad=10)
    ax1.grid(axis="x", linestyle=":", color=C_GRID, linewidth=0.7)
    ax1.set_axisbelow(True)

    # (b) Accuracy drop
    bars = ax2.barh(y, res["AccDrop"],
                    color=C_FP32, edgecolor=C_EDGE,
                    linewidth=0.9, height=0.55)
    for bar, val in zip(bars, res["AccDrop"]):
        ax2.text(
            val + 0.4, bar.get_y() + bar.get_height()/2,
            f"{val:.2f} pp", va="center", ha="left",
            fontsize=10, color=C_TEXT, fontweight="bold",
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(res["AccDrop"].max() * 1.35, 5))
    ax2.set_xlabel("Accuracy drop after INT8 (percentage points)")
    ax2.set_title("(b) Accuracy drop", loc="left",
                  color=C_TEXT, fontweight="bold", pad=10)
    ax2.grid(axis="x", linestyle=":", color=C_GRID, linewidth=0.7)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save_fig(fig, "Figure_3_Quantization_Effect")

    res.to_csv(OUTPUT_DIR / "Quantization_Effect_Summary.csv",
               index=False, float_format="%.4f")
    print("Saved Quantization_Effect_Summary.csv")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    figure1_comparison()
    figure2_tradeoff()
    figure3_quantization()
    print(f"\nAll figures saved to:\n{OUTPUT_DIR}")