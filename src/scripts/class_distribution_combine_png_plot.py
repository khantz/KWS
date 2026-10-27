import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# DATA (Extracted from your four images)
# ============================================================
CLASSES = ["no", "off", "on", "one", "stop", "three", "two", "unknown", "yes"]

DATA_FULL  = [3941, 3745, 3845, 3890, 3872, 3727, 3880, 5000, 4044]   # (a)
DATA_TRAIN = [3152, 2996, 3076, 3112, 3097, 2981, 3104, 4000, 3235]   # (b)
DATA_VAL   = [394,  374,  384,  389,  387,  372,  388,  500,  404]    # (c)
DATA_TEST  = [395,  375,  385,  389,  388,  374,  388,  500,  405]    # (d)

# ============================================================
# STYLE (Optimized for laser printing)
# ============================================================
C_BAR  = "#1E88E5"   # bright vivid blue
C_EDGE = "black"     # crisp black edges
C_TEXT = "#111111"
C_GRID = "#DDDDDD"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
})


def draw_panel(ax, data, title):
    x = np.arange(len(CLASSES))
    bars = ax.bar(x, data, color=C_BAR, edgecolor=C_EDGE,
                  linewidth=0.8, width=0.7)

    for bar, v in zip(bars, data):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max(data) * 0.015,
                f"{v:,}",
                ha="center", va="bottom",
                fontsize=8, color=C_TEXT, fontweight="bold")

    ax.set_title(title, loc="left", fontweight="bold",
                 color=C_TEXT, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, max(data) * 1.18)
    ax.yaxis.grid(True, linestyle=":", color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ============================================================
# 2 x 2 GRID  --  no caption text baked into the image
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.subplots_adjust(hspace=0.45, wspace=0.20)

draw_panel(axes[0, 0], DATA_FULL,  "(a) Full dataset")
draw_panel(axes[0, 1], DATA_TRAIN, "(b) Training set")
draw_panel(axes[1, 0], DATA_VAL,   "(c) Validation set")
draw_panel(axes[1, 1], DATA_TEST,  "(d) Test set")

# ---- Save WITHOUT any figure caption ----
plt.savefig("Figure_4_1_Combined_Distribution.png",
            dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig("Figure_4_1_Combined_Distribution.pdf",
            bbox_inches="tight", facecolor="white")
plt.show()

print("Saved Figure_4_1_Combined_Distribution.png / .pdf")