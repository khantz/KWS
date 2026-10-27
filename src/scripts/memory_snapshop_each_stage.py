import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---- Data ----
labels = ["Before\ninit", "Model\ninit", "After\ninit",
          "Input\nfilled", "After\ninference"]
values = [329_928, -63_708, 0, 0, 0]
starts = [0, 266_220, 266_220, 266_220, 266_220]

C_DROP  = "#DC2626"     # red for the drop
C_FLAT  = "#3B82F6"     # blue for the flat bar
C_TEXT  = "#111827"
C_LIGHT = "#E5E7EB"

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "text.color": C_TEXT,
    "axes.labelcolor": C_TEXT,
    "axes.edgecolor": C_LIGHT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
})

fig, ax = plt.subplots(figsize=(7.5, 4))

x = np.arange(len(labels))

# Draw the flat segments (After init → After inference)
ax.bar(0, 329_928, width=0.55, color=C_FLAT, edgecolor="none")
ax.bar(1, 63_708,  width=0.55, bottom=266_220,
       color=C_DROP, edgecolor="none")
ax.bar(2, 266_220, width=0.55, color=C_FLAT, edgecolor="none")
ax.bar(3, 266_220, width=0.55, color=C_FLAT, edgecolor="none", alpha=0.35)
ax.bar(4, 266_220, width=0.55, color=C_FLAT, edgecolor="none", alpha=0.35)

# Dashed guide line at 266,220
ax.axhline(266_220, color=C_LIGHT, linewidth=1,
           linestyle="--", zorder=0)

# Values above each bar
for i, v in enumerate(values):
    top = starts[i] + (v if v > 0 else 0)
    txt = f"{starts[i] + v:,}" if i == 0 else ""
    if i == 0:
        ax.text(0, 329_928 + 8_000, f"329,928",
                ha="center", fontsize=9, fontweight="bold", color=C_TEXT)
    elif i == 1:
        ax.text(1, 329_928 + 8_000, "−63,708",
                ha="center", fontsize=9, fontweight="bold", color=C_DROP)
    else:
        ax.text(i, 266_220 + 8_000, "266,220",
                ha="center", fontsize=9, color=C_TEXT)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Free heap (bytes)")
ax.set_ylim(0, 400_000)
ax.set_yticks([0, 100_000, 200_000, 300_000, 400_000])

ax.yaxis.grid(True, linestyle="-", color=C_LIGHT, linewidth=0.6, alpha=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.set_title("Free heap remains constant after model initialization",
             loc="left", fontweight="bold", pad=12)

plt.tight_layout()

OUT = Path(r"C:\Users\M S I\PycharmProjects\KWS\results\tflite_evaluation\plots\Figure_4_X_Memory_Usage.png")
plt.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.show()