import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

MODEL_FILES = {
    "DSCNN":       Path(r"C:\Users\M S I\PycharmProjects\KWS\Models\DSCNN_training_history.json"),
    "CNN1D":       Path(r"C:\Users\M S I\PycharmProjects\KWS\Models\CNN1D_training_history.json"),
    "MobileNetV2": Path(r"C:\Users\M S I\PycharmProjects\KWS\Models\MobileNetV2_training_history.json"),
}

OUTPUT_DIR = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\results\tflite_evaluation\plots"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 600

# ============================================================
# BRIGHT, HIGH-CONTRAST PALETTE
# ============================================================

# Strong vivid pair for training vs validation inside each panel
C_TRAIN = "#0066FF"   # vivid blue
C_VAL   = "#FF6B00"   # vivid orange

# One bright color per model (for combined figures)
MODEL_COLORS = {
    "CNN1D":       "#0066FF",   # vivid blue
    "DSCNN":       "#00A878",   # vivid teal-green
    "MobileNetV2": "#D62828",   # vivid red
}

C_TEXT = "#111111"
C_GRID = "#E6E6E6"
C_EDGE = "#111111"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.linewidth": 1.0,
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
    print(f"Saved: {name}.png / .pdf")


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        history = json.load(f)
    required = ["accuracy", "loss", "val_accuracy", "val_loss"]
    missing = [k for k in required if k not in history]
    if missing:
        raise KeyError(f"{path.name} is missing keys: {missing}")
    return {
        "accuracy":     np.asarray(history["accuracy"],     dtype=float) * 100,
        "val_accuracy": np.asarray(history["val_accuracy"], dtype=float) * 100,
        "loss":         np.asarray(history["loss"],         dtype=float),
        "val_loss":     np.asarray(history["val_loss"],     dtype=float),
    }


histories = {}
for name, path in MODEL_FILES.items():
    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        continue
    histories[name] = load_history(path)

if not histories:
    raise SystemExit("No training history files were found.")


# ============================================================
# SHARED SCALES
# ============================================================

ACC_YMIN, ACC_YMAX = 0, 100

loss_all = np.concatenate(
    [np.concatenate([h["loss"], h["val_loss"]]) for h in histories.values()]
)
LOSS_YMIN = 0
LOSS_YMAX = float(np.ceil(loss_all.max() * 1.10))
MAX_EPOCHS = max(len(h["accuracy"]) for h in histories.values())

print(f"\nShared y-limits:")
print(f"  Accuracy : {ACC_YMIN} – {ACC_YMAX} %")
print(f"  Loss     : {LOSS_YMIN} – {LOSS_YMAX:.2f}")
print(f"  Max epochs: {MAX_EPOCHS}\n")


# ============================================================
# SIDE-BY-SIDE FIGURE PER MODEL
# ============================================================

def plot_side_by_side(name, hist):
    acc      = hist["accuracy"]
    val_acc  = hist["val_accuracy"]
    loss     = hist["loss"]
    val_loss = hist["val_loss"]

    ep_acc  = np.arange(1, len(acc) + 1)
    ep_loss = np.arange(1, len(loss) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(
        1, 2, figsize=(11, 4.6),
        gridspec_kw={"wspace": 0.25}
    )

    # ---- Left panel: accuracy ----
    ax_acc.plot(ep_acc, acc, "-", color=C_TRAIN,
                linewidth=2.4, label="Training")
    ax_acc.plot(ep_acc, val_acc, "--", color=C_VAL,
                linewidth=2.4, label="Validation")

    ax_acc.set_xlim(1, MAX_EPOCHS)
    ax_acc.set_ylim(ACC_YMIN, ACC_YMAX)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("(a) Accuracy", loc="left",
                     color=C_TEXT, fontweight="bold", pad=10)
    ax_acc.grid(True, linestyle=":", color=C_GRID, linewidth=0.8)
    ax_acc.set_axisbelow(True)
    for s in ("top", "right"):
        ax_acc.spines[s].set_visible(False)
    ax_acc.legend(frameon=True, facecolor="white",
                  edgecolor=C_GRID, loc="lower right")

    # ---- Right panel: loss ----
    ax_loss.plot(ep_loss, loss, "-", color=C_TRAIN,
                 linewidth=2.4, label="Training")
    ax_loss.plot(ep_loss, val_loss, "--", color=C_VAL,
                 linewidth=2.4, label="Validation")

    ax_loss.set_xlim(1, MAX_EPOCHS)
    ax_loss.set_ylim(LOSS_YMIN, LOSS_YMAX)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("(b) Loss", loc="left",
                      color=C_TEXT, fontweight="bold", pad=10)
    ax_loss.grid(True, linestyle=":", color=C_GRID, linewidth=0.8)
    ax_loss.set_axisbelow(True)
    for s in ("top", "right"):
        ax_loss.spines[s].set_visible(False)
    ax_loss.legend(frameon=True, facecolor="white",
                   edgecolor=C_GRID, loc="upper right")

    # ---- Figure title ----
    fig.suptitle(
        f"{name}: Training and Validation Curves",
        fontsize=12.5, fontweight="bold", color=C_TEXT,
        x=0.02, ha="left", y=1.02,
    )

    fig.tight_layout()
    save_fig(fig, f"{name}_training_side_by_side")


# ============================================================
# COMBINED FIGURES (all models overlaid, bright per-model colors)
# ============================================================

def plot_combined_accuracy():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, hist in histories.items():
        color = MODEL_COLORS.get(name, C_TRAIN)
        ep = np.arange(1, len(hist["accuracy"]) + 1)
        ax.plot(ep, hist["accuracy"], "-",
                color=color, linewidth=2.4, label=f"{name} (train)")
        ax.plot(ep, hist["val_accuracy"], "--",
                color=color, linewidth=2.0, alpha=0.85,
                label=f"{name} (val)")

    ax.set_xlim(1, MAX_EPOCHS)
    ax.set_ylim(ACC_YMIN, ACC_YMAX)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Validation Accuracy — All Models",
                 loc="left", color=C_TEXT, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=True, facecolor="white",
              edgecolor=C_GRID, loc="lower right", ncol=2)

    fig.tight_layout()
    save_fig(fig, "Figure_All_Models_Accuracy")


def plot_combined_loss():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, hist in histories.items():
        color = MODEL_COLORS.get(name, C_TRAIN)
        ep = np.arange(1, len(hist["loss"]) + 1)
        ax.plot(ep, hist["loss"], "-",
                color=color, linewidth=2.4, label=f"{name} (train)")
        ax.plot(ep, hist["val_loss"], "--",
                color=color, linewidth=2.0, alpha=0.85,
                label=f"{name} (val)")

    ax.set_xlim(1, MAX_EPOCHS)
    ax.set_ylim(LOSS_YMIN, LOSS_YMAX)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss — All Models",
                 loc="left", color=C_TEXT, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=True, facecolor="white",
              edgecolor=C_GRID, loc="upper right", ncol=2)

    fig.tight_layout()
    save_fig(fig, "Figure_All_Models_Loss")


# ============================================================
# BEST VALIDATION ACCURACY BAR CHART
# ============================================================

def plot_best_validation():
    names, values, colors = [], [], []

    for name, hist in histories.items():
        names.append(name)
        values.append(float(np.max(hist["val_accuracy"])))
        colors.append(MODEL_COLORS.get(name, C_TRAIN))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors,
                  edgecolor=C_EDGE, linewidth=1.2, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.0, f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=11, color=C_TEXT, fontweight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Best Validation Accuracy per Model",
                 loc="left", color=C_TEXT, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle=":", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "Figure_Best_Validation_Accuracy")


# ============================================================
# RUN
# ============================================================

for name, hist in histories.items():
    plot_side_by_side(name, hist)

plot_combined_accuracy()
plot_combined_loss()
plot_best_validation()

print("Best validation accuracy per model:")
for name, hist in histories.items():
    best_epoch = int(np.argmax(hist["val_accuracy"])) + 1
    best_acc = float(np.max(hist["val_accuracy"]))
    print(f"  {name:<12}  {best_acc:.2f}%  (epoch {best_epoch})")

print(f"\nAll figures saved to:\n{OUTPUT_DIR}")