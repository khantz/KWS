from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# Configuration
# ============================================================

ROOT = Path(r"C:\Users\M S I\PycharmProjects\KWS")

BEFORE_CSV = (
    ROOT
    / "result_png"
    / "csv"
    / "audio_validation_report.csv"
)

AFTER_CSV = (
    ROOT
    / "result_png"
    / "csv"
    / "audio_validation_report_after_preprocessed.csv"
)

OUTPUT_DIR = ROOT / "result_png" / "validation_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Consistent blue palette (one color per status)
# ------------------------------------------------------------
STATUS_COLORS = {
    "VALID":   "#08519c",   # dark blue  -> Valid
    "WARNING": "#6baed6",   # mid blue   -> Warning
    "INVALID": "#08306b",   # navy       -> Failed
}

BLUE_TEXT  = "#08306b"
BLUE_GRID  = "#d9e6f2"
BLUE_EDGE  = "#08306b"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": BLUE_TEXT,
        "axes.labelcolor": BLUE_TEXT,
        "axes.edgecolor": BLUE_TEXT,
        "xtick.color": BLUE_TEXT,
        "ytick.color": BLUE_TEXT,
    }
)


# ============================================================
# Load report
# ============================================================

def load_report(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV report not found:\n{path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "status" not in df.columns:
        raise ValueError(
            f"Missing 'status' column in {path.name}.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df["status"] = (
        df["status"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df = df[["status"]].dropna()

    if df.empty:
        raise ValueError(f"The CSV report contains no records: {path}")

    return df


before = load_report(BEFORE_CSV)
after  = load_report(AFTER_CSV)

reports = {
    "Before preprocessing": before,
    "After preprocessing":  after,
}

STATUSES = ["VALID", "WARNING", "INVALID"]
DISPLAY  = {
    "VALID":   "Valid",
    "WARNING": "Warning",
    "INVALID": "Failed",
}


def status_counts(df: pd.DataFrame) -> dict:
    counts = df["status"].value_counts()
    return {s: int(counts.get(s, 0)) for s in STATUSES}


# ============================================================
# Figure 1: Validation status bar chart
#   - ONE color per status (consistent)
#   - Before = solid, After = hatched
# ============================================================

def plot_validation_status():
    x = np.arange(len(STATUSES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    hatches = {
        "Before preprocessing": "",
        "After preprocessing":  "///",
    }
    alphas = {
        "Before preprocessing": 1.0,
        "After preprocessing":  0.85,
    }

    for index, (name, df) in enumerate(reports.items()):
        counts = status_counts(df)
        values = [counts[s] for s in STATUSES]
        colors = [STATUS_COLORS[s] for s in STATUSES]

        positions = x + (index - 0.5) * width

        bars = ax.bar(
            positions,
            values,
            width,
            color=colors,
            edgecolor=BLUE_EDGE,
            linewidth=0.9,
            hatch=hatches[name],
            alpha=alphas[name],
        )

        ax.bar_label(
            bars,
            fmt="%d",
            padding=3,
            fontsize=9,
            color=BLUE_TEXT,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[s] for s in STATUSES])
    ax.set_ylabel("Number of audio files")
    ax.set_title(
        "Audio Validation Status: Valid, Warning, Failed",
        fontweight="bold",
        pad=12,
    )

    ax.yaxis.grid(True, linestyle="--",
                  color=BLUE_GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BLUE_TEXT)

    # Two legends: one for status colour, one for preprocessing stage.
    status_handles = [
        Patch(facecolor=STATUS_COLORS[s],
              edgecolor=BLUE_EDGE,
              label=DISPLAY[s])
        for s in STATUSES
    ]
    stage_handles = [
        Patch(facecolor="white",
              edgecolor=BLUE_EDGE,
              hatch=hatches["Before preprocessing"],
              label="Before preprocessing"),
        Patch(facecolor="white",
              edgecolor=BLUE_EDGE,
              hatch=hatches["After preprocessing"],
              label="After preprocessing"),
    ]

    legend1 = ax.legend(
        handles=status_handles,
        loc="upper left",
        title="Status",
        frameon=True,
        facecolor="white",
        edgecolor=BLUE_EDGE,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=stage_handles,
        loc="upper right",
        title="Stage",
        frameon=True,
        facecolor="white",
        edgecolor=BLUE_EDGE,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "figure1_validation_status.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Figure 2: Percentage breakdown (stacked horizontal bar)
#   - Same color per status, consistent with Figure 1
# ============================================================

def plot_status_breakdown():
    fig, ax = plt.subplots(figsize=(9, 3.8))

    names = list(reports.keys())
    y = np.arange(len(names))

    valid_pct   = []
    warning_pct = []
    failed_pct  = []

    for name in names:
        counts = status_counts(reports[name])
        total = sum(counts.values()) or 1
        valid_pct.append(   100 * counts["VALID"]   / total)
        warning_pct.append( 100 * counts["WARNING"] / total)
        failed_pct.append(  100 * counts["INVALID"] / total)

    ax.barh(
        y, valid_pct,
        color=STATUS_COLORS["VALID"],
        edgecolor=BLUE_EDGE, linewidth=0.8,
        label="Valid",
    )
    ax.barh(
        y, warning_pct, left=valid_pct,
        color=STATUS_COLORS["WARNING"],
        edgecolor=BLUE_EDGE, linewidth=0.8,
        label="Warning",
    )
    ax.barh(
        y, failed_pct,
        left=np.array(valid_pct) + np.array(warning_pct),
        color=STATUS_COLORS["INVALID"],
        edgecolor=BLUE_EDGE, linewidth=0.8,
        label="Failed",
    )

    for i in range(len(names)):
        if valid_pct[i] > 5:
            ax.text(
                valid_pct[i] / 2, y[i],
                f"{valid_pct[i]:.1f}%",
                ha="center", va="center",
                color="white", fontsize=9, fontweight="bold",
            )
        if warning_pct[i] > 3:
            ax.text(
                valid_pct[i] + warning_pct[i] / 2, y[i],
                f"{warning_pct[i]:.1f}%",
                ha="center", va="center",
                color="white", fontsize=9, fontweight="bold",
            )
        if failed_pct[i] > 3:
            ax.text(
                valid_pct[i] + warning_pct[i] + failed_pct[i] / 2, y[i],
                f"{failed_pct[i]:.1f}%",
                ha="center", va="center",
                color="white", fontsize=9, fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of files (%)")
    ax.set_title(
        "Validation Outcome Breakdown",
        fontweight="bold",
        pad=12,
    )

    ax.xaxis.grid(True, linestyle="--",
                  color=BLUE_GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BLUE_TEXT)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "figure2_status_breakdown.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Summary
# ============================================================

def print_summary():
    print("\n" + "=" * 65)
    print("AUDIO VALIDATION SUMMARY")
    print("=" * 65)

    for name, df in reports.items():
        counts = status_counts(df)
        total = sum(counts.values())

        print(f"\n{name}")
        print("-" * 40)
        print(f"Total records : {total}")
        print(f"  Valid       : {counts['VALID']}")
        print(f"  Warning     : {counts['WARNING']}")
        print(f"  Failed      : {counts['INVALID']}")

    print("\n" + "=" * 65)
    print(f"All figures saved to:\n{OUTPUT_DIR}")
    print("=" * 65)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    plot_validation_status()
    plot_status_breakdown()
    print_summary()