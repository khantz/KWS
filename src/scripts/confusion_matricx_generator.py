
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

JSON_PATH = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\evaluation_results\tflite_evaluation_results.json"
)

OUTPUT_DIR = Path(
    r"C:\Users\M S I\PycharmProjects\KWS\results\tflite_evaluation\plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "no", "off", "on", "one", "stop",
    "three", "two", "unknown", "yes"
]

with open(JSON_PATH, "r", encoding="utf-8") as file:
    results = json.load(file)["results"]

sns.set_theme(style="white")

for result in results:
    matrix = np.asarray(result["confusion_matrix"], dtype=float)

    model_name = result["model"]
    quantization = result["quantization"]
    labels = result.get("class_names", CLASS_NAMES)

    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix * 100,
        row_totals,
        out=np.zeros_like(matrix),
        where=row_totals != 0
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        percentages,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Percentage (%)"},
        annot_kws={"size": 10},
        ax=ax
    )

    ax.set_title(
        f"{model_name} ({quantization})",
        fontsize=16,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)

    fig.tight_layout()

    filename = (
        f"{model_name}_{quantization}_confusion_matrix_percent.png"
    )

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Saved: {OUTPUT_DIR / filename}")

print("\nCompleted: confusion matrices saved.")
