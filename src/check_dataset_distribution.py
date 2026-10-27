import os
from collections import defaultdict
import matplotlib.pyplot as plt


class DatasetReporter:

    def __init__(self, root_dir: str, extension: str = ".wav"):
        self.root_dir = root_dir
        self.extension = extension

    def count_per_class(self):
        class_counts = defaultdict(int)

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(self.extension):
                    # Class = parent folder name
                    class_name = os.path.basename(root)
                    class_counts[class_name] += 1

        return class_counts

    def print_report(self):
        counts = self.count_per_class()
        total = 0

        print("\n===== CLASS BALANCE REPORT =====")

        for cls, count in sorted(counts.items()):
            print(f"{cls:<15}: {count}")
            total += count

        print("----------------------------")
        print("TOTAL           :", total)

    def plot_class_distribution(self):
        counts = self.count_per_class()

        # Sort classes alphabetically
        classes = sorted(counts.keys())
        values = [counts[cls] for cls in classes]

        # Create bar chart
        plt.figure(figsize=(12, 6))

        bars = plt.bar(classes, values)

        # Add count labels on top of each bar
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom"
            )

        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.title("Class Distribution of Dataset")

        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()


# =========================================
# HOW TO RUN IT
# =========================================
if __name__ == "__main__":

    # Point this to your dataset folder
    DATASET_PATH = r"/dataset/after_splited\train"

    # Create reporter instance
    reporter = DatasetReporter(
        root_dir=DATASET_PATH,
        extension=".wav"
    )

    # Print class distribution report
    reporter.print_report()

    # Display class distribution bar chart
    reporter.plot_class_distribution()