import json

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class ModelEvaluator:

    def evaluate(
        self,
        model,
        x_test,
        y_test,
        class_names
    ):

        loss, accuracy = model.evaluate(
            x_test,
            y_test,
            verbose=0
        )

        probabilities = model.predict(
            x_test,
            verbose=0
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        test_accuracy = accuracy_score(
            y_test,
            predictions
        )

        report = classification_report(
            y_test,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        matrix = confusion_matrix(
            y_test,
            predictions
        )

        results = {
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "calculated_accuracy": float(test_accuracy),
            "classification_report": report,
            "confusion_matrix": matrix.tolist()
        }

        return results

    @staticmethod
    def print_results(results):

        print("\n========================================")
        print("Test Results")
        print("========================================")

        print(
            f"Test Loss     : "
            f"{results['test_loss']:.4f}"
        )

        print(
            f"Test Accuracy : "
            f"{results['test_accuracy'] * 100:.2f}%"
        )

        print("\nConfusion Matrix:")

        for row in results["confusion_matrix"]:
            print(row)

    @staticmethod
    def save_results(results, output_path):

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                results,
                file,
                indent=4
            )