import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data import ACTIVITIES, load_dataset
from train import MODEL_PATH

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
REPORT_PATH = os.path.join(RESULTS_DIR, "metrics.txt")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")


def save_confusion_matrix(matrix):
    display = ConfusionMatrixDisplay(matrix, display_labels=ACTIVITIES)
    fig, ax = plt.subplots(figsize=(9, 8))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    _, _, x_test, y_test, _, _ = load_dataset()
    model = tf.keras.models.load_model(MODEL_PATH)

    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=ACTIVITIES, digits=3)
    matrix = confusion_matrix(y_test, y_pred)

    output = (
        f"Accuracy: {accuracy:.3f}\n"
        f"Macro F1-score: {macro_f1:.3f}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (rows = true, columns = predicted):\n{matrix}\n"
    )
    print(output)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as file:
        file.write(output)
    save_confusion_matrix(matrix)

    print("Report saved in", REPORT_PATH)
    print("Confusion matrix saved in", CONFUSION_MATRIX_PATH)
