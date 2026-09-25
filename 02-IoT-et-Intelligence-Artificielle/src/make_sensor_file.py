import argparse
import csv
import os

import numpy as np

from data import ACTIVITIES, DATASET_DIR, SIGNALS, load_labels, load_signals

SENSOR_FILE_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "sensor_stream.csv"))

WINDOW_STEP = 64


def build_stream(windows, labels):
    samples = []
    sample_labels = []
    for window, label in zip(windows, labels):
        samples.append(window[:WINDOW_STEP])
        sample_labels += [label] * WINDOW_STEP

    return np.concatenate(samples), sample_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a continuous sensor file from the UCI HAR test set")
    parser.add_argument("--subject", type=int, default=2, help="test subject id (2, 4, 9, 10, 12, 13, 18, 20 or 24)")
    args = parser.parse_args()

    x_test = load_signals("test")
    y_test = load_labels("test")
    subjects = np.loadtxt(os.path.join(DATASET_DIR, "test", "subject_test.txt"), dtype=int)

    subject_rows = subjects == args.subject
    if not subject_rows.any():
        raise SystemExit(f"Subject {args.subject} is not in the test set")

    stream, stream_labels = build_stream(x_test[subject_rows], y_test[subject_rows])

    with open(SENSOR_FILE_PATH, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SIGNALS + ["label"])
        for sample, label in zip(stream, stream_labels):
            writer.writerow([f"{value:.6f}" for value in sample] + [ACTIVITIES[label]])

    print(f"Sensor file saved in {SENSOR_FILE_PATH}")
    print(f"{len(stream)} samples, i.e. {len(stream) / 50:.0f} seconds of data at 50 Hz")
