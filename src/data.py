import os

import numpy as np

DATASET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "UCI HAR Dataset"))

SIGNALS = [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

ACTIVITIES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def load_signals(split):
    signals = []
    for signal in SIGNALS:
        path = os.path.join(DATASET_DIR, split, "Inertial Signals", f"{signal}_{split}.txt")
        signals.append(np.loadtxt(path))
    return np.stack(signals, axis=-1)


def load_labels(split):
    path = os.path.join(DATASET_DIR, split, f"y_{split}.txt")
    return np.loadtxt(path, dtype=int) - 1


def load_split(split):
    x = load_signals(split)
    y = load_labels(split)

    valid_rows = ~np.isnan(x).any(axis=(1, 2))
    return x[valid_rows], y[valid_rows]


def compute_norm_stats(x_train):
    mean = x_train.mean(axis=(0, 1))
    std = x_train.std(axis=(0, 1))
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


def load_dataset():
    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")

    mean, std = compute_norm_stats(x_train)
    x_train = normalize(x_train, mean, std)
    x_test = normalize(x_test, mean, std)

    return x_train, y_train, x_test, y_test, mean, std


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, mean, std = load_dataset()

    print("x_train:", x_train.shape, "y_train:", y_train.shape)
    print("x_test: ", x_test.shape, "y_test: ", y_test.shape)
    print()
    print("mean per channel:", np.round(mean, 3))
    print("std per channel: ", np.round(std, 3))
    print()
    print("Train samples per class:")
    for index, activity in enumerate(ACTIVITIES):
        print(f"  {activity:<20} {np.sum(y_train == index)}")
