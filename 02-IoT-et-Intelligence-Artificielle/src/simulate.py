import argparse
import csv
import json
import os
import time
from collections import Counter, deque

import numpy as np
import tensorflow as tf

from data import ACTIVITIES, SIGNALS
from export_tflite import TFLITE_MODEL_PATH
from make_sensor_file import SENSOR_FILE_PATH
from train import NORM_STATS_PATH

SAMPLING_RATE = 50
WINDOW_SIZE = 128
PREDICTION_STEP = 64


def load_norm_stats():
    with open(NORM_STATS_PATH) as file:
        stats = json.load(file)
    return np.array(stats["mean"]), np.array(stats["std"])


def read_sensor(path, speed):
    with open(path) as file:
        for row in csv.DictReader(file):
            time.sleep(1 / (SAMPLING_RATE * speed))
            sample = [float(row[signal]) for signal in SIGNALS]
            yield sample, row["label"]


class ActivityClassifier:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.mean, self.std = load_norm_stats()

    def predict(self, window):
        x = ((np.array(window) - self.mean) / self.std).astype(np.float32)
        self.interpreter.set_tensor(self.input_index, x[np.newaxis])

        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start

        probabilities = self.interpreter.get_tensor(self.output_index)[0]
        predicted = int(np.argmax(probabilities))
        return ACTIVITIES[predicted], float(probabilities[predicted]), inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a sensor stream and run the TFLite model on it")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time at 50 Hz)")
    parser.add_argument("--file", default=SENSOR_FILE_PATH, help="CSV file with the sensor data")
    args = parser.parse_args()

    classifier = ActivityClassifier()
    buffer = deque(maxlen=WINDOW_SIZE)
    label_buffer = deque(maxlen=WINDOW_SIZE)

    inference_times = []
    correct = 0
    new_samples = 0

    print(f"Model: {TFLITE_MODEL_PATH} ({os.path.getsize(TFLITE_MODEL_PATH) / 1024:.1f} KB)")
    print(f"Reading sensor data from {args.file} (speed x{args.speed})\n")
    print(f"{'time':>7} | {'detected':<20} | {'confidence':>10} | {'true label':<20}")
    print("-" * 68)

    for sample_index, (sample, label) in enumerate(read_sensor(args.file, args.speed), start=1):
        buffer.append(sample)
        label_buffer.append(label)
        new_samples += 1

        if len(buffer) < WINDOW_SIZE or new_samples < PREDICTION_STEP:
            continue
        new_samples = 0

        activity, confidence, inference_time = classifier.predict(buffer)
        inference_times.append(inference_time)

        true_label = Counter(label_buffer).most_common(1)[0][0]
        correct += int(activity == true_label)
        status = "" if activity == true_label else "  <- wrong"

        seconds = sample_index / SAMPLING_RATE
        print(f"{seconds:6.1f}s | {activity:<20} | {confidence:>9.1%} | {true_label:<20}{status}")

    print("-" * 68)
    print(f"Predictions: {len(inference_times)}")
    print(f"Accuracy on the stream: {correct / len(inference_times):.1%}")
    print(f"Average inference time: {np.mean(inference_times) * 1000:.3f} ms")
